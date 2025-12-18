import argparse
import os
import random
import yaml
import time
import shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# torchmetrics
from torchmetrics.classification import (
    MulticlassAUROC, 
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy
)

# NNCF
import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

# Custom Modules
import models
import losses
from dataset import Dataset
from metrics import iou_score
from utils import str2bool
from get_transforms import get_training_transforms, get_validation_transforms

# ------------------------- Utils -------------------------

def list_type(s):
    return [int(a) for a in s.split(',')]

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

def build_criterion(name):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().cuda()
    return losses.__dict__[name]().cuda()

def save_ckpt(path, model, optimizer, scheduler, epoch, best_iou, best_dice, best_auroc, best_auprc, best_f1, config, compression_ctrl=None):
    # NNCF 모델 저장 시 unwrap 불필요 (구조 유지), 혹은 compression_ctrl 사용
    # 여기서는 일반 PyTorch 스타일로 저장하되, 나중에 로드 시 NNCF로 다시 감싸야 함
    state_dict = model.state_dict()
    ckpt = {
        'epoch': epoch,
        'model': state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_iou': best_iou,
        'best_dice': best_dice,
        'best_auroc': best_auroc,
        'best_auprc': best_auprc,
        'best_f1': best_f1,
        'config': config,
    }
    torch.save(ckpt, path)

# ------------------------- Data Wrapper for NNCF -------------------------
class QATCalibrationLoader:
    """NNCF 초기화를 위해 데이터를 모델 입력 형태(Tuple)로 변환하여 제공"""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            # batch = (img, mask, label, spacing, meta)
            # 모델 forward(x, spacing)에 맞춰 전달
            x = batch[0].to(self.device)
            spacing = batch[3].to(self.device)
            yield (x, spacing)

    def __len__(self):
        return len(self.loader)

# ------------------------- Train / Validate -------------------------

def train_qat_one_epoch(cfg, loader, model, compression_ctrl, criterion, cls_criterion, optimizer, device, loss_weight, num_classes):
    model.train()
    
    # Metrics
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    auprc_macro = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    auprc_per_class = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    sum_seg_loss = 0.0
    sum_cls_loss = 0.0
    sum_total_loss = 0.0
    sum_iou = 0.0
    n_samples = 0

    pbar = tqdm(total=len(loader), desc=f"Epoch {cfg['epoch']} [QAT Train]")
    
    for x, y, cls_y, spacing, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # Forward
        if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSP","UKANClsSSPScaleMLPAffine","UKANClsSSPScaleMLK","UKANClsSSPScaleMLP"]:
            out, cls_out = model(x, spacing)
        else:
            out, cls_out = model(x)
            
        cls_loss = cls_criterion(cls_out, cls_y)
        seg_loss = criterion(out, y)
        loss = seg_loss + loss_weight * cls_loss
        
        # [NNCF] Compression Loss 추가
        loss_compression = compression_ctrl.loss()
        total_loss = loss + loss_compression

        total_loss.backward()
        optimizer.step()
        
        # [NNCF] Scheduler Step (Batch 단위)
        compression_ctrl.scheduler.step()

        # Metrics Update
        iou, dice, _ = iou_score(out, y)
        
        auroc_macro.update(cls_out, cls_y)
        auprc_macro.update(cls_out, cls_y)
        f1_macro.update(cls_out, cls_y)
        acc_metric.update(cls_out, cls_y)
        auroc_per_class.update(cls_out, cls_y)
        auprc_per_class.update(cls_out, cls_y)
        f1_per_class.update(cls_out, cls_y)

        bs = x.size(0)
        sum_seg_loss += float(seg_loss.item()) * bs
        sum_cls_loss += float(cls_loss.item()) * bs
        sum_total_loss += float(total_loss.item()) * bs
        sum_iou += float(iou) * bs
        n_samples += bs

        pbar.set_postfix(OrderedDict(
            loss=sum_total_loss/max(n_samples,1),
            iou=sum_iou/max(n_samples,1)
        ))
        pbar.update(1)
    
    pbar.close()

    # Compute Metrics
    auroc_avg = auroc_macro.compute().item()
    auprc_avg = auprc_macro.compute().item()
    f1_avg = f1_macro.compute().item()
    accuracy = acc_metric.compute().item()
    
    auroc_classes = auroc_per_class.compute().cpu().numpy()
    auprc_classes = auprc_per_class.compute().cpu().numpy()
    f1_classes = f1_per_class.compute().cpu().numpy()

    return OrderedDict(
        loss=sum_total_loss/n_samples,
        seg_loss=sum_seg_loss/n_samples,
        cls_loss=sum_cls_loss/n_samples,
        iou=sum_iou/n_samples,
        auroc_avg=auroc_avg,
        auprc_avg=auprc_avg,
        f1_avg=f1_avg,
        accuracy=accuracy,
        auroc_class0=float(auroc_classes[0]),
        auroc_class1=float(auroc_classes[1]) if num_classes > 1 else 0.0,
        auprc_class0=float(auprc_classes[0]),
        auprc_class1=float(auprc_classes[1]) if num_classes > 1 else 0.0,
        f1_class0=float(f1_classes[0]),
        f1_class1=float(f1_classes[1]) if num_classes > 1 else 0.0,
    )

@torch.no_grad()
def validate_one_epoch(cfg, loader, model, criterion, cls_criterion, device, loss_weight, num_classes):
    model.eval()
    
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    auprc_macro = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    auprc_per_class = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    
    sum_seg_loss = 0.0
    sum_cls_loss = 0.0
    sum_total_loss = 0.0
    sum_iou = 0.0
    sum_dice = 0.0
    n_samples = 0
    
    pbar = tqdm(total=len(loader), desc="[QAT Val]")

    for x, y, cls_y, spacing, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)

        if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSP","UKANClsSSPScaleMLPAffine","UKANClsSSPScaleMLK","UKANClsSSPScaleMLP"]:
            out, cls_out = model(x, spacing)
        else:
            out, cls_out = model(x)
            
        cls_loss = cls_criterion(cls_out, cls_y)
        seg_loss = criterion(out, y)
        loss = seg_loss + loss_weight * cls_loss

        iou, dice, _ = iou_score(out, y)
        
        # Metrics Update
        auroc_macro.update(cls_out, cls_y)
        auprc_macro.update(cls_out, cls_y)
        f1_macro.update(cls_out, cls_y)
        acc_metric.update(cls_out, cls_y)
        auroc_per_class.update(cls_out, cls_y)
        auprc_per_class.update(cls_out, cls_y)
        f1_per_class.update(cls_out, cls_y)

        bs = x.size(0)
        sum_seg_loss += float(seg_loss.item()) * bs
        sum_cls_loss += float(cls_loss.item()) * bs
        sum_total_loss += float(loss.item()) * bs
        sum_iou += float(iou) * bs
        sum_dice += float(dice) * bs
        n_samples += bs
        
        pbar.update(1)
    
    pbar.close()

    # Compute Metrics
    auroc_avg = auroc_macro.compute().item()
    auprc_avg = auprc_macro.compute().item()
    f1_avg = f1_macro.compute().item()
    accuracy = acc_metric.compute().item()
    
    auroc_classes = auroc_per_class.compute().cpu().numpy()
    auprc_classes = auprc_per_class.compute().cpu().numpy()
    f1_classes = f1_per_class.compute().cpu().numpy()

    return OrderedDict(
        loss=sum_total_loss/n_samples,
        seg_loss=sum_seg_loss/n_samples,
        cls_loss=sum_cls_loss/n_samples,
        iou=sum_iou/n_samples,
        dice=sum_dice/n_samples,
        auroc_avg=auroc_avg,
        auprc_avg=auprc_avg,
        f1_avg=f1_avg,
        accuracy=accuracy,
        auroc_class0=float(auroc_classes[0]),
        auroc_class1=float(auroc_classes[1]) if num_classes > 1 else 0.0,
        auprc_class0=float(auprc_classes[0]),
        auprc_class1=float(auprc_classes[1]) if num_classes > 1 else 0.0,
        f1_class0=float(f1_classes[0]),
        f1_class1=float(f1_classes[1]) if num_classes > 1 else 0.0,
    )

# ------------------------- Args -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QAT Training Script")

    # Basics
    p.add_argument('--checkpoint', type=str, required=True, help='Pre-trained FP32 model path')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--epochs', default=5, type=int, help='Epochs for QAT fine-tuning')
    p.add_argument('--lr', default=1e-5, type=float, help='Low LR for QAT')
    p.add_argument('--batch_size', default=8, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--output_dir', default='qat_outputs')

    # Data Overrides
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--mask_dir', type=str, default=None)
    p.add_argument('--cls_df_path', type=str, default=None)
    
    # NNCF
    p.add_argument('--num_init_samples', type=int, default=100, help='Samples for NNCF calibration')

    return p.parse_args()

# ------------------------- Main -------------------------

def main():
    seed_all()
    args = parse_args()
    
    # Load Config
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.yml')
    
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override paths & params
    if args.image_dir: cfg['image_dir'] = args.image_dir
    if args.mask_dir: cfg['mask_dir'] = args.mask_dir
    if args.cls_df_path: cfg['cls_df_path'] = args.cls_df_path
    cfg['batch_size'] = args.batch_size
    
    # Output Dir
    save_dir = os.path.join(args.output_dir, f"{cfg['name']}_QAT")
    os.makedirs(save_dir, exist_ok=True)
    
    tb = SummaryWriter(save_dir)
    print(f"[Info] QAT Results will be saved to: {save_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device: {device}")

    # 1. Build Model (FP32)
    print(f"[Info] Building model: {cfg['arch']}")
    if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSP","UKANClsSSPScaleMLPAffine","UKANClsSSPScaleMLK","UKANClsSSPScaleMLP"]:
        model = models.__dict__[cfg['arch']](
            cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
            embed_dims=cfg['input_list'], no_kan=cfg['no_kan'], num_cls_classes=cfg['num_cls_classes'],
            reduction=cfg['reduction'], pooling_sizes=cfg['pooling_sizes']
        )
    else:
        model = models.__dict__[cfg['arch']](
            cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
            embed_dims=cfg['input_list'], no_kan=cfg['no_kan'], num_cls_classes=cfg['num_cls_classes']
        )
    model.to(device)

    # 2. Load Pre-trained Weights
    print(f"[Info] Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    # 3. Data Loaders
    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
    mirror_axes = (0, 1)
    train_tf = get_training_transforms(image_size=(cfg['input_h'], cfg['input_w']), rotation_for_DA=rotation_for_DA, mirror_axes=mirror_axes)
    val_tf = get_validation_transforms(image_size=(cfg['input_h'], cfg['input_w']))
    
    # Mask ext logic from original code
    mask_ext = '.nii.gz' if cfg['dataset'] == 'ngtube' else '.png'
    if cfg['dataset'] == 'busi': mask_ext = '_mask.png'

    train_ds = Dataset(cfg['image_dir'], cfg['mask_dir'], '_0000.nii.gz', mask_ext, target_size=(cfg['input_h'], cfg['input_w']), cls_df_path=cfg['cls_df_path'], transform=train_tf, mode='train')
    val_ds = Dataset(cfg['image_dir'], cfg['mask_dir'], '_0000.nii.gz', mask_ext, target_size=(cfg['input_h'], cfg['input_w']), cls_df_path=cfg['cls_df_path'], transform=val_tf, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 4. NNCF Configuration
    print("[NNCF] Configuring Quantization...")
    nncf_config_dict = {
        "input_info": [
            {"sample_size": [1, 1, cfg['input_h'], cfg['input_w']]}, # Image
            {"sample_size": [1, 2]}                                  # Spacing
        ],
        "compression": {
            "algorithm": "quantization",
            "preset": "mixed",
            "initializer": {
                "range": {"num_init_samples": args.num_init_samples},
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 50}
            }
        }
    }
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)
    
    # Register Init Args
    calibration_loader = QATCalibrationLoader(train_loader, device)
    nncf_config = register_default_init_args(nncf_config, calibration_loader)

    # 5. Wrap Model for QAT
    print("[NNCF] Creating Compressed Model...")
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    
    # 6. Optimizer & Loss
    optimizer = optim.AdamW(compressed_model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = build_criterion(cfg['loss'])
    cls_criterion = nn.CrossEntropyLoss().to(device)
    
    # QAT Epoch Loop
    best_auroc = 0.0
    best_iou, best_dice, best_auprc, best_f1 = 0.0, 0.0, 0.0, 0.0
    
    log = OrderedDict(
        epoch=[], 
        loss=[], seg_loss=[], cls_loss=[], iou=[],
        auroc_avg=[], auprc_avg=[], f1_avg=[], accuracy=[],
        val_loss=[], val_seg_loss=[], val_cls_loss=[], val_iou=[], val_dice=[],
        val_auroc_avg=[], val_auprc_avg=[], val_f1_avg=[], val_accuracy=[],
        # Per-class logs can be added here as in train.py
    )

    print(f"\n[Info] Starting QAT for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        cfg['epoch'] = epoch
        
        # Train
        tr = train_qat_one_epoch(cfg, train_loader, compressed_model, compression_ctrl, criterion, cls_criterion, optimizer, device, cfg['loss_weight'], cfg['num_cls_classes'])
        
        # Validate
        va = validate_one_epoch(cfg, val_loader, compressed_model, criterion, cls_criterion, device, cfg['loss_weight'], cfg['num_cls_classes'])
        
        # Logging
        log['epoch'].append(epoch)
        # Train
        log['loss'].append(tr['loss']); log['seg_loss'].append(tr['seg_loss']); log['cls_loss'].append(tr['cls_loss'])
        log['iou'].append(tr['iou'])
        log['auroc_avg'].append(tr['auroc_avg']); log['auprc_avg'].append(tr['auprc_avg']); log['f1_avg'].append(tr['f1_avg']); log['accuracy'].append(tr['accuracy'])
        
        # Val
        log['val_loss'].append(va['loss']); log['val_seg_loss'].append(va['seg_loss']); log['val_cls_loss'].append(va['cls_loss'])
        log['val_iou'].append(va['iou']); log['val_dice'].append(va['dice'])
        log['val_auroc_avg'].append(va['auroc_avg']); log['val_auprc_avg'].append(va['auprc_avg']); log['val_f1_avg'].append(va['f1_avg']); log['val_accuracy'].append(va['accuracy'])
        
        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log_qat.csv'), index=False)
        
        # TensorBoard
        tb.add_scalar('train/loss', tr['loss'], epoch)
        tb.add_scalar('train/auroc_avg', tr['auroc_avg'], epoch)
        tb.add_scalar('val/loss', va['loss'], epoch)
        tb.add_scalar('val/auroc_avg', va['auroc_avg'], epoch)
        tb.add_scalar('val/iou', va['iou'], epoch)
        
        # Print
        print(f"Epoch {epoch}: Tr_Loss={tr['loss']:.4f}, Val_AUROC={va['auroc_avg']:.4f}, Val_IoU={va['iou']:.4f}")

        # Save Best (based on AUROC as in train.py)
        if va['auroc_avg'] > best_auroc:
            best_auroc = va['auroc_avg']
            best_iou = va['iou']
            best_dice = va['dice']
            best_auprc = va['auprc_avg']
            best_f1 = va['f1_avg']
            
            save_ckpt(os.path.join(save_dir, 'best_qat.pth'),
                      compressed_model, optimizer, None, epoch,
                      best_iou, best_dice, best_auroc, best_auprc, best_f1, cfg)
            print(f" => Saved Best QAT Checkpoint (AUROC={best_auroc:.4f})")

    # 7. Export to ONNX / OpenVINO
    print("\n[Info] Exporting QAT Model...")
    onnx_path = os.path.join(save_dir, "model_qat.onnx")
    compression_ctrl.export_model(onnx_path)
    print(f" -> Saved ONNX: {onnx_path}")
    
    # Convert to OpenVINO IR
    try:
        import openvino as ov
        print(" -> Converting to OpenVINO IR...")
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, os.path.join(save_dir, "model_qat.xml"), compress_to_fp16=False)
        print(f" -> Saved IR: {os.path.join(save_dir, 'model_qat.xml')}")
    except ImportError:
        print("[Warning] OpenVINO not installed. Skipping IR conversion.")

    tb.close()
    print("[Done] QAT Finished.")

if __name__ == '__main__':
    main()