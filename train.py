import argparse, os, random, json, shutil, yaml
from collections import OrderedDict
import numpy as np, pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import albumentations as A

# torchmetrics import 추가
from torchmetrics.classification import (
    MulticlassAUROC, 
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy
)

import models, losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
from get_transforms import get_training_transforms, get_validation_transforms

# ------------------------- Utils -------------------------

def list_type(s):
    return [int(a) for a in s.split(',')]

def seed_all(seed=1029):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

def is_dist_env():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def init_distributed(backend="nccl"):
    if not is_dist_env():
        return False, 0, 0, 1
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def ddp_allreduce_mean(tensors, device):
    """tensors: dict(name -> scalar float). returns dict with global mean."""
    if not dist.is_initialized():
        return tensors
    keys = sorted(tensors.keys())
    vec = torch.tensor([float(tensors[k]) for k in keys],
                       device=device, dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec /= dist.get_world_size()
    return {k: v.item() for k, v in zip(keys, vec)}

def build_criterion(name):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().cuda()
    return losses.__dict__[name]().cuda()

def save_ckpt(path, model, optimizer, scheduler, epoch, best_iou, best_dice, best_auroc, best_auprc, best_f1, config):
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        'epoch': epoch,
        'model': model_to_save.state_dict(),
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

def load_ckpt(path, model, optimizer=None, scheduler=None, strict=True):
    ckpt = torch.load(path, map_location='cuda')
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(ckpt.get('optimizer'), dict):
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and isinstance(ckpt.get('scheduler'), dict):
        scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = int(ckpt.get('epoch', 0)) + 1
    best_iou = float(ckpt.get('best_iou', 0.0))
    best_dice = float(ckpt.get('best_dice', 0.0))
    best_auroc = float(ckpt.get('best_auroc', 0.0))
    best_auprc = float(ckpt.get('best_auprc', 0.0))
    best_f1 = float(ckpt.get('best_f1', 0.0))
    return start_epoch, best_iou, best_dice, best_auroc, best_auprc, best_f1


# ------------------------- Data -------------------------

def make_dataloaders(cfg, distributed, img_ext='_0000.nii.gz', mask_ext='.png'):
    if cfg['dataset'] == 'busi':
        mask_ext = '_mask.png'
    elif cfg['dataset'] in ['glas','cvc','isic2018','isic2017']:
        mask_ext = '.png'
    elif cfg['dataset'] == 'ngtube':
        mask_ext = '.nii.gz'
    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
    mirror_axes = (0, 1)
    train_tf = get_training_transforms(
        image_size=(cfg['input_h'], cfg['input_w']),
        rotation_for_DA=rotation_for_DA,
        mirror_axes=mirror_axes
    )
    val_tf = get_validation_transforms(
        image_size=(cfg['input_h'], cfg['input_w'])
    )

    train_ds = Dataset(cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, target_size=(cfg['input_h'], cfg['input_w']),cls_df_path=cfg['cls_df_path'],
                       transform=train_tf,mode="train")
    val_ds   = Dataset(cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, target_size=(cfg['input_h'], cfg['input_w']),cls_df_path=cfg['cls_df_path'],
                       transform=val_tf, mode="val")

    dl_common = dict(batch_size=cfg['batch_size'], pin_memory=True)
    if cfg['num_workers'] > 0:
        dl_common.update(dict(
            num_workers=cfg['num_workers'],
            persistent_workers=True,
            prefetch_factor=8
        ))
    else:
        dl_common.update(dict(num_workers=0))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_ds, sampler=train_sampler, shuffle=False, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   sampler=val_sampler,   shuffle=False, drop_last=False, **dl_common
        )
    else:
        train_sampler = val_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_ds, shuffle=True, drop_last=True, **dl_common
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,   shuffle=False, drop_last=False, **dl_common
        )

    return train_loader, val_loader, train_sampler, val_sampler


# ------------------------- Optim/Sched -------------------------

def build_optimizer(cfg, model):
    kan_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('layer' in n.lower()) and ('fc' in n.lower()):
            kan_params.append(p)
        else:
            base_params.append(p)
    groups = [
        {'params': base_params, 'lr': cfg['lr'],      'weight_decay': cfg['weight_decay']},
        {'params': kan_params,  'lr': cfg['kan_lr'], 'weight_decay': cfg['kan_weight_decay']},
    ]
    if cfg['optimizer'] == 'Adam':
        return optim.Adam(groups)
    elif cfg['optimizer'] == 'AdamW':
        return optim.AdamW(groups)
    return optim.SGD(groups, lr=cfg['lr'], momentum=cfg['momentum'],
                         nesterov=cfg['nesterov'], weight_decay=cfg['weight_decay'])

def build_scheduler(cfg, optimizer):
    if cfg['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    if cfg['scheduler'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['factor'],
                                              patience=cfg['patience'], verbose=True, min_lr=cfg['min_lr'])
    if cfg['scheduler'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in cfg['milestones'].split(',')], gamma=cfg['gamma'])
    if cfg['scheduler'] == 'ConstantLR':
        return None
    raise NotImplementedError


# ------------------------- Train/Valid (AMP 지원) -------------------------

def select_amp_dtype(mode: str):
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --------------------------------------------------------------------------------
# [수정 1] train_one_epoch 함수 시그니처 변경 (num_classes 추가)
# --------------------------------------------------------------------------------
def train_one_epoch(cfg, loader, model, criterion, optimizer, scaler, amp_dtype, device, is_main, loss_weight, cls_criterion, num_classes, sampler=None):
    if sampler is not None:
        sampler.set_epoch(cfg['epoch'])  # DDP: 매 에폭 셔플 시드 동기화
    model.train()

    # --------------------------------------------------------------------------------
    # [수정 2] train_one_epoch에 torchmetrics 초기화 코드 추가
    # --------------------------------------------------------------------------------
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    auprc_macro = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    auprc_per_class = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    # 글로벌 평균 계산용 누적치
    sum_seg_loss = 0.0
    sum_cls_loss = 0.0
    sum_total_loss = 0.0
    sum_iou = 0.0
    n_samples = 0

    pbar = tqdm(total=len(loader)) if is_main else None
    autocast_enabled = amp_dtype is not None
    
    for x, y, cls_y,spacing, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if cfg['arch'] in ['UKANClsSSPScale']:
                    out, cls_out = model(x,spacing)
                else:
                    out, cls_out = model(x)
                cls_loss = cls_criterion(cls_out, cls_y)
                seg_loss = criterion(out, y)
                loss = seg_loss + loss_weight*cls_loss
        else:
            if cfg['arch'] in ['UKANClsSSPScale']:
                out, cls_out = model(x,spacing)
            else:
                out, cls_out = model(x)
            cls_loss = cls_criterion(cls_out, cls_y)
            seg_loss = criterion(out, y)
            loss = seg_loss + loss_weight*cls_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iou, dice, _ = iou_score(out, y)

        # --------------------------------------------------------------------------------
        # [수정 3] train_one_epoch 루프 내에 metric.update() 호출 추가
        # --------------------------------------------------------------------------------
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
        sum_iou  += float(iou) * bs
        n_samples += bs

        if pbar:
            pbar.set_postfix(OrderedDict(
                loss=sum_total_loss/max(n_samples,1),
                seg_loss=sum_seg_loss/max(n_samples,1),
                cls_loss=sum_cls_loss/max(n_samples,1),
                iou=sum_iou/max(n_samples,1)
            ))
            pbar.update(1)
    if pbar: pbar.close()

    # --------------------------------------------------------------------------------
    # [수정 4] train_one_epoch 루프 종료 후 metric.compute() 호출 추가
    # --------------------------------------------------------------------------------
    auroc_avg = auroc_macro.compute().item()
    auprc_avg = auprc_macro.compute().item()
    f1_avg = f1_macro.compute().item()
    accuracy = acc_metric.compute().item()
    
    auroc_classes = auroc_per_class.compute().cpu().numpy()
    auprc_classes = auprc_per_class.compute().cpu().numpy()
    f1_classes = f1_per_class.compute().cpu().numpy()

    # DDP: 전 랭크 합산 후 평균
    stats = {
        'seg_loss_sum': sum_seg_loss,
        'cls_loss_sum': sum_cls_loss,
        'total_loss_sum': sum_total_loss,
        'iou_sum': sum_iou,
        'n': n_samples
    }
    stats = ddp_allreduce_mean(stats, device)
    tr_seg_loss = stats['seg_loss_sum'] / max(stats['n'], 1)
    tr_cls_loss = stats['cls_loss_sum'] / max(stats['n'], 1)
    tr_total_loss = stats['total_loss_sum'] / max(stats['n'], 1)
    tr_iou = stats['iou_sum'] / max(stats['n'], 1)
    
    # --------------------------------------------------------------------------------
    # [수정 5] train_one_epoch 반환 OrderedDict에 classification metric 추가
    # --------------------------------------------------------------------------------
    return OrderedDict(
        loss=tr_total_loss,
        seg_loss=tr_seg_loss,
        cls_loss=tr_cls_loss,
        iou=tr_iou,
        # Classification metrics
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
def validate_one_epoch(cfg, loader, model, criterion, amp_dtype, device, is_main, loss_weight, cls_criterion, num_classes=2):
    model.eval()
    
    # torchmetrics 메트릭 초기화
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    auprc_macro = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    # Per-class metrics
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    auprc_per_class = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    
    sum_seg_loss = 0.0
    sum_cls_loss = 0.0
    sum_total_loss = 0.0
    sum_iou = 0.0
    sum_dice = 0.0
    n_samples = 0
    
    pbar = tqdm(total=len(loader)) if is_main else None
    autocast_enabled = amp_dtype is not None

    for x, y, cls_y,spacing, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)

        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                if cfg['arch'] in ['UKANClsSSPScale']:
                    out, cls_out = model(x,spacing)
                else:
                    out, cls_out = model(x)
                cls_loss = cls_criterion(cls_out, cls_y)
                seg_loss = criterion(out, y)
                loss = seg_loss + loss_weight*cls_loss
        else:
            if cfg['arch'] in ['UKANClsSSPScale']:
                out, cls_out = model(x,spacing)
            else:
                out, cls_out = model(x)
            cls_loss = cls_criterion(cls_out, cls_y)
            seg_loss = criterion(out, y)
            loss = seg_loss + loss_weight*cls_loss

        iou, dice, _ = iou_score(out, y)
        bs = x.size(0)
        sum_seg_loss += float(seg_loss.item()) * bs
        sum_cls_loss += float(cls_loss.item()) * bs
        sum_total_loss += float(loss.item()) * bs
        sum_iou  += float(iou) * bs
        sum_dice += float(dice) * bs
        n_samples += bs
        
        # torchmetrics update (배치별 누적, 자동으로 DDP sync)
        auroc_macro.update(cls_out, cls_y)
        auprc_macro.update(cls_out, cls_y)
        f1_macro.update(cls_out, cls_y)
        acc_metric.update(cls_out, cls_y)
        auroc_per_class.update(cls_out, cls_y)
        auprc_per_class.update(cls_out, cls_y)
        f1_per_class.update(cls_out, cls_y)
        
        if pbar:
            pbar.set_postfix(OrderedDict(
                loss=sum_total_loss/max(n_samples,1),
                seg_loss=sum_seg_loss/max(n_samples,1),
                cls_loss=sum_cls_loss/max(n_samples,1),
                iou=sum_iou/max(n_samples,1),
                dice=sum_dice/max(n_samples,1)
            ))
            pbar.update(1)
    if pbar: pbar.close()

    # torchmetrics compute (자동으로 DDP sync 완료)
    auroc_avg = auroc_macro.compute().item()
    auprc_avg = auprc_macro.compute().item()
    f1_avg = f1_macro.compute().item()
    accuracy = acc_metric.compute().item()
    
    auroc_classes = auroc_per_class.compute().cpu().numpy()
    auprc_classes = auprc_per_class.compute().cpu().numpy()
    f1_classes = f1_per_class.compute().cpu().numpy()

    # Loss/IoU/Dice allreduce
    stats = {
        'seg_loss_sum': sum_seg_loss,
        'cls_loss_sum': sum_cls_loss,
        'total_loss_sum': sum_total_loss,
        'iou_sum': sum_iou,
        'dice_sum': sum_dice,
        'n': n_samples
    }
    stats = ddp_allreduce_mean(stats, device)
    
    va_seg_loss = stats['seg_loss_sum'] / max(stats['n'], 1)
    va_cls_loss = stats['cls_loss_sum'] / max(stats['n'], 1)
    va_total_loss = stats['total_loss_sum'] / max(stats['n'], 1)
    va_iou = stats['iou_sum'] / max(stats['n'], 1)
    va_dice = stats['dice_sum'] / max(stats['n'], 1)
    
    result = OrderedDict(
        loss=va_total_loss,
        seg_loss=va_seg_loss,
        cls_loss=va_cls_loss,
        iou=va_iou,
        dice=va_dice,
        # Classification metrics
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
    
    return result


# ------------------------- Args -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # basics
    p.add_argument('--name', default=None)
    p.add_argument('--epochs', default=300, type=int)
    p.add_argument('-b', '--batch_size', default=16, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--output_dir', default='outputs')

    # data
    p.add_argument('--dataset', default='busi')
    p.add_argument('--image_dir', required=True)
    p.add_argument('--mask_dir',  required=True)
    p.add_argument('--splits_final', type=str, required=True)
    p.add_argument('--cls_df_path', type=str, default=None)
    p.add_argument('--num_cls_classes', type=int, default=2, help='Number of classification classes')

    # model
    p.add_argument('--arch', default='UKAN')
    p.add_argument('--deep_supervision', default=False, type=str2bool)
    p.add_argument('--input_channels', default=3, type=int)
    p.add_argument('--num_classes', default=1, type=int)
    p.add_argument('--input_w', default=1024, type=int)
    p.add_argument('--input_h', default=1024, type=int)
    p.add_argument('--input_list', type=list_type, default=[128,160,256])
    p.add_argument('--no_kan', action='store_true')
    
    # loss
    LOSS_NAMES = losses.__all__ + ['BCEWithLogitsLoss']
    p.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    p.add_argument('--loss_weight', type=float, default=1.0)

    # optim
    p.add_argument('--optimizer', default='AdamW', choices=['AdamW','SGD',"Adam"])
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--nesterov', default=False, type=str2bool)
    p.add_argument('--kan_lr', default=1e-4, type=float)
    p.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # scheduler
    p.add_argument('--scheduler', default='CosineAnnealingLR',
                    choices=['CosineAnnealingLR','ReduceLROnPlateau','MultiStepLR','ConstantLR'])
    p.add_argument('--min_lr', default=1e-5, type=float)
    p.add_argument('--factor', default=0.1, type=float)
    p.add_argument('--patience', default=2, type=int)
    p.add_argument('--milestones', default='1,2', type=str)
    p.add_argument('--gamma', default=2/3, type=float)
    p.add_argument('--early_stopping', default=-1, type=int)

    # resume
    p.add_argument('--resume', type=str, default='',
                     help='checkpoint(.pth/.pt/.tar) 경로. 비우면 신규 학습')
    p.add_argument('--resume_strict', type=str2bool, default=True)
    p.add_argument('--resume_optim', type=str2bool, default=True)
    p.add_argument('--resume_sched', type=str2bool, default=True)

    # DDP/AMP
    p.add_argument('--ddp_backend', default='nccl', choices=['nccl','gloo','mpi'])
    p.add_argument('--bucket_cap_mb', type=int, default=None,
                     help='DDP gradient bucket size (MiB). None=PyTorch default(25MiB).')
    p.add_argument('--amp_dtype', default='auto', choices=['auto','bf16','fp16','off'],
                     help='auto: bf16 if supported else fp16')
    return p.parse_args()


# ------------------------- Main -------------------------

def main():
    seed_all()
    cfg = vars(parse_args())

    # DDP init (no-op on single GPU)
    distributed, rank, local_rank, world_size = init_distributed(cfg['ddp_backend'])
    is_main = (rank == 0)

    # 실험 폴더
    if cfg['name'] is None:
        cfg['name'] = f"{cfg['dataset']}_{cfg['arch']}_{'wDS' if cfg['deep_supervision'] else 'woDS'}"
    save_dir = os.path.join(cfg['output_dir'], cfg['name'])
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
            yaml.dump(cfg, f)
    if distributed: dist.barrier()

    tb = SummaryWriter(save_dir) if is_main else None

    # 모델/손실/옵티마/스케줄러
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[cfg['arch']](
        cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
        embed_dims=cfg['input_list'], no_kan=cfg['no_kan'], num_cls_classes=cfg['num_cls_classes']
    ).to(device)
    criterion = build_criterion(cfg['loss'])
    cls_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # --------- Resume (wrap 전에) ---------
    start_epoch = 0
    best_iou, best_dice = 0.0, 0.0
    best_auroc, best_auprc, best_f1 = 0.0, 0.0, 0.0
    
    if cfg['resume']:
        start_epoch, best_iou, best_dice, best_auroc, best_auprc, best_f1 = load_ckpt(
            cfg['resume'],
            model,
            optimizer if cfg['resume_optim'] else None,
            scheduler if (cfg['resume_sched'] and scheduler is not None) else None,
            strict=cfg['resume_strict']
        )
        if is_main:
            print(f"=> resumed from {cfg['resume']} | start_epoch={start_epoch} | "
                  f"best_iou={best_iou:.4f} | best_dice={best_dice:.4f} | "
                  f"best_auroc={best_auroc:.4f} | best_auprc={best_auprc:.4f} | best_f1={best_f1:.4f}")

    # DDP 래핑 (bucket_cap_mb로 통신-계산 겹침 조절)
    if distributed:
        ddp_kwargs = {}
        if cfg['bucket_cap_mb'] is not None:
            ddp_kwargs['bucket_cap_mb'] = cfg['bucket_cap_mb']
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)

    # 데이터
    train_loader, val_loader, train_sampler, val_sampler = make_dataloaders(cfg, distributed)

    # 코드 백업(선택)
    if is_main:
        for fname in ['train.py','archs.py']:
            if os.path.exists(fname):
                shutil.copy2(fname, save_dir)

    # AMP 설정
    amp_dtype = select_amp_dtype(cfg['amp_dtype'])
    use_scaler = (amp_dtype == torch.float16)  # FP16일 때만 권장
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # --------- Train Loop ---------
    # --------------------------------------------------------------------------------
    # [수정 6] main의 log OrderedDict 정의에 train classification metric 추가
    # --------------------------------------------------------------------------------
    log = OrderedDict(
        epoch=[], lr=[],
        # Train metrics
        loss=[], seg_loss=[], cls_loss=[], iou=[],
        auroc_avg=[], auprc_avg=[], f1_avg=[], accuracy=[],
        auroc_class0=[], auroc_class1=[],
        auprc_class0=[], auprc_class1=[],
        f1_class0=[], f1_class1=[],
        # Validation metrics
        val_loss=[], val_seg_loss=[], val_cls_loss=[], val_iou=[], val_dice=[],
        val_auroc_avg=[], val_auprc_avg=[], val_f1_avg=[], val_accuracy=[],
        val_auroc_class0=[], val_auroc_class1=[],
        val_auprc_class0=[], val_auprc_class1=[],
        val_f1_class0=[], val_f1_class1=[]
    )
    trigger = 0

    for epoch in range(start_epoch, cfg['epochs']):
        cfg['epoch'] = epoch  # sampler.set_epoch에 사용
        if is_main:
            print(f"Epoch [{epoch}/{cfg['epochs']}]")

        # --------------------------------------------------------------------------------
        # [수정 7] main의 train_one_epoch 호출 시 num_classes=cfg['num_cls_classes'] 전달
        # --------------------------------------------------------------------------------
        tr = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scaler, 
                             amp_dtype, device, is_main, loss_weight=cfg['loss_weight'],
                             cls_criterion=cls_criterion, num_classes=cfg['num_cls_classes'], 
                             sampler=train_sampler)
        
        va = validate_one_epoch(cfg, val_loader, model, criterion, amp_dtype, device, is_main, 
                                num_classes=cfg['num_cls_classes'], loss_weight=cfg['loss_weight'],
                                cls_criterion=cls_criterion)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va['loss'])
            else:
                scheduler.step()

        if is_main:
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            log['epoch'].append(epoch)
            log['lr'].append(current_lrs)
            
            # --------------------------------------------------------------------------------
            # [수정 8] main의 log.append() 호출에 train classification metric 추가
            # --------------------------------------------------------------------------------
            # Training metrics
            log['loss'].append(tr['loss'])
            log['seg_loss'].append(tr['seg_loss'])
            log['cls_loss'].append(tr['cls_loss'])
            log['iou'].append(tr['iou'])
            log['auroc_avg'].append(tr['auroc_avg'])
            log['auprc_avg'].append(tr['auprc_avg'])
            log['f1_avg'].append(tr['f1_avg'])
            log['accuracy'].append(tr['accuracy'])
            log['auroc_class0'].append(tr['auroc_class0'])
            log['auroc_class1'].append(tr['auroc_class1'])
            log['auprc_class0'].append(tr['auprc_class0'])
            log['auprc_class1'].append(tr['auprc_class1'])
            log['f1_class0'].append(tr['f1_class0'])
            log['f1_class1'].append(tr['f1_class1'])
            
            # Validation metrics
            log['val_loss'].append(va['loss'])
            log['val_seg_loss'].append(va['seg_loss'])
            log['val_cls_loss'].append(va['cls_loss'])
            log['val_iou'].append(va['iou'])
            log['val_dice'].append(va['dice'])
            log['val_auroc_avg'].append(va['auroc_avg'])
            log['val_auprc_avg'].append(va['auprc_avg'])
            log['val_f1_avg'].append(va['f1_avg'])
            log['val_accuracy'].append(va['accuracy'])
            log['val_auroc_class0'].append(va['auroc_class0'])
            log['val_auroc_class1'].append(va['auroc_class1'])
            log['val_auprc_class0'].append(va['auprc_class0'])
            log['val_auprc_class1'].append(va['auprc_class1'])
            log['val_f1_class0'].append(va['f1_class0'])
            log['val_f1_class1'].append(va['f1_class1'])
            
            pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log.csv'), index=False)

            # --------------------------------------------------------------------------------
            # [수정 9] main의 TensorBoard (tb) 로깅에 train classification metric 추가
            # --------------------------------------------------------------------------------
            if tb is not None:
                # Training
                tb.add_scalar('train/loss', tr['loss'], epoch)
                tb.add_scalar('train/seg_loss', tr['seg_loss'], epoch)
                tb.add_scalar('train/cls_loss', tr['cls_loss'], epoch)
                tb.add_scalar('train/iou', tr['iou'], epoch)
                tb.add_scalar('train/auroc_avg', tr['auroc_avg'], epoch)
                tb.add_scalar('train/auprc_avg', tr['auprc_avg'], epoch)
                tb.add_scalar('train/f1_avg', tr['f1_avg'], epoch)
                tb.add_scalar('train/accuracy', tr['accuracy'], epoch)
                tb.add_scalar('train/auroc_class0', tr['auroc_class0'], epoch)
                tb.add_scalar('train/auroc_class1', tr['auroc_class1'], epoch)
                tb.add_scalar('train/auprc_class0', tr['auprc_class0'], epoch)
                tb.add_scalar('train/auprc_class1', tr['auprc_class1'], epoch)
                tb.add_scalar('train/f1_class0', tr['f1_class0'], epoch)
                tb.add_scalar('train/f1_class1', tr['f1_class1'], epoch)
                
                # Validation - Segmentation
                tb.add_scalar('val/loss', va['loss'], epoch)
                tb.add_scalar('val/seg_loss', va['seg_loss'], epoch)
                tb.add_scalar('val/cls_loss', va['cls_loss'], epoch)
                tb.add_scalar('val/iou', va['iou'], epoch)
                tb.add_scalar('val/dice', va['dice'], epoch)
                
                # Validation - Classification
                tb.add_scalar('val/auroc_avg', va['auroc_avg'], epoch)
                tb.add_scalar('val/auprc_avg', va['auprc_avg'], epoch)
                tb.add_scalar('val/f1_avg', va['f1_avg'], epoch)
                tb.add_scalar('val/accuracy', va['accuracy'], epoch)
                tb.add_scalar('val/auroc_class0', va['auroc_class0'], epoch)
                tb.add_scalar('val/auroc_class1', va['auroc_class1'], epoch)
                tb.add_scalar('val/auprc_class0', va['auprc_class0'], epoch)
                tb.add_scalar('val/auprc_class1', va['auprc_class1'], epoch)
                tb.add_scalar('val/f1_class0', va['f1_class0'], epoch)
                tb.add_scalar('val/f1_class1', va['f1_class1'], epoch)
                
                # Best values
                tb.add_scalar('val/best_iou_value', best_iou, epoch)
                tb.add_scalar('val/best_dice_value', best_dice, epoch)
                tb.add_scalar('val/best_auroc_value', best_auroc, epoch)
                tb.add_scalar('val/best_auprc_value', best_auprc, epoch)
                tb.add_scalar('val/best_f1_value', best_f1, epoch)

            # 체크포인트: last (rank0만)
            save_ckpt(os.path.join(save_dir, 'last.pth'),
                      model, optimizer, scheduler, epoch, 
                      best_iou, best_dice, best_auroc, best_auprc, best_f1, cfg)

        # 베스트 모델 저장 (Dice 기준, rank0만)
        if is_main and va['dice'] > best_dice:
            best_iou = va['iou']
            best_dice = va['dice']
            best_auroc = va['auroc_avg']
            best_auprc = va['auprc_avg']
            best_f1 = va['f1_avg']
            trigger = 0
            
            save_ckpt(os.path.join(save_dir, 'best.pth'),
                      model, optimizer, scheduler, epoch,
                      best_iou, best_dice, best_auroc, best_auprc, best_f1, cfg)
            print(f"=> saved BEST checkpoint | IoU={best_iou:.4f} | Dice={best_dice:.4f} | "
                  f"AUROC={best_auroc:.4f} | AUPRC={best_auprc:.4f} | F1={best_f1:.4f}")
        else:
            trigger += 1

        if cfg['early_stopping'] >= 0 and trigger >= cfg['early_stopping']:
            if is_main: print("=> early stopping")
            break

    if tb is not None:
        tb.close()
    cleanup_distributed()


if __name__ == '__main__':
    main()