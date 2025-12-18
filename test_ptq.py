import argparse
import os
import random
import time
import json
import yaml
from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# GFLOPS 측정용 라이브러리
try:
    from thop import profile, file_count
except ImportError:
    print("[Warning] 'thop' library not found. GFLOPS calculation will be skipped.")
    print("Run 'pip install thop' to enable GFLOPS measurement.")

# Metrics
from torchmetrics.classification import (
    MulticlassAUROC, MulticlassAveragePrecision, MulticlassF1Score,
    MulticlassAccuracy, MulticlassConfusionMatrix
)

# OpenVINO & NNCF
import openvino as ov
import nncf
from nncf import quantize
from nncf.parameters import TargetDevice

# Custom Modules
import models
from dataset import Dataset
from metrics import iou_score
from utils import str2bool
from get_transforms import get_validation_transforms


# ------------------------- Utils & Wrappers -------------------------

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gflops(model, input_size=(1, 1, 1024, 1024), extra_input=None):
    """
    thop 라이브러리를 사용하여 GFLOPS 측정
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        
        # 더미 입력 생성
        dummy_input = torch.randn(input_size).to(device)
        
        # Spacing 등 추가 입력이 있는 경우 처리
        if extra_input is not None:
            # thop은 입력이 여러 개일 때 튜플로 전달해야 함 (args=(x, spacing))
            # 하지만 thop 버전별로 지원 방식이 다를 수 있어 가장 안전한 방법은 커스텀 래퍼를 쓰는 것임
            # 여기서는 단순화를 위해 args 튜플 전달 시도
            dummy_spacing = extra_input.to(device)
            macs, params = profile(model, inputs=(dummy_input, dummy_spacing), verbose=False)
        else:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            
        gflops = macs / 1e9 * 2  # MACs -> FLOPs (보통 1 MAC = 2 FLOPs)
        return gflops
    except Exception as e:
        print(f"[Warning] Failed to calculate GFLOPS: {e}")
        return 0.0

def get_file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return size_in_bytes / (1024 * 1024)  # Convert to MB

class OpenVINOModelWrapper:
    def __init__(self, compiled_model, device):
        self.model = compiled_model
        self.device = device
        self.input_layer_img = self.model.inputs[0]
        self.input_layer_spacing = self.model.inputs[1]
        self.output_layer_seg = self.model.outputs[0] 
        self.output_layer_cls = self.model.outputs[1]

    def __call__(self, x, spacing=None):
        x_np = x.cpu().numpy()
        spacing_np = spacing.cpu().numpy() if spacing is not None else None

        inputs = {
            self.input_layer_img: x_np,
            self.input_layer_spacing: spacing_np
        }
        results = self.model(inputs)

        seg_out = torch.from_numpy(results[self.output_layer_seg]).to(self.device)
        cls_out = torch.from_numpy(results[self.output_layer_cls]).to(self.device)

        return seg_out, cls_out
    
    def eval(self):
        pass

# ------------------------- Test Loop -------------------------

@torch.no_grad()
def test(cfg, loader, model, device, num_classes=2):
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    sum_iou = 0.0
    sum_dice = 0.0
    sum_hd95 = 0.0
    n_samples = 0

    pbar = tqdm(total=len(loader), desc='Evaluating Model')

    for x, y, cls_y, spacing, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)

        if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSPScaleMLK","UKANClsSSPScaleMLP","UKANClsSSPScaleMLPAffine"]:
            seg_out, cls_out = model(x, spacing)
        else:
            seg_out, cls_out = model(x)

        iou, dice, hd95_ = iou_score(seg_out, y)
        bs = x.size(0)
        sum_iou += float(iou) * bs
        sum_dice += float(dice) * bs
        sum_hd95 += float(hd95_) * bs
        n_samples += bs

        auroc_macro.update(cls_out, cls_y)
        f1_macro.update(cls_out, cls_y)
        acc_metric.update(cls_out, cls_y)

        pbar.update(1)

    pbar.close()

    results = OrderedDict(
        iou=sum_iou / max(n_samples, 1),
        dice=sum_dice / max(n_samples, 1),
        hd95=sum_hd95 / max(n_samples, 1),
        auroc_avg=auroc_macro.compute().item(),
        f1_avg=f1_macro.compute().item(),
        accuracy=acc_metric.compute().item()
    )
    
    return results

# ------------------------- Main Pipeline -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='OpenVINO Quantization & Evaluation')
    
    p.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--mask_dir', type=str, default=None)
    p.add_argument('--cls_df_path', type=str, default=None)
    
    p.add_argument('--output_dir', type=str, default='quantized_result')
    p.add_argument('--calib_samples', type=int, default=100, help='Number of samples for NNCF calibration')
    p.add_argument('--num_threads', type=int, default=8, help='OpenVINO inference threads')
    
    return p.parse_args()

def main():
    seed_all()
    args = parse_args()
    
    # 1. Config & Checkpoint Setup
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.yml')
    
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.image_dir: cfg['image_dir'] = args.image_dir
    if args.mask_dir: cfg['mask_dir'] = args.mask_dir
    if args.cls_df_path: cfg['cls_df_path'] = args.cls_df_path

    print(f"\n[1] Initializing PyTorch Model: {cfg['arch']}")
    
    if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSPScaleMLK","UKANClsSSPScaleMLP","UKANClsSSPScaleMLPAffine"]:
        model = models.__dict__[cfg['arch']](
            cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
            embed_dims=cfg['input_list'], no_kan=cfg.get('no_kan', False),
            num_cls_classes=cfg['num_cls_classes'],
            reduction=cfg['reduction'], pooling_sizes=cfg['pooling_sizes']
        )
    else:
        model = models.__dict__[cfg['arch']](
            cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
            embed_dims=cfg['input_list'], no_kan=cfg.get('no_kan', False),
            num_cls_classes=cfg['num_cls_classes']
        )

    # Load Weights
    print(f" -> Loading weights from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model') or ckpt.get('state_dict') or ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 파라미터 수 계산
    orig_params = count_parameters(model)
    print(f" -> Original PyTorch Parameters: {orig_params:,}")

    # -------------------------------------------------------------------------
    # [NEW] GFLOPS Calculation
    # -------------------------------------------------------------------------
    print("\n[1-1] Calculating GFLOPS...")
    # 모델에 필요한 추가 입력(Spacing) 준비
    dummy_spacing = torch.randn(1, 2) if 'SSPScale' in cfg['arch'] else None
    gflops = get_gflops(model, input_size=(1, 1, cfg['input_h'], cfg['input_w']), extra_input=dummy_spacing)
    print(f" -> Estimated GFLOPS: {gflops:.4f}")

    # -------------------------------------------------------------------------
    # 2. Calibration Data Preparation
    # -------------------------------------------------------------------------
    print("\n[2] Preparing VALIDATION Dataset for Calibration")
    val_tf = get_validation_transforms(image_size=(cfg['input_h'], cfg['input_w']))
    
    val_ds = Dataset(
        cfg['image_dir'], cfg['mask_dir'], '_0000.nii.gz', '.nii.gz',
        target_size=(cfg['input_h'], cfg['input_w']),
        cls_df_path=cfg['cls_df_path'], transform=val_tf, 
        mode='val'
    )
    
    if len(val_ds) < args.calib_samples:
        calib_indices = range(len(val_ds))
    else:
        calib_indices = np.random.choice(len(val_ds), args.calib_samples, replace=False)
        
    calib_ds = Subset(val_ds, calib_indices)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False, num_workers=4)
    print(f" -> Calibration samples: {len(calib_ds)}")

    # -------------------------------------------------------------------------
    # 3. OpenVINO Conversion & Quantization (PTQ)
    # -------------------------------------------------------------------------
    print("\n[3] Converting & Quantizing to OpenVINO INT8")
    
    dummy_img = torch.randn(1, 1, cfg['input_h'], cfg['input_w'])
    dummy_spacing_ov = torch.randn(1, 2)
    
    # FP32 변환
    ov_model = ov.convert_model(model, example_input=(dummy_img, dummy_spacing_ov))
    
    def transform_fn(data_item):
        img, _, _, spacing, _ = data_item 
        return {
            ov_model.inputs[0].any_name: img.numpy(),
            ov_model.inputs[1].any_name: spacing.numpy()
        }
    
    calibration_dataset = nncf.Dataset(calib_loader, transform_fn)
    
    # Quantize
    quantized_model = quantize(
        ov_model,
        calibration_dataset,
        subset_size=len(calib_ds),
        target_device=TargetDevice.CPU
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_path_xml = os.path.join(args.output_dir, 'quantized_model.xml')
    ov.save_model(quantized_model, model_path_xml)
    
    model_size_mb = get_file_size(os.path.join(args.output_dir, 'quantized_model.bin'))
    print(f" -> Quantized Model Saved: {model_path_xml}")
    print(f" -> Model Size (INT8 .bin): {model_size_mb:.2f} MB")

    # -------------------------------------------------------------------------
    # 4. Benchmarking Speed
    # -------------------------------------------------------------------------
    print("\n[4] Benchmarking Inference Speed (OpenVINO)")
    ie = ov.Core()
    ie.set_property("CPU", {"INFERENCE_NUM_THREADS": args.num_threads})
    compiled_model = ie.compile_model(quantized_model, "CPU")
    
    input_dict = {
        compiled_model.inputs[0]: dummy_img.numpy(),
        compiled_model.inputs[1]: dummy_spacing_ov.numpy()
    }
    for _ in range(20): compiled_model(input_dict)
    
    num_iter = 200
    start = time.perf_counter()
    for _ in range(num_iter):
        compiled_model(input_dict)
    end = time.perf_counter()
    
    total_time = end - start
    avg_latency = (total_time / num_iter) * 1000
    fps = num_iter / total_time
    
    print(f" -> Average Latency: {avg_latency:.2f} ms")
    print(f" -> Throughput: {fps:.2f} FPS")

    # -------------------------------------------------------------------------
    # 5. Evaluate Accuracy
    # -------------------------------------------------------------------------
    print("\n[5] Evaluating Accuracy on Full TEST Set")
    
    test_ds = Dataset(
        cfg['image_dir'], cfg['mask_dir'], '_0000.nii.gz', '.nii.gz',
        target_size=(cfg['input_h'], cfg['input_w']),
        cls_df_path=cfg['cls_df_path'], transform=val_tf, 
        mode='test'
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f" -> Test dataset size: {len(test_ds)}")
    
    wrapped_model = OpenVINOModelWrapper(compiled_model, device=torch.device('cpu'))
    
    results = test(cfg, test_loader, wrapped_model, device=torch.device('cpu'), num_classes=cfg['num_cls_classes'])
    
    # -------------------------------------------------------------------------
    # Final Report
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(" FINAL PERFORMANCE REPORT")
    print("="*50)
    print(f"1. Model Complexity:")
    print(f"   - Original Params: {orig_params:,}")
    print(f"   - GFLOPS (Est.):   {gflops:.4f}")
    print(f"   - Quantized Size:  {model_size_mb:.2f} MB")
    print("-" * 50)
    print(f"2. Inference Speed (CPU, {args.num_threads} threads):")
    print(f"   - Latency: {avg_latency:.2f} ms")
    print(f"   - FPS:     {fps:.2f}")
    print("-" * 50)
    print(f"3. Accuracy Metrics (INT8 Model on TEST Set):")
    print(f"   - IoU:      {results['iou']:.4f}")
    print(f"   - Dice:     {results['dice']:.4f}")
    print(f"   - HD95:     {results['hd95']:.4f}")
    print(f"   - AUROC:    {results['auroc_avg']:.4f}")
    print(f"   - F1 Score: {results['f1_avg']:.4f}")
    print(f"   - Accuracy: {results['accuracy']:.4f}")
    print("="*50)

    report = {
        'original_params': orig_params,
        'gflops': gflops,
        'quantized_size_mb': model_size_mb,
        'latency_ms': avg_latency,
        'fps': fps,
        'metrics': results
    }
    with open(os.path.join(args.output_dir, 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
if __name__ == '__main__':
    main()