import argparse
import os
import random
import json
import yaml
from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix
)

import models
from dataset import Dataset
from metrics import iou_score
from utils import str2bool
from get_transforms import get_validation_transforms


# ------------------------- Utils -------------------------

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_checkpoint(path, model, strict=True):
    """Load model checkpoint"""
    ckpt = torch.load(path, map_location='cuda')
    state = ckpt.get('model') or ckpt.get('state_dict') or ckpt

    # Remove 'module.' prefix if present (from DDP training)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)

    # Return checkpoint info if available
    info = {
        'epoch': ckpt.get('epoch', -1),
        'best_iou': ckpt.get('best_iou', 0.0),
        'best_dice': ckpt.get('best_dice', 0.0),
        'best_auroc': ckpt.get('best_auroc', 0.0),
        'best_auprc': ckpt.get('best_auprc', 0.0),
        'best_f1': ckpt.get('best_f1', 0.0),
    }
    return info


# ------------------------- Test -------------------------

@torch.no_grad()
def test(cfg, loader, model, device, num_classes=2, save_predictions=False, output_dir=None):
    """
    Run inference on test set and compute metrics
    """
    model.eval()

    # Initialize torchmetrics
    auroc_macro = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    auprc_macro = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # Per-class metrics
    auroc_per_class = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
    auprc_per_class = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    # Segmentation metrics
    sum_iou = 0.0
    sum_dice = 0.0
    sum_hd95 = 0.0
    n_samples = 0

    # Store predictions for analysis
    all_predictions = []

    pbar = tqdm(total=len(loader), desc='Testing')

    for x, y, cls_y, spacing, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)

        # Forward pass
        if cfg['arch'] in ['UKANClsSSPScale',"UKANClsSSPScaleMLK","UKANClsSSPScaleMLP","UKANClsSSPScaleMLPAffine"]:
            seg_out, cls_out = model(x, spacing)
        else:
            seg_out, cls_out = model(x)

        # Segmentation metrics
        iou, dice, hd95_ = iou_score(seg_out, y)
        bs = x.size(0)
        sum_iou += float(iou) * bs
        sum_dice += float(dice) * bs
        sum_hd95 += float(hd95_) * bs
        n_samples += bs

        # Classification metrics update
        auroc_macro.update(cls_out, cls_y)
        auprc_macro.update(cls_out, cls_y)
        f1_macro.update(cls_out, cls_y)
        acc_metric.update(cls_out, cls_y)
        confusion_matrix.update(cls_out, cls_y)
        auroc_per_class.update(cls_out, cls_y)
        auprc_per_class.update(cls_out, cls_y)
        f1_per_class.update(cls_out, cls_y)

        # Store predictions
        cls_probs = torch.softmax(cls_out, dim=1).cpu().numpy()
        cls_preds = torch.argmax(cls_out, dim=1).cpu().numpy()

        for i, img_id in enumerate(meta['img_id']):
            all_predictions.append({
                'img_id': img_id,
                'cls_target': cls_y[i].cpu().item(),
                'cls_pred': cls_preds[i],
                'cls_prob_0': cls_probs[i][0],
                'cls_prob_1': cls_probs[i][1] if num_classes > 1 else 0.0,
                'iou': float(iou),
                'dice': float(dice),
            })

        pbar.set_postfix(OrderedDict(
            iou=sum_iou / max(n_samples, 1),
            dice=sum_dice / max(n_samples, 1)
        ))
        pbar.update(1)

    pbar.close()

    # Compute final metrics
    auroc_avg = auroc_macro.compute().item()
    auprc_avg = auprc_macro.compute().item()
    f1_avg = f1_macro.compute().item()
    accuracy = acc_metric.compute().item()
    conf_matrix = confusion_matrix.compute().cpu().numpy()

    auroc_classes = auroc_per_class.compute().cpu().numpy()
    auprc_classes = auprc_per_class.compute().cpu().numpy()
    f1_classes = f1_per_class.compute().cpu().numpy()

    avg_iou = sum_iou / max(n_samples, 1)
    avg_dice = sum_dice / max(n_samples, 1)
    avg_hd95 = sum_hd95 / max(n_samples, 1)

    results = OrderedDict(
        # Segmentation metrics
        iou=avg_iou,
        dice=avg_dice,
        hd95=avg_hd95,
        # Classification metrics - macro average
        auroc_avg=auroc_avg,
        auprc_avg=auprc_avg,
        f1_avg=f1_avg,
        accuracy=accuracy,
        # Per-class metrics
        auroc_class0=float(auroc_classes[0]),
        auroc_class1=float(auroc_classes[1]) if num_classes > 1 else 0.0,
        auprc_class0=float(auprc_classes[0]),
        auprc_class1=float(auprc_classes[1]) if num_classes > 1 else 0.0,
        f1_class0=float(f1_classes[0]),
        f1_class1=float(f1_classes[1]) if num_classes > 1 else 0.0,
        # Sample count
        n_samples=n_samples,
    )

    # Save predictions if requested
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save per-sample predictions
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

        # Save confusion matrix
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), conf_matrix)

        # Save results summary
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(dict(results), f, indent=2)

        print(f"Predictions saved to {output_dir}")

    return results, conf_matrix


# ------------------------- Args -------------------------
def list_type(s):
    return [int(a) for a in s.split(',')]

def parse_args():
    p = argparse.ArgumentParser(description='Test script for segmentation + classification model')

    # Required arguments
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pth file)')
    p.add_argument('--config', type=str, default=None,
                   help='Path to config.yml (default: config.yml in checkpoint dir)')

    # Data arguments (override config)
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--mask_dir', type=str, default=None)
    p.add_argument('--cls_df_path', type=str, default=None)
    p.add_argument('--test_mode', type=str, default='test',
                   help='Dataset mode to use (test/val)')


    # Runtime arguments
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--device', type=str, default='cuda')

    # Output arguments
    p.add_argument('--output_dir', type=str, default=None,
                   help='Directory to save results (default: checkpoint dir)')
    p.add_argument('--save_predictions', type=str2bool, default=False)

    return p.parse_args()


def main():
    seed_all()
    args = parse_args()

    # Load config
    if args.config:
        config_path = args.config
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(ckpt_dir, 'config.yml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Loaded config from: {config_path}")

    # Override data paths if provided
    if args.image_dir:
        cfg['image_dir'] = args.image_dir
    if args.mask_dir:
        cfg['mask_dir'] = args.mask_dir
    if args.cls_df_path:
        cfg['cls_df_path'] = args.cls_df_path

    # Print config
    print('-' * 50)
    print('Test Configuration:')
    for key, value in cfg.items():
        print(f'  {key}: {value}')
    print('-' * 50)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model from config
    print(f"Building model: {cfg['arch']}")
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

    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_info = load_checkpoint(args.checkpoint, model, strict=True)
    print(f"Checkpoint info: epoch={ckpt_info['epoch']}, "
          f"best_iou={ckpt_info['best_iou']:.4f}, best_dice={ckpt_info['best_dice']:.4f}, "
          f"best_auroc={ckpt_info['best_auroc']:.4f}")

    # Prepare dataset
    img_ext = '_0000.nii.gz'
    mask_ext = '.nii.gz'

    val_tf = get_validation_transforms(
        image_size=(cfg['input_h'], cfg['input_w'])
    )

    test_ds = Dataset(
        cfg['image_dir'], cfg['mask_dir'],
        img_ext, mask_ext,
        target_size=(cfg['input_h'], cfg['input_w']),
        cls_df_path=cfg['cls_df_path'],
        transform=val_tf,
        mode=args.test_mode
    )

    print(f"Test dataset size: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Output directory setup
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), f'test_results_{args.test_mode}')

    # Run test
    print("\nRunning inference...")
    results, conf_matrix = test(
        cfg, test_loader, model, device,
        num_classes=cfg['num_cls_classes'],
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )

    # ---------------------------------------------------------
    # Generate and Save Report
    # ---------------------------------------------------------
    
    # 1. Construct the result string
    log_message = []
    log_message.append("\n" + "=" * 60)
    log_message.append("TEST RESULTS")
    log_message.append("=" * 60)
    
    log_message.append("\nSegmentation Metrics:")
    log_message.append(f"  IoU:  {results['iou']:.4f}")
    log_message.append(f"  Dice: {results['dice']:.4f}")
    log_message.append(f"  HD95: {results['hd95']:.4f}")

    log_message.append("\nClassification Metrics (Macro Average):")
    log_message.append(f"  AUROC:    {results['auroc_avg']:.4f}")
    log_message.append(f"  AUPRC:    {results['auprc_avg']:.4f}")
    log_message.append(f"  F1:       {results['f1_avg']:.4f}")
    log_message.append(f"  Accuracy: {results['accuracy']:.4f}")

    log_message.append("\nPer-Class Metrics:")
    log_message.append(f"  Class 0 (Complete):   AUROC={results['auroc_class0']:.4f}, AUPRC={results['auprc_class0']:.4f}, F1={results['f1_class0']:.4f}")
    log_message.append(f"  Class 1 (Incomplete): AUROC={results['auroc_class1']:.4f}, AUPRC={results['auprc_class1']:.4f}, F1={results['f1_class1']:.4f}")

    log_message.append("\nConfusion Matrix:")
    log_message.append(f"  Predicted ->")
    log_message.append(f"  Actual v    Complete  Incomplete")
    log_message.append(f"  Complete      {int(conf_matrix[0, 0]):5d}      {int(conf_matrix[0, 1]):5d}")
    log_message.append(f"  Incomplete    {int(conf_matrix[1, 0]):5d}      {int(conf_matrix[1, 1]):5d}")

    log_message.append(f"\nTotal samples: {results['n_samples']}")
    log_message.append("=" * 60)
    
    # Join list into a single string
    full_report = "\n".join(log_message)

    # 2. Print to console
    print(full_report)

    # 3. Save to file
    # Ensure directory exists (might not exist if save_predictions was False)
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w') as f:
        f.write(full_report)
    
    print(f"\n[Info] Full test report saved to: {report_path}")

    if args.save_predictions:
        print(f"[Info] Detailed predictions saved to: {output_dir}")


if __name__ == '__main__':
    main()

