#!/usr/bin/env python3
"""
SlideCheck Phase 1 Training - Unified Script

Supports multiple training variants:
- baseline: Pure baseline (LayerNorm + 2-layer MLP)
- mixup: Baseline + Mixup augmentation
- feat_perturb: Baseline + Feature Perturbation
- mixup_fp: Baseline + Mixup + Feature Perturbation (BEST)
- concatv2: ConcatV2 architecture (raw + zscore)
- concatv2_mixup: ConcatV2 + Mixup
- concatv2_mixup_fp: ConcatV2 + Mixup + FP

Usage:
    python tools/train_phase1.py --variant mixup_fp --feat_dir /path/to/features
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.models import build_slidecheck_model
from slidecheck.utils import save_checkpoint


class SlideCheckDataset(Dataset):
    """Patch-level dataset for SlideCheck training"""

    def __init__(self, embeddings, na_labels, cc_labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.na_labels = torch.from_numpy(na_labels).long()
        self.cc_labels = torch.from_numpy(cc_labels).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.na_labels[idx], self.cc_labels[idx]


def load_merged_data(feat_dir):
    """Load and merge 8 datasets"""
    files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.pt')])

    all_embeddings = []
    all_na_labels = []
    all_cc_labels = []

    print(f"\nLoading datasets from {feat_dir}:")
    for f in files:
        path = os.path.join(feat_dir, f)
        data = torch.load(path, map_location='cpu', weights_only=False)

        emb = data['embeddings']
        na_labels = data['normal_abnormal_labels']
        cc_labels = data['cancer_noncancer_labels']

        all_embeddings.append(emb)
        all_na_labels.append(na_labels)
        all_cc_labels.append(cc_labels)

        print(f"  {f}: {emb.shape[0]} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)
    na_labels = np.concatenate(all_na_labels, axis=0)
    cc_labels = np.concatenate(all_cc_labels, axis=0)

    print(f"\nMerged total: {embeddings.shape[0]} samples")
    print(f"  Normal: {(na_labels == 0).sum()} | Abnormal: {(na_labels == 1).sum()}")
    print(f"  Non-cancer: {(cc_labels == 0).sum()} | Cancer: {(cc_labels == 1).sum()}")

    return embeddings, na_labels, cc_labels


def compute_metrics(y_true, y_pred, y_score):
    """Compute classification metrics"""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 1:
        auc = 1.0 if acc == 1.0 else 0.0
        auprc = 1.0 if acc == 1.0 else 0.0
    else:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'acc': float(acc),
        'bacc': float(bacc),
        'auc': float(auc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }


def apply_feature_perturbation(embeddings, sigma=0.1):
    """Apply feature perturbation: x' = x * (1 + noise)"""
    noise = torch.randn_like(embeddings) * sigma
    return embeddings * (1 + noise)


def apply_mixup(embeddings, labels_a, labels_b, alpha=0.4):
    """Apply Mixup augmentation"""
    batch_size = embeddings.size(0)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

    index = torch.randperm(batch_size, device=embeddings.device)

    mixed_emb = lam * embeddings + (1 - lam) * embeddings[index]
    mixed_labels_a = lam * labels_a + (1 - lam) * labels_a[index]
    mixed_labels_b = lam * labels_b + (1 - lam) * labels_b[index]

    return mixed_emb, mixed_labels_a, mixed_labels_b


def train_epoch(model, loader, optimizer, device, args, pos_weight_abn=None, pos_weight_can=None):
    """Training epoch with configurable augmentations"""
    model.train()
    total_loss = 0

    criterion_abn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_abn)
    criterion_can = nn.BCEWithLogitsLoss(pos_weight=pos_weight_can)

    for embeddings, na_labels, cc_labels in tqdm(loader, desc='Train', leave=False):
        embeddings = embeddings.to(device)
        na_labels = na_labels.to(device).float()
        cc_labels = cc_labels.to(device).float()

        # Apply augmentations based on variant
        if 'feat_perturb' in args.variant or 'fp' in args.variant:
            embeddings = apply_feature_perturbation(embeddings, args.perturb_sigma)

        if 'mixup' in args.variant:
            embeddings, na_labels, cc_labels = apply_mixup(
                embeddings, na_labels, cc_labels, args.mixup_alpha
            )

        optimizer.zero_grad()
        output = model(embeddings)
        logit_abn = output.logit_abn
        logit_can = output.logit_can

        loss_abn = criterion_abn(logit_abn, na_labels)
        loss_can = criterion_can(logit_can, cc_labels)

        # Constraint loss: cancer => abnormal
        p_abn = torch.sigmoid(logit_abn)
        p_can = torch.sigmoid(logit_can)
        constraint = F.relu(p_can - p_abn)
        loss_constraint = (constraint ** 2).mean()

        loss = loss_abn + loss_can + args.lambda_constraint * loss_constraint
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluation without augmentation"""
    model.eval()

    all_logit_abn = []
    all_logit_can = []
    all_na_labels = []
    all_cc_labels = []

    for embeddings, na_labels, cc_labels in tqdm(loader, desc='Eval', leave=False):
        embeddings = embeddings.to(device)
        output = model(embeddings)

        all_logit_abn.append(output.logit_abn.cpu().numpy())
        all_logit_can.append(output.logit_can.cpu().numpy())
        all_na_labels.append(na_labels.numpy())
        all_cc_labels.append(cc_labels.numpy())

    logit_abn = np.concatenate(all_logit_abn)
    logit_can = np.concatenate(all_logit_can)
    na_labels = np.concatenate(all_na_labels)
    cc_labels = np.concatenate(all_cc_labels)

    prob_abn = 1 / (1 + np.exp(-logit_abn))
    prob_can = 1 / (1 + np.exp(-logit_can))
    pred_abn = (prob_abn >= 0.5).astype(int)
    pred_can = (prob_can >= 0.5).astype(int)

    metrics_abn = compute_metrics(na_labels, pred_abn, prob_abn)
    metrics_can = compute_metrics(cc_labels, pred_can, prob_can)

    violation_rate = (prob_can > prob_abn).mean()

    return {
        'abnormal': metrics_abn,
        'cancer': metrics_can,
        'violation_rate': float(violation_rate),
        'mean_bacc': (metrics_abn['bacc'] + metrics_can['bacc']) / 2
    }


def main():
    parser = argparse.ArgumentParser(description='SlideCheck Phase 1 Training')

    # Variant selection
    parser.add_argument('--variant', type=str, required=True,
                        choices=['baseline', 'mixup', 'feat_perturb', 'mixup_fp',
                                'concatv2', 'concatv2_mixup', 'concatv2_mixup_fp'],
                        help='Training variant')

    # Data arguments
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory containing feature .pt files')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')

    # Model arguments
    parser.add_argument('--foundation_model', type=str, default='virchow2',
                        help='Foundation model name (default: virchow2)')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension (default: 768)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size (default: 2048)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--lambda_constraint', type=float, default=1.0,
                        help='Constraint loss weight (default: 1.0)')

    # Augmentation arguments
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup alpha (default: 0.4)')
    parser.add_argument('--perturb_sigma', type=float, default=0.1,
                        help='Feature perturbation sigma (default: 0.1)')

    # Output arguments
    parser.add_argument('--log_dir', type=str, default='./logs/phase1',
                        help='Log directory (default: ./logs/phase1)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (default: 2024)')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine architecture from variant
    if 'concatv2' in args.variant:
        arch = 'concatv2'
        if args.hidden_dim == 768:  # Auto-adjust for concatv2
            args.hidden_dim = 1024
    else:
        arch = 'baseline'

    # Create experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{timestamp}_{args.variant}"

    # Create log directory
    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SlideCheck Phase 1 Training")
    print(f"{'='*60}")
    print(f"Variant: {args.variant}")
    print(f"Architecture: {arch}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Save config
    config = vars(args).copy()
    config['arch'] = arch
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    embeddings, na_labels, cc_labels = load_merged_data(args.feat_dir)

    # Create dataset
    dataset = SlideCheckDataset(embeddings, na_labels, cc_labels)

    # Split train/test
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Build model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_slidecheck_model(
        arch=arch,
        foundation_model=args.foundation_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Compute class weights from training set only
    train_indices = train_dataset.indices
    train_na_labels = na_labels[train_indices]
    train_cc_labels = cc_labels[train_indices]
    na_pos = max((train_na_labels == 1).sum(), 1)
    cc_pos = max((train_cc_labels == 1).sum(), 1)
    na_pos_weight = torch.tensor([(train_na_labels == 0).sum() / na_pos]).to(device)
    cc_pos_weight = torch.tensor([(train_cc_labels == 0).sum() / cc_pos]).to(device)

    # Training loop
    best_mean_bacc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args,
            pos_weight_abn=na_pos_weight,
            pos_weight_can=cc_pos_weight
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Abnormal - BACC: {test_metrics['abnormal']['bacc']:.4f}, "
              f"AUC: {test_metrics['abnormal']['auc']:.4f}")
        print(f"Test Cancer - BACC: {test_metrics['cancer']['bacc']:.4f}, "
              f"AUC: {test_metrics['cancer']['auc']:.4f}")
        print(f"Mean BACC: {test_metrics['mean_bacc']:.4f}")
        print(f"Violation Rate: {test_metrics['violation_rate']:.4f}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_metrics': test_metrics,
            'lr': scheduler.get_last_lr()[0]
        })

        # Save best model
        if test_metrics['mean_bacc'] > best_mean_bacc:
            best_mean_bacc = test_metrics['mean_bacc']
            patience_counter = 0

            save_checkpoint(
                model=model,
                save_path=str(log_dir / 'best_model.pt'),
                foundation_model=args.foundation_model,
                epoch=epoch,
                metrics=test_metrics,
                config={'arch': arch, 'hidden_dim': args.hidden_dim, 'dropout': args.dropout}
            )

            with open(log_dir / 'best_metrics.json', 'w') as f:
                json.dump(test_metrics, f, indent=2)

            print(f"★ New best model saved! Mean BACC: {best_mean_bacc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final model
    save_checkpoint(
        model=model,
        save_path=str(log_dir / 'final_model.pt'),
        foundation_model=args.foundation_model,
        epoch=epoch,
        metrics=test_metrics,
        config={'arch': arch, 'hidden_dim': args.hidden_dim, 'dropout': args.dropout}
    )

    # Save history
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Mean BACC: {best_mean_bacc:.4f}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
