#!/usr/bin/env python3
"""
Phase 2: Gated Attention MIL Training

Step 2.1: Pure MIL training (bag-level CE)
Step 2.2 (optional): SlideCheck-guided training (KL attention prior)

Usage:
    # Step 2.1 only
    python tools/train_mil.py \
        --train_manifest bag_manifest.csv \
        --epochs 30

    # Step 2.1 + Step 2.2
    python tools/train_mil.py \
        --train_manifest bag_manifest.csv \
        --epochs 30 \
        --slidecheck_ckpt logs/phase1/best_model.pt \
        --epochs_guided 10
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.models import build_gated_abmil_2560, build_slidecheck_model
from slidecheck.models.mil import GatedABMIL_2560
from slidecheck.datasets.h5_bag_dataset import H5BagDataset, bag_collate_fn
from slidecheck.utils.checkpoint import load_checkpoint


def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: torch.Tensor, y_prob: torch.Tensor) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).long()
    y_true = y_true.long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    bacc = (sens + spec) / 2
    precision_val = tp / (tp + fp + 1e-8)
    f1 = 2 * precision_val * sens / (precision_val + sens + 1e-8)

    return {'acc': acc, 'bacc': bacc, 'sens': sens, 'spec': spec, 'f1': f1}


# ==================== Step 2.1: Pure MIL Training ====================

def train_epoch_mil(
    mil: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    """Train one epoch with per-bag gradient updates and CE loss"""
    mil.train()

    total_loss = 0.0
    n_bags = 0
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc='MIL Training', leave=False):
        for bag in batch:
            features = bag['features'].to(device)
            label = bag['label'].to(device).unsqueeze(0)

            optimizer.zero_grad()

            mil_output = mil(features)
            logits = mil_output.logits  # [1, 2]
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_bags += 1

            prob = F.softmax(logits, dim=1)[0, 1].detach().cpu()
            all_labels.append(label.cpu().item())
            all_probs.append(prob.item())

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return {'loss': total_loss / max(n_bags, 1), **metrics}


# ==================== Step 2.2: SlideCheck-Guided MIL ====================

def train_epoch_mil_guided(
    mil: nn.Module,
    slidecheck: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lambda_guide: float = 0.5,
    mask_threshold: float = 0.5,
) -> Dict[str, float]:
    """Train with SlideCheck attention prior via KL divergence"""
    mil.train()
    slidecheck.eval()

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_guide = 0.0
    n_bags = 0
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc='Guided MIL Training', leave=False):
        for bag in batch:
            features = bag['features'].to(device)
            label = bag['label'].to(device).unsqueeze(0)

            optimizer.zero_grad()

            # MIL forward
            mil_output = mil(features)
            logits = mil_output.logits  # [1, 2]
            attn = mil_output.attention_normalized.squeeze(-1)  # [N]

            loss_ce = F.cross_entropy(logits, label)

            # SlideCheck guided loss
            loss_guide = torch.tensor(0.0, device=device)
            with torch.no_grad():
                sc_output = slidecheck(features)
                cancer_prob = torch.sigmoid(sc_output.logit_can)  # [N]

            mask = (cancer_prob > mask_threshold).float()
            if mask.sum() > 0:
                mask_norm = mask / (mask.sum() + 1e-8)
                loss_guide = F.kl_div(
                    torch.log(attn + 1e-8),
                    mask_norm,
                    reduction='sum',
                )

            loss = loss_ce + lambda_guide * loss_guide
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_ce += loss_ce.item()
            total_loss_guide += loss_guide.item()
            n_bags += 1

            prob = F.softmax(logits, dim=1)[0, 1].detach().cpu()
            all_labels.append(label.cpu().item())
            all_probs.append(prob.item())

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return {
        'loss': total_loss / max(n_bags, 1),
        'loss_ce': total_loss_ce / max(n_bags, 1),
        'loss_guide': total_loss_guide / max(n_bags, 1),
        **metrics,
    }


# ==================== Evaluation ====================

@torch.no_grad()
def evaluate(
    mil: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    mil.eval()

    all_labels = []
    all_probs = []

    for batch in dataloader:
        for bag in batch:
            features = bag['features'].to(device)
            label = bag['label'].item()

            mil_output = mil(features)
            prob = F.softmax(mil_output.logits, dim=1)[0, 1].cpu().item()

            all_labels.append(label)
            all_probs.append(prob)

    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    return compute_metrics(all_labels, all_probs)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Gated Attention MIL Training')

    # Data
    parser.add_argument('--train_manifest', type=str, required=True,
                        help='Path to training manifest CSV')
    parser.add_argument('--val_manifest', type=str, default=None,
                        help='Path to validation manifest CSV')

    # MIL architecture
    parser.add_argument('--attn_hidden', type=int, default=384,
                        help='Attention hidden dimension (default: 384)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate (default: 0.25)')

    # Step 2.1: Pure MIL
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for Step 2.1 (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')

    # Step 2.2: SlideCheck-guided (optional)
    parser.add_argument('--slidecheck_ckpt', type=str, default=None,
                        help='Path to SlideCheck checkpoint for guided training')
    parser.add_argument('--epochs_guided', type=int, default=0,
                        help='Number of epochs for Step 2.2 (default: 0, disabled)')
    parser.add_argument('--lambda_guide', type=float, default=0.5,
                        help='Weight for KL guided loss (default: 0.5)')

    # General
    parser.add_argument('--batch_size_bag', type=int, default=1,
                        help='Bag batch size (default: 1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader num_workers (default: 2)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (default: 2024)')
    parser.add_argument('--log_dir', type=str, default='./logs/phase2',
                        help='Log directory (default: ./logs/phase2)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or "phase2_gated_mil"
    log_dir = Path(args.log_dir) / f"{timestamp}_{exp_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log dir: {log_dir}")

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ========== Load data ==========
    print("Loading bag data...")
    train_dataset = H5BagDataset(args.train_manifest, label_filter=[0, 1])
    val_dataset = H5BagDataset(args.val_manifest, label_filter=[0, 1]) if args.val_manifest else None

    print(f"Train: {train_dataset.get_statistics()}")
    if val_dataset:
        print(f"Val: {val_dataset.get_statistics()}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size_bag, shuffle=True,
        collate_fn=bag_collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size_bag, shuffle=False,
        collate_fn=bag_collate_fn, num_workers=args.num_workers,
    ) if val_dataset else None

    # ========== Build GatedABMIL_2560 ==========
    print(f"Building GatedABMIL_2560 (attn_hidden={args.attn_hidden})...")
    mil = build_gated_abmil_2560(
        in_dim=2560,
        attn_hidden=args.attn_hidden,
        num_classes=2,
        dropout=args.dropout,
    )
    mil = mil.to(device)
    print(f"MIL parameters: {sum(p.numel() for p in mil.parameters()):,}")

    # (Optional) Load SlideCheck for Step 2.2
    slidecheck = None
    if args.slidecheck_ckpt and args.epochs_guided > 0:
        print(f"Loading SlideCheck from {args.slidecheck_ckpt}...")
        ckpt = load_checkpoint(args.slidecheck_ckpt)
        sc_config = ckpt['config']
        slidecheck = build_slidecheck_model(
            arch=sc_config.get('arch', 'baseline'),
            foundation_model=sc_config.get('foundation_model', 'virchow2'),
            hidden_dim=sc_config.get('hidden_dim', 768),
            dropout=sc_config.get('dropout', 0.1),
        )
        slidecheck.load_state_dict(ckpt['state_dict'])
        slidecheck = slidecheck.to(device)
        slidecheck.eval()
        for param in slidecheck.parameters():
            param.requires_grad = False
        print("SlideCheck loaded (frozen)")

    # ========== Step 2.1: Pure MIL Training ==========
    print(f"\n{'='*60}")
    print("Step 2.1: Pure Gated-MIL Training (no SlideCheck)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        mil.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_metric = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Step 2.1] Epoch {epoch}/{args.epochs}")

        train_metrics = train_epoch_mil(mil, train_loader, optimizer, str(device))
        val_metrics = evaluate(mil, val_loader, str(device)) if val_loader else {}
        scheduler.step()

        record = {
            'phase': 'step2.1',
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train': train_metrics,
            'val': val_metrics,
        }
        history.append(record)

        print(f"  Train Loss: {train_metrics['loss']:.4f} | BACC: {train_metrics['bacc']:.4f}")
        if val_metrics:
            print(f"  Val BACC: {val_metrics['bacc']:.4f} | F1: {val_metrics['f1']:.4f}")

        metric = val_metrics.get('bacc', train_metrics['bacc'])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'mil_state_dict': mil.state_dict(),
                'best_metric': best_metric,
                'config': vars(args),
            }, log_dir / 'step1_best.pt')
            print(f"  -> New best! (BACC={best_metric:.4f})")

    print(f"\nStep 2.1 complete. Best BACC: {best_metric:.4f} @ epoch {best_epoch}")

    # ========== Step 2.2 (optional): SlideCheck-Guided ==========
    if slidecheck is not None and args.epochs_guided > 0:
        print(f"\n{'='*60}")
        print("Step 2.2: Guided MIL Training (SlideCheck attention prior)")
        print(f"{'='*60}")

        optimizer_guided = torch.optim.AdamW(
            mil.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs_guided + 1):
            global_epoch = args.epochs + epoch
            print(f"\n[Step 2.2] Epoch {epoch}/{args.epochs_guided} (Global: {global_epoch})")

            train_metrics = train_epoch_mil_guided(
                mil, slidecheck, train_loader, optimizer_guided,
                str(device), lambda_guide=args.lambda_guide,
            )
            val_metrics = evaluate(mil, val_loader, str(device)) if val_loader else {}

            record = {
                'phase': 'step2.2',
                'epoch': epoch,
                'global_epoch': global_epoch,
                'train': train_metrics,
                'val': val_metrics,
            }
            history.append(record)

            print(f"  Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['loss_ce']:.4f}, Guide: {train_metrics['loss_guide']:.4f})")
            if val_metrics:
                print(f"  Val BACC: {val_metrics['bacc']:.4f}")

            metric = val_metrics.get('bacc', train_metrics['bacc'])
            if metric > best_metric:
                best_metric = metric
                best_epoch = global_epoch
                torch.save({
                    'epoch': global_epoch,
                    'mil_state_dict': mil.state_dict(),
                    'best_metric': best_metric,
                    'config': vars(args),
                }, log_dir / 'step2_guided_best.pt')
                print(f"  -> New best! (BACC={best_metric:.4f})")

    # Save final model
    torch.save({
        'epoch': best_epoch,
        'mil_state_dict': mil.state_dict(),
        'best_metric': best_metric,
        'config': vars(args),
    }, log_dir / 'phase2_final.pt')

    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Phase 2 Complete! Best BACC: {best_metric:.4f} @ epoch {best_epoch}")
    print(f"Saved to: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
