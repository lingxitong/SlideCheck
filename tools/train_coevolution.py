#!/usr/bin/env python3
"""
Co-Evolution Training - Unified script for V4a and V6 routes

This script implements co-evolution training where:
1. MIL model mines high-attention patches
2. SlideCheck model generates pseudo-labels
3. Both models are trained together with mined data

Supports two routes:
- V4a: Negative bag enhancement + Type2 weight=0
- V6: V3c confidence weighting + stratified sampling

Usage:
    # V4a route (finetune)
    python tools/train_coevolution.py \
        --route v4a \
        --mode ft \
        --slidecheck_ckpt logs/phase1/best_model.pt \
        --pseudo_labels pseudo_labels.pt \
        --epochs 5 \
        --lr 1e-5

    # V6 route (from scratch)
    python tools/train_coevolution.py \
        --route v6 \
        --mode scratch \
        --pseudo_labels pseudo_labels.pt \
        --epochs 10 \
        --lr 1e-3
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.models import build_slidecheck_model
from slidecheck.utils import save_checkpoint, compute_dual_head_metrics


class CoEvolutionDataset(torch.utils.data.Dataset):
    """
    Dataset for co-evolution training with cancer_weight and abn_mask

    Args:
        embeddings: Patch embeddings [N, feature_dim]
        cancer_labels: Binary cancer labels [N]
        abnormal_labels: Binary abnormal labels [N]
        cancer_weights: Per-sample cancer weights [N]
        abn_mask: Abnormal loss mask [N] (1=use, 0=ignore)
    """
    def __init__(self, embeddings, cancer_labels, abnormal_labels,
                 cancer_weights, abn_mask):
        self.embeddings = embeddings
        self.cancer_labels = cancer_labels
        self.abnormal_labels = abnormal_labels
        self.cancer_weights = cancer_weights
        self.abn_mask = abn_mask

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.cancer_labels[idx],
                self.abnormal_labels[idx], self.cancer_weights[idx],
                self.abn_mask[idx])


def create_stratified_sampler(labels_abn, labels_can, balance_mode='sqrt'):
    """
    Create stratified sampler for balanced training

    Args:
        labels_abn: Abnormal labels
        labels_can: Cancer labels
        balance_mode: 'equal' or 'sqrt'

    Returns:
        WeightedRandomSampler
    """
    # Create combined labels for stratification
    combined = labels_abn * 2 + labels_can  # 0: N-NC, 1: N-C, 2: A-NC, 3: A-C

    # Count samples per stratum
    unique, counts = np.unique(combined, return_counts=True)
    stratum_counts = dict(zip(unique, counts))

    # Compute weights
    if balance_mode == 'equal':
        # Equal weight for each stratum
        weights = np.zeros(len(combined))
        for stratum in unique:
            mask = combined == stratum
            weights[mask] = 1.0 / stratum_counts[stratum]
    elif balance_mode == 'sqrt':
        # Square root of inverse frequency
        weights = np.zeros(len(combined))
        for stratum in unique:
            mask = combined == stratum
            weights[mask] = 1.0 / np.sqrt(stratum_counts[stratum])
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    # Normalize weights
    weights = weights / weights.sum() * len(weights)

    return WeightedRandomSampler(weights, len(weights), replacement=True)


def train_epoch_v4a(
    model, loader, optimizer, device,
    lambda_constraint=1.0,
    type2_cancer_weight=0.0,
    use_v3c_weighting=True,
    mixup_alpha=0.4,
    perturb_sigma=0.1,
    use_mixup=True,
    use_perturb=True,
    pos_weight_can=None,
    pos_weight_abn=None
):
    """
    Training epoch for V4a route with Mixup + Feature Perturbation

    V4a features:
    - Type 2 cancer weight = 0 (ignore Type 2 cancer samples)
    - V3c confidence weighting
    - Negative bag enhancement
    - Mixup (alpha=0.4) + Feature Perturbation (sigma=0.1)
    - cancer_weight mixing during Mixup
    """
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc='Train', leave=False):
        embeddings, cc_labels, na_labels, cancer_weight, abn_mask = batch
        embeddings = embeddings.to(device)
        na_labels = na_labels.to(device).float()
        cc_labels = cc_labels.to(device).float()
        cancer_weight = cancer_weight.to(device).float()
        abn_mask = abn_mask.to(device).float()

        batch_size = embeddings.size(0)

        # Feature Perturbation
        if use_perturb:
            noise = torch.randn_like(embeddings) * perturb_sigma
            embeddings = embeddings * (1 + noise)

        # Mixup
        if use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)
            idx = torch.randperm(batch_size, device=device)

            embeddings = lam * embeddings + (1 - lam) * embeddings[idx]
            cc_labels = lam * cc_labels + (1 - lam) * cc_labels[idx]
            na_labels = lam * na_labels + (1 - lam) * na_labels[idx]
            cancer_weight = lam * cancer_weight + (1 - lam) * cancer_weight[idx]
            abn_mask = lam * abn_mask + (1 - lam) * abn_mask[idx]

        optimizer.zero_grad()
        output = model(embeddings)
        logit_abn = output.logit_abn
        logit_can = output.logit_can

        # Cancer loss: weighted BCE with mixed cancer_weight
        if pos_weight_can is not None:
            loss_can_raw = F.binary_cross_entropy_with_logits(
                logit_can, cc_labels, pos_weight=pos_weight_can, reduction='none')
        else:
            loss_can_raw = F.binary_cross_entropy_with_logits(
                logit_can, cc_labels, reduction='none')
        loss_can = (loss_can_raw * cancer_weight).mean()

        # Abnormal loss: use mixed abn_mask as soft weight
        if abn_mask.sum() > 0:
            if pos_weight_abn is not None:
                loss_abn_raw = F.binary_cross_entropy_with_logits(
                    logit_abn, na_labels, pos_weight=pos_weight_abn, reduction='none')
            else:
                loss_abn_raw = F.binary_cross_entropy_with_logits(
                    logit_abn, na_labels, reduction='none')
            loss_abn = (loss_abn_raw * abn_mask).mean()
        else:
            loss_abn = torch.tensor(0.0, device=device)

        # Constraint loss
        p_abn = torch.sigmoid(logit_abn)
        p_can = torch.sigmoid(logit_can)
        constraint = torch.relu(p_can - p_abn).mean()

        loss = loss_can + loss_abn + lambda_constraint * constraint

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_epoch_v6(
    model, loader, optimizer, device,
    lambda_constraint=1.0,
    use_confidence_weighting=True
):
    """
    Training epoch for V6 route

    V6 features:
    - V3c confidence weighting
    - Stratified sampling (handled by dataloader)
    """
    model.train()
    total_loss = 0

    criterion_abn = nn.BCEWithLogitsLoss(reduction='none')
    criterion_can = nn.BCEWithLogitsLoss(reduction='none')

    for batch in tqdm(loader, desc='Train', leave=False):
        # CoEvolutionDataset returns 5-tuple: (emb, cc, na, cw, am)
        embeddings, cc_labels, na_labels, _, _ = batch
        embeddings = embeddings.to(device)
        na_labels = na_labels.to(device).float()
        cc_labels = cc_labels.to(device).float()

        optimizer.zero_grad()
        output = model(embeddings)
        logit_abn = output.logit_abn
        logit_can = output.logit_can

        # Compute losses
        loss_abn = criterion_abn(logit_abn, na_labels)
        loss_can = criterion_can(logit_can, cc_labels)

        # V3c confidence weighting
        if use_confidence_weighting:
            with torch.no_grad():
                prob_abn = torch.sigmoid(logit_abn)
                prob_can = torch.sigmoid(logit_can)
                conf_abn = torch.where(na_labels == 1, prob_abn, 1 - prob_abn)
                conf_can = torch.where(cc_labels == 1, prob_can, 1 - prob_can)
                weights = (conf_abn + conf_can) / 2

            loss_abn = (loss_abn * weights).mean()
            loss_can = (loss_can * weights).mean()
        else:
            loss_abn = loss_abn.mean()
            loss_can = loss_can.mean()

        # Constraint loss
        p_abn = torch.sigmoid(logit_abn)
        p_can = torch.sigmoid(logit_can)
        constraint = F.relu(p_can - p_abn)
        loss_constraint = (constraint ** 2).mean()

        loss = loss_abn + loss_can + lambda_constraint * loss_constraint
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model"""
    model.eval()

    all_logit_abn = []
    all_logit_can = []
    all_na_labels = []
    all_cc_labels = []

    for batch in tqdm(loader, desc='Eval', leave=False):
        # Handle both old format (emb, na, cc) and new format (emb, cc, na, cw, am)
        if len(batch) == 3:
            embeddings, na_labels, cc_labels = batch
        else:
            embeddings, cc_labels, na_labels, _, _ = batch

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

    metrics = compute_dual_head_metrics(
        na_labels, pred_abn, prob_abn,
        cc_labels, pred_can, prob_can
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Co-Evolution Training')

    # Route selection
    parser.add_argument('--route', choices=['v4a', 'v6'], required=True,
                        help='Training route')

    # Mode selection
    parser.add_argument('--mode', choices=['ft', 'scratch'], required=True,
                        help='Training mode: finetune or from scratch')

    # Data arguments
    parser.add_argument('--v2_cache', required=True,
                        help='Path to V2 training cache .pt file (pl_emb, pl_can, pl_abn, da_emb, da_can, da_abn)')
    parser.add_argument('--neg_bag_cache', default=None,
                        help='Path to negative bag cache .pt file (neg_emb) - V4a only')
    parser.add_argument('--mining_signals', default=None,
                        help='Path to mining signals .pt file (pl_cancer_prob) - V4a only')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (for scratch mode, split from Dataset A)')

    # Model arguments
    parser.add_argument('--slidecheck_ckpt', default=None,
                        help='Path to SlideCheck checkpoint (for finetune mode)')
    parser.add_argument('--foundation_model', default='virchow2',
                        help='Foundation model name')
    parser.add_argument('--arch', default='baseline',
                        choices=['baseline', 'concatv2'],
                        help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: ft=5, scratch=200)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: ft=1e-5, scratch=1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lambda_constraint', type=float, default=1.0,
                        help='Constraint loss weight')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Early stopping patience (scratch mode only, default: 30)')

    # V4a specific arguments
    parser.add_argument('--type2_cancer_weight', type=float, default=0.0,
                        help='V4a: Weight for Type 2 cancer samples (default: 0.0)')
    parser.add_argument('--use_v3c_weighting', action='store_true',
                        help='V4a: Use V3c confidence weighting')

    # Data augmentation arguments
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup alpha parameter (default: 0.4)')
    parser.add_argument('--perturb_sigma', type=float, default=0.1,
                        help='Feature perturbation sigma (default: 0.1)')
    parser.add_argument('--no_mixup', action='store_true',
                        help='Disable Mixup augmentation')
    parser.add_argument('--no_perturb', action='store_true',
                        help='Disable Feature Perturbation')

    # V6 specific arguments
    parser.add_argument('--balance_mode', choices=['equal', 'sqrt'], default='sqrt',
                        help='V6: Stratified sampling balance mode')
    parser.add_argument('--no_stratified', action='store_true',
                        help='V6: Disable stratified sampling')

    # Output arguments
    parser.add_argument('--log_dir', default='./logs/coevolution',
                        help='Log directory')
    parser.add_argument('--exp_name', default=None,
                        help='Experiment name')

    # Other arguments
    parser.add_argument('--device', default='cuda:0',
                        help='Device')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'ft' and args.slidecheck_ckpt is None:
        parser.error("--slidecheck_ckpt is required for finetune mode")

    if args.route == 'v4a':
        if args.neg_bag_cache is None:
            parser.error("V4a route requires --neg_bag_cache")
        if args.mining_signals is None:
            parser.error("V4a route requires --mining_signals")

    # Default epochs/lr based on mode
    if args.epochs is None:
        args.epochs = 5 if args.mode == 'ft' else 200
    if args.lr is None:
        args.lr = 1e-5 if args.mode == 'ft' else 1e-3

    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create experiment name
    if args.exp_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{timestamp}_{args.route}_{args.mode}"

    # Create log directory
    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Co-Evolution Training - {args.route.upper()} Route")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Route: {args.route}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Save config
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ===== Load V2 Cache =====
    print("Loading V2 cache...")
    cache = torch.load(args.v2_cache, map_location='cpu', weights_only=False)
    pl_emb = cache['pl_emb']
    pl_can = cache['pl_can']
    pl_abn = cache['pl_abn']
    da_emb = cache['da_emb']
    da_can = cache['da_can']
    da_abn = cache['da_abn']
    del cache

    n_pl = len(pl_can)
    n_da = len(da_can)
    in_dim = pl_emb.shape[1]

    print(f"  Pseudo-label patches: {n_pl:,}")
    print(f"  Dataset A patches: {n_da:,}")
    print(f"  Feature dimension: {in_dim}")

    # ===== V4a: Load additional data and compute cancer_weight =====
    if args.route == 'v4a':
        print("\nLoading negative bag cache...")
        neg_cache = torch.load(args.neg_bag_cache, map_location='cpu', weights_only=False)
        neg_emb = neg_cache['neg_emb'].float() if 'neg_emb' in neg_cache else neg_cache['embeddings'].float()
        del neg_cache
        n_neg = len(neg_emb)
        neg_can = torch.zeros(n_neg)
        neg_abn = torch.zeros(n_neg)
        print(f"  Negative bag patches: {n_neg:,}")

        print("\nLoading mining signals...")
        signals = torch.load(args.mining_signals, map_location='cpu', weights_only=False)
        pl_cancer_prob = signals['pl_cancer_prob']
        assert len(pl_cancer_prob) == n_pl, f"Mining signals mismatch: {len(pl_cancer_prob)} != {n_pl}"

        # Compute V4a cancer weights
        print("\nComputing V4a cancer weights...")
        type1 = (pl_can == 1) & (pl_abn == 1)
        type2 = (pl_can == 0) & (pl_abn == 1)  # ← FIXED: Non-Cancer + Abnormal
        type3 = (pl_can == 0) & (pl_abn == 0)

        pl_cancer_weight = torch.where(
            pl_can == 1, pl_cancer_prob, 1.0 - pl_cancer_prob)
        pl_cancer_weight[type2] = args.type2_cancer_weight  # Default: 0.0

        neg_cancer_weight = torch.ones(n_neg)
        da_cancer_weight = torch.ones(n_da)

        print(f"  Type 1 (Cancer+Abnormal):     {type1.sum():>8,}  weight mean={pl_cancer_weight[type1].mean():.4f}")
        print(f"  Type 2 (NonCancer+Abnormal):   {type2.sum():>8,}  weight = {args.type2_cancer_weight}")
        print(f"  Type 3 (Normal):               {type3.sum():>8,}  weight mean={pl_cancer_weight[type3].mean():.4f}")
        print(f"  Negative bags:                 {n_neg:>8,}  weight = 1.0")
        print(f"  Dataset A:                     {n_da:>8,}  weight = 1.0")

        # V4a: All samples use abnormal loss
        pl_abn_mask = torch.ones(n_pl)
        neg_abn_mask = torch.ones(n_neg)
        da_abn_mask = torch.ones(n_da)

    else:  # V6 route
        # V6: No negative bags, no cancer_weight
        neg_emb = None
        n_neg = 0
        pl_cancer_weight = torch.ones(n_pl)
        da_cancer_weight = torch.ones(n_da)
        pl_abn_mask = torch.ones(n_pl)
        da_abn_mask = torch.ones(n_da)

    # ===== Dataset A validation split (scratch mode) =====
    if args.mode == 'scratch':
        n_val = int(n_da * args.val_ratio)
        n_train_da = n_da - n_val
        perm = torch.randperm(n_da, generator=torch.Generator().manual_seed(args.seed))
        da_train_idx, da_val_idx = perm[:n_train_da], perm[n_train_da:]

        val_emb = da_emb[da_val_idx]
        val_can = da_can[da_val_idx]
        val_abn = da_abn[da_val_idx]
        print(f"\nScratch mode: Dataset A split {n_train_da:,} train / {n_val:,} val")

        # Replace DA with train portion only
        da_emb = da_emb[da_train_idx]
        da_can = da_can[da_train_idx]
        da_abn = da_abn[da_train_idx]
        da_cancer_weight = da_cancer_weight[da_train_idx]
        da_abn_mask = da_abn_mask[da_train_idx]
        n_da = n_train_da

    # ===== Combine all training data =====
    print("\nCombining training data...")
    if args.route == 'v4a':
        all_emb = torch.cat([pl_emb, neg_emb, da_emb], dim=0)
        all_can = torch.cat([pl_can, neg_can, da_can], dim=0)
        all_abn = torch.cat([pl_abn, neg_abn, da_abn], dim=0)
        all_cw = torch.cat([pl_cancer_weight, neg_cancer_weight, da_cancer_weight], dim=0)
        all_abn_mask = torch.cat([pl_abn_mask, neg_abn_mask, da_abn_mask], dim=0)
    else:  # V6
        all_emb = torch.cat([pl_emb, da_emb], dim=0)
        all_can = torch.cat([pl_can, da_can], dim=0)
        all_abn = torch.cat([pl_abn, da_abn], dim=0)
        all_cw = torch.cat([pl_cancer_weight, da_cancer_weight], dim=0)
        all_abn_mask = torch.cat([pl_abn_mask, da_abn_mask], dim=0)

    print(f"  Total training patches: {len(all_emb):,}")

    # Free memory
    del pl_emb, da_emb
    if args.route == 'v4a':
        del neg_emb
    import gc
    gc.collect()

    # ===== Create dataset and dataloader =====
    train_ds = CoEvolutionDataset(all_emb, all_can, all_abn, all_cw, all_abn_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)

    # Compute pos_weight
    pw_can = (len(all_can) - all_can.sum()) / (all_can.sum() + 1e-8)
    pw_abn = (len(all_abn) - all_abn.sum()) / (all_abn.sum() + 1e-8)
    print(f"  pos_weight: cancer={pw_can:.4f}  abnormal={pw_abn:.4f}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pw_can_t = torch.tensor([pw_can], device=device)
    pw_abn_t = torch.tensor([pw_abn], device=device)

    # Validation loader (scratch only)
    if args.mode == 'scratch':
        val_ds = CoEvolutionDataset(val_emb, val_can, val_abn,
                                     torch.ones(len(val_can)), torch.ones(len(val_can)))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.mode == 'ft':
        # Finetune mode: load pretrained model
        print(f"Loading pretrained model from {args.slidecheck_ckpt}")
        from slidecheck.utils import load_checkpoint

        ckpt = load_checkpoint(args.slidecheck_ckpt, foundation_model=args.foundation_model)
        config = ckpt['config']

        model = build_slidecheck_model(
            arch=config.get('arch', args.arch),
            foundation_model=args.foundation_model,
            hidden_dim=config.get('hidden_dim', args.hidden_dim),
            dropout=config.get('dropout', args.dropout)
        )
        model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded model from epoch {ckpt['epoch']}\n")
    else:
        # From scratch mode
        print("Training from scratch")
        model = build_slidecheck_model(
            arch=args.arch,
            foundation_model=args.foundation_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.mode == 'scratch' else None

    # Training loop
    best_mean_bacc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        if args.route == 'v4a':
            train_loss = train_epoch_v4a(
                model, train_loader, optimizer, device,
                lambda_constraint=args.lambda_constraint,
                type2_cancer_weight=args.type2_cancer_weight,
                use_v3c_weighting=args.use_v3c_weighting,
                mixup_alpha=args.mixup_alpha,
                perturb_sigma=args.perturb_sigma,
                use_mixup=not args.no_mixup,
                use_perturb=not args.no_perturb,
                pos_weight_can=pw_can_t,
                pos_weight_abn=pw_abn_t
            )
        else:  # v6
            train_loss = train_epoch_v6(
                model, train_loader, optimizer, device,
                lambda_constraint=args.lambda_constraint,
                use_confidence_weighting=True
            )

        # Evaluate
        if args.mode == 'scratch':
            val_metrics = evaluate(model, val_loader, device)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Abnormal - BACC: {val_metrics['abnormal']['bacc']:.4f}, "
                  f"AUC: {val_metrics['abnormal']['auc']:.4f}")
            print(f"Val Cancer - BACC: {val_metrics['cancer']['bacc']:.4f}, "
                  f"AUC: {val_metrics['cancer']['auc']:.4f}")
            print(f"Mean BACC: {val_metrics['mean_bacc']:.4f}")
            print(f"Violation Rate: {val_metrics['violation_rate']:.4f}")

            # Save history
            lr_now = optimizer.param_groups[0]['lr']
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'lr': lr_now
            })

            # Save best model + early stopping
            if val_metrics['mean_bacc'] > best_mean_bacc:
                best_mean_bacc = val_metrics['mean_bacc']
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    model=model,
                    save_path=str(log_dir / 'best_model.pt'),
                    foundation_model=args.foundation_model,
                    epoch=epoch,
                    metrics=val_metrics,
                    config={'arch': args.arch, 'hidden_dim': args.hidden_dim, 'dropout': args.dropout}
                )
                print(f"  -> New best model (Mean BACC: {best_mean_bacc:.4f})")
            else:
                patience_counter += 1
                if patience_counter % 10 == 0:
                    print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            # Finetune mode: no validation, just log training loss
            print(f"Train Loss: {train_loss:.4f}")

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

    # Save final model
    if args.mode == 'ft':
        # Finetune mode: save final model
        save_checkpoint(
            model=model,
            save_path=str(log_dir / 'final_model.pt'),
            foundation_model=args.foundation_model,
            epoch=args.epochs,
            metrics={},
            config={'arch': args.arch, 'hidden_dim': args.hidden_dim, 'dropout': args.dropout}
        )
    else:
        # Scratch mode: final model already saved as best_model.pt
        pass

    # Save history
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    if args.mode == 'scratch':
        print(f"Best Mean BACC: {best_mean_bacc:.4f}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
