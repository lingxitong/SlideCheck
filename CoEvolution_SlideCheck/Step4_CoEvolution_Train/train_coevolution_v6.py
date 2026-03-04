#!/usr/bin/env python3
"""
V6 — SlideCheck 训练 (V3c 置信度加权 + 分层采样)

两种模式:
  --mode scratch:  随机初始化, lr=1e-3, CosineAnnealingLR, Mixup+FP, early stop
  --mode finetune: Phase1 checkpoint, lr=1e-5, 5 epochs, 无增强

改进 vs V5a:
  1. V3c 置信度加权: cancer loss 按 mining cancer_prob 逐样本加权
  2. 分层采样: 每个 batch 从 11 个分层 (3 PL types + 8 DA domains) 按配额抽样

--balance_mode:
  equal: 各分层等额 (V6, 过于激进)
  sqrt:  按 sqrt(count) 分配 (V6b, 温和版)
"""

import argparse, json, logging, os, random, sys
from pathlib import Path

import numpy as np
# Compatibility shim: files saved with numpy 2.x use numpy._core
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import build_slidecheck_mlp
from models.slidecheck_mlp import SlideCheckOutput

try:
    from models import load_slidecheck_from_ckpt
except ImportError:
    from models.slidecheck_mlp import load_slidecheck_from_ckpt


# ===========================================================================
#  ConcatV2 architecture
# ===========================================================================
class SlideCheckMLP_InputConcatV2(nn.Module):
    def __init__(self, in_dim=2560, hidden_dim=1024, dropout=0.1):
        super().__init__()
        combined_dim = in_dim * 2
        self.input_norm = nn.LayerNorm(combined_dim, elementwise_affine=False)
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_zscore = (x - mean) / std
        x_combined = torch.cat([x, x_zscore], dim=-1)
        x_combined = self.input_norm(x_combined)
        feat = self.backbone(x_combined)
        logit_abn = self.head_abn(feat).squeeze(-1)
        logit_can = self.head_can(feat).squeeze(-1)
        return SlideCheckOutput(logit_abn=logit_abn, logit_can=logit_can)


# ===========================================================================
#  Dataset
# ===========================================================================
class PatchDatasetV6(Dataset):
    def __init__(self, embeddings, cancer_labels, abnormal_labels, cancer_weights):
        self.embeddings = embeddings
        self.cancer_labels = cancer_labels
        self.abnormal_labels = abnormal_labels
        self.cancer_weights = cancer_weights

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.cancer_labels[idx],
                self.abnormal_labels[idx], self.cancer_weights[idx])


# ===========================================================================
#  Stratified Batch Sampler
# ===========================================================================
class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampling with per-stratum quotas.
    Small strata cycle with reshuffling when exhausted.
    Epoch length = total training samples / batch_size.
    """
    def __init__(self, strata, quotas, n_batches):
        """
        strata:    dict {name: np.array of global indices}
        quotas:    dict {name: int samples per batch}
        n_batches: batches per epoch
        """
        self.strata = strata
        self.quotas = quotas
        self.n_batches = n_batches

    def __iter__(self):
        shuffled = {k: np.random.permutation(v) for k, v in self.strata.items()}
        pos = {k: 0 for k in shuffled}

        for _ in range(self.n_batches):
            batch = []
            for k in shuffled:
                n = self.quotas[k]
                for _ in range(n):
                    if pos[k] >= len(shuffled[k]):
                        shuffled[k] = np.random.permutation(self.strata[k])
                        pos[k] = 0
                    batch.append(int(shuffled[k][pos[k]]))
                    pos[k] += 1
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ===========================================================================
#  Augmentations
# ===========================================================================
def mixup(x, y_can, y_abn, w, alpha=0.4):
    """Mixup augmentation on embeddings, labels, and weights."""
    if alpha <= 0:
        return x, y_can, y_abn, w
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return (
        lam * x + (1 - lam) * x[idx],
        lam * y_can + (1 - lam) * y_can[idx],
        lam * y_abn + (1 - lam) * y_abn[idx],
        lam * w + (1 - lam) * w[idx],
    )


def feature_perturbation(x, sigma=0.1):
    """Additive Gaussian noise on embeddings."""
    return x + torch.randn_like(x) * sigma


# ===========================================================================
#  Metrics
# ===========================================================================
def balanced_accuracy(preds, targets):
    pbin = (preds > 0.5).float()
    pos = targets == 1
    neg = targets == 0
    tpr = pbin[pos].mean().item() if pos.sum() > 0 else 0.0
    tnr = (1 - pbin[neg]).mean().item() if neg.sum() > 0 else 0.0
    return (tpr + tnr) / 2.0


# ===========================================================================
#  Args
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train SlideCheck V6")
    p.add_argument("--mode", type=str, required=True, choices=["scratch", "finetune"])
    p.add_argument("--arch", type=str, default="baseline", choices=["baseline", "concatv2"])
    p.add_argument("--cache_path", type=str, required=True)
    p.add_argument("--signals_path", type=str, required=True,
                   help="pl_mining_signals_v5a.pt with cancer_prob + DA sizes")
    p.add_argument("--slidecheck_ckpt", type=str, default=None,
                   help="Phase1 checkpoint (required for finetune)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs (default: 200 for scratch, 5 for finetune)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override lr (default: 1e-3 for scratch, 1e-5 for finetune)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--pl_ratio", type=float, default=0.5,
                   help="Fraction of batch allocated to PL (default 0.5)")
    p.add_argument("--balance_mode", type=str, default="sqrt",
                   choices=["equal", "sqrt"],
                   help="Stratum allocation: equal (V6) or sqrt (V6b, default)")
    p.add_argument("--no_stratified", action="store_true",
                   help="Disable stratified sampling, use random shuffle (V6c)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--lambda_constraint", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--perturb_sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--exp_name", type=str, default=None)
    return p.parse_args()


# ===========================================================================
#  Main
# ===========================================================================
def main():
    args = parse_args()

    # Defaults per mode
    if args.epochs is None:
        args.epochs = 200 if args.mode == "scratch" else 5
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "scratch" else 1e-5
    if args.exp_name is None:
        args.exp_name = f"{'scratch' if args.mode == 'scratch' else 'finetune'}_v6_baseline"

    if args.mode == "finetune" and not args.slidecheck_ckpt:
        raise ValueError("--slidecheck_ckpt required for finetune mode")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(out_dir, "train.log")),
                  logging.StreamHandler()])
    log = logging.getLogger(__name__)
    log.info("Mode: %s", args.mode)
    log.info("Args: %s", vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load data ----
    log.info("Loading cache: %s", args.cache_path)
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    pl_emb, pl_can, pl_abn = cache["pl_emb"], cache["pl_can"], cache["pl_abn"]
    da_emb, da_can, da_abn = cache["da_emb"], cache["da_can"], cache["da_abn"]
    del cache
    n_pl = len(pl_can)
    n_da = len(da_can)
    log.info("Pseudo-label patches: %d", n_pl)
    log.info("Dataset A patches: %d", n_da)

    # ---- load mining signals + DA metadata ----
    log.info("Loading signals: %s", args.signals_path)
    signals = torch.load(args.signals_path, map_location="cpu", weights_only=False)
    pl_cancer_prob = signals["pl_cancer_prob"]
    da_sizes = signals["da_sizes"]
    da_names = signals["da_names"]
    assert len(pl_cancer_prob) == n_pl, \
        f"Signal length {len(pl_cancer_prob)} != cache PL {n_pl}"
    assert sum(da_sizes) == n_da, \
        f"DA sizes sum {sum(da_sizes)} != cache DA {n_da}"

    # ---- V3c confidence weighting ----
    pl_cancer_weight = torch.where(
        pl_can == 1,
        pl_cancer_prob,          # Type 1: w = cancer_prob
        1.0 - pl_cancer_prob,    # Type 2/3: w = 1 - cancer_prob
    )
    da_cancer_weight = torch.ones(n_da)

    # Log weight statistics by type
    pl_can_np = pl_can.numpy()
    pl_abn_np = pl_abn.numpy()
    type1_mask = (pl_can_np == 1) & (pl_abn_np == 1)
    type2_mask = (pl_can_np == 0) & (pl_abn_np == 1)
    type3_mask = (pl_can_np == 0) & (pl_abn_np == 0)

    log.info("=== V3c Confidence Weights ===")
    log.info("  Type 1 (%d): cancer_prob mean=%.4f -> weight mean=%.4f",
             type1_mask.sum(), pl_cancer_prob[type1_mask].mean(),
             pl_cancer_weight[type1_mask].mean())
    log.info("  Type 2 (%d): cancer_prob mean=%.4f -> weight mean=%.4f",
             type2_mask.sum(), pl_cancer_prob[type2_mask].mean(),
             pl_cancer_weight[type2_mask].mean())
    log.info("  Type 3 (%d): cancer_prob mean=%.4f -> weight mean=%.4f",
             type3_mask.sum(), pl_cancer_prob[type3_mask].mean(),
             pl_cancer_weight[type3_mask].mean())
    log.info("  Dataset A (%d): weight=1.0", n_da)

    # ---- combine all data ----
    all_emb = torch.cat([pl_emb, da_emb], dim=0)
    all_can = torch.cat([pl_can, da_can], dim=0)
    all_abn = torch.cat([pl_abn, da_abn], dim=0)
    all_weight = torch.cat([pl_cancer_weight, da_cancer_weight], dim=0)
    in_dim = all_emb.shape[-1]
    n_total = len(all_emb)
    log.info("Combined: %d patches, in_dim=%d", n_total, in_dim)

    del pl_emb, da_emb
    import gc; gc.collect()

    # ---- build strata indices ----
    # PL types (indices 0..n_pl-1)
    type1_idx = np.where(type1_mask)[0]
    type2_idx = np.where(type2_mask)[0]
    type3_idx = np.where(type3_mask)[0]

    # DA domains (indices n_pl..n_pl+n_da-1)
    da_domain_idx = {}
    offset = n_pl
    for name, size in zip(da_names, da_sizes):
        da_domain_idx[name] = np.arange(offset, offset + size)
        offset += size

    # ---- train/val split (scratch only) ----
    if args.mode == "scratch":
        rng = np.random.default_rng(args.seed)
        all_idx = np.arange(n_total)
        rng.shuffle(all_idx)
        n_val = int(n_total * args.val_ratio)
        val_indices = np.sort(all_idx[:n_val])
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[val_indices] = False
        log.info("Train/Val split: %d train, %d val", train_mask.sum(), n_val)
    else:
        train_mask = np.ones(n_total, dtype=bool)
        val_indices = np.array([], dtype=np.int64)

    # Build strata from training indices only
    pl_strata = {
        "PL_T1": type1_idx[train_mask[type1_idx]],
        "PL_T2": type2_idx[train_mask[type2_idx]],
        "PL_T3": type3_idx[train_mask[type3_idx]],
    }
    da_strata = {}
    for name, idx in da_domain_idx.items():
        da_strata[name] = idx[train_mask[idx]]

    log.info("=== Strata (training) ===")
    for k, v in pl_strata.items():
        log.info("  %s: %d samples", k, len(v))
    for k, v in da_strata.items():
        log.info("  DA_%s: %d samples", k, len(v))

    # ---- dataset and dataloaders ----
    full_ds = PatchDatasetV6(all_emb, all_can, all_abn, all_weight)

    if args.no_stratified:
        # V6c: random shuffle, no stratified sampling
        if args.mode == "scratch":
            train_indices = np.where(train_mask)[0].tolist()
            train_ds = Subset(full_ds, train_indices)
        else:
            train_ds = full_ds
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
        log.info("Sampling: random shuffle (no stratified)")
        log.info("Train: %d batches/epoch", len(train_loader))
    else:
        # V6/V6b: stratified batch sampling
        # Merge all strata into one dict
        all_strata = dict(pl_strata)  # PL_T1, PL_T2, PL_T3
        for name, idx in da_strata.items():
            all_strata[f"DA_{name}"] = idx

        pl_keys = [k for k in all_strata if k.startswith("PL_")]
        da_keys = [k for k in all_strata if k.startswith("DA_")]
        pl_batch = int(args.batch_size * args.pl_ratio)
        da_batch = args.batch_size - pl_batch

        def _allocate(keys, budget, mode):
            sizes = [len(all_strata[k]) for k in keys]
            if mode == "sqrt":
                weights = [np.sqrt(s) for s in sizes]
            else:
                weights = [1.0] * len(keys)
            total_w = sum(weights)
            raw = [budget * w / total_w for w in weights]
            quotas = [max(1, round(r)) for r in raw]
            diff = budget - sum(quotas)
            if diff != 0:
                max_i = sizes.index(max(sizes))
                quotas[max_i] += diff
            return {k: q for k, q in zip(keys, quotas)}

        quotas = {}
        quotas.update(_allocate(pl_keys, pl_batch, args.balance_mode))
        quotas.update(_allocate(da_keys, da_batch, args.balance_mode))

        log.info("=== Batch quotas (mode=%s) ===", args.balance_mode)
        for k in pl_keys + da_keys:
            natural_pct = 100.0 * len(all_strata[k]) / sum(len(all_strata[g]) for g in pl_keys + da_keys)
            batch_pct = 100.0 * quotas[k] / args.batch_size
            oversample = batch_pct / natural_pct if natural_pct > 0 else 0
            log.info("  %s: %d/batch (%.1f%%, natural %.1f%%, %.1fx)",
                     k, quotas[k], batch_pct, natural_pct, oversample)

        n_train = sum(len(all_strata[k]) for k in all_strata)
        n_batches = n_train // args.batch_size

        train_sampler = StratifiedBatchSampler(all_strata, quotas, n_batches)
        train_loader = DataLoader(full_ds, batch_sampler=train_sampler,
                                  num_workers=args.num_workers)
        log.info("Train: %d batches/epoch", len(train_sampler))

    if args.mode == "scratch" and len(val_indices) > 0:
        val_loader = DataLoader(
            Subset(full_ds, val_indices.tolist()),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
    else:
        val_loader = None

    # ---- pos_weight (from ALL data) ----
    pw_can = (n_total - all_can.sum()) / (all_can.sum() + 1e-8)
    pw_abn = (n_total - all_abn.sum()) / (all_abn.sum() + 1e-8)
    log.info("pos_weight: cancer=%.4f  abnormal=%.4f", pw_can.item(), pw_abn.item())
    pw_can_t = torch.tensor([pw_can], device=device)
    pw_abn_t = torch.tensor([pw_abn], device=device)

    # ---- build model ----
    if args.mode == "finetune":
        if args.arch == "baseline":
            model = load_slidecheck_from_ckpt(args.slidecheck_ckpt)
            log.info("Loaded baseline from %s", args.slidecheck_ckpt)
        else:
            model = SlideCheckMLP_InputConcatV2(in_dim=in_dim)
            ckpt = torch.load(args.slidecheck_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            log.info("Loaded concatv2 from %s", args.slidecheck_ckpt)
    else:
        if args.arch == "baseline":
            model = build_slidecheck_mlp(in_dim=in_dim)
        else:
            model = SlideCheckMLP_InputConcatV2(in_dim=in_dim)
        log.info("Random init (%s)", args.arch)

    model.to(device)
    log.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # ---- optimizer / scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) \
        if args.mode == "scratch" else None

    crit_abn = nn.BCEWithLogitsLoss(pos_weight=pw_abn_t)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---- training loop ----
    best_bacc = 0.0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_can, t_abn, t_con, nb = 0.0, 0.0, 0.0, 0.0, 0

        for emb, c, a, w in train_loader:
            emb, c, a, w = emb.to(device), c.to(device), a.to(device), w.to(device)

            # Augmentation (scratch only)
            if args.mode == "scratch":
                emb, c, a, w = mixup(emb, c, a, w, alpha=args.mixup_alpha)
                emb = feature_perturbation(emb, sigma=args.perturb_sigma)

            out = model(emb)

            # Cancer loss: V3c weighted
            loss_can_raw = F.binary_cross_entropy_with_logits(
                out.logit_can, c, pos_weight=pw_can_t, reduction='none')
            loss_can = (loss_can_raw * w).mean()

            # Abnormal loss: standard BCE
            loss_abn = crit_abn(out.logit_abn, a)

            # Constraint: p(cancer) <= p(abnormal)
            p_can = torch.sigmoid(out.logit_can)
            p_abn = torch.sigmoid(out.logit_abn)
            constraint = torch.relu(p_can - p_abn).mean()

            loss = loss_can + loss_abn + args.lambda_constraint * constraint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_can += loss_can.item()
            t_abn += loss_abn.item()
            t_con += constraint.item()
            nb += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = t_loss / max(nb, 1)

        # ---- validation (scratch only) ----
        if val_loader is not None:
            model.eval()
            pc_all, pa_all, lc_all, la_all = [], [], [], []
            v_loss, vnb = 0.0, 0
            with torch.no_grad():
                for emb, c, a, w in val_loader:
                    emb, c, a = emb.to(device), c.to(device), a.to(device)
                    out = model(emb)
                    l_can = F.binary_cross_entropy_with_logits(
                        out.logit_can, c, pos_weight=pw_can_t)
                    l_abn = crit_abn(out.logit_abn, a)
                    cst = torch.relu(
                        torch.sigmoid(out.logit_can) - torch.sigmoid(out.logit_abn)
                    ).mean()
                    v_loss += (l_can + l_abn + args.lambda_constraint * cst).item()
                    vnb += 1
                    pc_all.append(torch.sigmoid(out.logit_can).cpu())
                    pa_all.append(torch.sigmoid(out.logit_abn).cpu())
                    lc_all.append(c.cpu())
                    la_all.append(a.cpu())

            bacc_c = balanced_accuracy(torch.cat(pc_all), torch.cat(lc_all))
            bacc_a = balanced_accuracy(torch.cat(pa_all), torch.cat(la_all))
            mean_bacc = (bacc_c + bacc_a) / 2.0

            log.info(
                "Epoch %3d/%d  TrLoss %.4f  VaLoss %.4f  "
                "BACC_can %.4f  BACC_abn %.4f  MeanBACC %.4f",
                epoch, args.epochs, avg_loss, v_loss / max(vnb, 1),
                bacc_c, bacc_a, mean_bacc)

            if mean_bacc > best_bacc:
                best_bacc = mean_bacc
                patience_ctr = 0
                torch.save(
                    {"state_dict": model.state_dict(),
                     "model_state_dict": model.state_dict(),
                     "epoch": epoch, "mean_bacc": mean_bacc},
                    os.path.join(out_dir, "best_model.pt"))
                log.info("  => saved best_model.pt (mean_bacc=%.4f)", mean_bacc)
            else:
                patience_ctr += 1

            if patience_ctr >= args.early_stop_patience:
                log.info("Early stopping at epoch %d (patience=%d)",
                         epoch, args.early_stop_patience)
                break
        else:
            # Finetune: just log loss
            log.info("Epoch %d/%d  Loss %.4f (can=%.4f abn=%.4f con=%.4f)",
                     epoch, args.epochs, avg_loss,
                     t_can / max(nb, 1), t_abn / max(nb, 1), t_con / max(nb, 1))

    # ---- save final ----
    if args.mode == "finetune":
        torch.save(
            {"state_dict": model.state_dict(),
             "slidecheck_state_dict": model.state_dict(), "epoch": epoch},
            os.path.join(out_dir, "final.pt"))
        log.info("Finetune complete. Saved final.pt")
    else:
        torch.save(
            {"state_dict": model.state_dict(),
             "model_state_dict": model.state_dict(),
             "epoch": epoch, "mean_bacc": mean_bacc if val_loader else 0.0},
            os.path.join(out_dir, "final_model.pt"))
        log.info("Training complete. Best mean_bacc=%.4f", best_bacc)


if __name__ == "__main__":
    main()
