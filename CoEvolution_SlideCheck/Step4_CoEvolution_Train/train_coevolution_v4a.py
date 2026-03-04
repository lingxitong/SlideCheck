#!/usr/bin/env python3
"""
V4a + Mixup+FP 训练脚本

两种模式:
  --mode ft      : Finetune (5ep, lr=1e-5, 从 Phase1 checkpoint)
  --mode scratch : Scratch (200ep, lr=1e-3, 随机初始化, CosineAnnealingLR, early stop)

V4a 数据配置 + Mixup(alpha=0.4) + Feature Perturbation(sigma=0.1)
- 全样本 abnormal loss + constraint + pos_weight
- Type 2 cancer weight=0, V3c 动态权重
- cancer_weight 在 Mixup 时线性混合
"""

import argparse, json, logging, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


class PatchDataset(Dataset):
    def __init__(self, embeddings, cancer_labels, abnormal_labels, cancer_weights, abn_mask):
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


def parse_args():
    p = argparse.ArgumentParser(description="V4a + Mixup+FP Training")
    p.add_argument("--mode", type=str, required=True, choices=["ft", "scratch"])
    p.add_argument("--cache_path", type=str, required=True, help="V2 training cache")
    p.add_argument("--neg_bag_cache", type=str, required=True, help="neg_bag_cache.pt")
    p.add_argument("--signals_path", type=str, required=True, help="pl_mining_signals.pt")
    p.add_argument("--slidecheck_ckpt", type=str, default=None, help="Phase1 ckpt (for ft mode)")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (default: ft=5, scratch=200)")
    p.add_argument("--lr", type=float, default=None, help="Override lr (default: ft=1e-5, scratch=1e-3)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_constraint", type=float, default=1.0)
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--perturb_sigma", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1, help="DA validation split (scratch only)")
    p.add_argument("--early_stop_patience", type=int, default=30, help="Scratch only")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="v4a_mixup")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main():
    args = parse_args()

    # Defaults based on mode
    if args.epochs is None:
        args.epochs = 5 if args.mode == "ft" else 200
    if args.lr is None:
        args.lr = 1e-5 if args.mode == "ft" else 1e-3

    random.seed(args.seed); np.random.seed(args.seed)
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

    # ---- load V2 cache ----
    log.info("Loading V2 cache: %s", args.cache_path)
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    pl_emb, pl_can, pl_abn = cache["pl_emb"], cache["pl_can"], cache["pl_abn"]
    da_emb, da_can, da_abn = cache["da_emb"], cache["da_can"], cache["da_abn"]
    del cache
    n_pl = len(pl_can)
    n_da = len(da_can)
    log.info("Pseudo-label patches: %d", n_pl)
    log.info("Dataset A patches: %d", n_da)

    # ---- load negative bag cache ----
    log.info("Loading neg bag cache: %s", args.neg_bag_cache)
    neg_cache = torch.load(args.neg_bag_cache, map_location="cpu", weights_only=False)
    neg_emb = neg_cache["neg_emb"].float()
    del neg_cache
    n_neg = len(neg_emb)
    neg_can = torch.zeros(n_neg)
    neg_abn = torch.zeros(n_neg)
    log.info("Negative bag patches: %d", n_neg)

    # ---- load mining signals ----
    log.info("Loading mining signals: %s", args.signals_path)
    signals = torch.load(args.signals_path, map_location="cpu", weights_only=False)
    pl_cancer_prob = signals["pl_cancer_prob"]
    assert len(pl_cancer_prob) == n_pl

    # ---- V4a: V3c dynamic weights + Type 2 cancer weight = 0 ----
    type1 = (pl_can == 1) & (pl_abn == 1)
    type2 = (pl_can == 0) & (pl_abn == 1)
    type3 = (pl_can == 0) & (pl_abn == 0)

    pl_cancer_weight = torch.where(
        pl_can == 1, pl_cancer_prob, 1.0 - pl_cancer_prob)
    pl_cancer_weight[type2] = 0.0

    neg_cancer_weight = torch.ones(n_neg)
    da_cancer_weight = torch.ones(n_da)

    log.info("=== V4a Cancer Weights ===")
    log.info("  Type 1 (%d): weight mean=%.4f", type1.sum(), pl_cancer_weight[type1].mean())
    log.info("  Type 2 (%d): weight = 0", type2.sum())
    log.info("  Type 3 (%d): weight mean=%.4f", type3.sum(), pl_cancer_weight[type3].mean())
    log.info("  Neg bags (%d): weight = 1.0", n_neg)
    log.info("  Dataset A (%d): weight = 1.0", n_da)

    # ---- abnormal loss mask: V4a = all samples ----
    pl_abn_mask = torch.ones(n_pl)
    neg_abn_mask = torch.ones(n_neg)
    da_abn_mask = torch.ones(n_da)

    # ---- Dataset A validation split (scratch mode) ----
    if args.mode == "scratch":
        n_val = int(n_da * args.val_ratio)
        n_train_da = n_da - n_val
        perm = torch.randperm(n_da, generator=torch.Generator().manual_seed(args.seed))
        da_train_idx, da_val_idx = perm[:n_train_da], perm[n_train_da:]

        val_emb = da_emb[da_val_idx]
        val_can = da_can[da_val_idx]
        val_abn = da_abn[da_val_idx]
        log.info("Scratch mode: DA split %d train / %d val", n_train_da, n_val)

        # Replace DA with train portion only
        da_emb = da_emb[da_train_idx]
        da_can = da_can[da_train_idx]
        da_abn = da_abn[da_train_idx]
        da_cancer_weight = da_cancer_weight[da_train_idx]
        da_abn_mask = da_abn_mask[da_train_idx]
        n_da = n_train_da

    # ---- combine all data ----
    all_emb = torch.cat([pl_emb, neg_emb, da_emb], dim=0)
    all_can = torch.cat([pl_can, neg_can, da_can], dim=0)
    all_abn = torch.cat([pl_abn, neg_abn, da_abn], dim=0)
    all_cw = torch.cat([pl_cancer_weight, neg_cancer_weight, da_cancer_weight], dim=0)
    all_abn_mask = torch.cat([pl_abn_mask, neg_abn_mask, da_abn_mask], dim=0)
    in_dim = all_emb.shape[-1]
    log.info("Total training patches: %d   in_dim: %d", len(all_emb), in_dim)

    # Free memory
    del pl_emb, neg_emb, da_emb
    import gc; gc.collect()

    train_ds = PatchDataset(all_emb, all_can, all_abn, all_cw, all_abn_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)

    # ---- pos_weight ----
    pw_can = (len(all_can) - all_can.sum()) / (all_can.sum() + 1e-8)
    pw_abn = (len(all_abn) - all_abn.sum()) / (all_abn.sum() + 1e-8)
    log.info("pos_weight: cancer=%.4f  abnormal=%.4f", pw_can.item(), pw_abn.item())
    pw_can_t = torch.tensor([pw_can], device=device)
    pw_abn_t = torch.tensor([pw_abn], device=device)

    # ---- Validation loader (scratch only) ----
    if args.mode == "scratch":
        val_ds = PatchDataset(val_emb, val_can, val_abn,
                              torch.ones(len(val_can)), torch.ones(len(val_can)))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    # ---- load model ----
    if args.mode == "ft":
        assert args.slidecheck_ckpt is not None, "ft mode requires --slidecheck_ckpt"
        model = load_slidecheck_from_ckpt(args.slidecheck_ckpt)
        log.info("Loaded Phase1 model from %s", args.slidecheck_ckpt)
    else:
        model = build_slidecheck_mlp(in_dim=in_dim, hidden_dim=768, dropout=0.1)
        log.info("Random initialized model (scratch)")

    model.to(device)
    log.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.mode == "scratch" else None

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info("Training: %s, %d epochs, lr=%g", args.mode, args.epochs, args.lr)
    log.info("  Mixup alpha=%.2f, FP sigma=%.2f", args.mixup_alpha, args.perturb_sigma)
    log.info("  Constraint lambda=%.1f", args.lambda_constraint)
    log.info("=" * 60)

    # ---- training ----
    best_bacc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_can, t_abn, t_con, nb = 0.0, 0.0, 0.0, 0.0, 0

        for emb, c, a, w, am in train_loader:
            emb = emb.to(device)
            c, a, w, am = c.to(device), a.to(device), w.to(device), am.to(device)
            B = emb.size(0)

            # ---- Feature Perturbation ----
            noise = torch.randn_like(emb) * args.perturb_sigma
            emb_fp = emb * (1 + noise)

            # ---- Mixup ----
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            lam = max(lam, 1 - lam)
            idx = torch.randperm(B, device=device)

            emb_mix = lam * emb_fp + (1 - lam) * emb_fp[idx]
            c_mix = lam * c + (1 - lam) * c[idx]
            a_mix = lam * a + (1 - lam) * a[idx]
            w_mix = lam * w + (1 - lam) * w[idx]
            am_mix = lam * am + (1 - lam) * am[idx]

            out = model(emb_mix)

            # Cancer loss: weighted BCE with mixed weights
            loss_can_raw = F.binary_cross_entropy_with_logits(
                out.logit_can, c_mix, pos_weight=pw_can_t, reduction='none')
            loss_can = (loss_can_raw * w_mix).mean()

            # Abnormal loss: use mixed abn_mask as soft weight
            if am_mix.sum() > 0:
                loss_abn_raw = F.binary_cross_entropy_with_logits(
                    out.logit_abn, a_mix, pos_weight=pw_abn_t, reduction='none')
                loss_abn = (loss_abn_raw * am_mix).mean()
            else:
                loss_abn = torch.tensor(0.0, device=device)

            # Constraint
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
        avg_can = t_can / max(nb, 1)
        avg_abn = t_abn / max(nb, 1)
        avg_con = t_con / max(nb, 1)

        lr_now = optimizer.param_groups[0]['lr']
        log.info("Epoch %d/%d  Loss=%.4f (can=%.4f abn=%.4f con=%.4f) LR=%.6f",
                 epoch, args.epochs, avg_loss, avg_can, avg_abn, avg_con, lr_now)

        # ---- Validation (scratch only) ----
        if args.mode == "scratch" and epoch % 1 == 0:
            model.eval()
            all_la, all_lc, all_a, all_c = [], [], [], []
            with torch.no_grad():
                for ve, vc, va, vw, vm in val_loader:
                    ve = ve.to(device)
                    vout = model(ve)
                    all_la.append(vout.logit_abn.cpu())
                    all_lc.append(vout.logit_can.cpu())
                    all_a.append(va)
                    all_c.append(vc)

            prob_abn = torch.sigmoid(torch.cat(all_la)).numpy()
            prob_can = torch.sigmoid(torch.cat(all_lc)).numpy()
            labels_abn = torch.cat(all_a).numpy()
            labels_can = torch.cat(all_c).numpy()

            from sklearn.metrics import roc_auc_score, balanced_accuracy_score
            auc_abn = roc_auc_score(labels_abn, prob_abn)
            auc_can = roc_auc_score(labels_can, prob_can)
            bacc_abn = balanced_accuracy_score(labels_abn, (prob_abn > 0.5).astype(int))
            bacc_can = balanced_accuracy_score(labels_can, (prob_can > 0.5).astype(int))
            mean_bacc = 0.5 * (bacc_abn + bacc_can)

            log.info("  Val: AbnAUC=%.4f CanAUC=%.4f MeanBACC=%.4f", auc_abn, auc_can, mean_bacc)

            if mean_bacc > best_bacc:
                best_bacc = mean_bacc
                best_epoch = epoch
                patience_counter = 0
                torch.save({"state_dict": model.state_dict(),
                             "slidecheck_state_dict": model.state_dict(),
                             "epoch": epoch, "best_bacc": best_bacc},
                           os.path.join(out_dir, "best_model.pt"))
                log.info("  -> New best (MeanBACC=%.4f)", best_bacc)
            else:
                patience_counter += 1
                if patience_counter % 10 == 0:
                    log.info("  No improvement (%d/%d)", patience_counter, args.early_stop_patience)

            if patience_counter >= args.early_stop_patience:
                log.info("Early stopping at epoch %d", epoch)
                break

        # Save per-epoch (ft mode) or periodic (scratch mode)
        if args.mode == "ft":
            torch.save({"state_dict": model.state_dict(),
                        "slidecheck_state_dict": model.state_dict(), "epoch": epoch},
                       os.path.join(out_dir, f"epoch{epoch}.pt"))

    # ---- Save final ----
    torch.save({"state_dict": model.state_dict(),
                "slidecheck_state_dict": model.state_dict(), "epoch": epoch},
               os.path.join(out_dir, "final.pt"))
    if args.mode == "scratch":
        log.info("Best epoch: %d, Best MeanBACC: %.4f", best_epoch, best_bacc)
    log.info("Done. Saved final.pt to %s", out_dir)


if __name__ == "__main__":
    main()
