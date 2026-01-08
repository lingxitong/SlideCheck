# train.py
import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from SlideCheck_Model import build_model, list_models, ModelOutput


# -------------------------
# Utils
# -------------------------
def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint and return the state_dict.
    Supports:
      1) {'state_dict': ...}
      2) raw state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # raw state_dict
    raise ValueError(
        "Unrecognized checkpoint format. Expect checkpoint['state_dict'] or a raw state_dict."
    )
    

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_2d_embeddings(x: torch.Tensor) -> torch.Tensor:
    # [N,D] or [1,N,D] -> [N,D]
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f"embeddings has 3 dims but first dim != 1: {tuple(x.shape)}")
        x = x.squeeze(0)
    if x.dim() != 2:
        raise ValueError(f"embeddings must be [N,D] or [1,N,D], got {tuple(x.shape)}")
    return x


def stratified_split_indices(y_abn: torch.Tensor, y_can: torch.Tensor, val_ratio: float, seed: int):
    """
    3 strata (because cancer implies abnormal):
      - normal: (abn=0, can=0)
      - abnormal-noncancer: (abn=1, can=0)
      - cancer: (abn=1, can=1)
    """
    n = y_abn.numel()
    raw = (2 * y_abn + y_can).tolist()
    groups = []
    for r in raw:
        if r == 0:
            groups.append(0)
        elif r == 2:
            groups.append(1)
        elif r == 3:
            groups.append(2)
        else:
            raise ValueError("Invalid label combo (abn=0, can=1) violates cancer=>abnormal.")

    g2idx = {0: [], 1: [], 2: []}
    for i, g in enumerate(groups):
        g2idx[g].append(i)

    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []

    for g, idxs in g2idx.items():
        if not idxs:
            continue
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        idxs = [idxs[p] for p in perm]
        v = int(round(len(idxs) * val_ratio))
        if val_ratio > 0 and len(idxs) >= 20 and v == 0:
            v = 1
        val_idx.extend(idxs[:v])
        train_idx.extend(idxs[v:])

    if len(val_idx) == 0 and val_ratio > 0:
        all_idx = list(range(n))
        perm = torch.randperm(n, generator=gen).tolist()
        all_idx = [all_idx[p] for p in perm]
        v = max(1, int(round(n * val_ratio)))
        val_idx = all_idx[:v]
        train_idx = all_idx[v:]

    return train_idx, val_idx


def compute_pos_weight(y01: torch.Tensor) -> torch.Tensor:
    pos = float((y01 == 1).sum().item())
    neg = float((y01 == 0).sum().item())
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def math_isnan(x: float) -> bool:
    return x != x


# -------------------------
# Metrics (pure python / torch)
# -------------------------
@torch.no_grad()
def _auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    ROC-AUC for binary labels in {0,1}. Handles ties.
    Returns nan if only one class exists.
    """
    y_true = y_true.detach().cpu().to(torch.float64)
    y_score = y_score.detach().cpu().to(torch.float64)

    pos = (y_true == 1).sum().item()
    neg = (y_true == 0).sum().item()
    if pos == 0 or neg == 0:
        return float("nan")

    order = torch.argsort(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    n = y_true_sorted.numel()
    ranks = torch.empty(n, dtype=torch.float64)

    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        ranks[i: j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    sum_pos_ranks = ranks[y_true_sorted == 1].sum().item()
    u = sum_pos_ranks - (pos * (pos + 1) / 2.0)
    auc = u / (pos * neg)
    return float(auc)


@torch.no_grad()
def _auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Average Precision (AUPRC). Returns nan if no positive samples.
    """
    y_true = y_true.detach().cpu().to(torch.float64)
    y_score = y_score.detach().cpu().to(torch.float64)

    pos = (y_true == 1).sum().item()
    if pos == 0:
        return float("nan")

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    tp = torch.cumsum(y_true == 1, dim=0).to(torch.float64)
    fp = torch.cumsum(y_true == 0, dim=0).to(torch.float64)

    precision = tp / torch.clamp(tp + fp, min=1.0)

    ap = precision[y_true == 1].sum().item() / pos
    return float(ap)


@torch.no_grad()
def compute_binary_metrics(
    y_true01: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    y_true01: shape [N], values {0,1}
    y_prob:   shape [N], probabilities in [0,1]
    Positive label = 1
    """
    y_true01 = y_true01.detach().cpu().to(torch.long)
    y_prob = y_prob.detach().cpu().to(torch.float64)

    y_pred = (y_prob >= threshold).to(torch.long)

    tp = int(((y_pred == 1) & (y_true01 == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true01 == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true01 == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true01 == 1)).sum().item())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else float("nan")

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    bacc = (sens + spec) / 2.0 if (not math_isnan(sens) and not math_isnan(spec)) else float("nan")

    auc = _auc_roc(y_true01, y_prob)
    auprc = _auprc(y_true01, y_prob)

    return {
        "acc": float(acc),
        "bacc": float(bacc),
        "auc": float(auc),
        "auprc": float(auprc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# -------------------------
# Dataset
# -------------------------
class EmbDataset(Dataset):
    def __init__(self, X: torch.Tensor, y_abn: torch.Tensor, y_can: torch.Tensor, indices: List[int]):
        self.X = X[indices]
        self.y_abn = y_abn[indices].float()
        self.y_can = y_can[indices].float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_abn[idx], self.y_can[idx]


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_y_abn, all_p_abn = [], []
    all_y_can, all_p_can = [], []

    for X, y_abn, y_can in loader:
        X = X.to(device)
        y_abn = y_abn.to(device)
        y_can = y_can.to(device)

        out: ModelOutput = model(X)
        p_abn = torch.sigmoid(out.logit_abn)
        p_can = torch.sigmoid(out.logit_can)

        all_y_abn.append(y_abn.detach().cpu())
        all_p_abn.append(p_abn.detach().cpu())
        all_y_can.append(y_can.detach().cpu())
        all_p_can.append(p_can.detach().cpu())

    y_abn = torch.cat(all_y_abn).to(torch.long)
    p_abn = torch.cat(all_p_abn)
    y_can = torch.cat(all_y_can).to(torch.long)
    p_can = torch.cat(all_p_can)

    violation_rate = float((p_can > p_abn).float().mean().item())

    m_abn = compute_binary_metrics(y_abn, p_abn, threshold=0.5)
    m_can = compute_binary_metrics(y_can, p_can, threshold=0.5)

    return {
        "abnormal": m_abn,
        "cancer": m_can,
        "constraint": {"violation_rate": violation_rate},
    }


def mean_bacc_abn_can(val_metrics: Dict[str, Any]) -> float:
    """best / early-stop 指标：两个任务 bacc 的均值"""
    bacc_abn = float(val_metrics["abnormal"]["bacc"])
    bacc_can = float(val_metrics["cancer"]["bacc"])
    if math_isnan(bacc_abn) or math_isnan(bacc_can):
        return float("nan")
    return 0.5 * (bacc_abn + bacc_can)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Dual-head training + constraint + exp logging")
    ap.add_argument("--pt_path", type=str, required=True)
    ap.add_argument("--log_root_dir", type=str, default="./Logs_SlideCheck")
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--resume_model_ckpt", type=str, default='')

    ap.add_argument("--model_tag", type=str, default="mlp_v1",
                    help=f"Model tag. Available: {list(list_models().keys())}")
    ap.add_argument("--hidden_dim", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--layers", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1000000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--use_pos_weight", action="store_true")

    ap.add_argument("--lambda_constraint", type=float, default=1.0)

    # ====== Early Stop ======
    ap.add_argument("--early_stop_patience", type=int, default=30,
                    help="Stop if no improvement on mean_bacc for this many epochs.")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum improvement over best_key to be counted as improvement.")
    ap.add_argument("--early_stop_warmup", type=int, default=0,
                    help="Do not early-stop before this many epochs.")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- build exp dir ----
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    exp_folder = ts if (args.exp_name is None or str(args.exp_name).strip() == "") else f"{ts}__{args.exp_name}"
    exp_dir = os.path.join(args.log_root_dir, exp_folder)
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- save hparams ----
    hparams = vars(args).copy()
    hparams["exp_dir"] = exp_dir
    save_json(hparams, os.path.join(exp_dir, "hparams.json"))

    # ---- load data ----
    data: Dict[str, Any] = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    for k in ["embeddings", "normal_abnormal_labels", "cancer_noncancer_labels"]:
        if k not in data:
            raise KeyError(f"Missing key in pt dict: {k}")

    X = data["embeddings"]
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    X = ensure_2d_embeddings(X.float())

    y_abn = torch.tensor(data["normal_abnormal_labels"], dtype=torch.long).view(-1)
    y_can = torch.tensor(data["cancer_noncancer_labels"], dtype=torch.long).view(-1)

    if X.shape[0] != y_abn.numel() or X.shape[0] != y_can.numel():
        raise ValueError(f"Length mismatch: N={X.shape[0]} y_abn={y_abn.numel()} y_can={y_can.numel()}")

    bad = ((y_can == 1) & (y_abn == 0)).sum().item()
    if bad > 0:
        raise ValueError(f"Found {bad} samples with cancer=1 but abnormal=0 (violates cancer=>abnormal).")

    train_idx, val_idx = stratified_split_indices(y_abn, y_can, args.val_ratio, args.seed)
    ds_train = EmbDataset(X, y_abn, y_can, train_idx)
    ds_val = EmbDataset(X, y_abn, y_can, val_idx)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    in_dim = X.shape[1]

    # ---- build model by tag ----
    model_kwargs = dict(hidden_dim=args.hidden_dim, dropout=args.dropout, layers=args.layers)
    model = build_model(args.model_tag, in_dim=in_dim, **model_kwargs).to(args.device)
    if os.path.exists(args.resume_model_ckpt):
        try:
            state_dict = load_checkpoint_state_dict(args.ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                assert f"Missing keys when loading: {missing}"
            if unexpected:
                assert f"Unexpected keys when loading: {unexpected}"
        except:
            print('Training from the begining.')

    # ---- losses ----
    if args.use_pos_weight:
        pw_abn = compute_pos_weight(ds_train.y_abn.long()).to(args.device)
        pw_can = compute_pos_weight(ds_train.y_can.long()).to(args.device)
    else:
        pw_abn = torch.tensor(1.0, device=args.device)
        pw_can = torch.tensor(1.0, device=args.device)

    bce_abn = nn.BCEWithLogitsLoss(pos_weight=pw_abn)
    bce_can = nn.BCEWithLogitsLoss(pos_weight=pw_can)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- tracking ----
    best_key = -1e18
    best_epoch = -1
    best_path = os.path.join(ckpt_dir, "best.pt")

    best_json_path = os.path.join(exp_dir, "best.json")  # NEW
    history_path = os.path.join(exp_dir, "metrics.json")

    history: List[Dict[str, Any]] = []

    # early stop state
    bad_epochs = 0  # number of epochs without improvement

    print(f"[EXP] {exp_dir}")
    print(f"[INFO] model_tag={args.model_tag} in_dim={in_dim} kwargs={model_kwargs}")
    print(f"[INFO] N={X.shape[0]} D={X.shape[1]} | train={len(ds_train)} val={len(ds_val)}")
    print(f"[INFO] pos_weight: abn={pw_abn.item():.4f} can={pw_can.item():.4f}")
    print(f"[INFO] early_stop: patience={args.early_stop_patience} min_delta={args.early_stop_min_delta} warmup={args.early_stop_warmup}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for Xb, y_abn_b, y_can_b in dl_train:
            Xb = Xb.to(args.device, non_blocking=True)
            y_abn_b = y_abn_b.to(args.device, non_blocking=True)
            y_can_b = y_can_b.to(args.device, non_blocking=True)

            out: ModelOutput = model(Xb)

            loss1 = bce_abn(out.logit_abn, y_abn_b)
            loss2 = bce_can(out.logit_can, y_can_b)

            p_abn = torch.sigmoid(out.logit_abn)
            p_can = torch.sigmoid(out.logit_can)
            c = torch.relu(p_can - p_abn)
            loss_c = (c * c).mean()

            loss = loss1 + loss2 + args.lambda_constraint * loss_c

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = Xb.size(0)
            loss_sum += loss.item() * bs
            n_seen += bs

        train_loss = loss_sum / max(n_seen, 1)

        val_metrics = evaluate(model, dl_val, args.device)

        abn = val_metrics["abnormal"]
        can = val_metrics["cancer"]
        vio = val_metrics["constraint"]["violation_rate"]

        cur_key = mean_bacc_abn_can(val_metrics)

        line = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "best_key_so_far": float(best_key),
            "cur_key": float(cur_key),
            "val": val_metrics,
        }
        history.append(line)
        save_json({"history": history}, history_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"[ABN] bacc={abn['bacc']:.4f} acc={abn['acc']:.4f} auc={abn['auc']:.4f} "
            f"sens={abn['sensitivity']:.4f} spec={abn['specificity']:.4f} auprc={abn['auprc']:.4f} | "
            f"[CAN] bacc={can['bacc']:.4f} acc={can['acc']:.4f} auc={can['auc']:.4f} "
            f"sens={can['sensitivity']:.4f} spec={can['specificity']:.4f} auprc={can['auprc']:.4f} | "
            f"viol={vio:.4f} | mean_bacc={cur_key:.4f}"
        )

        # ---------- update best + write best.json ----------
        improved = (not math_isnan(cur_key)) and (cur_key > (best_key + args.early_stop_min_delta))
        if improved:
            best_key = float(cur_key)
            best_epoch = int(epoch)
            bad_epochs = 0

            torch.save(
                {
                    "model_tag": args.model_tag,
                    "model_kwargs": model_kwargs,
                    "in_dim": in_dim,
                    "state_dict": model.state_dict(),
                    "train_args": vars(args),
                    "best_key": best_key,
                    "best_key_name": "mean_bacc_abn_can",
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "train_loss_at_best": float(train_loss),
                },
                best_path,
            )

            # NEW: best.json（你要的“详细指标+epoch”）
            best_json = {
                "best_epoch": best_epoch,
                "best_key_name": "mean_bacc_abn_can",
                "best_key": best_key,
                "train_loss_at_best": float(train_loss),
                "val": val_metrics,                 # abnormal/cancer/constraint 全部详细指标
                "model_tag": args.model_tag,
                "model_kwargs": model_kwargs,
                "in_dim": int(in_dim),
                "pt_path": args.pt_path,
                "exp_dir": exp_dir,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            save_json(best_json, best_json_path)

            print(
                f"[SAVE] best (mean_bacc={best_key:.4f}; "
                f"abn_bacc={float(val_metrics['abnormal']['bacc']):.4f}, "
                f"can_bacc={float(val_metrics['cancer']['bacc']):.4f}) -> {best_path}"
            )
        else:
            bad_epochs += 1

        # ---------- early stopping ----------
        # warmup 期间不早停
        if epoch >= args.early_stop_warmup:
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                print(
                    f"[EARLY STOP] epoch={epoch} | "
                    f"no improvement on mean_bacc for {bad_epochs} epochs "
                    f"(best={best_key:.6f} at epoch={best_epoch})"
                )
                break

    print(f"[DONE] best_key={best_key:.6f} best_epoch={best_epoch} best_json={best_json_path}")


if __name__ == "__main__":
    main()
