# train.py
import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

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
class ImageDataset(Dataset):
    """Dataset for image inputs (for MobileNetV3 end-to-end training/finetuning)"""
    def __init__(self, image_paths: List[str], y_abn: torch.Tensor, y_can: torch.Tensor, 
                 indices: List[int], transform: Optional[transforms.Compose] = None):
        self.image_paths = [image_paths[i] for i in indices]
        self.y_abn = y_abn[indices].float()
        self.y_can = y_can[indices].float()
        
        # Default transform: resize to 224x224 and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {img_path}, error: {e}")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.y_abn[idx], self.y_can[idx]


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_y_abn, all_p_abn = [], []
    all_y_can, all_p_can = [], []

    for batch in loader:
        X, y_abn, y_can = batch[0], batch[1], batch[2]
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
    """Best / early-stop metric: mean of bacc from two tasks"""
    bacc_abn = float(val_metrics["abnormal"]["bacc"])
    bacc_can = float(val_metrics["cancer"]["bacc"])
    if math_isnan(bacc_abn) or math_isnan(bacc_can):
        return float("nan")
    return 0.5 * (bacc_abn + bacc_can)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Dual-head finetuning + constraint + exp logging (MobileNetV3)")
    
    # Data input: CSV file with image paths and labels
    ap.add_argument("--image_csv_path", type=str, required=True,
                    help="Path to CSV file with columns: image_path, normal_abnormal_label, cancer_noncancer_label")
    
    # Pre-trained model checkpoint
    ap.add_argument("--resume_model_ckpt", type=str, default="",
                    help="Path to pre-trained model checkpoint to finetune from (optional).")
    
    ap.add_argument("--log_root_dir", type=str, default="./Logs_SlideCheck")
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.05)

    ap.add_argument("--model_tag", type=str, default="mobilenetv3",
                    help=f"Model tag. Available: {list(list_models().keys())}")
    ap.add_argument("--hidden_dim", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze MobileNetV3 backbone (only train classification heads)")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
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
    # Load image paths and labels from CSV
    df = pd.read_csv(args.image_csv_path)
    
    # Check required columns
    required_cols = ["image_path", "normal_abnormal_label", "cancer_noncancer_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    # Read data
    image_paths = df["image_path"].dropna().tolist()
    y_abn_raw = df["normal_abnormal_label"].dropna().tolist()
    y_can_raw = df["cancer_noncancer_label"].dropna().tolist()
    
    # Ensure all lists have the same length
    min_len = min(len(image_paths), len(y_abn_raw), len(y_can_raw))
    image_paths = image_paths[:min_len]
    y_abn_raw = y_abn_raw[:min_len]
    y_can_raw = y_can_raw[:min_len]
    
    # Convert labels to numeric values (supports both string and numeric)
    def normalize_label(val, label_type="abnormal"):
        if isinstance(val, (int, float)):
            return int(val)
        val_str = str(val).strip().lower()
        if label_type == "abnormal":
            label_map = {"normal": 0, "abnormal": 1, "0": 0, "1": 1}
        else:  # cancer
            label_map = {"noncancer": 0, "cancer": 1, "0": 0, "1": 1}
        if val_str in label_map:
            return label_map[val_str]
        try:
            return int(float(val_str))
        except:
            raise ValueError(f"Unable to parse label value: {val} (type: {label_type})")
    
    y_abn = torch.tensor([normalize_label(v, "abnormal") for v in y_abn_raw], dtype=torch.long)
    y_can = torch.tensor([normalize_label(v, "cancer") for v in y_can_raw], dtype=torch.long)
    
    # Validate label consistency
    bad = ((y_can == 1) & (y_abn == 0)).sum().item()
    if bad > 0:
        raise ValueError(f"Found {bad} samples with cancer=1 but abnormal=0 (violates cancer=>abnormal).")
    
    train_idx, val_idx = stratified_split_indices(y_abn, y_can, args.val_ratio, args.seed)
    
    # Image data augmentation (for training)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For validation: only resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds_train = ImageDataset(image_paths, y_abn, y_can, train_idx, transform=train_transform)
    ds_val = ImageDataset(image_paths, y_abn, y_can, val_idx, transform=val_transform)
    

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ---- build model by tag ----
    if args.model_tag == "mobilenetv3":
        model_kwargs = dict(
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout, 
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    else:
        model_kwargs = dict(hidden_dim=args.hidden_dim, dropout=args.dropout, layers=args.layers)
    in_dim = 0
    model = build_model(args.model_tag, in_dim=in_dim, **model_kwargs).to(args.device)

    # ---- load pre-trained checkpoint if provided ----
    if args.resume_model_ckpt and os.path.exists(args.resume_model_ckpt):
        try:
            print(f"[INFO] Loading pre-trained checkpoint from: {args.resume_model_ckpt}")
            state_dict = load_checkpoint_state_dict(args.resume_model_ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading: {unexpected}")
            print("[INFO] Pre-trained checkpoint loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            print("[INFO] Training from scratch.")
    else:
        if args.resume_model_ckpt:
            print(f"[WARN] Checkpoint path provided but file not found: {args.resume_model_ckpt}")
            print("[INFO] Training from scratch.")
        else:
            print("[INFO] No checkpoint provided. Training from scratch.")

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
    print(f"[INFO] Data mode: Images | N={len(image_paths)} | train={len(ds_train)} val={len(ds_val)}")
    print(f"[INFO] pos_weight: abn={pw_abn.item():.4f} can={pw_can.item():.4f}")
    print(f"[INFO] early_stop: patience={args.early_stop_patience} min_delta={args.early_stop_min_delta} warmup={args.early_stop_warmup}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for batch in dl_train:
            Xb, y_abn_b, y_can_b = batch[0], batch[1], batch[2]
            # Xb: [B, 3, 224, 224] - image data
            Xb = Xb.to(args.device, non_blocking=True)
            y_abn_b = y_abn_b.to(args.device, non_blocking=True)
            y_can_b = y_can_b.to(args.device, non_blocking=True)

            # Forward pass:
            # 1. MobileNetV3 backbone: [B, 3, 224, 224] -> [B, backbone_dim] (e.g., 1280)
            # 2. mlp_v1 head: [B, backbone_dim] -> [B, hidden_dim]
            # 3. Classification heads: [B, hidden_dim] -> [B] (logit_abn, logit_can)
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

            # NEW: best.json (detailed metrics + epoch)
            best_json = {
                "best_epoch": best_epoch,
                "best_key_name": "mean_bacc_abn_can",
                "best_key": best_key,
                "train_loss_at_best": float(train_loss),
                "val": val_metrics,                 # All detailed metrics for abnormal/cancer/constraint
                "model_tag": args.model_tag,
                "model_kwargs": model_kwargs,
                "in_dim": int(in_dim),
                "image_csv_path": args.image_csv_path,
                "resume_model_ckpt": args.resume_model_ckpt,
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
        # Do not early stop during warmup period
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

