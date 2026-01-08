#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SlideCheck inference script (GitHub-friendly).

- Loads patch features from an HDF5 file (dataset key: "features")
- Builds SlideCheck model via build_model(tag, in_dim, **kwargs)
- Loads checkpoint (expects checkpoint["state_dict"] or a raw state_dict)
- Runs inference and saves:
  - sigmoid probabilities (optional)
  - binary predictions with threshold

"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import h5py
import torch

from SlideCheck_Model import build_model  # adjust to your repo layout if needed


def save_json(data: Dict[Any, Any], save_path: str, indent: int = 2) -> None:
    """Save a dict as a JSON file."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_features_from_h5(h5_path: str, key: str = "features") -> torch.Tensor:
    """Load features from an HDF5 file and return a torch.FloatTensor."""
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in H5 file. Available keys: {list(f.keys())}")
        feats = f[key][:]
    x = torch.from_numpy(feats).float()
    return x


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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="SlideCheck inference on HDF5 features.")
    parser.add_argument("--features_h5", type=str, required=True, help="Path to the HDF5 features file.")
    parser.add_argument("--h5_key", type=str, default="features", help="Dataset key in HDF5 (default: features).")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to SlideCheck checkpoint (.pt).")
    parser.add_argument("--out_json", type=str, default=None, help="Path to output JSON file.")

    # Model args
    parser.add_argument("--model_tag", type=str, default="mlp_v1", help="Model tag registered in build_model().")
    parser.add_argument("--in_dim", type=int, required=True, help="Input feature dimension.")

    # Inference args
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary outputs.")
    parser.add_argument("--batch_size", type=int, default=0, help="Optional batching. 0 means no batching.")
    parser.add_argument("--save_probs", default=True, help="Also save sigmoid probabilities to JSON.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2).")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) Build model
    hidden_dim = 768
    dropout = 01.
    model = build_model(
        args.model_tag,
        in_dim=args.in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # 2) Load weights
    state_dict = load_checkpoint_state_dict(args.ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading: {unexpected}")

    model.eval().to(device)
    print("✅ SlideCheck checkpoint loaded and model set to eval().")

    # 3) Load features
    x = load_features_from_h5(args.features_h5, key=args.h5_key)  # [N, D] or [B, D]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.size(-1) != args.in_dim:
        raise ValueError(f"Feature dim mismatch: got {x.size(-1)}, expected {args.in_dim}")
    x = x.to(device)

    # 4) Inference (optional batching)
    def _forward(t: torch.Tensor):
        y = model(t)
        p_abn = torch.sigmoid(y.logit_abn)
        p_can = torch.sigmoid(y.logit_can)
        return p_abn, p_can

    p_abn, p_can = _forward(x)
    p_abn = p_abn.detach().cpu()
    p_can = p_can.detach().cpu()

    # 5) Binarize
    abn_bin = (p_abn > args.threshold).int().tolist()
    can_bin = (p_can > args.threshold).int().tolist()
    out: Dict[str, Any] = {
        "threshold": args.threshold,
        "logit_abn_binary": abn_bin,
        "logit_can_binary": can_bin,
    }
    if args.save_probs:
        out["logit_abn"] = p_abn.tolist()
        out["logit_can"] = p_can.tolist()
    save_json(out, args.out_json, indent=args.indent)
    print(f"Saved JSON to: {args.out_json}")

if __name__ == "__main__":
    main()
