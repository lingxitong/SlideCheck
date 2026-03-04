#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SlideCheck inference script (backward compatibility wrapper).

This script now internally uses the slidecheck library's SlideCheckPredictor
while maintaining the original interface for backward compatibility.

Usage:
    python SlideCheck_Infer.py --features_h5 features.h5 --ckpt model.pt --out_json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Import from slidecheck library
sys.path.insert(0, str(Path(__file__).parent.parent))
from slidecheck.inference import SlideCheckPredictor


def save_json(data: Dict[Any, Any], save_path: str, indent: int = 2) -> None:
    """Save a dict as a JSON file."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="SlideCheck inference on HDF5 features.")
    parser.add_argument("--features_h5", type=str, required=True, help="Path to the HDF5 features file.")
    parser.add_argument("--h5_key", type=str, default="features", help="Dataset key in HDF5 (default: features).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SlideCheck checkpoint (.pt).")
    parser.add_argument("--out_json", type=str, required=True, help="Path to output JSON file.")

    # Inference args
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary outputs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
    parser.add_argument("--save_probs", action="store_true", default=True, help="Also save sigmoid probabilities to JSON.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2).")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"Loading SlideCheck model from: {args.ckpt}")
    print(f"Device: {device}")

    # Use slidecheck library's predictor
    predictor = SlideCheckPredictor(
        ckpt_path=args.ckpt,
        device=device
    )

    print(f"Loading features from: {args.features_h5}")

    # Run inference
    results = predictor.predict_from_h5(
        h5_path=args.features_h5,
        h5_key=args.h5_key,
        batch_size=args.batch_size
    )

    # Extract probabilities
    prob_abn = torch.from_numpy(results['prob_abn'])
    prob_can = torch.from_numpy(results['prob_can'])

    # Binarize
    abn_bin = (prob_abn > args.threshold).int().tolist()
    can_bin = (prob_can > args.threshold).int().tolist()

    # Prepare output
    out: Dict[str, Any] = {
        "threshold": args.threshold,
        "logit_abn_binary": abn_bin,
        "logit_can_binary": can_bin,
    }

    if args.save_probs:
        out["logit_abn"] = prob_abn.tolist()
        out["logit_can"] = prob_can.tolist()

    # Save results
    save_json(out, args.out_json, indent=args.indent)
    print(f"✅ Inference complete! Results saved to: {args.out_json}")
    print(f"   Total patches: {len(abn_bin)}")
    print(f"   Abnormal predictions: {sum(abn_bin)} / {len(abn_bin)}")
    print(f"   Cancer predictions: {sum(can_bin)} / {len(can_bin)}")


if __name__ == "__main__":
    main()
