#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SlideCheck MobileNetV3 inference script.

- Loads images from a CSV file (with column: image_path)
- Builds SlideCheck MobileNetV3 model via build_model(tag="mobilenetv3", ...)
- Loads checkpoint (expects checkpoint["state_dict"] or a raw state_dict)
- Runs inference and saves JSON with:
  - sigmoid probabilities (logit_abn, logit_can)
  - binary predictions with threshold (logit_abn_binary, logit_can_binary)

Output format matches SlideCheck_Infer.py for consistency.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from SlideCheck_Model import build_model, ModelOutput


def save_json(data: Dict[Any, Any], save_path: str, indent: int = 2) -> None:
    """Save a dict as a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


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


class ImageDataset(Dataset):
    """Dataset for image inputs (for MobileNetV3 inference)"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        
        # Default transform: resize to 224x224, normalize (same as validation in training)
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
        
        return img, img_path


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="SlideCheck MobileNetV3 inference on images from CSV.")
    parser.add_argument("--image_csv", type=str, required=True, 
                        help="Path to CSV file with column 'image_path' containing image file paths.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to SlideCheck checkpoint (.pt).")
    parser.add_argument("--out_json", type=str, default=None, help="Path to output JSON file.")

    # Model args
    parser.add_argument("--model_tag", type=str, default="mobilenetv3", 
                        help="Model tag (default: mobilenetv3).")
    parser.add_argument("--hidden_dim", type=int, default=768, 
                        help="Hidden dimension for classification head (default: 768).")
    parser.add_argument("--dropout", type=float, default=0.1, 
                        help="Dropout rate (default: 0.1).")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained backbone (default: True).")

    # Inference args
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Sigmoid threshold for binary outputs (default: 0.5).")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for inference (default: 32).")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of workers for DataLoader (default: 8).")
    parser.add_argument("--save_probs", action="store_true", default=True,
                        help="Also save sigmoid probabilities to JSON (default: True).")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2).")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) Load image paths from CSV
    df = pd.read_csv(args.image_csv)
    if "image_path" not in df.columns:
        raise ValueError(f"CSV file must contain 'image_path' column. Available columns: {list(df.columns)}")
    
    image_paths = df["image_path"].dropna().tolist()
    if len(image_paths) == 0:
        raise ValueError("No valid image paths found in CSV file.")
    
    print(f"[INFO] Loaded {len(image_paths)} image paths from CSV.")

    # 2) Build model
    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=False  # For inference, backbone doesn't need to be frozen
    )
    in_dim = 0  # For MobileNetV3, in_dim is ignored (images are input)
    model = build_model(args.model_tag, in_dim=in_dim, **model_kwargs)

    # 3) Load weights
    state_dict = load_checkpoint_state_dict(args.ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading: {unexpected}")

    model.eval().to(device)
    print("✅ SlideCheck checkpoint loaded and model set to eval().")

    # 4) Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5) Inference
    all_p_abn = []
    all_p_can = []
    
    print(f"[INFO] Running inference on {len(image_paths)} images...")
    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)
        
        out: ModelOutput = model(images)
        p_abn = torch.sigmoid(out.logit_abn)
        p_can = torch.sigmoid(out.logit_can)
        
        all_p_abn.append(p_abn.detach().cpu())
        all_p_can.append(p_can.detach().cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"[INFO] Processed {min((batch_idx + 1) * args.batch_size, len(image_paths))} / {len(image_paths)} images...")

    p_abn = torch.cat(all_p_abn)
    p_can = torch.cat(all_p_can)

    # 6) Binarize
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
    
    # 7) Save JSON
    if args.out_json is None:
        # Generate output path from input CSV path
        base_name = os.path.splitext(os.path.basename(args.image_csv))[0]
        output_dir = os.path.dirname(os.path.abspath(args.image_csv))
        args.out_json = os.path.join(output_dir, f"{base_name}_inference_results.json")
    
    save_json(out, args.out_json, indent=args.indent)
    print(f"✅ Saved JSON to: {args.out_json}")
    print(f"[INFO] Inference completed. Processed {len(image_paths)} images.")


if __name__ == "__main__":
    main()

