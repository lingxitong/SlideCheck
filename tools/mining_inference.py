#!/usr/bin/env python3
"""
Mining Inference: Forward pass only, saves per-bag cancer_prob + attention

Uses BOTH SlideCheck AND GatedABMIL_2560
  - SlideCheck provides: cancer_prob [N], abnormal_prob [N] per patch
  - MIL provides: attention [N] (softmax), bag_prob (cancer probability)
  - No inline selection (done later by generate_pseudo_labels_v5a.py)

Output: mining_inference.pt
  {
    wsi_id: {
      'cancer_prob': [N],       # SlideCheck sigmoid output
      'abnormal_prob': [N],     # SlideCheck sigmoid output
      'attention': [N],         # MIL softmax attention
      'bag_prob': float,        # MIL bag-level cancer prob
      'bag_label': int,
      'n_patches': int,
    },
    ...
  }

Usage:
    python tools/mining_inference.py \
        --manifest bag_manifest.csv \
        --slidecheck_ckpt logs/phase1/best_model.pt \
        --mil_ckpt logs/phase2/step1_best.pt \
        --output mining_inference.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.models import build_slidecheck_model, build_gated_abmil_2560
from slidecheck.datasets.h5_bag_dataset import H5BagDataset, bag_collate_fn
from slidecheck.utils.checkpoint import load_checkpoint


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SlideCheck
    print(f"Loading SlideCheck from {args.slidecheck_ckpt}...")
    sc_ckpt = load_checkpoint(args.slidecheck_ckpt)
    sc_config = sc_ckpt['config']
    sc = build_slidecheck_model(
        arch=sc_config.get('arch', 'baseline'),
        foundation_model=sc_config.get('foundation_model', 'virchow2'),
        hidden_dim=sc_config.get('hidden_dim', 768),
        dropout=sc_config.get('dropout', 0.1),
    )
    sc.load_state_dict(sc_ckpt['state_dict'])
    sc = sc.to(device).eval()

    # Load GatedABMIL_2560
    print(f"Loading GatedABMIL from {args.mil_ckpt}...")
    mil_ckpt = torch.load(args.mil_ckpt, map_location='cpu', weights_only=False)
    mil = build_gated_abmil_2560(
        in_dim=2560, attn_hidden=args.attn_hidden,
        num_classes=2, dropout=args.dropout,
    )
    mil.load_state_dict(mil_ckpt['mil_state_dict'])
    mil = mil.to(device).eval()

    # Load data
    print(f"Loading data from {args.manifest}...")
    dataset = H5BagDataset(args.manifest, label_filter=[0, 1])
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=bag_collate_fn, num_workers=0)
    print(f"Total bags: {len(dataset)}")

    # Inference
    results = {}
    for batch in tqdm(loader, desc='Inference'):
        bag = batch[0]
        features = bag['features'].to(device)
        wsi_id = bag['wsi_id']
        bag_label = bag['label'].item()
        N = features.shape[0]

        # SlideCheck: patch-level cancer/abnormal prob
        sc_out = sc(features)
        cancer_prob = torch.sigmoid(sc_out.logit_can).cpu()  # [N]
        abnormal_prob = torch.sigmoid(sc_out.logit_abn).cpu()  # [N]

        # MIL: attention + bag-level prob
        mil_out = mil(features)
        attn = mil_out.attention_normalized.squeeze(-1).cpu()  # [N]
        bag_prob = F.softmax(mil_out.logits, dim=1)[0, 1].cpu().item()

        results[wsi_id] = {
            'cancer_prob': cancer_prob,
            'abnormal_prob': abnormal_prob,
            'attention': attn,
            'bag_prob': bag_prob,
            'bag_label': bag_label,
            'n_patches': N,
        }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(out_path))
    print(f"\nSaved {len(results)} bags to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mining Inference')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to bag manifest CSV')
    parser.add_argument('--slidecheck_ckpt', type=str, required=True,
                        help='Path to SlideCheck checkpoint')
    parser.add_argument('--mil_ckpt', type=str, required=True,
                        help='Path to MIL checkpoint')
    parser.add_argument('--attn_hidden', type=int, default=384,
                        help='MIL attention hidden dim (default: 384)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='MIL dropout rate (default: 0.25)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--output', type=str, default='mining_inference.pt',
                        help='Output path (default: mining_inference.pt)')
    args = parser.parse_args()
    run_inference(args)
