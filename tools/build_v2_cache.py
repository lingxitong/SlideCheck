#!/usr/bin/env python3
"""
Build V2 Training Cache - Merge Pseudo-Labels + Dataset A

This script merges:
1. Pseudo-labeled patches (from build_pseudo_label_cache.py)
2. Dataset A patches (original training data)

Output format (V2 cache):
{
    'pl_emb': torch.Tensor [N_pl, D],
    'pl_can': torch.Tensor [N_pl],
    'pl_abn': torch.Tensor [N_pl],
    'da_emb': torch.Tensor [N_da, D],
    'da_can': torch.Tensor [N_da],
    'da_abn': torch.Tensor [N_da],
}

Usage:
    python tools/build_v2_cache.py \
        --pseudo_label_cache pseudo_labels_cached.pt \
        --dataset_a_path dataset_a.pt \
        --output training_cache_v2.pt
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Build V2 training cache')
    parser.add_argument('--pseudo_label_cache', required=True,
                        help='Path to cached pseudo-labels .pt file')
    parser.add_argument('--dataset_a_path', required=True,
                        help='Path to Dataset A .pt file')
    parser.add_argument('--output', default='training_cache_v2.pt',
                        help='Output path for V2 cache')
    args = parser.parse_args()

    print("Loading pseudo-label cache...")
    pl_cache = torch.load(args.pseudo_label_cache, map_location='cpu', weights_only=False)
    pl_emb = pl_cache['embeddings']
    pl_can = pl_cache['cancer_labels']
    pl_abn = pl_cache['abnormal_labels']
    print(f"  Pseudo-labels: {len(pl_emb):,} patches, dim={pl_emb.shape[1]}")

    print("\nLoading Dataset A...")
    da_data = torch.load(args.dataset_a_path, map_location='cpu', weights_only=False)

    # Handle different Dataset A formats
    if 'embeddings' in da_data:
        da_emb = da_data['embeddings']
        da_can = da_data.get('cancer_noncancer_labels', da_data.get('cancer_labels'))
        da_abn = da_data.get('normal_abnormal_labels', da_data.get('abnormal_labels'))
    else:
        raise ValueError(f"Unknown Dataset A format. Keys: {list(da_data.keys())}")

    print(f"  Dataset A: {len(da_emb):,} patches, dim={da_emb.shape[1]}")

    # Verify dimensions match
    if pl_emb.shape[1] != da_emb.shape[1]:
        raise ValueError(f"Dimension mismatch: pseudo-labels={pl_emb.shape[1]}, Dataset A={da_emb.shape[1]}")

    # Build V2 cache
    print("\nBuilding V2 cache...")
    v2_cache = {
        'pl_emb': pl_emb,
        'pl_can': pl_can,
        'pl_abn': pl_abn,
        'da_emb': da_emb,
        'da_can': da_can,
        'da_abn': da_abn,
    }

    print(f"\n{'='*60}")
    print(f"V2 Cache Statistics:")
    print(f"{'='*60}")
    print(f"  Pseudo-labels:")
    print(f"    Patches: {len(pl_emb):,}")
    print(f"    Cancer:   1={(pl_can==1).sum().item():,}  0={(pl_can==0).sum().item():,}")
    print(f"    Abnormal: 1={(pl_abn==1).sum().item():,}  0={(pl_abn==0).sum().item():,}")
    print(f"  Dataset A:")
    print(f"    Patches: {len(da_emb):,}")
    print(f"    Cancer:   1={(da_can==1).sum().item():,}  0={(da_can==0).sum().item():,}")
    print(f"    Abnormal: 1={(da_abn==1).sum().item():,}  0={(da_abn==0).sum().item():,}")
    print(f"  Total: {len(pl_emb) + len(da_emb):,} patches")
    print(f"{'='*60}")

    # Save V2 cache
    print(f"\nSaving to {args.output}...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(v2_cache, str(out_path))

    size_gb = out_path.stat().st_size / 1e9
    print(f"Done! File size: {size_gb:.2f} GB")


if __name__ == '__main__':
    main()
