#!/usr/bin/env python3
"""
Build Training Cache for Co-Evolution Training

This script:
1. Loads pseudo-labels from V5a generation
2. Extracts corresponding embeddings from H5 files
3. Caches everything as .pt file for fast training

Output format:
{
    'embeddings': torch.Tensor [N, D],
    'cancer_labels': torch.Tensor [N],
    'abnormal_labels': torch.Tensor [N],
}

Usage:
    python tools/build_pseudo_label_cache.py \
        --pseudo_labels pseudo_labels_v5a.pt \
        --manifest bag_manifest_train.csv \
        --output pseudo_labels_cached.pt
"""

import argparse
import csv
import sys
import time
import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Build training cache from pseudo-labels')
    parser.add_argument('--pseudo_labels', required=True,
                        help='Path to pseudo_labels_v5a.pt')
    parser.add_argument('--manifest', required=True,
                        help='Path to bag manifest CSV (with h5_path column)')
    parser.add_argument('--output', required=True,
                        help='Output path for cached .pt file')
    parser.add_argument('--wsi_id_column', default='wsi_id',
                        help='Column name for WSI ID in manifest (default: wsi_id)')
    parser.add_argument('--h5_path_column', default='h5_path',
                        help='Column name for H5 path in manifest (default: h5_path)')
    args = parser.parse_args()

    print("Loading pseudo-labels...")
    pl_data = torch.load(args.pseudo_labels, map_location='cpu', weights_only=False)
    pseudo_labels = pl_data['pseudo_labels']
    print(f"  {len(pseudo_labels)} WSIs with pseudo-labels")

    print("\nBuilding manifest dict...")
    manifest = {}
    with open(args.manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            wsi_id = row[args.wsi_id_column]
            h5_path = row[args.h5_path_column]
            # Handle both full path and stem
            manifest[wsi_id] = h5_path
            manifest[Path(h5_path).stem] = h5_path
    print(f"  {len(manifest)} entries in manifest")

    print("\nExtracting embeddings from H5 files...")
    all_emb, all_can, all_abn = [], [], []
    missing = 0
    t0 = time.time()

    for i, (wsi_id, info) in enumerate(tqdm(pseudo_labels.items(), desc='Processing')):
        if wsi_id not in manifest:
            missing += 1
            continue

        h5_path = manifest[wsi_id]
        try:
            with h5py.File(h5_path, 'r') as hf:
                feats = torch.from_numpy(hf['features'][:]).float()
        except Exception as e:
            print(f"\nWarning: Failed to load {h5_path}: {e}")
            missing += 1
            continue

        indices = info['indices']
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)

        cancer_labels = info['cancer_labels']
        if not isinstance(cancer_labels, torch.Tensor):
            cancer_labels = torch.tensor(cancer_labels, dtype=torch.float32)

        abnormal_labels = info['abnormal_labels']
        if not isinstance(abnormal_labels, torch.Tensor):
            abnormal_labels = torch.tensor(abnormal_labels, dtype=torch.float32)

        repeat = int(info.get('repeat', 1))
        patch_feats = feats[indices]

        # Repeat for hard bags
        for _ in range(repeat):
            all_emb.append(patch_feats)
            all_can.append(cancer_labels)
            all_abn.append(abnormal_labels)

    if not all_emb:
        print("\nError: No embeddings extracted! Check manifest and pseudo-labels.")
        sys.exit(1)

    # Concatenate all data
    emb = torch.cat(all_emb, dim=0)
    can = torch.cat(all_can, dim=0)
    abn = torch.cat(all_abn, dim=0)

    print(f"\n{'='*60}")
    print(f"Cache Statistics:")
    print(f"{'='*60}")
    print(f"  Total patches: {emb.shape[0]:,}")
    print(f"  Feature dimension: {emb.shape[1]}")
    print(f"  Cancer labels:   1={(can==1).sum().item():,}  0={(can==0).sum().item():,}")
    print(f"  Abnormal labels: 1={(abn==1).sum().item():,}  0={(abn==0).sum().item():,}")
    print(f"  Missing WSIs: {missing}")
    print(f"  Processing time: {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    # Save cache
    print(f"\nSaving to {args.output}...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'embeddings': emb,
        'cancer_labels': can,
        'abnormal_labels': abn,
    }, str(out_path))

    size_gb = out_path.stat().st_size / 1e9
    print(f"Done! File size: {size_gb:.2f} GB")


if __name__ == '__main__':
    main()
