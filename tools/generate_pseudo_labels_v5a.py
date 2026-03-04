#!/usr/bin/env python3
"""
Generate Pseudo Labels V5a - Three-class intersection strategy

V5a Strategy:
- Positive bags: Type 1 (cancer_top ∩ attn_top), Type 2 (cancer_bottom ∩ abn_top ∩ attn_bottom), Type 3 (all_bottom)
- Negative bags (V5a widened): Type 3 (cancer_bottom ∩ abn_bottom ∩ attn_top), Type 2 (cancer_bottom ∩ abn_top ∩ attn_top)

Input: mining_inference_results.pt (from mining_inference.py)
Output: pseudo_labels_v5a.pt

Usage:
    python tools/generate_pseudo_labels_v5a.py \
        --inference_results mining_results.pt \
        --output pseudo_labels_v5a.pt \
        --pct_cancer_top 0.05 \
        --pct_attn_top 0.10 \
        --neg_pct_bottom 0.50
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_pseudo_labels_v5a(
    results,
    # Positive bag parameters (same as V5)
    pct_cancer_top=0.05,
    pct_attn_top=0.10,
    pct_abn_top=0.10,
    pct_bottom=0.05,
    pct_cancer_bot_t2=0.20,
    pct_attn_bot_t2=0.20,
    # Negative bag parameters (V5a widened)
    neg_pct_bottom=0.50,
    neg_pct_attn_top=0.30,
    neg_pct_abn_top=0.30,
    neg_pct_cancer_bot_t2=0.50,
    # Hard bag weighting
    hard_bag_lo=0.4,
    hard_bag_hi=0.6,
    hard_bag_weight=3
):
    """
    Generate pseudo-labels using V5a three-class intersection strategy

    Args:
        results: Dict[wsi_id -> {cancer_prob, abnormal_prob, attention, bag_prob, bag_label}]
        pct_cancer_top: Top K% for cancer probability (positive bags)
        pct_attn_top: Top K% for attention scores (positive bags)
        pct_abn_top: Top K% for abnormal probability (positive bags)
        pct_bottom: Bottom K% for all three signals (positive bags Type 3)
        pct_cancer_bot_t2: Bottom K% for cancer (positive bags Type 2)
        pct_attn_bot_t2: Bottom K% for attention (positive bags Type 2)
        neg_pct_bottom: Bottom K% for cancer/abn (negative bags, V5a=0.50)
        neg_pct_attn_top: Top K% for attention (negative bags, V5a=0.30)
        neg_pct_abn_top: Top K% for abnormal (negative bags Type 2, V5a=0.30)
        neg_pct_cancer_bot_t2: Bottom K% for cancer (negative bags Type 2, V5a=0.50)
        hard_bag_lo: Lower bound for hard bag probability
        hard_bag_hi: Upper bound for hard bag probability
        hard_bag_weight: Repeat weight for hard bags

    Returns:
        pseudo: Dict[wsi_id -> {indices, cancer_labels, abnormal_labels, repeat}]
        stats: Statistics dict
    """
    pseudo = {}
    stats = {
        'pos_type1': 0, 'pos_type2': 0, 'pos_type3': 0,
        'neg_type2': 0, 'neg_type3': 0,
        'n_hard_bags': 0,
        'n_total_patches': 0,
        'n_labeled_patches': 0,
        'n_pos_bags': 0,
        'n_neg_bags': 0,
        '_type1_cp': [], '_type2_cp': [], '_type3_cp': [],
        '_type2_ap': [], '_type3_ap': [],
    }

    for wsi_id, data in tqdm(results.items(), desc='Generating V5a pseudo labels'):
        cp = data['cancer_prob']
        ap = data['abnormal_prob']
        attn = data['attention']
        bag_prob = data.get('bag_prob', None)
        bag_label = data.get('bag_label', None)
        N = len(cp)

        stats['n_total_patches'] += N

        # Check if this is a hard bag
        is_hard = (bag_prob is not None and hard_bag_lo <= bag_prob <= hard_bag_hi)
        repeat = hard_bag_weight if is_hard else 1
        if is_hard:
            stats['n_hard_bags'] += 1

        # Initialize labels as -1 (unlabeled)
        can_lab = torch.full((N,), -1, dtype=torch.long)
        abn_lab = torch.full((N,), -1, dtype=torch.long)

        if bag_label == 1:  # ===== Positive bags (same as V5) =====
            stats['n_pos_bags'] += 1

            # Compute top-k and bottom-k indices
            k_cancer_top = max(1, int(N * pct_cancer_top))
            k_attn_top = max(1, int(N * pct_attn_top))
            k_abn_top = max(1, int(N * pct_abn_top))
            k_bottom = max(1, int(N * pct_bottom))
            k_cancer_bot_t2 = max(1, int(N * pct_cancer_bot_t2))
            k_attn_bot_t2 = max(1, int(N * pct_attn_bot_t2))

            cancer_topk = set(cp.topk(k_cancer_top).indices.tolist())
            attn_topk = set(attn.topk(k_attn_top).indices.tolist())
            abn_topk = set(ap.topk(k_abn_top).indices.tolist())

            cancer_botk = set(cp.topk(k_bottom, largest=False).indices.tolist())
            abn_botk = set(ap.topk(k_bottom, largest=False).indices.tolist())
            attn_botk = set(attn.topk(k_bottom, largest=False).indices.tolist())

            cancer_botk_t2 = set(cp.topk(k_cancer_bot_t2, largest=False).indices.tolist())
            attn_botk_t2 = set(attn.topk(k_attn_bot_t2, largest=False).indices.tolist())

            # Type 1: Cancer + Abnormal (cancer_top ∩ attn_top)
            type1_idx = sorted(cancer_topk & attn_topk)

            # Type 2: Non-Cancer + Abnormal (cancer_bottom ∩ abn_top ∩ attn_bottom)
            type2_idx = sorted(cancer_botk_t2 & abn_topk & attn_botk_t2)
            type1_set = set(type1_idx)
            type2_idx = [i for i in type2_idx if i not in type1_set]

            # Type 3: Normal (cancer_bottom ∩ abn_bottom ∩ attn_bottom)
            type3_idx = sorted(cancer_botk & abn_botk & attn_botk)
            type12_set = type1_set | set(type2_idx)
            type3_idx = [i for i in type3_idx if i not in type12_set]

            # Assign labels
            for i in type1_idx:
                can_lab[i] = 1
                abn_lab[i] = 1
            for i in type2_idx:
                can_lab[i] = 0
                abn_lab[i] = 1
            for i in type3_idx:
                can_lab[i] = 0
                abn_lab[i] = 0

            # Update stats
            n1, n2, n3 = len(type1_idx), len(type2_idx), len(type3_idx)
            stats['pos_type1'] += n1 * repeat
            stats['pos_type2'] += n2 * repeat
            stats['pos_type3'] += n3 * repeat

            # Collect signal quality stats
            if type1_idx:
                stats['_type1_cp'].extend([cp[i].item() for i in type1_idx])
            if type2_idx:
                stats['_type2_cp'].extend([cp[i].item() for i in type2_idx])
                stats['_type2_ap'].extend([ap[i].item() for i in type2_idx])
            if type3_idx:
                stats['_type3_cp'].extend([cp[i].item() for i in type3_idx])
                stats['_type3_ap'].extend([ap[i].item() for i in type3_idx])

        else:  # ===== Negative bags (V5a: widened parameters) =====
            stats['n_neg_bags'] += 1

            # V5a: Use widened parameters for negative bags
            k_bottom_neg = max(1, int(N * neg_pct_bottom))
            k_attn_top_neg = max(1, int(N * neg_pct_attn_top))
            k_abn_top_neg = max(1, int(N * neg_pct_abn_top))
            k_cancer_bot_t2_neg = max(1, int(N * neg_pct_cancer_bot_t2))

            cancer_botk = set(cp.topk(k_bottom_neg, largest=False).indices.tolist())
            abn_botk = set(ap.topk(k_bottom_neg, largest=False).indices.tolist())
            abn_topk = set(ap.topk(k_abn_top_neg).indices.tolist())
            attn_topk = set(attn.topk(k_attn_top_neg).indices.tolist())
            cancer_botk_t2 = set(cp.topk(k_cancer_bot_t2_neg, largest=False).indices.tolist())

            # Negative bag Type 3: Normal (cancer_bottom ∩ abn_bottom ∩ attn_TOP)
            neg_type3_idx = sorted(cancer_botk & abn_botk & attn_topk)

            # Negative bag Type 2: Non-Cancer + Abnormal (cancer_bottom ∩ abn_TOP ∩ attn_TOP)
            neg_type2_idx = sorted(cancer_botk_t2 & abn_topk & attn_topk)
            neg_type3_set = set(neg_type3_idx)
            neg_type2_idx = [i for i in neg_type2_idx if i not in neg_type3_set]

            # Assign labels
            for i in neg_type3_idx:
                can_lab[i] = 0
                abn_lab[i] = 0
            for i in neg_type2_idx:
                can_lab[i] = 0
                abn_lab[i] = 1

            # Update stats
            n2, n3 = len(neg_type2_idx), len(neg_type3_idx)
            stats['neg_type2'] += n2 * repeat
            stats['neg_type3'] += n3 * repeat

            # Collect signal quality stats
            if neg_type2_idx:
                stats['_type2_cp'].extend([cp[i].item() for i in neg_type2_idx])
                stats['_type2_ap'].extend([ap[i].item() for i in neg_type2_idx])
            if neg_type3_idx:
                stats['_type3_cp'].extend([cp[i].item() for i in neg_type3_idx])
                stats['_type3_ap'].extend([ap[i].item() for i in neg_type3_idx])

        # Save labeled patches
        labeled_mask = (can_lab >= 0)
        if labeled_mask.any():
            sel_idx = labeled_mask.nonzero(as_tuple=True)[0]
            pseudo[wsi_id] = {
                'indices': sel_idx,
                'cancer_labels': can_lab[sel_idx],
                'abnormal_labels': abn_lab[sel_idx],
                'repeat': repeat,
            }
            stats['n_labeled_patches'] += len(sel_idx) * repeat

    # Compute signal quality statistics
    for prefix in ['type1', 'type2', 'type3']:
        cp_vals = stats.pop(f'_{prefix}_cp')
        if cp_vals:
            arr = np.array(cp_vals)
            stats[f'{prefix}_cancer_prob_mean'] = float(np.mean(arr))
            stats[f'{prefix}_cancer_prob_std'] = float(np.std(arr))
            stats[f'{prefix}_cancer_prob_gt05'] = float((arr > 0.5).mean())
            stats[f'{prefix}_cancer_prob_gt09'] = float((arr > 0.9).mean())
            stats[f'{prefix}_cancer_prob_lt01'] = float((arr < 0.1).mean())

    for prefix in ['type2', 'type3']:
        ap_vals = stats.pop(f'_{prefix}_ap')
        if ap_vals:
            arr = np.array(ap_vals)
            stats[f'{prefix}_abnormal_prob_mean'] = float(np.mean(arr))
            stats[f'{prefix}_abnormal_prob_gt05'] = float((arr > 0.5).mean())

    return pseudo, stats


def main():
    parser = argparse.ArgumentParser(description='Generate Pseudo Labels V5a')

    # Input/Output
    parser.add_argument('--inference_results', required=True,
                        help='Path to mining inference results .pt file')
    parser.add_argument('--output', default='pseudo_labels_v5a.pt',
                        help='Output path for pseudo-labels')

    # Positive bag parameters (same as V5)
    parser.add_argument('--pct_cancer_top', type=float, default=0.05,
                        help='Positive bag: cancer_prob top K%% (default: 0.05)')
    parser.add_argument('--pct_attn_top', type=float, default=0.10,
                        help='Positive bag: attention top K%% (default: 0.10)')
    parser.add_argument('--pct_abn_top', type=float, default=0.10,
                        help='Positive bag: abnormal_prob top K%% (default: 0.10)')
    parser.add_argument('--pct_bottom', type=float, default=0.05,
                        help='Positive bag Type 3: all bottom K%% (default: 0.05)')
    parser.add_argument('--pct_cancer_bot_t2', type=float, default=0.20,
                        help='Positive bag Type 2: cancer_prob bottom K%% (default: 0.20)')
    parser.add_argument('--pct_attn_bot_t2', type=float, default=0.20,
                        help='Positive bag Type 2: attention bottom K%% (default: 0.20)')

    # Negative bag parameters (V5a widened)
    parser.add_argument('--neg_pct_bottom', type=float, default=0.50,
                        help='Negative bag Type 3: cancer/abn bottom K%% (V5=0.05, V5a=0.50)')
    parser.add_argument('--neg_pct_attn_top', type=float, default=0.30,
                        help='Negative bag: attention top K%% (V5=0.10, V5a=0.30)')
    parser.add_argument('--neg_pct_abn_top', type=float, default=0.30,
                        help='Negative bag Type 2: abnormal_prob top K%% (V5=0.10, V5a=0.30)')
    parser.add_argument('--neg_pct_cancer_bot_t2', type=float, default=0.50,
                        help='Negative bag Type 2: cancer_prob bottom K%% (V5=0.20, V5a=0.50)')

    # Hard bag weighting
    parser.add_argument('--hard_bag_lo', type=float, default=0.4,
                        help='Hard bag lower bound (default: 0.4)')
    parser.add_argument('--hard_bag_hi', type=float, default=0.6,
                        help='Hard bag upper bound (default: 0.6)')
    parser.add_argument('--hard_bag_weight', type=int, default=3,
                        help='Hard bag repeat weight (default: 3)')

    args = parser.parse_args()

    print(f"Loading inference results from {args.inference_results}...")
    results = torch.load(args.inference_results, map_location='cpu', weights_only=False)
    print(f"  {len(results)} bags\n")

    print("V5a Configuration:")
    print("  Positive bag parameters (same as V5):")
    print(f"    pct_cancer_top={args.pct_cancer_top}, pct_attn_top={args.pct_attn_top}")
    print(f"    pct_abn_top={args.pct_abn_top}, pct_bottom={args.pct_bottom}")
    print(f"    pct_cancer_bot_t2={args.pct_cancer_bot_t2}, pct_attn_bot_t2={args.pct_attn_bot_t2}")
    print("  Negative bag parameters (V5a widened):")
    print(f"    neg_pct_bottom={args.neg_pct_bottom}")
    print(f"    neg_pct_attn_top={args.neg_pct_attn_top}")
    print(f"    neg_pct_abn_top={args.neg_pct_abn_top}")
    print(f"    neg_pct_cancer_bot_t2={args.neg_pct_cancer_bot_t2}\n")

    pseudo, stats = generate_pseudo_labels_v5a(
        results,
        pct_cancer_top=args.pct_cancer_top,
        pct_attn_top=args.pct_attn_top,
        pct_abn_top=args.pct_abn_top,
        pct_bottom=args.pct_bottom,
        pct_cancer_bot_t2=args.pct_cancer_bot_t2,
        pct_attn_bot_t2=args.pct_attn_bot_t2,
        neg_pct_bottom=args.neg_pct_bottom,
        neg_pct_attn_top=args.neg_pct_attn_top,
        neg_pct_abn_top=args.neg_pct_abn_top,
        neg_pct_cancer_bot_t2=args.neg_pct_cancer_bot_t2,
        hard_bag_lo=args.hard_bag_lo,
        hard_bag_hi=args.hard_bag_hi,
        hard_bag_weight=args.hard_bag_weight,
    )

    print("=" * 60)
    print("Pseudo Labels V5a Statistics:")
    print("=" * 60)

    print("\n--- Counts ---")
    print(f"  Positive bags: {stats['n_pos_bags']}, Negative bags: {stats['n_neg_bags']}")
    print(f"  Hard bags: {stats['n_hard_bags']}")
    print(f"  Total patches in all bags: {stats['n_total_patches']}")
    print(f"  Labeled patches (with repeat): {stats['n_labeled_patches']}")
    print(f"  Labeling rate: {stats['n_labeled_patches']/max(stats['n_total_patches'],1)*100:.1f}%")

    print("\n--- By Type (with repeat) ---")
    t1 = stats['pos_type1']
    t2 = stats['pos_type2'] + stats['neg_type2']
    t3 = stats['pos_type3'] + stats['neg_type3']
    total_pl = t1 + t2 + t3
    print(f"  Type 1 (Cancer+Abnormal):     {t1:>8d} ({t1/max(total_pl,1)*100:5.1f}%)  [pos bags only]")
    print(f"  Type 2 (NonCancer+Abnormal):   {t2:>8d} ({t2/max(total_pl,1)*100:5.1f}%)  [pos: {stats['pos_type2']}, neg: {stats['neg_type2']}]")
    print(f"  Type 3 (Normal):               {t3:>8d} ({t3/max(total_pl,1)*100:5.1f}%)  [pos: {stats['pos_type3']}, neg: {stats['neg_type3']}]")
    print(f"  TOTAL:                         {total_pl:>8d}")

    print("\n--- Signal Quality ---")
    for prefix, label in [('type1', 'Type 1 (should be cancer)'),
                           ('type2', 'Type 2 (should NOT be cancer)'),
                           ('type3', 'Type 3 (should be normal)')]:
        cp_key = f'{prefix}_cancer_prob_mean'
        if cp_key in stats:
            print(f"  {label}:")
            print(f"    cancer_prob:   mean={stats[f'{prefix}_cancer_prob_mean']:.4f}, "
                  f"std={stats[f'{prefix}_cancer_prob_std']:.4f}, "
                  f">0.5={stats[f'{prefix}_cancer_prob_gt05']*100:.1f}%, "
                  f">0.9={stats[f'{prefix}_cancer_prob_gt09']*100:.1f}%, "
                  f"<0.1={stats[f'{prefix}_cancer_prob_lt01']*100:.1f}%")
            ap_key = f'{prefix}_abnormal_prob_mean'
            if ap_key in stats:
                print(f"    abnormal_prob: mean={stats[ap_key]:.4f}, "
                      f">0.5={stats[f'{prefix}_abnormal_prob_gt05']*100:.1f}%")
    print("=" * 60)

    # Label distribution
    all_can, all_abn = [], []
    for wsi_id, d in pseudo.items():
        r = d['repeat']
        for _ in range(r):
            all_can.append(d['cancer_labels'])
            all_abn.append(d['abnormal_labels'])
    if all_can:
        all_can = torch.cat(all_can)
        all_abn = torch.cat(all_abn)
        print(f"\nLabel Distribution (with repeat):")
        print(f"  Cancer:   1={(all_can==1).sum().item()}, 0={(all_can==0).sum().item()}")
        print(f"  Abnormal: 1={(all_abn==1).sum().item()}, 0={(all_abn==0).sum().item()}")
        print(f"  Total patches: {len(all_can)}")

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'pseudo_labels': pseudo,
        'stats': stats,
        'config': vars(args),
    }, str(out_path))
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
