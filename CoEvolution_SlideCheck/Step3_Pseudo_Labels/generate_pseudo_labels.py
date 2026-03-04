#!/usr/bin/env python3
"""
伪标签生成 V5a — 在 V5 基础上加宽负袋选择范围

与 V5 的唯一区别: 负袋使用独立的（更宽的）百分位参数。
正袋逻辑完全不变。

负袋参数变化 (V5 → V5a):
  Type 3: cancer_bottom 5%→50%, abn_bottom 5%→50%, attn_top 10%→30%
  Type 2: cancer_bottom 20%→50%, abn_top 10%→30%, attn_top 10%→30%

输入: mining_inference_round1_v2.pt
输出: pseudo_labels_v5a.pt
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm


def generate_pseudo_labels_v5a(results,
                               # 正袋参数 (同 V5)
                               pct_cancer_top=0.05,
                               pct_attn_top=0.10,
                               pct_abn_top=0.10,
                               pct_bottom=0.05,
                               pct_cancer_bot_t2=0.20,
                               pct_attn_bot_t2=0.20,
                               # 负袋参数 (V5a 加宽)
                               neg_pct_bottom=0.50,
                               neg_pct_attn_top=0.30,
                               neg_pct_abn_top=0.30,
                               neg_pct_cancer_bot_t2=0.50,
                               # Hard bag
                               hard_bag_lo=0.4,
                               hard_bag_hi=0.6,
                               hard_bag_weight=3):
    """
    V5a 伪标签生成: 正袋同 V5，负袋加宽选择范围。
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

        is_hard = (bag_prob is not None
                   and hard_bag_lo <= bag_prob <= hard_bag_hi)
        repeat = hard_bag_weight if is_hard else 1
        if is_hard:
            stats['n_hard_bags'] += 1

        can_lab = torch.full((N,), -1, dtype=torch.long)
        abn_lab = torch.full((N,), -1, dtype=torch.long)

        if bag_label == 1:  # ===== 正袋 (同 V5，完全不变) =====
            stats['n_pos_bags'] += 1

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

            # Type 1: Cancer+Abnormal
            type1_idx = sorted(cancer_topk & attn_topk)

            # Type 2: NonCancer+Abnormal
            type2_idx = sorted(cancer_botk_t2 & abn_topk & attn_botk_t2)
            type1_set = set(type1_idx)
            type2_idx = [i for i in type2_idx if i not in type1_set]

            # Type 3: Normal
            type3_idx = sorted(cancer_botk & abn_botk & attn_botk)
            type12_set = type1_set | set(type2_idx)
            type3_idx = [i for i in type3_idx if i not in type12_set]

            for i in type1_idx:
                can_lab[i] = 1; abn_lab[i] = 1
            for i in type2_idx:
                can_lab[i] = 0; abn_lab[i] = 1
            for i in type3_idx:
                can_lab[i] = 0; abn_lab[i] = 0

            n1, n2, n3 = len(type1_idx), len(type2_idx), len(type3_idx)
            stats['pos_type1'] += n1 * repeat
            stats['pos_type2'] += n2 * repeat
            stats['pos_type3'] += n3 * repeat

            if type1_idx:
                stats['_type1_cp'].extend([cp[i].item() for i in type1_idx])
            if type2_idx:
                stats['_type2_cp'].extend([cp[i].item() for i in type2_idx])
                stats['_type2_ap'].extend([ap[i].item() for i in type2_idx])
            if type3_idx:
                stats['_type3_cp'].extend([cp[i].item() for i in type3_idx])
                stats['_type3_ap'].extend([ap[i].item() for i in type3_idx])

        else:  # ===== 负袋 (V5a: 使用加宽的独立参数) =====
            stats['n_neg_bags'] += 1

            # V5a: 负袋使用独立的更宽参数
            k_bottom_neg = max(1, int(N * neg_pct_bottom))
            k_attn_top_neg = max(1, int(N * neg_pct_attn_top))
            k_abn_top_neg = max(1, int(N * neg_pct_abn_top))
            k_cancer_bot_t2_neg = max(1, int(N * neg_pct_cancer_bot_t2))

            cancer_botk = set(cp.topk(k_bottom_neg, largest=False).indices.tolist())
            abn_botk = set(ap.topk(k_bottom_neg, largest=False).indices.tolist())
            abn_topk = set(ap.topk(k_abn_top_neg).indices.tolist())
            attn_topk = set(attn.topk(k_attn_top_neg).indices.tolist())
            cancer_botk_t2 = set(cp.topk(k_cancer_bot_t2_neg, largest=False).indices.tolist())

            # 负袋 Type 3: Normal (cancer bottom ∩ abn bottom ∩ attn TOP)
            neg_type3_idx = sorted(cancer_botk & abn_botk & attn_topk)

            # 负袋 Type 2: NonCancer+Abnormal (cancer bottom ∩ abn TOP ∩ attn TOP)
            neg_type2_idx = sorted(cancer_botk_t2 & abn_topk & attn_topk)
            neg_type3_set = set(neg_type3_idx)
            neg_type2_idx = [i for i in neg_type2_idx if i not in neg_type3_set]

            for i in neg_type3_idx:
                can_lab[i] = 0; abn_lab[i] = 0
            for i in neg_type2_idx:
                can_lab[i] = 0; abn_lab[i] = 1

            n2, n3 = len(neg_type2_idx), len(neg_type3_idx)
            stats['neg_type2'] += n2 * repeat
            stats['neg_type3'] += n3 * repeat

            if neg_type2_idx:
                stats['_type2_cp'].extend([cp[i].item() for i in neg_type2_idx])
                stats['_type2_ap'].extend([ap[i].item() for i in neg_type2_idx])
            if neg_type3_idx:
                stats['_type3_cp'].extend([cp[i].item() for i in neg_type3_idx])
                stats['_type3_ap'].extend([ap[i].item() for i in neg_type3_idx])

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

    # 计算信号质量统计
    import numpy as np
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


def main(args):
    print(f"Loading inference results from {args.inference_results}...")
    results = torch.load(args.inference_results, map_location='cpu', weights_only=False)
    print(f"  {len(results)} bags")

    print(f"\nV5a Config:")
    print(f"  Positive bag params (same as V5):")
    print(f"    pct_cancer_top={args.pct_cancer_top}, pct_attn_top={args.pct_attn_top}")
    print(f"    pct_abn_top={args.pct_abn_top}, pct_bottom={args.pct_bottom}")
    print(f"    pct_cancer_bot_t2={args.pct_cancer_bot_t2}, pct_attn_bot_t2={args.pct_attn_bot_t2}")
    print(f"  Negative bag params (V5a widened):")
    print(f"    neg_pct_bottom={args.neg_pct_bottom}")
    print(f"    neg_pct_attn_top={args.neg_pct_attn_top}")
    print(f"    neg_pct_abn_top={args.neg_pct_abn_top}")
    print(f"    neg_pct_cancer_bot_t2={args.neg_pct_cancer_bot_t2}")

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

    print(f"\n{'='*60}")
    print(f"Pseudo Labels V5a Stats:")
    print(f"{'='*60}")

    print(f"\n--- Counts ---")
    print(f"  Pos bags: {stats['n_pos_bags']}, Neg bags: {stats['n_neg_bags']}")
    print(f"  Hard bags: {stats['n_hard_bags']}")
    print(f"  Total patches in all bags: {stats['n_total_patches']}")
    print(f"  Labeled patches (with repeat): {stats['n_labeled_patches']}")
    print(f"  Labeling rate: {stats['n_labeled_patches']/max(stats['n_total_patches'],1)*100:.1f}%")

    print(f"\n--- By Type (with repeat) ---")
    t1 = stats['pos_type1']
    t2 = stats['pos_type2'] + stats['neg_type2']
    t3 = stats['pos_type3'] + stats['neg_type3']
    total_pl = t1 + t2 + t3
    print(f"  Type 1 (Cancer+Abnormal):     {t1:>8d} ({t1/max(total_pl,1)*100:5.1f}%)  [pos bags only]")
    print(f"  Type 2 (NonCancer+Abnormal):   {t2:>8d} ({t2/max(total_pl,1)*100:5.1f}%)  [pos: {stats['pos_type2']}, neg: {stats['neg_type2']}]")
    print(f"  Type 3 (Normal):               {t3:>8d} ({t3/max(total_pl,1)*100:5.1f}%)  [pos: {stats['pos_type3']}, neg: {stats['neg_type3']}]")
    print(f"  TOTAL:                         {total_pl:>8d}")

    # 与 V5 对比 (V5 没有负袋 patches 的贡献很多)
    print(f"\n--- V5a vs V5 对比 ---")
    print(f"  V5 labeled patches: ~341,115")
    print(f"  V5a labeled patches: {stats['n_labeled_patches']}")
    print(f"  V5a neg_type2: {stats['neg_type2']} (V5: ~5,000)")
    print(f"  V5a neg_type3: {stats['neg_type3']} (V5: ~7,000)")

    print(f"\n--- Signal Quality ---")
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
    print(f"{'='*60}")

    # 标签分布
    all_can, all_abn = [], []
    for wsi_id, d in pseudo.items():
        r = d['repeat']
        for _ in range(r):
            all_can.append(d['cancer_labels'])
            all_abn.append(d['abnormal_labels'])
    if all_can:
        all_can = torch.cat(all_can)
        all_abn = torch.cat(all_abn)
        print(f"\nLabel distribution (with repeat):")
        print(f"  Cancer:   1={(all_can==1).sum().item()}, 0={(all_can==0).sum().item()}")
        print(f"  Abnormal: 1={(all_abn==1).sum().item()}, 0={(all_abn==0).sum().item()}")
        print(f"  Total patches: {len(all_can)}")

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
    p = argparse.ArgumentParser(description='Generate Pseudo Labels V5a (widened negative bags)')
    # 正袋参数 (同 V5)
    p.add_argument('--inference_results', type=str, required=True)
    p.add_argument('--pct_cancer_top', type=float, default=0.05)
    p.add_argument('--pct_attn_top', type=float, default=0.10)
    p.add_argument('--pct_abn_top', type=float, default=0.10)
    p.add_argument('--pct_bottom', type=float, default=0.05)
    p.add_argument('--pct_cancer_bot_t2', type=float, default=0.20)
    p.add_argument('--pct_attn_bot_t2', type=float, default=0.20)
    # 负袋参数 (V5a 加宽)
    p.add_argument('--neg_pct_bottom', type=float, default=0.50,
                   help='Neg bag Type 3: cancer/abn bottom K%% (V5=0.05, V5a=0.50)')
    p.add_argument('--neg_pct_attn_top', type=float, default=0.30,
                   help='Neg bag: attention top K%% (V5=0.10, V5a=0.30)')
    p.add_argument('--neg_pct_abn_top', type=float, default=0.30,
                   help='Neg bag Type 2: abnormal_prob top K%% (V5=0.10, V5a=0.30)')
    p.add_argument('--neg_pct_cancer_bot_t2', type=float, default=0.50,
                   help='Neg bag Type 2: cancer_prob bottom K%% (V5=0.20, V5a=0.50)')
    # Hard bag
    p.add_argument('--hard_bag_lo', type=float, default=0.4)
    p.add_argument('--hard_bag_hi', type=float, default=0.6)
    p.add_argument('--hard_bag_weight', type=int, default=3)
    # Output
    p.add_argument('--output', type=str, default='pseudo_labels_v5a.pt')
    args = p.parse_args()
    main(args)
