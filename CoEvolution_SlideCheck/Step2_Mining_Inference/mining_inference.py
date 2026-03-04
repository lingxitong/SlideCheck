"""
Mining 推理: 只做 forward pass, 保存每个 bag 的 cancer_prob + attention

输出: mining_inference.pt
  {
    wsi_id: {
      'cancer_prob': [N],     # SlideCheck sigmoid output
      'attention': [N],       # MIL softmax attention
      'bag_prob': float,      # MIL bag-level cancer prob
      'bag_label': int,
      'n_patches': int,
    },
    ...
  }

后续用 apply_topk_strategy.py 在这个结果上跑不同的 TopK 策略
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
_code_root = str(Path(__file__).parent.parent)
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from models import load_slidecheck_from_ckpt
from models.gated_abmil_2560 import build_gated_abmil_2560
from datasets.h5_bag_dataset import H5BagDataset, bag_collate_fn


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading SlideCheck from {args.slidecheck_ckpt}...")
    sc = load_slidecheck_from_ckpt(args.slidecheck_ckpt, device=str(device))
    sc.eval()

    print(f"Loading GatedABMIL from {args.mil_ckpt}...")
    mil_ckpt = torch.load(args.mil_ckpt, map_location='cpu', weights_only=False)
    mil = build_gated_abmil_2560(
        in_dim=2560, attn_hidden=args.attn_hidden,
        num_classes=2, dropout=args.dropout,
    )
    mil.load_state_dict(mil_ckpt['mil_state_dict'])
    mil = mil.to(device).eval()

    # 加载数据
    print(f"Loading data from {args.manifest}...")
    dataset = H5BagDataset(args.manifest, label_filter=[0, 1])
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=bag_collate_fn, num_workers=0)
    print(f"Total bags: {len(dataset)}")

    # 推理
    results = {}
    for batch in tqdm(loader, desc='Inference'):
        bag = batch[0]
        features = bag['features'].to(device)
        wsi_id = bag['wsi_id']
        bag_label = bag['label'].item()
        N = features.shape[0]

        sc_out = sc(features)
        cancer_prob = torch.sigmoid(sc_out.logit_can).cpu()  # [N]
        abnormal_prob = torch.sigmoid(sc_out.logit_abn).cpu()  # [N]

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

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(out_path))
    print(f"\nSaved {len(results)} bags to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mining Inference Only')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--slidecheck_ckpt', type=str, required=True)
    parser.add_argument('--mil_ckpt', type=str, required=True)
    parser.add_argument('--attn_hidden', type=int, default=384)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='mining_inference.pt')
    args = parser.parse_args()
    run_inference(args)
