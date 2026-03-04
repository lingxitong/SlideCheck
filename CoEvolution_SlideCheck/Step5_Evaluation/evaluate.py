#!/usr/bin/env python3
"""
SlideCheck 跨域评测脚本

评测一个或多个 checkpoint 在外部数据集上的表现。
支持 Baseline (SlideCheckMLP) 和 ConcatV2 两种架构。

用法示例:
  # 评测单个模型
  python evaluate.py \
    --models "V4a BL FT:path/to/final.pt:baseline" \
    --bracs_path /path/to/BRACS.pt \
    --unitopatho_path /path/to/UNITOPATHO.pt \
    --output_csv results.csv

  # 评测多个模型
  python evaluate.py \
    --models "Phase1:ckpt1.pt:baseline" "V4a FT:ckpt2.pt:baseline" "V6c SCR:ckpt3.pt:concatv2" \
    --bracs_path /path/to/BRACS.pt \
    --output_csv results.csv

  # 通过配置文件批量评测
  python evaluate.py \
    --model_list models.txt \
    --bracs_path /path/to/BRACS.pt \
    --output_csv results.csv

models.txt 格式 (每行: name:path:arch):
  Phase1 BL:checkpoints/phase1_best.pt:baseline
  V4a BL FT:checkpoints/v4a_ft_final.pt:baseline
"""

import argparse, csv, os, sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from dataclasses import dataclass


# ---- 模型定义 ----
@dataclass
class SlideCheckOutput:
    logit_abn: torch.Tensor
    logit_can: torch.Tensor


class SlideCheckMLP(nn.Module):
    """Baseline architecture"""
    def __init__(self, in_dim=2560, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_abn(feat).squeeze(-1), self.head_can(feat).squeeze(-1)


class SlideCheckMLP_InputConcatV2(nn.Module):
    """ConcatV2 architecture: raw(2560) + zscore(2560) = 5120d"""
    def __init__(self, in_dim=2560, hidden_dim=1024, dropout=0.1):
        super().__init__()
        combined_dim = in_dim * 2
        self.input_norm = nn.LayerNorm(combined_dim, elementwise_affine=False)
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_zscore = (x - mean) / std
        x_combined = torch.cat([x, x_zscore], dim=-1)
        x_combined = self.input_norm(x_combined)
        feat = self.backbone(x_combined)
        return self.head_abn(feat).squeeze(-1), self.head_can(feat).squeeze(-1)


# ---- 加载模型 ----
CKPT_KEY_PRIORITY = ['state_dict', 'model_state_dict', 'slidecheck_state_dict']


def load_model(path, arch, device):
    """加载 checkpoint，自动适配不同 key 格式"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if arch == 'concatv2':
        model = SlideCheckMLP_InputConcatV2()
    else:
        model = SlideCheckMLP()

    # 尝试多种 key
    for key in CKPT_KEY_PRIORITY:
        if key in ckpt:
            model.load_state_dict(ckpt[key])
            return model.to(device).eval()

    raise KeyError(f"Checkpoint {path} 中未找到可用的 state_dict key. "
                   f"可用 keys: {list(ckpt.keys())}")


# ---- 加载数据集 ----
def load_bracs(path):
    """BRACS: 7 类, abnormal=非Normal, cancer=IDC+IC(5,6)"""
    bracs = torch.load(path, map_location='cpu', weights_only=False)
    emb = bracs['features']
    cls_token = emb[:, 0, :]
    mean_pool = emb[:, 1:, :].mean(dim=1)
    emb = torch.cat([cls_token, mean_pool], dim=-1).float()
    labels = bracs['labels'].numpy()
    return {
        'emb': emb,
        'abn': (labels != 0).astype(int),
        'can': np.isin(labels, [5, 6]).astype(int),
        'n': len(labels),
    }


def load_unitopatho(path):
    """UNITOPATHO: 6 类, abnormal=非NORM(1), cancer=HG(2,4)"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    emb = data['embeddings']
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    emb = emb.float()
    glob = data['glob_labels']
    if isinstance(glob, torch.Tensor):
        glob = glob.numpy()
    return {
        'emb': emb,
        'abn': (glob != 1).astype(int),
        'can': np.isin(glob, [2, 4]).astype(int),
        'n': len(glob),
    }


def load_camel(path):
    """CAMEL: 2 头标签 (normal_abnormal + cancer_noncancer)"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    emb = data['embeddings']
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    emb = emb.float()
    na = data['normal_abnormal_labels']
    cc = data['cancer_noncancer_labels']
    if isinstance(na, torch.Tensor):
        na = na.numpy()
    if isinstance(cc, torch.Tensor):
        cc = cc.numpy()
    return {'emb': emb, 'abn': na, 'can': cc, 'n': len(na)}


def load_mhist(path):
    """MHIST: 只有 test set, HP(0) vs SSA(1), 全部为 abnormal"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    emb = data['embeddings'].float()
    labels = data['labels'].numpy()
    test_mask = data['test_mask'].numpy().astype(bool)
    emb = emb[test_mask]
    labels = labels[test_mask]
    return {
        'emb': emb,
        'abn': np.ones(len(labels), dtype=int),
        'can': np.zeros(len(labels), dtype=int),
        'n': len(labels),
        'raw_labels': labels,
    }


# ---- 评测 ----
def evaluate_model(model, emb, device, batch_size=512):
    """推理并返回概率"""
    all_la, all_lc = [], []
    with torch.no_grad():
        for i in range(0, len(emb), batch_size):
            batch = emb[i:i + batch_size].to(device)
            la, lc = model(batch)
            all_la.append(la.cpu())
            all_lc.append(lc.cpu())
    prob_abn = torch.sigmoid(torch.cat(all_la)).numpy()
    prob_can = torch.sigmoid(torch.cat(all_lc)).numpy()
    return prob_abn, prob_can


def parse_model_spec(spec):
    """解析 'name:path:arch' 格式"""
    parts = spec.strip().split(':')
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()
    elif len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), 'baseline'
    else:
        raise ValueError(f"Invalid model spec: {spec}. Expected 'name:path:arch' or 'name:path'")


def parse_args():
    p = argparse.ArgumentParser(description="SlideCheck 跨域评测")
    p.add_argument("--models", nargs='+', default=[],
                   help="模型规格, 格式: 'name:path:arch' (arch=baseline|concatv2)")
    p.add_argument("--model_list", type=str, default=None,
                   help="模型列表文件, 每行 'name:path:arch'")
    p.add_argument("--bracs_path", type=str, default=None, help="BRACS 特征文件")
    p.add_argument("--unitopatho_path", type=str, default=None, help="UNITOPATHO 特征文件")
    p.add_argument("--camel_path", type=str, default=None, help="CAMEL 特征文件")
    p.add_argument("--mhist_path", type=str, default=None, help="MHIST 特征文件")
    p.add_argument("--output_csv", type=str, default="eval_results.csv", help="输出 CSV 路径")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 收集模型规格
    model_specs = []
    for spec in args.models:
        model_specs.append(parse_model_spec(spec))
    if args.model_list and os.path.exists(args.model_list):
        with open(args.model_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    model_specs.append(parse_model_spec(line))

    if not model_specs:
        print("Error: 请通过 --models 或 --model_list 指定至少一个模型")
        sys.exit(1)

    # 检查模型文件存在性
    available = []
    for name, path, arch in model_specs:
        if os.path.exists(path):
            available.append((name, path, arch))
            print(f"[OK]   {name}: {path} ({arch})")
        else:
            print(f"[SKIP] {name}: {path} 不存在")
    print(f"\n可用模型: {len(available)}/{len(model_specs)}")

    if not available:
        print("Error: 没有可用的模型")
        sys.exit(1)

    # 加载数据集
    datasets = {}
    loaders = [
        ('BRACS', args.bracs_path, load_bracs),
        ('UNITOPATHO', args.unitopatho_path, load_unitopatho),
        ('CAMEL', args.camel_path, load_camel),
        ('MHIST_test', args.mhist_path, load_mhist),
    ]
    for ds_name, ds_path, loader_fn in loaders:
        if ds_path and os.path.exists(ds_path):
            print(f"\n加载 {ds_name}...")
            datasets[ds_name] = loader_fn(ds_path)
            print(f"  {ds_name}: {datasets[ds_name]['n']} 样本")

    if not datasets:
        print("Error: 请至少指定一个数据集路径")
        sys.exit(1)

    # 评测
    all_results = []
    for ds_name, ds in datasets.items():
        emb, abn, can = ds['emb'], ds['abn'], ds['can']
        has_abn = len(np.unique(abn)) >= 2
        has_can = len(np.unique(can)) >= 2

        print()
        print("=" * 110)
        print(f"{ds_name} ({ds['n']} 样本)")
        print("=" * 110)

        if ds_name == 'MHIST_test':
            raw_labels = ds['raw_labels']
            print(f"{'Model':<16} {'Abn Prob':>10} {'HP Abn':>10} {'SSA Abn':>10} "
                  f"{'Can Prob':>10} {'HP Can':>10} {'SSA Can':>10}")
            print("-" * 110)
            for name, path, arch in available:
                try:
                    model = load_model(path, arch, device)
                    prob_abn, prob_can = evaluate_model(model, emb, device)
                    print(f"{name:<16} {prob_abn.mean():>10.4f} {prob_abn[raw_labels==0].mean():>10.4f} "
                          f"{prob_abn[raw_labels==1].mean():>10.4f} {prob_can.mean():>10.4f} "
                          f"{prob_can[raw_labels==0].mean():>10.4f} {prob_can[raw_labels==1].mean():>10.4f}")
                    all_results.append({
                        'dataset': ds_name, 'model': name,
                        'abn_auc': float('nan'), 'abn_bacc': float('nan'),
                        'can_auc': float('nan'), 'can_bacc': float('nan'),
                    })
                except Exception as e:
                    print(f"{name:<16} ERROR: {e}")
        else:
            print(f"{'Model':<16} {'Abn AUC':>9} {'Abn BACC':>9} {'Can AUC':>9} {'Can BACC':>9}")
            print("-" * 110)
            for name, path, arch in available:
                try:
                    model = load_model(path, arch, device)
                    prob_abn, prob_can = evaluate_model(model, emb, device)
                    abn_auc = roc_auc_score(abn, prob_abn) if has_abn else float('nan')
                    abn_bacc = balanced_accuracy_score(abn, (prob_abn > 0.5).astype(int)) if has_abn else float('nan')
                    can_auc = roc_auc_score(can, prob_can) if has_can else float('nan')
                    can_bacc = balanced_accuracy_score(can, (prob_can > 0.5).astype(int)) if has_can else float('nan')
                    print(f"{name:<16} {abn_auc:>9.4f} {abn_bacc:>9.4f} {can_auc:>9.4f} {can_bacc:>9.4f}")
                    all_results.append({
                        'dataset': ds_name, 'model': name,
                        'abn_auc': abn_auc, 'abn_bacc': abn_bacc,
                        'can_auc': can_auc, 'can_bacc': can_bacc,
                    })
                except Exception as e:
                    print(f"{name:<16} ERROR: {e}")
        print("=" * 110)

    # 保存结果
    if all_results:
        with open(args.output_csv, 'w', newline='') as f:
            fieldnames = ['dataset', 'model', 'abn_auc', 'abn_bacc', 'can_auc', 'can_bacc']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n结果已保存到 {args.output_csv}")

    print("Done!")


if __name__ == "__main__":
    main()
