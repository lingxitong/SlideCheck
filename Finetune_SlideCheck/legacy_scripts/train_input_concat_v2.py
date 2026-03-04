# -*- coding: utf-8 -*-
"""
Input Concat V2: Raw Features + Z-score Normalized Features
修正V1的尺度不平衡问题：
- 归一化分支：InstanceNorm (Z-score) 替代 L2 Norm
- 拼接后：LayerNorm(affine=False) 平衡两部分尺度
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm


class SlideCheckDataset(Dataset):
    def __init__(self, embeddings, na_labels, cc_labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.na_labels = torch.from_numpy(na_labels).long()
        self.cc_labels = torch.from_numpy(cc_labels).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.na_labels[idx], self.cc_labels[idx]


class SlideCheckMLP_InputConcatV2(nn.Module):
    """Input Concat V2: raw(2560) + zscore(2560) = 5120 -> LN(no affine) -> 3层h=1024
    修正：Z-score替代L2 Norm + LayerNorm(affine=False)平衡尺度
    """
    def __init__(self, in_dim=2560, hidden_dim=1024, dropout=0.1):
        super().__init__()
        combined_dim = in_dim * 2  # 5120

        # 拼接后先用LayerNorm(affine=False)平衡两部分尺度
        self.input_norm = nn.LayerNorm(combined_dim, elementwise_affine=False)

        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, 2560]
        # Z-score归一化 (InstanceNorm): 去除样本级均值和方差
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_zscore = (x - mean) / std

        # 拼接: [raw, zscore]
        x_combined = torch.cat([x, x_zscore], dim=-1)  # [B, 5120]

        # LayerNorm(affine=False) 平衡两部分尺度
        x_combined = self.input_norm(x_combined)

        feat = self.backbone(x_combined)
        logit_abn = self.head_abn(feat).squeeze(-1)
        logit_can = self.head_can(feat).squeeze(-1)
        return logit_abn, logit_can


def load_merged_data(feat_dir):
    files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.pt')])
    all_embeddings = []
    all_na_labels = []
    all_cc_labels = []

    print(f"\n加载数据集:")
    for f in files:
        path = os.path.join(feat_dir, f)
        data = torch.load(path, map_location='cpu', weights_only=False)
        emb = data['embeddings']
        na_labels = data['normal_abnormal_labels']
        cc_labels = data['cancer_noncancer_labels']
        all_embeddings.append(emb)
        all_na_labels.append(na_labels)
        all_cc_labels.append(cc_labels)
        print(f"  {f}: {emb.shape[0]} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)
    na_labels = np.concatenate(all_na_labels, axis=0)
    cc_labels = np.concatenate(all_cc_labels, axis=0)

    print(f"\n合并后总样本数: {embeddings.shape[0]}")
    print(f"  Normal: {(na_labels == 0).sum()} | Abnormal: {(na_labels == 1).sum()}")
    print(f"  Non-cancer: {(cc_labels == 0).sum()} | Cancer: {(cc_labels == 1).sum()}")
    return embeddings, na_labels, cc_labels


def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) == 1:
        auc = 1.0 if acc == 1.0 else 0.0
        auprc = 1.0 if acc == 1.0 else 0.0
    else:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {'acc': acc, 'bacc': bacc, 'auc': auc, 'auprc': auprc,
            'sensitivity': sensitivity, 'specificity': specificity}


def train_epoch(model, loader, optimizer, device, lambda_constraint=1.0, pos_weight_abn=None, pos_weight_can=None):
    model.train()
    total_loss = 0
    criterion_abn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_abn)
    criterion_can = nn.BCEWithLogitsLoss(pos_weight=pos_weight_can)

    for embeddings, na_labels, cc_labels in tqdm(loader, desc='Train', leave=False):
        embeddings = embeddings.to(device)
        na_labels = na_labels.to(device).float()
        cc_labels = cc_labels.to(device).float()
        optimizer.zero_grad()
        logit_abn, logit_can = model(embeddings)
        loss_abn = criterion_abn(logit_abn, na_labels)
        loss_can = criterion_can(logit_can, cc_labels)
        p_abn = torch.sigmoid(logit_abn)
        p_can = torch.sigmoid(logit_can)
        constraint = F.relu(p_can - p_abn)
        loss_constraint = (constraint ** 2).mean()
        loss = loss_abn + loss_can + lambda_constraint * loss_constraint
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logit_abn, all_logit_can = [], []
    all_na_labels, all_cc_labels = [], []

    for embeddings, na_labels, cc_labels in tqdm(loader, desc='Eval', leave=False):
        embeddings = embeddings.to(device)
        logit_abn, logit_can = model(embeddings)
        all_logit_abn.append(logit_abn.cpu().numpy())
        all_logit_can.append(logit_can.cpu().numpy())
        all_na_labels.append(na_labels.numpy())
        all_cc_labels.append(cc_labels.numpy())

    logit_abn = np.concatenate(all_logit_abn)
    logit_can = np.concatenate(all_logit_can)
    na_labels = np.concatenate(all_na_labels)
    cc_labels = np.concatenate(all_cc_labels)

    prob_abn = 1 / (1 + np.exp(-logit_abn))
    prob_can = 1 / (1 + np.exp(-logit_can))
    pred_abn = (prob_abn > 0.5).astype(int)
    pred_can = (prob_can > 0.5).astype(int)

    metrics_abn = compute_metrics(na_labels, pred_abn, prob_abn)
    metrics_can = compute_metrics(cc_labels, pred_can, prob_can)
    violation_rate = (prob_can > prob_abn).mean()
    mean_bacc = 0.5 * (metrics_abn['bacc'] + metrics_can['bacc'])

    return {'abnormal': metrics_abn, 'cancer': metrics_can,
            'mean_bacc': mean_bacc, 'constraint_violation_rate': violation_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True, help='Directory containing Dataset A .pt feature files')
    parser.add_argument('--log_root_dir', type=str, default='./Logs_SlideCheck')
    parser.add_argument('--exp_name', type=str, default='exp_input_concat_v2_zscore')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--in_dim', type=int, default=2560)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--lambda_constraint', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = os.path.join(args.log_root_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config = vars(args).copy()
    config['experiment_description'] = 'Input Concat V2: raw + Z-score + LN(no affine), 3层h=1024'
    config['input'] = 'concat(raw, zscore) = 5120d -> LayerNorm(affine=False)'
    config['fix'] = 'V1用L2Norm导致尺度不平衡; V2改用Z-score + LN(no affine)平衡尺度'
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"实验: {args.exp_name}")
    print(f"输入: raw(2560) + Z-score(2560) = 5120d -> LN(no affine)")
    print(f"架构: 3层 h={args.hidden_dim}")
    print(f"修正: Z-score替代L2Norm + LN(no affine)平衡尺度")
    print(f"日志目录: {exp_dir}")
    print(f"{'='*60}")

    embeddings, na_labels, cc_labels = load_merged_data(args.feat_dir)
    dataset = SlideCheckDataset(embeddings, na_labels, cc_labels)

    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                                generator=torch.Generator().manual_seed(args.seed))

    print(f"\n数据划分: 训练集 {len(train_dataset)} | 测试集 {len(test_dataset)}")

    train_indices = train_dataset.indices
    train_na_labels = na_labels[train_indices]
    train_cc_labels = cc_labels[train_indices]
    pos_weight_abn = torch.tensor([(train_na_labels == 0).sum() / (train_na_labels == 1).sum()]).to(args.device)
    pos_weight_can = torch.tensor([(train_cc_labels == 0).sum() / (train_cc_labels == 1).sum()]).to(args.device)
    print(f"  Abnormal pos_weight: {pos_weight_abn.item():.4f}")
    print(f"  Cancer pos_weight: {pos_weight_can.item():.4f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SlideCheckMLP_InputConcatV2(in_dim=args.in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(args.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_bacc = 0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}\n开始训练\n{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, args.device,
                                 lambda_constraint=args.lambda_constraint,
                                 pos_weight_abn=pos_weight_abn, pos_weight_can=pos_weight_can)
        metrics = evaluate(model, test_loader, args.device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Mean BACC: {metrics['mean_bacc']:.4f}")
        print(f"    Abnormal - ACC: {metrics['abnormal']['acc']:.4f}, BACC: {metrics['abnormal']['bacc']:.4f}, AUC: {metrics['abnormal']['auc']:.4f}")
        print(f"    Cancer   - ACC: {metrics['cancer']['acc']:.4f}, BACC: {metrics['cancer']['bacc']:.4f}, AUC: {metrics['cancer']['auc']:.4f}")
        print(f"    Constraint Violation: {metrics['constraint_violation_rate']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if metrics['mean_bacc'] > best_bacc:
            best_bacc = metrics['mean_bacc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_bacc': best_bacc, 'metrics': metrics},
                       os.path.join(exp_dir, 'best_model.pt'))
            with open(os.path.join(exp_dir, 'best_metrics.json'), 'w') as f:
                json.dump({'epoch': epoch, 'mean_bacc': float(metrics['mean_bacc']),
                           'abnormal': {k: float(v) for k, v in metrics['abnormal'].items()},
                           'cancer': {k: float(v) for k, v in metrics['cancer'].items()},
                           'constraint_violation_rate': float(metrics['constraint_violation_rate'])}, f, indent=2)
            print(f"  -> New best (BACC={best_bacc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")
        print()

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            final_metrics = evaluate(model, test_loader, args.device)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'final_bacc': final_metrics['mean_bacc'], 'metrics': final_metrics},
                       os.path.join(exp_dir, 'final_model.pt'))
            with open(os.path.join(exp_dir, 'final_metrics.json'), 'w') as f:
                json.dump({'epoch': epoch, 'mean_bacc': float(final_metrics['mean_bacc']),
                           'abnormal': {k: float(v) for k, v in final_metrics['abnormal'].items()},
                           'cancer': {k: float(v) for k, v in final_metrics['cancer'].items()},
                           'constraint_violation_rate': float(final_metrics['constraint_violation_rate'])}, f, indent=2)
            break

    print(f"\n{'='*60}")
    print(f"训练完成 | 最佳Epoch: {best_epoch} | 最佳Mean BACC: {best_bacc:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
