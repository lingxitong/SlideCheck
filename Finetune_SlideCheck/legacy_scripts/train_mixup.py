# -*- coding: utf-8 -*-
"""
SlideCheck Training Script - Baseline + Mixup
在Baseline架构(LayerNorm + 2层h=768)基础上，仅在训练时添加Mixup数据增强
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


class SlideCheckMLP(nn.Module):
    """与Baseline完全一致的模型架构"""
    def __init__(self, in_dim=2560, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        logit_abn = self.head_abn(feat).squeeze(-1)
        logit_can = self.head_can(feat).squeeze(-1)
        return logit_abn, logit_can


def load_merged_data(feat_dir):
    """加载8个数据集并合并"""
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

    return {
        'acc': acc,
        'bacc': bacc,
        'auc': auc,
        'auprc': auprc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def train_epoch_mixup(model, loader, optimizer, device, mixup_alpha=0.4,
                       lambda_constraint=1.0, pos_weight_abn=None, pos_weight_can=None):
    """训练一个epoch，加入Mixup数据增强"""
    model.train()
    total_loss = 0

    criterion_abn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_abn)
    criterion_can = nn.BCEWithLogitsLoss(pos_weight=pos_weight_can)

    for embeddings, na_labels, cc_labels in tqdm(loader, desc='Train', leave=False):
        embeddings = embeddings.to(device)
        na_labels = na_labels.to(device).float()
        cc_labels = cc_labels.to(device).float()

        batch_size = embeddings.size(0)

        # ===== Mixup =====
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(lam, 1 - lam)  # 确保 λ >= 0.5，主样本主导

        index = torch.randperm(batch_size, device=device)

        mixed_emb = lam * embeddings + (1 - lam) * embeddings[index]
        mixed_na = lam * na_labels + (1 - lam) * na_labels[index]
        mixed_cc = lam * cc_labels + (1 - lam) * cc_labels[index]
        # ===== End Mixup =====

        optimizer.zero_grad()

        logit_abn, logit_can = model(mixed_emb)

        # BCE损失（soft labels）
        loss_abn = criterion_abn(logit_abn, mixed_na)
        loss_can = criterion_can(logit_can, mixed_cc)

        # 约束损失: p(cancer) <= p(abnormal)，用混合后的预测
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
    """验证集评估 - 不做Mixup"""
    model.eval()

    all_logit_abn = []
    all_logit_can = []
    all_na_labels = []
    all_cc_labels = []

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

    return {
        'abnormal': metrics_abn,
        'cancer': metrics_can,
        'mean_bacc': mean_bacc,
        'constraint_violation_rate': violation_rate
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True, help='Directory containing Dataset A .pt feature files')
    parser.add_argument('--log_root_dir', type=str, default='./Logs_SlideCheck')
    parser.add_argument('--exp_name', type=str, default='exp_mixup_alpha04')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--in_dim', type=int, default=2560)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--lambda_constraint', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建日志目录
    exp_dir = os.path.join(args.log_root_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"实验: {args.exp_name}")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"日志目录: {exp_dir}")
    print(f"{'='*60}")

    # 加载数据
    embeddings, na_labels, cc_labels = load_merged_data(args.feat_dir)

    # 创建数据集
    dataset = SlideCheckDataset(embeddings, na_labels, cc_labels)

    # 划分训练集和测试集
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                                generator=torch.Generator().manual_seed(args.seed))

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_dataset)} samples")
    print(f"  测试集: {len(test_dataset)} samples")

    # 计算类别权重
    train_indices = train_dataset.indices
    train_na_labels = na_labels[train_indices]
    train_cc_labels = cc_labels[train_indices]

    na_pos = (train_na_labels == 1).sum()
    na_neg = (train_na_labels == 0).sum()
    cc_pos = (train_cc_labels == 1).sum()
    cc_neg = (train_cc_labels == 0).sum()

    pos_weight_abn = torch.tensor([na_neg / na_pos]).to(args.device)
    pos_weight_can = torch.tensor([cc_neg / cc_pos]).to(args.device)

    print(f"\n类别权重:")
    print(f"  Abnormal pos_weight: {pos_weight_abn.item():.4f}")
    print(f"  Cancer pos_weight: {pos_weight_can.item():.4f}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型 - 与Baseline完全一致
    model = SlideCheckMLP(in_dim=args.in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(args.device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_bacc = 0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print("开始训练 (with Mixup)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # 训练 - 使用Mixup
        train_loss = train_epoch_mixup(model, train_loader, optimizer, args.device,
                                        mixup_alpha=args.mixup_alpha,
                                        lambda_constraint=args.lambda_constraint,
                                        pos_weight_abn=pos_weight_abn,
                                        pos_weight_can=pos_weight_can)

        # 评估 - 不做Mixup
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

            ckpt_path = os.path.join(exp_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bacc': best_bacc,
                'metrics': metrics
            }, ckpt_path)

            with open(os.path.join(exp_dir, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'mean_bacc': float(metrics['mean_bacc']),
                    'abnormal': {k: float(v) for k, v in metrics['abnormal'].items()},
                    'cancer': {k: float(v) for k, v in metrics['cancer'].items()},
                    'constraint_violation_rate': float(metrics['constraint_violation_rate'])
                }, f, indent=2)

            print(f"  -> New best (BACC={best_bacc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

        print()

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 保存最终模型
    final_path = os.path.join(exp_dir, 'final_model.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, final_path)

    with open(os.path.join(exp_dir, 'final_metrics.json'), 'w') as f:
        json.dump({
            'epoch': epoch,
            'mean_bacc': float(metrics['mean_bacc']),
            'abnormal': {k: float(v) for k, v in metrics['abnormal'].items()},
            'cancer': {k: float(v) for k, v in metrics['cancer'].items()},
            'constraint_violation_rate': float(metrics['constraint_violation_rate'])
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"训练完成")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳Mean BACC: {best_bacc:.4f}")
    print(f"  Best model: {os.path.join(exp_dir, 'best_model.pt')}")
    print(f"  Final model: {final_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
