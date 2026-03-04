"""
ABMIL_2560: 标准 Attention-Based MIL，直接以 Virchow2 2560d 特征为输入

与旧版 MIL_Z 的区别:
  - 输入: 2560d (Virchow2 原始特征), 不是 z (256d/512d)
  - 不依赖 SlideCheck 的前向传播
  - SlideCheck 和 MIL 是完全独立的两个模型

架构:
  features [N, 2560]
    → feature_transform: Linear(2560, proj_dim) → ReLU → Dropout
    → attention: Linear(proj_dim, attn_hidden) → Tanh → Linear(attn_hidden, 1) → Softmax
    → weighted sum → bag_feature [1, proj_dim]
    → classifier: Linear(proj_dim, num_classes)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MILOutput:
    logits: torch.Tensor               # [1, num_classes]
    attention: torch.Tensor            # [N, 1] 未归一化
    attention_normalized: torch.Tensor # [N, 1] softmax归一化
    bag_feature: Optional[torch.Tensor] = None  # [1, proj_dim]


class ABMIL_2560(nn.Module):
    """标准 Attention-Based MIL, 2560d 输入"""

    def __init__(
        self,
        in_dim: int = 2560,
        proj_dim: int = 512,
        attn_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim

        # 降维投影 (2560 → proj_dim)
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention 机制
        self.attention = nn.Sequential(
            nn.Linear(proj_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # 分类器
        self.classifier = nn.Linear(proj_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        return_bag_feature: bool = False,
    ) -> MILOutput:
        """
        Args:
            features: [N, 2560] 或 [1, N, 2560]
            return_bag_feature: 是否返回 bag 级特征

        Returns:
            MILOutput
        """
        if features.dim() == 3:
            features = features.squeeze(0)

        # 降维
        h = self.feature_transform(features)  # [N, proj_dim]

        # Attention scores
        A = self.attention(h)                  # [N, 1]
        A_normalized = F.softmax(A, dim=0)     # [N, 1]

        # 加权聚合
        bag_feature = (A_normalized * h).sum(dim=0, keepdim=True)  # [1, proj_dim]

        # 分类
        logits = self.classifier(bag_feature)  # [1, num_classes]

        return MILOutput(
            logits=logits,
            attention=A,
            attention_normalized=A_normalized,
            bag_feature=bag_feature if return_bag_feature else None,
        )

    def get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """仅获取归一化注意力权重 [N, 1]"""
        return self.forward(features).attention_normalized


class GuidedABMIL_2560(ABMIL_2560):
    """
    带 Guided Loss 的 ABMIL_2560

    Phase 3 Step 3.3 中使用:
    L = L_CE^Bag + λ_guide · KL(Attention ∥ SlideCheck_mask_norm)
    """

    def compute_guided_loss(
        self,
        features: torch.Tensor,
        bag_label: torch.Tensor,
        mask: torch.Tensor,
        lambda_guide: float = 1.0,
    ) -> dict:
        """
        Args:
            features: [N, 2560]
            bag_label: [1] bag 标签
            mask: [N] SlideCheck 生成的 attention mask (cancer_prob > threshold)
            lambda_guide: guided loss 权重

        Returns:
            dict: {loss, loss_ce, loss_guide, attention}
        """
        out = self.forward(features)

        # CE Loss
        loss_ce = F.cross_entropy(out.logits, bag_label)

        # Guided Loss: KL(Attention ∥ Mask_norm)
        mask = mask.view(-1)
        if mask.sum() > 0:
            mask_norm = mask / (mask.sum() + 1e-8)
        else:
            mask_norm = torch.ones_like(mask) / len(mask)

        attn = out.attention_normalized.squeeze(-1)  # [N]
        loss_guide = F.kl_div(
            torch.log(attn + 1e-8),
            mask_norm,
            reduction='sum',
        )

        loss = loss_ce + lambda_guide * loss_guide

        return {
            'loss': loss,
            'loss_ce': loss_ce,
            'loss_guide': loss_guide,
            'attention': out.attention_normalized,
        }


def build_abmil_2560(
    in_dim: int = 2560,
    proj_dim: int = 512,
    attn_hidden: int = 256,
    num_classes: int = 2,
    dropout: float = 0.1,
    use_guided: bool = False,
) -> ABMIL_2560:
    cls = GuidedABMIL_2560 if use_guided else ABMIL_2560
    return cls(
        in_dim=in_dim,
        proj_dim=proj_dim,
        attn_hidden=attn_hidden,
        num_classes=num_classes,
        dropout=dropout,
    )
