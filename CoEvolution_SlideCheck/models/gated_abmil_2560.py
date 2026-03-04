"""
Gated Attention MIL (CLAM-style), 2560d Virchow2 特征输入

与 bowel ABMIL 相同的门控注意力架构:
  attention_a: Linear(in_dim, attn_hidden) → Tanh
  attention_b: Linear(in_dim, attn_hidden) → Sigmoid  (gate)
  attention_c: Linear(attn_hidden, 1)
  → softmax → weighted sum → classifier

相比简单 ABMIL_2560 的优势:
  - 门控机制能更精细地筛选 patch (CLAM, ICML 2021)
  - Tanh 分支学 "哪些 patch 重要", Sigmoid 分支学 "哪些 patch 该被抑制"
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


class GatedABMIL_2560(nn.Module):
    """Gated Attention MIL, 2560d 输入"""

    def __init__(
        self,
        in_dim: int = 2560,
        attn_hidden: int = 384,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.attn_hidden = attn_hidden

        # Gated Attention
        self.attention_a = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Tanh(),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Sigmoid(),
        )
        self.attention_c = nn.Linear(attn_hidden, 1)

        # Dropout on attention
        self.attn_dropout = nn.Dropout(dropout)

        # 分类器
        self.classifier = nn.Linear(in_dim, num_classes)

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
        Returns:
            MILOutput
        """
        if features.dim() == 3:
            features = features.squeeze(0)

        # Gated Attention
        a = self.attention_a(features)          # [N, attn_hidden]
        b = self.attention_b(features)          # [N, attn_hidden]
        A = self.attention_c(self.attn_dropout(a * b))  # [N, 1]
        A_normalized = F.softmax(A, dim=0)      # [N, 1]

        # 加权聚合
        bag_feature = (A_normalized * features).sum(dim=0, keepdim=True)  # [1, in_dim]

        # 分类
        logits = self.classifier(bag_feature)   # [1, num_classes]

        return MILOutput(
            logits=logits,
            attention=A,
            attention_normalized=A_normalized,
            bag_feature=bag_feature if return_bag_feature else None,
        )

    def get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """仅获取归一化注意力权重 [N, 1]"""
        return self.forward(features).attention_normalized


def build_gated_abmil_2560(
    in_dim: int = 2560,
    attn_hidden: int = 384,
    num_classes: int = 2,
    dropout: float = 0.25,
) -> GatedABMIL_2560:
    return GatedABMIL_2560(
        in_dim=in_dim,
        attn_hidden=attn_hidden,
        num_classes=num_classes,
        dropout=dropout,
    )
