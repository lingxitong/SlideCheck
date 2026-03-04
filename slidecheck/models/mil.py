"""
Multiple Instance Learning (MIL) models for SlideCheck

Implements:
- GatedABMIL_2560: Gated attention MIL matching original CLAM-style architecture
  (used in co-evolution pipeline for mining)
- ABMIL: Simple attention-based MIL (legacy)
- GatedABMIL: Gated attention MIL with batched input (legacy)
- DualHeadABMIL / DualHeadGatedABMIL: Dual-head variants (legacy)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GatedABMIL_2560: Original architecture from Server 239
# ============================================================

@dataclass
class MILOutput:
    logits: torch.Tensor               # [1, num_classes]
    attention: torch.Tensor            # [N, 1] unnormalized
    attention_normalized: torch.Tensor # [N, 1] softmax-normalized
    bag_feature: Optional[torch.Tensor] = None  # [1, in_dim]


class GatedABMIL_2560(nn.Module):
    """
    Gated Attention MIL (CLAM-style), 2560d Virchow2 input

    Architecture:
      attention_a: Linear(in_dim, attn_hidden) -> Tanh
      attention_b: Linear(in_dim, attn_hidden) -> Sigmoid  (gate)
      attention_c: Linear(attn_hidden, 1)
      -> softmax -> weighted sum -> classifier

    Args:
        in_dim: Input feature dimension (default: 2560)
        attn_hidden: Attention hidden dimension (default: 384)
        num_classes: Number of output classes (default: 2)
        dropout: Dropout rate on gated product (default: 0.25)
    """

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

        # Dropout on gated product (NOT inside each branch)
        self.attn_dropout = nn.Dropout(dropout)

        # Single-layer classifier
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
            features: [N, 2560] or [1, N, 2560]
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

        # Weighted aggregation
        bag_feature = (A_normalized * features).sum(dim=0, keepdim=True)  # [1, in_dim]

        # Classification
        logits = self.classifier(bag_feature)   # [1, num_classes]

        return MILOutput(
            logits=logits,
            attention=A,
            attention_normalized=A_normalized,
            bag_feature=bag_feature if return_bag_feature else None,
        )

    def get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Get normalized attention weights [N, 1]"""
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


# ============================================================
# Legacy models (kept for backward compatibility)
# ============================================================

class ABMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning (legacy, batched input)
    """

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        n_classes: int = 2
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Compute attention weights
        attention_logits = self.attention(x)  # [B, N, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N, 1]

        # Aggregate features
        aggregated = torch.sum(attention_weights * x, dim=1)  # [B, in_dim]

        # Classify
        logits = self.classifier(aggregated)  # [B, n_classes]

        if return_attention:
            return logits, attention_weights.squeeze(-1)  # [B, n_classes], [B, N]
        else:
            return logits, None


class GatedABMIL(nn.Module):
    """
    Gated Attention-based MIL (legacy, batched input)
    """

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        n_classes: int = 2
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Gated attention network
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_w = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Gated attention
        A_V = self.attention_V(x)  # [B, N, hidden_dim]
        A_U = self.attention_U(x)  # [B, N, hidden_dim]
        A = self.attention_w(A_V * A_U)  # [B, N, 1]

        attention_weights = F.softmax(A, dim=1)  # [B, N, 1]

        # Aggregate features
        aggregated = torch.sum(attention_weights * x, dim=1)  # [B, in_dim]

        # Classify
        logits = self.classifier(aggregated)  # [B, n_classes]

        if return_attention:
            return logits, attention_weights.squeeze(-1)
        else:
            return logits, None


class DualHeadABMIL(nn.Module):
    """Dual-head ABMIL (legacy)"""

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mil_abn = ABMIL(hidden_dim, hidden_dim // 2, dropout, n_classes=1)
        self.mil_can = ABMIL(hidden_dim, hidden_dim // 2, dropout, n_classes=1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        features = self.feature_extractor(x)

        logit_abn, attn_abn = self.mil_abn(features, return_attention)
        logit_abn = logit_abn.squeeze(-1)

        logit_can, attn_can = self.mil_can(features, return_attention)
        logit_can = logit_can.squeeze(-1)

        return logit_abn, logit_can, attn_abn, attn_can


class DualHeadGatedABMIL(nn.Module):
    """Dual-head Gated ABMIL (legacy)"""

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mil_abn = GatedABMIL(hidden_dim, hidden_dim // 2, dropout, n_classes=1)
        self.mil_can = GatedABMIL(hidden_dim, hidden_dim // 2, dropout, n_classes=1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        features = self.feature_extractor(x)

        logit_abn, attn_abn = self.mil_abn(features, return_attention)
        logit_abn = logit_abn.squeeze(-1)

        logit_can, attn_can = self.mil_can(features, return_attention)
        logit_can = logit_can.squeeze(-1)

        return logit_abn, logit_can, attn_abn, attn_can
