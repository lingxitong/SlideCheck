"""
SlideCheck model definitions - unified version with multi-FM support

Supported architectures:
- baseline: LayerNorm + 2-layer MLP (h=768)
- concatv2: raw+zscore concatenation (5120d) + 3-layer MLP (h=1024)

Key design:
- Automatically adapts to FM feature dimensions
- Unified output format via SlideCheckOutput
- Factory function for easy model creation
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


@dataclass
class SlideCheckOutput:
    """Unified output format for SlideCheck models"""
    logit_abn: torch.Tensor  # Abnormal logits
    logit_can: torch.Tensor  # Cancer logits

    def to_probs(self) -> Dict[str, torch.Tensor]:
        """Convert logits to probabilities"""
        return {
            'prob_abn': torch.sigmoid(self.logit_abn),
            'prob_can': torch.sigmoid(self.logit_can)
        }


class SlideCheckMLP(nn.Module):
    """
    Baseline SlideCheck architecture

    Architecture:
        LayerNorm -> Linear(in_dim, hidden_dim) -> GELU -> Dropout
                  -> Linear(hidden_dim, hidden_dim) -> GELU -> Dropout
                  -> Two heads (abnormal, cancer)

    Args:
        in_dim: Input feature dimension (from foundation model)
        hidden_dim: Hidden layer dimension (default: 768)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.arch = 'baseline'
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> SlideCheckOutput:
        """
        Forward pass

        Args:
            x: Input features [batch_size, in_dim]

        Returns:
            SlideCheckOutput with logit_abn and logit_can
        """
        feat = self.backbone(x)
        return SlideCheckOutput(
            logit_abn=self.head_abn(feat).squeeze(-1),
            logit_can=self.head_can(feat).squeeze(-1)
        )


class SlideCheckConcatV2(nn.Module):
    """
    ConcatV2 architecture: raw + zscore concatenation

    Architecture:
        Concatenate [raw_features, zscore_normalized_features]
        -> LayerNorm (no affine) -> 3-layer MLP -> Two heads

    Args:
        in_dim: Input feature dimension (from foundation model)
        hidden_dim: Hidden layer dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.arch = 'concatv2'
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        combined_dim = in_dim * 2  # raw + zscore

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

    def forward(self, x: torch.Tensor) -> SlideCheckOutput:
        """
        Forward pass with zscore normalization

        Args:
            x: Input features [batch_size, in_dim]

        Returns:
            SlideCheckOutput with logit_abn and logit_can
        """
        # Z-score normalization
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        x_zscore = (x - mean) / std

        # Concatenate raw and normalized
        x_combined = torch.cat([x, x_zscore], dim=-1)
        x_combined = self.input_norm(x_combined)

        feat = self.backbone(x_combined)
        return SlideCheckOutput(
            logit_abn=self.head_abn(feat).squeeze(-1),
            logit_can=self.head_can(feat).squeeze(-1)
        )


def build_slidecheck_model(
    arch: str = 'baseline',
    foundation_model: str = 'virchow2',
    **kwargs
) -> nn.Module:
    """
    Factory function to build SlideCheck models

    Automatically adapts to foundation model feature dimensions.

    Args:
        arch: Architecture type ('baseline' or 'concatv2')
        foundation_model: Foundation model name ('virchow2', 'uni', 'gigapath')
        **kwargs: Additional arguments (hidden_dim, dropout, etc.)

    Returns:
        SlideCheck model instance

    Example:
        >>> model = build_slidecheck_model('baseline', 'virchow2', hidden_dim=768)
        >>> model = build_slidecheck_model('concatv2', 'uni', hidden_dim=1024)
    """
    from ..foundation_models import get_foundation_model

    # Get feature dimension from foundation model
    fm_adapter = get_foundation_model(foundation_model)
    fm_config = fm_adapter.get_config()
    in_dim = fm_config.feature_dim

    print(f"Building SlideCheck model:")
    print(f"  Foundation Model: {foundation_model}")
    print(f"  Feature Dim: {in_dim}")
    print(f"  Architecture: {arch}")

    if arch == 'baseline':
        return SlideCheckMLP(in_dim=in_dim, **kwargs)
    elif arch == 'concatv2':
        return SlideCheckConcatV2(in_dim=in_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. "
            f"Available: ['baseline', 'concatv2']"
        )
