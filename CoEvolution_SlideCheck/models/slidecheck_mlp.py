"""
SlideCheck MLP (V1/V2 architecture) — standalone copy

Architecture matches SlideCheck_Model.py DualHeadMLP_v1:
  LayerNorm(2560) → Linear(2560, 768) → GELU → Dropout
  → Linear(768, 768) → GELU → Dropout
  → head_abn: Linear(768, 1)
  → head_can: Linear(768, 1)

Checkpoint 格式 (best_model.pt):
  {
      'epoch': int,
      'model_state_dict': OrderedDict,  # backbone.0/1/4 + head_abn + head_can
      'optimizer_state_dict': ...,
      'best_bacc': float,
      'metrics': dict,
  }
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SlideCheckOutput:
    logit_abn: torch.Tensor  # [B]
    logit_can: torch.Tensor  # [B]


class SlideCheckMLP(nn.Module):
    """
    Same architecture as DualHeadMLP_v1 (tag='mlp_v1').

    state_dict keys:
        backbone.0.weight / bias  → LayerNorm(in_dim)
        backbone.1.weight / bias  → Linear(in_dim, hidden_dim)
        backbone.4.weight / bias  → Linear(hidden_dim, hidden_dim)
        head_abn.weight / bias    → Linear(hidden_dim, 1)
        head_can.weight / bias    → Linear(hidden_dim, 1)
    """

    def __init__(self, in_dim: int = 2560, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),               # 0
            nn.Linear(in_dim, hidden_dim),      # 1
            nn.GELU(),                          # 2
            nn.Dropout(dropout),                # 3
            nn.Linear(hidden_dim, hidden_dim),  # 4
            nn.GELU(),                          # 5
            nn.Dropout(dropout),                # 6
        )
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> SlideCheckOutput:
        """
        Args:
            x: [B, 2560] 或 [N, 2560]
        Returns:
            SlideCheckOutput(logit_abn=[B], logit_can=[B])
        """
        h = self.backbone(x)
        logit_abn = self.head_abn(h).squeeze(-1)
        logit_can = self.head_can(h).squeeze(-1)
        return SlideCheckOutput(logit_abn=logit_abn, logit_can=logit_can)


def build_slidecheck_mlp(
    in_dim: int = 2560,
    hidden_dim: int = 768,
    dropout: float = 0.1,
) -> SlideCheckMLP:
    return SlideCheckMLP(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)


def load_slidecheck_from_ckpt(
    ckpt_path: str,
    in_dim: int = 2560,
    hidden_dim: int = 768,
    dropout: float = 0.1,
    device: str = 'cpu',
) -> SlideCheckMLP:
    """
    从 Phase 1 checkpoint 加载 SlideCheckMLP。

    兼容两种 checkpoint 格式:
      1. {'model_state_dict': ...}         (归一化实验的格式)
      2. {'slidecheck_state_dict': ...}    (Phase 2/3 的格式)
    """
    model = build_slidecheck_mlp(in_dim, hidden_dim, dropout)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'slidecheck_state_dict' in ckpt:
        state_dict = ckpt['slidecheck_state_dict']
    else:
        # 假设整个 ckpt 就是 state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict)
    return model.to(device)
