# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, Any
from abc import ABC, abstractmethod
import inspect

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Unified model output: logits from two heads (shape [B])."""
    logit_abn: torch.Tensor
    logit_can: torch.Tensor


class BaseDualHeadModel(nn.Module, ABC):
    """
    Abstract interface: all models must implement forward(x) -> ModelOutput
    and must have a class-level 'tag' field for selection.
    """
    tag: str = "base"

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput:
        raise NotImplementedError


# -------------------------
# Registry (tag -> class)
# -------------------------
_MODEL_REGISTRY: Dict[str, Type[BaseDualHeadModel]] = {}


def register_model(tag: str):
    """Decorator: register a model class under the given tag."""
    def _wrap(cls: Type[BaseDualHeadModel]):
        if tag in _MODEL_REGISTRY:
            raise KeyError(f"Duplicated model tag: {tag}")
        cls.tag = tag
        _MODEL_REGISTRY[tag] = cls
        return cls
    return _wrap


def list_models() -> Dict[str, Type[BaseDualHeadModel]]:
    """Return a copy of the current model registry."""
    return dict(_MODEL_REGISTRY)


def _filter_kwargs_for_class(cls: Type[BaseDualHeadModel], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs supported by cls.__init__. Others are ignored.
    If __init__ accepts **kwargs, no filtering is needed (everything is allowed).
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # If **kwargs exists, return kwargs as-is (the class will handle it).
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return kwargs

    allowed = {name for name in params.keys() if name not in ("self", "in_dim")}
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_model(tag: str, in_dim: int, **kwargs) -> BaseDualHeadModel:
    """
    Build a model by tag, automatically ignoring unsupported __init__ kwargs.
    """
    if tag not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model tag={tag}. Available: {list(_MODEL_REGISTRY.keys())}")
    cls = _MODEL_REGISTRY[tag]

    filtered = _filter_kwargs_for_class(cls, kwargs)
    ignored = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if ignored:
        print(f"[WARN] Ignored kwargs for model '{tag}': {ignored}")

    return cls(in_dim=in_dim, **filtered)


# -------------------------
# Model variants
# -------------------------
@register_model("linear")
class LinearDualHead(BaseDualHeadModel):
    """Simplest variant: two linear heads directly on normalized features."""
    def __init__(self, in_dim: int):
        super().__init__(in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.head_abn = nn.Linear(in_dim, 1)
        self.head_can = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x = self.norm(x)
        logit_abn = self.head_abn(x).squeeze(-1)
        logit_can = self.head_can(x).squeeze(-1)
        return ModelOutput(logit_abn=logit_abn, logit_can=logit_can)


@register_model("mlp_v1")
class DualHeadMLP_v1(BaseDualHeadModel):
    """MLP backbone + two heads (recommended default)."""
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__(in_dim)
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

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.backbone(x)
        logit_abn = self.head_abn(h).squeeze(-1)
        logit_can = self.head_can(h).squeeze(-1)
        return ModelOutput(logit_abn=logit_abn, logit_can=logit_can)


@register_model("mlp_deep")
class DualHeadMLP_Deep(BaseDualHeadModel):
    """A deeper MLP variant with configurable depth."""
    def __init__(self, in_dim: int, hidden_dim: int = 768, dropout: float = 0.2, layers: int = 4):
        super().__init__(in_dim)
        blocks = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*blocks)
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.backbone(x)
        return ModelOutput(
            logit_abn=self.head_abn(h).squeeze(-1),
            logit_can=self.head_can(h).squeeze(-1),
        )

@register_model("mobilenetv3")
class MobileNetV3DualHead(BaseDualHeadModel):
    """Use timm MobileNetV3 as backbone + mlp_v1 structure as classification head"""
    def __init__(self, in_dim: int = None, hidden_dim: int = 512, dropout: float = 0.1, 
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            in_dim: For image input, this parameter will be ignored (kept for interface compatibility)
            hidden_dim: Hidden dimension of mlp_v1 classification head
            dropout: Dropout rate
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone (only train classification head)
        """
        # Temporarily set in_dim, will be updated based on actual backbone output dimension
        super().__init__(in_dim=1280)
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm library is required: pip install timm")
        
        # Create MobileNetV3 backbone, num_classes=0 means output features only
        self.backbone = timm.create_model(
            model_name='mobilenetv3_large_100',
            pretrained=pretrained,
            num_classes=0  # Output embedding instead of classification results
        )
        
        # Get actual output dimension of backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            backbone_dim = dummy_output.shape[1]
        
        self.in_dim = backbone_dim
        
        # If freeze backbone, only train classification head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Use mlp_v1 structure as classification head: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Two classification heads: from mlp_head output to final classification results
        self.head_abn = nn.Linear(hidden_dim, 1)
        self.head_can = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Args:
            x: Input image tensor, shape [B, 3, 224, 224]
        Returns:
            ModelOutput: Contains logit_abn and logit_can
        """
        # Extract features
        features = self.backbone(x)  # [B, backbone_dim]
        
        # Through mlp_v1 structure classification head
        h = self.mlp_head(features)  # [B, hidden_dim]
        
        # Two classification heads
        logit_abn = self.head_abn(h).squeeze(-1)  # [B]
        logit_can = self.head_can(h).squeeze(-1)  # [B]
        
        return ModelOutput(logit_abn=logit_abn, logit_can=logit_can)
