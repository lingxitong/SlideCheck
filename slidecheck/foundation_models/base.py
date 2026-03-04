"""
Foundation Model abstract base class

Key design:
1. Each FM has different feature dimensions (Virchow2=2560, UNI=1024, GigaPath=1536)
2. SlideCheck model's in_dim must match FM's output dimension
3. Checkpoints must record the FM used to avoid dimension mismatches
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class FoundationModelConfig:
    """Foundation Model configuration"""
    name: str              # 'virchow2', 'uni', 'gigapath'
    feature_dim: int       # Feature dimension
    patch_size: int        # Patch size
    normalize: bool        # Whether normalization is needed


class FoundationModelAdapter(ABC):
    """Foundation Model adapter base class"""

    @abstractmethod
    def get_config(self) -> FoundationModelConfig:
        """Return FM configuration"""
        pass

    @abstractmethod
    def extract_features(self, images):
        """Extract features from images"""
        pass

    @abstractmethod
    def load_pretrained(self, ckpt_path: Optional[str] = None):
        """Load pretrained weights"""
        pass


# Registry for all foundation models
FM_REGISTRY = {}


def register_foundation_model(name: str):
    """Decorator to register a foundation model"""
    def decorator(cls):
        FM_REGISTRY[name] = cls
        return cls
    return decorator


def get_foundation_model(name: str) -> FoundationModelAdapter:
    """Get FM adapter by name"""
    if name not in FM_REGISTRY:
        raise ValueError(
            f"Unknown foundation model: {name}. "
            f"Available: {list(FM_REGISTRY.keys())}"
        )
    return FM_REGISTRY[name]()
