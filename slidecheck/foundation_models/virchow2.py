"""
Virchow2 Foundation Model adapter
"""

from typing import Optional
from .base import (
    FoundationModelAdapter,
    FoundationModelConfig,
    register_foundation_model
)


@register_foundation_model('virchow2')
class Virchow2Adapter(FoundationModelAdapter):
    """Virchow2 adapter - 2560d features"""

    def get_config(self) -> FoundationModelConfig:
        return FoundationModelConfig(
            name='virchow2',
            feature_dim=2560,
            patch_size=224,
            normalize=True
        )

    def extract_features(self, images):
        """Extract 2560d features from images"""
        raise NotImplementedError(
            "Feature extraction not implemented. "
            "Use Dataset_Preprocess scripts for feature extraction."
        )

    def load_pretrained(self, ckpt_path: Optional[str] = None):
        """Load pretrained Virchow2 weights"""
        raise NotImplementedError(
            "Pretrained loading not implemented. "
            "Use Dataset_Preprocess scripts for feature extraction."
        )
