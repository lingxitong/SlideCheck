"""
Foundation Model adapters for SlideCheck

Supports multiple foundation models with different feature dimensions:
- Virchow2: 2560d
- UNI: 1024d (planned)
- GigaPath: 1536d (planned)
"""

from .base import (
    FoundationModelConfig,
    FoundationModelAdapter,
    get_foundation_model,
    FM_REGISTRY
)
from .virchow2 import Virchow2Adapter

__all__ = [
    'FoundationModelConfig',
    'FoundationModelAdapter',
    'get_foundation_model',
    'FM_REGISTRY',
    'Virchow2Adapter',
]
