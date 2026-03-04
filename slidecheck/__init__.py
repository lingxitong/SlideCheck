"""
SlideCheck: A library for weakly-supervised pathology image analysis

Core modules:
- models: SlideCheck classifier and MIL models
- datasets: Data loading and augmentation
- training: Training frameworks
- inference: Unified prediction interface
- mining: Pseudo-label generation and data mining
- foundation_models: Foundation model adapters
"""

__version__ = "1.0.0"

# Lazy imports to avoid h5py dependency issues
def __getattr__(name):
    if name == 'SlideCheckPredictor':
        from .inference import SlideCheckPredictor
        return SlideCheckPredictor
    elif name == 'build_slidecheck_model':
        from .models import build_slidecheck_model
        return build_slidecheck_model
    elif name == 'SlideCheckMLP':
        from .models import SlideCheckMLP
        return SlideCheckMLP
    elif name == 'SlideCheckConcatV2':
        from .models import SlideCheckConcatV2
        return SlideCheckConcatV2
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'build_slidecheck_model',
    'SlideCheckMLP',
    'SlideCheckConcatV2',
    'SlideCheckPredictor',
]
