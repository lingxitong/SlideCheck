"""
Model definitions
"""

from .slidecheck import (
    SlideCheckOutput,
    SlideCheckMLP,
    SlideCheckConcatV2,
    build_slidecheck_model
)
from .mil import (
    MILOutput,
    GatedABMIL_2560,
    build_gated_abmil_2560,
    ABMIL,
    GatedABMIL,
    DualHeadABMIL,
    DualHeadGatedABMIL
)

__all__ = [
    'SlideCheckOutput',
    'SlideCheckMLP',
    'SlideCheckConcatV2',
    'build_slidecheck_model',
    'MILOutput',
    'GatedABMIL_2560',
    'build_gated_abmil_2560',
    'ABMIL',
    'GatedABMIL',
    'DualHeadABMIL',
    'DualHeadGatedABMIL',
]
