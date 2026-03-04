"""
Dataset modules for SlideCheck

Provides:
- PatchDataset: Patch-level dataset for training
- BagDataset: Bag-level dataset for MIL
- H5BagDataset: Lazy-loading bag dataset from h5 files (requires h5py)
- Transforms: Data augmentation (Mixup, FeaturePerturbation)
"""

from .patch_dataset import PatchDataset
from .bag_dataset import BagDataset, BagDatasetWithAbnormalCancer
from .transforms import Mixup, FeaturePerturbation, Compose


def __getattr__(name):
    """Lazy import for h5py-dependent modules"""
    if name in ('H5BagDataset', 'bag_collate_fn', 'create_bag_dataloader'):
        from .h5_bag_dataset import H5BagDataset, bag_collate_fn, create_bag_dataloader
        globals()['H5BagDataset'] = H5BagDataset
        globals()['bag_collate_fn'] = bag_collate_fn
        globals()['create_bag_dataloader'] = create_bag_dataloader
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'PatchDataset',
    'BagDataset',
    'BagDatasetWithAbnormalCancer',
    'H5BagDataset',
    'bag_collate_fn',
    'create_bag_dataloader',
    'Mixup',
    'FeaturePerturbation',
    'Compose',
]
