"""
Bag-level dataset for MIL training
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path


class BagDataset(Dataset):
    """
    Bag-level dataset for Multiple Instance Learning

    Each bag contains multiple instances (patches) from a single WSI.

    Args:
        bag_features: List of bag features, each [num_instances, feature_dim]
        bag_labels: Bag-level labels [num_bags]
        bag_ids: Optional bag identifiers
        transform: Optional transform to apply

    Example:
        >>> bag_features = [np.random.randn(100, 2560), np.random.randn(150, 2560)]
        >>> bag_labels = np.array([0, 1])
        >>> dataset = BagDataset(bag_features, bag_labels)
        >>> features, label = dataset[0]
    """

    def __init__(
        self,
        bag_features: List[np.ndarray],
        bag_labels: np.ndarray,
        bag_ids: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        assert len(bag_features) == len(bag_labels), \
            "Number of bags must match number of labels"

        self.bag_features = [torch.from_numpy(f).float() for f in bag_features]
        self.bag_labels = torch.from_numpy(bag_labels).long()
        self.bag_ids = bag_ids if bag_ids is not None else [f"bag_{i}" for i in range(len(bag_features))]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.bag_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single bag

        Returns:
            Tuple of (bag_features, bag_label)
        """
        features = self.bag_features[idx]
        label = self.bag_labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label

    def get_bag_id(self, idx: int) -> str:
        """Get bag identifier"""
        return self.bag_ids[idx]

    def get_bag_sizes(self) -> List[int]:
        """Get number of instances in each bag"""
        return [len(f) for f in self.bag_features]

    def get_class_distribution(self) -> dict:
        """
        Get class distribution statistics

        Returns:
            Dict with class counts
        """
        unique, counts = torch.unique(self.bag_labels, return_counts=True)
        return {
            'total_bags': len(self),
            'total_instances': sum(self.get_bag_sizes()),
            'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)},
            'avg_bag_size': np.mean(self.get_bag_sizes()),
            'min_bag_size': min(self.get_bag_sizes()),
            'max_bag_size': max(self.get_bag_sizes())
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        feature_dir: str,
        label_key: str = 'label',
        transform: Optional[callable] = None
    ):
        """
        Load dataset from manifest file

        Manifest format (CSV):
            bag_id,feature_path,label
            slide001,slide001.pt,0
            slide002,slide002.pt,1

        Args:
            manifest_path: Path to manifest CSV
            feature_dir: Directory containing feature files
            label_key: Column name for labels
            transform: Optional transform

        Returns:
            BagDataset instance
        """
        import pandas as pd

        df = pd.read_csv(manifest_path)
        feature_dir = Path(feature_dir)

        bag_features = []
        bag_labels = []
        bag_ids = []

        for _, row in df.iterrows():
            # Load features
            feature_path = feature_dir / row['feature_path']
            features = torch.load(feature_path, map_location='cpu', weights_only=False)

            # Handle different feature formats
            if isinstance(features, dict):
                if 'features' in features:
                    features = features['features']
                elif 'embeddings' in features:
                    features = features['embeddings']
                else:
                    raise ValueError(f"Unknown feature format in {feature_path}")

            if isinstance(features, torch.Tensor):
                features = features.numpy()

            bag_features.append(features)
            bag_labels.append(row[label_key])
            bag_ids.append(row['bag_id'])

        bag_labels = np.array(bag_labels)

        return cls(bag_features, bag_labels, bag_ids, transform)


class BagDatasetWithAbnormalCancer(BagDataset):
    """
    Bag-level dataset with both abnormal and cancer labels

    Args:
        bag_features: List of bag features
        abnormal_labels: Bag-level abnormal labels [num_bags]
        cancer_labels: Bag-level cancer labels [num_bags]
        bag_ids: Optional bag identifiers
        transform: Optional transform
    """

    def __init__(
        self,
        bag_features: List[np.ndarray],
        abnormal_labels: np.ndarray,
        cancer_labels: np.ndarray,
        bag_ids: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        super().__init__(bag_features, abnormal_labels, bag_ids, transform)
        self.cancer_labels = torch.from_numpy(cancer_labels).long()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single bag with both labels

        Returns:
            Tuple of (bag_features, abnormal_label, cancer_label)
        """
        features = self.bag_features[idx]
        abn_label = self.bag_labels[idx]
        can_label = self.cancer_labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, abn_label, can_label

    def get_class_distribution(self) -> dict:
        """Get class distribution for both tasks"""
        abn_unique, abn_counts = torch.unique(self.bag_labels, return_counts=True)
        can_unique, can_counts = torch.unique(self.cancer_labels, return_counts=True)

        return {
            'total_bags': len(self),
            'total_instances': sum(self.get_bag_sizes()),
            'abnormal_distribution': {int(u): int(c) for u, c in zip(abn_unique, abn_counts)},
            'cancer_distribution': {int(u): int(c) for u, c in zip(can_unique, can_counts)},
            'avg_bag_size': np.mean(self.get_bag_sizes()),
            'min_bag_size': min(self.get_bag_sizes()),
            'max_bag_size': max(self.get_bag_sizes())
        }
