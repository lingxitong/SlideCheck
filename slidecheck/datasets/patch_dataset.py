"""
Patch-level dataset for SlideCheck training
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple


class PatchDataset(Dataset):
    """
    Patch-level dataset for SlideCheck training

    Loads pre-extracted patch features with binary labels.

    Args:
        embeddings: Patch embeddings [N, feature_dim]
        abnormal_labels: Binary abnormal labels [N] (0=normal, 1=abnormal)
        cancer_labels: Binary cancer labels [N] (0=non-cancer, 1=cancer)
        transform: Optional transform to apply

    Example:
        >>> embeddings = np.random.randn(1000, 2560)
        >>> abnormal_labels = np.random.randint(0, 2, 1000)
        >>> cancer_labels = np.random.randint(0, 2, 1000)
        >>> dataset = PatchDataset(embeddings, abnormal_labels, cancer_labels)
        >>> emb, abn, can = dataset[0]
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        abnormal_labels: np.ndarray,
        cancer_labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        assert len(embeddings) == len(abnormal_labels) == len(cancer_labels), \
            "Embeddings and labels must have same length"

        self.embeddings = torch.from_numpy(embeddings).float()
        self.abnormal_labels = torch.from_numpy(abnormal_labels).long()
        self.cancer_labels = torch.from_numpy(cancer_labels).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
            Tuple of (embedding, abnormal_label, cancer_label)
        """
        emb = self.embeddings[idx]
        abn = self.abnormal_labels[idx]
        can = self.cancer_labels[idx]

        if self.transform:
            emb = self.transform(emb)

        return emb, abn, can

    @classmethod
    def from_pt_file(cls, pt_path: str, transform: Optional[callable] = None):
        """
        Load dataset from .pt file

        Expected format:
        {
            'embeddings': np.ndarray [N, feature_dim],
            'normal_abnormal_labels': np.ndarray [N],
            'cancer_noncancer_labels': np.ndarray [N],
            ...
        }

        Args:
            pt_path: Path to .pt file
            transform: Optional transform

        Returns:
            PatchDataset instance
        """
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        embeddings = data['embeddings']
        abnormal_labels = data['normal_abnormal_labels']
        cancer_labels = data['cancer_noncancer_labels']

        return cls(embeddings, abnormal_labels, cancer_labels, transform)

    @classmethod
    def from_multiple_files(cls, pt_paths: list, transform: Optional[callable] = None):
        """
        Load and merge multiple .pt files

        Args:
            pt_paths: List of paths to .pt files
            transform: Optional transform

        Returns:
            PatchDataset instance with merged data
        """
        all_embeddings = []
        all_abnormal_labels = []
        all_cancer_labels = []

        for pt_path in pt_paths:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            all_embeddings.append(data['embeddings'])
            all_abnormal_labels.append(data['normal_abnormal_labels'])
            all_cancer_labels.append(data['cancer_noncancer_labels'])

        embeddings = np.concatenate(all_embeddings, axis=0)
        abnormal_labels = np.concatenate(all_abnormal_labels, axis=0)
        cancer_labels = np.concatenate(all_cancer_labels, axis=0)

        return cls(embeddings, abnormal_labels, cancer_labels, transform)

    def get_class_distribution(self) -> dict:
        """
        Get class distribution statistics

        Returns:
            Dict with class counts
        """
        return {
            'total': len(self),
            'abnormal': {
                'normal': (self.abnormal_labels == 0).sum().item(),
                'abnormal': (self.abnormal_labels == 1).sum().item()
            },
            'cancer': {
                'non_cancer': (self.cancer_labels == 0).sum().item(),
                'cancer': (self.cancer_labels == 1).sum().item()
            }
        }

    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class weights for balanced training

        Returns:
            Tuple of (abnormal_weight, cancer_weight)
        """
        abn_pos = (self.abnormal_labels == 1).sum().item()
        abn_neg = (self.abnormal_labels == 0).sum().item()
        abn_weight = torch.tensor([abn_neg / max(abn_pos, 1)])

        can_pos = (self.cancer_labels == 1).sum().item()
        can_neg = (self.cancer_labels == 0).sum().item()
        can_weight = torch.tensor([can_neg / max(can_pos, 1)])

        return abn_weight, can_weight
