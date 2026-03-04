"""
Data augmentation transforms for SlideCheck

Implements:
- Mixup: Interpolation-based augmentation
- FeaturePerturbation: Multiplicative noise augmentation
"""

import torch
import numpy as np
from typing import Tuple


class Mixup:
    """
    Mixup augmentation for feature embeddings

    Reference:
        mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)

    Args:
        alpha: Beta distribution parameter (default: 0.4)
        prob: Probability of applying mixup (default: 1.0)
    """

    def __init__(self, alpha: float = 0.4, prob: float = 1.0):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        embeddings: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Mixup augmentation

        Args:
            embeddings: Input embeddings [batch_size, feature_dim]
            labels_a: First set of labels [batch_size]
            labels_b: Second set of labels [batch_size]

        Returns:
            Tuple of (mixed_embeddings, mixed_labels_a, mixed_labels_b)
        """
        if np.random.rand() > self.prob:
            return embeddings, labels_a, labels_b

        batch_size = embeddings.size(0)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

        # Random permutation
        index = torch.randperm(batch_size, device=embeddings.device)

        # Mix embeddings and labels
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        mixed_labels_a = lam * labels_a + (1 - lam) * labels_a[index]
        mixed_labels_b = lam * labels_b + (1 - lam) * labels_b[index]

        return mixed_embeddings, mixed_labels_a, mixed_labels_b


class FeaturePerturbation:
    """
    Feature perturbation with multiplicative noise

    Applies element-wise multiplicative noise: x' = x * (1 + ε)
    where ε ~ N(0, σ²)

    Args:
        sigma: Standard deviation of noise (default: 0.1)
        prob: Probability of applying perturbation (default: 1.0)
    """

    def __init__(self, sigma: float = 0.1, prob: float = 1.0):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply feature perturbation

        Args:
            embeddings: Input embeddings [batch_size, feature_dim]

        Returns:
            Perturbed embeddings
        """
        if np.random.rand() > self.prob:
            return embeddings

        noise = torch.randn_like(embeddings) * self.sigma
        return embeddings * (1 + noise)


class Compose:
    """
    Compose multiple transforms

    Args:
        transforms: List of transform functions
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        """Apply transforms sequentially"""
        result = args
        for transform in self.transforms:
            if isinstance(result, tuple):
                result = transform(*result)
            else:
                result = transform(result)
        return result
