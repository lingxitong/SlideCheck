"""
Unified inference interface for SlideCheck

Usage:
    predictor = SlideCheckPredictor('model.pt')
    results = predictor.predict_from_h5('features.h5')
    results = predictor.predict_from_embeddings(embeddings)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional

from ..models import build_slidecheck_model
from ..utils.checkpoint import load_checkpoint


class SlideCheckPredictor:
    """
    Unified predictor for SlideCheck models

    Supports:
    - Loading from checkpoint with automatic format detection
    - Inference from HDF5 files
    - Inference from numpy/torch embeddings
    - Batch processing for memory efficiency

    Example:
        >>> predictor = SlideCheckPredictor('phase1_best.pt')
        >>> results = predictor.predict_from_h5('slide001.h5')
        >>> print(results['prob_abn'].mean())
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = 'cuda:0',
        foundation_model: Optional[str] = None
    ):
        """
        Initialize predictor

        Args:
            ckpt_path: Path to checkpoint file
            device: Device to run inference on
            foundation_model: FM name (optional, will be read from checkpoint)
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )

        # Load checkpoint
        ckpt = load_checkpoint(ckpt_path, foundation_model=foundation_model)
        config = ckpt['config']

        # Extract model config
        self.foundation_model = config.get('foundation_model', foundation_model or 'virchow2')
        arch = config.get('arch', 'baseline')
        hidden_dim = config.get('hidden_dim', 768)
        dropout = config.get('dropout', 0.1)

        print(f"Loading SlideCheck model from {ckpt_path}")
        print(f"  Foundation Model: {self.foundation_model}")
        print(f"  Architecture: {arch}")
        print(f"  Epoch: {ckpt['epoch']}")

        # Build model
        self.model = build_slidecheck_model(
            arch=arch,
            foundation_model=self.foundation_model,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Load weights
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict_from_h5(
        self,
        h5_path: str,
        h5_key: str = 'features',
        batch_size: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        Predict from HDF5 file

        Args:
            h5_path: Path to HDF5 file
            h5_key: Key in HDF5 file (default: 'features')
            batch_size: Batch size for inference

        Returns:
            Dict with keys: logit_abn, logit_can, prob_abn, prob_can
        """
        import h5py
        with h5py.File(h5_path, 'r') as f:
            if h5_key not in f:
                raise KeyError(
                    f"Key '{h5_key}' not found in {h5_path}. "
                    f"Available keys: {list(f.keys())}"
                )
            features = torch.from_numpy(f[h5_key][:]).float()

        return self.predict_from_embeddings(features, batch_size)

    def predict_from_embeddings(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        batch_size: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        Predict from embeddings

        Args:
            embeddings: Input embeddings [N, feature_dim]
            batch_size: Batch size for inference

        Returns:
            Dict with keys: logit_abn, logit_can, prob_abn, prob_can
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()

        all_logit_abn = []
        all_logit_can = []

        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].to(self.device)
                output = self.model(batch)
                all_logit_abn.append(output.logit_abn.cpu())
                all_logit_can.append(output.logit_can.cpu())

        logit_abn = torch.cat(all_logit_abn)
        logit_can = torch.cat(all_logit_can)

        return {
            'logit_abn': logit_abn.numpy(),
            'logit_can': logit_can.numpy(),
            'prob_abn': torch.sigmoid(logit_abn).numpy(),
            'prob_can': torch.sigmoid(logit_can).numpy(),
        }

    def predict_batch(
        self,
        h5_paths: list,
        h5_key: str = 'features',
        batch_size: int = 512
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict from multiple HDF5 files

        Args:
            h5_paths: List of HDF5 file paths
            h5_key: Key in HDF5 files
            batch_size: Batch size for inference

        Returns:
            Dict mapping file paths to prediction results
        """
        results = {}
        for h5_path in h5_paths:
            results[h5_path] = self.predict_from_h5(h5_path, h5_key, batch_size)
        return results
