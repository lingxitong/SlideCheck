"""
Base trainer class for SlideCheck models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import json

from ..utils.checkpoint import save_checkpoint
from ..utils.metrics import compute_dual_head_metrics


class BaseTrainer:
    """
    Base trainer for SlideCheck models

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        log_dir: Directory to save logs and checkpoints
        foundation_model: Foundation model name (for checkpoint saving)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda:0',
        log_dir: str = './logs',
        foundation_model: str = 'virchow2'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.log_dir = Path(log_dir)
        self.foundation_model = foundation_model

        self.model.to(self.device)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = []

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average training loss
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")

    def validate(self) -> Dict[str, Any]:
        """
        Validate the model

        Returns:
            Dict with validation metrics
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def train(
        self,
        num_epochs: int,
        early_stop_patience: int = 30,
        metric_key: str = 'mean_bacc'
    ):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            early_stop_patience: Early stopping patience
            metric_key: Metric to use for early stopping
        """
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log
            print(f"Train Loss: {train_loss:.4f}")
            self._print_metrics(val_metrics)

            # Save history
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Check for improvement
            current_metric = val_metrics.get(metric_key, 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                patience_counter = 0

                # Save best model
                self.save_checkpoint('best_model.pt', val_metrics)
                print(f"★ New best model saved! {metric_key}: {self.best_metric:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save final model
        self.save_checkpoint('final_model.pt', val_metrics)

        # Save history
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best {metric_key}: {self.best_metric:.4f}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, filename: str, metrics: Dict[str, Any]):
        """Save model checkpoint"""
        save_path = self.log_dir / filename

        # Get model config
        config = {
            'arch': getattr(self.model, 'arch', 'unknown'),
            'hidden_dim': getattr(self.model, 'hidden_dim', None),
            'dropout': getattr(self.model, 'dropout', None)
        }

        save_checkpoint(
            model=self.model,
            save_path=str(save_path),
            foundation_model=self.foundation_model,
            epoch=self.current_epoch,
            metrics=metrics,
            config=config
        )

        # Save metrics separately
        metrics_path = self.log_dir / filename.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way"""
        if 'abnormal' in metrics and 'cancer' in metrics:
            # Dual-head metrics
            print(f"Val Abnormal - BACC: {metrics['abnormal']['bacc']:.4f}, "
                  f"AUC: {metrics['abnormal']['auc']:.4f}")
            print(f"Val Cancer - BACC: {metrics['cancer']['bacc']:.4f}, "
                  f"AUC: {metrics['cancer']['auc']:.4f}")
            print(f"Mean BACC: {metrics.get('mean_bacc', 0.0):.4f}")
            if 'violation_rate' in metrics:
                print(f"Violation Rate: {metrics['violation_rate']:.4f}")
        else:
            # Generic metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
