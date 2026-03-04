#!/usr/bin/env python3
"""
Evaluate SlideCheck models on BRACS dataset

Usage:
    python tools/evaluate_bracs.py \
        --ckpt logs/phase1/best_model.pt \
        --bracs_path /path/to/BRACS.pt \
        --output results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.inference import SlideCheckPredictor


def load_bracs_data(bracs_path):
    """
    Load BRACS dataset

    Expected format:
    {
        'features': torch.Tensor [N, 261, 1280],
        'labels': torch.Tensor [N] (0-6),
        'paths': list [N],
        'classes': list ['0_N', '1_PB', ..., '6_IC']
    }
    """
    data = torch.load(bracs_path, map_location='cpu', weights_only=False)

    features = data['features']  # [N, 261, 1280]
    labels = data['labels'].numpy()  # [N]

    # Convert to 2560d: CLS + mean(patches)
    cls_features = features[:, 0, :]  # [N, 1280]
    patch_features = features[:, 5:, :].mean(dim=1)  # [N, 1280] skip CLS+4 registers
    features_2560 = torch.cat([cls_features, patch_features], dim=-1)  # [N, 2560]

    # Binary labels
    cancer_labels = ((labels == 5) | (labels == 6)).astype(int)  # DCIS(5) + IC(6)
    abnormal_labels = (labels > 0).astype(int)  # PB-IC(1-6) vs N(0)

    return {
        'features': features_2560.numpy(),
        'labels': labels,
        'cancer_labels': cancer_labels,
        'abnormal_labels': abnormal_labels,
        'paths': data.get('paths', []),
        'classes': data.get('classes', [])
    }


def compute_metrics(y_true, y_pred, y_score):
    """Compute classification metrics"""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 1:
        auc = 1.0 if acc == 1.0 else 0.0
        auprc = 1.0 if acc == 1.0 else 0.0
    else:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'acc': float(acc),
        'bacc': float(bacc),
        'auc': float(auc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on BRACS dataset')

    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--bracs_path', required=True, help='Path to BRACS .pt file')
    parser.add_argument('--output', default=None, help='Output JSON file')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BRACS Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"BRACS data: {args.bracs_path}")
    print(f"{'='*60}\n")

    # Load BRACS data
    print("Loading BRACS dataset...")
    bracs_data = load_bracs_data(args.bracs_path)
    print(f"  Total samples: {len(bracs_data['features'])}")
    print(f"  Cancer: {bracs_data['cancer_labels'].sum()} / {len(bracs_data['cancer_labels'])}")
    print(f"  Abnormal: {bracs_data['abnormal_labels'].sum()} / {len(bracs_data['abnormal_labels'])}\n")

    # Initialize predictor
    print("Loading model...")
    predictor = SlideCheckPredictor(args.ckpt, device=args.device)
    print()

    # Run inference
    print("Running inference...")
    results = predictor.predict_from_embeddings(
        bracs_data['features'],
        batch_size=args.batch_size
    )
    print("Done!\n")

    # Compute metrics
    print("Computing metrics...")

    # Cancer metrics
    pred_can = (results['prob_can'] >= 0.5).astype(int)
    metrics_can = compute_metrics(
        bracs_data['cancer_labels'],
        pred_can,
        results['prob_can']
    )

    # Abnormal metrics
    pred_abn = (results['prob_abn'] >= 0.5).astype(int)
    metrics_abn = compute_metrics(
        bracs_data['abnormal_labels'],
        pred_abn,
        results['prob_abn']
    )

    # Constraint violation
    violation_rate = (results['prob_can'] > results['prob_abn']).mean()

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print("\nCancer Classification:")
    print(f"  Accuracy: {metrics_can['acc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_can['bacc']:.4f}")
    print(f"  AUC: {metrics_can['auc']:.4f}")
    print(f"  AUPRC: {metrics_can['auprc']:.4f}")
    print(f"  Sensitivity: {metrics_can['sensitivity']:.4f}")
    print(f"  Specificity: {metrics_can['specificity']:.4f}")

    print("\nAbnormal Classification:")
    print(f"  Accuracy: {metrics_abn['acc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_abn['bacc']:.4f}")
    print(f"  AUC: {metrics_abn['auc']:.4f}")
    print(f"  AUPRC: {metrics_abn['auprc']:.4f}")
    print(f"  Sensitivity: {metrics_abn['sensitivity']:.4f}")
    print(f"  Specificity: {metrics_abn['specificity']:.4f}")

    print(f"\nConstraint Violation Rate: {violation_rate:.4f}")
    print(f"{'='*60}\n")

    # Save results
    output_data = {
        'checkpoint': args.ckpt,
        'bracs_path': args.bracs_path,
        'num_samples': len(bracs_data['features']),
        'cancer': metrics_can,
        'abnormal': metrics_abn,
        'violation_rate': float(violation_rate),
        'mean_bacc': (metrics_can['bacc'] + metrics_abn['bacc']) / 2
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}\n")

    return output_data


if __name__ == '__main__':
    main()
