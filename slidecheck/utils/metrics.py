"""
Metrics computation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics

    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_score: Prediction scores [N]

    Returns:
        Dict with metrics: acc, bacc, auc, auprc, sensitivity, specificity
    """
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # Handle single-class case
    if len(np.unique(y_true)) == 1:
        auc = 1.0 if acc == 1.0 else 0.0
        auprc = 1.0 if acc == 1.0 else 0.0
    else:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # All samples belong to one class and all predicted correctly
        only_class = y_true[0]
        count = int(cm[0, 0])
        if only_class == 1:
            tp, tn, fp, fn = count, 0, 0, 0
        else:
            tn, tp, fp, fn = count, 0, 0, 0
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'acc': float(acc),
        'bacc': float(bacc),
        'auc': float(auc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_dual_head_metrics(
    abnormal_true: np.ndarray,
    abnormal_pred: np.ndarray,
    abnormal_score: np.ndarray,
    cancer_true: np.ndarray,
    cancer_pred: np.ndarray,
    cancer_score: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for dual-head classification

    Args:
        abnormal_true: True abnormal labels
        abnormal_pred: Predicted abnormal labels
        abnormal_score: Abnormal prediction scores
        cancer_true: True cancer labels
        cancer_pred: Predicted cancer labels
        cancer_score: Cancer prediction scores

    Returns:
        Dict with 'abnormal' and 'cancer' metrics
    """
    abnormal_metrics = compute_binary_metrics(
        abnormal_true, abnormal_pred, abnormal_score
    )

    cancer_metrics = compute_binary_metrics(
        cancer_true, cancer_pred, cancer_score
    )

    # Compute constraint violation rate
    violation_rate = float((cancer_score > abnormal_score).mean())

    return {
        'abnormal': abnormal_metrics,
        'cancer': cancer_metrics,
        'violation_rate': violation_rate,
        'mean_bacc': (abnormal_metrics['bacc'] + cancer_metrics['bacc']) / 2
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute multiclass classification metrics

    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_score: Prediction scores [N, num_classes] (optional)

    Returns:
        Dict with metrics
    """
    from sklearn.metrics import classification_report

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'acc': float(acc),
        'bacc': float(bacc)
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class'] = report

    # Multiclass AUC if scores provided
    if y_score is not None and len(np.unique(y_true)) > 2:
        from sklearn.preprocessing import label_binarize
        n_classes = y_score.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        try:
            auc_ovr = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
            metrics['auc_macro'] = float(auc_ovr)
        except Exception:
            metrics['auc_macro'] = 0.0

    return metrics
