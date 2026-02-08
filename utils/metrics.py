"""Evaluation metrics for plant disease classification."""

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report


def compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute top-1 accuracy.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground truth class indices, shape (N,).

    Returns:
        Top-1 accuracy as a float.
    """
    return accuracy_score(targets, preds)


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute macro-averaged F1-score.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground truth class indices, shape (N,).

    Returns:
        Macro F1-score as a float.
    """
    return f1_score(targets, preds, average="macro")


def compute_classification_report(
    preds: np.ndarray, targets: np.ndarray, class_names: list[str] = None
) -> str:
    """Generate a full classification report.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground truth class indices, shape (N,).
        class_names: Optional list of class names for labeling.

    Returns:
        Formatted classification report string.
    """
    return classification_report(targets, preds, target_names=class_names)
