"""Binary classification metrics, written by hand."""
from typing import Dict

from torch import Tensor


def binary_metrics(preds: Tensor, labels: Tensor) -> Dict[str, float]:
    """preds, labels: 1-D integer tensors with values 0/1 (1 = positive class)."""
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "acc": (tp + tn) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }