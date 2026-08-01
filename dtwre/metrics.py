"""Evaluation metrics (Section 3.4, Eq. 8-11)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = ["binary_metrics"]


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """AUC plus precision / recall / F1 / accuracy at ``threshold``.

    Eq. 8   Precision = TP / (TP + FP)
    Eq. 9   Recall    = TP / (TP + FN)
    Eq. 10  F1        = 2 * P * R / (P + R)
    Eq. 11  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    auc = (float(roc_auc_score(y_true, y_score))
           if len(np.unique(y_true)) > 1 else float("nan"))

    return {"auc": auc, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
