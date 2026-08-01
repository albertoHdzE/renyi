"""Metrics matching the Performance column of Tables 1-2.

Those tables report ACC for most rows, Macro-F1 for three (Zhiyuan et al., Yuan
et al., Weizhi et al.) and AUC for one (Nguyen et al.). All three are computed
for every run so a comparison never has to convert between them -- and so the
gap between accuracy and macro-F1 stays visible on the imbalanced corpora, where
PHEME's 63% non-rumor majority makes accuracy alone misleading.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

__all__ = ["classification_metrics", "softmax"]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(y_true: np.ndarray, logits: np.ndarray,
                           label_names: list[str]) -> dict:
    """Accuracy, macro-F1, per-class F1, AUC and the confusion matrix."""
    proba = softmax(logits)
    y_pred = proba.argmax(axis=1)
    n_classes = len(label_names)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro",
                                   zero_division=0)),
        "majority_baseline": float(np.bincount(y_true, minlength=n_classes).max()
                                   / max(len(y_true), 1)),
    }

    # AUC needs every class present in y_true; a held-out event split can drop
    # one, in which case the score is undefined rather than zero.
    try:
        if n_classes == 2:
            out["auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        elif len(np.unique(y_true)) == n_classes:
            out["auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                             average="macro"))
        else:
            out["auc"] = float("nan")
    except ValueError:
        out["auc"] = float("nan")

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n_classes)))
    out["per_class_f1"] = {lab: float(v)
                           for lab, v in zip(label_names, per_class)}
    out["confusion"] = confusion_matrix(
        y_true, y_pred, labels=list(range(n_classes))).tolist()
    return out
