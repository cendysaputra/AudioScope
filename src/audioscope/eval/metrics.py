"""Classification metrics computed from integer labels (no external deps)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of clips of true class ``i`` predicted as ``j``."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1 over classes present in the ground truth."""
    present = cm.sum(axis=1) > 0
    return float(per_class_f1(cm)[present].mean())


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall per class; NaN for classes absent from the ground truth."""
    true = cm.sum(axis=1).astype(float)
    return np.divide(np.diag(cm), true, out=np.full(len(cm), np.nan), where=true > 0)


def fold_metrics(y_true, y_pred, num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(cm),
        "per_class_accuracy": per_class_accuracy(cm).tolist(),
        "confusion_matrix": cm,
    }


def summarize(values) -> dict:
    """Mean and sample standard deviation (ddof=1) across folds."""
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }
