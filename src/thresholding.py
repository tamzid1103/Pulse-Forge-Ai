"""
Threshold calibration and classification evaluation engine for PulseForge AI (Milestone M5).

Provides rigorous, reproducible threshold selection rules derived strictly
from validation-set normal acoustic recordings, avoiding test-set leakage.
Also computes standard binary classification metrics (AUROC, PR-AUC, F1, precision, recall, FPR).
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn import metrics

logger = logging.getLogger("pulseforge.thresholding")


def select_threshold_quantile(
    normal_scores: np.ndarray | list[float],
    target_fpr: float = 0.10,
) -> float:
    """
    Select decision threshold from normal validation scores at a target false-positive rate.
    
    For a target FPR of 0.10 (10%), the threshold is set at the (1.0 - 0.10) = 90th percentile
    of the validation normal score distribution.
    
    Args:
        normal_scores: 1D array of anomaly or distance scores for normal validation samples.
        target_fpr: Desired false-positive rate on normal validation data (0 < target_fpr < 1).
        
    Returns:
        Calibrated threshold as a float.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot select threshold from an empty score array.")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError(f"target_fpr must be strictly between 0 and 1, got {target_fpr}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Score array contains NaN or infinite values.")

    quantile_level = (1.0 - target_fpr) * 100.0
    threshold = float(np.percentile(scores, quantile_level))
    logger.info(
        "Calibrated threshold %.6f at target FPR=%.2f (percentile=%.1f%%) on %d samples.",
        threshold,
        target_fpr,
        quantile_level,
        scores.size,
    )
    return threshold


def select_threshold_iqr(
    normal_scores: np.ndarray | list[float],
    k: float = 1.5,
) -> float:
    """
    Select threshold using Tukey's Interquartile Range (IQR) fence.
    
    Formula: threshold = Q3 + k * (Q3 - Q1)
    
    Args:
        normal_scores: 1D array of normal validation scores.
        k: Outlier multiplier (default 1.5 for mild outliers, 3.0 for extreme outliers).
        
    Returns:
        Threshold value.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot select threshold from an empty score array.")
    q25, q75 = np.percentile(scores, [25.0, 75.0])
    iqr = q75 - q25
    threshold = float(q75 + k * iqr)
    return threshold


def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float,
) -> dict[str, Any]:
    """
    Compute binary classification evaluation metrics for a given decision threshold.
    
    Args:
        y_true: 1D array of binary ground-truth labels (0 = normal, 1 = abnormal).
        scores: 1D array of continuous anomaly / distance scores (higher = more anomalous).
        threshold: Decision threshold for classification (score >= threshold -> predicted 1).
        
    Returns:
        Dictionary containing:
        - auroc, pr_auc
        - precision, recall, f1_score, false_positive_rate
        - confusion_matrix (TP, FP, TN, FN)
        - threshold
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)

    if y_t.shape[0] != s.shape[0]:
        raise ValueError(
            f"y_true length ({y_t.shape[0]}) does not match scores length ({s.shape[0]})"
        )
    if y_t.size == 0:
        raise ValueError("Cannot evaluate metrics on empty inputs.")

    y_pred = (s >= threshold).astype(np.int32)

    # Confusion matrix
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(np.unique(y_t)) > 1:
        cm = metrics.confusion_matrix(y_t, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        # Single-class case
        if y_t[0] == 0:
            fp = int(np.sum(y_pred == 1))
            tn = int(np.sum(y_pred == 0))
        else:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))

    precision = float(metrics.precision_score(y_t, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_t, y_pred, zero_division=0))
    f1 = float(metrics.f1_score(y_t, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Curve areas (requires both positive and negative samples)
    if len(np.unique(y_t)) > 1:
        auroc = float(metrics.roc_auc_score(y_t, s))
        pr_auc = float(metrics.average_precision_score(y_t, s))
    else:
        auroc = float("nan")
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "auroc": round(auroc, 4) if np.isfinite(auroc) else None,
        "pr_auc": round(pr_auc, 4) if np.isfinite(pr_auc) else None,
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }
