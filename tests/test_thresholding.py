"""
Unit tests for threshold calibration and classification metrics (Milestone M5).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.thresholding import (
    compute_binary_metrics,
    select_threshold_iqr,
    select_threshold_quantile,
)


class TestThresholding:
    """Test quantile-based and IQR threshold calibration rules."""

    def test_select_threshold_quantile_uniform(self) -> None:
        # Uniform numbers 0 to 100
        scores = np.arange(101, dtype=np.float64)
        # target_fpr = 0.10 -> 90th percentile -> 90.0
        thresh = select_threshold_quantile(scores, target_fpr=0.10)
        assert pytest.approx(thresh, rel=1e-3) == 90.0

    def test_select_threshold_invalid_fpr(self) -> None:
        scores = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="target_fpr must be strictly between 0 and 1"):
            select_threshold_quantile(scores, target_fpr=0.0)

        with pytest.raises(ValueError, match="target_fpr must be strictly between 0 and 1"):
            select_threshold_quantile(scores, target_fpr=1.0)

    def test_select_threshold_empty_or_nan(self) -> None:
        with pytest.raises(ValueError, match="empty score array"):
            select_threshold_quantile([])

        with pytest.raises(ValueError, match="NaN or infinite"):
            select_threshold_quantile([1.0, np.nan, 3.0])

    def test_select_threshold_iqr(self) -> None:
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Q1 = 20, Q3 = 40, IQR = 20 -> Q3 + 1.5 * 20 = 70
        thresh = select_threshold_iqr(scores, k=1.5)
        assert pytest.approx(thresh, rel=1e-2) == 70.0


class TestBinaryMetrics:
    """Test evaluation metric calculations."""

    def test_compute_binary_metrics_perfect_separation(self) -> None:
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        scores = [0.1, 0.2, 0.15, 0.25, 0.18, 0.85, 0.90, 0.78, 0.95, 0.88]
        threshold = 0.5

        metrics = compute_binary_metrics(y_true, scores, threshold)
        assert metrics["auroc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["confusion_matrix"]["true_positive"] == 5
        assert metrics["confusion_matrix"]["true_negative"] == 5

    def test_compute_binary_metrics_length_mismatch(self) -> None:
        y_true = [0, 1]
        scores = [0.2, 0.8, 0.9]
        with pytest.raises(ValueError, match="does not match"):
            compute_binary_metrics(y_true, scores, threshold=0.5)
