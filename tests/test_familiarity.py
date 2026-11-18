"""
Unit tests for latent embedding familiarity / domain-shift estimation (Milestone M5).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.familiarity import LatentFamiliarityEstimator


class TestLatentFamiliarityEstimator:
    """Test fitting, scoring, thresholding, and serialization of familiarity estimator."""

    @pytest.fixture
    def normal_train_embeddings(self) -> np.ndarray:
        # 100 samples from N(0, 1) in 32 dimensions
        np.random.seed(42)
        return np.random.randn(100, 32).astype(np.float32)

    def test_fit_and_is_fitted(self, normal_train_embeddings: np.ndarray) -> None:
        estimator = LatentFamiliarityEstimator()
        assert estimator.is_fitted is False
        estimator.fit(normal_train_embeddings)
        assert estimator.is_fitted is True
        assert estimator.latent_dim == 32
        assert estimator.centroid.shape == (32,)
        assert estimator.variance.shape == (32,)

    def test_unfitted_raises(self) -> None:
        estimator = LatentFamiliarityEstimator()
        with pytest.raises(RuntimeError, match="not fitted"):
            estimator.score_segments(np.zeros((5, 32)))

    def test_score_exact_centroid(self, normal_train_embeddings: np.ndarray) -> None:
        estimator = LatentFamiliarityEstimator().fit(normal_train_embeddings)
        # Centroid should have 0 distance
        centroid_sample = estimator.centroid.reshape(1, -1)
        dist = estimator.score_segments(centroid_sample)
        assert pytest.approx(dist[0], abs=1e-5) == 0.0

    def test_threshold_selection_and_outlier_detection(
        self, normal_train_embeddings: np.ndarray
    ) -> None:
        np.random.seed(42)
        estimator = LatentFamiliarityEstimator().fit(normal_train_embeddings)

        # Validation set: 20 normal recordings (each 10 segments)
        val_recordings = [
            np.random.randn(10, 32).astype(np.float32) for _ in range(20)
        ]
        threshold = estimator.select_threshold(val_recordings, target_fpr=0.10)
        assert threshold > 0.0

        # Normal test recording (drawn from same distribution)
        normal_test = np.random.randn(10, 32).astype(np.float32)
        normal_pred = estimator.predict(normal_test)
        # Should have reasonable distance
        assert normal_pred.threshold == round(threshold, 4)

        # Outlier / domain-shifted recording (mean shifted by 10.0)
        outlier_test = (np.random.randn(10, 32) + 10.0).astype(np.float32)
        outlier_pred = estimator.predict(outlier_test)
        assert outlier_pred.score > threshold
        assert outlier_pred.is_familiar is False

    def test_save_load_roundtrip(
        self, normal_train_embeddings: np.ndarray, tmp_path: Path
    ) -> None:
        estimator = LatentFamiliarityEstimator().fit(normal_train_embeddings)
        estimator.threshold = 4.25

        save_path = tmp_path / "familiarity.joblib"
        estimator.save(save_path)
        assert save_path.is_file()

        loaded = LatentFamiliarityEstimator.load(save_path)
        assert loaded.is_fitted is True
        assert loaded.threshold == 4.25
        assert loaded.latent_dim == 32
        np.testing.assert_allclose(loaded.centroid, estimator.centroid)
        np.testing.assert_allclose(loaded.variance, estimator.variance)
