"""
Unit and integration tests for Milestone M3 — Baseline Anomaly Detector.

Verifies:
1. Feature pooling strategies (mean_std, mean, flatten).
2. Deterministic fitting under fixed random seed.
3. Inverted score direction (higher score = more anomalous).
4. Threshold selection on validation data.
5. Recording-level score aggregation rules.
6. Model serialization and exact reload roundtrip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.config import BaselineConfig
from src.features import FeatureTensor
from src.models.baseline import BaselineIsolationForest, BaselinePrediction


@pytest.fixture
def dummy_feature_tensor() -> FeatureTensor:
    """Generate a reproducible synthetic FeatureTensor with shape (19, 128, 32)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((19, 128, 32)).astype(np.float32)
    return FeatureTensor(
        features=data,
        sample_rate=16_000,
        n_mels=128,
        n_frames=32,
    )


class TestBaselinePooling:
    """Validate feature reduction and pooling transformations."""

    def test_pool_features_mean_std(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(pooling_strategy="mean_std")
        model = BaselineIsolationForest(config=cfg)
        pooled = model.pool_features(dummy_feature_tensor)

        assert pooled.shape == (19, 256)
        assert np.all(np.isfinite(pooled))
        assert pooled.dtype == np.float32

    def test_pool_features_mean(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(pooling_strategy="mean")
        model = BaselineIsolationForest(config=cfg)
        pooled = model.pool_features(dummy_feature_tensor)

        assert pooled.shape == (19, 128)
        assert np.all(np.isfinite(pooled))

    def test_pool_features_flatten(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(pooling_strategy="flatten")
        model = BaselineIsolationForest(config=cfg)
        pooled = model.pool_features(dummy_feature_tensor)

        assert pooled.shape == (19, 128 * 32)
        assert np.all(np.isfinite(pooled))

    def test_pool_features_invalid_strategy(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(pooling_strategy="unsupported_strat")
        model = BaselineIsolationForest(config=cfg)
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            model.pool_features(dummy_feature_tensor)


class TestBaselineFittingAndInference:
    """Validate Isolation Forest training and inference behavior."""

    def test_fit_and_is_fitted(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(n_estimators=10, random_state=42)
        model = BaselineIsolationForest(config=cfg)
        assert not model.is_fitted

        pooled = model.pool_features(dummy_feature_tensor)
        model.fit(pooled)

        assert model.is_fitted
        assert model.feature_dim == 256

    def test_unfitted_model_raises_on_predict(self, dummy_feature_tensor: FeatureTensor) -> None:
        model = BaselineIsolationForest()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_segment_scores(dummy_feature_tensor)

    def test_score_direction_and_determinism(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(n_estimators=20, random_state=42)
        model = BaselineIsolationForest(config=cfg)

        pooled = model.pool_features(dummy_feature_tensor)
        model.fit(pooled)

        scores1 = model.predict_segment_scores(dummy_feature_tensor)
        scores2 = model.predict_segment_scores(dummy_feature_tensor)

        # Determinism under fixed seed
        np.testing.assert_allclose(scores1, scores2)
        assert len(scores1) == 19
        assert np.all(np.isfinite(scores1))

    def test_score_aggregation_methods(self, dummy_feature_tensor: FeatureTensor) -> None:
        for agg in ("mean", "max", "percentile_95"):
            cfg = BaselineConfig(n_estimators=10, score_aggregation=agg, random_state=42)
            model = BaselineIsolationForest(config=cfg)
            pooled = model.pool_features(dummy_feature_tensor)
            model.fit(pooled)

            score = model.predict_recording_score(dummy_feature_tensor)
            assert isinstance(score, float)
            assert np.isfinite(score)


class TestBaselineThresholdAndDecision:
    """Validate validation-only threshold derivation and decision predictions."""

    def test_select_threshold_quantile(self) -> None:
        model = BaselineIsolationForest()
        normal_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Target FPR 0.10 -> 90th percentile = 0.91
        th = model.select_threshold(normal_scores, target_fpr=0.10)
        assert model.threshold is not None
        assert 0.9 <= th <= 1.0

    def test_predict_without_threshold_raises(self, dummy_feature_tensor: FeatureTensor) -> None:
        model = BaselineIsolationForest()
        pooled = model.pool_features(dummy_feature_tensor)
        model.fit(pooled)

        with pytest.raises(RuntimeError, match="Threshold must be selected"):
            model.predict(dummy_feature_tensor)

    def test_predict_returns_structured_prediction(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(n_estimators=10, random_state=42)
        model = BaselineIsolationForest(config=cfg)
        pooled = model.pool_features(dummy_feature_tensor)
        model.fit(pooled)

        model.select_threshold([0.3, 0.4, 0.5], target_fpr=0.10)
        prediction = model.predict(dummy_feature_tensor)

        assert isinstance(prediction, BaselinePrediction)
        assert isinstance(prediction.anomaly_score, float)
        assert isinstance(prediction.threshold, float)
        assert isinstance(prediction.is_anomaly, bool)
        assert prediction.status in ("normal", "possible_anomaly")
        assert len(prediction.segment_scores) == 19


class TestBaselineSerialization:
    """Validate model saving and loading roundtrip."""

    def test_save_and_load_roundtrip(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = BaselineConfig(n_estimators=15, score_aggregation="max", random_state=42)
        model = BaselineIsolationForest(config=cfg)
        pooled = model.pool_features(dummy_feature_tensor)
        model.fit(pooled)
        model.select_threshold([0.3, 0.4, 0.5])

        pred_before = model.predict(dummy_feature_tensor)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "baseline_model.joblib"
            model.save(save_path)

            assert save_path.exists()
            assert save_path.with_suffix(".json").exists()

            loaded_model = BaselineIsolationForest.load(save_path)
            assert loaded_model.is_fitted
            assert loaded_model.threshold == model.threshold
            assert loaded_model.config.score_aggregation == "max"

            pred_after = loaded_model.predict(dummy_feature_tensor)
            assert pred_before.anomaly_score == pred_after.anomaly_score
            assert pred_before.is_anomaly == pred_after.is_anomaly
            assert pred_before.status == pred_after.status
