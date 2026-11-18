"""
Unit and integration tests for Milestone M4 — Autoencoder Anomaly Model.

Verifies:
1. AcousticAutoencoder PyTorch architecture dimensions (forward, encode, decode).
2. Autoencoder training loop and convergence on feature segments.
3. Reconstruction error calculations and finite values.
4. Latent embedding extraction for downstream familiarity checks.
5. Threshold selection strictly on validation normal data.
6. Checkpoint save, load, and deterministic inference roundtrip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.features import FeatureTensor
from src.models.autoencoder import (
    AcousticAutoencoder,
    AutoencoderDetector,
    AutoencoderPrediction,
)


@pytest.fixture
def dummy_feature_tensor() -> FeatureTensor:
    """Generate a reproducible synthetic FeatureTensor (19 segments, 128 Mel bins, 32 frames)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((19, 128, 32)).astype(np.float32)
    return FeatureTensor(
        features=data,
        sample_rate=16_000,
        n_mels=128,
        n_frames=32,
    )


class TestAcousticAutoencoderArchitecture:
    """Validate PyTorch module tensor dimensions and layer operations."""

    def test_forward_shape(self) -> None:
        model = AcousticAutoencoder(input_dim=4096, hidden_dims=(512, 128, 64), latent_dim=32)
        model.eval()
        x = torch.randn(8, 4096)
        with torch.no_grad():
            x_rec = model(x)
        assert x_rec.shape == (8, 4096)
        assert not torch.isnan(x_rec).any()

    def test_latent_encoding_shape(self) -> None:
        model = AcousticAutoencoder(input_dim=4096, hidden_dims=(512, 128, 64), latent_dim=32)
        model.eval()
        x = torch.randn(4, 4096)
        with torch.no_grad():
            z = model.encode(x)
        assert z.shape == (4, 32)
        assert not torch.isnan(z).any()

    def test_decoder_shape(self) -> None:
        model = AcousticAutoencoder(input_dim=4096, hidden_dims=(512, 128, 64), latent_dim=32)
        model.eval()
        z = torch.randn(4, 32)
        with torch.no_grad():
            x_rec = model.decode(z)
        assert x_rec.shape == (4, 4096)


class TestAutoencoderDetectorTrainingAndInference:
    """Validate high-level detector training, reconstruction error, and embeddings."""

    def test_fit_and_loss_decrease(self) -> None:
        cfg = ModelConfig(
            input_dim=4096,
            hidden_dims=(64, 32),
            latent_dim=16,
            epochs=5,
            batch_size=8,
            learning_rate=1e-2,
            seed=42,
        )
        detector = AutoencoderDetector(config=cfg, device="cpu")

        rng = np.random.default_rng(42)
        # Create normal training data around a mean signature
        train_data = rng.standard_normal((32, 4096)).astype(np.float32) * 0.1
        val_data = rng.standard_normal((8, 4096)).astype(np.float32) * 0.1

        history = detector.fit(train_data, val_features=val_data, epochs=5, verbose=False)

        assert detector.is_trained
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        # Loss should be positive and decreasing/stabilizing
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_reconstruct_shapes_and_finite(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = ModelConfig(input_dim=4096, hidden_dims=(64,), latent_dim=16, seed=42)
        detector = AutoencoderDetector(config=cfg, device="cpu")

        rec_arr, segment_mse = detector.reconstruct(dummy_feature_tensor)

        assert rec_arr.shape == (19, 128, 32)
        assert segment_mse.shape == (19,)
        assert np.all(np.isfinite(rec_arr))
        assert np.all(np.isfinite(segment_mse))
        assert np.all(segment_mse >= 0.0)

    def test_recording_score_aggregation_methods(self, dummy_feature_tensor: FeatureTensor) -> None:
        for agg in ("mean", "max", "percentile_95"):
            cfg = ModelConfig(input_dim=4096, hidden_dims=(64,), latent_dim=16, score_aggregation=agg, seed=42)
            detector = AutoencoderDetector(config=cfg, device="cpu")

            rec_score = detector.predict_recording_score(dummy_feature_tensor)
            assert isinstance(rec_score, float)
            assert np.isfinite(rec_score)
            assert rec_score >= 0.0

    def test_extract_latent_embeddings(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = ModelConfig(input_dim=4096, hidden_dims=(64,), latent_dim=16, seed=42)
        detector = AutoencoderDetector(config=cfg, device="cpu")

        z = detector.extract_latent_embeddings(dummy_feature_tensor)
        assert z.shape == (19, 16)
        assert np.all(np.isfinite(z))


class TestAutoencoderThresholdAndPrediction:
    """Validate validation-only threshold selection and structured predictions."""

    def test_select_threshold_quantile(self) -> None:
        detector = AutoencoderDetector()
        val_normal = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]

        # Target FPR 0.10 -> 90th percentile = 0.305
        th = detector.select_threshold(val_normal, target_fpr=0.10)
        assert detector.threshold is not None
        assert 0.30 <= th <= 0.35

    def test_predict_without_threshold_raises(self, dummy_feature_tensor: FeatureTensor) -> None:
        detector = AutoencoderDetector()
        with pytest.raises(RuntimeError, match="Threshold must be selected"):
            detector.predict(dummy_feature_tensor)

    def test_predict_returns_structured_prediction(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = ModelConfig(input_dim=4096, hidden_dims=(64,), latent_dim=16, seed=42)
        detector = AutoencoderDetector(config=cfg, device="cpu")
        detector.select_threshold([0.1, 0.2, 0.3], target_fpr=0.10)

        pred = detector.predict(dummy_feature_tensor)
        assert isinstance(pred, AutoencoderPrediction)
        assert isinstance(pred.anomaly_score, float)
        assert isinstance(pred.threshold, float)
        assert isinstance(pred.is_anomaly, bool)
        assert pred.status in ("normal", "possible_anomaly")
        assert len(pred.reconstruction_errors) == 19


class TestAutoencoderSerialization:
    """Validate model saving to disk and exact reload roundtrip."""

    def test_save_and_load_roundtrip(self, dummy_feature_tensor: FeatureTensor) -> None:
        cfg = ModelConfig(
            input_dim=4096,
            hidden_dims=(64, 32),
            latent_dim=16,
            score_aggregation="max",
            seed=42,
        )
        detector = AutoencoderDetector(config=cfg, device="cpu")
        detector.select_threshold([0.1, 0.2, 0.3], target_fpr=0.10)

        pred_orig = detector.predict(dummy_feature_tensor)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "autoencoder_test.pt"
            detector.save(save_path)

            assert save_path.exists()
            assert save_path.with_suffix(".json").exists()

            loaded = AutoencoderDetector.load(save_path, device="cpu")
            assert loaded.threshold == detector.threshold
            assert loaded.config.score_aggregation == "max"
            assert loaded.config.latent_dim == 16

            pred_loaded = loaded.predict(dummy_feature_tensor)
            assert pred_orig.anomaly_score == pred_loaded.anomaly_score
            assert pred_orig.is_anomaly == pred_loaded.is_anomaly
            assert pred_orig.status == pred_loaded.status
            np.testing.assert_allclose(pred_orig.reconstruction_errors, pred_loaded.reconstruction_errors, atol=1e-5)
