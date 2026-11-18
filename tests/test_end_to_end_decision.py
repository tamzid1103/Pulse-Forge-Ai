"""
Integration tests for end-to-end decision pipeline (Milestone M5).

Verifies the full path:
Audio signal -> Quality checks -> Feature extraction -> Autoencoder inference ->
Familiarity check -> Industrial decision policy.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.audio_io import AudioMetadata as IOAudioMetadata, AudioSignal
from src.config import AppConfig
from src.decision import evaluate_decision_policy
from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
from src.quality import check_signal_quality
from src.schemas import DecisionStatus, ModelPrediction


@pytest.fixture(scope="module")
def app_config() -> AppConfig:
    return AppConfig()


@pytest.fixture(scope="module")
def detector(app_config: AppConfig) -> AutoencoderDetector:
    ckpt_path = app_config.paths.model_artifacts_dir / "autoencoder_fan_id_00.pt"
    if not ckpt_path.is_file():
        pytest.skip(f"Autoencoder checkpoint not found: {ckpt_path}")
    return AutoencoderDetector.load(ckpt_path)


@pytest.fixture(scope="module")
def scaler(app_config: AppConfig) -> FeatureScaler:
    scaler_path = app_config.paths.model_artifacts_dir / "scaler_params.json"
    if not scaler_path.is_file():
        pytest.skip(f"FeatureScaler artifact not found: {scaler_path}")
    return FeatureScaler.load(scaler_path)


@pytest.fixture(scope="module")
def familiarity_estimator(app_config: AppConfig) -> LatentFamiliarityEstimator:
    fam_path = app_config.paths.model_artifacts_dir / "familiarity_fan_id_00.joblib"
    if not fam_path.is_file():
        pytest.skip(f"Familiarity estimator artifact not found: {fam_path}")
    return LatentFamiliarityEstimator.load(fam_path)


class TestEndToEndPipeline:
    """Test full decision pipeline on normal, anomalous, and degraded inputs."""

    def test_pipeline_silent_audio_triggers_needs_review(
        self, app_config: AppConfig
    ) -> None:
        sr = app_config.audio.sample_rate
        silent = np.zeros(sr * 5, dtype=np.float32)

        quality = check_signal_quality(silent, sr, app_config.quality)
        result = evaluate_decision_policy(quality=quality)

        assert result.status == DecisionStatus.NEEDS_REVIEW
        assert result.quality.is_silent is True
        assert "quality is poor" in result.explanation.lower()

    def test_pipeline_clipped_audio_triggers_needs_review(
        self, app_config: AppConfig
    ) -> None:
        sr = app_config.audio.sample_rate
        t = np.linspace(0, 5.0, sr * 5, endpoint=False)
        clipped = np.clip(10.0 * np.sin(2 * np.pi * 100 * t), -1.0, 1.0).astype(np.float32)

        quality = check_signal_quality(clipped, sr, app_config.quality)
        result = evaluate_decision_policy(quality=quality)

        assert result.status == DecisionStatus.NEEDS_REVIEW
        assert result.quality.is_clipped is True
        assert "quality is poor" in result.explanation.lower()

    def test_pipeline_short_audio_triggers_invalid_input(
        self, app_config: AppConfig
    ) -> None:
        sr = app_config.audio.sample_rate
        short_sig = 0.1 * np.ones(int(sr * 0.2), dtype=np.float32)

        quality = check_signal_quality(short_sig, sr, app_config.quality)
        result = evaluate_decision_policy(quality=quality)

        assert result.status == DecisionStatus.INVALID_INPUT
        assert result.quality.is_duration_valid is False
        assert "validation" in result.explanation.lower()

    def test_pipeline_synthetic_domain_shift(
        self,
        app_config: AppConfig,
        detector: AutoencoderDetector,
        scaler: FeatureScaler,
        familiarity_estimator: LatentFamiliarityEstimator,
    ) -> None:
        sr = app_config.audio.sample_rate
        # Synthetic white noise audio
        np.random.seed(99)
        noise = np.random.uniform(-0.3, 0.3, sr * 5).astype(np.float32)

        quality = check_signal_quality(noise, sr, app_config.quality)
        assert quality.passed is True

        dummy_meta = IOAudioMetadata(
            filename="noise.wav",
            size_bytes=noise.nbytes,
            sample_rate=sr,
            channels=1,
            duration_seconds=5.0,
            format="WAV",
            subtype="PCM_16",
        )
        audio = AudioSignal(data=noise, sample_rate=sr, duration_seconds=5.0, metadata=dummy_meta)
        features = extract_features(audio, config=app_config.features, scaler=scaler)

        rec_score = detector.predict_recording_score(features)
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=round(rec_score, 4),
            threshold=round(detector.threshold, 4),
            is_anomaly=bool(rec_score >= detector.threshold),
        )

        latents = detector.extract_latent_embeddings(features)
        fam = familiarity_estimator.predict(latents)

        result = evaluate_decision_policy(
            quality=quality,
            model_prediction=pred,
            familiarity=fam,
        )

        # White noise has severe domain shift compared to fan acoustic recordings
        assert fam.is_familiar is False
        assert result.status == DecisionStatus.NEEDS_REVIEW
        assert "differs from the training conditions" in result.explanation.lower()
