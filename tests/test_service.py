"""
Unit and integration tests for core application service (Milestone M6).
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.config import AppConfig
from src.persistence import AnalysisStore
from src.schemas import DecisionStatus
from src.service import PulseForgeService


@pytest.fixture(scope="module")
def app_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def service(app_config: AppConfig) -> PulseForgeService:
    store = AnalysisStore(db_path=":memory:")
    return PulseForgeService(config=app_config, store=store)


def _create_wav_bytes(signal: np.ndarray, sr: int = 16000) -> bytes:
    """Helper to encode a numpy waveform into genuine WAV in-memory bytes."""
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestPulseForgeService:
    """Test full service lifecycle and error handling."""

    def test_service_initialization(self, service: PulseForgeService) -> None:
        status = service.get_model_status()
        assert status["model_ready"] is True
        assert status["autoencoder_loaded"] is True
        assert status["scaler_loaded"] is True
        assert status["familiarity_loaded"] is True
        assert status["autoencoder_threshold"] > 0.0

    def test_analyze_real_normal_audio(
        self, service: PulseForgeService, app_config: AppConfig
    ) -> None:
        audio_path = app_config.paths.data_dir / "raw" / "id_00" / "normal" / "00000000.wav"
        if not audio_path.is_file():
            pytest.skip(f"MIMII normal file not found: {audio_path}")

        result, signal, run_id = service.analyze_audio(
            source=audio_path,
            filename="00000000.wav",
            source_kind="demo_sample",
        )
        assert result.status == DecisionStatus.NORMAL
        assert result.quality.passed is True
        assert result.model_prediction is not None
        assert result.model_prediction.is_anomaly is False
        assert signal is not None
        assert run_id is not None

        # Verify record in SQLite store
        record = service.store.get_run_by_id(run_id)
        assert record is not None
        assert record["decision_status"] == "normal"

    def test_analyze_real_abnormal_audio(
        self, service: PulseForgeService, app_config: AppConfig
    ) -> None:
        audio_path = app_config.paths.data_dir / "raw" / "id_00" / "abnormal" / "00000009.wav"
        if not audio_path.is_file():
            pytest.skip(f"MIMII abnormal file not found: {audio_path}")

        result, signal, run_id = service.analyze_audio(
            source=audio_path,
            filename="00000009.wav",
            source_kind="demo_sample",
        )
        assert result.status == DecisionStatus.POSSIBLE_ANOMALY
        assert result.quality.passed is True
        assert result.model_prediction is not None
        assert result.model_prediction.is_anomaly is True

    def test_analyze_corrupt_bytes_yields_invalid_input(
        self, service: PulseForgeService
    ) -> None:
        corrupt_bytes = b"this is not a valid RIFF WAV audio stream"
        result, signal, run_id = service.analyze_audio(
            source=corrupt_bytes,
            filename="corrupt.wav",
        )
        assert result.status == DecisionStatus.INVALID_INPUT
        assert signal is None
        assert run_id is not None

    def test_analyze_silent_audio_yields_needs_review(
        self, service: PulseForgeService
    ) -> None:
        sr = 16000
        silent = np.zeros(sr * 3, dtype=np.float32)
        wav_bytes = _create_wav_bytes(silent, sr)

        result, signal, run_id = service.analyze_audio(
            source=wav_bytes,
            filename="silent.wav",
        )
        assert result.status == DecisionStatus.NEEDS_REVIEW
        assert result.quality.is_silent is True

    def test_analyze_clipped_audio_yields_needs_review(
        self, service: PulseForgeService
    ) -> None:
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        clipped = np.clip(10.0 * np.sin(2 * np.pi * 100 * t), -1.0, 1.0).astype(np.float32)
        wav_bytes = _create_wav_bytes(clipped, sr)

        result, signal, run_id = service.analyze_audio(
            source=wav_bytes,
            filename="clipped.wav",
        )
        assert result.status == DecisionStatus.NEEDS_REVIEW
        assert result.quality.is_clipped is True
