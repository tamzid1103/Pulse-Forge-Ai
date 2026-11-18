"""
Unit tests for signal quality assessment (Milestone M5).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import QualityConfig
from src.quality import (
    check_signal_quality,
    compute_clipping_ratio,
    compute_dc_offset,
    compute_rms,
)


class TestQualityMetrics:
    """Test raw metric calculations."""

    def test_compute_rms_silence(self) -> None:
        zero_signal = np.zeros(16000, dtype=np.float32)
        assert compute_rms(zero_signal) == 0.0

    def test_compute_rms_sine(self) -> None:
        # RMS of sine wave with peak amplitude A is A / sqrt(2) ~ 0.7071 * A
        t = np.linspace(0, 1, 16000, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms = compute_rms(sine)
        assert pytest.approx(rms, rel=1e-2) == 1.0 / np.sqrt(2.0)

    def test_compute_clipping_ratio(self) -> None:
        # Half clipped at 1.0
        signal = np.array([0.1, 0.5, 1.0, 1.0, -1.0, -1.0, 0.2, 0.3], dtype=np.float32)
        ratio = compute_clipping_ratio(signal, threshold=0.999)
        assert ratio == 4.0 / 8.0 == 0.5

    def test_compute_dc_offset(self) -> None:
        signal = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        assert pytest.approx(compute_dc_offset(signal), rel=1e-3) == 0.4


class TestSignalQualityChecks:
    """Test comprehensive signal quality assessment logic."""

    @pytest.fixture
    def default_config(self) -> QualityConfig:
        return QualityConfig()

    def test_clean_audio_passes(self, default_config: QualityConfig) -> None:
        # 2 seconds of 440Hz sine wave at amplitude 0.2
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        clean = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        report = check_signal_quality(clean, sample_rate=sr, config=default_config)
        assert report.passed is True
        assert report.is_silent is False
        assert report.is_clipped is False
        assert report.is_duration_valid is True
        assert len(report.errors) == 0

    def test_silent_audio_fails(self, default_config: QualityConfig) -> None:
        sr = 16000
        silent = np.zeros(sr * 2, dtype=np.float32)

        report = check_signal_quality(silent, sample_rate=sr, config=default_config)
        assert report.passed is False
        assert report.is_silent is True
        assert any("silent" in err.lower() for err in report.errors)

    def test_severely_clipped_audio_fails(self, default_config: QualityConfig) -> None:
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Saturated sine wave exceeding +/- 1.0, clipped at boundary
        signal = np.clip(5.0 * np.sin(2 * np.pi * 100 * t), -1.0, 1.0).astype(np.float32)

        report = check_signal_quality(signal, sample_rate=sr, config=default_config)
        assert report.is_clipped is True
        assert report.passed is False
        assert any("clipping" in w.lower() for w in report.warnings)

    def test_too_short_audio_fails(self, default_config: QualityConfig) -> None:
        sr = 16000
        # 0.2 seconds is shorter than min_analysis_duration (0.5s)
        short_signal = 0.1 * np.ones(int(0.2 * sr), dtype=np.float32)

        report = check_signal_quality(short_signal, sample_rate=sr, config=default_config)
        assert report.passed is False
        assert report.is_duration_valid is False
        assert any("shorter than minimum" in err.lower() for err in report.errors)

    def test_too_long_audio_fails(self, default_config: QualityConfig) -> None:
        sr = 16000
        # 35 seconds exceeds max_analysis_duration (30.0s)
        long_signal = 0.1 * np.ones(int(35.0 * sr), dtype=np.float32)

        report = check_signal_quality(long_signal, sample_rate=sr, config=default_config)
        assert report.passed is False
        assert report.is_duration_valid is False
        assert any("exceeds maximum" in err.lower() for err in report.errors)

    def test_nan_or_inf_fails(self, default_config: QualityConfig) -> None:
        sr = 16000
        corrupt = np.ones(sr * 2, dtype=np.float32)
        corrupt[100] = np.nan

        report = check_signal_quality(corrupt, sample_rate=sr, config=default_config)
        assert report.passed is False
        assert any("nan" in err.lower() for err in report.errors)

    def test_dc_bias_warning(self, default_config: QualityConfig) -> None:
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Sine wave with large DC bias (+0.2)
        signal = (0.2 + 0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        report = check_signal_quality(signal, sample_rate=sr, config=default_config)
        assert abs(report.dc_offset) >= default_config.dc_offset_threshold
        assert any("dc bias" in w.lower() for w in report.warnings)
