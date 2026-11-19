"""
Unit and smoke tests for the Streamlit dashboard components (Milestone M6).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dashboard import plot_acoustic_features, get_service
from src.config import AppConfig


def test_service_initialization() -> None:
    service = get_service()
    assert service is not None
    assert service.is_model_ready is True
    assert service.detector is not None


def test_acoustic_feature_plotting() -> None:
    cfg = AppConfig()
    sr = cfg.audio.sample_rate
    # 2 seconds of synthetic audio
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    synthetic_audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    fig = plot_acoustic_features(synthetic_audio, sr)
    assert fig is not None
    assert len(fig.axes) == 3  # ax1 (waveform), ax2 (spectrogram), cbar_ax
