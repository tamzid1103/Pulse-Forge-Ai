"""
Unit and smoke tests for the Streamlit dashboard components (Milestone M6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streamlit.testing.v1 import AppTest

from app.dashboard import plot_acoustic_features, get_service
from src.config import AppConfig

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"


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


def test_dashboard_equipment_selector_sync() -> None:
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=20).run()
    assert not at.exception
    assert at.session_state["active_machine_type"] == "fan"
    assert at.session_state["studio_machine_selector"] == "fan"

    # Switch to demo mode so curated machine-specific demo scenarios render
    at.radio(key="input_mode_radio").set_value("Try demo scenarios").run()
    assert not at.exception

    # Switching sidebar Equipment Family updates studio selector & demo scenarios
    at.sidebar.selectbox(key="selected_machine_type_box").select("pump").run()
    assert not at.exception
    assert at.session_state["active_machine_type"] == "pump"
    assert at.session_state["studio_machine_selector"] == "pump"
    assert any(
        "Centrifugal Pump normal" in opt for opt in at.radio(key="demo_scenario_radio_pump").options
    )

    # Switching to valve & slider also stays valid without exceptions
    at.sidebar.selectbox(key="selected_machine_type_box").select("valve").run()
    assert not at.exception
    assert at.session_state["active_machine_type"] == "valve"
    assert at.session_state["studio_machine_selector"] == "valve"
    assert any(
        "Control Valve normal" in opt for opt in at.radio(key="demo_scenario_radio_valve").options
    )
