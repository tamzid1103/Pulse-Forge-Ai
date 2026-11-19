"""
Tests for curated demo audio files in demo_audio/ (Milestone M6).
Verifies that all 5 demonstration audio scenarios exist and produce their exact expected decisions.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config import AppConfig
from src.decision import DecisionStatus
from src.persistence import AnalysisStore
from src.service import PulseForgeService


@pytest.fixture(scope="module")
def service() -> PulseForgeService:
    cfg = AppConfig()
    store = AnalysisStore(":memory:")
    return PulseForgeService(config=cfg, store=store)


DEMO_EXPECTATIONS = [
    ("01_normal_fan.wav", DecisionStatus.NORMAL),
    ("02_bearing_anomaly.wav", DecisionStatus.POSSIBLE_ANOMALY),
    ("03_silent_sensor_failure.wav", DecisionStatus.NEEDS_REVIEW),
    ("04_clipped_sensor_saturation.wav", DecisionStatus.NEEDS_REVIEW),
    ("05_unfamiliar_operating_noise.wav", DecisionStatus.NEEDS_REVIEW),
]


@pytest.mark.parametrize("filename,expected_status", DEMO_EXPECTATIONS)
def test_demo_audio_scenarios(
    service: PulseForgeService, filename: str, expected_status: DecisionStatus
) -> None:
    cfg = AppConfig()
    demo_file = cfg.paths.project_root / "demo_audio" / filename

    assert demo_file.is_file(), f"Demo file not found: {demo_file}"

    result, sig, run_id = service.analyze_audio(
        demo_file, filename=filename, record_to_db=False
    )

    assert result.status == expected_status, (
        f"{filename} produced status {result.status.value}, expected {expected_status.value}. "
        f"Reason: {result.explanation}"
    )
