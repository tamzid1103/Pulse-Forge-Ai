"""
Unit tests for the Factory Brain AI Copilot (Milestone M8).

Validates:
- Knowledge base loading from disk.
- Telemetry prompt formatting from AnalysisResult.
- Deterministic work order generation for all 4 decision states.
- API key resolution hierarchy.
- Mocked Groq LLM API invocation (ensuring 100% offline pass).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from src.copilot import IndustrialCopilot
from src.decision import DecisionStatus
from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)


@pytest.fixture
def sample_normal_result() -> AnalysisResult:
    return AnalysisResult(
        status=DecisionStatus.NORMAL,
        status_label="Normal",
        quality=QualityReport(
            passed=True,
            rms_level=0.035,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        ),
        model_prediction=ModelPrediction(
            model_type="autoencoder",
            anomaly_score=0.0021,
            threshold=0.0032,
            is_anomaly=False,
            segment_scores=[0.0020, 0.0022],
        ),
        familiarity=FamiliarityReport(
            score=4.12,
            threshold=9.25,
            is_familiar=True,
        ),
        explanation="Normal acoustic operation confirmed.",
        recommended_action="Continue standard monitoring.",
        machine_id="fan_id_00",
    )


@pytest.fixture
def sample_anomaly_result() -> AnalysisResult:
    return AnalysisResult(
        status=DecisionStatus.POSSIBLE_ANOMALY,
        status_label="Possible Anomaly",
        quality=QualityReport(
            passed=True,
            rms_level=0.042,
            is_silent=False,
            clipping_ratio=0.0001,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        ),
        model_prediction=ModelPrediction(
            model_type="autoencoder",
            anomaly_score=0.0048,
            threshold=0.0032,
            is_anomaly=True,
            segment_scores=[0.0035, 0.0062],
        ),
        familiarity=FamiliarityReport(
            score=5.80,
            threshold=9.25,
            is_familiar=True,
        ),
        explanation="Acoustic anomaly detected. Reconstruction error exceeds threshold.",
        recommended_action="Inspect bearing assembly and belt tension.",
        machine_id="fan_id_00",
    )


@pytest.fixture
def sample_review_result() -> AnalysisResult:
    return AnalysisResult(
        status=DecisionStatus.NEEDS_REVIEW,
        status_label="Needs Review",
        quality=QualityReport(
            passed=False,
            rms_level=0.0002,
            is_silent=True,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        ),
        model_prediction=None,
        familiarity=None,
        explanation="Sensor signal near silence. Quality check failed.",
        recommended_action="Inspect acoustic sensor wiring.",
        machine_id="fan_id_00",
    )


def test_copilot_knowledge_base_loading() -> None:
    copilot = IndustrialCopilot()
    kb = copilot.load_knowledge_base()
    assert isinstance(kb, str)
    assert len(kb) > 100
    assert "ISO 10816" in kb
    assert "Lockout" in kb


def test_copilot_build_telemetry_prompt(sample_anomaly_result: AnalysisResult) -> None:
    copilot = IndustrialCopilot()
    prompt = copilot.build_telemetry_prompt(sample_anomaly_result, machine_id="fan_id_00")
    assert "POSSIBLE_ANOMALY" in prompt
    assert "0.004800" in prompt
    assert "fan_id_00" in prompt


def test_deterministic_work_order_normal(sample_normal_result: AnalysisResult) -> None:
    copilot = IndustrialCopilot(api_key=None)
    wo = copilot.generate_work_order(sample_normal_result, machine_id="fan_id_00")
    assert wo["diagnostic_status"] == "normal"
    assert wo["urgency"] == "ROUTINE"
    assert len(wo["recommended_steps"]) >= 2
    assert "required_tools" in wo
    assert "required_ppe" in wo


def test_deterministic_work_order_anomaly(sample_anomaly_result: AnalysisResult) -> None:
    copilot = IndustrialCopilot(api_key=None)
    wo = copilot.generate_work_order(sample_anomaly_result, machine_id="fan_id_00")
    assert wo["diagnostic_status"] == "possible_anomaly"
    assert wo["urgency"] == "PRIORITY_INSPECTION"
    assert "Zone C" in wo["iso_severity_zone"]
    assert any("LOTO" in step or "Lockout" in step for step in wo["safety_checklist"])


def test_deterministic_work_order_needs_review(sample_review_result: AnalysisResult) -> None:
    copilot = IndustrialCopilot(api_key=None)
    wo = copilot.generate_work_order(sample_review_result, machine_id="fan_id_00")
    assert wo["diagnostic_status"] == "needs_review"
    assert "sensor" in wo["suspected_mechanism"].lower()


def test_deterministic_chat_response(sample_anomaly_result: AnalysisResult) -> None:
    copilot = IndustrialCopilot(api_key=None)
    resp = copilot.chat_response(
        messages=[{"role": "user", "content": "What bearing inspection is needed?"}],
        result=sample_anomaly_result,
    )
    assert "Bearing Inspection Protocol" in resp
    assert "Lockout/Tagout" in resp


def test_api_key_resolution_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    copilot = IndustrialCopilot(api_key="manual_key_123")
    assert copilot.get_api_key() == "manual_key_123"

    # Argument override takes highest precedence
    assert copilot.get_api_key(override_key="override_key_456") == "override_key_456"

    # When manual is None, falls back to env
    copilot_env = IndustrialCopilot(api_key=None)
    monkeypatch.setenv("GROQ_API_KEY", "env_gsk_789")
    assert copilot_env.get_api_key() == "env_gsk_789"


@patch("src.copilot.Groq")
def test_mocked_groq_work_order(mock_groq_cls: MagicMock, sample_anomaly_result: AnalysisResult) -> None:
    mock_client = MagicMock()
    mock_groq_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_work_order_json = {
        "work_order_id": "WO-FAN-2026-9999",
        "asset_id": "fan_id_00",
        "diagnostic_status": "possible_anomaly",
        "urgency": "PRIORITY_INSPECTION",
        "suspected_mechanism": "High-frequency bearing defect",
        "iso_severity_zone": "Zone C",
        "recommended_steps": ["Inspect inboard bearing", "Measure temperature"],
        "safety_checklist": ["Lockout/Tagout 480V MCC", "Zero-speed confirmation"],
        "required_tools": ["Vibration pen", "IR thermometer"],
        "required_ppe": ["Safety glasses", "Hearing protection"],
        "engineering_summary": "Reconstruction error elevated; immediate inspection required."
    }
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(mock_work_order_json)))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    copilot = IndustrialCopilot(api_key="gsk_mock_test_key")
    wo = copilot.generate_work_order(sample_anomaly_result, machine_id="fan_id_00")

    assert wo["work_order_id"] == "WO-FAN-2026-9999"
    assert wo["urgency"] == "PRIORITY_INSPECTION"
    assert "Groq LPU" in wo["ai_engine"]
    mock_client.chat.completions.create.assert_called_once()
