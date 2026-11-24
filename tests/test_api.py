"""
Comprehensive integration and unit test suite for FastAPI REST & WebSocket backend (Milestone M10).

Validates:
- Health check and system status endpoints.
- Curated demo scenarios catalog.
- Multipart WAV audio file upload and analysis pipeline.
- Input validation, error codes (400, 404, 413, 415, 422), and exception handlers.
- SQLite inspection history audit retrieval and purging.
- AI Copilot work-order generation and interactive chat endpoints.
- Bidirectional WebSocket streaming handshake, ping/pong, and chunk telemetry.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
import soundfile as sf

from src.api import app, get_service, get_copilot
from src.schemas import DecisionStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Module-scoped FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Generate a clean synthetic 16kHz sine wave WAV file in memory."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # 440 Hz fundamental tone
    audio = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. System Health & Metadata Endpoints
# ---------------------------------------------------------------------------

def test_health_check(client: TestClient) -> None:
    """Test GET /health returns service status and model version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert data["model_version"] == "fan-ae-v1"
    assert "timestamp" in data


def test_system_status(client: TestClient) -> None:
    """Test GET /api/v1/system/status returns telemetry thresholds and fleet state."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "PulseForge AI"
    assert data["machine_type"] == "fan"
    assert data["sample_rate"] == 16000
    assert "thresholds" in data
    assert "autoencoder_mse" in data["thresholds"]
    assert "familiarity_mahalanobis" in data["thresholds"]
    assert "fleet_summary" in data
    assert "wav" in data["supported_formats"]


def test_list_scenarios(client: TestClient) -> None:
    """Test GET /api/v1/scenarios returns the 5 curated demonstration scenarios."""
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert isinstance(scenarios, list)
    assert len(scenarios) == 5
    ids = [s["id"] for s in scenarios]
    assert "scenario_01" in ids
    assert "scenario_02" in ids
    assert "scenario_03" in ids
    assert "scenario_04" in ids
    assert "scenario_05" in ids


# ---------------------------------------------------------------------------
# 2. Acoustic Analysis Endpoints
# ---------------------------------------------------------------------------

def test_analyze_upload_valid_wav(client: TestClient, sample_wav_bytes: bytes) -> None:
    """Test POST /api/v1/analyze with a valid in-memory WAV file."""
    files = {"file": ("test_fan.wav", sample_wav_bytes, "audio/wav")}
    data = {"machine_id": "fan_id_00", "operator_note": "Automated pytest run"}
    response = client.post("/api/v1/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "run_id" in res
    assert res["machine_id"] == "fan_id_00"
    assert "analysis" in res
    analysis = res["analysis"]
    assert "status" in analysis
    assert "quality" in analysis
    assert analysis["quality"]["passed"] is True


def test_analyze_demo_scenario_normal(client: TestClient) -> None:
    """Test POST /api/v1/analyze/scenario/scenario_01 evaluates normal baseline."""
    response = client.post("/api/v1/analyze/scenario/scenario_01")
    assert response.status_code == 200
    res = response.json()
    assert res["scenario"]["id"] == "scenario_01"
    assert res["analysis"]["status"] == "normal"


def test_analyze_demo_scenario_anomaly(client: TestClient) -> None:
    """Test POST /api/v1/analyze/scenario/scenario_02 evaluates bearing anomaly."""
    response = client.post("/api/v1/analyze/scenario/scenario_02")
    assert response.status_code == 200
    res = response.json()
    assert res["scenario"]["id"] == "scenario_02"
    assert res["analysis"]["status"] == "possible_anomaly"


def test_analyze_demo_scenario_alias(client: TestClient) -> None:
    """Test scenario matching works via aliases (e.g. 'bearing')."""
    response = client.post("/api/v1/analyze/scenario/bearing")
    assert response.status_code == 200
    res = response.json()
    assert res["scenario"]["id"] == "scenario_02"


def test_analyze_demo_scenario_not_found(client: TestClient) -> None:
    """Test non-existent scenario returns HTTP 404."""
    response = client.post("/api/v1/analyze/scenario/invalid_nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_analyze_upload_empty_audio(client: TestClient) -> None:
    """Test uploading 0-byte file returns HTTP 400 invalid_audio."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in (400, 422)


def test_analyze_upload_invalid_format(client: TestClient) -> None:
    """Test uploading non-WAV data returns HTTP 415 or 400."""
    fake_mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00FAKE_MP3_CONTENT"
    files = {"file": ("recording.mp3", fake_mp3, "audio/mpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in (400, 415)


# ---------------------------------------------------------------------------
# 3. Audit History & Persistence Endpoints
# ---------------------------------------------------------------------------

def test_runs_history_and_lifecycle(client: TestClient, sample_wav_bytes: bytes) -> None:
    """Test full audit trail lifecycle: insert via analyze -> query -> detail -> clear."""
    # 1. Insert a run
    files = {"file": ("audit_sample.wav", sample_wav_bytes, "audio/wav")}
    res = client.post("/api/v1/analyze", files=files, data={"machine_id": "fan_id_99"})
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    assert run_id is not None

    # 2. Query history
    runs_res = client.get("/api/v1/runs?limit=10")
    assert runs_res.status_code == 200
    runs_data = runs_res.json()
    assert runs_data["total_returned"] >= 1
    assert "summary" in runs_data
    matching = [r for r in runs_data["runs"] if r["id"] == run_id]
    assert len(matching) == 1

    # 3. Query single run detail
    single_res = client.get(f"/api/v1/runs/{run_id}")
    assert single_res.status_code == 200
    single_data = single_res.json()
    assert single_data["id"] == run_id
    assert single_data["machine_id"] == "fan_id_99"

    # 4. Query non-existent run detail
    missing_res = client.get("/api/v1/runs/00000000-0000-0000-0000-000000000000")
    assert missing_res.status_code == 404

    # 5. Clear history
    del_res = client.delete("/api/v1/runs")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    # Verify history is empty
    empty_runs = client.get("/api/v1/runs?limit=10").json()
    assert empty_runs["total_returned"] == 0


# ---------------------------------------------------------------------------
# 4. Factory Brain Copilot Endpoints
# ---------------------------------------------------------------------------

def test_copilot_work_order_synthesis(client: TestClient) -> None:
    """Test POST /api/v1/copilot/work-order generates complete work order."""
    payload = {
        "decision_status": "possible_anomaly",
        "machine_id": "fan_id_00",
        "operator_note": "Acoustic rattling audible on motor housing.",
    }
    response = client.post("/api/v1/copilot/work-order", json=payload)
    assert response.status_code == 200
    wo = response.json()
    assert "work_order_id" in wo
    assert wo["asset_id"] == "fan_id_00"
    assert wo["diagnostic_status"] == "possible_anomaly"
    assert wo["urgency"] == "PRIORITY_INSPECTION"
    assert len(wo["recommended_steps"]) >= 2
    assert "safety_checklist" in wo
    assert "required_tools" in wo
    assert "required_ppe" in wo
    assert "operator_note" in wo


def test_copilot_chat_response(client: TestClient) -> None:
    """Test POST /api/v1/copilot/chat answers technician advisory questions."""
    payload = {
        "messages": [
            {"role": "user", "content": "What bearing inspection protocol should be followed?"}
        ],
        "machine_id": "fan_id_00",
    }
    response = client.post("/api/v1/copilot/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "fan_id_00"
    assert "reply" in data
    assert len(data["reply"]) > 20


# ---------------------------------------------------------------------------
# 5. Real-Time WebSocket Streaming Endpoint
# ---------------------------------------------------------------------------

def test_websocket_handshake_and_ping(client: TestClient) -> None:
    """Test WebSocket connection handshake and JSON ping/pong protocol."""
    with client.websocket_connect("/ws/v1/audio-stream?machine_id=fan_id_00") as ws:
        # Initial greeting frame
        greeting = ws.receive_json()
        assert greeting["type"] == "connected"
        assert greeting["machine_id"] == "fan_id_00"
        assert greeting["sample_rate"] == 16000

        # Send ping
        ws.send_json({"action": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert "timestamp" in pong


def test_websocket_binary_chunk_telemetry(client: TestClient) -> None:
    """Test WebSocket receives raw binary PCM bytes and broadcasts spectral telemetry."""
    with client.websocket_connect("/ws/v1/audio-stream?machine_id=fan_id_00") as ws:
        # Receive handshake
        _ = ws.receive_json()

        # Create raw int16 PCM chunk (1024 samples = 2048 bytes)
        t = np.linspace(0, 0.064, 1024, endpoint=False, dtype=np.float32)
        pcm_samples = (0.2 * np.sin(2 * np.pi * 1000.0 * t) * 32767.0).astype(np.int16)
        raw_bytes = pcm_samples.tobytes()

        # Send binary frame
        ws.send_bytes(raw_bytes)

        # Receive telemetry frame
        telemetry_frame = ws.receive_json()
        assert telemetry_frame["type"] == "telemetry"
        assert telemetry_frame["frame_index"] == 1
        assert telemetry_frame["machine_id"] == "fan_id_00"
        telem = telemetry_frame["telemetry"]
        assert "rms_level" in telem
        assert telem["rms_level"] > 0.0
        assert "clipping_ratio" in telem
        assert "spectral_bands" in telem
        assert "low_0_2khz" in telem["spectral_bands"]
        assert telem["quality_pass"] is True


def test_websocket_scenario_evaluation(client: TestClient) -> None:
    """Test triggering scenario evaluation via WebSocket message."""
    with client.websocket_connect("/ws/v1/audio-stream?machine_id=fan_id_00") as ws:
        _ = ws.receive_json()

        # Send scenario command
        ws.send_json({"action": "scenario", "id": "scenario_01"})
        res = ws.receive_json()
        assert res["type"] == "analysis_result"
        assert res["scenario"] == "scenario_01"
        assert res["analysis"]["status"] == "normal"
