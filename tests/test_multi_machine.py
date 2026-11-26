"""
Unit and integration tests for Milestone M12: Multi-Machine Industrial Fleet Extension
(Centrifugal Pumps, Industrial Valves, and Linear Slide Rails alongside Fans).
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app
from src.config import AppConfig, MACHINE_FAMILY_METADATA
from src.copilot import IndustrialCopilot
from src.mcp_server import create_server
from src.models.registry import ModelRegistry
from src.persistence import AnalysisStore
from src.schemas import AnalysisResult, DecisionStatus, ModelPrediction, QualityReport
from src.service import PulseForgeService

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_wav_bytes(freq: float = 440.0, duration: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = 0.25 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestMultiMachineConfigAndMigration:
    """Verify config metadata and non-destructive SQLite schema migration."""

    def test_supported_machine_families_and_metadata(self) -> None:
        cfg = AppConfig()
        assert cfg.supported_machine_types == ("fan", "pump", "valve", "slider")
        for mt in cfg.supported_machine_types:
            assert mt in MACHINE_FAMILY_METADATA
            assert "display_name" in MACHINE_FAMILY_METADATA[mt]
            assert "inspection_focus" in MACHINE_FAMILY_METADATA[mt]

    def test_sqlite_migrates_legacy_fan_only_schema(self, tmp_path: Path) -> None:
        db_file = tmp_path / "legacy_m6.db"
        # Create legacy M6 database with strict fan-only CHECK constraint
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute(
                """
                CREATE TABLE analysis_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    machine_type TEXT NOT NULL CHECK (machine_type = 'fan'),
                    machine_id TEXT NOT NULL,
                    input_label TEXT,
                    source_kind TEXT NOT NULL CHECK (source_kind IN ('uploaded', 'demo_sample', 'synthetic')),
                    audio_duration_seconds REAL NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    decision_status TEXT NOT NULL CHECK (
                        decision_status IN ('normal', 'possible_anomaly', 'needs_review', 'invalid_input')
                    ),
                    quality_status TEXT NOT NULL,
                    quality_reasons_json TEXT,
                    unfamiliarity_score REAL,
                    uncertainty_reasons_json TEXT,
                    model_version TEXT NOT NULL,
                    processing_ms INTEGER,
                    operator_note TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO analysis_runs VALUES (
                    'legacy-row-1', '2026-11-19T00:00:00Z', 'fan', 'id_00', 'old_fan.wav',
                    'demo_sample', 10.0, 16000, 0.45, 0.89, 'normal', 'good', '[]', 3.2, '[]',
                    'autoencoder-v1', 25, ''
                );
                """
            )
            conn.commit()

        # Re-open with M12 AnalysisStore (should migrate automatically)
        store = AnalysisStore(db_path=db_file)
        assert store.get_run_by_id("legacy-row-1") is not None

        # Verify non-fan machine types can now be inserted cleanly
        for mt in ("pump", "valve", "slider"):
            res = AnalysisResult(
                status=DecisionStatus.NORMAL,
                status_label="Normal",
                quality=QualityReport(
                    passed=True,
                    rms_level=0.04,
                    is_silent=False,
                    clipping_ratio=0.0,
                    is_clipped=False,
                    duration_seconds=10.0,
                    is_duration_valid=True,
                ),
                machine_id="id_00",
                machine_type=mt,
            )
            rid = store.save_analysis(res, input_label=f"{mt}.wav")
            fetched = store.get_run_by_id(rid)
            assert fetched is not None
            assert fetched["machine_type"] == mt


class TestMultiMachineRegistryAndService:
    """Verify ModelRegistry loading and PulseForgeService dynamic routing."""

    def test_model_registry_loads_all_four_families(self) -> None:
        cfg = AppConfig()
        registry = ModelRegistry(
            model_dir=cfg.paths.model_artifacts_dir,
            supported_types=cfg.supported_machine_types,
        )
        available = registry.available_machine_types()
        assert available == ["fan", "pump", "valve", "slider"]
        assert registry.get_bundle("unknown_machine") is None

        summary = registry.get_status_summary()
        for mt in ("fan", "pump", "valve", "slider"):
            assert summary[mt]["model_ready"] is True
            assert summary[mt]["autoencoder_threshold"] > 0.0
            assert summary[mt]["familiarity_threshold"] > 0.0

    def test_service_rejects_unsupported_machine_type(self) -> None:
        svc = PulseForgeService(store=AnalysisStore(":memory:"))
        wav_bytes = _make_wav_bytes()
        res, _, rid = svc.analyze_audio(
            source=wav_bytes,
            filename="turbine.wav",
            machine_type="turbine",
            record_to_db=True,
        )
        assert res.status == DecisionStatus.INVALID_INPUT
        assert "unsupported machine type" in res.explanation.lower()
        assert rid is None

    @pytest.mark.parametrize(
        "machine_type,demo_file,expected",
        [
            ("fan", "01_normal_fan.wav", DecisionStatus.NORMAL),
            ("fan", "02_bearing_anomaly.wav", DecisionStatus.POSSIBLE_ANOMALY),
            ("pump", "06_pump_normal.wav", DecisionStatus.NORMAL),
            ("pump", "07_pump_anomaly.wav", DecisionStatus.POSSIBLE_ANOMALY),
            ("valve", "08_valve_normal.wav", DecisionStatus.NORMAL),
            ("valve", "09_valve_anomaly.wav", DecisionStatus.POSSIBLE_ANOMALY),
            ("slider", "10_slider_normal.wav", DecisionStatus.NORMAL),
            ("slider", "11_slider_anomaly.wav", DecisionStatus.POSSIBLE_ANOMALY),
            ("pump", "05_unfamiliar_operating_noise.wav", DecisionStatus.NEEDS_REVIEW),
            ("valve", "05_unfamiliar_operating_noise.wav", DecisionStatus.NEEDS_REVIEW),
            ("slider", "05_unfamiliar_operating_noise.wav", DecisionStatus.NEEDS_REVIEW),
        ],
    )
    def test_demo_inference_across_all_machine_families(
        self, machine_type: str, demo_file: str, expected: DecisionStatus
    ) -> None:
        wav_path = PROJECT_ROOT / "demo_audio" / demo_file
        if not wav_path.is_file():
            pytest.skip(f"Demo file missing: {wav_path}")

        svc = PulseForgeService(store=AnalysisStore(":memory:"))
        res, _, _ = svc.analyze_audio(
            source=wav_path,
            filename=demo_file,
            machine_id="id_00",
            machine_type=machine_type,
            record_to_db=False,
        )
        assert res.status == expected
        assert res.machine_type == machine_type


class TestMultiMachineCopilotAPIAndMCP:
    """Verify Copilot SOPs, FastAPI endpoints, and MCP multi-machine integration."""

    @pytest.mark.parametrize(
        "machine_type,expected_prefix,keyword",
        [
            ("pump", "WO-PUMP-", "cavitation"),
            ("valve", "WO-VALVE-", "seat"),
            ("slider", "WO-SLDR-", "ball-screw"),
        ],
    )
    def test_copilot_machine_specific_work_orders(
        self, machine_type: str, expected_prefix: str, keyword: str
    ) -> None:
        copilot = IndustrialCopilot(api_key=None)
        res = AnalysisResult(
            status=DecisionStatus.POSSIBLE_ANOMALY,
            status_label="Possible Acoustic Anomaly",
            quality=QualityReport(
                passed=True,
                rms_level=0.05,
                is_silent=False,
                clipping_ratio=0.0,
                is_clipped=False,
                duration_seconds=10.0,
                is_duration_valid=True,
            ),
            model_prediction=ModelPrediction(
                model_type=f"{machine_type}-autoencoder-v1",
                anomaly_score=1.85,
                threshold=0.90,
                is_anomaly=True,
            ),
            machine_id="id_00",
            machine_type=machine_type,
        )
        wo = copilot.generate_work_order(res, machine_id=f"{machine_type}_id_00")
        assert wo["work_order_id"].startswith(expected_prefix)
        assert wo["machine_type"] == machine_type
        assert keyword in wo["suspected_mechanism"].lower()

    def test_api_fleet_endpoints_and_analysis(self) -> None:
        with TestClient(app) as client:
            status_resp = client.get("/api/v1/system/status")
            assert status_resp.status_code == 200
            s_data = status_resp.json()
            assert s_data["supported_machine_types"] == ["fan", "pump", "valve", "slider"]
            assert "pump" in s_data["model_registry"]

            fleet_resp = client.get("/api/v1/fleet/scenarios")
            assert fleet_resp.status_code == 200
            fleet_scenarios = fleet_resp.json()
            assert len(fleet_scenarios) == 11

            pump_scen_resp = client.post("/api/v1/analyze/scenario/scenario_06")
            assert pump_scen_resp.status_code == 200
            assert pump_scen_resp.json()["analysis"]["machine_type"] == "pump"
            assert pump_scen_resp.json()["analysis"]["status"] == "normal"

    @pytest.mark.anyio
    async def test_mcp_fleet_status_and_pump_analysis(self) -> None:
        server = create_server()
        contents = await server.read_resource("pulseforge://fleet/status")
        text = contents[0].content
        assert "pump_id_00" in text
        assert "valve_id_00" in text
        assert "slider_id_00" in text

        pump_wav = PROJECT_ROOT / "demo_audio" / "06_pump_normal.wav"
        if pump_wav.is_file():
            res = await server.call_tool(
                "analyze_fan_audio",
                {
                    "audio_path": str(pump_wav),
                    "machine_id": "pump_id_00",
                    "machine_type": "pump",
                },
            )
            assert not res.is_error
