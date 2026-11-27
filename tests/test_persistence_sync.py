"""
Unit and integration tests for Hybrid Supabase + SQLite Offline-First Persistence & Auto-Sync.

Uses deterministic in-memory SQLite / tmp_path and httpx.MockTransport (no live internet required).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import resolve_supabase_credentials, sanitize_secret_text, validate_supabase_url
from src.persistence import AnalysisStore
from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)


def _make_sample_result(
    status: DecisionStatus = DecisionStatus.NORMAL,
    machine_type: str = "fan",
    machine_id: str = "id_00",
    score: float = 0.42,
) -> AnalysisResult:
    return AnalysisResult(
        status=status,
        status_label="Normal Acoustic Operation" if status == DecisionStatus.NORMAL else "Anomaly",
        quality=QualityReport(
            passed=True,
            rms_level=0.04,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        ),
        model_prediction=ModelPrediction(
            model_type=f"{machine_type}-autoencoder-v1",
            anomaly_score=score,
            threshold=0.89,
            is_anomaly=(status == DecisionStatus.POSSIBLE_ANOMALY),
        ),
        familiarity=FamiliarityReport(score=2.5, threshold=9.25, is_familiar=True),
        metadata=AudioMetadata(
            sample_rate=16000,
            channels=1,
            duration_seconds=10.0,
            num_samples=160000,
            file_size_bytes=320044,
        ),
        explanation="Deterministic diagnostic check.",
        recommended_action="Routine monitoring.",
        machine_id=machine_id,
        machine_type=machine_type,
    )


class FakeSupabaseBackend:
    """In-memory mock of Supabase PostgREST table supporting online/offline toggling."""

    def __init__(self, secret_token: str = "sb_secret_TESTTOKEN_DO_NOT_LEAK") -> None:
        self.online: bool = True
        self.secret_token = secret_token
        self.rows_by_id: dict[str, dict[str, Any]] = {}
        self.upsert_calls: int = 0

    def build_client(self) -> httpx.Client:
        backend = self

        def handler(request: httpx.Request) -> httpx.Response:
            # Verify apikey header is always present, and Bearer is omitted for modern sb_secret_ keys
            assert request.headers.get("apikey") == backend.secret_token
            if backend.secret_token.startswith("eyJ"):
                assert request.headers.get("Authorization") == f"Bearer {backend.secret_token}"
            else:
                assert "Authorization" not in request.headers

            if not backend.online:
                raise httpx.ConnectError("Simulated plant Wi-Fi outage", request=request)

            if request.method == "GET" and "select=id&limit=1" in str(request.url):
                return httpx.Response(200, json=[])

            if request.method == "POST" and "on_conflict=id" in str(request.url):
                backend.upsert_calls += 1
                items = json.loads(request.content.decode("utf-8"))
                for item in items:
                    # Idempotent upsert keyed by UUID
                    backend.rows_by_id[item["id"]] = item
                return httpx.Response(201, json=[])

            if request.method == "GET" and "select=decision_status,processing_ms" in str(request.url):
                ordered = sorted(
                    backend.rows_by_id.values(),
                    key=lambda r: r.get("created_at", ""),
                    reverse=True,
                )
                return httpx.Response(
                    200,
                    json=[
                        {
                            "decision_status": r["decision_status"],
                            "processing_ms": r["processing_ms"],
                        }
                        for r in ordered
                    ],
                    headers={"Content-Range": f"0-{max(len(ordered)-1, 0)}/{len(ordered)}"},
                )

            if request.method == "GET":
                ordered = sorted(
                    backend.rows_by_id.values(),
                    key=lambda r: r.get("created_at", ""),
                    reverse=True,
                )
                return httpx.Response(200, json=ordered)

            if request.method == "DELETE":
                backend.rows_by_id.clear()
                return httpx.Response(204)

            return httpx.Response(404, json={"error": "not found"})

        return httpx.Client(transport=httpx.MockTransport(handler))


class TestSecurityAndConfigValidation:
    def test_strict_https_validation(self) -> None:
        assert validate_supabase_url("https://xyzabc.supabase.co/") == "https://xyzabc.supabase.co"
        assert validate_supabase_url("http://xyzabc.supabase.co") is None
        assert validate_supabase_url("file:///etc/passwd") is None
        assert validate_supabase_url("https://your-project-ref.supabase.co") is None
        assert validate_supabase_url("") is None

    def test_secret_sanitization_redacts_tokens(self) -> None:
        raw = (
            "Failed call with Bearer eyJhbGciOiJIUzI1NiJ9.payload12345.sig67890 "
            "and sb_secret_ABCDEF123456 and gsk_1234567890abcdef"
        )
        cleaned = sanitize_secret_text(raw)
        assert "payload12345" not in cleaned
        assert "ABCDEF123456" not in cleaned
        assert "1234567890abcdef" not in cleaned

    def test_credentials_resolver_rejects_placeholders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPABASE_URL", "https://your-project-ref.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "your_supabase_server_secret_key_here")
        url, key = resolve_supabase_credentials()
        assert url is None
        assert key is None


class TestHybridSupabaseOfflineSync:
    def test_online_default_offline_failover_and_auto_reconnect_sync(self, tmp_path) -> None:
        mock_sb = FakeSupabaseBackend()
        http_client = mock_sb.build_client()
        db_file = tmp_path / "hybrid_sync.db"

        store = AnalysisStore(
            db_path=db_file,
            supabase_url="https://demo-plant.supabase.co",
            supabase_key=mock_sb.secret_token,
            enable_cloud=True,
            probe_ttl_seconds=0.0,
            http_client=http_client,
        )

        # 1. Online default: saves directly to Supabase AND local SQLite
        id_online = store.save_analysis(
            _make_sample_result(DecisionStatus.NORMAL, "fan"),
            processing_ms=42,
            input_label="/var/tmp/secret/../../01_normal_fan.wav",
        )
        status = store.get_status()
        assert status["mode"] == "supabase"
        assert status["is_online"] is True
        assert status["pending_sync_count"] == 0
        assert id_online in mock_sb.rows_by_id
        # Input path traversal was stripped before cloud upload
        assert mock_sb.rows_by_id[id_online]["input_label"] == "01_normal_fan.wav"

        # 2. Internet drops -> Fails over smoothly to local SQLite
        mock_sb.online = False
        id_off_1 = store.save_analysis(
            _make_sample_result(DecisionStatus.POSSIBLE_ANOMALY, "pump", score=1.45),
            processing_ms=55,
            input_label="07_pump_anomaly.wav",
        )
        id_off_2 = store.save_analysis(
            _make_sample_result(DecisionStatus.NORMAL, "valve"),
            processing_ms=38,
            input_label="08_valve_normal.wav",
        )

        offline_status = store.get_status()
        assert offline_status["mode"] == "offline_sqlite"
        assert offline_status["label"] == "Offline Mode : Operating Locally"
        assert offline_status["is_online"] is False
        assert offline_status["pending_sync_count"] == 2
        assert mock_sb.secret_token not in str(offline_status)

        # Reads serve all 3 runs locally from SQLite while offline
        local_runs = store.get_recent_runs(limit=10)
        assert len(local_runs) == 3
        assert {r["id"] for r in local_runs} == {id_online, id_off_1, id_off_2}

        # 3. Internet returns -> Reconnection auto-uploads queued SQLite records
        mock_sb.online = True
        reconnected_status = store.sync_pending_runs(force=True)
        assert reconnected_status["mode"] == "supabase"
        assert reconnected_status["is_online"] is True
        assert reconnected_status["pending_sync_count"] == 0
        assert reconnected_status["last_synced_count"] == 2
        assert {id_online, id_off_1, id_off_2}.issubset(mock_sb.rows_by_id.keys())

        # Re-syncing is a clean no-op with 0 pending duplicates
        upserts_before = mock_sb.upsert_calls
        second_sync = store.sync_pending_runs(force=True)
        assert second_sync["last_synced_count"] == 0
        assert mock_sb.upsert_calls == upserts_before

        # Cloud summary stats reflect all synced inspections
        stats = store.get_summary_stats()
        assert stats["total_inspections"] == 3
        assert stats["status_counts"]["normal"] == 2
        assert stats["status_counts"]["possible_anomaly"] == 1

    def test_pre_sync_legacy_db_migrates_non_destructively(self, tmp_path) -> None:
        db_file = tmp_path / "legacy_18col.db"
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute(
                """
                CREATE TABLE analysis_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    machine_type TEXT NOT NULL CHECK (machine_type IN ('fan', 'pump', 'valve', 'slider')),
                    machine_id TEXT NOT NULL,
                    input_label TEXT,
                    source_kind TEXT NOT NULL,
                    audio_duration_seconds REAL NOT NULL,
                    sample_rate_hz INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    decision_status TEXT NOT NULL,
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
                    'legacy-uuid-1', '2026-11-26T00:00:00+00:00', 'fan', 'fan_id_00',
                    '01_normal_fan.wav', 'demo_sample', 10.0, 16000, 0.41, 0.89,
                    'normal', 'good', '[]', 2.0, '[]', 'fan-autoencoder-v1', 25, ''
                );
                """
            )
            conn.commit()

        store = AnalysisStore(db_path=db_file, enable_cloud=False)
        runs = store.get_recent_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["id"] == "legacy-uuid-1"
        assert runs[0]["sync_status"] == "synced"

    def test_api_persistence_and_sync_endpoints(self) -> None:
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert "persistence" in health
            assert "mode" in health["persistence"]

            runs_resp = client.get("/api/v1/runs").json()
            assert "persistence" in runs_resp

            sync_resp = client.post("/api/v1/runs/sync")
            assert sync_resp.status_code == 200
            sync_data = sync_resp.json()
            assert sync_data["status"] in ("synced", "offline")
            assert "persistence" in sync_data
