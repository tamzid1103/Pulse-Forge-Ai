"""
SQLite metadata audit persistence for PulseForge AI (Milestone M6).

Implements the lightweight audit trail defined in docs/DATABASE_SCHEMA.md.
Persists analysis metadata, scores, quality flags, and decisions.
CRITICAL: Never persists raw audio data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_default_config
from src.schemas import AnalysisResult, DecisionStatus

logger = logging.getLogger("pulseforge.persistence")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_runs (
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

CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
ON analysis_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_machine_id
ON analysis_runs(machine_id);
"""


class AnalysisStore:
    """Embedded SQLite persistence manager for inspection run records."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._mem_conn: sqlite3.Connection | None = None
        if db_path is None:
            cfg = get_default_config()
            self.db_path = cfg.paths.data_dir / "pulseforge.db"
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = Path(db_path)

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database tables, concurrency pragmas, and indexes."""
        with self._get_connection() as conn:
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
            conn.executescript(CREATE_TABLE_SQL)
            conn.commit()

    def save_analysis(
        self,
        result: AnalysisResult,
        processing_ms: int = 0,
        source_kind: str = "uploaded",
        input_label: str = "audio.wav",
        operator_note: str = "",
    ) -> str:
        """
        Persist an AnalysisResult into the SQLite audit log.
        
        Returns:
            Unique record ID (UUID).
        """
        record_id = str(uuid.uuid4())
        created_at = result.timestamp or datetime.now(timezone.utc).isoformat()

        # Quality status derivation
        if result.quality.passed:
            quality_status = "good"
        elif result.quality.warnings and not result.quality.errors:
            quality_status = "warning"
        else:
            quality_status = "poor"

        quality_reasons = result.quality.errors + result.quality.warnings
        quality_reasons_json = json.dumps(quality_reasons)

        anomaly_score = (
            result.model_prediction.anomaly_score
            if result.model_prediction is not None
            else 0.0
        )
        threshold = (
            result.model_prediction.threshold
            if result.model_prediction is not None
            else 0.0
        )

        unfamiliarity_score = (
            result.familiarity.score if result.familiarity is not None else None
        )
        uncertainty_reasons: list[str] = []
        if result.familiarity and not result.familiarity.is_familiar:
            uncertainty_reasons.append("Latent acoustic unfamiliarity detected.")
        uncertainty_reasons_json = json.dumps(uncertainty_reasons)

        duration = (
            result.metadata.duration_seconds
            if result.metadata
            else result.quality.duration_seconds
        )
        sr = result.metadata.sample_rate if result.metadata else 16000
        model_version = (
            result.model_prediction.model_type
            if result.model_prediction
            else "autoencoder-v1"
        )

        # Ensure source_kind is valid
        if source_kind not in ("uploaded", "demo_sample", "synthetic"):
            source_kind = "uploaded"

        query = """
        INSERT INTO analysis_runs (
            id, created_at, machine_type, machine_id, input_label, source_kind,
            audio_duration_seconds, sample_rate_hz, anomaly_score, threshold,
            decision_status, quality_status, quality_reasons_json, unfamiliarity_score,
            uncertainty_reasons_json, model_version, processing_ms, operator_note
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        params = (
            record_id,
            created_at,
            result.machine_type,
            result.machine_id,
            input_label,
            source_kind,
            float(duration),
            int(sr),
            float(anomaly_score),
            float(threshold),
            result.status.value,
            quality_status,
            quality_reasons_json,
            unfamiliarity_score,
            uncertainty_reasons_json,
            model_version,
            int(processing_ms),
            operator_note,
        )

        with self._get_connection() as conn:
            conn.execute(query, params)
            conn.commit()

        logger.info(
            "Saved analysis run %s (status: %s, label: %s)",
            record_id,
            result.status.value,
            input_label,
        )
        return record_id

    def get_recent_runs(
        self,
        limit: int = 10,
        machine_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve recent analysis runs ordered by newest first."""
        base_query = "SELECT * FROM analysis_runs"
        params: list[Any] = []
        if machine_id:
            base_query += " WHERE machine_id = ?"
            params.append(machine_id)
        base_query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["quality_reasons"] = json.loads(d["quality_reasons_json"] or "[]")
            d["uncertainty_reasons"] = json.loads(d["uncertainty_reasons_json"] or "[]")
            results.append(d)
        return results

    def get_run_by_id(self, record_id: str) -> dict[str, Any] | None:
        """Fetch a single analysis run by primary key."""
        query = "SELECT * FROM analysis_runs WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (record_id,))
            row = cursor.fetchone()

        if row is None:
            return None
        d = dict(row)
        d["quality_reasons"] = json.loads(d["quality_reasons_json"] or "[]")
        d["uncertainty_reasons"] = json.loads(d["uncertainty_reasons_json"] or "[]")
        return d

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute aggregated statistics for telemetry dashboards."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
            status_counts = conn.execute(
                "SELECT decision_status, COUNT(*) as cnt FROM analysis_runs GROUP BY decision_status"
            ).fetchall()
            avg_latency = conn.execute(
                "SELECT AVG(processing_ms) FROM analysis_runs WHERE processing_ms > 0"
            ).fetchone()[0]

        counts = {
            "normal": 0,
            "possible_anomaly": 0,
            "needs_review": 0,
            "invalid_input": 0,
        }
        for row in status_counts:
            counts[row["decision_status"]] = row["cnt"]

        return {
            "total_inspections": total,
            "status_counts": counts,
            "average_latency_ms": round(avg_latency, 1) if avg_latency else 0.0,
        }

    def clear_history(self) -> None:
        """Purge all records (for testing or reset)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_runs")
            conn.commit()
