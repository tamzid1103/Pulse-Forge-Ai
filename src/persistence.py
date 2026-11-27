"""
Hybrid Cloud + Local Audit Persistence for PulseForge AI (M6 -> M12 -> Supabase Offline-First).

Implements metadata audit persistence conforming to docs/DATABASE_SCHEMA.md:
- Default primary backend (when configured & online): Supabase Postgres over HTTPS.
- Automatic offline failover: When network/Supabase is unavailable, operates seamlessly
  on embedded local SQLite (`sync_status = 'pending'`) and reports "Offline Mode : Operating Locally".
- Automatic reconnection sync: When Supabase connectivity is restored, batch-upserts
  pending local SQLite records idempotently by UUID primary key (`on_conflict=id`).
- CRITICAL: Strictly stores inspection metadata only — never raw audio data or secrets.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from src.config import (
    get_default_config,
    resolve_supabase_credentials,
    sanitize_secret_text,
    validate_supabase_url,
)
from src.schemas import AnalysisResult

logger = logging.getLogger("pulseforge.persistence")

CORE_COLUMNS: tuple[str, ...] = (
    "id",
    "created_at",
    "machine_type",
    "machine_id",
    "input_label",
    "source_kind",
    "audio_duration_seconds",
    "sample_rate_hz",
    "anomaly_score",
    "threshold",
    "decision_status",
    "quality_status",
    "quality_reasons_json",
    "unfamiliarity_score",
    "uncertainty_reasons_json",
    "model_version",
    "processing_ms",
    "operator_note",
)

CORE_COLUMNS_CSV = ", ".join(CORE_COLUMNS)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    machine_type TEXT NOT NULL CHECK (machine_type IN ('fan', 'pump', 'valve', 'slider')),
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
    operator_note TEXT,
    sync_status TEXT NOT NULL DEFAULT 'synced' CHECK (sync_status IN ('synced', 'pending')),
    synced_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
ON analysis_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_machine_id
ON analysis_runs(machine_id);
"""


def _sanitize_label(label: str | None) -> str:
    """Strip directory paths from filenames and redact accidental tokens."""
    if not label:
        return "audio.wav"
    clean = Path(str(label)).name.strip() or "audio.wav"
    return sanitize_secret_text(clean)[:255]


def _row_to_supabase_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Extract only the 18 canonical metadata columns expected by Supabase Postgres."""
    payload: dict[str, Any] = {}
    for col in CORE_COLUMNS:
        payload[col] = row.get(col)
    payload["input_label"] = _sanitize_label(payload.get("input_label"))
    if payload.get("operator_note"):
        payload["operator_note"] = sanitize_secret_text(str(payload["operator_note"]))[:1000]
    return payload


def _hydrate_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Attach parsed JSON list fields and normalized sync metadata to a row dict."""
    d = dict(raw)
    try:
        d["quality_reasons"] = json.loads(d.get("quality_reasons_json") or "[]")
    except Exception:
        d["quality_reasons"] = []
    try:
        d["uncertainty_reasons"] = json.loads(d.get("uncertainty_reasons_json") or "[]")
    except Exception:
        d["uncertainty_reasons"] = []
    d.setdefault("sync_status", "synced")
    return d


class SupabasePostgRESTClient:
    """
    Lightweight, security-hardened HTTPS REST client for Supabase PostgREST.
    Never logs or leaks credentials in exceptions or object representations.
    """

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = "analysis_runs",
        timeout_seconds: float = 2.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        validated = validate_supabase_url(url)
        if not validated or not key.strip():
            raise ValueError("Invalid Supabase URL or empty API key.")
        self._base_url = f"{validated}/rest/v1/{table_name}"
        self._key = key.strip()
        self._timeout = float(timeout_seconds)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=self._timeout)

    def __repr__(self) -> str:
        return f"SupabasePostgRESTClient(endpoint={self._base_url!r})"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self._key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Only send Authorization Bearer for legacy JWT keys (eyJ...);
        # modern Supabase sb_secret_/sb_publishable_ keys are opaque and fail JWT parsing if put in Authorization.
        if self._key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self._key}"
        if extra:
            headers.update(extra)
        return headers

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def probe(self) -> tuple[bool, str | None]:
        """Verify internet + Supabase PostgREST table reachability cleanly."""
        try:
            resp = self._client.get(
                f"{self._base_url}?select=id&limit=1",
                headers=self._headers(),
                timeout=self._timeout,
            )
            if resp.status_code == 200:
                return True, None
            return False, f"Supabase returned HTTP {resp.status_code}"
        except (httpx.RequestError, OSError):
            return False, "Network unreachable or Supabase timeout"
        except Exception as exc:
            return False, sanitize_secret_text(exc)[:160]

    def upsert_rows(self, rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
        """Idempotently upload/merge records into Supabase keyed by UUID `id`."""
        if not rows:
            return True, None
        payload = [_row_to_supabase_payload(r) for r in rows]
        try:
            resp = self._client.post(
                f"{self._base_url}?on_conflict=id",
                headers=self._headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
                json=payload,
                timeout=max(self._timeout, 4.0),
            )
            if resp.status_code in (200, 201, 204):
                return True, None
            return False, f"Supabase upsert failed with HTTP {resp.status_code}"
        except (httpx.RequestError, OSError):
            return False, "Network unreachable during Supabase upload"
        except Exception as exc:
            return False, sanitize_secret_text(exc)[:160]

    def fetch_recent_runs(
        self, limit: int = 10, machine_id: str | None = None
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Fetch newest inspection runs from Supabase."""
        safe_limit = max(1, min(1000, int(limit)))
        url = f"{self._base_url}?select=*&order=created_at.desc&limit={safe_limit}"
        if machine_id:
            url += f"&machine_id=eq.{quote(str(machine_id), safe='')}"
        try:
            resp = self._client.get(url, headers=self._headers(), timeout=self._timeout)
            if resp.status_code != 200:
                return None, f"Supabase query returned HTTP {resp.status_code}"
            rows = resp.json()
            return [_hydrate_row(r) for r in rows], None
        except (httpx.RequestError, OSError):
            return None, "Network unreachable while querying Supabase"
        except Exception as exc:
            return None, sanitize_secret_text(exc)[:160]

    def fetch_run_by_id(self, record_id: str) -> tuple[dict[str, Any] | None, str | None, bool]:
        """Fetch a single inspection record by UUID. Returns (row_or_none, error, ok)."""
        url = f"{self._base_url}?select=*&id=eq.{quote(str(record_id), safe='')}&limit=1"
        try:
            resp = self._client.get(url, headers=self._headers(), timeout=self._timeout)
            if resp.status_code != 200:
                return None, f"Supabase lookup returned HTTP {resp.status_code}", False
            rows = resp.json()
            if not rows:
                return None, None, True
            return _hydrate_row(rows[0]), None, True
        except (httpx.RequestError, OSError):
            return None, "Network unreachable during Supabase lookup", False
        except Exception as exc:
            return None, sanitize_secret_text(exc)[:160], False

    def fetch_summary_stats(self) -> tuple[dict[str, Any] | None, str | None]:
        """Compute summary telemetry stats cleanly from Supabase Postgres."""
        url = f"{self._base_url}?select=decision_status,processing_ms&order=created_at.desc&limit=1000"
        try:
            resp = self._client.get(
                url,
                headers=self._headers({"Prefer": "count=exact"}),
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return None, f"Supabase summary returned HTTP {resp.status_code}"
            rows = resp.json()

            total = len(rows)
            content_range = resp.headers.get("content-range", "")
            if "/" in content_range:
                tail = content_range.split("/")[-1].strip()
                if tail.isdigit():
                    total = int(tail)

            counts = {
                "normal": 0,
                "possible_anomaly": 0,
                "needs_review": 0,
                "invalid_input": 0,
            }
            latencies: list[float] = []
            for r in rows:
                st_val = str(r.get("decision_status", ""))
                if st_val in counts:
                    counts[st_val] += 1
                ms = r.get("processing_ms")
                if isinstance(ms, (int, float)) and ms > 0:
                    latencies.append(float(ms))

            avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
            return {
                "total_inspections": total,
                "status_counts": counts,
                "average_latency_ms": avg_lat,
            }, None
        except (httpx.RequestError, OSError):
            return None, "Network unreachable while loading Supabase summary"
        except Exception as exc:
            return None, sanitize_secret_text(exc)[:160]

    def clear_remote(self) -> tuple[bool, str | None]:
        """Clear inspection history rows in Supabase (when explicitly requested)."""
        try:
            resp = self._client.delete(
                f"{self._base_url}?id=not.is.null",
                headers=self._headers({"Prefer": "return=minimal"}),
                timeout=max(self._timeout, 4.0),
            )
            if resp.status_code in (200, 204):
                return True, None
            return False, f"Supabase clear failed with HTTP {resp.status_code}"
        except (httpx.RequestError, OSError):
            return False, "Network unreachable while clearing Supabase"
        except Exception as exc:
            return False, sanitize_secret_text(exc)[:160]


class AnalysisStore:
    """
    Hybrid persistence manager:
    - Uses Supabase Postgres as default cloud database when configured and reachable.
    - Automatically fails over to local embedded SQLite ("Offline Mode : Operating Locally")
      when internet/Supabase is unavailable, queueing writes with `sync_status='pending'`.
    - Automatically flushes pending SQLite records to Supabase upon reconnection.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        enable_cloud: bool | None = None,
        timeout_seconds: float | None = None,
        probe_ttl_seconds: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        cfg = get_default_config()
        self._mem_conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

        if db_path is None:
            self.db_path = cfg.paths.data_dir / "pulseforge.db"
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path = Path(db_path)

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

        # Resolve whether cloud sync should be enabled cleanly and safely
        running_under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        if enable_cloud is False:
            should_use_cloud = False
        elif enable_cloud is True or (supabase_url and supabase_key) or http_client is not None:
            should_use_cloud = True
        elif str(self.db_path) == ":memory:" or running_under_pytest:
            # Never touch live remote Supabase implicitly inside isolated tests
            should_use_cloud = False
        else:
            should_use_cloud = cfg.supabase.enabled

        self._supabase: SupabasePostgRESTClient | None = None
        if should_use_cloud:
            res_url, res_key = resolve_supabase_credentials(
                override_url=supabase_url,
                override_key=supabase_key,
            )
            if res_url and res_key:
                try:
                    self._supabase = SupabasePostgRESTClient(
                        url=res_url,
                        key=res_key,
                        table_name=cfg.supabase.table_name,
                        timeout_seconds=timeout_seconds or cfg.supabase.timeout_seconds,
                        http_client=http_client,
                    )
                except ValueError as exc:
                    logger.warning("Supabase disabled due to invalid config: %s", exc)

        self._probe_ttl = (
            float(probe_ttl_seconds)
            if probe_ttl_seconds is not None
            else cfg.supabase.probe_ttl_seconds
        )
        self._last_probe_monotonic: float = 0.0
        self._is_online: bool = False
        self._last_synced_count: int = 0
        self._total_synced_session: int = 0
        self._last_error: str | None = None
        self._last_checked_at: str | None = None

    @property
    def cloud_configured(self) -> bool:
        """True if valid Supabase credentials are configured for cloud sync."""
        return self._supabase is not None

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize local SQLite table, concurrency pragmas, and non-destructive migrations."""
        with self._lock, self._get_connection() as conn:
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=30000;")

            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='analysis_runs'"
            ).fetchone()
            if row and row[0] and "machine_type = 'fan'" in row[0]:
                logger.info("Migrating analysis_runs table CHECK constraint for multi-machine support...")
                conn.execute("ALTER TABLE analysis_runs RENAME TO analysis_runs_legacy_m6;")
                conn.executescript(CREATE_TABLE_SQL)
                conn.execute(
                    f"INSERT INTO analysis_runs ({CORE_COLUMNS_CSV}) "
                    f"SELECT {CORE_COLUMNS_CSV} FROM analysis_runs_legacy_m6;"
                )
                conn.execute("DROP TABLE analysis_runs_legacy_m6;")
                conn.commit()
            else:
                conn.executescript(CREATE_TABLE_SQL)
                conn.commit()

            # Non-destructive upgrade for pre-sync databases missing sync_status / synced_at
            cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(analysis_runs);").fetchall()
            }
            if "sync_status" not in cols:
                conn.execute(
                    "ALTER TABLE analysis_runs ADD COLUMN sync_status TEXT NOT NULL DEFAULT 'synced';"
                )
            if "synced_at" not in cols:
                conn.execute("ALTER TABLE analysis_runs ADD COLUMN synced_at TEXT;")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_sync_status ON analysis_runs(sync_status);"
            )
            conn.commit()

    def _get_pending_count(self) -> int:
        with self._lock, self._get_connection() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM analysis_runs WHERE sync_status = 'pending'"
                ).fetchone()[0]
            )

    def _get_pending_rows(self) -> list[dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_runs WHERE sync_status = 'pending' ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def _mark_rows_synced(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in record_ids)
        with self._lock, self._get_connection() as conn:
            conn.execute(
                f"UPDATE analysis_runs SET sync_status = 'synced', synced_at = ? WHERE id IN ({placeholders})",
                [now_iso, *record_ids],
            )
            conn.commit()

    def sync_pending_runs(self, force: bool = False) -> dict[str, Any]:
        """
        Check Supabase reachability (respecting TTL unless force=True) and flush any
        queued local SQLite records up to Supabase idempotently.
        """
        with self._lock:
            if self._supabase is None:
                return self.get_status()

            now_mono = time.monotonic()
            should_probe = force or self._last_probe_monotonic == 0.0 or (
                now_mono - self._last_probe_monotonic >= self._probe_ttl
            )

            self._last_synced_count = 0
            if should_probe:
                self._last_probe_monotonic = now_mono
                self._last_checked_at = datetime.now(timezone.utc).isoformat()
                ok, err = self._supabase.probe()
                self._is_online = ok
                self._last_error = err

            if not self._is_online:
                return self.get_status()

            pending_rows = self._get_pending_rows()
            if pending_rows:
                batch_size = 50
                uploaded_total = 0
                for start in range(0, len(pending_rows), batch_size):
                    chunk = pending_rows[start : start + batch_size]
                    ok, err = self._supabase.upsert_rows(chunk)
                    if not ok:
                        self._is_online = False
                        self._last_error = err
                        logger.warning(
                            "Supabase sync interrupted after %d/%d rows (%s); remaining queued in SQLite.",
                            uploaded_total,
                            len(pending_rows),
                            err,
                        )
                        break
                    chunk_ids = [str(r["id"]) for r in chunk]
                    self._mark_rows_synced(chunk_ids)
                    uploaded_total += len(chunk_ids)

                self._last_synced_count = uploaded_total
                self._total_synced_session += uploaded_total
                if uploaded_total > 0:
                    logger.info(
                        "Synced %d offline SQLite record(s) to Supabase.", uploaded_total
                    )

            return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Return safe, secret-free persistence and connectivity telemetry."""
        pending = self._get_pending_count()
        if self._supabase is None:
            mode = "local_only"
            label = "Local SQLite Only"
        elif self._is_online:
            mode = "supabase"
            label = "Cloud Connection: Online"
        else:
            mode = "offline_sqlite"
            label = "Offline Mode : Operating Locally"

        return {
            "cloud_configured": self._supabase is not None,
            "mode": mode,
            "label": label,
            "is_online": self._is_online if self._supabase is not None else False,
            "pending_sync_count": pending,
            "last_synced_count": self._last_synced_count,
            "total_synced_session": self._total_synced_session,
            "last_checked_at": self._last_checked_at,
            "last_error": self._last_error,
        }

    def save_analysis(
        self,
        result: AnalysisResult,
        processing_ms: int = 0,
        source_kind: str = "uploaded",
        input_label: str = "audio.wav",
        operator_note: str = "",
    ) -> str:
        """
        Persist an AnalysisResult durably:
        1. Write immediately to local SQLite (`sync_status='pending'` when cloud is configured).
        2. When Supabase is configured and reachable, upsert immediately (and flush any
           earlier offline backlog) and mark `sync_status='synced'`.
        3. If offline, fail over gracefully without raising and leave queued in SQLite.
        """
        record_id = str(uuid.uuid4())
        created_at = result.timestamp or datetime.now(timezone.utc).isoformat()

        if result.quality.passed:
            quality_status = "good"
        elif result.quality.warnings and not result.quality.errors:
            quality_status = "warning"
        else:
            quality_status = "poor"

        quality_reasons_json = json.dumps(result.quality.errors + result.quality.warnings)
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

        if source_kind not in ("uploaded", "demo_sample", "synthetic"):
            source_kind = "uploaded"

        safe_label = _sanitize_label(input_label)
        safe_note = sanitize_secret_text(operator_note)[:1000] if operator_note else ""
        initial_sync_status = "pending" if self._supabase is not None else "synced"

        query = """
        INSERT INTO analysis_runs (
            id, created_at, machine_type, machine_id, input_label, source_kind,
            audio_duration_seconds, sample_rate_hz, anomaly_score, threshold,
            decision_status, quality_status, quality_reasons_json, unfamiliarity_score,
            uncertainty_reasons_json, model_version, processing_ms, operator_note,
            sync_status, synced_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        params = (
            record_id,
            created_at,
            result.machine_type,
            result.machine_id,
            safe_label,
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
            safe_note,
            initial_sync_status,
            None,
        )

        with self._lock, self._get_connection() as conn:
            conn.execute(query, params)
            conn.commit()

        if self._supabase is not None:
            self.sync_pending_runs(force=True)

        logger.info(
            "Saved analysis run %s (status: %s, label: %s, cloud_online: %s)",
            record_id,
            result.status.value,
            safe_label,
            self._is_online,
        )
        return record_id

    def get_recent_runs(
        self,
        limit: int = 10,
        machine_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve recent analysis runs (from Supabase when online, else SQLite)."""
        if self._supabase is not None:
            self.sync_pending_runs(force=False)
            if self._is_online:
                remote_rows, err = self._supabase.fetch_recent_runs(
                    limit=limit, machine_id=machine_id
                )
                if remote_rows is not None:
                    return remote_rows
                with self._lock:
                    self._is_online = False
                    self._last_error = err

        base_query = "SELECT * FROM analysis_runs"
        params: list[Any] = []
        if machine_id:
            base_query += " WHERE machine_id = ?"
            params.append(machine_id)
        base_query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(base_query, params).fetchall()

        return [_hydrate_row(dict(r)) for r in rows]

    def get_run_by_id(self, record_id: str) -> dict[str, Any] | None:
        """Fetch a single analysis run by primary key (Supabase when online, else SQLite)."""
        if self._supabase is not None and self._is_online:
            row, err, ok = self._supabase.fetch_run_by_id(record_id)
            if ok and row is not None:
                return row
            if not ok:
                with self._lock:
                    self._is_online = False
                    self._last_error = err

        query = "SELECT * FROM analysis_runs WHERE id = ?"
        with self._lock, self._get_connection() as conn:
            row = conn.execute(query, (record_id,)).fetchone()

        if row is None:
            return None
        return _hydrate_row(dict(row))

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute aggregated statistics for telemetry dashboards."""
        if self._supabase is not None:
            self.sync_pending_runs(force=False)
            if self._is_online:
                remote_stats, err = self._supabase.fetch_summary_stats()
                if remote_stats is not None:
                    return remote_stats
                with self._lock:
                    self._is_online = False
                    self._last_error = err

        with self._lock, self._get_connection() as conn:
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

    def clear_history(self, include_cloud: bool = True) -> None:
        """Purge local SQLite records (and Supabase records when online and enabled)."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_runs")
            conn.commit()
        if include_cloud and self._supabase is not None:
            self.sync_pending_runs(force=False)
            if self._is_online:
                ok, err = self._supabase.clear_remote()
                if not ok:
                    with self._lock:
                        self._last_error = err
