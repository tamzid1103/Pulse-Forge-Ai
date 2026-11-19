# Database Schema & Metadata Audit Trail

## 1. Storage strategy: Embedded SQLite Audit Store

In accordance with Milestone M6 and the CloudCamp BD Data Lifecycle standards, PulseForge AI implements an embedded **SQLite metadata audit store** (`src/persistence.py`).

Key architectural principles:
- **Zero Configuration:** Embedded local database (`data/pulseforge.db`), requiring zero database server setup.
- **Privacy & Safety:** Strictly persists **analysis metadata, scores, and timestamps only**. Raw audio is never persisted by default.
- **Unified Interface:** Used by the central service (`src/service.py`) to power both Streamlit and FastAPI `/api/v1/history` endpoints.

## 2. Table: `analysis_runs`

| Column | Type | Required | Description |
|---|---|---:|---|
| `id` | TEXT | Yes | UUID primary key |
| `created_at` | TEXT | Yes | UTC ISO-8601 timestamp |
| `machine_type` | TEXT | Yes | MVP value: `fan` |
| `input_label` | TEXT | No | Sanitized display label; not a filesystem path |
| `source_kind` | TEXT | Yes | `uploaded` or `demo_sample` |
| `audio_duration_seconds` | REAL | Yes | Decoded duration |
| `sample_rate_hz` | INTEGER | Yes | Source or normalized sample rate; name clearly indicates which |
| `anomaly_score` | REAL | Yes | Raw score from model |
| `threshold` | REAL | Yes | Threshold used for decision |
| `decision_status` | TEXT | Yes | `normal`, `possible_anomaly`, `needs_review`, `invalid_input` |
| `quality_status` | TEXT | Yes | `good`, `warning`, or `poor` |
| `quality_reasons_json` | TEXT | No | JSON array of quality flags |
| `unfamiliarity_score` | REAL | No | Distance / score if implemented |
| `uncertainty_reasons_json` | TEXT | No | JSON array of uncertainty flags |
| `model_version` | TEXT | Yes | Model artifact version/hash |
| `processing_ms` | INTEGER | No | Measured inference duration |
| `operator_note` | TEXT | No | Optional short user note; avoid sensitive content |

### SQL sketch

```sql
CREATE TABLE IF NOT EXISTS analysis_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    machine_type TEXT NOT NULL CHECK (machine_type = 'fan'),
    input_label TEXT,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('uploaded', 'demo_sample')),
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
```

## 3. ER diagram

There are no user/account tables in the MVP. If metadata history is enabled, the minimal model is one analysis record per run:

```mermaid
erDiagram
    ANALYSIS_RUNS {
        TEXT id PK
        TEXT created_at
        TEXT machine_type
        TEXT source_kind
        REAL audio_duration_seconds
        INTEGER sample_rate_hz
        REAL anomaly_score
        REAL threshold
        TEXT decision_status
        TEXT quality_status
        TEXT quality_reasons_json
        REAL unfamiliarity_score
        TEXT uncertainty_reasons_json
        TEXT model_version
        INTEGER processing_ms
    }
```

## 4. Data retention and privacy

- Default: no raw audio persistence.
- Uploaded bytes should be held in memory or a generated temporary file and removed after analysis.
- If the operator explicitly exports an audio file, that is a separate user action and must be explained.
- Retain only metadata that is useful for demo history and evaluation.
- Do not log full local paths or raw uploaded data.
- A public/multi-user deployment requires an approved data-retention policy and access control.

## 5. Model artifacts are not database rows

Keep model checkpoints, preprocessing config, threshold, and metrics as versioned files under a local `models/` or `model_artifacts/` directory. Do not store large weight tensors in SQLite.
