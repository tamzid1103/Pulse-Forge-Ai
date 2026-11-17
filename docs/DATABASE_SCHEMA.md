# Database Schema

## 1. MVP decision: no database required

The first MVP can work without a database. A user uploads one recording, receives a result, and can download or screenshot the result if needed. This minimizes setup and failure points during the hackathon.

Do not add a database before the model pipeline and dashboard work end to end.

## 2. Optional SQLite history

If time permits, add SQLite to store **analysis metadata only**. Do not store raw audio by default. Keep the database local and outside version control.

### Proposed table: `analysis_runs`

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
