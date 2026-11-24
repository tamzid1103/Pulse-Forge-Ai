# PulseForge AI — FastAPI REST & WebSocket API Specification

**Module:** `src/api.py`  
**Framework:** FastAPI 0.110+ on Uvicorn  
**Compliance:** ISO 13374 Condition Monitoring and ISO 10816-3 Vibration Severity Standards  
**Status:** ✅ Fully Implemented and Verified (16 tests in `tests/test_api.py`)  
**Interactive Docs:** `http://127.0.0.1:8000/docs` (Swagger UI) & `http://127.0.0.1:8000/redoc` (ReDoc)

---

## 1. Architecture & Execution

The FastAPI service decouples backend acoustic anomaly detection, SQLite audit persistence, and the AI Maintenance Copilot into a high-performance asynchronous API server. It provides headless REST endpoints for plant SCADA/MES systems and real-time WebSockets for modern frontend dashboards (such as the Lovable React frontend in Milestone M11).

### Launching the Backend Server

```bash
# Run locally with hot reload
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Production deployment
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 2. API Endpoints Overview

| Method | Path | Category | Description |
|---|---|---|---|
| `GET` | `/health` | System | Lightweight health and model readiness check |
| `GET` | `/api/v1/system/status` | System | Telemetry thresholds, feature dimensions, and fleet metrics |
| `GET` | `/api/v1/scenarios` | Analysis | Catalog of 5 curated demonstration audio scenarios |
| `POST` | `/api/v1/analyze` | Analysis | Multipart WAV audio upload and 4-state diagnostic analysis |
| `POST` | `/api/v1/analyze/scenario/{id}` | Analysis | 1-click evaluation of curated demo scenarios |
| `GET` | `/api/v1/runs` | Audit | Paginated historical inspection runs from SQLite |
| `GET` | `/api/v1/runs/{run_id}` | Audit | Single inspection run details by UUID |
| `DELETE` | `/api/v1/runs` | Audit | Purge inspection run history (reset demo) |
| `POST` | `/api/v1/copilot/work-order` | Copilot | Generate structured ISO 10816 maintenance work order |
| `POST` | `/api/v1/copilot/chat` | Copilot | Interactive technician diagnostic advisory Q&A |
| `WS` | `/ws/v1/audio-stream` | Streaming | Real-time bidirectional WebSocket for microphone streaming & telemetry |

---

## 3. Endpoint Specifications

### A. System & Health Endpoints

#### `GET /health`
- **Purpose:** Fast container/load-balancer readiness probe.
- **Response `200 OK`:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "fan-ae-v1",
  "timestamp": "2026-11-24T06:30:00Z"
}
```

#### `GET /api/v1/system/status`
- **Purpose:** Comprehensive operational telemetry for plant dashboards.
- **Response `200 OK`:**
```json
{
  "application": "PulseForge AI",
  "machine_type": "fan",
  "model_type": "autoencoder",
  "sample_rate": 16000,
  "feature_dims": 4096,
  "thresholds": {
    "autoencoder_mse": 0.0032,
    "familiarity_mahalanobis": 9.2549
  },
  "fleet_summary": {
    "total_inspections": 12,
    "status_counts": {
      "normal": 8,
      "possible_anomaly": 2,
      "needs_review": 2,
      "invalid_input": 0
    },
    "average_latency_ms": 14.5
  },
  "model_status": {
    "model_ready": true,
    "autoencoder_loaded": true,
    "scaler_loaded": true,
    "familiarity_loaded": true
  },
  "supported_formats": ["wav"]
}
```

---

### B. Acoustic Analysis Endpoints

#### `POST /api/v1/analyze`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: Required WAV binary.
  - `machine_id`: Asset identifier (default: `"fan_id_00"`).
  - `source_kind`: Source label (`"uploaded"`, `"sensor"`, etc.).
  - `record_to_db`: Boolean (`true` to persist into SQLite audit log).
  - `operator_note`: Optional technician note.
- **Example cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F "file=@demo_audio/01_normal_fan.wav" \
  -F "machine_id=fan_id_00" \
  -F "operator_note=Routine intake plenum check"
```
- **Response `200 OK`:**
```json
{
  "run_id": "3c8a14fd-9f82-411a-874b-e7b51b312781",
  "machine_id": "fan_id_00",
  "filename": "01_normal_fan.wav",
  "analysis": {
    "status": "normal",
    "status_label": "Normal",
    "explanation": "Normal acoustic signature verified. Reconstruction error remains within baseline boundaries.",
    "recommended_action": "Maintain routine 4-hour monitoring interval.",
    "quality": {
      "passed": true,
      "rms_level": 0.0345,
      "is_silent": false,
      "clipping_ratio": 0.0,
      "is_clipped": false,
      "duration_seconds": 10.0,
      "warnings": [],
      "errors": []
    },
    "model_prediction": {
      "model_type": "autoencoder",
      "anomaly_score": 0.00185,
      "threshold": 0.0032,
      "is_anomaly": false
    },
    "familiarity": {
      "score": 4.12,
      "threshold": 9.2549,
      "is_familiar": true
    }
  }
}
```

#### `POST /api/v1/analyze/scenario/{scenario_id}`
- **Purpose:** 1-click execution of preloaded demo recordings without uploading files.
- **Supported IDs & Aliases:**
  - `scenario_01` / `normal` / `01`: Healthy baseline fan audio.
  - `scenario_02` / `anomaly` / `bearing`: Authentic bearing raceway defect.
  - `scenario_03` / `silent` / `disconnection`: Zero-energy sensor line fault.
  - `scenario_04` / `clipped` / `saturation`: ADC digital clipping (>1%).
  - `scenario_05` / `domain_shift` / `unfamiliar`: Factory background noise shift.
- **Example cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze/scenario/bearing
```

---

### C. SQLite Audit & Persistence Endpoints

#### `GET /api/v1/runs?limit=50&machine_id=fan_id_00`
- **Purpose:** Query paginated inspection logs for compliance audits.
- **Response `200 OK`:**
```json
{
  "total_returned": 2,
  "summary": {
    "total_inspections": 2,
    "status_counts": { "normal": 1, "possible_anomaly": 1 },
    "average_latency_ms": 13.8
  },
  "runs": [
    {
      "id": "3c8a14fd-9f82-411a-874b-e7b51b312781",
      "timestamp": "2026-11-24 06:30:15 UTC",
      "machine_id": "fan_id_00",
      "decision_status": "normal",
      "anomaly_score": 0.00185,
      "threshold": 0.0032,
      "rms_level": 0.0345,
      "clipping_ratio": 0.0,
      "processing_ms": 14.1
    }
  ]
}
```

#### `GET /api/v1/runs/{run_id}`
- **Response `200 OK`:** Returns full single run metadata or `404 Not Found`.

#### `DELETE /api/v1/runs`
- **Response `200 OK`:** Clears historical database records for test resets.

---

### D. Factory Brain AI Copilot Endpoints

#### `POST /api/v1/copilot/work-order`
- **Request Body (JSON):**
```json
{
  "run_id": "3c8a14fd-9f82-411a-874b-e7b51b312781",
  "decision_status": "possible_anomaly",
  "machine_id": "fan_id_00",
  "operator_note": "Acoustic rattling near inboard bearing housing."
}
```
- **Response `200 OK`:**
```json
{
  "work_order_id": "WO-FAN-2026-7842",
  "asset_id": "fan_id_00",
  "diagnostic_status": "possible_anomaly",
  "urgency": "PRIORITY_INSPECTION",
  "suspected_mechanism": "High-frequency acoustic transients indicate raceway micro-spalling or insufficient grease.",
  "iso_severity_zone": "Zone C (Restricted Operation)",
  "recommended_steps": [
    "Execute OSHA 1910.147 Lockout/Tagout on 480V motor supply.",
    "Verify zero-motion rotation before removing belt guard.",
    "Perform acoustic probe listening at 2–8 kHz around bearing housing.",
    "Measure housing temperature with infrared thermometer."
  ],
  "safety_checklist": [
    "Lockout/Tagout 480V motor disconnect",
    "Verify zero rotation stroboscopically",
    "Wear ANSI Z87.1 safety glasses and Class 2 hearing protection"
  ],
  "required_tools": ["Acoustic stethoscope", "IR thermometer", "Grease gun"],
  "required_ppe": ["Safety glasses", "Steel-toe boots", "Earplugs"],
  "ai_engine": "llama-3.1-8b-instant (Groq LPU)"
}
```

#### `POST /api/v1/copilot/chat`
- **Request Body (JSON):**
```json
{
  "messages": [
    { "role": "user", "content": "How do I calculate re-lubrication quantity for SKF 6308?" }
  ],
  "machine_id": "fan_id_00"
}
```
- **Response `200 OK`:** Returns technician advisory grounded in ISO standards and fan SOPs.

---

### E. Real-Time WebSocket Audio Streaming

#### `WS /ws/v1/audio-stream?machine_id=fan_id_00`
- **Protocol:** Bidirectional WebSocket (text JSON commands and binary PCM chunks).
- **Handshake Frame (Server $\to$ Client):**
```json
{
  "type": "connected",
  "machine_id": "fan_id_00",
  "sample_rate": 16000,
  "message": "PulseForge real-time acoustic telemetry stream active."
}
```
- **Streaming Chunks (Client $\to$ Server):**
  - Raw 16-bit PCM bytes (e.g. 1024 or 2048 sample buffers).
- **Telemetry Broadcast (Server $\to$ Client):**
```json
{
  "type": "telemetry",
  "frame_index": 12,
  "machine_id": "fan_id_00",
  "telemetry": {
    "rms_level": 0.03452,
    "peak_amplitude": 0.2841,
    "clipping_ratio": 0.0,
    "quality_pass": true,
    "spectral_bands": {
      "low_0_2khz": 0.0142,
      "mid_2_4khz": 0.0261,
      "high_4_8khz": 0.0094
    }
  }
}
```

---

## 4. Structured Error Handling

All pipeline exceptions are mapped to standard HTTP status codes with structured JSON error payloads:

| HTTP Status | Error Code | Description |
|---|---|---|
| `400 Bad Request` | `invalid_audio` | Corrupted container header or empty/truncated file |
| `404 Not Found` | `not_found` | Requested scenario or SQLite audit run ID does not exist |
| `413 Payload Too Large` | `file_too_large` | Audio exceeds maximum configured byte limit (25 MB) |
| `415 Unsupported Media Type` | `unsupported_media_type` | File is not valid 16-bit PCM WAV audio |
| `422 Unprocessable Entity` | `audio_not_suitable` | Recording fails minimum duration (<1.0s) or quality bounds |
| `503 Service Unavailable` | `model_unavailable` | Autoencoder weights or feature scalers are missing |
| `500 Internal Error` | `analysis_failed` | Unexpected pipeline failure (logged locally, no leaks) |

**Error Payload Schema:**
```json
{
  "error_code": "unsupported_media_type",
  "message": "Unsupported audio format 'mp3'. Supported formats: wav.",
  "detail": "Only standard 16-bit PCM WAV audio is supported for MVP edge inference."
}
```
