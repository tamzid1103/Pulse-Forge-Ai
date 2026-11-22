# API and Application Service Specification

## 1. API decision

The MVP does **not** require a separate HTTP backend. Streamlit can call the modular Python application service directly. This is the fastest and least fragile option for a single-machine hackathon demo.

The service boundary must still be clean so a FastAPI adapter can be added later without rewriting model logic. Endpoints below are an optional interface specification, not implemented endpoints.

## 2. Core application service contract

Conceptual Python interface:

`analyze_audio(upload: AudioInput, model_bundle: ModelBundle, config: AppConfig) -> AnalysisResult`

`AudioInput` should contain validated bytes or a safe temporary-file handle, a sanitized display filename, and source metadata. Do not pass an arbitrary user-supplied path into model code.

`AnalysisResult` should contain:

```json
{
  "analysis_id": "optional-uuid",
  "machine_type": "fan",
  "model_version": "fan-ae-v1",
  "decision_status": "possible_anomaly",
  "anomaly_score": 0.0,
  "threshold": 0.0,
  "quality": {
    "status": "good",
    "reasons": []
  },
  "unfamiliarity": {
    "flagged": false,
    "score": null,
    "reasons": []
  },
  "evidence": ["Example evidence text based on measured signal features"],
  "recommendation": "Inspect the machine and verify the recording.",
  "processing_ms": 0
}
```

The numbers above are schema examples only, not benchmark output. Use actual finite results at runtime. `anomaly_score` is not a probability.

## 3. Optional FastAPI endpoints

### `GET /health`

Purpose: lightweight service-health check.

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "fan-ae-v1"
}
```

If the model is not loaded, return a clear degraded/unavailable response rather than `ok` with misleading state.

### `POST /api/v1/analyze`

Content type: `multipart/form-data`

Fields:

- `file` — required WAV file.
- `machine_type` — optional; accepted MVP value is `fan`.
- `source_kind` — optional internal/demo marker; do not trust client-supplied values for security-sensitive decisions.

Success response: `200 OK` with an `AnalysisResult` object.

Errors:

| HTTP | Error code | Meaning |
|---:|---|---|
| 400 | `invalid_audio` | Cannot decode or file content is invalid |
| 413 | `file_too_large` | File exceeds configured upload limit |
| 415 | `unsupported_media_type` | File is not supported WAV content |
| 422 | `audio_not_suitable` | Decodable but fails minimum duration / mandatory quality requirements |
| 503 | `model_unavailable` | Required model artifact is missing or incompatible |
| 500 | `analysis_failed` | Unexpected analysis error; log details locally |

Do not return full stack traces, server paths, or internal secrets to clients.

## 4. Example request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F "machine_type=fan" \
  -F "file=@./demo_audio/fan_normal.wav"
```

This is the intended future request pattern, not evidence that the endpoint exists today.

## 5. API security requirements if FastAPI is enabled

- Enforce upload size and duration limits.
- Validate decoded content, not just extension or MIME header.
- Generate server-side temporary names.
- Use a request timeout and bound concurrent inference jobs.
- Keep the service on localhost during local development.
- Do not expose publicly without HTTPS, authentication, rate limiting, and a documented retention policy.
- CORS should be disabled or allowlisted unless a separate browser frontend requires it.

## 6. API/UI consistency

Streamlit and FastAPI must call the same application service and decision policy. Do not implement separate anomaly thresholds or reliability logic inside the UI/API layer.


## 7. Model Context Protocol (MCP) Server Specification (M7)

PulseForge AI exposes its core industrial intelligence via a custom Model Context Protocol (MCP) server (`src/mcp_server.py`) supporting Stdio and Streamable HTTP/SSE transports.

### Tools Exposed to AI Agents:

1. **`analyze_fan_audio`**:
   - Description: "Executes the full PulseForge AI acoustic pipeline on a target audio file: quality checks -> log-Mel feature extraction -> PyTorch Autoencoder reconstruction -> latent familiarity check -> 4-state industrial decision."
   - Parameters:
     - `file_path` (string, required): Path to WAV recording.
     - `machine_id` (string, optional, default "id_00"): Target fan identifier.
   - Returns: Formatted `AnalysisResult` JSON.

2. **`inspect_signal_quality`**:
   - Description: "Evaluates physical acoustic signal suitability before running ML inference (checks for silence, clipping, duration, and DC bias)."
   - Parameters:
     - `file_path` (string, required): Path to WAV recording.
   - Returns: Formatted `QualityReport` JSON.

3. **`get_machine_health_status`**:
   - Description: "Retrieves recent historical analysis records, anomaly rates, and quality flags for a specific machine ID from the SQLite audit store."
   - Parameters:
     - `machine_id` (string, optional, default "id_00"): Machine identifier.
     - `limit` (integer, optional, default 10): Maximum number of recent records.
   - Returns: List of recent `AnalysisResult` summaries.

### Resources Exposed:
- `pulseforge://telemetry/latest`: Real-time machine health state summary.
- `pulseforge://manifest/fan`: Catalog of available benchmark audio files and split details.

---

## 8. Factory Brain Copilot Service Interface (`src/copilot.py`)

The `IndustrialCopilot` class provides programmatic maintenance advisory generation:

### `generate_work_order(result: AnalysisResult, audio_meta: AudioMetadata | None = None) -> WorkOrderResponse`

Generates an actionable OSHA 1910.147 and ISO 10816 compliant maintenance ticket.

**Execution Flow:**
1. Formats deterministic telemetry string from `AnalysisResult` (anomaly score, threshold, quality flags, Mahalanobis unfamiliarity).
2. Attempts ultra-low latency Groq LPU inference (model cascade: `llama-3.1-8b-instant` -> `llama3-8b-8192` -> `llama-3.3-70b-versatile`).
3. If Groq API key is missing or network is unreachable, automatically triggers `_generate_deterministic_fallback()`: parses local SOP files in `data/knowledge_base/` and returns complete work order without throwing an unhandled exception.

**Return Structure:**
```json
{
  "work_order_id": "WO-20261120-FAN00-781A",
  "urgency": "HIGH",
  "recommended_action": "Bearing lubrication breakdown or micro-spalling detected. Execute Lockout/Tagout (LOTO) per OSHA 1910.147, remove belt guard, and measure ISO 10816 vibration velocity.",
  "safety_lockout_required": true,
  "required_ppe": ["Hard hat", "Safety glasses", "Cut-resistant gloves", "Steel-toe boots"],
  "estimated_downtime_minutes": 60,
  "inspection_checklist": [
    "Verify zero electrical energy state (LOTO step 1)",
    "Inspect SKF 6308 pillow block bearings for thermal discoloration",
    "Check drive belt tension and pulley alignment"
  ],
  "model_used": "llama-3.1-8b-instant"
}
```

### `ask_question(question: str, result: AnalysisResult | None = None) -> str`

Interactive technical assistant answering technician questions grounded in rotating equipment manuals and ISO condition monitoring standards.
