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
