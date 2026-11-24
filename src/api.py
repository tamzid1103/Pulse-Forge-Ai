"""
FastAPI REST & WebSocket Backend for PulseForge AI (Milestone M10).

Provides:
- High-performance headless REST endpoints for plant SCADA/MES and external agents.
- Real-time WebSocket audio chunk streaming for live sensor diagnostics.
- Model lifecycle management, CORS configuration, and structured ISO-aligned error handling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel, Field

from src.config import AppConfig, get_default_config
from src.copilot import IndustrialCopilot
from src.decision import DecisionStatus
from src.exceptions import (
    AudioValidationError,
    CorruptAudioError,
    FileTooLargeError,
    InvalidFormatError,
    ModelNotFoundError,
    PulseForgeError,
)
from src.schemas import (
    AnalysisResult,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)
from src.service import PulseForgeService

logger = logging.getLogger("pulseforge.api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Pydantic Schemas for Requests
# ---------------------------------------------------------------------------
class WorkOrderRequest(BaseModel):
    run_id: str | None = Field(
        default=None,
        description="Optional SQLite analysis run ID to ground the work order in empirical telemetry.",
    )
    decision_status: str = Field(
        default="possible_anomaly",
        description="Operational decision state if run_id is omitted (normal, possible_anomaly, needs_review).",
    )
    machine_id: str = Field(
        default="fan_id_00",
        description="Target machine equipment identifier (e.g. 'fan_id_00').",
    )
    operator_note: str = Field(
        default="",
        description="Field technician observations or dispatch notes.",
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Optional Groq API key override for live LPU reasoning.",
    )


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="Message author role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message text content.")


class CopilotChatRequest(BaseModel):
    messages: list[ChatMessageItem] = Field(
        ...,
        description="Sequential list of conversation messages.",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional analysis run ID to ground copilot responses in telemetry.",
    )
    machine_id: str = Field(
        default="fan_id_00",
        description="Target machine equipment identifier.",
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Optional Groq API key override.",
    )

# Global state instances accessible across routes
service: PulseForgeService | None = None
copilot: IndustrialCopilot | None = None


def get_service() -> PulseForgeService:
    """Return the initialized PulseForgeService instance."""
    global service
    if service is None:
        service = PulseForgeService()
    return service


def get_copilot() -> IndustrialCopilot:
    """Return the initialized IndustrialCopilot instance."""
    global copilot
    if copilot is None:
        copilot = IndustrialCopilot()
    return copilot


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager:
    Initializes models and services at startup, safely cleans up at shutdown.
    """
    global service, copilot
    logger.info("Initializing PulseForge AI Service and Copilot...")
    try:
        service = get_service()
        # Pre-warm service models
        is_ready = service.is_ready()
        logger.info("PulseForge Service initialized. Model ready: %s", is_ready)
    except Exception as e:
        logger.error("Failed to initialize PulseForge service at startup: %s", e)

    try:
        copilot = get_copilot()
        logger.info("PulseForge Copilot initialized.")
    except Exception as e:
        logger.error("Failed to initialize PulseForge copilot: %s", e)

    yield

    logger.info("Shutting down PulseForge AI API...")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="PulseForge AI Industrial Maintenance API",
        description=(
            "Autonomous acoustic anomaly detection and predictive maintenance backend "
            "conforming to ISO 13374 Condition Monitoring and ISO 10816-3 Vibration Severity Standards."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS for multi-client access (Streamlit, Lovable, React, SCADA)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Structured Exception Handlers conforming to docs/API_DOCUMENTATION.md
    # ---------------------------------------------------------------------------
    @app.exception_handler(InvalidFormatError)
    async def invalid_format_handler(request: Request, exc: InvalidFormatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "error_code": "unsupported_media_type",
                "message": str(exc),
                "detail": "Only standard 16-bit PCM WAV audio is supported for MVP edge inference.",
            },
        )

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(request: Request, exc: FileTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error_code": "file_too_large",
                "message": str(exc),
                "detail": "Uploaded audio exceeds maximum configured size boundary.",
            },
        )

    @app.exception_handler(CorruptAudioError)
    async def corrupt_audio_handler(request: Request, exc: CorruptAudioError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "invalid_audio",
                "message": str(exc),
                "detail": "Audio container header or sample stream is corrupted and cannot be decoded.",
            },
        )

    @app.exception_handler(AudioValidationError)
    async def audio_validation_handler(request: Request, exc: AudioValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error_code": "audio_not_suitable",
                "message": str(exc),
                "detail": "Audio file does not satisfy minimum duration, sampling rate, or signal quality criteria.",
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_missing_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": "model_unavailable",
                "message": str(exc),
                "detail": "Required neural network weights or feature scalers are missing.",
            },
        )

    @app.exception_handler(PulseForgeError)
    async def pulseforge_error_handler(request: Request, exc: PulseForgeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "analysis_failed",
                "message": str(exc),
                "detail": "An internal domain error occurred in the acoustic analysis pipeline.",
            },
        )

    # ---------------------------------------------------------------------------
    # Health & System Status Endpoints
    # ---------------------------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, Any]:
        """
        Lightweight health and operational readiness check conforming to docs/API_DOCUMENTATION.md.
        """
        svc = get_service()
        model_ready = svc.is_ready()
        return {
            "status": "ok" if model_ready else "degraded",
            "model_loaded": model_ready,
            "model_version": "fan-ae-v1",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    @app.get("/api/v1/system/status", tags=["System"])
    async def system_status() -> dict[str, Any]:
        """
        Detailed equipment and algorithm telemetry: sample rate, feature dimensions,
        decision thresholds, and fleet operational metrics.
        """
        svc = get_service()
        return {
            "application": "PulseForge AI",
            "machine_type": "fan",
            "model_type": "autoencoder",
            "sample_rate": svc.config.audio.sample_rate,
            "feature_dims": svc.config.features.n_mels * svc.config.features.frames_per_sample,
            "thresholds": {
                "autoencoder_mse": 0.0032,
                "familiarity_mahalanobis": 9.2549,
            },
            "fleet_summary": svc.get_fleet_status(),
            "supported_formats": list(svc.config.audio.supported_formats),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    @app.get("/api/v1/scenarios", tags=["Analysis"])
    async def list_scenarios() -> list[dict[str, Any]]:
        """
        Returns the catalog of bundled curated demonstration scenarios.
        """
        return get_scenario_catalog()

    # ---------------------------------------------------------------------------
    # Acoustic Analysis Endpoints
    # ---------------------------------------------------------------------------
    @app.post("/api/v1/analyze", tags=["Analysis"])
    async def analyze_audio_upload(
        file: UploadFile = File(..., description="Multipart WAV audio recording"),
        machine_id: str = Form("fan_id_00", description="Equipment asset ID"),
        source_kind: str = Form("uploaded", description="Source classification"),
        record_to_db: bool = Form(True, description="Persist audit record in SQLite"),
        operator_note: str = Form("", description="Optional technician note"),
    ) -> dict[str, Any]:
        """
        Main acoustic anomaly detection pipeline endpoint.
        Performs technical audio decoding, physical signal quality assessment,
        Log-Mel feature extraction, PyTorch Autoencoder scoring, latent Mahalanobis
        familiarity gating, and 4-state deterministic decision policy.
        """
        svc = get_service()
        contents = await file.read()
        filename = file.filename or "uploaded.wav"

        result, signal, run_id = svc.analyze_audio(
            source=contents,
            filename=filename,
            machine_id=machine_id,
            machine_type="fan",
            source_kind=source_kind,
            record_to_db=record_to_db,
            operator_note=operator_note,
        )

        response: dict[str, Any] = {
            "run_id": run_id,
            "machine_id": machine_id,
            "filename": filename,
            "analysis": result.to_dict(),
        }
        return response

    @app.post("/api/v1/analyze/scenario/{scenario_id}", tags=["Analysis"])
    async def analyze_scenario(
        scenario_id: str,
        machine_id: str = Form("fan_id_00"),
        record_to_db: bool = Form(True),
    ) -> dict[str, Any]:
        """
        Executes a 1-click evaluation of a bundled demonstration scenario.
        Accepts scenario identifiers: scenario_01 (normal), scenario_02 (anomaly),
        scenario_03 (silent), scenario_04 (clipped), scenario_05 (domain_shift).
        """
        catalog = get_scenario_catalog()
        matched = None
        s_id_clean = scenario_id.lower().strip()
        for item in catalog:
            if s_id_clean == item["id"] or s_id_clean in item["aliases"]:
                matched = item
                break

        if not matched:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario '{scenario_id}' not found. Available: {[s['id'] for s in catalog]}",
            )

        file_path = Path(matched["file_path"])
        if not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario audio file missing at {file_path}",
            )

        svc = get_service()
        result, signal, run_id = svc.analyze_audio(
            source=file_path,
            filename=file_path.name,
            machine_id=machine_id,
            machine_type="fan",
            source_kind="demo_sample",
            record_to_db=record_to_db,
            operator_note=f"Demo Scenario: {matched['label']}",
        )

        return {
            "run_id": run_id,
            "scenario": matched,
            "analysis": result.to_dict(),
        }

    # ---------------------------------------------------------------------------
    # Audit Trail & Persistence Endpoints
    # ---------------------------------------------------------------------------
    @app.get("/api/v1/runs", tags=["Audit & Persistence"])
    async def get_inspection_runs(
        machine_id: str | None = Query(default=None, description="Filter by machine asset ID"),
        limit: int = Query(default=50, ge=1, le=500, description="Max records to retrieve"),
    ) -> dict[str, Any]:
        """
        Retrieves historical acoustic inspection records and aggregated fleet statistics from SQLite.
        """
        svc = get_service()
        runs = svc.get_history(limit=limit, machine_id=machine_id)
        summary = svc.get_summary_stats()
        return {
            "total_returned": len(runs),
            "summary": summary,
            "runs": runs,
        }

    @app.get("/api/v1/runs/{run_id}", tags=["Audit & Persistence"])
    async def get_inspection_run(run_id: str) -> dict[str, Any]:
        """
        Retrieves a single comprehensive inspection record by UUID primary key.
        """
        svc = get_service()
        run = svc.store.get_run_by_id(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inspection record '{run_id}' not found in SQLite store.",
            )
        return run

    @app.delete("/api/v1/runs", tags=["Audit & Persistence"])
    async def clear_inspection_runs() -> dict[str, Any]:
        """
        Purges historical SQLite inspection records for demonstration resets.
        """
        svc = get_service()
        svc.store.clear_history()
        return {
            "status": "cleared",
            "message": "SQLite audit inspection log purged successfully.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # ---------------------------------------------------------------------------
    # Factory Brain Copilot Endpoints
    # ---------------------------------------------------------------------------
    @app.post("/api/v1/copilot/work-order", tags=["Factory Brain Copilot"])
    async def generate_maintenance_work_order(req: WorkOrderRequest) -> dict[str, Any]:
        """
        Generates a structured, ISO 10816-grounded maintenance work order.
        Integrates deterministic acoustic telemetry with OEM maintenance SOPs.
        Uses Groq LPU inference with automated fallback and zero-crash offline SOP mode.
        """
        svc = get_service()
        cop = get_copilot()
        result = _resolve_analysis_result(
            svc=svc,
            run_id=req.run_id,
            decision_status=req.decision_status,
            machine_id=req.machine_id,
            operator_note=req.operator_note,
        )
        wo = cop.generate_work_order(
            result=result,
            machine_id=req.machine_id,
            override_key=req.groq_api_key,
        )
        if req.operator_note:
            wo["operator_note"] = req.operator_note
        return wo

    @app.post("/api/v1/copilot/chat", tags=["Factory Brain Copilot"])
    async def copilot_chat(req: CopilotChatRequest) -> dict[str, Any]:
        """
        Interactive plant maintenance assistant providing technician diagnostic Q&A.
        Answers vibration severity queries, ISO threshold interpretations, and safety protocols.
        """
        svc = get_service()
        cop = get_copilot()
        result = None
        if req.run_id:
            result = _resolve_analysis_result(
                svc=svc,
                run_id=req.run_id,
                decision_status="normal",
                machine_id=req.machine_id,
            )
        messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
        reply = cop.chat_response(
            messages=messages_dicts,
            result=result,
            machine_id=req.machine_id,
            override_key=req.groq_api_key,
        )
        return {
            "machine_id": req.machine_id,
            "reply": reply,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # ---------------------------------------------------------------------------
    # Real-Time WebSocket Streaming Endpoint
    # ---------------------------------------------------------------------------
    @app.websocket("/ws/v1/audio-stream")
    async def audio_stream_websocket(
        websocket: WebSocket,
        machine_id: str = "fan_id_00",
    ) -> None:
        """
        Bidirectional WebSocket endpoint for real-time acoustic streaming.
        Accepts:
        - Binary PCM chunks: returns instantaneous RMS, clipping, and spectral bands.
        - JSON text commands: ping/pong, full scenario evaluation, or telemetry queries.
        """
        await websocket.accept()
        logger.info("WebSocket client connected for machine %s", machine_id)
        await websocket.send_json({
            "type": "connected",
            "machine_id": machine_id,
            "sample_rate": 16000,
            "message": "PulseForge real-time acoustic telemetry stream active.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        frame_counter = 0
        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    raw_bytes = message["bytes"]
                    frame_counter += 1
                    telemetry = compute_chunk_telemetry(raw_bytes)
                    await websocket.send_json({
                        "type": "telemetry",
                        "frame_index": frame_counter,
                        "machine_id": machine_id,
                        "telemetry": telemetry,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
                elif "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                    except Exception:
                        payload = {"action": message["text"]}

                    action = payload.get("action", "")
                    if action == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                    elif action == "scenario":
                        scenario_id = payload.get("id", "scenario_01")
                        catalog = get_scenario_catalog()
                        matched = next(
                            (s for s in catalog if s["id"] == scenario_id or scenario_id in s["aliases"]),
                            None,
                        )
                        if matched and Path(matched["file_path"]).is_file():
                            svc = get_service()
                            res, _, run_id = svc.analyze_audio(
                                source=Path(matched["file_path"]),
                                filename=matched["filename"],
                                machine_id=machine_id,
                                source_kind="demo_sample",
                            )
                            await websocket.send_json({
                                "type": "analysis_result",
                                "run_id": run_id,
                                "scenario": matched["id"],
                                "analysis": res.to_dict(),
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Scenario {scenario_id} not found.",
                            })
                    else:
                        await websocket.send_json({
                            "type": "ack",
                            "received_action": action,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for machine %s", machine_id)
        except Exception as e:
            logger.warning("WebSocket error for machine %s: %s", machine_id, e)

    return app


def compute_chunk_telemetry(raw_bytes: bytes) -> dict[str, Any]:
    """
    Compute rapid physical telemetry and spectral energy bands from raw 16-bit PCM or WAV audio bytes.
    Optimized for high-throughput edge streaming (<1ms execution).
    """
    if len(raw_bytes) % 2 == 0:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw_bytes[: len(raw_bytes) - 1], dtype=np.int16).astype(np.float32) / 32768.0

    if len(samples) == 0:
        return {
            "rms_level": 0.0,
            "peak_amplitude": 0.0,
            "clipping_ratio": 0.0,
            "quality_pass": False,
            "spectral_bands": {
                "low_0_2khz": 0.0,
                "mid_2_4khz": 0.0,
                "high_4_8khz": 0.0,
            },
        }

    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    clipping = float(np.mean(np.abs(samples) >= 0.99))
    quality_pass = (rms >= 0.001) and (clipping <= 0.01)

    # Rapid FFT spectral energy distribution
    fft_vals = np.abs(np.fft.rfft(samples))
    n_bins = len(fft_vals)
    if n_bins >= 3:
        third = n_bins // 3
        low = float(np.mean(fft_vals[:third]))
        mid = float(np.mean(fft_vals[third : 2 * third]))
        high = float(np.mean(fft_vals[2 * third :]))
    else:
        low = float(np.mean(fft_vals)) if n_bins > 0 else 0.0
        mid, high = low, low

    return {
        "rms_level": round(rms, 5),
        "peak_amplitude": round(peak, 4),
        "clipping_ratio": round(clipping, 4),
        "quality_pass": quality_pass,
        "spectral_bands": {
            "low_0_2khz": round(low, 4),
            "mid_2_4khz": round(mid, 4),
            "high_4_8khz": round(high, 4),
        },
    }


def _resolve_analysis_result(
    svc: PulseForgeService,
    run_id: str | None,
    decision_status: str,
    machine_id: str,
    operator_note: str = "",
) -> AnalysisResult:
    """
    Resolve an AnalysisResult either from an existing SQLite run or by synthesizing
    a representative result matching the requested status.
    """
    if run_id:
        record = svc.store.get_run_by_id(run_id)
        if record:
            status_enum = DecisionStatus.NORMAL
            for s in DecisionStatus:
                if s.value.lower() == record.get("decision_status", "").lower():
                    status_enum = s
                    break

            is_anom = record.get("anomaly_flag", 0) == 1
            score = float(record.get("anomaly_score", 0.0) or 0.0)
            thresh = float(record.get("threshold", 0.0032) or 0.0032)
            rms = float(record.get("rms_level", 0.035) or 0.035)
            clip = float(record.get("clipping_ratio", 0.0) or 0.0)
            duration = float(record.get("duration_seconds", 10.0) or 10.0)
            fam_score = (
                float(record["uncertainty_score"])
                if record.get("uncertainty_score") is not None
                else 4.2
            )
            fam_thresh = (
                float(record["uncertainty_threshold"])
                if record.get("uncertainty_threshold") is not None
                else 9.25
            )
            is_fam = record.get("uncertainty_flag", 0) == 0

            return AnalysisResult(
                status=status_enum,
                status_label=status_enum.value.replace("_", " ").title(),
                quality=QualityReport(
                    passed=record.get("quality_pass", 1) == 1,
                    rms_level=rms,
                    is_silent=(rms < 0.001),
                    clipping_ratio=clip,
                    is_clipped=(clip > 0.01),
                    duration_seconds=duration,
                    is_duration_valid=True,
                    warnings=record.get("quality_reasons", []),
                ),
                model_prediction=ModelPrediction(
                    model_type=record.get("model_version", "autoencoder"),
                    anomaly_score=score,
                    threshold=thresh,
                    is_anomaly=is_anom,
                ),
                familiarity=FamiliarityReport(
                    score=fam_score,
                    threshold=fam_thresh,
                    is_familiar=is_fam,
                ),
                explanation=record.get("evidence", f"Empirical run record {run_id}"),
                recommended_action=record.get("recommendation", "Execute standard plant SOP."),
                machine_id=record.get("machine_id", machine_id),
            )

    # Synthesize typed result
    status_enum = DecisionStatus.NORMAL
    for s in DecisionStatus:
        if s.value.lower() == decision_status.lower():
            status_enum = s
            break

    is_anomaly = status_enum == DecisionStatus.POSSIBLE_ANOMALY
    is_review = status_enum == DecisionStatus.NEEDS_REVIEW
    return AnalysisResult(
        status=status_enum,
        status_label=status_enum.value.replace("_", " ").title(),
        quality=QualityReport(
            passed=(not is_review and status_enum != DecisionStatus.INVALID_INPUT),
            rms_level=0.0002 if is_review else 0.035,
            is_silent=is_review,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        ),
        model_prediction=ModelPrediction(
            model_type="autoencoder",
            anomaly_score=0.0055 if is_anomaly else 0.0018,
            threshold=0.0032,
            is_anomaly=is_anomaly,
        ),
        familiarity=FamiliarityReport(
            score=12.4 if is_review else 4.2,
            threshold=9.25,
            is_familiar=(not is_review),
        ),
        explanation=f"Telemetry assessment for {machine_id}: {status_enum.value}. {operator_note}".strip(),
        recommended_action="Execute maintenance review per standard operating procedures.",
        machine_id=machine_id,
    )


def get_scenario_catalog() -> list[dict[str, Any]]:
    """Helper returning metadata for bundled demonstration scenarios."""
    demo_dir = PROJECT_ROOT / "demo_audio"
    return [
        {
            "id": "scenario_01",
            "aliases": ["01", "normal", "scenario_01", "01_normal_fan"],
            "label": "01 Normal Fan Baseline",
            "filename": "01_normal_fan.wav",
            "file_path": str((demo_dir / "01_normal_fan.wav").resolve()),
            "expected_status": "normal",
            "description": "Authentic MIMII Fan ID 00 healthy baseline recording in 6 dB industrial noise.",
        },
        {
            "id": "scenario_02",
            "aliases": ["02", "anomaly", "bearing", "scenario_02", "02_bearing_anomaly"],
            "label": "02 Bearing Raceway Anomaly",
            "filename": "02_bearing_anomaly.wav",
            "file_path": str((demo_dir / "02_bearing_anomaly.wav").resolve()),
            "expected_status": "possible_anomaly",
            "description": "Authentic MIMII Fan ID 00 bearing degradation exhibiting high-frequency impact bursts.",
        },
        {
            "id": "scenario_03",
            "aliases": ["03", "silent", "disconnection", "scenario_03", "03_silent_sensor_failure"],
            "label": "03 Silent Sensor Failure",
            "filename": "03_silent_sensor_failure.wav",
            "file_path": str((demo_dir / "03_silent_sensor_failure.wav").resolve()),
            "expected_status": "needs_review",
            "description": "Sensor line disconnection / zero-voltage ADC drop. Fails RMS energy quality check.",
        },
        {
            "id": "scenario_04",
            "aliases": ["04", "clipped", "saturation", "scenario_04", "04_clipped_sensor_saturation"],
            "label": "04 Clipped Sensor Saturation",
            "filename": "04_clipped_sensor_saturation.wav",
            "file_path": str((demo_dir / "04_clipped_sensor_saturation.wav").resolve()),
            "expected_status": "needs_review",
            "description": "Preamplifier gain overflow causing harsh digital square-wave clipping (>1.0% clipping ratio).",
        },
        {
            "id": "scenario_05",
            "aliases": ["05", "domain_shift", "unfamiliar", "scenario_05", "05_unfamiliar_operating_noise"],
            "label": "05 Unfamiliar Operating Noise",
            "filename": "05_unfamiliar_operating_noise.wav",
            "file_path": str((demo_dir / "05_unfamiliar_operating_noise.wav").resolve()),
            "expected_status": "needs_review",
            "description": "Acoustic background noise shift (Mahalanobis distance > 9.2549). Flags domain shift.",
        },
    ]


# Default app instance for uvicorn
app = create_app()
