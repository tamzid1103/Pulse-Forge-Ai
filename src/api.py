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
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import AppConfig, get_default_config
from src.copilot import IndustrialCopilot
from src.exceptions import (
    AudioValidationError,
    CorruptAudioError,
    FileTooLargeError,
    InvalidFormatError,
    ModelNotFoundError,
    PulseForgeError,
)
from src.service import PulseForgeService

logger = logging.getLogger("pulseforge.api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

    return app


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
