"""
FastAPI REST & WebSocket Backend for PulseForge AI (Milestone M10).

Provides:
- High-performance headless REST endpoints for plant SCADA/MES and external agents.
- Real-time WebSocket audio chunk streaming for live sensor diagnostics.
- Model lifecycle management, CORS configuration, and structured ISO-aligned error handling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, status
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

    return app


# Default app instance for uvicorn
app = create_app()
