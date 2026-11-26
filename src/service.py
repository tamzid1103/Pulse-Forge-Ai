"""
Core application service layer for PulseForge AI (Milestone M6).

Central coordinator that integrates all subsystem components:
1. Audio loading and validation (src/audio_io.py)
2. Pre-inference signal quality assessment (src/quality.py)
3. Log-Mel spectrogram feature extraction and normalization (src/features.py)
4. PyTorch Autoencoder anomaly detection (src/models/autoencoder.py)
5. Latent standardized Euclidean domain familiarity check (src/familiarity.py)
6. 4-state industrial decision policy and explanation generation (src/decision.py)
7. Embedded SQLite audit metadata logging (src/persistence.py)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, BinaryIO

from src.audio_io import AudioSignal, load_audio
from src.config import AppConfig, get_default_config
from src.decision import create_invalid_input_result, evaluate_decision_policy
from src.exceptions import (
    AudioValidationError,
    PulseForgeError,
)
from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
from src.models.registry import MachineModelBundle, ModelRegistry
from src.persistence import AnalysisStore
from src.quality import check_signal_quality
from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
)

logger = logging.getLogger("pulseforge.service")


class PulseForgeService:
    """
    Stateful application service holding cached models and persistent connections.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        store: AnalysisStore | None = None,
    ) -> None:
        self.config = config or get_default_config()
        self.store = store or AnalysisStore()

        self.registry = ModelRegistry(
            model_dir=self.config.paths.model_artifacts_dir,
            supported_types=self.config.supported_machine_types,
        )

    def _get_bundle(self, machine_type: str = "fan") -> MachineModelBundle | None:
        """Retrieve cached MachineModelBundle for a machine family."""
        return self.registry.get_bundle(machine_type)

    @property
    def detector(self) -> AutoencoderDetector | None:
        """Loaded default (fan) AutoencoderDetector instance for backward compatibility."""
        b = self._get_bundle(self.config.default_machine_type)
        return b.detector if b else None

    @property
    def scaler(self) -> FeatureScaler | None:
        """Loaded default (fan) FeatureScaler instance for backward compatibility."""
        b = self._get_bundle(self.config.default_machine_type)
        return b.scaler if b else None

    @property
    def familiarity(self) -> LatentFamiliarityEstimator | None:
        """Loaded default (fan) LatentFamiliarityEstimator instance for backward compatibility."""
        b = self._get_bundle(self.config.default_machine_type)
        return b.familiarity if b else None

    def get_history(
        self, limit: int = 10, machine_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch recent analysis runs from the audit store."""
        return self.store.get_recent_runs(limit=limit, machine_id=machine_id)

    def get_summary_stats(self) -> dict[str, Any]:
        """Fetch aggregated operational summary stats."""
        return self.store.get_summary_stats()

    def clear_history(self) -> None:
        """Purge all stored audit records."""
        self.store.clear_history()

    def analyze_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        machine_id: str = "id_00",
        machine_type: str = "fan",
        source_kind: str = "uploaded",
        record_to_db: bool = True,
        operator_note: str = "",
    ) -> tuple[AnalysisResult, AudioSignal | None, str | None]:
        """Convenience alias for analyze_audio with raw bytes."""
        return self.analyze_audio(
            source=audio_bytes,
            filename=filename,
            machine_id=machine_id,
            machine_type=machine_type,
            source_kind=source_kind,
            record_to_db=record_to_db,
            operator_note=operator_note,
        )

    @property
    def is_model_ready(self) -> bool:
        """Return True if the default machine model artifacts are loaded and ready."""
        return self.registry.is_ready(self.config.default_machine_type)

    def get_model_status(self, machine_type: str | None = None) -> dict[str, Any]:
        """Return operational status and metadata of loaded models."""
        target = (machine_type or self.config.default_machine_type).lower().strip()
        bundle = self._get_bundle(target)
        base_status = (
            bundle.to_status_dict()
            if bundle is not None
            else {
                "machine_type": target,
                "model_ready": False,
                "autoencoder_loaded": False,
                "autoencoder_threshold": None,
                "scaler_loaded": False,
                "familiarity_loaded": False,
                "familiarity_threshold": None,
            }
        )
        base_status["supported_machine_types"] = list(self.config.supported_machine_types)
        base_status["available_machine_types"] = self.registry.available_machine_types()
        base_status["registry"] = self.registry.get_status_summary()
        return base_status

    def analyze_audio(
        self,
        source: bytes | Path | str | BinaryIO,
        filename: str = "audio.wav",
        machine_id: str = "id_00",
        machine_type: str = "fan",
        source_kind: str = "uploaded",
        record_to_db: bool = True,
        operator_note: str = "",
    ) -> tuple[AnalysisResult, AudioSignal | None, str | None]:
        """
        Execute full end-to-end industrial acoustic analysis pipeline.
        
        Args:
            source: Raw bytes, file path, or file-like stream of WAV audio.
            filename: Display filename hint.
            machine_id: Equipment identifier (default: "id_00").
            machine_type: Machine category (must be "fan" in MVP).
            source_kind: "uploaded", "demo_sample", or "synthetic".
            record_to_db: Whether to persist analysis metadata into SQLite.
            operator_note: Optional user note.
            
        Returns:
            Tuple of:
            - AnalysisResult: Structured decision, explanations, and metrics.
            - AudioSignal | None: Decoded single-channel audio signal (for plots/playback).
            - str | None: SQLite record ID if persisted.
        """
        start_time = time.perf_counter()
        m_type = machine_type.lower().strip()
        can_persist = record_to_db and (m_type in self.config.supported_machine_types)

        # Step 1: Safe Audio Decoding & Technical Header Validation
        try:
            if hasattr(source, "read") and callable(getattr(source, "read")):
                raw_bytes = source.read()
            elif isinstance(source, (Path, str)):
                p = Path(source)
                filename = p.name
                raw_bytes = p.read_bytes()
            else:
                raw_bytes = source

            audio_signal = load_audio(
                raw_bytes,
                config=self.config.audio,
                filename_hint=filename,
            )
        except (
            AudioValidationError,
            PulseForgeError,
            ValueError,
        ) as e:
            logger.warning("Audio validation failed for '%s': %s", filename, e)
            result = create_invalid_input_result(
                error_message=str(e),
                machine_id=machine_id,
                machine_type=m_type,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            rec_id = None
            if can_persist:
                rec_id = self.store.save_analysis(
                    result=result,
                    processing_ms=elapsed_ms,
                    source_kind=source_kind,
                    input_label=filename,
                    operator_note=operator_note,
                )
            return result, None, rec_id

        # Convert audio metadata to schema dataclass
        schema_metadata = AudioMetadata(
            sample_rate=audio_signal.sample_rate,
            channels=audio_signal.metadata.channels,
            duration_seconds=audio_signal.duration_seconds,
            num_samples=audio_signal.data.size,
            file_size_bytes=audio_signal.metadata.size_bytes,
            format=audio_signal.metadata.format,
        )

        # Step 2: Signal Quality Inspection (Silence, Clipping, DC bias)
        quality_report = check_signal_quality(
            signal=audio_signal.data,
            sample_rate=audio_signal.sample_rate,
            config=self.config.quality,
        )

        # If mandatory technical quality checks failed (e.g. silent, clipped, or invalid duration)
        if not quality_report.passed:
            result = evaluate_decision_policy(
                quality=quality_report,
                model_prediction=None,
                familiarity=None,
                machine_id=machine_id,
                machine_type=m_type,
                metadata=schema_metadata,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            rec_id = None
            if can_persist:
                rec_id = self.store.save_analysis(
                    result=result,
                    processing_ms=elapsed_ms,
                    source_kind=source_kind,
                    input_label=filename,
                    operator_note=operator_note,
                )
            return result, audio_signal, rec_id

        # Step 3: Resolve machine-specific model bundle & extract features
        if m_type not in self.config.supported_machine_types:
            result = create_invalid_input_result(
                error_message=(
                    f"Unsupported machine type '{machine_type}'. "
                    f"Supported: {', '.join(self.config.supported_machine_types)}."
                ),
                metadata=schema_metadata,
                machine_id=machine_id,
                machine_type=m_type,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            rec_id = None
            if record_to_db and m_type in self.config.supported_machine_types:
                rec_id = self.store.save_analysis(
                    result=result,
                    processing_ms=elapsed_ms,
                    source_kind=source_kind,
                    input_label=filename,
                    operator_note=operator_note,
                )
            return result, audio_signal, rec_id

        bundle = self._get_bundle(m_type)
        if bundle is None or not bundle.is_ready:
            result = evaluate_decision_policy(
                quality=quality_report,
                model_prediction=None,
                familiarity=None,
                machine_id=machine_id,
                machine_type=m_type,
                metadata=schema_metadata,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            rec_id = None
            if record_to_db:
                rec_id = self.store.save_analysis(
                    result=result,
                    processing_ms=elapsed_ms,
                    source_kind=source_kind,
                    input_label=filename,
                    operator_note=operator_note,
                )
            return result, audio_signal, rec_id

        scaled_features = extract_features(
            audio=audio_signal,
            config=self.config.features,
            scaler=bundle.scaler,
        )

        # Step 4: Machine-Specific Autoencoder Anomaly Scoring
        assert bundle.detector is not None
        assert bundle.familiarity is not None
        assert bundle.detector.threshold is not None

        rec_score = bundle.detector.predict_recording_score(scaled_features)
        seg_scores = bundle.detector.predict_segment_scores(scaled_features)
        is_anomaly = bool(rec_score >= bundle.detector.threshold)

        model_prediction = ModelPrediction(
            model_type=f"{m_type}-autoencoder-v1",
            anomaly_score=round(rec_score, 4),
            threshold=round(bundle.detector.threshold, 4),
            is_anomaly=is_anomaly,
            aggregation_method=bundle.detector.config.score_aggregation,
            segment_scores=[round(float(s), 4) for s in seg_scores],
        )

        # Step 5: Latent Familiarity / Domain Shift Verification
        latents = bundle.detector.extract_latent_embeddings(scaled_features)
        familiarity_report = bundle.familiarity.predict(latents)

        # Step 6: 4-State Industrial Decision Policy & Explanation
        result = evaluate_decision_policy(
            quality=quality_report,
            model_prediction=model_prediction,
            familiarity=familiarity_report,
            machine_id=machine_id,
            machine_type=m_type,
            metadata=schema_metadata,
        )

        # Step 7: SQLite Persistence
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        rec_id = None
        if record_to_db:
            rec_id = self.store.save_analysis(
                result=result,
                processing_ms=elapsed_ms,
                source_kind=source_kind,
                input_label=filename,
                operator_note=operator_note,
            )

        logger.info(
            "Analysis complete for '%s': status=%s, score=%.4f (latency=%dms)",
            filename,
            result.status.value,
            model_prediction.anomaly_score,
            elapsed_ms,
        )
        return result, audio_signal, rec_id


# ---------------------------------------------------------------------------
# Global Singleton Helper
# ---------------------------------------------------------------------------

_SERVICE_INSTANCE: PulseForgeService | None = None


def get_service() -> PulseForgeService:
    """Return cached application service singleton instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = PulseForgeService()
    return _SERVICE_INSTANCE
