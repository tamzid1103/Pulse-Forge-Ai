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
from typing import BinaryIO

from src.audio_io import AudioSignal, load_audio
from src.config import AppConfig, get_default_config
from src.decision import create_invalid_input_result, evaluate_decision_policy
from src.exceptions import (
    AudioValidationError,
    CorruptAudioError,
    FileTooLargeError,
    InvalidFormatError,
    PulseForgeError,
)
from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
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

        self._detector: AutoencoderDetector | None = None
        self._scaler: FeatureScaler | None = None
        self._familiarity: LatentFamiliarityEstimator | None = None
        self._load_models()

    def _load_models(self) -> None:
        """Load trained model artifacts from the configured artifact directory."""
        model_dir = self.config.paths.model_artifacts_dir

        ae_path = model_dir / "autoencoder_fan_id_00.pt"
        if ae_path.is_file():
            try:
                self._detector = AutoencoderDetector.load(ae_path)
                logger.info("Loaded Autoencoder detector from %s", ae_path)
            except Exception as e:
                logger.error("Failed to load Autoencoder from %s: %s", ae_path, e)

        scaler_path = model_dir / "scaler_params.json"
        if scaler_path.is_file():
            try:
                self._scaler = FeatureScaler.load(scaler_path)
                logger.info("Loaded FeatureScaler from %s", scaler_path)
            except Exception as e:
                logger.error("Failed to load FeatureScaler from %s: %s", scaler_path, e)

        fam_path = model_dir / "familiarity_fan_id_00.joblib"
        if fam_path.is_file():
            try:
                self._familiarity = LatentFamiliarityEstimator.load(fam_path)
                logger.info("Loaded Familiarity estimator from %s", fam_path)
            except Exception as e:
                logger.error("Failed to load Familiarity from %s: %s", fam_path, e)

    @property
    def is_model_ready(self) -> bool:
        """Return True if all ML model artifacts are loaded and ready for inference."""
        return (
            self._detector is not None
            and self._scaler is not None
            and self._familiarity is not None
            and self._detector.is_trained
            and self._detector.threshold is not None
        )

    def get_model_status(self) -> dict[str, Any]:
        """Return operational status and metadata of loaded models."""
        return {
            "model_ready": self.is_model_ready,
            "autoencoder_loaded": self._detector is not None,
            "autoencoder_threshold": (
                self._detector.threshold if self._detector else None
            ),
            "scaler_loaded": self._scaler is not None,
            "familiarity_loaded": self._familiarity is not None,
            "familiarity_threshold": (
                self._familiarity.threshold if self._familiarity else None
            ),
        }

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
                machine_type=machine_type,
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
                machine_type=machine_type,
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

        # Step 3: Feature Extraction (128 Log-Mel Bins, 1.0s window, 0.5s hop)
        if not self.is_model_ready:
            result = evaluate_decision_policy(
                quality=quality_report,
                model_prediction=None,
                familiarity=None,
                machine_id=machine_id,
                machine_type=machine_type,
                metadata=schema_metadata,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return result, audio_signal, None

        scaled_features = extract_features(
            audio=audio_signal,
            config=self.config.features,
            scaler=self._scaler,
        )

        # Step 4: Primary Autoencoder Anomaly Scoring
        rec_score = self._detector.predict_recording_score(scaled_features)
        seg_scores = self._detector.predict_segment_scores(scaled_features)
        is_anomaly = bool(rec_score >= self._detector.threshold)

        model_prediction = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=round(rec_score, 4),
            threshold=round(self._detector.threshold, 4),
            is_anomaly=is_anomaly,
            aggregation_method=self._detector.config.score_aggregation,
            segment_scores=[round(float(s), 4) for s in seg_scores],
        )

        # Step 5: Latent Familiarity / Domain Shift Verification
        latents = self._detector.extract_latent_embeddings(scaled_features)
        familiarity_report = self._familiarity.predict(latents)

        # Step 6: 4-State Industrial Decision Policy & Explanation
        result = evaluate_decision_policy(
            quality=quality_report,
            model_prediction=model_prediction,
            familiarity=familiarity_report,
            machine_id=machine_id,
            machine_type=machine_type,
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
