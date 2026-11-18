"""
Domain schemas and typed data contracts for PulseForge AI (Milestone M5).

This module defines standard dataclasses and enums representing:
- Audio file metadata
- Signal quality inspection reports
- Latent embedding familiarity / domain shift reports
- Anomaly detector predictions
- Explicit 4-state decision policies and user-facing analysis results
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DecisionStatus(str, Enum):
    """
    Explicit 4-state operational outcome of the PulseForge AI pipeline.
    
    States:
    - NORMAL: Signal quality and familiarity pass; anomaly score < threshold.
    - POSSIBLE_ANOMALY: Signal quality and familiarity pass; anomaly score >= threshold.
    - NEEDS_REVIEW: Signal quality is poor, or input shows significant domain shift /
                    unfamiliarity compared to training distribution. Automatic decision withheld.
    - INVALID_INPUT: Audio failed decoding, file size / duration violated boundaries,
                     or contains non-finite samples.
    """

    NORMAL = "normal"
    POSSIBLE_ANOMALY = "possible_anomaly"
    NEEDS_REVIEW = "needs_review"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class AudioMetadata:
    """Decoded audio technical characteristics."""

    sample_rate: int
    channels: int
    duration_seconds: float
    num_samples: int
    file_size_bytes: int
    format: str = "wav"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    """
    Audio signal quality assessment report.
    
    Evaluates physical acoustic signal suitability before passing to ML models.
    """

    passed: bool
    rms_level: float
    is_silent: bool
    clipping_ratio: float
    is_clipped: bool
    duration_seconds: float
    is_duration_valid: bool
    dc_offset: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        return cls(
            passed=bool(data["passed"]),
            rms_level=float(data["rms_level"]),
            is_silent=bool(data["is_silent"]),
            clipping_ratio=float(data["clipping_ratio"]),
            is_clipped=bool(data["is_clipped"]),
            duration_seconds=float(data["duration_seconds"]),
            is_duration_valid=bool(data["is_duration_valid"]),
            dc_offset=float(data.get("dc_offset", 0.0)),
            warnings=list(data.get("warnings", [])),
            errors=list(data.get("errors", [])),
        )


@dataclass(frozen=True)
class FamiliarityReport:
    """
    Domain-shift / latent unfamiliarity evaluation report.
    
    Measures acoustic distance from the known normal baseline training distribution.
    A high score flags an unfamiliar acoustic environment (warning/review signal),
    NOT that the anomaly detector is guaranteed incorrect.
    """

    score: float
    threshold: float
    is_familiar: bool
    method: str = "latent_mahalanobis"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FamiliarityReport:
        return cls(
            score=float(data["score"]),
            threshold=float(data["threshold"]),
            is_familiar=bool(data["is_familiar"]),
            method=str(data.get("method", "latent_mahalanobis")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class ModelPrediction:
    """
    Structured output from an acoustic anomaly detector (Autoencoder or Baseline).
    
    IMPORTANT: Anomaly score is an uncalibrated reconstruction error / isolation score.
    It is NEVER a probability or failure percentage.
    """

    model_type: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    aggregation_method: str = "mean"
    segment_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPrediction:
        return cls(
            model_type=str(data["model_type"]),
            anomaly_score=float(data["anomaly_score"]),
            threshold=float(data["threshold"]),
            is_anomaly=bool(data["is_anomaly"]),
            aggregation_method=str(data.get("aggregation_method", "mean")),
            segment_scores=[float(s) for s in data.get("segment_scores", [])],
        )


@dataclass(frozen=True)
class AnalysisResult:
    """
    Complete end-to-end industrial diagnosis report for an uploaded audio recording.
    
    Combines raw signal quality checks, ML model anomaly detection, and
    domain-familiarity checks into an actionable operational decision.
    """

    status: DecisionStatus
    status_label: str
    quality: QualityReport
    model_prediction: ModelPrediction | None = None
    familiarity: FamiliarityReport | None = None
    explanation: str = ""
    recommended_action: str = ""
    machine_id: str = "id_00"
    machine_type: str = "fan"
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: AudioMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        status = DecisionStatus(data["status"])
        quality = QualityReport.from_dict(data["quality"])
        pred = (
            ModelPrediction.from_dict(data["model_prediction"])
            if data.get("model_prediction")
            else None
        )
        fam = (
            FamiliarityReport.from_dict(data["familiarity"])
            if data.get("familiarity")
            else None
        )
        meta = (
            AudioMetadata(**data["metadata"])
            if data.get("metadata")
            else None
        )
        return cls(
            status=status,
            status_label=str(data.get("status_label", status.value)),
            quality=quality,
            model_prediction=pred,
            familiarity=fam,
            explanation=str(data.get("explanation", "")),
            recommended_action=str(data.get("recommended_action", "")),
            machine_id=str(data.get("machine_id", "id_00")),
            machine_type=str(data.get("machine_type", "fan")),
            timestamp=str(data.get("timestamp", "")),
            metadata=meta,
        )
