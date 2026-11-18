"""
Industrial decision policy and explanation engine for PulseForge AI (Milestone M5).

Translates acoustic signal quality metrics, autoencoder anomaly predictions,
and latent domain-familiarity assessments into four explicit operational states:
1. NORMAL: Passed quality and familiarity checks; anomaly score < threshold.
2. POSSIBLE_ANOMALY: Passed quality and familiarity; anomaly score >= threshold.
3. NEEDS_REVIEW: Quality is poor (silent/clipped) or acoustic characteristics show
                 significant domain shift from training conditions.
4. INVALID_INPUT: Corrupted audio, non-finite values, or duration limits violated.

Adheres strictly to AGENTS.md language guidelines:
- Never labels anomaly scores as failure probabilities or confidence percentages.
- Recommends physical inspection and human review without confirming specific internal component failures.
"""

from __future__ import annotations

import logging
from typing import Any

from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)

logger = logging.getLogger("pulseforge.decision")


def create_invalid_input_result(
    error_message: str,
    metadata: AudioMetadata | None = None,
    machine_id: str = "id_00",
    machine_type: str = "fan",
) -> AnalysisResult:
    """Helper to construct an explicit INVALID_INPUT AnalysisResult."""
    qr = QualityReport(
        passed=False,
        rms_level=0.0,
        is_silent=False,
        clipping_ratio=0.0,
        is_clipped=False,
        duration_seconds=metadata.duration_seconds if metadata else 0.0,
        is_duration_valid=False,
        errors=[error_message],
    )
    return AnalysisResult(
        status=DecisionStatus.INVALID_INPUT,
        status_label="Invalid Audio Input",
        quality=qr,
        model_prediction=None,
        familiarity=None,
        explanation=f"Audio input could not be validated: {error_message}",
        recommended_action="Please provide a valid, uncorrupted WAV recording between 0.5s and 30.0s duration.",
        machine_id=machine_id,
        machine_type=machine_type,
        metadata=metadata,
    )


def evaluate_decision_policy(
    quality: QualityReport,
    model_prediction: ModelPrediction | None = None,
    familiarity: FamiliarityReport | None = None,
    machine_id: str = "id_00",
    machine_type: str = "fan",
    metadata: AudioMetadata | None = None,
) -> AnalysisResult:
    """
    Evaluate the full PulseForge AI decision policy.
    
    Deterministic evaluation order:
    1. Critical input invalidity -> INVALID_INPUT
    2. Quality failure (silence, severe clipping) -> NEEDS_REVIEW
    3. Unfamiliar domain shift -> NEEDS_REVIEW (with explanation of anomaly if present)
    4. Model prediction evaluated -> POSSIBLE_ANOMALY or NORMAL
    
    Args:
        quality: Signal quality report.
        model_prediction: Model prediction (optional if quality failed prior to inference).
        familiarity: Latent familiarity report (optional).
        machine_id: Machine identifier string.
        machine_type: Machine family string (must be "fan" in MVP).
        metadata: Audio metadata.
        
    Returns:
        Structured AnalysisResult.
    """
    # 1. Critical Input Errors (duration invalid, non-finite samples)
    if not quality.is_duration_valid or (
        quality.errors and any("nan" in e.lower() or "duration" in e.lower() for e in quality.errors)
    ):
        err_msg = "; ".join(quality.errors) if quality.errors else "Duration boundary violation."
        return AnalysisResult(
            status=DecisionStatus.INVALID_INPUT,
            status_label="Invalid Audio Input",
            quality=quality,
            model_prediction=model_prediction,
            familiarity=familiarity,
            explanation=f"Audio file failed technical input validation: {err_msg}",
            recommended_action="Please supply an uncorrupted WAV audio file with duration between 0.5s and 30.0s.",
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )

    # 2. Quality Failure (silent or clipped)
    if not quality.passed:
        reasons = quality.errors + quality.warnings
        detail = "; ".join(reasons) if reasons else "Audio quality metrics below acceptable thresholds."
        return AnalysisResult(
            status=DecisionStatus.NEEDS_REVIEW,
            status_label="Signal Quality Rejected — Needs Review",
            quality=quality,
            model_prediction=model_prediction,
            familiarity=familiarity,
            explanation=f"Recording quality is poor; result may be unreliable. {detail}",
            recommended_action="Please inspect sensor placement and re-record in clean operating conditions.",
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )

    # 3. Domain Shift / Unfamiliarity Check
    if familiarity is not None and not familiarity.is_familiar:
        fam_text = (
            f"Input differs from the training conditions; human review is recommended. "
            f"Acoustic unfamiliarity score ({familiarity.score:.2f}) exceeds reference limit "
            f"({familiarity.threshold:.2f})."
        )
        if model_prediction is not None and model_prediction.is_anomaly:
            fam_text += (
                f" In addition, elevated reconstruction score ({model_prediction.anomaly_score:.4f} >= "
                f"threshold {model_prediction.threshold:.4f}) suggests possible anomalous behavior."
            )
            rec_action = (
                "Please inspect the machine and verify the recording conditions. "
                "Both acoustic unfamiliarity and elevated reconstruction error were detected."
            )
        else:
            rec_action = (
                "Please inspect the machine environment and verify operating conditions before "
                "relying on automated baseline screening."
            )

        return AnalysisResult(
            status=DecisionStatus.NEEDS_REVIEW,
            status_label="Acoustic Pattern Needs Human Review",
            quality=quality,
            model_prediction=model_prediction,
            familiarity=familiarity,
            explanation=fam_text,
            recommended_action=rec_action,
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )

    # If no model prediction is provided but quality passed, needs review
    if model_prediction is None:
        return AnalysisResult(
            status=DecisionStatus.NEEDS_REVIEW,
            status_label="Model Prediction Unavailable — Needs Review",
            quality=quality,
            model_prediction=None,
            familiarity=familiarity,
            explanation="Acoustic quality verified, but model inference result is unavailable.",
            recommended_action="Verify model checkpoint availability and rerun analysis.",
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )

    # 4. Anomaly Decision Paths
    if model_prediction.is_anomaly:
        return AnalysisResult(
            status=DecisionStatus.POSSIBLE_ANOMALY,
            status_label="Possible Acoustic Anomaly Detected",
            quality=quality,
            model_prediction=model_prediction,
            familiarity=familiarity,
            explanation=(
                f"Unusual acoustic pattern detected. Recording anomaly score "
                f"({model_prediction.anomaly_score:.4f}) exceeds the calibrated baseline threshold "
                f"({model_prediction.threshold:.4f})."
            ),
            recommended_action="Please inspect the machine and verify the recording.",
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )
    else:
        return AnalysisResult(
            status=DecisionStatus.NORMAL,
            status_label="Normal Acoustic Pattern",
            quality=quality,
            model_prediction=model_prediction,
            familiarity=familiarity,
            explanation=(
                f"Acoustic pattern matches normal baseline operating parameters. "
                f"Anomaly score ({model_prediction.anomaly_score:.4f}) is below the threshold "
                f"({model_prediction.threshold:.4f})."
            ),
            recommended_action="Routine operational monitoring. No acoustic anomaly detected.",
            machine_id=machine_id,
            machine_type=machine_type,
            metadata=metadata,
        )
