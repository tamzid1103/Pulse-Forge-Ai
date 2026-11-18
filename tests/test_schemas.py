"""
Unit tests for domain schemas and data structures (Milestone M5).
"""

from __future__ import annotations

import pytest

from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)


class TestSchemas:
    """Validate serialization and integrity of M5 domain objects."""

    def test_decision_status_enum_values(self) -> None:
        assert DecisionStatus.NORMAL.value == "normal"
        assert DecisionStatus.POSSIBLE_ANOMALY.value == "possible_anomaly"
        assert DecisionStatus.NEEDS_REVIEW.value == "needs_review"
        assert DecisionStatus.INVALID_INPUT.value == "invalid_input"

    def test_audio_metadata_dict(self) -> None:
        meta = AudioMetadata(
            sample_rate=16000,
            channels=1,
            duration_seconds=10.0,
            num_samples=160000,
            file_size_bytes=320044,
            format="wav",
        )
        d = meta.to_dict()
        assert d["sample_rate"] == 16000
        assert d["duration_seconds"] == 10.0

    def test_quality_report_roundtrip(self) -> None:
        qr = QualityReport(
            passed=True,
            rms_level=0.045,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
            dc_offset=0.001,
            warnings=["Slight DC bias detected"],
            errors=[],
        )
        d = qr.to_dict()
        reconstructed = QualityReport.from_dict(d)
        assert reconstructed == qr
        assert reconstructed.passed is True
        assert reconstructed.warnings == ["Slight DC bias detected"]

    def test_familiarity_report_roundtrip(self) -> None:
        fr = FamiliarityReport(
            score=1.42,
            threshold=2.15,
            is_familiar=True,
            method="latent_mahalanobis",
            details={"dimension": 32, "top_component": 0.8},
        )
        d = fr.to_dict()
        reconstructed = FamiliarityReport.from_dict(d)
        assert reconstructed == fr
        assert reconstructed.is_familiar is True
        assert reconstructed.details["dimension"] == 32

    def test_model_prediction_roundtrip(self) -> None:
        mp = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=0.942,
            threshold=0.890,
            is_anomaly=True,
            aggregation_method="mean",
            segment_scores=[0.91, 0.95, 0.97],
        )
        d = mp.to_dict()
        reconstructed = ModelPrediction.from_dict(d)
        assert reconstructed == mp
        assert reconstructed.is_anomaly is True
        assert len(reconstructed.segment_scores) == 3

    def test_analysis_result_roundtrip(self) -> None:
        qr = QualityReport(
            passed=True,
            rms_level=0.05,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        )
        mp = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=0.95,
            threshold=0.89,
            is_anomaly=True,
        )
        fr = FamiliarityReport(
            score=1.2,
            threshold=2.0,
            is_familiar=True,
        )
        result = AnalysisResult(
            status=DecisionStatus.POSSIBLE_ANOMALY,
            status_label="Possible Acoustic Anomaly",
            quality=qr,
            model_prediction=mp,
            familiarity=fr,
            explanation="Unusual acoustic pattern detected above baseline reconstruction error threshold.",
            recommended_action="Please inspect the machine and verify the recording.",
            machine_id="id_00",
            machine_type="fan",
        )
        d = result.to_dict()
        assert d["status"] == "possible_anomaly"
        reconstructed = AnalysisResult.from_dict(d)
        assert reconstructed.status == DecisionStatus.POSSIBLE_ANOMALY
        assert reconstructed.quality.passed is True
        assert reconstructed.model_prediction is not None
        assert reconstructed.model_prediction.is_anomaly is True
        assert reconstructed.familiarity is not None
        assert reconstructed.familiarity.is_familiar is True
