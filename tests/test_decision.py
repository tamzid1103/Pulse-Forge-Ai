"""
Unit tests for industrial decision policy and explanation engine (Milestone M5).
"""

from __future__ import annotations

import pytest

from src.decision import create_invalid_input_result, evaluate_decision_policy
from src.schemas import (
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)


class TestDecisionPolicy:
    """Validate all deterministic branches of the 4-state decision policy."""

    @pytest.fixture
    def passing_quality(self) -> QualityReport:
        return QualityReport(
            passed=True,
            rms_level=0.05,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
        )

    @pytest.fixture
    def familiar_report(self) -> FamiliarityReport:
        return FamiliarityReport(
            score=1.5,
            threshold=3.0,
            is_familiar=True,
        )

    @pytest.fixture
    def unfamiliar_report(self) -> FamiliarityReport:
        return FamiliarityReport(
            score=5.8,
            threshold=3.0,
            is_familiar=False,
        )

    def test_decision_normal(
        self, passing_quality: QualityReport, familiar_report: FamiliarityReport
    ) -> None:
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=0.45,
            threshold=0.89,
            is_anomaly=False,
        )
        res = evaluate_decision_policy(
            quality=passing_quality,
            model_prediction=pred,
            familiarity=familiar_report,
        )
        assert res.status == DecisionStatus.NORMAL
        assert "normal" in res.status_label.lower()
        assert "below the threshold" in res.explanation.lower()

    def test_decision_possible_anomaly(
        self, passing_quality: QualityReport, familiar_report: FamiliarityReport
    ) -> None:
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=1.25,
            threshold=0.89,
            is_anomaly=True,
        )
        res = evaluate_decision_policy(
            quality=passing_quality,
            model_prediction=pred,
            familiarity=familiar_report,
        )
        assert res.status == DecisionStatus.POSSIBLE_ANOMALY
        assert "possible acoustic anomaly" in res.status_label.lower()
        assert "unusual acoustic pattern detected" in res.explanation.lower()
        assert "inspect the machine" in res.recommended_action.lower()

    def test_decision_silent_input_needs_review(
        self, familiar_report: FamiliarityReport
    ) -> None:
        silent_quality = QualityReport(
            passed=False,
            rms_level=0.00001,
            is_silent=True,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=10.0,
            is_duration_valid=True,
            errors=["Recording is silent or near-silent."],
        )
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=0.2,
            threshold=0.89,
            is_anomaly=False,
        )
        res = evaluate_decision_policy(
            quality=silent_quality,
            model_prediction=pred,
            familiarity=familiar_report,
        )
        assert res.status == DecisionStatus.NEEDS_REVIEW
        assert "quality is poor" in res.explanation.lower()

    def test_decision_clipped_input_needs_review(
        self, familiar_report: FamiliarityReport
    ) -> None:
        clipped_quality = QualityReport(
            passed=False,
            rms_level=0.5,
            is_silent=False,
            clipping_ratio=0.15,
            is_clipped=True,
            duration_seconds=10.0,
            is_duration_valid=True,
            warnings=["Severe digital clipping detected."],
        )
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=1.5,
            threshold=0.89,
            is_anomaly=True,
        )
        res = evaluate_decision_policy(
            quality=clipped_quality,
            model_prediction=pred,
            familiarity=familiar_report,
        )
        assert res.status == DecisionStatus.NEEDS_REVIEW
        assert "quality is poor" in res.explanation.lower()

    def test_decision_invalid_duration(self) -> None:
        invalid_quality = QualityReport(
            passed=False,
            rms_level=0.05,
            is_silent=False,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=0.2,
            is_duration_valid=False,
            errors=["Audio duration (0.20s) is shorter than minimum required (0.50s)."],
        )
        res = evaluate_decision_policy(quality=invalid_quality)
        assert res.status == DecisionStatus.INVALID_INPUT
        assert "technical input validation" in res.explanation.lower()

    def test_decision_unfamiliar_domain_shift(
        self, passing_quality: QualityReport, unfamiliar_report: FamiliarityReport
    ) -> None:
        # Score is normal, but unfamiliarity is high
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=0.5,
            threshold=0.89,
            is_anomaly=False,
        )
        res = evaluate_decision_policy(
            quality=passing_quality,
            model_prediction=pred,
            familiarity=unfamiliar_report,
        )
        assert res.status == DecisionStatus.NEEDS_REVIEW
        assert "differs from the training conditions" in res.explanation.lower()

    def test_decision_unfamiliar_and_anomalous_both_reported(
        self, passing_quality: QualityReport, unfamiliar_report: FamiliarityReport
    ) -> None:
        # Both unfamiliarity and reconstruction error are high
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=1.65,
            threshold=0.89,
            is_anomaly=True,
        )
        res = evaluate_decision_policy(
            quality=passing_quality,
            model_prediction=pred,
            familiarity=unfamiliar_report,
        )
        assert res.status == DecisionStatus.NEEDS_REVIEW
        # Must report both signals prominently
        assert "unfamiliarity" in res.explanation.lower()
        assert "reconstruction score" in res.explanation.lower()

    def test_language_rules_compliance(
        self, passing_quality: QualityReport, familiar_report: FamiliarityReport
    ) -> None:
        """Verify forbidden speculative claims never appear in generated text."""
        forbidden_phrases = [
            "fail in",
            "bearing failure confirmed",
            "probability of failure",
            "maintenance guaranteed",
            "will fail",
        ]
        pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=2.0,
            threshold=0.89,
            is_anomaly=True,
        )
        res = evaluate_decision_policy(
            quality=passing_quality,
            model_prediction=pred,
            familiarity=familiar_report,
        )
        combined_text = (res.explanation + " " + res.recommended_action).lower()
        for phrase in forbidden_phrases:
            assert phrase not in combined_text, f"Forbidden phrase '{phrase}' found in decision output"
