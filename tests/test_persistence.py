"""
Unit tests for SQLite audit persistence (Milestone M6).
"""

from __future__ import annotations

import pytest

from src.persistence import AnalysisStore
from src.schemas import (
    AnalysisResult,
    AudioMetadata,
    DecisionStatus,
    FamiliarityReport,
    ModelPrediction,
    QualityReport,
)


@pytest.fixture
def memory_store() -> AnalysisStore:
    """Create an isolated in-memory SQLite store."""
    return AnalysisStore(db_path=":memory:")


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    qr = QualityReport(
        passed=True,
        rms_level=0.045,
        is_silent=False,
        clipping_ratio=0.0,
        is_clipped=False,
        duration_seconds=10.0,
        is_duration_valid=True,
    )
    mp = ModelPrediction(
        model_type="AcousticAutoencoder",
        anomaly_score=0.42,
        threshold=0.89,
        is_anomaly=False,
    )
    fam = FamiliarityReport(
        score=2.1,
        threshold=4.46,
        is_familiar=True,
    )
    meta = AudioMetadata(
        sample_rate=16000,
        channels=1,
        duration_seconds=10.0,
        num_samples=160000,
        file_size_bytes=320044,
    )
    return AnalysisResult(
        status=DecisionStatus.NORMAL,
        status_label="Normal Acoustic Pattern",
        quality=qr,
        model_prediction=mp,
        familiarity=fam,
        explanation="Normal operation.",
        recommended_action="Routine monitoring.",
        machine_id="id_00",
        machine_type="fan",
        metadata=meta,
    )


class TestPersistence:
    """Validate SQLite audit logging and retrieval."""

    def test_save_and_get_by_id(
        self, memory_store: AnalysisStore, sample_analysis_result: AnalysisResult
    ) -> None:
        rec_id = memory_store.save_analysis(
            result=sample_analysis_result,
            processing_ms=45,
            source_kind="demo_sample",
            input_label="01_normal_fan.wav",
        )
        assert rec_id is not None
        assert len(rec_id) > 10

        run = memory_store.get_run_by_id(rec_id)
        assert run is not None
        assert run["id"] == rec_id
        assert run["decision_status"] == "normal"
        assert run["machine_id"] == "id_00"
        assert run["source_kind"] == "demo_sample"
        assert run["input_label"] == "01_normal_fan.wav"
        assert run["anomaly_score"] == pytest.approx(0.42, abs=1e-3)
        assert run["threshold"] == pytest.approx(0.89, abs=1e-3)
        assert run["processing_ms"] == 45

    def test_recent_runs_ordering_and_limit(
        self, memory_store: AnalysisStore, sample_analysis_result: AnalysisResult
    ) -> None:
        # Save 3 records
        for i in range(3):
            memory_store.save_analysis(
                sample_analysis_result,
                input_label=f"test_{i}.wav",
            )

        runs = memory_store.get_recent_runs(limit=2)
        assert len(runs) == 2
        assert runs[0]["input_label"] == "test_2.wav"

    def test_summary_stats(
        self, memory_store: AnalysisStore, sample_analysis_result: AnalysisResult
    ) -> None:
        # Save normal
        memory_store.save_analysis(sample_analysis_result, processing_ms=30)

        # Save anomaly
        mp_anomaly = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=1.5,
            threshold=0.89,
            is_anomaly=True,
        )
        anomaly_result = AnalysisResult(
            status=DecisionStatus.POSSIBLE_ANOMALY,
            status_label="Possible Acoustic Anomaly Detected",
            quality=sample_analysis_result.quality,
            model_prediction=mp_anomaly,
        )
        memory_store.save_analysis(anomaly_result, processing_ms=50)

        stats = memory_store.get_summary_stats()
        assert stats["total_inspections"] == 2
        assert stats["status_counts"]["normal"] == 1
        assert stats["status_counts"]["possible_anomaly"] == 1
        assert stats["average_latency_ms"] == 40.0

    def test_clear_history(
        self, memory_store: AnalysisStore, sample_analysis_result: AnalysisResult
    ) -> None:
        memory_store.save_analysis(sample_analysis_result)
        assert len(memory_store.get_recent_runs()) == 1

        memory_store.clear_history()
        assert len(memory_store.get_recent_runs()) == 0

    def test_file_db_concurrency_pragmas(self, tmp_path) -> None:
        """Verify WAL mode and busy timeout are configured on file-backed databases."""
        file_db = tmp_path / "test_concurrency.db"
        store = AnalysisStore(db_path=file_db)
        with store._get_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"

