"""
Baseline Anomaly Detector for PulseForge AI using Isolation Forest.

Milestone M3: Implements a reproducible baseline anomaly detector on
pooled log-Mel spectrogram feature representations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import BaselineConfig
from src.features import FeatureTensor

logger = logging.getLogger("pulseforge.models.baseline")


@dataclass(frozen=True)
class BaselinePrediction:
    """Prediction result from the Baseline Anomaly Detector."""

    anomaly_score: float
    threshold: float
    is_anomaly: bool
    status: Literal["normal", "possible_anomaly"]
    segment_scores: list[float]
    score_aggregation: str


class BaselineIsolationForest:
    """
    Isolation Forest anomaly detector for acoustic features.

    Trains exclusively on normal acoustic patterns. Anomaly scores are
    inverted such that higher score indicates higher degree of anomaly.
    Threshold is determined strictly from validation data.
    """

    def __init__(
        self,
        config: BaselineConfig | None = None,
        threshold: float | None = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self.threshold = threshold
        self.model: IsolationForest | None = None
        self.feature_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def pool_features(self, tensor: FeatureTensor) -> np.ndarray:
        """
        Convert a 3D FeatureTensor (num_segments, n_mels, n_frames) into a 2D
        feature matrix (num_segments, feature_dim) using the configured pooling strategy.

        Strategies:
            - 'mean_std': per Mel band mean & std across time frames -> (num_segments, 2 * n_mels)
            - 'mean': per Mel band mean across time frames -> (num_segments, n_mels)
            - 'flatten': full segment unrolled -> (num_segments, n_mels * n_frames)
        """
        feats = tensor.features  # Shape: (N, n_mels, n_frames)
        if feats.ndim != 3:
            raise ValueError(f"Expected 3D feature array, got shape {feats.shape}")

        strategy = self.config.pooling_strategy

        if strategy == "mean_std":
            # Axis 2 is time frames
            mean = np.mean(feats, axis=2)  # (N, n_mels)
            std = np.std(feats, axis=2)    # (N, n_mels)
            pooled = np.concatenate([mean, std], axis=1)  # (N, 2 * n_mels)
        elif strategy == "mean":
            pooled = np.mean(feats, axis=2)  # (N, n_mels)
        elif strategy == "flatten":
            pooled = tensor.to_flattened()  # (N, n_mels * n_frames)
        else:
            raise ValueError(f"Unknown pooling strategy: '{strategy}'")

        if not np.all(np.isfinite(pooled)):
            raise ValueError("Pooled baseline features contain NaN or infinite values.")

        return pooled.astype(np.float32)

    def fit(self, training_vectors: np.ndarray) -> BaselineIsolationForest:
        """
        Fit the Isolation Forest strictly on normal training feature vectors.

        Parameters:
            training_vectors: 2D array of shape (num_samples, feature_dim).
        """
        if training_vectors.ndim != 2:
            raise ValueError(f"Expected 2D training matrix (samples, features), got {training_vectors.shape}")

        self.feature_dim = training_vectors.shape[1]
        logger.info(
            f"Fitting IsolationForest on {training_vectors.shape[0]} samples with "
            f"{self.feature_dim} features (n_estimators={self.config.n_estimators}, "
            f"random_state={self.config.random_state})."
        )

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(training_vectors)
        logger.info("Isolation Forest baseline fitting completed.")
        return self

    def predict_segment_scores(self, tensor: FeatureTensor) -> np.ndarray:
        """
        Compute anomaly score for each segment in the FeatureTensor.

        In scikit-learn, score_samples() returns negative anomaly scores
        (more negative = more anomalous). We invert the sign so that higher
        scores indicate greater anomaly.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("BaselineIsolationForest must be fitted before scoring.")

        pooled = self.pool_features(tensor)
        if pooled.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {pooled.shape[1]}"
            )

        # Invert score_samples so higher = more anomalous
        scores = -self.model.score_samples(pooled)
        return scores.astype(np.float32)

    def predict_recording_score(self, tensor: FeatureTensor) -> float:
        """
        Compute an aggregated recording-level anomaly score from segment scores.
        """
        seg_scores = self.predict_segment_scores(tensor)
        agg = self.config.score_aggregation

        if agg == "mean":
            rec_score = float(np.mean(seg_scores))
        elif agg == "max":
            rec_score = float(np.max(seg_scores))
        elif agg == "percentile_95":
            rec_score = float(np.percentile(seg_scores, 95))
        else:
            rec_score = float(np.mean(seg_scores))

        return rec_score

    def select_threshold(
        self,
        val_normal_scores: list[float] | np.ndarray,
        val_abnormal_scores: list[float] | np.ndarray | None = None,
        target_fpr: float | None = None,
    ) -> float:
        """
        Derive anomaly threshold strictly from validation data.

        If only normal validation scores are provided (or by default),
        threshold is set at the (1 - target_fpr) quantile of normal scores.
        If abnormal scores are also provided, an optimal F1 threshold is also
        evaluated and logged.
        """
        arr_norm = np.array(val_normal_scores, dtype=np.float32)
        if len(arr_norm) == 0:
            raise ValueError("val_normal_scores cannot be empty.")

        fpr = target_fpr if target_fpr is not None else self.config.target_fpr
        # Quantile-based threshold: (1 - target_fpr) percentile of normal validation scores
        quantile = (1.0 - fpr) * 100.0
        threshold = float(np.percentile(arr_norm, quantile))
        logger.info(f"Selected baseline threshold {threshold:.4f} at target FPR={fpr:.2f} (quantile={quantile:.1f}%)")

        if val_abnormal_scores is not None and len(val_abnormal_scores) > 0:
            arr_abn = np.array(val_abnormal_scores, dtype=np.float32)
            # Evaluate empirical precision, recall, F1 at this threshold on validation
            y_true = np.array([0] * len(arr_norm) + [1] * len(arr_abn))
            scores = np.concatenate([arr_norm, arr_abn])
            preds = (scores >= threshold).astype(int)

            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            logger.info(
                f"Validation set performance at threshold {threshold:.4f}: "
                f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
            )

        self.threshold = threshold
        return threshold

    def predict(self, tensor: FeatureTensor) -> BaselinePrediction:
        """
        Run end-to-end inference on a FeatureTensor and return a structured BaselinePrediction.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold must be selected or provided before running prediction.")

        seg_scores = self.predict_segment_scores(tensor)
        rec_score = self.predict_recording_score(tensor)

        is_anomaly = bool(rec_score >= self.threshold)
        status: Literal["normal", "possible_anomaly"] = (
            "possible_anomaly" if is_anomaly else "normal"
        )

        return BaselinePrediction(
            anomaly_score=round(rec_score, 4),
            threshold=round(self.threshold, 4),
            is_anomaly=is_anomaly,
            status=status,
            segment_scores=[round(float(s), 4) for s in seg_scores],
            score_aggregation=self.config.score_aggregation,
        )

    def save(self, filepath: Path | str) -> tuple[Path, Path]:
        """
        Save the fitted model (.joblib) and metadata (.json) to disk.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted BaselineIsolationForest.")

        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.suffix != ".joblib":
            model_path = model_path.with_suffix(".joblib")

        meta_path = model_path.with_suffix(".json")

        joblib.dump(self.model, model_path)
        logger.info(f"Saved baseline IsolationForest weights to {model_path}")

        meta = {
            "model_type": "IsolationForest",
            "threshold": self.threshold,
            "feature_dim": self.feature_dim,
            "config": {
                "n_estimators": self.config.n_estimators,
                "max_samples": self.config.max_samples,
                "contamination": self.config.contamination,
                "random_state": self.config.random_state,
                "pooling_strategy": self.config.pooling_strategy,
                "score_aggregation": self.config.score_aggregation,
                "target_fpr": self.config.target_fpr,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved baseline metadata to {meta_path}")

        return model_path, meta_path

    @classmethod
    def load(cls, filepath: Path | str) -> BaselineIsolationForest:
        """
        Load a fitted baseline model from disk.
        """
        model_path = Path(filepath)
        if model_path.suffix != ".joblib":
            model_path = model_path.with_suffix(".joblib")

        meta_path = model_path.with_suffix(".json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_model = joblib.load(model_path)

        meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        cfg_dict = meta.get("config", {})
        config = BaselineConfig(
            n_estimators=cfg_dict.get("n_estimators", 100),
            max_samples=cfg_dict.get("max_samples", "auto"),
            contamination=cfg_dict.get("contamination", "auto"),
            random_state=cfg_dict.get("random_state", 42),
            pooling_strategy=cfg_dict.get("pooling_strategy", "mean_std"),
            score_aggregation=cfg_dict.get("score_aggregation", "mean"),
            target_fpr=cfg_dict.get("target_fpr", 0.10),
        )

        detector = cls(config=config, threshold=meta.get("threshold"))
        detector.model = loaded_model
        detector.feature_dim = meta.get("feature_dim", loaded_model.n_features_in_)
        logger.info(f"Loaded BaselineIsolationForest from {model_path} (threshold={detector.threshold})")
        return detector
