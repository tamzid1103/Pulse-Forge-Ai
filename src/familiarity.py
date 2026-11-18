"""
Latent embedding unfamiliarity and domain-shift estimator for PulseForge AI (Milestone M5).

Quantifies how much an acoustic recording's latent representation deviates
from the known normal training distribution (e.g. standard machine speed, noise environment).
A high unfamiliarity score signals domain shift or unsupported operating conditions,
triggering human review (needs_review) rather than an ungrounded automatic decision.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np

from src.schemas import FamiliarityReport
from src.thresholding import select_threshold_quantile

logger = logging.getLogger("pulseforge.familiarity")


class LatentFamiliarityEstimator:
    """
    Standardized Euclidean distance estimator in latent representation space.
    
    Fits the mean centroid and per-dimension variance of normal training embeddings.
    Computes recording-level acoustic distance:
        d(z, mu) = sqrt(sum_i ((z_i - mu_i)^2 / (var_i + epsilon)))
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        aggregation: str = "mean",
    ) -> None:
        self.epsilon = float(epsilon)
        self.aggregation = aggregation
        self.centroid: np.ndarray | None = None
        self.variance: np.ndarray | None = None
        self.threshold: float | None = None
        self.num_samples_fitted: int = 0
        self.latent_dim: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.centroid is not None and self.variance is not None

    def fit(self, embeddings: np.ndarray) -> LatentFamiliarityEstimator:
        """
        Fit reference centroid and variance on normal training embeddings.
        
        Args:
            embeddings: 2D array of shape [N, latent_dim].
        """
        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array [N, latent_dim], got shape {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples to fit familiarity distribution, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Embeddings contain NaN or infinite values.")

        self.num_samples_fitted = arr.shape[0]
        self.latent_dim = arr.shape[1]
        self.centroid = np.mean(arr, axis=0)
        self.variance = np.var(arr, axis=0) + self.epsilon

        logger.info(
            "Fitted LatentFamiliarityEstimator on %d samples (dim=%d, eps=%.1e)",
            self.num_samples_fitted,
            self.latent_dim,
            self.epsilon,
        )
        return self

    def score_segments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute standardized distance for each segment embedding.
        
        Args:
            embeddings: 2D array of shape [num_segments, latent_dim].
            
        Returns:
            1D array of non-negative distances of shape [num_segments].
        """
        if not self.is_fitted:
            raise RuntimeError("Familiarity estimator is not fitted.")
        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected embeddings shape [N, {self.latent_dim}], got {arr.shape}"
            )

        diff = arr - self.centroid
        norm_sq = np.sum((diff ** 2) / self.variance, axis=1)
        distances = np.sqrt(np.maximum(norm_sq, 0.0))
        return distances

    def score_recording(
        self,
        embeddings: np.ndarray,
        aggregation: str | None = None,
    ) -> float:
        """
        Compute recording-level unfamiliarity distance.
        
        Args:
            embeddings: 2D array of segment embeddings.
            aggregation: "mean", "max", or "percentile_95".
        """
        agg = aggregation or self.aggregation
        seg_distances = self.score_segments(embeddings)
        if seg_distances.size == 0:
            return 0.0

        if agg == "mean":
            return float(np.mean(seg_distances))
        elif agg == "max":
            return float(np.max(seg_distances))
        elif agg in ("percentile_95", "p95"):
            return float(np.percentile(seg_distances, 95.0))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    def select_threshold(
        self,
        validation_embeddings_list: list[np.ndarray],
        target_fpr: float = 0.05,
    ) -> float:
        """
        Derive unfamiliarity threshold from validation normal recording embeddings.
        
        Target FPR defaults to 0.05 (5%), meaning 95% of standard normal recordings
        are accepted as familiar.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot select threshold on an unfitted estimator.")
        if not validation_embeddings_list:
            raise ValueError("Validation embeddings list cannot be empty.")

        val_scores = [
            self.score_recording(emb) for emb in validation_embeddings_list
        ]
        self.threshold = select_threshold_quantile(val_scores, target_fpr=target_fpr)
        logger.info(
            "Selected unfamiliarity threshold: %.6f on %d validation recordings (target FPR=%.2f)",
            self.threshold,
            len(val_scores),
            target_fpr,
        )
        return self.threshold

    def predict(
        self,
        embeddings: np.ndarray,
        threshold: float | None = None,
    ) -> FamiliarityReport:
        """
        Evaluate recording embeddings and return a typed FamiliarityReport.
        """
        active_thresh = threshold if threshold is not None else self.threshold
        if active_thresh is None:
            raise RuntimeError(
                "Threshold is not calibrated. Run select_threshold() or pass explicit threshold."
            )

        score = self.score_recording(embeddings)
        is_familiar = bool(score <= active_thresh)

        return FamiliarityReport(
            score=round(score, 4),
            threshold=round(active_thresh, 4),
            is_familiar=is_familiar,
            method="latent_standardized_euclidean",
            details={
                "latent_dim": self.latent_dim,
                "num_segments": embeddings.shape[0] if embeddings.ndim == 2 else 0,
                "aggregation": self.aggregation,
            },
        )

    def save(self, file_path: str | Path) -> None:
        """Save fitted estimator parameters to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted familiarity estimator.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "centroid": self.centroid,
            "variance": self.variance,
            "threshold": self.threshold,
            "latent_dim": self.latent_dim,
            "num_samples_fitted": self.num_samples_fitted,
            "epsilon": self.epsilon,
            "aggregation": self.aggregation,
        }
        joblib.dump(payload, path)
        logger.info("Saved LatentFamiliarityEstimator to %s", path)

    @classmethod
    def load(cls, file_path: str | Path) -> LatentFamiliarityEstimator:
        """Load fitted estimator parameters from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Familiarity estimator artifact not found: {path}")

        payload = joblib.load(path)
        instance = cls(
            epsilon=float(payload.get("epsilon", 1e-4)),
            aggregation=str(payload.get("aggregation", "mean")),
        )
        instance.centroid = payload["centroid"]
        instance.variance = payload["variance"]
        instance.threshold = payload.get("threshold")
        instance.latent_dim = int(payload.get("latent_dim", instance.centroid.shape[0]))
        instance.num_samples_fitted = int(payload.get("num_samples_fitted", 0))
        return instance
