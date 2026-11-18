"""
Centralized configuration for PulseForge AI.

All configurable parameters (sample rate, feature settings, thresholds,
upload limits, paths) live here. Do not duplicate these values across
modules. Configuration can be overridden by environment variables
prefixed with PULSEFORGE_.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pulseforge")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("true", "1", "yes")


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# ---------------------------------------------------------------------------
# Audio / preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioConfig:
    """Audio loading and preprocessing parameters."""

    # Target sample rate after resampling (Hz).
    # MIMII source audio is documented as 16 kHz.
    sample_rate: int = 16_000

    # Channel selection strategy: "first" selects channel 0, "mean" averages.
    channel_strategy: str = "first"

    # Supported audio format(s) — content-validated, not by extension alone.
    supported_formats: tuple[str, ...] = ("wav",)

    # Upload constraints
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """Log-Mel spectrogram and feature parameters."""

    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float | None = None  # None → Nyquist
    segment_length_seconds: float = 1.0  # seconds per model input segment
    segment_hop_seconds: float = 0.5     # overlap between segments
    power: float = 2.0  # power spectrogram


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineConfig:
    """Isolation Forest baseline model parameters."""

    n_estimators: int = 100
    max_samples: float | str = "auto"
    contamination: float | str = "auto"
    random_state: int = 42
    # Feature pooling strategy: "mean_std" (mean and std across time per Mel band: 256 dims),
    # "mean" (mean across time per Mel band: 128 dims), or "flatten" (flattened segments: 4096 dims)
    pooling_strategy: str = "mean_std"
    # Aggregation across segments for recording score: "mean", "max", "percentile_95"
    score_aggregation: str = "mean"
    # Target false-positive rate on validation normal samples for threshold selection
    target_fpr: float = 0.10


@dataclass(frozen=True)
class ModelConfig:
    """Autoencoder model parameters."""

    # Autoencoder architecture
    input_dim: int = 4096  # 128 Mel bins * 32 time frames
    hidden_dims: tuple[int, ...] = (512, 128, 64)
    latent_dim: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    seed: int = 42

    # Score aggregation: "mean", "max", "percentile_95"
    score_aggregation: str = "mean"

    # Target false-positive rate on validation normal samples for threshold selection
    target_fpr: float = 0.10


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityConfig:
    """Signal-quality check thresholds."""

    # RMS below this → near-silent
    silence_rms_threshold: float = 0.001

    # Fraction of samples at ±1.0 above this → clipping warning
    clipping_ratio_threshold: float = 0.01

    # Minimum acceptable duration after decode (seconds)
    min_analysis_duration: float = 0.5

    # Maximum acceptable duration after decode (seconds)
    max_analysis_duration: float = 30.0

    # DC bias magnitude above which a warning is issued
    dc_offset_threshold: float = 0.1


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionConfig:
    """Decision policy parameters."""

    # These are placeholders; real thresholds are set from validation data.
    anomaly_threshold: float = 0.0  # to be determined from training

    # Unfamiliarity distance above this triggers review
    unfamiliarity_threshold: float = 0.0  # to be determined from training


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Project directory paths. Relative to project root."""

    project_root: Path = field(default_factory=_project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def manifests_dir(self) -> Path:
        return self.project_root / "data" / "manifests"

    @property
    def demo_audio_dir(self) -> Path:
        raw = _env("PULSEFORGE_DEMO_AUDIO_DIR", "")
        if raw:
            return Path(raw)
        return self.project_root / "demo_audio"

    @property
    def model_artifacts_dir(self) -> Path:
        raw = _env("PULSEFORGE_MODEL_DIR", "")
        if raw:
            return Path(raw)
        return self.project_root / "model_artifacts"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"


# ---------------------------------------------------------------------------
# Top-level app config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration bundle."""

    env: str = field(default_factory=lambda: _env("PULSEFORGE_ENV", "development"))
    log_level: str = field(default_factory=lambda: _env("PULSEFORGE_LOG_LEVEL", "INFO"))
    enable_history: bool = field(
        default_factory=lambda: _env_bool("PULSEFORGE_ENABLE_HISTORY", False)
    )

    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Machine type — fan only in MVP
    supported_machine_types: tuple[str, ...] = ("fan",)
    default_machine_type: str = "fan"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(numeric_level)


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

def get_default_config() -> AppConfig:
    """Return the default application configuration."""
    return AppConfig()
