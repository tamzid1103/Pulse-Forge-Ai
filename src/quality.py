"""
Acoustic signal quality evaluation for PulseForge AI (Milestone M5).

Evaluates whether decoded audio meets industrial analysis standards
before passing features into machine learning anomaly detection models.
Detects:
- Audio duration boundaries (too short or too long)
- Silence or near-silent sensor/microphone failure
- Severe digital clipping (microphone gain overflow / saturation)
- Severe DC offset / electronic sensor drift
- Non-finite samples (NaN / Inf)
"""

from __future__ import annotations

import logging
import numpy as np

from src.config import QualityConfig
from src.schemas import QualityReport

logger = logging.getLogger("pulseforge.quality")


def compute_rms(signal: np.ndarray) -> float:
    """
    Compute Root-Mean-Square (RMS) amplitude of a 1D audio signal.
    
    Formula: RMS = sqrt(mean(signal^2))
    """
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def compute_clipping_ratio(signal: np.ndarray, threshold: float = 0.999) -> float:
    """
    Calculate proportion of samples saturated near digital maximum (+/- 1.0).
    
    Args:
        signal: 1D audio waveform array in range [-1.0, 1.0].
        threshold: Absolute value threshold considered clipped (default 0.999).
        
    Returns:
        Fraction of samples exceeding threshold [0.0, 1.0].
    """
    if signal.size == 0:
        return 0.0
    clipped_count = np.count_nonzero(np.abs(signal) >= threshold)
    return float(clipped_count / signal.size)


def compute_dc_offset(signal: np.ndarray) -> float:
    """Compute the mean DC offset / electronic drift of the signal."""
    if signal.size == 0:
        return 0.0
    return float(np.mean(signal, dtype=np.float64))


def check_signal_quality(
    signal: np.ndarray,
    sample_rate: int,
    config: QualityConfig | None = None,
) -> QualityReport:
    """
    Perform comprehensive quality verification on a single-channel audio waveform.
    
    Args:
        signal: 1D numpy array of audio samples.
        sample_rate: Sampling frequency in Hz.
        config: Quality check configuration thresholds (defaults to QualityConfig()).
        
    Returns:
        QualityReport with passed flag, metrics, warnings, and errors.
    """
    if config is None:
        config = QualityConfig()

    warnings: list[str] = []
    errors: list[str] = []

    # 1. Non-finite values check
    if not np.all(np.isfinite(signal)):
        errors.append("Audio signal contains NaN or infinite values.")
        return QualityReport(
            passed=False,
            rms_level=0.0,
            is_silent=True,
            clipping_ratio=0.0,
            is_clipped=False,
            duration_seconds=0.0,
            is_duration_valid=False,
            dc_offset=0.0,
            warnings=warnings,
            errors=errors,
        )

    # 2. Duration calculation & check
    duration = float(signal.size / sample_rate) if sample_rate > 0 else 0.0
    is_duration_valid = True

    if duration < config.min_analysis_duration:
        is_duration_valid = False
        errors.append(
            f"Audio duration ({duration:.2f}s) is shorter than minimum required "
            f"({config.min_analysis_duration:.2f}s)."
        )
    elif duration > config.max_analysis_duration:
        is_duration_valid = False
        errors.append(
            f"Audio duration ({duration:.2f}s) exceeds maximum analysis limit "
            f"({config.max_analysis_duration:.2f}s)."
        )

    # 3. RMS & Silence check
    rms = compute_rms(signal)
    is_silent = False
    if rms < config.silence_rms_threshold:
        is_silent = True
        errors.append(
            f"Recording is silent or near-silent (RMS {rms:.6f} < "
            f"threshold {config.silence_rms_threshold:.6f}). Sensor or connection may be disconnected."
        )

    # 4. Clipping check
    clipping_ratio = compute_clipping_ratio(signal)
    is_clipped = False
    if clipping_ratio >= config.clipping_ratio_threshold:
        is_clipped = True
        warnings.append(
            f"Severe digital clipping detected ({clipping_ratio:.2%} of samples saturated). "
            f"Recording quality is poor; result may be unreliable."
        )

    # 5. DC offset check
    dc_offset = compute_dc_offset(signal)
    if abs(dc_offset) >= config.dc_offset_threshold:
        warnings.append(
            f"Significant DC bias detected ({dc_offset:+.3f}). Sensor electronic drift suspected."
        )

    # Acceptance criteria: must have valid duration, cannot be silent, cannot have errors,
    # and if severely clipped (> threshold), passed is False to prevent false normal results.
    passed = bool(
        is_duration_valid
        and not is_silent
        and not is_clipped
        and len(errors) == 0
    )

    return QualityReport(
        passed=passed,
        rms_level=rms,
        is_silent=is_silent,
        clipping_ratio=clipping_ratio,
        is_clipped=is_clipped,
        duration_seconds=duration,
        is_duration_valid=is_duration_valid,
        dc_offset=dc_offset,
        warnings=warnings,
        errors=errors,
    )
