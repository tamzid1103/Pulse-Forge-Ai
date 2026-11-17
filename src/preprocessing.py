"""
Audio Preprocessing and Window Framing Module for PulseForge AI.

Standardizes sample rates, applies windowed segmentation, and formats
audio signals into deterministic segment arrays for feature extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.audio_io import AudioSignal
from src.config import AudioConfig, FeatureConfig

logger = logging.getLogger("pulseforge.preprocessing")


@dataclass(frozen=True)
class AudioSegments:
    """Framed audio segments ready for feature extraction."""

    segments: np.ndarray  # Shape: (N, W), dtype: float32
    sample_rate: int
    segment_length_s: float
    segment_hop_s: float

    @property
    def num_segments(self) -> int:
        return self.segments.shape[0]

    @property
    def window_samples(self) -> int:
        return self.segments.shape[1]


def frame_signal(
    signal: np.ndarray,
    sample_rate: int = 16_000,
    segment_length_s: float = 1.0,
    segment_hop_s: float = 0.5,
) -> np.ndarray:
    """
    Split a 1D audio signal into overlapping fixed-length windows.

    Parameters:
        signal: 1D float numpy array.
        sample_rate: Audio sampling frequency in Hz.
        segment_length_s: Duration of each analysis segment in seconds.
        segment_hop_s: Hop length between consecutive segments in seconds.

    Returns:
        2D numpy array of shape (num_segments, window_samples), dtype float32.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D audio signal, got shape {signal.shape}")

    window_samples = int(round(segment_length_s * sample_rate))
    hop_samples = int(round(segment_hop_s * sample_rate))

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop sample counts must be positive integers.")

    total_samples = len(signal)

    # Edge case: signal is shorter than one full window (e.g. 0.5s - 0.99s)
    if total_samples < window_samples:
        pad_needed = window_samples - total_samples
        # Symmetric reflection padding preserves spectral continuity
        padded = np.pad(signal, (0, pad_needed), mode="reflect")
        return padded[np.newaxis, :].astype(np.float32)

    # Standard sliding window framing
    num_segments = 1 + (total_samples - window_samples) // hop_samples
    segments = np.empty((num_segments, window_samples), dtype=np.float32)

    for i in range(num_segments):
        start = i * hop_samples
        end = start + window_samples
        segments[i] = signal[start:end]

    return segments


def preprocess_audio(
    audio: AudioSignal,
    feature_config: FeatureConfig | None = None,
) -> AudioSegments:
    """
    Preprocess an AudioSignal into structured, framed AudioSegments.
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    segments = frame_signal(
        signal=audio.data,
        sample_rate=audio.sample_rate,
        segment_length_s=feature_config.segment_length_seconds,
        segment_hop_s=feature_config.segment_hop_seconds,
    )

    return AudioSegments(
        segments=segments,
        sample_rate=audio.sample_rate,
        segment_length_s=feature_config.segment_length_seconds,
        segment_hop_s=feature_config.segment_hop_seconds,
    )
