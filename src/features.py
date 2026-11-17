"""
Feature Extraction and Normalization Module for PulseForge AI.

Extracts fixed-shape log-Mel spectrogram representations from audio segments,
fits normalization statistics strictly on normal training data, and provides
diagnostic plotting functions.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

# Ensure writable matplotlib config dir
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_pulseforge")

import matplotlib
matplotlib.use("Agg")  # Headless rendering safe for servers/sandboxes
import matplotlib.pyplot as plt

from src.audio_io import AudioSignal
from src.config import FeatureConfig
from src.exceptions import FeatureExtractionError
from src.preprocessing import AudioSegments, preprocess_audio

logger = logging.getLogger("pulseforge.features")


@dataclass(frozen=True)
class FeatureTensor:
    """Extracted feature representations for an audio recording."""

    # Shape: (num_segments, n_mels, n_frames), e.g. (19, 128, 32)
    features: np.ndarray
    sample_rate: int
    n_mels: int
    n_frames: int

    @property
    def num_segments(self) -> int:
        return self.features.shape[0]

    @property
    def flattened_dim(self) -> int:
        """Flattened dimension per segment (n_mels * n_frames)."""
        return self.n_mels * self.n_frames

    def to_flattened(self) -> np.ndarray:
        """Reshape to (num_segments, n_mels * n_frames) for dense models/baselines."""
        return self.features.reshape(self.num_segments, -1)


class FeatureScaler:
    """
    Per-frequency-bin standardizer (mean / std scaler).

    Strictly fitted on normal training data only. Preserves and serializes
    statistics to ensure identical normalization during inference.
    """

    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.mean = mean
        self.std = std
        self.eps = eps

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, features: np.ndarray) -> FeatureScaler:
        """
        Compute per-Mel-bin mean and std from training feature array.

        Parameters:
            features: 3D array of shape (N, n_mels, n_frames).
        """
        if features.ndim != 3:
            raise ValueError(f"Expected 3D feature array (N, n_mels, n_frames), got shape {features.shape}")

        # Compute statistics over segments (axis 0) and time frames (axis 2), per Mel bin (axis 1)
        # Resulting mean and std have shape (1, n_mels, 1) for broadcasable operations
        self.mean = np.mean(features, axis=(0, 2), keepdims=True).astype(np.float32)
        raw_std = np.std(features, axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = np.maximum(raw_std, self.eps)
        logger.info(f"Fitted FeatureScaler across {len(features)} segments ({self.mean.shape[1]} Mel bins).")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply standardization: (X - mean) / std."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transforming data.")
        normed = (features - self.mean) / self.std
        if not np.all(np.isfinite(normed)):
            raise FeatureExtractionError("Non-finite values encountered after feature scaling.")
        return normed.astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, filepath: Path | str) -> None:
        """Save scaler parameters and metadata to JSON."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureScaler.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.flatten().tolist(),
            "std": self.std.flatten().tolist(),
            "n_mels": self.mean.shape[1],
            "eps": self.eps,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved FeatureScaler parameters to {path}")

    @classmethod
    def load(cls, filepath: Path | str) -> FeatureScaler:
        """Load scaler parameters from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Scaler parameters file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        n_mels = payload["n_mels"]
        mean = np.array(payload["mean"], dtype=np.float32).reshape(1, n_mels, 1)
        std = np.array(payload["std"], dtype=np.float32).reshape(1, n_mels, 1)
        return cls(mean=mean, std=std, eps=payload.get("eps", 1e-6))


def extract_log_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16_000,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Compute log-Mel power spectrogram for a single 1D audio segment.

    Returns:
        2D numpy array of shape (n_mels, n_frames), dtype float32 (values in dB).
    """
    if config is None:
        config = FeatureConfig()

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=config.power,
        center=True,
    )

    # Use constant reference 1.0 to preserve absolute acoustic amplitude differences
    log_mel = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

    if not np.all(np.isfinite(log_mel)):
        raise FeatureExtractionError("Non-finite values encountered in log-Mel spectrogram.")

    return log_mel.astype(np.float32)


def extract_features(
    audio: AudioSignal,
    config: FeatureConfig | None = None,
    scaler: FeatureScaler | None = None,
) -> FeatureTensor:
    """
    Full feature pipeline:
    1. Frames audio into 1-second segments.
    2. Computes log-Mel spectrogram for each segment.
    3. Optionally applies fitted FeatureScaler.
    """
    if config is None:
        config = FeatureConfig()

    audio_segments = preprocess_audio(audio, feature_config=config)
    num_segs = audio_segments.num_segments

    # Extract first segment to establish shape
    first_spec = extract_log_mel_spectrogram(audio_segments.segments[0], sr=audio.sample_rate, config=config)
    n_mels, n_frames = first_spec.shape

    feature_array = np.empty((num_segs, n_mels, n_frames), dtype=np.float32)
    feature_array[0] = first_spec

    for i in range(1, num_segs):
        feature_array[i] = extract_log_mel_spectrogram(
            audio_segments.segments[i],
            sr=audio.sample_rate,
            config=config,
        )

    if scaler is not None:
        feature_array = scaler.transform(feature_array)

    return FeatureTensor(
        features=feature_array,
        sample_rate=audio.sample_rate,
        n_mels=n_mels,
        n_frames=n_frames,
    )


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def compute_waveform_plot_data(signal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_seconds, amplitude) arrays for waveform plotting."""
    time_axis = np.linspace(0.0, len(signal) / sr, num=len(signal), endpoint=False, dtype=np.float32)
    return time_axis, signal


def compute_spectrogram_plot_data(
    signal: np.ndarray,
    sr: int = 16_000,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (log_mel_matrix, time_axis_seconds, mel_frequencies_hz) for full audio display.
    """
    if config is None:
        config = FeatureConfig()

    mel_spec = extract_log_mel_spectrogram(signal, sr=sr, config=config)
    time_axis = librosa.frames_to_time(
        np.arange(mel_spec.shape[1]),
        sr=sr,
        hop_length=config.hop_length,
    )
    freq_axis = librosa.mel_frequencies(
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax or (sr / 2.0),
    )
    return mel_spec, time_axis, freq_axis


def save_diagnostic_plot(
    audio: AudioSignal,
    output_path: Path | str,
    title: str = "PulseForge AI — Acoustic Signal Diagnostics",
    config: FeatureConfig | None = None,
) -> Path:
    """
    Generate and save a dual-panel figure (waveform + log-Mel spectrogram) to disk.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    time_wave, amp = compute_waveform_plot_data(audio.data, audio.sample_rate)
    mel_db, time_spec, freqs = compute_spectrogram_plot_data(audio.data, audio.sample_rate, config=config)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # 1. Waveform
    ax1.plot(time_wave, amp, color="#2563EB", linewidth=0.6, alpha=0.85)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{title} | {audio.metadata.filename}")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Log-Mel Spectrogram
    img = ax2.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        extent=[time_spec[0], time_spec[-1], freqs[0], freqs[-1]],
        cmap="magma",
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(img, ax=ax2, orientation="horizontal", pad=0.15, shrink=0.6)
    cbar.set_label("Log-Mel Energy (dB)")

    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Diagnostic plot saved to {out_file}")
    return out_file
