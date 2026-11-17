"""
Audio I/O and Content Validation Module for PulseForge AI.

Handles safe, content-verified loading of WAV files from memory or disk,
enforcing upload size, duration limits, channel policies, and format integrity.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import librosa

from src.config import AudioConfig
from src.exceptions import (
    AudioTooLongError,
    AudioTooShortError,
    CorruptAudioError,
    FileTooLargeError,
    InvalidFormatError,
)

logger = logging.getLogger("pulseforge.audio_io")

AudioSource = Union[bytes, Path, str]


@dataclass(frozen=True)
class AudioMetadata:
    """Decoded audio technical metadata."""

    filename: str
    size_bytes: int
    sample_rate: int
    channels: int
    duration_seconds: float
    format: str
    subtype: str


@dataclass(frozen=True)
class AudioSignal:
    """Standardized single-channel audio signal ready for analysis."""

    data: np.ndarray  # 1D float32 array
    sample_rate: int
    duration_seconds: float
    metadata: AudioMetadata

    def __post_init__(self) -> None:
        if self.data.ndim != 1:
            raise ValueError(f"AudioSignal data must be 1D, got shape {self.data.shape}")
        if self.data.dtype != np.float32:
            object.__setattr__(self, "data", self.data.astype(np.float32))


def validate_wav_header(data: bytes, max_bytes: int) -> None:
    """Validate WAV magic bytes and size constraints before decoding."""
    size = len(data)
    if size > max_bytes:
        raise FileTooLargeError(file_size=size, max_size=max_bytes)

    if size < 12:
        raise InvalidFormatError(detected_format="empty_or_truncated", supported_formats=("wav",))

    # Check for RIFF header and WAVE form type
    if not (data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        # Check if user uploaded MP3, OGG, FLAC, or other binary
        magic = data[:4]
        if magic.startswith(b"ID3") or magic.startswith(b"\xff\xfb"):
            detected = "mp3"
        elif magic.startswith(b"OggS"):
            detected = "ogg"
        elif magic.startswith(b"fLaC"):
            detected = "flac"
        else:
            detected = "unknown_binary"
        raise InvalidFormatError(detected_format=detected, supported_formats=("wav",))


def get_audio_metadata(source: AudioSource, filename_hint: str = "audio.wav") -> AudioMetadata:
    """Extract audio metadata from bytes or path without reading full waveform into memory."""
    if isinstance(source, bytes):
        size_bytes = len(source)
        file_obj = io.BytesIO(source)
        name = filename_hint
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        size_bytes = path.stat().st_size
        file_obj = path.as_posix()
        name = path.name

    try:
        info = sf.info(file_obj)
    except Exception as e:
        raise CorruptAudioError(str(e)) from e

    return AudioMetadata(
        filename=name,
        size_bytes=size_bytes,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=float(info.duration),
        format=info.format,
        subtype=info.subtype,
    )


def load_audio(
    source: AudioSource,
    config: AudioConfig | None = None,
    filename_hint: str = "upload.wav",
) -> AudioSignal:
    """
    Safely load, validate, and standardize audio from bytes or file path.

    - Validates file size and WAV magic bytes.
    - Decodes samples into float32 [-1.0, 1.0].
    - Selects or downmixes channels to 1D mono (channel 0 by default).
    - Enforces duration bounds.
    - Resamples to target sample rate (16,000 Hz) if needed.
    """
    if config is None:
        config = AudioConfig()

    # Step 1: Read raw bytes if path, or validate existing bytes
    if isinstance(source, (Path, str)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        file_size = path.stat().st_size
        if file_size > config.max_upload_bytes:
            raise FileTooLargeError(file_size=file_size, max_size=config.max_upload_bytes)
        raw_bytes = path.read_bytes()
        filename = path.name
    else:
        raw_bytes = source
        filename = filename_hint

    # Step 2: Content format and size validation
    validate_wav_header(raw_bytes, max_bytes=config.max_upload_bytes)

    # Step 3: Decode with soundfile
    try:
        data_io = io.BytesIO(raw_bytes)
        samples, sr = sf.read(data_io, dtype="float32", always_2d=True)
        # Re-read info for clean metadata
        data_io.seek(0)
        info = sf.info(data_io)
    except Exception as e:
        logger.warning(f"SoundFile decode failed for {filename}: {e}. Attempting Librosa fallback.")
        try:
            samples_1d, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=False)
            if samples_1d.ndim == 1:
                samples = samples_1d[:, np.newaxis]
            else:
                samples = samples_1d.T
            info_duration = len(samples) / sr
            info = sf.SoundFile(io.BytesIO(raw_bytes))
        except Exception as fallback_err:
            raise CorruptAudioError(f"Cannot decode audio file: {fallback_err}") from fallback_err

    channels = samples.shape[1]
    duration_s = float(len(samples) / sr)

    # Step 4: Duration validation
    if duration_s < config.min_duration_seconds:
        raise AudioTooShortError(duration=duration_s, min_duration=config.min_duration_seconds)
    if duration_s > config.max_duration_seconds:
        raise AudioTooLongError(duration=duration_s, max_duration=config.max_duration_seconds)

    # Step 5: Channel selection / downmixing to 1D mono
    if channels == 1:
        mono_signal = samples[:, 0]
    else:
        if config.channel_strategy == "first":
            # Primary microphone channel (MIMII circular array channel 0)
            mono_signal = samples[:, 0]
            logger.debug(f"Selected channel 0 from {channels} available channels.")
        elif config.channel_strategy == "mean":
            mono_signal = np.mean(samples, axis=1)
            logger.debug(f"Downmixed {channels} channels using mean average.")
        else:
            mono_signal = samples[:, 0]

    # Step 6: Resampling if sample rate differs from target (e.g. 16,000 Hz)
    if sr != config.sample_rate:
        logger.info(f"Resampling audio from {sr} Hz to target {config.sample_rate} Hz.")
        mono_signal = librosa.resample(
            y=mono_signal,
            orig_sr=sr,
            target_sr=config.sample_rate,
            res_type="soxr_hq",
        )
        final_sr = config.sample_rate
        final_duration = len(mono_signal) / final_sr
    else:
        final_sr = sr
        final_duration = duration_s

    # Ensure float32 array
    mono_signal = mono_signal.astype(np.float32)

    metadata = AudioMetadata(
        filename=filename,
        size_bytes=len(raw_bytes),
        sample_rate=sr,
        channels=channels,
        duration_seconds=round(duration_s, 3),
        format=info.format,
        subtype=info.subtype,
    )

    return AudioSignal(
        data=mono_signal,
        sample_rate=final_sr,
        duration_seconds=round(final_duration, 3),
        metadata=metadata,
    )
