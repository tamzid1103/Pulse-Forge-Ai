"""
Unit tests for src/audio_io.py.

Verifies:
1. Valid WAV loading from bytes and disk.
2. Content format and magic bytes validation.
3. Upload size limit enforcement.
4. Duration constraints (min / max duration).
5. Multichannel reduction (8-channel MIMII audio to 1D mono).
6. Resampling to target sample rate (16,000 Hz).
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import (
    AudioMetadata,
    AudioSignal,
    get_audio_metadata,
    load_audio,
    validate_wav_header,
)
from src.config import AudioConfig
from src.exceptions import (
    AudioTooLongError,
    AudioTooShortError,
    CorruptAudioError,
    FileTooLargeError,
    InvalidFormatError,
)

SAMPLE_WAV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "id_00" / "normal" / "00000000.wav"


class TestAudioValidation:
    """Test format and safety constraints."""

    def test_validate_wav_header_valid(self, valid_wav_bytes: bytes) -> None:
        # Should not raise
        validate_wav_header(valid_wav_bytes, max_bytes=10 * 1024 * 1024)

    def test_validate_wav_header_too_large(self, valid_wav_bytes: bytes) -> None:
        with pytest.raises(FileTooLargeError) as exc_info:
            validate_wav_header(valid_wav_bytes, max_bytes=100)
        assert exc_info.value.file_size > 100
        assert "exceeds maximum allowed limit" in str(exc_info.value)

    def test_validate_wav_header_not_riff(self) -> None:
        bad_data = b"NOT_A_RIFF_HEADER_123456789"
        with pytest.raises(InvalidFormatError):
            validate_wav_header(bad_data, max_bytes=1024 * 1024)

    def test_validate_wav_header_mp3_detected(self) -> None:
        mp3_magic = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
        with pytest.raises(InvalidFormatError) as exc_info:
            validate_wav_header(mp3_magic, max_bytes=1024 * 1024)
        assert exc_info.value.detected_format == "mp3"

    def test_validate_wav_header_truncated(self) -> None:
        with pytest.raises(InvalidFormatError):
            validate_wav_header(b"RIFF", max_bytes=1024 * 1024)


class TestAudioLoading:
    """Test audio decoding and channel handling."""

    def test_load_audio_from_bytes(self, valid_wav_bytes: bytes) -> None:
        cfg = AudioConfig(sample_rate=16_000)
        signal = load_audio(valid_wav_bytes, config=cfg, filename_hint="test.wav")
        assert isinstance(signal, AudioSignal)
        assert signal.data.ndim == 1
        assert signal.data.dtype == np.float32
        assert signal.sample_rate == 16_000
        assert signal.duration_seconds == pytest.approx(1.0, abs=0.05)
        assert signal.metadata.filename == "test.wav"

    def test_load_audio_too_short(self, short_wav_bytes: bytes) -> None:
        cfg = AudioConfig(min_duration_seconds=0.5)
        with pytest.raises(AudioTooShortError) as exc_info:
            load_audio(short_wav_bytes, config=cfg)
        assert exc_info.value.duration < 0.5

    def test_load_audio_too_long(self, valid_wav_bytes: bytes) -> None:
        cfg = AudioConfig(max_duration_seconds=0.5)
        with pytest.raises(AudioTooLongError):
            load_audio(valid_wav_bytes, config=cfg)

    def test_load_audio_resampling(self) -> None:
        # Generate 22050 Hz 1-second audio in memory
        sr_orig = 22050
        num_samples = sr_orig
        t = np.linspace(0, 1.0, num_samples, endpoint=False)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, waveform, sr_orig, format="WAV")
        wav_bytes = buf.getvalue()

        cfg = AudioConfig(sample_rate=16_000)
        signal = load_audio(wav_bytes, config=cfg)
        assert signal.sample_rate == 16_000
        assert signal.duration_seconds == pytest.approx(1.0, abs=0.05)
        assert len(signal.data) == pytest.approx(16_000, abs=50)

    @pytest.mark.skipif(not SAMPLE_WAV_PATH.exists(), reason="MIMII audio sample not found on disk")
    def test_load_real_mimii_multichannel(self) -> None:
        cfg = AudioConfig(sample_rate=16_000, channel_strategy="first")
        signal = load_audio(SAMPLE_WAV_PATH, config=cfg)

        assert signal.data.ndim == 1
        assert signal.metadata.channels == 8
        assert signal.duration_seconds == pytest.approx(10.0, abs=0.05)
        assert signal.sample_rate == 16_000
        assert np.all(np.isfinite(signal.data))
        # Ensure amplitude is within standard audio range
        assert np.max(np.abs(signal.data)) <= 1.0

    @pytest.mark.skipif(not SAMPLE_WAV_PATH.exists(), reason="MIMII audio sample not found on disk")
    def test_load_real_mimii_mean_channel_strategy(self) -> None:
        cfg = AudioConfig(sample_rate=16_000, channel_strategy="mean")
        signal = load_audio(SAMPLE_WAV_PATH, config=cfg)
        assert signal.data.ndim == 1
        assert signal.metadata.channels == 8
        assert np.all(np.isfinite(signal.data))

    def test_get_audio_metadata(self, valid_wav_bytes: bytes) -> None:
        meta = get_audio_metadata(valid_wav_bytes)
        assert isinstance(meta, AudioMetadata)
        assert meta.sample_rate == 16_000
        assert meta.channels == 1
        assert meta.duration_seconds == pytest.approx(1.0, abs=0.05)
