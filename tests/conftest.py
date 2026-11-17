"""Shared test fixtures for PulseForge AI."""

from __future__ import annotations

import io
import struct
import pytest

from src.config import AppConfig, get_default_config


@pytest.fixture
def app_config() -> AppConfig:
    """Provide a default AppConfig for tests."""
    return get_default_config()


def _make_wav_bytes(
    num_samples: int = 16000,
    sample_rate: int = 16000,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Generate a minimal valid WAV file in memory.

    Produces a sine-like pattern (not silent, not clipped) for testing.
    """
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    buf = io.BytesIO()

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")

    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))   # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    buf.write(struct.pack("<I", byte_rate))
    block_align = num_channels * (bits_per_sample // 8)
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))

    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    import math
    max_val = 2 ** (bits_per_sample - 1) - 1
    for i in range(num_samples):
        # Simple sine wave at 440 Hz, amplitude ~50%
        val = int(0.5 * max_val * math.sin(2 * math.pi * 440 * i / sample_rate))
        for _ in range(num_channels):
            buf.write(struct.pack("<h", val))

    return buf.getvalue()


@pytest.fixture
def valid_wav_bytes() -> bytes:
    """Return bytes of a valid 1-second 16 kHz mono WAV."""
    return _make_wav_bytes(num_samples=16000, sample_rate=16000)


@pytest.fixture
def short_wav_bytes() -> bytes:
    """Return bytes of a too-short WAV (0.01 seconds)."""
    return _make_wav_bytes(num_samples=160, sample_rate=16000)


@pytest.fixture
def silent_wav_bytes() -> bytes:
    """Return bytes of a silent WAV (all zeros)."""
    data = _make_wav_bytes(num_samples=16000, sample_rate=16000)
    # Replace data section with zeros
    buf = bytearray(data)
    data_offset = data.index(b"data") + 8  # after "data" + size field
    for i in range(data_offset, len(buf)):
        buf[i] = 0
    return bytes(buf)
