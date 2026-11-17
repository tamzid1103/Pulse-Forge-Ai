"""
Unit tests for src/features.py and src/preprocessing.py.

Verifies:
1. Signal framing into fixed-duration overlapping segments.
2. Short signal padding.
3. Log-Mel spectrogram dimension and finite values.
4. Determinism: identical input yields bitwise identical features.
5. FeatureScaler fit, transform, serialization, and round-trip consistency.
6. Diagnostic plotting helpers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.audio_io import AudioMetadata, AudioSignal
from src.config import FeatureConfig
from src.features import (
    FeatureScaler,
    FeatureTensor,
    compute_spectrogram_plot_data,
    compute_waveform_plot_data,
    extract_features,
    extract_log_mel_spectrogram,
    save_diagnostic_plot,
)
from src.preprocessing import frame_signal

SAMPLE_WAV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "id_00" / "normal" / "00000000.wav"


def _make_mock_signal(duration_s: float = 1.0, sr: int = 16_000) -> AudioSignal:
    num_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Sine wave + small noise
    data = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.cos(2 * np.pi * 1200 * t)).astype(np.float32)
    meta = AudioMetadata(
        filename="mock.wav",
        size_bytes=len(data) * 2,
        sample_rate=sr,
        channels=1,
        duration_seconds=duration_s,
        format="WAV",
        subtype="PCM_16",
    )
    return AudioSignal(data=data, sample_rate=sr, duration_seconds=duration_s, metadata=meta)


class TestPreprocessing:
    """Test window framing and signal preparation."""

    def test_frame_signal_standard(self) -> None:
        sr = 16_000
        signal = np.zeros(sr * 10, dtype=np.float32)  # 10.0 seconds
        segments = frame_signal(signal, sample_rate=sr, segment_length_s=1.0, segment_hop_s=0.5)

        # Total samples = 160,000, window = 16,000, hop = 8,000
        # Number of segments = 1 + (160,000 - 16,000) / 8,000 = 1 + 18 = 19
        assert segments.shape == (19, 16_000)
        assert segments.dtype == np.float32

    def test_frame_signal_short_padded(self) -> None:
        sr = 16_000
        # 0.6 seconds (less than 1.0s window)
        signal = np.ones(int(0.6 * sr), dtype=np.float32)
        segments = frame_signal(signal, sample_rate=sr, segment_length_s=1.0, segment_hop_s=0.5)

        assert segments.shape == (1, 16_000)
        assert np.all(np.isfinite(segments))


class TestFeatureExtraction:
    """Test log-Mel extraction and shape contracts."""

    def test_extract_log_mel_spectrogram_shape(self) -> None:
        sr = 16_000
        segment = np.zeros(sr, dtype=np.float32)  # 1.0 second
        cfg = FeatureConfig(n_mels=128, n_fft=1024, hop_length=512)
        mel = extract_log_mel_spectrogram(segment, sr=sr, config=cfg)

        assert mel.ndim == 2
        assert mel.shape[0] == 128  # Mel bins
        assert mel.shape[1] == 32   # Time frames (1 + 16000 // 512)
        assert np.all(np.isfinite(mel))

    def test_extract_features_full_pipeline(self) -> None:
        audio = _make_mock_signal(duration_s=2.0, sr=16_000)
        cfg = FeatureConfig()
        tensor = extract_features(audio, config=cfg)

        assert isinstance(tensor, FeatureTensor)
        # 2.0s audio with 1.0s window and 0.5s hop -> (2.0 - 1.0)/0.5 + 1 = 3 segments
        assert tensor.num_segments == 3
        assert tensor.features.shape == (3, 128, 32)
        assert tensor.flattened_dim == 128 * 32
        assert tensor.to_flattened().shape == (3, 4096)
        assert np.all(np.isfinite(tensor.features))

    def test_feature_extraction_determinism(self) -> None:
        audio = _make_mock_signal(duration_s=3.0, sr=16_000)
        cfg = FeatureConfig()

        t1 = extract_features(audio, config=cfg)
        t2 = extract_features(audio, config=cfg)

        np.testing.assert_array_equal(t1.features, t2.features)


class TestFeatureScaler:
    """Test train-only normalization and serialization."""

    def test_scaler_fit_transform(self) -> None:
        np.random.seed(42)
        # 20 segments, 128 Mel bins, 32 time frames
        mock_data = np.random.randn(20, 128, 32).astype(np.float32) * 5.0 + 10.0

        scaler = FeatureScaler()
        assert not scaler.is_fitted

        normed = scaler.fit_transform(mock_data)
        assert scaler.is_fitted
        assert normed.shape == mock_data.shape

        # Verify mean is close to 0 and std is close to 1 across the dataset
        per_bin_mean = np.mean(normed, axis=(0, 2))
        per_bin_std = np.std(normed, axis=(0, 2))

        np.testing.assert_allclose(per_bin_mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(per_bin_std, 1.0, atol=1e-4)

    def test_scaler_save_load_roundtrip(self) -> None:
        np.random.seed(42)
        mock_data = np.random.randn(10, 128, 32).astype(np.float32)

        scaler = FeatureScaler().fit(mock_data)
        original_normed = scaler.transform(mock_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "scaler_test.json"
            scaler.save(save_path)
            assert save_path.exists()

            loaded_scaler = FeatureScaler.load(save_path)
            assert loaded_scaler.is_fitted

            reloaded_normed = loaded_scaler.transform(mock_data)
            np.testing.assert_allclose(original_normed, reloaded_normed, atol=1e-6)


class TestVisualizationHelpers:
    """Test plotting and visual array generation."""

    def test_compute_waveform_plot_data(self) -> None:
        audio = _make_mock_signal(duration_s=1.0, sr=16_000)
        t, y = compute_waveform_plot_data(audio.data, audio.sample_rate)
        assert len(t) == len(y) == 16_000
        assert t[0] == 0.0
        assert t[-1] < 1.0

    def test_compute_spectrogram_plot_data(self) -> None:
        audio = _make_mock_signal(duration_s=1.0, sr=16_000)
        mel_db, t, freqs = compute_spectrogram_plot_data(audio.data, audio.sample_rate)
        assert mel_db.shape == (128, 32)
        assert len(t) == 32
        assert len(freqs) == 128
        assert np.all(np.isfinite(mel_db))

    def test_save_diagnostic_plot(self) -> None:
        audio = _make_mock_signal(duration_s=1.0, sr=16_000)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "diagnostic.png"
            res = save_diagnostic_plot(audio, output_path=out_img)
            assert res.exists()
            assert res.stat().st_size > 1000  # Valid non-empty PNG
