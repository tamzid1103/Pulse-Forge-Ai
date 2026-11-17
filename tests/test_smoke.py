"""
M0 smoke tests — verify that all core dependencies import
and that the basic project structure is functional.
"""

from __future__ import annotations

import importlib
import sys


class TestImports:
    """Verify all required libraries are importable."""

    def test_numpy(self) -> None:
        import numpy
        assert hasattr(numpy, "__version__")

    def test_pandas(self) -> None:
        import pandas
        assert hasattr(pandas, "__version__")

    def test_scipy(self) -> None:
        import scipy
        assert hasattr(scipy, "__version__")

    def test_matplotlib(self) -> None:
        import matplotlib
        assert hasattr(matplotlib, "__version__")

    def test_librosa(self) -> None:
        import librosa
        assert hasattr(librosa, "__version__")

    def test_soundfile(self) -> None:
        import soundfile
        assert hasattr(soundfile, "__version__")

    def test_sklearn(self) -> None:
        import sklearn
        assert hasattr(sklearn, "__version__")

    def test_torch(self) -> None:
        import torch
        assert hasattr(torch, "__version__")

    def test_streamlit(self) -> None:
        import streamlit
        assert hasattr(streamlit, "__version__")

    def test_plotly(self) -> None:
        import plotly
        assert hasattr(plotly, "__version__")

    def test_pytest_available(self) -> None:
        import pytest
        assert hasattr(pytest, "__version__")


class TestProjectStructure:
    """Verify the project package structure."""

    def test_src_package_importable(self) -> None:
        import src
        assert src is not None

    def test_config_importable(self) -> None:
        from src.config import AppConfig, get_default_config
        cfg = get_default_config()
        assert isinstance(cfg, AppConfig)

    def test_config_defaults(self) -> None:
        from src.config import get_default_config
        cfg = get_default_config()
        assert cfg.default_machine_type == "fan"
        assert cfg.audio.sample_rate == 16_000
        assert cfg.audio.max_upload_bytes == 25 * 1024 * 1024
        assert cfg.features.n_mels == 128
        assert cfg.supported_machine_types == ("fan",)

    def test_models_package_importable(self) -> None:
        import src.models
        assert src.models is not None

    def test_paths_config(self) -> None:
        from src.config import get_default_config
        cfg = get_default_config()
        # Paths should be Path objects
        from pathlib import Path
        assert isinstance(cfg.paths.data_dir, Path)
        assert isinstance(cfg.paths.model_artifacts_dir, Path)
        assert isinstance(cfg.paths.demo_audio_dir, Path)
        assert isinstance(cfg.paths.results_dir, Path)

    def test_python_version(self) -> None:
        """Verify Python >= 3.10 (type hint syntax requirement)."""
        assert sys.version_info >= (3, 10)


class TestGitignore:
    """Verify that sensitive/large files are ignored."""

    def test_gitignore_covers_data(self) -> None:
        from pathlib import Path
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text()
        assert "/data/" in content
        assert "*.wav" in content
        assert "model_artifacts/" in content
        assert "*.pt" in content
        assert ".env" in content
        assert "*.db" in content

    def test_env_example_not_ignored(self) -> None:
        from pathlib import Path
        gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore.read_text()
        assert "!.env.example" in content
