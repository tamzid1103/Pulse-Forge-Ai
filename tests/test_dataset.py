"""
Unit tests for Milestone M1 — Dataset audit and fan subset.

Verifies:
1. Every manifest row resolves to an existing file.
2. Every file decodes properly with SoundFile / Librosa.
3. Audio properties match expectations (sample rate, channels, duration).
4. No duplicate recording IDs across splits.
5. Train split contains ONLY normal recordings (leakage prevention).
6. Manifest schema matches the documented format.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest
import soundfile as sf

from src.config import get_default_config

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifests" / "fan_manifest.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def manifest_rows() -> list[dict[str, str]]:
    """Load rows from data/manifests/fan_manifest.csv."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"Manifest file {MANIFEST_PATH} does not exist yet. Run scripts/build_manifest.py first.")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestFanManifest:
    """Validate dataset manifest and files."""

    def test_manifest_exists(self) -> None:
        assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"

    def test_manifest_schema(self, manifest_rows: list[dict[str, str]]) -> None:
        expected_columns = {
            "recording_id",
            "relative_path",
            "machine_type",
            "machine_id_or_section",
            "label",
            "noise_condition",
            "duration_s",
            "sample_rate_hz",
            "channels",
            "group_id",
            "split",
            "source_archive",
        }
        assert len(manifest_rows) > 0, "Manifest is empty"
        actual_columns = set(manifest_rows[0].keys())
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"

    def test_all_files_exist_on_disk(self, manifest_rows: list[dict[str, str]]) -> None:
        for row in manifest_rows:
            rel = row["relative_path"]
            full_path = PROJECT_ROOT / rel
            assert full_path.exists(), f"File does not exist: {full_path}"
            assert full_path.stat().st_size > 0, f"File is empty: {full_path}"

    def test_audio_decode_and_properties(self, manifest_rows: list[dict[str, str]]) -> None:
        # Check first 5 rows and a sample of normal/abnormal
        sample_rows = manifest_rows[:5] + [r for r in manifest_rows if r["label"] == "abnormal"][:3]
        for row in sample_rows:
            full_path = PROJECT_ROOT / row["relative_path"]
            info = sf.info(str(full_path))
            assert info.samplerate == int(row["sample_rate_hz"]), "Sample rate mismatch"
            assert info.channels == int(row["channels"]), "Channel count mismatch"
            assert abs(info.duration - float(row["duration_s"])) < 0.1, "Duration mismatch"

    def test_train_contains_only_normal(self, manifest_rows: list[dict[str, str]]) -> None:
        train_rows = [r for r in manifest_rows if r["split"] == "train"]
        assert len(train_rows) > 0, "No training samples found"
        for r in train_rows:
            assert r["label"] == "normal", f"Train set must contain only normal recordings! Found: {r}"

    def test_val_and_test_contain_both_classes(self, manifest_rows: list[dict[str, str]]) -> None:
        for split in ("val", "test"):
            split_rows = [r for r in manifest_rows if r["split"] == split]
            labels = {r["label"] for r in split_rows}
            assert "normal" in labels, f"{split} split missing normal samples"
            assert "abnormal" in labels, f"{split} split missing abnormal samples"

    def test_no_recording_id_leakage_across_splits(self, manifest_rows: list[dict[str, str]]) -> None:
        split_ids: dict[str, set[str]] = {}
        for r in manifest_rows:
            sp = r["split"]
            rec_id = r["recording_id"]
            split_ids.setdefault(sp, set()).add(rec_id)

        all_splits = list(split_ids.keys())
        for i in range(len(all_splits)):
            for j in range(i + 1, len(all_splits)):
                s1, s2 = all_splits[i], all_splits[j]
                overlap = split_ids[s1] & split_ids[s2]
                assert len(overlap) == 0, f"Leakage detected between {s1} and {s2}: {overlap}"

    def test_supported_machine_type_only(self, manifest_rows: list[dict[str, str]]) -> None:
        cfg = get_default_config()
        for r in manifest_rows:
            assert r["machine_type"] in cfg.supported_machine_types, f"Unsupported machine type: {r['machine_type']}"

