"""
Build reproducible dataset manifest for PulseForge AI.

Scans the local fan audio directory, decodes audio headers to verify
sample rate, channels, and duration, assigns group-aware train/val/test splits,
and writes data/manifests/fan_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.manifest")

MANIFEST_COLUMNS = [
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
]


def build_manifest(
    data_dir: Path,
    output_manifest: Path,
    machine_type: str = "fan",
    noise_condition: str = "6_dB",
    source_archive: str = "6_dB_fan.zip",
    train_normal_ratio: float = 0.70,
    val_normal_ratio: float = 0.15,
    val_abnormal_ratio: float = 0.50,
) -> list[dict[str, Any]]:
    """Scan data_dir, extract metadata, assign splits, and save CSV manifest."""
    project_root = output_manifest.resolve().parent.parent.parent

    wav_files = sorted(data_dir.glob("**/*.wav"))
    logger.info(f"Scanning {len(wav_files)} WAV files in {data_dir}...")

    normal_records: list[dict[str, Any]] = []
    abnormal_records: list[dict[str, Any]] = []

    for path in wav_files:
        rel_to_root = path.resolve().relative_to(project_root).as_posix()
        parts = path.relative_to(data_dir).parts
        # Expected layout: [id_00, normal, 00000000.wav] or similar
        machine_id = "id_00"
        label = "normal"
        for p in parts:
            if p.startswith("id_"):
                machine_id = p
            if p in ("normal", "abnormal"):
                label = p

        info = sf.info(str(path))
        rec_id = f"{machine_type}_{machine_id}_{label}_{path.stem}"

        record = {
            "recording_id": rec_id,
            "relative_path": rel_to_root,
            "machine_type": machine_type,
            "machine_id_or_section": machine_id,
            "label": label,
            "noise_condition": noise_condition,
            "duration_s": round(info.duration, 2),
            "sample_rate_hz": info.samplerate,
            "channels": info.channels,
            "group_id": "",
            "split": "",
            "source_archive": source_archive,
        }

        if label == "normal":
            normal_records.append(record)
        else:
            abnormal_records.append(record)

    logger.info(f"Discovered: {len(normal_records)} normal, {len(abnormal_records)} abnormal.")

    # Split normal records: train / val / test
    n_norm = len(normal_records)
    n_train = int(n_norm * train_normal_ratio)
    n_val_norm = int(n_norm * val_normal_ratio)

    for i, r in enumerate(normal_records):
        if i < n_train:
            r["split"] = "train"
            r["group_id"] = f"{r['machine_id_or_section']}_train_grp_{i // 10}"
        elif i < n_train + n_val_norm:
            r["split"] = "val"
            r["group_id"] = f"{r['machine_id_or_section']}_val_grp_{i // 10}"
        else:
            r["split"] = "test"
            r["group_id"] = f"{r['machine_id_or_section']}_test_grp_{i // 10}"

    # Split abnormal records: val / test (NEVER train on abnormal)
    n_abnorm = len(abnormal_records)
    n_val_abnorm = int(n_abnorm * val_abnormal_ratio)

    for i, r in enumerate(abnormal_records):
        if i < n_val_abnorm:
            r["split"] = "val"
            r["group_id"] = f"{r['machine_id_or_section']}_val_abn_grp_{i // 5}"
        else:
            r["split"] = "test"
            r["group_id"] = f"{r['machine_id_or_section']}_test_abn_grp_{i // 5}"

    all_records = normal_records + abnormal_records
    # Sort deterministically
    all_records.sort(key=lambda x: (x["split"], x["label"], x["recording_id"]))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_records)

    logger.info(f"Wrote manifest with {len(all_records)} entries to {output_manifest}.")
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fan dataset manifest")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to audio directory")
    parser.add_argument("--output", type=str, default="data/manifests/fan_manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / args.data_dir
    output_path = project_root / args.output

    records = build_manifest(data_dir=data_path, output_manifest=output_path)
    # Print split counts
    splits: dict[str, dict[str, int]] = {}
    for r in records:
        sp = r["split"]
        lb = r["label"]
        splits.setdefault(sp, {})[lb] = splits.setdefault(sp, {}).get(lb, 0) + 1

    print("\nManifest summary by split and label:")
    for sp, counts in sorted(splits.items()):
        total = sum(counts.values())
        print(f"  {sp.upper()} ({total} files): {counts}")


if __name__ == "__main__":
    main()

