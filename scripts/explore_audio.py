"""
Audio Exploration & Baseline Feature Fitting Script for Milestone M2.

1. Loads authentic normal and abnormal fan audio clips.
2. Generates and saves waveform & log-Mel diagnostic plots.
3. Fits FeatureScaler strictly on the normal training split.
4. Serializes scaler parameters to model_artifacts/scaler_params.json.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.audio_io import load_audio
from src.config import AudioConfig, FeatureConfig, get_default_config
from src.features import (
    FeatureScaler,
    extract_features,
    save_diagnostic_plot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.explore")


def run_exploration() -> None:
    cfg = get_default_config()
    project_root = cfg.paths.project_root
    manifest_path = cfg.paths.manifests_dir / "fan_manifest.csv"
    plots_dir = cfg.paths.results_dir / "plots"
    artifacts_dir = cfg.paths.model_artifacts_dir

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    logger.info(f"Loaded {len(records)} records from manifest.")

    # 1. Select sample normal and abnormal recordings for visualization
    normal_samples = [r for r in records if r["label"] == "normal"]
    abnormal_samples = [r for r in records if r["label"] == "abnormal"]

    if not normal_samples or not abnormal_samples:
        raise ValueError("Manifest must contain both normal and abnormal recordings.")

    normal_rec = normal_samples[0]
    abnormal_rec = abnormal_samples[0]

    logger.info(f"Visualizing Normal sample: {normal_rec['recording_id']} ({normal_rec['relative_path']})")
    audio_norm = load_audio(project_root / normal_rec["relative_path"], config=cfg.audio)
    plot_norm = save_diagnostic_plot(
        audio=audio_norm,
        output_path=plots_dir / "exploration_normal.png",
        title="PulseForge AI — Normal Fan Acoustic Signature",
        config=cfg.features,
    )

    logger.info(f"Visualizing Abnormal sample: {abnormal_rec['recording_id']} ({abnormal_rec['relative_path']})")
    audio_abn = load_audio(project_root / abnormal_rec["relative_path"], config=cfg.audio)
    plot_abn = save_diagnostic_plot(
        audio=audio_abn,
        output_path=plots_dir / "exploration_abnormal.png",
        title="PulseForge AI — Anomalous Fan Acoustic Signature",
        config=cfg.features,
    )

    # 2. Extract features from training samples and fit FeatureScaler
    train_records = [r for r in records if r["split"] == "train"]
    logger.info(f"Fitting FeatureScaler on {len(train_records)} normal training recordings...")

    train_feature_list: list[np.ndarray] = []
    for r in train_records:
        sig = load_audio(project_root / r["relative_path"], config=cfg.audio)
        tensor = extract_features(sig, config=cfg.features)
        train_feature_list.append(tensor.features)

    # Concatenate all training segments along axis 0
    all_train_features = np.concatenate(train_feature_list, axis=0)
    logger.info(f"Total training segments: {all_train_features.shape[0]}, feature shape per segment: {all_train_features.shape[1:]}")

    scaler = FeatureScaler()
    scaler.fit(all_train_features)

    # 3. Save scaler parameters
    scaler_out = artifacts_dir / "scaler_params.json"
    scaler.save(scaler_out)

    # 4. Verify normalized features
    normed_features = scaler.transform(all_train_features)
    mean_val = float(np.mean(normed_features))
    std_val = float(np.std(normed_features))
    logger.info(f"FeatureScaler verified: normalized train mean={mean_val:.4f}, std={std_val:.4f}")
    logger.info("Exploration and preprocessing setup complete.")


if __name__ == "__main__":
    run_exploration()
