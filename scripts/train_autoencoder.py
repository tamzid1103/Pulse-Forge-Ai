"""
Training Script for Deep Autoencoder Anomaly Model (Milestone M4).

Trains AcousticAutoencoder on normal fan acoustic feature segments,
tracks train and validation loss, derives anomaly threshold on validation
normal recordings, generates loss curve diagnostics, and serializes the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_audio
from src.config import ModelConfig, get_default_config
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.train_autoencoder")


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read rows from the dataset manifest CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_features_for_split(
    records: list[dict[str, str]],
    project_root: Path,
    scaler: FeatureScaler,
    audio_cfg: Any,
    feat_cfg: Any,
) -> list[dict[str, Any]]:
    """Load audio and extract normalized features for each record in the split."""
    data = []
    for r in records:
        full_path = project_root / r["relative_path"]
        signal = load_audio(full_path, config=audio_cfg)
        feat_tensor = extract_features(signal, config=feat_cfg, scaler=scaler)
        data.append({
            "record": r,
            "feature_tensor": feat_tensor,
            "label_int": 1 if r["label"] == "abnormal" else 0,
        })
    return data


def plot_loss_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save training and validation loss curve diagnostic plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train MSE Loss", color="#2563EB", linewidth=1.8)
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], label="Validation MSE Loss", color="#DC2626", linewidth=1.8, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PulseForge AI — Autoencoder Training & Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved loss curve plot to {output_path}")


def train_autoencoder_pipeline(
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    target_fpr: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute full Autoencoder training, validation threshold selection, and serialization."""
    cfg = get_default_config()
    project_root = cfg.paths.project_root

    manifest_p = manifest_path or (cfg.paths.manifests_dir / "fan_manifest.csv")
    artifacts_d = output_dir or cfg.paths.model_artifacts_dir
    results_d = results_dir or cfg.paths.results_dir

    artifacts_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_p)
    logger.info(f"Loaded {len(records)} records from manifest: {manifest_p.name}")

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    logger.info(
        f"Split partition: Train={len(train_recs)} normal, "
        f"Val={len(val_recs)} ({sum(1 for r in val_recs if r['label'] == 'normal')}N / {sum(1 for r in val_recs if r['label'] == 'abnormal')}A), "
        f"Test={len(test_recs)} ({sum(1 for r in test_recs if r['label'] == 'normal')}N / {sum(1 for r in test_recs if r['label'] == 'abnormal')}A)"
    )

    # 1. FeatureScaler: load existing or fit on train
    scaler_path = artifacts_d / "scaler_params.json"
    if scaler_path.exists():
        logger.info(f"Loading existing FeatureScaler from {scaler_path}")
        scaler = FeatureScaler.load(scaler_path)
    else:
        logger.info("Fitting FeatureScaler on normal training recordings...")
        train_feats_raw = []
        for r in train_recs:
            sig = load_audio(project_root / r["relative_path"], config=cfg.audio)
            ft = extract_features(sig, config=cfg.features)
            train_feats_raw.append(ft.features)
        scaler = FeatureScaler()
        scaler.fit(np.concatenate(train_feats_raw, axis=0))
        scaler.save(scaler_path)

    # 2. Extract features for training and validation splits
    logger.info("Extracting normalized features for training and validation splits...")
    train_data = extract_features_for_split(train_recs, project_root, scaler, cfg.audio, cfg.features)
    val_data = extract_features_for_split(val_recs, project_root, scaler, cfg.audio, cfg.features)

    train_matrix = np.concatenate([item["feature_tensor"].to_flattened() for item in train_data], axis=0)
    # Validation normal segments for loss tracking
    val_norm_items = [item for item in val_data if item["record"]["label"] == "normal"]
    val_matrix = np.concatenate([item["feature_tensor"].to_flattened() for item in val_norm_items], axis=0)

    logger.info(f"Training segments: {train_matrix.shape[0]}, Validation normal segments: {val_matrix.shape[0]}")

    model_cfg = ModelConfig(
        input_dim=4096,
        hidden_dims=(512, 128, 64),
        latent_dim=32,
        learning_rate=lr or cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        batch_size=batch_size or cfg.model.batch_size,
        epochs=epochs or cfg.model.epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        seed=seed,
        score_aggregation=cfg.model.score_aggregation,
        target_fpr=target_fpr,
    )

    detector = AutoencoderDetector(config=model_cfg)

    # 3. Fit Autoencoder
    t0_fit = time.perf_counter()
    history = detector.fit(
        train_features=train_matrix,
        val_features=val_matrix,
        epochs=model_cfg.epochs,
        batch_size=model_cfg.batch_size,
        lr=model_cfg.learning_rate,
        verbose=True,
    )
    fit_duration_s = time.perf_counter() - t0_fit
    logger.info(f"Autoencoder fitting completed in {fit_duration_s:.2f}s ({len(history['train_loss'])} epochs run).")

    # Save loss plot
    loss_plot_path = results_d / "plots" / "autoencoder_loss_curve.png"
    plot_loss_curves(history, loss_plot_path)

    # 4. Derive validation threshold
    logger.info("Computing validation recording reconstruction scores for threshold selection...")
    val_norm_scores = [
        detector.predict_recording_score(item["feature_tensor"])
        for item in val_data if item["record"]["label"] == "normal"
    ]
    val_abn_scores = [
        detector.predict_recording_score(item["feature_tensor"])
        for item in val_data if item["record"]["label"] == "abnormal"
    ]

    threshold = detector.select_threshold(
        val_normal_scores=val_norm_scores,
        val_abnormal_scores=val_abn_scores,
        target_fpr=target_fpr,
    )

    # 5. Save model checkpoint and training metadata
    model_save_path = artifacts_d / "autoencoder_fan_id_00.pt"
    detector.save(model_save_path)

    summary = {
        "model_type": "AcousticAutoencoder",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_type": cfg.default_machine_type,
        "machine_id": "id_00",
        "noise_condition": "6_dB",
        "split_counts": {
            "train": len(train_recs),
            "val_normal": len(val_norm_scores),
            "val_abnormal": len(val_abn_scores),
            "test": len(test_recs),
        },
        "training_duration_seconds": round(fit_duration_s, 2),
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": round(history["train_loss"][-1], 6),
        "final_val_loss": round(history["val_loss"][-1], 6) if history.get("val_loss") else None,
        "selected_threshold": round(threshold, 4),
        "target_fpr": target_fpr,
        "architecture": {
            "input_dim": model_cfg.input_dim,
            "hidden_dims": list(model_cfg.hidden_dims),
            "latent_dim": model_cfg.latent_dim,
        },
    }

    summary_file = results_d / "autoencoder_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved Autoencoder training summary to {summary_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Acoustic Autoencoder for fan anomaly detection")
    parser.add_argument("--manifest", type=str, default="", help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--target-fpr", type=float, default=0.10, help="Target FPR for validation threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    manifest_p = Path(args.manifest) if args.manifest else None
    summary = train_autoencoder_pipeline(
        manifest_path=manifest_p,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_fpr=args.target_fpr,
        seed=args.seed,
    )
    print("\nTraining summary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
