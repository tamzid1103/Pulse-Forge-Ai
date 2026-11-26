"""
Multi-Machine Training, Familiarity Calibration & Benchmark Script (Milestone M12).

Trains per-machine-family FeatureScaler, AcousticAutoencoder, and LatentFamiliarityEstimator
bundles across industrial rotating and fluid equipment families (fan, pump, valve, slider)
using leakage-aware splits from data/manifests/<machine_type>_manifest.csv.

Outputs:
- model_artifacts/scaler_<machine_type>_id_00.json
- model_artifacts/autoencoder_<machine_type>_id_00.pt (+ .json)
- model_artifacts/familiarity_<machine_type>_id_00.joblib (+ .json)
- demo_audio/06..11_<machine_type>_{normal,anomaly}.wav (320KB mono 16kHz held-out test demos)
- results/multi_machine_summary.json
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

import numpy as np
import soundfile as sf

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_audio
from src.config import ModelConfig, get_default_config
from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
from src.thresholding import compute_binary_metrics, select_threshold_iqr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.train_multi_machine")

DEMO_FILENAMES: dict[str, tuple[str, str]] = {
    "pump": ("06_pump_normal.wav", "07_pump_anomaly.wav"),
    "valve": ("08_valve_normal.wav", "09_valve_anomaly.wav"),
    "slider": ("10_slider_normal.wav", "11_slider_anomaly.wav"),
}


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read manifest rows from CSV."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_and_evaluate_machine(
    machine_type: str,
    manifest_path: Path | None = None,
    artifacts_dir: Path | None = None,
    results_dir: Path | None = None,
    demo_dir: Path | None = None,
    epochs: int = 80,
    batch_size: int = 64,
    target_fpr: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Train, calibrate, evaluate, and export artifacts for a single machine type."""
    cfg = get_default_config()
    root = cfg.paths.project_root
    m_type = machine_type.lower().strip()

    manifest_p = manifest_path or (cfg.paths.manifests_dir / f"{m_type}_manifest.csv")
    art_d = artifacts_dir or cfg.paths.model_artifacts_dir
    res_d = results_dir or cfg.paths.results_dir
    dmo_d = demo_dir or cfg.paths.demo_audio_dir

    art_d.mkdir(parents=True, exist_ok=True)
    res_d.mkdir(parents=True, exist_ok=True)
    dmo_d.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_p)
    train_recs = [r for r in records if r["split"] == "train" and r["label"] == "normal"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    if not train_recs or not val_recs or not test_recs:
        raise ValueError(f"[{m_type}] Incomplete splits in {manifest_p}")

    logger.info(
        "[%s] Manifest loaded: train=%d normal, val=%d, test=%d",
        m_type,
        len(train_recs),
        len(val_recs),
        len(test_recs),
    )

    # 1. Fit & Save machine-specific FeatureScaler strictly on normal train split
    train_raw_feats = []
    train_signals = []
    for r in train_recs:
        sig = load_audio(root / r["relative_path"], config=cfg.audio)
        train_signals.append(sig)
        ft_raw = extract_features(sig, config=cfg.features)
        train_raw_feats.append(ft_raw.features)

    scaler = FeatureScaler()
    scaler.fit(np.concatenate(train_raw_feats, axis=0))

    scaler_path = (
        art_d / "scaler_params.json"
        if m_type == "fan" and (art_d / "scaler_params.json").is_file()
        else art_d / f"scaler_{m_type}_id_00.json"
    )
    scaler.save(scaler_path)
    # Also save standardized fan scaler name if not present so registry can resolve both
    if m_type == "fan":
        scaler.save(art_d / "scaler_fan_id_00.json")

    # Extract normalized training features
    train_tensors = [extract_features(sig, config=cfg.features, scaler=scaler) for sig in train_signals]
    train_matrix = np.concatenate([ft.to_flattened() for ft in train_tensors], axis=0)

    # Extract normalized validation features
    val_items = []
    for r in val_recs:
        sig = load_audio(root / r["relative_path"], config=cfg.audio)
        ft = extract_features(sig, config=cfg.features, scaler=scaler)
        val_items.append({"record": r, "ft": ft, "label": 1 if r["label"] == "abnormal" else 0})

    val_norm_items = [x for x in val_items if x["label"] == 0]
    val_matrix = np.concatenate([x["ft"].to_flattened() for x in val_norm_items], axis=0)

    # 2. Train AutoencoderDetector
    model_cfg = ModelConfig(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        latent_dim=cfg.model.latent_dim,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        seed=seed,
        score_aggregation=cfg.model.score_aggregation,
        target_fpr=target_fpr,
    )
    detector = AutoencoderDetector(config=model_cfg)
    t0 = time.perf_counter()
    history = detector.fit(
        train_features=train_matrix,
        val_features=val_matrix,
        epochs=epochs,
        batch_size=batch_size,
        lr=model_cfg.learning_rate,
        verbose=False,
    )
    train_s = time.perf_counter() - t0

    val_norm_scores = [detector.predict_recording_score(x["ft"]) for x in val_norm_items]
    val_abn_scores = [detector.predict_recording_score(x["ft"]) for x in val_items if x["label"] == 1]
    ae_threshold = detector.select_threshold(
        val_normal_scores=val_norm_scores,
        val_abnormal_scores=val_abn_scores,
        target_fpr=target_fpr,
    )

    ae_path = art_d / f"autoencoder_{m_type}_id_00.pt"
    detector.save(ae_path)

    # 3. Fit & Calibrate LatentFamiliarityEstimator
    train_latents_list = [detector.extract_latent_embeddings(ft) for ft in train_tensors]
    stacked_train_latents = np.vstack(train_latents_list)
    familiarity = LatentFamiliarityEstimator(epsilon=1e-4, aggregation="mean")
    familiarity.fit(stacked_train_latents)

    val_latents_list = [detector.extract_latent_embeddings(x["ft"]) for x in val_norm_items]
    fam_train_scores = [familiarity.score_recording(lat) for lat in train_latents_list]
    fam_val_scores = [familiarity.score_recording(lat) for lat in val_latents_list]
    combined_fam = fam_train_scores + fam_val_scores
    # Combine Tukey extreme fence (k=3.0) with 2.25x max normal baseline so strong
    # machine faults (e.g. pump cavitation) trigger POSSIBLE_ANOMALY while true
    # environmental OOD noise (40-70+ latent distance) triggers NEEDS_REVIEW.
    fam_threshold = float(max(select_threshold_iqr(combined_fam, k=3.0), 2.25 * max(combined_fam)))
    familiarity.threshold = fam_threshold

    fam_path = art_d / f"familiarity_{m_type}_id_00.joblib"
    familiarity.save(fam_path)
    fam_meta_path = art_d / f"familiarity_{m_type}_id_00.json"
    with open(fam_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "machine_type": m_type,
                "machine_id": "id_00",
                "model_type": "LatentFamiliarityEstimator",
                "latent_dim": familiarity.latent_dim,
                "threshold": round(fam_threshold, 6),
                "num_training_samples": familiarity.num_samples_fitted,
            },
            f,
            indent=2,
        )

    # 4. Evaluate on held-out Test split (and export mono 16kHz demo samples for non-fan machines)
    test_scores: list[float] = []
    test_labels: list[int] = []
    best_norm_candidate: tuple[float, Path] | None = None
    best_abn_candidate: tuple[float, Path] | None = None

    for r in test_recs:
        p = root / r["relative_path"]
        sig = load_audio(p, config=cfg.audio)
        ft = extract_features(sig, config=cfg.features, scaler=scaler)
        sc = float(detector.predict_recording_score(ft))
        fam_sc = float(familiarity.score_recording(detector.extract_latent_embeddings(ft)))
        lbl = 1 if r["label"] == "abnormal" else 0
        test_scores.append(sc)
        test_labels.append(lbl)

        if lbl == 0:
            if fam_sc <= fam_threshold and (best_norm_candidate is None or sc < best_norm_candidate[0]):
                best_norm_candidate = (sc, p)
        else:
            if sc >= ae_threshold and fam_sc <= fam_threshold:
                if best_abn_candidate is None or sc > best_abn_candidate[0]:
                    best_abn_candidate = (sc, p)

    metrics = compute_binary_metrics(
        y_true=test_labels,
        scores=test_scores,
        threshold=ae_threshold,
    )

    # Export representative held-out test mono 16kHz demo audio clips for pump/valve/slider
    exported_demos: dict[str, str] = {}
    if m_type in DEMO_FILENAMES:
        norm_name, abn_name = DEMO_FILENAMES[m_type]
        if best_norm_candidate is not None:
            sig_n = load_audio(best_norm_candidate[1], config=cfg.audio)
            out_n = dmo_d / norm_name
            sf.write(str(out_n), sig_n.data, sig_n.sample_rate, subtype="PCM_16")
            exported_demos["normal"] = out_n.name
        if best_abn_candidate is not None:
            sig_a = load_audio(best_abn_candidate[1], config=cfg.audio)
            out_a = dmo_d / abn_name
            sf.write(str(out_a), sig_a.data, sig_a.sample_rate, subtype="PCM_16")
            exported_demos["abnormal"] = out_a.name

    summary = {
        "machine_type": m_type,
        "machine_id": "id_00",
        "noise_condition": "6_dB",
        "train_files": len(train_recs),
        "val_files": len(val_recs),
        "test_files": len(test_recs),
        "epochs_trained": len(history["train_loss"]),
        "training_seconds": round(train_s, 2),
        "ae_threshold": round(float(ae_threshold), 6),
        "familiarity_threshold": round(float(fam_threshold), 6),
        "test_metrics": {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        "exported_demos": exported_demos,
    }
    logger.info(
        "[%s] Complete! AE_thr=%.4f | Fam_thr=%.4f | Test AUROC=%.4f | Recall=%.4f | F1=%.4f",
        m_type,
        ae_threshold,
        fam_threshold,
        metrics.get("auroc", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1_score", 0.0),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-machine acoustic anomaly detectors")
    parser.add_argument(
        "--machines",
        nargs="+",
        default=["pump", "valve", "slider"],
        help="Machine types to train (e.g. pump valve slider)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs per machine")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--target-fpr", type=float, default=0.10, help="Target validation FPR")
    args = parser.parse_args()

    cfg = get_default_config()
    results_summary: dict[str, Any] = {}

    for m_type in args.machines:
        results_summary[m_type] = train_and_evaluate_machine(
            machine_type=m_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_fpr=args.target_fpr,
        )

    out_path = cfg.paths.results_dir / "multi_machine_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Saved multi-machine evaluation summary to %s", out_path)
    print(json.dumps(results_summary, indent=2))


if __name__ == "__main__":
    main()
