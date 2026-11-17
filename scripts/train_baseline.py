"""
Training and Evaluation Script for Baseline Anomaly Detector (Milestone M3).

Executes:
1. Loads dataset manifest and separates train, val, and test splits.
2. Extracts and normalizes features using FeatureScaler fitted strictly on train.
3. Fits BaselineIsolationForest exclusively on normal training recordings.
4. Selects anomaly threshold on validation split (without touching test split).
5. Evaluates metrics once on the held-out test split (AUROC, PR-AUC, F1, Precision, Recall, FPR).
6. Serializes the trained baseline model to model_artifacts/ and metrics to results/baseline_metrics.json.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_audio
from src.config import BaselineConfig, get_default_config
from src.features import FeatureScaler, extract_features
from src.models.baseline import BaselineIsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.train_baseline")


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate rows from the dataset manifest CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest at {manifest_path} is empty.")
    return rows


def extract_features_for_records(
    records: list[dict[str, str]],
    project_root: Path,
    scaler: FeatureScaler,
    audio_cfg: Any,
    feat_cfg: Any,
) -> list[dict[str, Any]]:
    """Load audio and extract normalized features for each recording record."""
    processed: list[dict[str, Any]] = []
    for r in records:
        full_path = project_root / r["relative_path"]
        audio_signal = load_audio(full_path, config=audio_cfg)
        feature_tensor = extract_features(audio_signal, config=feat_cfg, scaler=scaler)
        processed.append({
            "record": r,
            "feature_tensor": feature_tensor,
            "label_int": 1 if r["label"] == "abnormal" else 0,
        })
    return processed


def run_baseline_pipeline(
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
    target_fpr: float = 0.10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute full baseline training, validation-based thresholding, and test evaluation."""
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
        f"Split partition: Train={len(train_recs)} (all normal), "
        f"Val={len(val_recs)} ({sum(1 for r in val_recs if r['label'] == 'normal')}N / {sum(1 for r in val_recs if r['label'] == 'abnormal')}A), "
        f"Test={len(test_recs)} ({sum(1 for r in test_recs if r['label'] == 'normal')}N / {sum(1 for r in test_recs if r['label'] == 'abnormal')}A)"
    )

    # 1. FeatureScaler: load existing or fit on train
    scaler_path = artifacts_d / "scaler_params.json"
    if scaler_path.exists():
        logger.info(f"Loading existing FeatureScaler from {scaler_path}")
        scaler = FeatureScaler.load(scaler_path)
    else:
        logger.info("Fitting new FeatureScaler on normal training recordings...")
        train_features_raw = []
        for r in train_recs:
            sig = load_audio(project_root / r["relative_path"], config=cfg.audio)
            ft = extract_features(sig, config=cfg.features)
            train_features_raw.append(ft.features)
        scaler = FeatureScaler()
        scaler.fit(np.concatenate(train_features_raw, axis=0))
        scaler.save(scaler_path)

    # 2. Extract normalized features
    logger.info("Extracting normalized features for training split...")
    train_data = extract_features_for_records(train_recs, project_root, scaler, cfg.audio, cfg.features)

    baseline_cfg = BaselineConfig(
        n_estimators=100,
        random_state=random_state,
        pooling_strategy="mean_std",
        score_aggregation="mean",
        target_fpr=target_fpr,
    )
    detector = BaselineIsolationForest(config=baseline_cfg)

    # Collect pooled segment feature vectors for training
    train_vectors_list = [detector.pool_features(item["feature_tensor"]) for item in train_data]
    train_matrix = np.concatenate(train_vectors_list, axis=0)
    logger.info(f"Total training segments: {train_matrix.shape[0]}, feature dim: {train_matrix.shape[1]}")

    # 3. Fit Isolation Forest
    t0_fit = time.perf_counter()
    detector.fit(train_matrix)
    fit_duration_s = time.perf_counter() - t0_fit
    logger.info(f"Baseline fit finished in {fit_duration_s:.3f}s")

    # 4. Validation: extract features and select threshold
    logger.info("Evaluating on validation split for threshold selection...")
    val_data = extract_features_for_records(val_recs, project_root, scaler, cfg.audio, cfg.features)

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
    logger.info(f"Selected anomaly threshold: {threshold:.4f}")

    # 5. Held-out Test Set Evaluation
    logger.info("Evaluating on held-out test split...")
    test_data = extract_features_for_records(test_recs, project_root, scaler, cfg.audio, cfg.features)

    y_test: list[int] = []
    test_scores: list[float] = []
    latencies: list[float] = []

    for item in test_data:
        t0_inf = time.perf_counter()
        pred = detector.predict(item["feature_tensor"])
        dt_ms = (time.perf_counter() - t0_inf) * 1000.0

        y_test.append(item["label_int"])
        test_scores.append(pred.anomaly_score)
        latencies.append(dt_ms)

    y_test_arr = np.array(y_test, dtype=int)
    scores_arr = np.array(test_scores, dtype=np.float32)
    preds_arr = (scores_arr >= threshold).astype(int)

    # Compute metrics
    auroc = float(roc_auc_score(y_test_arr, scores_arr))
    pr_auc = float(average_precision_score(y_test_arr, scores_arr))
    f1 = float(f1_score(y_test_arr, preds_arr, zero_division=0))
    prec = float(precision_score(y_test_arr, preds_arr, zero_division=0))
    rec = float(recall_score(y_test_arr, preds_arr, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test_arr, preds_arr).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    mean_latency_ms = float(np.mean(latencies))

    logger.info("=" * 60)
    logger.info("BASELINE TEST EVALUATION RESULTS:")
    logger.info(f"  AUROC:     {auroc:.4f}")
    logger.info(f"  PR-AUC:    {pr_auc:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  FPR:       {fpr:.4f}")
    logger.info(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.info(f"  Mean inference latency: {mean_latency_ms:.2f} ms")
    logger.info("=" * 60)

    # 6. Save model and metrics report
    model_save_path = artifacts_d / "baseline_isolation_forest.joblib"
    detector.save(model_save_path)

    metrics_payload = {
        "model_type": "BaselineIsolationForest",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_type": cfg.default_machine_type,
        "machine_id": "id_00",
        "noise_condition": "6_dB",
        "split_counts": {
            "train": len(train_recs),
            "val_normal": len(val_norm_scores),
            "val_abnormal": len(val_abn_scores),
            "test_normal": int(tn + fp),
            "test_abnormal": int(tp + fn),
            "total_test": len(test_recs),
        },
        "config": {
            "n_estimators": baseline_cfg.n_estimators,
            "pooling_strategy": baseline_cfg.pooling_strategy,
            "score_aggregation": baseline_cfg.score_aggregation,
            "target_fpr": baseline_cfg.target_fpr,
            "random_state": baseline_cfg.random_state,
            "feature_dim": detector.feature_dim,
        },
        "threshold": round(threshold, 4),
        "test_metrics": {
            "auroc": round(auroc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
            },
            "mean_inference_latency_ms": round(mean_latency_ms, 2),
        },
    }

    metrics_file = results_d / "baseline_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved baseline evaluation metrics to {metrics_file}")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Isolation Forest baseline")
    parser.add_argument("--manifest", type=str, default="", help="Path to manifest CSV")
    parser.add_argument("--target-fpr", type=float, default=0.10, help="Target FPR for threshold selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    manifest_p = Path(args.manifest) if args.manifest else None
    summary = run_baseline_pipeline(
        manifest_path=manifest_p,
        target_fpr=args.target_fpr,
        random_state=args.seed,
    )
    print("\nTraining summary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
