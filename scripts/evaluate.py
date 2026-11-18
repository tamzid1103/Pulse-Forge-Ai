"""
Unified Evaluation and Comparative Benchmark Script (Milestone M4).

Evaluates the Deep Autoencoder and the Isolation Forest baseline
side-by-side on the identical held-out test split (14 recordings: 6 normal, 8 abnormal).
Generates comparative metrics, ROC curves, reconstruction error distribution plots,
and serializes results to results/evaluation_report.json.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_audio
from src.config import get_default_config
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
from src.models.baseline import BaselineIsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.evaluate")


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load records from manifest CSV."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute comprehensive classification metrics at specified threshold."""
    preds = (scores >= threshold).astype(int)
    auroc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
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
    }


def plot_score_distributions(
    ae_scores: list[float],
    ae_threshold: float,
    labels: list[int],
    output_path: Path,
) -> None:
    """Plot anomaly score distributions for normal and abnormal test recordings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm_scores = [s for s, y in zip(ae_scores, labels) if y == 0]
    abn_scores = [s for s, y in zip(ae_scores, labels) if y == 1]

    plt.figure(figsize=(9, 5))
    bins = np.linspace(min(ae_scores) * 0.9, max(ae_scores) * 1.1, 15)
    plt.hist(norm_scores, bins=bins, alpha=0.6, color="#2563EB", label=f"Normal Fan (n={len(norm_scores)})", edgecolor="black")
    plt.hist(abn_scores, bins=bins, alpha=0.6, color="#DC2626", label=f"Anomalous Fan (n={len(abn_scores)})", edgecolor="black")

    plt.axvline(ae_threshold, color="#16A34A", linestyle="--", linewidth=2.0, label=f"Threshold = {ae_threshold:.4f}")

    plt.xlabel("Recording Reconstruction Error (MSE)")
    plt.ylabel("Recording Count")
    plt.title("PulseForge AI — Test Set Reconstruction Error Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved score distribution plot to {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    ae_scores: np.ndarray,
    baseline_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC curves comparing Autoencoder against Baseline Isolation Forest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr_ae, tpr_ae, _ = roc_curve(y_true, ae_scores)
    auroc_ae = roc_auc_score(y_true, ae_scores)

    fpr_base, tpr_base, _ = roc_curve(y_true, baseline_scores)
    auroc_base = roc_auc_score(y_true, baseline_scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_ae, tpr_ae, color="#2563EB", linewidth=2.0, label=f"Autoencoder (AUROC = {auroc_ae:.4f})")
    plt.plot(fpr_base, tpr_base, color="#F59E0B", linewidth=2.0, linestyle="--", label=f"Isolation Forest Baseline (AUROC = {auroc_base:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.7, label="Chance")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("PulseForge AI — ROC Curve Comparison (Test Split)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC comparison plot to {output_path}")


def evaluate_models(
    manifest_path: Path | None = None,
    ae_model_path: Path | None = None,
    baseline_model_path: Path | None = None,
    scaler_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute rigorous side-by-side evaluation of Autoencoder and Baseline on held-out test data."""
    cfg = get_default_config()
    project_root = cfg.paths.project_root

    manifest_p = manifest_path or (cfg.paths.manifests_dir / "fan_manifest.csv")
    ae_p = ae_model_path or (cfg.paths.model_artifacts_dir / "autoencoder_fan_id_00.pt")
    base_p = baseline_model_path or (cfg.paths.model_artifacts_dir / "baseline_isolation_forest.joblib")
    scaler_p = scaler_path or (cfg.paths.model_artifacts_dir / "scaler_params.json")
    results_d = results_dir or cfg.paths.results_dir

    if not ae_p.exists():
        raise FileNotFoundError(f"Autoencoder model not found: {ae_p}. Run scripts/train_autoencoder.py first.")
    if not base_p.exists():
        raise FileNotFoundError(f"Baseline model not found: {base_p}. Run scripts/train_baseline.py first.")
    if not scaler_p.exists():
        raise FileNotFoundError(f"FeatureScaler not found: {scaler_p}.")

    # Load models in fresh context
    logger.info(f"Loading Autoencoder from {ae_p}...")
    ae_detector = AutoencoderDetector.load(ae_p, device="cpu")

    logger.info(f"Loading Baseline from {base_p}...")
    base_detector = BaselineIsolationForest.load(base_p)

    scaler = FeatureScaler.load(scaler_p)

    records = load_manifest(manifest_p)
    test_recs = [r for r in records if r["split"] == "test"]
    logger.info(f"Loaded {len(test_recs)} test recordings (Normal={sum(1 for r in test_recs if r['label'] == 'normal')}, Abnormal={sum(1 for r in test_recs if r['label'] == 'abnormal')}).")

    y_true: list[int] = []
    ae_scores: list[float] = []
    base_scores: list[float] = []
    ae_latencies: list[float] = []
    base_latencies: list[float] = []

    for idx, r in enumerate(test_recs, 1):
        full_path = project_root / r["relative_path"]
        label_int = 1 if r["label"] == "abnormal" else 0
        y_true.append(label_int)

        sig = load_audio(full_path, config=cfg.audio)
        feat_tensor = extract_features(sig, config=cfg.features, scaler=scaler)

        # 1. Score Autoencoder
        t0 = time.perf_counter()
        ae_pred = ae_detector.predict(feat_tensor)
        ae_ms = (time.perf_counter() - t0) * 1000.0
        ae_scores.append(ae_pred.anomaly_score)
        ae_latencies.append(ae_ms)

        # 2. Score Baseline
        t0 = time.perf_counter()
        base_pred = base_detector.predict(feat_tensor)
        base_ms = (time.perf_counter() - t0) * 1000.0
        base_scores.append(base_pred.anomaly_score)
        base_latencies.append(base_ms)

    y_arr = np.array(y_true, dtype=int)
    ae_scores_arr = np.array(ae_scores, dtype=np.float32)
    base_scores_arr = np.array(base_scores, dtype=np.float32)

    ae_metrics = compute_metrics(y_arr, ae_scores_arr, ae_detector.threshold or 0.0)
    ae_metrics["mean_inference_latency_ms"] = round(float(np.mean(ae_latencies)), 2)
    ae_metrics["threshold"] = ae_detector.threshold

    base_metrics = compute_metrics(y_arr, base_scores_arr, base_detector.threshold or 0.0)
    base_metrics["mean_inference_latency_ms"] = round(float(np.mean(base_latencies)), 2)
    base_metrics["threshold"] = base_detector.threshold

    logger.info("=" * 60)
    logger.info("COMPARATIVE EVALUATION SUMMARY (Held-Out Test Split):")
    logger.info(f"  AUTOENCODER:  AUROC={ae_metrics['auroc']} | PR-AUC={ae_metrics['pr_auc']} | F1={ae_metrics['f1_score']} | Rec={ae_metrics['recall']} | Latency={ae_metrics['mean_inference_latency_ms']} ms")
    logger.info(f"  BASELINE IF:  AUROC={base_metrics['auroc']} | PR-AUC={base_metrics['pr_auc']} | F1={base_metrics['f1_score']} | Rec={base_metrics['recall']} | Latency={base_metrics['mean_inference_latency_ms']} ms")
    logger.info("=" * 60)

    # Generate diagnostic plots
    plot_score_distributions(
        ae_scores=ae_scores,
        ae_threshold=ae_detector.threshold or 0.0,
        labels=y_true,
        output_path=results_d / "plots" / "reconstruction_error_distribution.png",
    )

    plot_roc_curves(
        y_true=y_arr,
        ae_scores=ae_scores_arr,
        baseline_scores=base_scores_arr,
        output_path=results_d / "plots" / "roc_comparison.png",
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_type": "fan",
        "machine_id": "id_00",
        "noise_condition": "6_dB",
        "test_counts": {
            "total": len(test_recs),
            "normal": int(np.sum(y_arr == 0)),
            "abnormal": int(np.sum(y_arr == 1)),
        },
        "models": {
            "primary_autoencoder": {
                "model_type": "AcousticAutoencoder",
                "checkpoint": str(ae_p.name),
                "metrics": ae_metrics,
            },
            "baseline_isolation_forest": {
                "model_type": "BaselineIsolationForest",
                "checkpoint": str(base_p.name),
                "metrics": base_metrics,
            },
        },
        "comparison_summary": {
            "auroc_difference": round(ae_metrics["auroc"] - base_metrics["auroc"], 4),
            "f1_difference": round(ae_metrics["f1_score"] - base_metrics["f1_score"], 4),
            "latency_ratio": round(ae_metrics["mean_inference_latency_ms"] / max(0.01, base_metrics["mean_inference_latency_ms"]), 2),
        },
    }

    report_path = results_d / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved unified evaluation report to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified comparative evaluation of Autoencoder and Baseline")
    parser.add_argument("--manifest", type=str, default="", help="Path to manifest CSV")
    args = parser.parse_args()

    manifest_p = Path(args.manifest) if args.manifest else None
    report = evaluate_models(manifest_path=manifest_p)
    print("\nEvaluation Report JSON:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
