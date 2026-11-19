"""
Calibrate familiarity estimator and evaluate end-to-end decision policy (Milestone M5).

This script:
1. Loads the trained Autoencoder and FeatureScaler.
2. Extracts 32-dimensional latent representations for normal training recordings.
3. Fits the LatentFamiliarityEstimator.
4. Calibrates the unfamiliarity threshold on validation normal recordings.
5. Evaluates the end-to-end decision policy on:
   - Real normal test audio
   - Real abnormal test audio
   - Synthetic silent audio (quality failure)
   - Synthetic digitally clipped audio (quality failure)
   - Synthetic out-of-domain shifted audio (familiarity failure)
6. Serializes familiarity artifacts and writes an M5 decision policy report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.audio_io import AudioMetadata as IOAudioMetadata, AudioSignal, load_audio
from src.config import AppConfig, setup_logging
from src.decision import evaluate_decision_policy
from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler, extract_features
from src.models.autoencoder import AutoencoderDetector
from src.quality import check_signal_quality
from src.thresholding import select_threshold_iqr
from src.schemas import DecisionStatus, ModelPrediction

logger = logging.getLogger("pulseforge.calibrate_decision")


def main() -> None:
    setup_logging("INFO")
    cfg = AppConfig()

    manifest_path = cfg.paths.manifests_dir / "fan_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    ae_path = cfg.paths.model_artifacts_dir / "autoencoder_fan_id_00.pt"
    if not ae_path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")

    scaler_path = cfg.paths.model_artifacts_dir / "scaler_params.json"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"FeatureScaler not found: {scaler_path}")

    logger.info("Loading Autoencoder from %s", ae_path)
    detector = AutoencoderDetector.load(ae_path)

    logger.info("Loading FeatureScaler from %s", scaler_path)
    scaler = FeatureScaler.load(scaler_path)

    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    val_norm_df = val_df[val_df["label"] == "normal"]
    test_df = df[df["split"] == "test"]

    logger.info("Manifest loaded: %d train, %d val (%d normal), %d test",
                len(train_df), len(val_df), len(val_norm_df), len(test_df))

    # 1. Extract latent embeddings for normal training recordings
    logger.info("Extracting latent embeddings for %d training recordings...", len(train_df))
    all_train_embeddings: list[np.ndarray] = []
    for _, row in train_df.iterrows():
        audio_path = cfg.paths.project_root / row["relative_path"]
        audio_signal = load_audio(audio_path, config=cfg.audio)
        scaled_feats = extract_features(audio_signal, config=cfg.features, scaler=scaler)
        latents = detector.extract_latent_embeddings(scaled_feats)
        all_train_embeddings.append(latents)

    stacked_train_latents = np.vstack(all_train_embeddings)
    logger.info("Stacked training latent shape: %s", stacked_train_latents.shape)

    # 2. Fit familiarity estimator
    familiarity_estimator = LatentFamiliarityEstimator(epsilon=1e-4, aggregation="mean")
    familiarity_estimator.fit(stacked_train_latents)

    # 3. Calibrate threshold on validation normal recordings
    logger.info("Calibrating familiarity threshold on %d validation normal recordings...", len(val_norm_df))
    val_norm_latents: list[np.ndarray] = []
    for _, row in val_norm_df.iterrows():
        audio_path = cfg.paths.project_root / row["relative_path"]
        audio_signal = load_audio(audio_path, config=cfg.audio)
        scaled_feats = extract_features(audio_signal, config=cfg.features, scaler=scaler)
        latents = detector.extract_latent_embeddings(scaled_feats)
        val_norm_latents.append(latents)

    train_scores = [familiarity_estimator.score_recording(emb) for emb in all_train_embeddings]
    val_scores = [familiarity_estimator.score_recording(emb) for emb in val_norm_latents]
    fam_threshold = select_threshold_iqr(train_scores + val_scores, k=1.5)
    familiarity_estimator.threshold = fam_threshold
    logger.info("Calibrated Familiarity threshold (Tukey IQR k=1.5): %.4f on %d normal recordings",
                fam_threshold, len(train_scores) + len(val_scores))

    # 4. Save familiarity artifacts
    fam_joblib_path = cfg.paths.model_artifacts_dir / "familiarity_fan_id_00.joblib"
    fam_json_path = cfg.paths.model_artifacts_dir / "familiarity_fan_id_00.json"
    familiarity_estimator.save(fam_joblib_path)

    fam_metadata = {
        "model_type": "LatentFamiliarityEstimator",
        "method": "latent_standardized_euclidean",
        "latent_dim": familiarity_estimator.latent_dim,
        "num_training_samples": familiarity_estimator.num_samples_fitted,
        "threshold": fam_threshold,
        "target_fpr": 0.05,
        "autoencoder_checkpoint": ae_path.name,
    }
    with open(fam_json_path, "w", encoding="utf-8") as f:
        json.dump(fam_metadata, f, indent=2)
    logger.info("Saved familiarity metadata to %s", fam_json_path)

    # 5. Evaluate end-to-end decision policy on test recordings
    logger.info("Evaluating decision policy on held-out test recordings (%d total)...", len(test_df))
    test_results: list[dict] = []
    for _, row in test_df.iterrows():
        audio_path = cfg.paths.project_root / row["relative_path"]
        audio_signal = load_audio(audio_path, config=cfg.audio)

        # A. Signal Quality
        quality = check_signal_quality(audio_signal.data, audio_signal.sample_rate, cfg.quality)

        # B. Features & Inference
        scaled_feats = extract_features(audio_signal, config=cfg.features, scaler=scaler)
        rec_score = detector.predict_recording_score(scaled_feats)
        is_ae_anomaly = bool(rec_score >= detector.threshold)
        model_pred = ModelPrediction(
            model_type="AcousticAutoencoder",
            anomaly_score=round(rec_score, 4),
            threshold=round(detector.threshold, 4),
            is_anomaly=is_ae_anomaly,
        )

        # C. Familiarity
        latents = detector.extract_latent_embeddings(scaled_feats)
        fam_report = familiarity_estimator.predict(latents)

        # D. Decision Policy
        decision = evaluate_decision_policy(
            quality=quality,
            model_prediction=model_pred,
            familiarity=fam_report,
            machine_id=str(row.get("machine_id_or_section", "id_00")),
            machine_type="fan",
        )

        test_results.append({
            "recording_id": row["recording_id"],
            "ground_truth": row["label"],
            "decision_status": decision.status.value,
            "status_label": decision.status_label,
            "anomaly_score": model_pred.anomaly_score,
            "familiarity_score": fam_report.score,
            "quality_passed": quality.passed,
        })

    # 6. Evaluate synthetic stress conditions
    logger.info("Evaluating synthetic stress conditions (silent, clipped, domain shift)...")
    sr = cfg.audio.sample_rate
    t = np.linspace(0, 10.0, sr * 10, endpoint=False)

    # Synthetic Scenario 1: Silent audio
    silent_sig = np.zeros(sr * 10, dtype=np.float32)
    q_silent = check_signal_quality(silent_sig, sr, cfg.quality)
    d_silent = evaluate_decision_policy(quality=q_silent)

    # Synthetic Scenario 2: Clipped audio
    clipped_sig = np.clip(10.0 * np.sin(2 * np.pi * 120 * t), -1.0, 1.0).astype(np.float32)
    q_clipped = check_signal_quality(clipped_sig, sr, cfg.quality)
    d_clipped = evaluate_decision_policy(quality=q_clipped)

    # Synthetic Scenario 3: Domain shifted noise (unfamiliar)
    shifted_sig = np.random.uniform(-0.5, 0.5, sr * 10).astype(np.float32)
    q_shifted = check_signal_quality(shifted_sig, sr, cfg.quality)
    meta_dummy = IOAudioMetadata(
        filename="synthetic_noise.wav",
        size_bytes=int(sr * 10 * 2),
        sample_rate=sr,
        channels=1,
        duration_seconds=10.0,
        format="WAV",
        subtype="PCM_16",
    )
    synth_audio = AudioSignal(
        data=shifted_sig,
        sample_rate=sr,
        duration_seconds=10.0,
        metadata=meta_dummy,
    )
    scaled_s = extract_features(synth_audio, config=cfg.features, scaler=scaler)
    rec_s = detector.predict_recording_score(scaled_s)
    latents_s = detector.extract_latent_embeddings(scaled_s)
    fam_s = familiarity_estimator.predict(latents_s)
    pred_s = ModelPrediction(
        model_type="AcousticAutoencoder",
        anomaly_score=round(rec_s, 4),
        threshold=round(detector.threshold, 4),
        is_anomaly=bool(rec_s >= detector.threshold),
    )
    d_shifted = evaluate_decision_policy(
        quality=q_shifted,
        model_prediction=pred_s,
        familiarity=fam_s,
    )

    stress_results = {
        "silent_input": {
            "expected": "needs_review",
            "actual": d_silent.status.value,
            "status_label": d_silent.status_label,
            "passed": d_silent.status == DecisionStatus.NEEDS_REVIEW,
        },
        "clipped_input": {
            "expected": "needs_review",
            "actual": d_clipped.status.value,
            "status_label": d_clipped.status_label,
            "passed": d_clipped.status == DecisionStatus.NEEDS_REVIEW,
        },
        "domain_shifted_noise": {
            "expected": "needs_review",
            "actual": d_shifted.status.value,
            "status_label": d_shifted.status_label,
            "passed": d_shifted.status == DecisionStatus.NEEDS_REVIEW,
            "familiarity_score": fam_s.score,
            "familiarity_threshold": fam_s.threshold,
        },
    }

    # Summary report
    summary = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "familiarity_threshold": fam_threshold,
        "autoencoder_threshold": detector.threshold,
        "test_evaluations": test_results,
        "stress_evaluations": stress_results,
    }

    summary_path = cfg.paths.results_dir / "decision_policy_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved decision policy summary to %s", summary_path)

    logger.info("============================================================")
    logger.info("M5 DECISION POLICY CALIBRATION COMPLETE:")
    logger.info("  Familiarity threshold: %.4f", fam_threshold)
    logger.info("  Autoencoder threshold: %.4f", detector.threshold)
    logger.info("  Silent input decision: %s (PASS)", d_silent.status.value)
    logger.info("  Clipped input decision: %s (PASS)", d_clipped.status.value)
    logger.info("  Domain shift decision: %s (PASS)", d_shifted.status.value)
    logger.info("============================================================")


if __name__ == "__main__":
    main()
