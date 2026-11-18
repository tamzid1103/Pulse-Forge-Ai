"""
Autoencoder Anomaly Model for PulseForge AI.

Milestone M4: Implements a PyTorch-based Deep Autoencoder trained
on normal industrial fan acoustic features to detect anomalous patterns
via reconstruction error.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig
from src.features import FeatureTensor

logger = logging.getLogger("pulseforge.models.autoencoder")


@dataclass(frozen=True)
class AutoencoderPrediction:
    """Prediction output from the Autoencoder Anomaly Detector."""

    anomaly_score: float
    threshold: float
    is_anomaly: bool
    status: Literal["normal", "possible_anomaly"]
    reconstruction_errors: list[float]
    score_aggregation: str


class AcousticAutoencoder(nn.Module):
    """
    Symmetric fully connected Autoencoder for acoustic log-Mel spectrogram segments.

    Architecture:
        Input: (B, 4096) [128 Mel bands * 32 time frames]
        Encoder: 4096 -> 512 -> 128 -> 64 -> 32 (latent space)
        Decoder: 32 -> 64 -> 128 -> 512 -> 4096 (reconstruction)
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: tuple[int, ...] = (512, 128, 64),
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder construction
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1, inplace=True),
            ])
            in_dim = h_dim
        encoder_layers.extend([
            nn.Linear(in_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder construction (mirrors encoder)
        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1, inplace=True),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features into low-dimensional latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct full feature dimension from latent vector."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass: x -> latent -> x_hat."""
        z = self.encode(x)
        return self.decode(z)


class AutoencoderDetector:
    """
    High-level Autoencoder Anomaly Detector for acoustic FeatureTensors.

    Handles training on normal acoustic segments, inference, reconstruction
    error calculation, validation thresholding, and model serialization.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        threshold: float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.threshold = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = AcousticAutoencoder(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            latent_dim=self.config.latent_dim,
        ).to(self.device)

        self.is_trained: bool = False

    def fit(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the autoencoder on normal acoustic feature segments.

        Parameters:
            train_features: 2D array of shape (N_train_segments, 4096).
            val_features: Optional 2D array of shape (N_val_segments, 4096).
            epochs: Maximum training epochs (defaults to config.epochs).
            batch_size: Mini-batch size (defaults to config.batch_size).
            lr: Learning rate (defaults to config.learning_rate).
            verbose: Whether to log training progress.

        Returns:
            Dictionary with 'train_loss' and 'val_loss' history lists.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        n_epochs = epochs or self.config.epochs
        bs = batch_size or self.config.batch_size
        learning_rate = lr or self.config.learning_rate

        if train_features.ndim != 2 or train_features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected train_features with shape (N, {self.config.input_dim}), got {train_features.shape}"
            )

        train_tensor = torch.from_numpy(train_features.astype(np.float32))
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=len(train_tensor) > bs)

        val_loader: DataLoader | None = None
        if val_features is not None and len(val_features) > 0:
            val_tensor = torch.from_numpy(val_features.astype(np.float32))
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights: dict[str, Any] | None = None
        patience_counter = 0

        self.model.train()
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(1, num_batches)
            history["train_loss"].append(avg_train_loss)

            # Validation loss evaluation
            avg_val_loss = avg_train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for (v_batch,) in val_loader:
                        v_batch = v_batch.to(self.device)
                        v_rec = self.model(v_batch)
                        val_loss_sum += criterion(v_rec, v_batch).item()
                        val_batches += 1
                avg_val_loss = val_loss_sum / max(1, val_batches)
                self.model.train()
                history["val_loss"].append(avg_val_loss)
                scheduler.step(avg_val_loss)

            # Early stopping check
            eval_loss = avg_val_loss if val_loader is not None else avg_train_loss
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == n_epochs):
                val_str = f" | Val MSE: {avg_val_loss:.6f}" if val_loader else ""
                logger.info(f"Epoch [{epoch:03d}/{n_epochs:03d}] Train MSE: {avg_train_loss:.6f}{val_str}")

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best loss: {best_val_loss:.6f})")
                break

        # Restore best model state
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.model.eval()
        self.is_trained = True
        return history

    def reconstruct(self, tensor: FeatureTensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct a FeatureTensor and return:
        - reconstructed segments array: shape (num_segments, n_mels, n_frames)
        - segment MSE reconstruction errors: shape (num_segments,)
        """
        self.model.eval()
        feats = tensor.to_flattened()  # (num_segments, 4096)
        n_segs, n_mels, n_frames = tensor.features.shape

        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).to(self.device)
            x_rec = self.model(x)

            # Per-segment MSE along axis 1 (dimension 4096)
            segment_mse = torch.mean((x - x_rec) ** 2, dim=1).cpu().numpy()
            rec_array = x_rec.cpu().numpy().reshape(n_segs, n_mels, n_frames)

        return rec_array.astype(np.float32), segment_mse.astype(np.float32)

    def predict_segment_scores(self, tensor: FeatureTensor) -> np.ndarray:
        """
        Compute reconstruction error (MSE) for each segment.
        Higher score = poorer reconstruction = higher likelihood of anomaly.
        """
        _, segment_mse = self.reconstruct(tensor)
        return segment_mse

    def predict_recording_score(self, tensor: FeatureTensor) -> float:
        """
        Compute recording-level anomaly score by aggregating segment MSEs.
        """
        seg_scores = self.predict_segment_scores(tensor)
        agg = self.config.score_aggregation

        if agg == "mean":
            rec_score = float(np.mean(seg_scores))
        elif agg == "max":
            rec_score = float(np.max(seg_scores))
        elif agg == "percentile_95":
            rec_score = float(np.percentile(seg_scores, 95))
        else:
            rec_score = float(np.mean(seg_scores))

        return rec_score

    def extract_latent_embeddings(self, tensor: FeatureTensor) -> np.ndarray:
        """
        Extract low-dimensional latent embeddings (num_segments, latent_dim).
        Useful for familiarity / domain shift checking in Milestone M5.
        """
        self.model.eval()
        feats = tensor.to_flattened()
        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).to(self.device)
            z = self.model.encode(x)
            return z.cpu().numpy().astype(np.float32)

    def select_threshold(
        self,
        val_normal_scores: list[float] | np.ndarray,
        val_abnormal_scores: list[float] | np.ndarray | None = None,
        target_fpr: float | None = None,
    ) -> float:
        """
        Determine anomaly threshold strictly from validation data.

        Quantile threshold: (1 - target_fpr) percentile of normal validation scores.
        """
        arr_norm = np.array(val_normal_scores, dtype=np.float32)
        if len(arr_norm) == 0:
            raise ValueError("val_normal_scores cannot be empty.")

        fpr = target_fpr if target_fpr is not None else self.config.target_fpr
        quantile = (1.0 - fpr) * 100.0
        threshold = float(np.percentile(arr_norm, quantile))
        logger.info(f"Selected Autoencoder threshold {threshold:.4f} at target FPR={fpr:.2f} (quantile={quantile:.1f}%)")

        if val_abnormal_scores is not None and len(val_abnormal_scores) > 0:
            arr_abn = np.array(val_abnormal_scores, dtype=np.float32)
            y_true = np.array([0] * len(arr_norm) + [1] * len(arr_abn))
            scores = np.concatenate([arr_norm, arr_abn])
            preds = (scores >= threshold).astype(int)

            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            logger.info(
                f"Validation performance at threshold {threshold:.4f}: "
                f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
            )

        self.threshold = threshold
        return threshold

    def predict(self, tensor: FeatureTensor) -> AutoencoderPrediction:
        """
        Run end-to-end inference and return a structured AutoencoderPrediction.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold must be selected or provided before running prediction.")

        seg_scores = self.predict_segment_scores(tensor)
        rec_score = self.predict_recording_score(tensor)

        is_anomaly = bool(rec_score >= self.threshold)
        status: Literal["normal", "possible_anomaly"] = (
            "possible_anomaly" if is_anomaly else "normal"
        )

        return AutoencoderPrediction(
            anomaly_score=round(rec_score, 4),
            threshold=round(self.threshold, 4),
            is_anomaly=is_anomaly,
            status=status,
            reconstruction_errors=[round(float(s), 4) for s in seg_scores],
            score_aggregation=self.config.score_aggregation,
        )

    def save(self, filepath: Path | str) -> tuple[Path, Path]:
        """
        Save the trained model checkpoint (.pt) and architecture metadata (.json).
        """
        pt_path = Path(filepath)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        if pt_path.suffix != ".pt":
            pt_path = pt_path.with_suffix(".pt")

        json_path = pt_path.with_suffix(".json")

        torch.save(self.model.state_dict(), pt_path)
        logger.info(f"Saved Autoencoder weights to {pt_path}")

        meta = {
            "model_type": "AcousticAutoencoder",
            "threshold": self.threshold,
            "architecture": {
                "input_dim": self.config.input_dim,
                "hidden_dims": list(self.config.hidden_dims),
                "latent_dim": self.config.latent_dim,
            },
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "seed": self.config.seed,
                "score_aggregation": self.config.score_aggregation,
                "target_fpr": self.config.target_fpr,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved Autoencoder metadata to {json_path}")

        return pt_path, json_path

    @classmethod
    def load(cls, filepath: Path | str, device: str = "cpu") -> AutoencoderDetector:
        """
        Load a trained AutoencoderDetector from checkpoint and metadata.
        """
        pt_path = Path(filepath)
        if pt_path.suffix != ".pt":
            pt_path = pt_path.with_suffix(".pt")

        json_path = pt_path.with_suffix(".json")

        if not pt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {pt_path}")

        meta: dict[str, Any] = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        arch = meta.get("architecture", {})
        tr_cfg = meta.get("training_config", {})

        cfg = ModelConfig(
            input_dim=arch.get("input_dim", 4096),
            hidden_dims=tuple(arch.get("hidden_dims", (512, 128, 64))),
            latent_dim=arch.get("latent_dim", 32),
            learning_rate=tr_cfg.get("learning_rate", 1e-3),
            weight_decay=tr_cfg.get("weight_decay", 1e-5),
            batch_size=tr_cfg.get("batch_size", 64),
            epochs=tr_cfg.get("epochs", 100),
            early_stopping_patience=tr_cfg.get("early_stopping_patience", 10),
            seed=tr_cfg.get("seed", 42),
            score_aggregation=tr_cfg.get("score_aggregation", "mean"),
            target_fpr=tr_cfg.get("target_fpr", 0.10),
        )

        detector = cls(config=cfg, threshold=meta.get("threshold"), device=device)
        state_dict = torch.load(pt_path, map_location=detector.device)
        detector.model.load_state_dict(state_dict)
        detector.model.eval()
        detector.is_trained = True

        logger.info(f"Loaded AutoencoderDetector from {pt_path} (threshold={detector.threshold})")
        return detector
