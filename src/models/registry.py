"""
Dynamic Multi-Machine Model Registry for PulseForge AI (Milestone M12).

Manages loading, caching, and lifecycle of machine-specific anomaly detection bundles
(PyTorch Autoencoder, FeatureScaler, LatentFamiliarityEstimator) across industrial
equipment families: fan, pump, valve, and slider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.familiarity import LatentFamiliarityEstimator
from src.features import FeatureScaler
from src.models.autoencoder import AutoencoderDetector

logger = logging.getLogger("pulseforge.registry")


@dataclass
class MachineModelBundle:
    """Complete ML inference bundle for a specific machine family."""

    machine_type: str
    machine_id: str = "id_00"
    detector: AutoencoderDetector | None = None
    scaler: FeatureScaler | None = None
    familiarity: LatentFamiliarityEstimator | None = None

    @property
    def is_ready(self) -> bool:
        """Return True if detector, scaler, and familiarity estimator are ready."""
        return (
            self.detector is not None
            and self.scaler is not None
            and self.familiarity is not None
            and self.detector.is_trained
            and self.detector.threshold is not None
        )

    def to_status_dict(self) -> dict[str, Any]:
        """Serialize bundle readiness and calibrated thresholds."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "model_ready": self.is_ready,
            "autoencoder_loaded": self.detector is not None,
            "autoencoder_threshold": (
                round(float(self.detector.threshold), 6)
                if self.detector is not None and self.detector.threshold is not None
                else None
            ),
            "scaler_loaded": self.scaler is not None,
            "familiarity_loaded": self.familiarity is not None,
            "familiarity_threshold": (
                round(float(self.familiarity.threshold), 6)
                if self.familiarity is not None and self.familiarity.threshold is not None
                else None
            ),
        }


class ModelRegistry:
    """
    Registry discovering and routing acoustic anomaly detection model bundles
    by machine type ('fan', 'pump', 'valve', 'slider').
    """

    def __init__(
        self,
        model_dir: Path,
        supported_types: tuple[str, ...] = ("fan", "pump", "valve", "slider"),
        default_machine_id: str = "id_00",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.supported_types = tuple(t.lower().strip() for t in supported_types)
        self.default_machine_id = default_machine_id
        self._bundles: dict[str, MachineModelBundle] = {}
        self.load_all()

    def _resolve_artifact_paths(
        self, machine_type: str, machine_id: str
    ) -> tuple[Path, Path, Path]:
        """Resolve Autoencoder, Scaler, and Familiarity artifact file paths."""
        ae_path = self.model_dir / f"autoencoder_{machine_type}_{machine_id}.pt"
        scaler_path = self.model_dir / f"scaler_{machine_type}_{machine_id}.json"

        # Preserve M2/M4 backward-compatible scaler filename for 'fan'
        if machine_type == "fan" and not scaler_path.is_file():
            legacy_scaler = self.model_dir / "scaler_params.json"
            if legacy_scaler.is_file():
                scaler_path = legacy_scaler

        fam_path = self.model_dir / f"familiarity_{machine_type}_{machine_id}.joblib"
        return ae_path, scaler_path, fam_path

    def load_machine(
        self, machine_type: str, machine_id: str | None = None
    ) -> MachineModelBundle:
        """Load and cache the model bundle for a specific machine type."""
        m_type = machine_type.lower().strip()
        m_id = machine_id or self.default_machine_id
        ae_path, scaler_path, fam_path = self._resolve_artifact_paths(m_type, m_id)

        detector: AutoencoderDetector | None = None
        scaler: FeatureScaler | None = None
        familiarity: LatentFamiliarityEstimator | None = None

        if ae_path.is_file():
            try:
                detector = AutoencoderDetector.load(ae_path)
                logger.info("[%s] Loaded Autoencoder from %s", m_type, ae_path.name)
            except Exception as exc:
                logger.error("[%s] Failed to load Autoencoder %s: %s", m_type, ae_path, exc)

        if scaler_path.is_file():
            try:
                scaler = FeatureScaler.load(scaler_path)
                logger.info("[%s] Loaded FeatureScaler from %s", m_type, scaler_path.name)
            except Exception as exc:
                logger.error("[%s] Failed to load FeatureScaler %s: %s", m_type, scaler_path, exc)

        if fam_path.is_file():
            try:
                familiarity = LatentFamiliarityEstimator.load(fam_path)
                logger.info("[%s] Loaded Familiarity estimator from %s", m_type, fam_path.name)
            except Exception as exc:
                logger.error("[%s] Failed to load Familiarity %s: %s", m_type, fam_path, exc)

        bundle = MachineModelBundle(
            machine_type=m_type,
            machine_id=m_id,
            detector=detector,
            scaler=scaler,
            familiarity=familiarity,
        )
        self._bundles[m_type] = bundle
        return bundle

    def load_all(self) -> None:
        """Scan and load bundles for all configured supported machine types."""
        for m_type in self.supported_types:
            self.load_machine(m_type, self.default_machine_id)

    def get_bundle(self, machine_type: str = "fan") -> MachineModelBundle | None:
        """Retrieve cached model bundle for machine_type, returning None if unsupported."""
        m_type = machine_type.lower().strip()
        if m_type not in self.supported_types:
            return None
        if m_type not in self._bundles:
            return self.load_machine(m_type, self.default_machine_id)
        return self._bundles[m_type]

    def is_ready(self, machine_type: str = "fan") -> bool:
        """Check whether all required ML artifacts are loaded for machine_type."""
        bundle = self.get_bundle(machine_type)
        return bool(bundle is not None and bundle.is_ready)

    def available_machine_types(self) -> list[str]:
        """Return list of machine types whose full model bundles are ready for inference."""
        return [m for m in self.supported_types if self.is_ready(m)]

    def get_status_summary(self) -> dict[str, dict[str, Any]]:
        """Return detailed readiness and calibrated threshold telemetry per machine type."""
        return {
            m_type: self._bundles[m_type].to_status_dict()
            for m_type in self.supported_types
            if m_type in self._bundles
        }
