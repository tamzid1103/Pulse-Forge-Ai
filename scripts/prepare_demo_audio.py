"""
Prepare curated offline demonstration audio files for PulseForge AI (Milestone M6).

Creates 5 self-contained WAV audio files in demo_audio/ for offline demonstration:
1. 01_normal_fan.wav: Healthy operating fan (Expected: NORMAL)
2. 02_bearing_anomaly.wav: Genuine abnormal rattle/impact (Expected: POSSIBLE_ANOMALY)
3. 03_silent_sensor_failure.wav: Disconnected / zero energy sensor (Expected: NEEDS_REVIEW)
4. 04_clipped_sensor_saturation.wav: Severe gain overflow clipping (Expected: NEEDS_REVIEW)
5. 05_unfamiliar_operating_noise.wav: Out-of-domain background acoustic noise (Expected: NEEDS_REVIEW)
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from src.config import AppConfig, setup_logging
from src.service import PulseForgeService
from src.persistence import AnalysisStore

logger = logging.getLogger("pulseforge.prepare_demo_audio")


def main() -> None:
    setup_logging("INFO")
    cfg = AppConfig()
    demo_dir = cfg.paths.project_root / "demo_audio"
    demo_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.audio.sample_rate

    raw_dir = cfg.paths.data_dir / "raw" / "id_00"

    # 1. 01_normal_fan.wav
    norm_src = raw_dir / "normal" / "00000000.wav"
    norm_dest = demo_dir / "01_normal_fan.wav"
    if norm_src.is_file():
        shutil.copy2(norm_src, norm_dest)
        logger.info("Created %s from %s", norm_dest.name, norm_src)
    else:
        logger.error("Source normal file not found: %s", norm_src)

    # 2. 02_bearing_anomaly.wav
    abn_src = raw_dir / "abnormal" / "00000009.wav"
    abn_dest = demo_dir / "02_bearing_anomaly.wav"
    if abn_src.is_file():
        shutil.copy2(abn_src, abn_dest)
        logger.info("Created %s from %s", abn_dest.name, abn_src)
    else:
        logger.error("Source abnormal file not found: %s", abn_src)

    # 3. 03_silent_sensor_failure.wav (10s of silence with 1e-6 noise)
    silent_dest = demo_dir / "03_silent_sensor_failure.wav"
    np.random.seed(42)
    silent_data = (np.random.randn(sr * 10) * 1e-6).astype(np.float32)
    sf.write(str(silent_dest), silent_data, sr, subtype="PCM_16")
    logger.info("Created %s (RMS: %.8f)", silent_dest.name, float(np.sqrt(np.mean(silent_data**2))))

    # 4. 04_clipped_sensor_saturation.wav (10s sine wave clipped to +/- 1.0)
    clipped_dest = demo_dir / "04_clipped_sensor_saturation.wav"
    t = np.linspace(0, 10.0, sr * 10, endpoint=False)
    raw_clipped = 8.0 * np.sin(2 * np.pi * 120 * t)
    clipped_data = np.clip(raw_clipped, -1.0, 1.0).astype(np.float32)
    sf.write(str(clipped_dest), clipped_data, sr, subtype="PCM_16")
    logger.info("Created %s (clipping ratio: %.2f%%)", clipped_dest.name, float(np.mean(np.abs(clipped_data) >= 0.999) * 100))

    # 5. 05_unfamiliar_operating_noise.wav (10s frequency modulated noise)
    unfam_dest = demo_dir / "05_unfamiliar_operating_noise.wav"
    noise_mod = np.random.uniform(-0.4, 0.4, sr * 10).astype(np.float32)
    carrier = 0.2 * np.sin(2 * np.pi * 1500 * t).astype(np.float32)
    unfam_data = (noise_mod + carrier).astype(np.float32)
    sf.write(str(unfam_dest), unfam_data, sr, subtype="PCM_16")
    logger.info("Created %s (out-of-domain synthetic)", unfam_dest.name)

    # Verify all 5 demo scenarios with PulseForgeService
    logger.info("--- Verifying all 5 demo files with PulseForgeService ---")
    service = PulseForgeService(config=cfg, store=AnalysisStore(":memory:"))

    demo_files = [
        ("01_normal_fan.wav", "normal"),
        ("02_bearing_anomaly.wav", "possible_anomaly"),
        ("03_silent_sensor_failure.wav", "needs_review"),
        ("04_clipped_sensor_saturation.wav", "needs_review"),
        ("05_unfamiliar_operating_noise.wav", "needs_review"),
    ]

    all_passed = True
    for fname, expected_status in demo_files:
        fpath = demo_dir / fname
        result, sig, _ = service.analyze_audio(fpath, filename=fname, record_to_db=False)
        passed = (result.status.value == expected_status)
        status_icon = "PASS" if passed else "FAIL"
        logger.info(
            "[%s] %s -> Result: %s (Expected: %s)",
            status_icon,
            fname,
            result.status.value,
            expected_status,
        )
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("SUCCESS: All 5 demo scenarios produce their exact expected status!")
    else:
        logger.error("FAILURE: One or more demo scenarios failed verification.")


if __name__ == "__main__":
    main()
