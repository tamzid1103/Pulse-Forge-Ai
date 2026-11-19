# ISO 10816 & ISO 13374: Industrial Vibration and Acoustic Monitoring Standards
**Governing Standard:** ISO 10816-3 (Mechanical vibration — Evaluation of machine vibration by measurements on non-rotating parts) & ISO 13374 (Condition Monitoring and Diagnostics of Machine Systems)  
**Target Asset Class:** Class II and Class III Medium/Large Industrial Fans and Blowers (15 kW to 300 kW, Rigid & Flexible Foundations)

---

## 1. Vibration Severity Classification (ISO 10816-3)

While PulseForge AI evaluates airborne and structure-borne acoustic emissions, international maintenance audits cross-correlate acoustic reconstruction anomalies with vibration velocity RMS thresholds (measured in mm/s RMS from 10 Hz to 1000 Hz):

| Vibration Severity Zone | Velocity RMS (Rigid Foundation) | Operational Meaning | PulseForge AI Anomaly Equivalent | Recommended Maintenance Timeline |
| :--- | :--- | :--- | :--- | :--- |
| **Zone A (Excellent)** | $< 1.4\text{ mm/s}$ | Newly commissioned machine; optimal dynamic balance. | Low reconstruction error ($< 0.0020$). Latent space centered. | Routine monitoring. No action. |
| **Zone B (Acceptable)** | $1.4 - 2.8\text{ mm/s}$ | Long-term unrestricted continuous industrial service. | Baseline nominal error ($0.0020 - 0.0032$). | Standard scheduled lubrication. |
| **Zone C (Alert / Unsatisfactory)** | $2.8 - 4.5\text{ mm/s}$ | Machine can run temporarily, but corrective action must be scheduled. | Exceeds threshold ($> 0.0032$). Elevated high-frequency log-mel energy. | Priority physical inspection within 48 hours. |
| **Zone D (Unacceptable / Danger)** | $> 4.5\text{ mm/s}$ | Immediate risk of catastrophic bearing seizure or impeller destruction. | Massive reconstruction spikes ($> 0.0100$) or persistent transient clipping. | Immediate controlled shutdown and teardown. |

---

## 2. Acoustic Anomaly vs. Vibration Correlation

- **Early Warning Lead Time:** Acoustic anomaly detection (especially high-frequency spectral reconstruction via autoencoders) detects micro-surface distress (subsurface fatigue, lubrication starvation) **2 to 6 weeks earlier** than low-frequency 10–1000 Hz broadband vibration velocity rises into Zone C.
- **Root Cause Mapping:**
  - *Acoustic transient clicks without low-frequency vibration:* Bearing raceway micro-spalling or lack of hydrodynamic grease film.
  - *Acoustic rumbling synchronized with high 1X vibration:* Impeller unbalance (blade buildup) or severe shaft misalignment.
  - *High-frequency acoustic hissing:* Clogged intake filter or air flow aerodynamic vortex shedding.

---

## 3. ISO 13374 Architecture Mapping

PulseForge AI adheres directly to the six processing blocks defined in **ISO 13374-1 (Condition Monitoring Architecture)**:

1. **Data Acquisition (DA):** 16 kHz / 16-bit uncompressed WAV decoding with rigorous boundary validation (`src/audio_io.py`).
2. **Data Manipulation (DM):** 128-band Log-mel spectrogram calculation with standardized FFT frames (`src/features.py`).
3. **State Detection (SD):** Deep PyTorch Autoencoder reconstruction error calculation against calibrated thresholds (`src/autoencoder.py`).
4. **Health Assessment (HA):** Signal quality verification (RMS, clipping) combined with latent space Mahalanobis familiarity distance (`src/quality.py`, `src/familiarity.py`).
5. **Prognostic Assessment (PA):** Triage into explicit 4-state operational outcomes (`src/decision.py`).
6. **Advisory Generation (AG):** Factory Brain Copilot generating structured maintenance work orders and technician guidance (`src/copilot.py`).
