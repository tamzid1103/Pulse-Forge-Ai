# Hackathon Demonstration Guide

## 1. Demo objective

Show a complete, understandable, and honest product workflow: PulseForge AI receives a fan recording, analyzes it, exposes an anomaly score, checks input quality / familiarity, and tells the operator when a human should review the result.

The live demo must use the trained model and real local audio samples, not hardcoded predictions.

## 2. Recommended 3–5 minute demo sequence

### 0:00–0:30 — State the problem

“Factory machines can develop unusual operating patterns before a breakdown. A sound recording may contain a warning, but background noise and unfamiliar conditions can make an AI result unreliable. A simple anomaly label is not enough; operators also need to know when to trust or review the result.”

### 0:30–0:50 — Introduce the solution

“PulseForge AI is a fan acoustic predictive-maintenance prototype. It learns normal sound patterns, detects unusual recordings, checks signal quality and familiarity, and gives an evidence-based inspection recommendation.”

### 0:50–1:40 — Demo a normal case

1. Open the dashboard.
2. Choose the “Normal fan” demo sample.
3. Run analysis.
4. Show waveform and spectrogram.
5. Point to the score and validation-selected threshold.
6. Explain that this sample is below threshold and passes the configured checks.

### 1:40–2:30 — Demo an anomaly case

1. Choose the “Anomalous fan” sample.
2. Run analysis.
3. Show the score relative to threshold.
4. Explain the measurable evidence shown by the system.
5. Read the inspection recommendation.

### 2:30–3:20 — Demo a review-needed case

1. Choose a noisy / clipped / unfamiliar example.
2. Run analysis.
3. Show the quality or unfamiliarity warning.
4. Explain that PulseForge does not pretend to be certain when the input is unreliable.

### 3:20–4:00 — Explain implementation and impact

Highlight the MIMII data source, Autoencoder, baseline comparison, validation threshold, and reliability checks. Explain that a real factory pilot would require sensor evaluation, expert review, and operational validation.

## 3. Demo assets and curated scenario suite

A local `demo_audio/` folder is bundled in the repository containing 5 curated, representative audio files covering all operational branches. They allow offline, one-click demonstration during the video recording:

| File | Scenario | Acoustic Characteristics | Expected Decision Status |
| :--- | :--- | :--- | :--- |
| `01_normal_fan.wav` | Baseline Healthy Fan | Continuous steady 16 kHz fan hum; passes quality and familiarity checks. | **`NORMAL` (Green)** |
| `02_bearing_anomaly.wav` | Genuine Bearing Anomaly | Acoustic rattle and periodic impact pulses from genuine MIMII abnormal test set. Score exceeds threshold (0.8904). | **`POSSIBLE_ANOMALY` (Red)** |
| `03_silent_sensor_failure.wav` | Disconnected Microphone | Low-energy signal (RMS < 0.0010). Triggers silent sensor warning. | **`NEEDS_REVIEW` (Amber)** |
| `04_clipped_sensor_saturation.wav` | Sensor Gain Overflow | Heavy digital saturation (> 1% samples at +/- 1.0). Triggers clipping warning. | **`NEEDS_REVIEW` (Amber)** |
| `05_unfamiliar_operating_noise.wav` | Domain Shift / Background Noise | Acoustic distribution deviates from fan training baseline. Latent familiarity distance > 4.4682. | **`NEEDS_REVIEW` (Amber)** |

## 4. Live-demo checklist

### Before event

- [ ] Model artifact loads in a clean environment.
- [ ] App launches without network access.
- [ ] Demo WAV files are local and small.
- [ ] All three scenarios were tested end to end.
- [ ] The threshold and metric report match the shipped model version.
- [ ] Dataset attribution is present in slides/README.
- [ ] No large dataset archive is needed during the presentation.
- [ ] A backup laptop or a second local copy is available if possible.
- [ ] Presentation screenshots are saved.

### Immediately before presenting

- [ ] Close unnecessary applications.
- [ ] Confirm audio file paths.
- [ ] Confirm model-ready status.
- [ ] Keep the dashboard already open.
- [ ] Ensure screen resolution and plot text are readable.
- [ ] Have a backup slide with architecture and captured results.

## 5. Potential failure points and backups

| Failure point | Prevention | Backup |
|---|---|---|
| Internet unavailable | Keep dataset sample and model local | Use local demo samples; no live downloads |
| Dataset archive too large | Download one fan subset in advance | Use small approved subset already extracted |
| Model fails to load | Test clean restart and artifact checksum | Keep a verified model copy and captured result slides |
| Live inference is slow | Measure in advance; cache model; use short clips | Use a pre-run local analysis result with clear disclosure, while explaining the live pipeline |
| Upload codec / WAV issue | Test exact files using app | Use known-good WAV demo files |
| App crashes | Run regression and keep terminal visible to team | Restart from prepared run script; backup screenshots/video |
| Result is not as expected | Do not tune on the demo test sample | Explain current limitation honestly; use an independently verified sample |
| Network / external dependency | Keep MVP local | No API or cloud dependency required |

If a precomputed output is used as a backup, state clearly that it is a captured previous run; do not present it as a live prediction.

## 6. Suggested presentation structure

1. Title and one-line product promise.
2. Industrial downtime problem.
3. Why ordinary anomaly labels are insufficient.
4. PulseForge workflow and architecture.
5. Dataset and model approach.
6. Trustworthiness: signal quality and unfamiliarity checks.
7. Live three-case demo.
8. Measured evaluation results and limitations.
9. Business / deployment hypothesis and Bangladesh-first pilot path.
10. Future direction and closing.

## 7. Short project pitch structure

**Problem:** Unexpected machine downtime can interrupt factory production. Sound changes may provide useful warning signals, but noisy or unfamiliar recordings can mislead an AI detector.

**Solution:** PulseForge AI analyzes fan audio to detect unusual acoustic patterns and checks whether the recording is suitable and familiar enough to support the result.

**Innovation:** It communicates anomaly evidence and uncertainty separately, so the user knows when to investigate and when to be cautious.

**Technology:** MIMII fan audio, signal processing, Autoencoder anomaly detection, baseline comparison, and a Streamlit dashboard.

**Impact path:** Help maintenance personnel prioritize inspections. A real factory pilot would be the next validation step; the hackathon prototype does not yet prove downtime reduction.

## 8. Technical achievements worth highlighting (only when completed)

- Real industrial sound dataset inspected and prepared.
- Reproducible audio feature pipeline.
- Normal-only Autoencoder anomaly detector.
- Baseline comparison.
- Validation-selected anomaly threshold.
- Explicit signal-quality checks.
- Unfamiliarity / domain-shift warning.
- Clear human-review decision policy.
- Measured held-out evaluation and inference latency.
- End-to-end local dashboard.

Do not describe planned achievements as completed achievements. Replace this list with verified implementation evidence after building.

## 9. Competition alignment — planning assumption to verify

The earlier project-planning discussion used the following working judging frame:

| Area | Weight used in planning |
|---|---:|
| Innovation | 20% |
| Technical execution | 20% |
| Business model and global readiness | 20% |
| Real-world impact and ethical AI | 20% |
| Scalability and NRB collaboration | 10% |
| Presentation | 10% |

**Important:** these weights came from the prior planning discussion and have not been verified here against the event's official current judging rubric. Confirm the official rubric before preparing final slides. A previously discussed target score of 84/100 was an aspiration only, not a measured score or a prediction of winning.

How to address these areas honestly:

- **Innovation:** emphasize separate anomaly, signal-quality, and unfamiliarity communication.
- **Technical execution:** show real data, reproducible splits, a baseline comparison, and measured metrics.
- **Business/global readiness:** present subscription or per-machine pricing only as a business hypothesis; explain how machine adapters could enable expansion.
- **Impact/ethical AI:** explain human review, no automatic machine control, input limitations, and the need for a real factory pilot.
- **Scalability/NRB:** outline a potential Bangladesh-first test and possible expert advice from Bangladeshi professionals abroad; do not imply an existing collaboration.
- **Presentation:** run the three-case live demo and explain one limitation clearly.
