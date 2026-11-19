# PulseForge AI — Preliminary Submission Dossier

**Competition:** International AI Builders Congress 2026 (CloudCamp Bangladesh)  
**Submission Milestone:** Preliminary Screening (November 20, 2026)  
**Track:** IndustrySphere AI — Primary: Predictive Plant | Supporting: Factory Brain  
**GitHub Repository:** `https://github.com/tamzid-bd/Pulse-Forge-Ai` (Branch: `tamzid` / `main`)  
**Demo Video Link:** `[INSERT YOUTUBE / DRIVE LINK HERE]`  

---

## 1. Executive Summary & Elevator Pitch

### Project Title
**PulseForge AI** — Self-Validating Edge Acoustic Anomaly Detection for Industrial Equipment

### One-Line Pitch (Tagline)
*An intelligent acoustic predictive maintenance platform that detects mechanical degradation before failure while actively self-validating physical sensor health and ambient domain shift.*

### Short Summary (100–150 words)
PulseForge AI is an edge-first industrial AI platform designed to eliminate catastrophic downtime in industrial rotating equipment (focusing on industrial centrifugal fans). Unlike conventional acoustic solutions that suffer from high false-alarm rates when factory noise fluctuates, PulseForge AI combines a deep PyTorch Autoencoder with an active signal-quality validator and a latent Mahalanobis familiarity space. 

Operating under a strict 4-state deterministic decision policy, the system alerts technicians to genuine bearing anomalies, identifies sensor hardware failures (cable disconnections, clipping), and detects ambient acoustic domain shifts. The solution features an offline Streamlit operator console, an automated SQLite audit trail, 129 passing unit tests, and a planned Phase 2 expansion incorporating a local Ollama AI copilot and an enterprise Model Context Protocol (MCP) server.

---

## 2. Problem Statement (>200 words)

In continuous process manufacturing plants—such as chemical refineries, power generation facilities, and automotive assembly lines—unplanned machine breakdowns cost industrial enterprises over **$50 billion annually** in lost productivity, emergency repairs, and secondary collateral damage.

Industrial centrifugal fans are mission-critical assets responsible for furnace ventilation, toxic gas exhaust, and cooling. When fan bearings degrade or blades suffer aerodynamic imbalance, early failure warnings appear first in high-frequency acoustic sound signatures—often days or weeks before gross physical vibration triggers conventional accelerometers.

However, existing acoustic monitoring solutions face three severe operational roadblocks that prevent plant-floor adoption:
1. **The Ambient Noise & Domain Shift Dilemma:** Factory acoustic environments are dynamic. When adjacent forklifts, compressors, or tooling start up, standard anomaly detectors falsely flag these normal background changes as critical machine faults, destroying operator trust.
2. **Blind Failure on Sensor Malfunctions:** When acoustic sensors become physically disconnected, suffer cable faults, or experience ADC clipping due to improper gain calibration, black-box neural networks hallucinate anomalous failure predictions instead of diagnosing the broken sensor.
3. **Irresponsible "Black Box" Probability Outputs:** Presenting technicians with uncalibrated "95% probability of bearing failure" or predicting arbitrary failure dates without physical evidence causes alert fatigue and regulatory non-compliance.

Factories urgently require an acoustic intelligence platform that is **accurate, explainable, self-validating, and deterministic in its safety recommendations**.

---

## 3. Proposed Solution & Innovation (>250 words)

PulseForge AI solves these fundamental problems through a layered, self-validating edge acoustic architecture:

### A. Physics-First Signal Quality Gating
Before any neural network computation occurs, the system's signal-processing engine evaluates raw 16 kHz acoustic frames against physical hardware integrity criteria:
- **Zero-Energy / Silence Detection:** Catches loose wiring, unplugged microphones, or broken preamplifiers (RMS < 0.0010).
- **ADC Saturation & Digital Clipping Check:** Identifies acoustic sensor gain overflow (> 1.0% samples at full dynamic scale).
- **Duration and Finite Value Guardrails:** Guarantees strict input validity and prevents pipeline NaN crashes.

### B. Deep Unsupervised Acoustic Autoencoder
To bypass the industrial scarcity of labelled failure data, PulseForge AI utilizes a deep symmetrical Autoencoder implemented in PyTorch. Trained exclusively on baseline normal recordings:
- Raw audio is converted to 128-bin Log-Mel spectrograms with temporal frame concatenation (5-frame context $\to$ 200 input dimensions).
- The network compresses representations through decreasing dense layers ($200 \to 128 \to 64 \to 32$) into a compact **16-dimensional bottleneck latent space**.
- Healthy operating sounds reconstruct with minimal loss, while abnormal bearing rattle and mechanical friction generate prominent reconstruction errors that exceed our validation-calibrated threshold ($0.8904$).

### C. Latent Space Familiarity & Domain Shift Detection
To solve the factory background noise dilemma, PulseForge AI fits an empirical Mahalanobis covariance model over the 16-dimensional latent representations of normal training data. When ambient factory noise shifts, the latent vector's distance spikes beyond our Tukey IQR threshold ($9.2549$). Instead of falsely reporting a machine breakdown, PulseForge AI flags an unfamiliar acoustic environment.

### D. 4-State Industrial Decision Policy
Decisions are mapped into four unambiguous, ISO 13374-1 aligned operational states:
1. **`NORMAL`:** Acoustic signature verified below anomaly threshold with high signal quality.
2. **`POSSIBLE_ANOMALY`:** Signal quality verified; acoustic reconstruction error exceeds threshold. Actionable physical inspection recommended.
3. **`NEEDS_REVIEW`:** Sensor quality degraded or acoustic domain shift detected. Flags human review without crying wolf.
4. **`INVALID_INPUT`:** Corrupted file or invalid technical header.

---

## 4. Technical Architecture & Stack

| Layer | Component | Technologies Used |
| :--- | :--- | :--- |
| **Edge Audio I/O** | Safe WAV decoding, header validation | `soundfile`, `librosa`, `numpy` |
| **Feature Extraction** | Log-Mel filterbanks, temporal framing, scaling | `librosa.feature`, `scikit-learn` |
| **Primary Detector** | Deep Acoustic Autoencoder (16-latent bottleneck) | `PyTorch 2.0+` (Optimized for CPU edge inference) |
| **Validation Baseline** | Ensemble Isolation Forest (50 trees) | `scikit-learn.ensemble` |
| **Domain Shift** | Latent Mahalanobis familiarity estimator | `scipy.spatial.distance`, `joblib` |
| **Decision Policy** | 4-state deterministic state machine | Python 3.12, strict Pydantic/dataclass schemas |
| **Audit Persistence** | Zero-audio SQLite metadata trail | `sqlite3` (`analysis_runs` schema) |
| **Operator Interface** | Interactive Plant Console with 1-click demos | `Streamlit 1.67+`, `matplotlib` |
| **Testing & CI** | Comprehensive regression test suite | `pytest` (**129 passed, 0 failed, 100% pass rate**) |

---

## 5. Dataset Attribution & Validation Methodology

### Dataset
- **MIMII Dataset:** *Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection* (Hitachi Ltd. / IEEE DCASE).
- **Target Machine:** Industrial Centrifugal Fan, Model ID `id_00`.
- **Operating Conditions:** 16 kHz sampling rate, 16-bit PCM, multi-channel acoustic array averaged to robust single-channel telemetry.

### Train / Validation / Test Discipline
- **Train Set:** 24 pure normal operating clips (strictly zero anomaly leakage).
- **Validation Set:** 5 normal clips and 5 abnormal clips used strictly to calibrate decision thresholds (Autoencoder 95th percentile = $0.8904$; Familiarity Tukey IQR = $9.2549$).
- **Test Set:** 20 held-out clips used exclusively for final unbiased evaluation.
- **Reproducible Manifest:** Every single recording is hashed and indexed in `data/manifests/fan_id_00_manifest.csv`.

---

## 6. CloudCamp BD Rubrics Alignment & Pre-Score Auto-Evaluation

| Rubric Area | Required Standard | PulseForge AI Implementation & Evidence |
| :--- | :--- | :--- |
| **Core Problem Relevance** | Predictive Plant track / Industrial efficiency | Solves catastrophic machine downtime via acoustic predictive maintenance. |
| **Working Software Slice** | Functional demo without manual notebook runs | Full end-to-end Streamlit app with 5 preloaded scenarios and live audio upload. |
| **ML Rigor & Baseline** | Legitimate model + comparative baseline | PyTorch Deep Autoencoder compared against an Isolation Forest baseline. |
| **Reliability & Self-Validation** | Signal quality & domain shift handling | Active checks for silence, clipping, DC offset, and latent Mahalanobis distance. |
| **Engineering Quality** | Modular code, comprehensive tests | 129 automated unit tests passing (100% pass rate), zero hardcoded mock predictions. |
| **CloudCamp Ecosystem** | MCP & Event integration | Connected CloudCamp SSE MCP server; full knowledge base ingested; Phase 2 custom MCP mapped. |
| **Responsible AI** | Honest communication, no hallucinations | Zero fabricated failure probabilities or uncalibrated failure dates; ISO-compliant language. |

---

## 7. Roadmap: Final Round Expansion (Phase 2 / Milestones M8–M13)

1. **Factory Brain Local AI Copilot (Ollama + RAG):**
   - Run a fully local, zero-subscription quantized LLM (`deepseek-r1:8b` or `qwen2.5:7b` via Ollama) to translate raw acoustic metrics and sensor logs into natural language maintenance work-orders.
   - Grounded with RAG over OEM industrial fan service manuals.
2. **Dedicated PulseForge MCP Server:**
   - Expose tools (`inspect_fan_acoustic_signature`, `get_equipment_telemetry`, `get_sensor_audit_history`) conforming to the Model Context Protocol (MCP), enabling Antigravity and external enterprise agents to autonomously query plant health.
3. **Dual Frontend (Lovable React Dashboard):**
   - Deploy a high-throughput, multi-asset React frontend for plant operations teams, complementing the edge Streamlit console.
4. **Cloud Deployment & Secure Tunneling:**
   - Expose the live inference endpoints and dashboard via secure tunneling (Cloudflare Tunnel / ngrok) for live remote evaluation during the final congress.
