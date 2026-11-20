# PulseForge AI — Complete Project Master Guide
**Everything you need to know about PulseForge AI: concepts, implemented features, and future roadmap.**

---

## 1. Executive Summary & Core Mission

**PulseForge AI** is an industrial acoustic intelligence and autonomous maintenance platform designed for predictive plant operations. 

- **Competition:** CloudCamp BD "International AI Builders Congress 2026"
- **Domain:** IndustrySphere AI
- **Primary Sub-Track:** **Predictive Plant** (Self-validating edge acoustic anomaly detection)
- **Supporting Sub-Track:** **Factory Brain** (Autonomous LLM maintenance copilot grounded in industrial SOPs)
- **Deployment URL:** `https://pulseforge-ai.streamlit.app`
- **Source Code Repository:** `https://github.com/tamzid1103/Pulse-Forge-Ai`

### The Core Problem
Unplanned mechanical breakdowns cost global manufacturing over **$50 billion annually**. Industrial ventilation fans, blowers, and pumps run 24/7 in harsh plant environments. When an internal bearing seizes or an impeller unbalances, it triggers catastrophic line shutdowns.

### The Innovation
Acoustic sound signatures contain the earliest physical indicators of micro-surface degradation—detectable **2 to 6 weeks before** traditional vibration sensors or thermal cameras register a fault. 

PulseForge AI solves the fatal flaws of existing industrial AI tools:
1. **No Hallucinations:** Never predicts fabricated failure dates or uncalibrated failure probabilities.
2. **Signal Quality Self-Validation:** Detects dead microphones, clipping, or digital saturation before AI inference.
3. **Domain Shift Awareness:** Measures latent space distance to detect unfamiliar factory noise rather than triggering false alarms.
4. **Autonomous Factory Brain Copilot:** Translates acoustic telemetry into OSHA-compliant maintenance work orders using **Groq LPU (Llama 3.3)**.

---

## 2. End-to-End Technical Pipeline (How the AI Works)

PulseForge AI operates a 6-stage deterministic edge pipeline adhering directly to the international standard **ISO 13374** (Condition Monitoring and Diagnostics of Machine Systems):

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Raw Audio In   │ ──> │ Signal Quality Gate  │ ──> │ Log-Mel Spectrogram    │
│  (16 kHz WAV)   │     │ (Silence/Clip/RMS)   │     │ (128-band, STFT 1024)  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Factory Brain   │ <── │ 4-State State Machine│ <── │ PyTorch Autoencoder    │
│ Copilot (Groq)  │     │ (Decision Policy)    │     │ & Latent Familiarity   │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

### Stage 1: Audio Ingestion & Decoding (`src/audio_io.py`)
- Accepts standard uncompressed 16-bit PCM WAV audio.
- Standardizes sampling rate to **16,000 Hz (16 kHz)**.
- Automatically downmixes multi-channel or stereo audio to a single mono channel.
- Enforces strict boundary limits: duration must be between **0.5s and 30.0s** (10 seconds nominal), and file size must not exceed 25 MB.

### Stage 2: Signal Quality Gate (`src/quality.py`)
Physical microphones on factory floors get damaged, disconnected, or covered in oil mist. Before running expensive neural networks, PulseForge verifies signal integrity:
- **Silence Detection:** Checks if RMS energy is below `0.001` (detects unplugged or broken microphones).
- **Clipping Detection:** Checks if audio samples saturate at $\pm 1.0$ (detects digital gain overflow).
- **DC Bias / Offset:** Checks for electrical ground loop leakage.
- *Outcome:* If quality fails, the system immediately flags `NEEDS_REVIEW` and directs the technician to inspect the sensor hardware instead of condemning the fan.

### Stage 3: Acoustic Feature Extraction (`src/features.py`)
- Computes Short-Time Fourier Transforms (STFT) with a frame size of `1024` samples (64 ms) and hop length of `512` samples (32 ms).
- Maps raw audio frequencies into a **128-band Log-Mel Spectrogram**.
- Mel filterbanks mirror human and physical acoustic perception, isolating high-frequency harmonic clicks (bearing wear) and low-frequency aerodynamic rumble (blade unbalance).

### Stage 4: Deep Autoencoder Anomaly Detection (`src/autoencoder.py`)
- **Unsupervised Anomaly Paradigm:** In real industrial plants, machines run healthy 99% of the time; genuine failure recordings are extremely rare.
- **Architecture:** Symmetric 4-layer fully connected deep Autoencoder with a compressed **16-dimensional latent bottleneck**.
- **Training Strategy:** Trained purely on **normal, healthy fan sounds** from the MIMII industrial dataset.
- **Inference:** The Autoencoder attempts to compress and reconstruct the sound. Because it only understands healthy patterns, when anomalous acoustic patterns (bearing spalling, blade rattling) occur, the network cannot reconstruct them cleanly.
- **Metric:** The Mean Squared Error (MSE) between the input and reconstructed spectrogram is the **Reconstruction Error (Anomaly Score)**.
- **Calibrated Threshold:** Derived empirically from validation data (`0.0032` MSE).

### Stage 5: Domain Familiarity & Latent Space Distance (`src/familiarity.py`)
- Background noise in plants fluctuates (e.g. overhead cranes, forklifts, shift changes).
- PulseForge measures the **Mahalanobis Distance** of the audio's embedding inside the Autoencoder's 16-dimensional bottleneck space.
- If the sound deviates from known operational regimes (distance > `9.25`), the system flags an **Acoustic Domain Shift**.

### Stage 6: Deterministic 4-State Decision Policy (`src/decision.py`)
The pipeline outputs one of 4 unambiguous operational states:
1. **`NORMAL` (Green):** Signal quality passed, latent space is familiar, and anomaly score is below threshold (`< 0.0032`). Machine is healthy.
2. **`POSSIBLE_ANOMALY` (Red):** Signal quality passed, latent space is familiar (authentic fan noise), but anomaly score exceeds threshold (`≥ 0.0032`). Developing mechanical defect.
3. **`NEEDS_REVIEW` (Amber):** Sensor quality degraded (silent/clipped) OR acoustic domain shift detected. Automatic decision is safely withheld to prevent false alarms.
4. **`INVALID_INPUT` (Gray):** File corrupted or out of duration bounds.

---

## 3. Factory Brain: Industrial AI Copilot (`src/copilot.py`)

Milestone M8 integrates an autonomous reliability engineering layer fulfilling the **Factory Brain** sub-track:

```
[ Diagnostic Telemetry ] + [ Industrial SOP & ISO 10816 ] ──> [ Groq LPU (Llama 3.3 70B) ]
                                                                        │
                                       ┌────────────────────────────────┴────────────────────────────────┐
                                       ▼                                                                 ▼
                         [ 1-Click Digital Work Order ]                                    [ Interactive Technician Chat ]
                         - Work Order ID (WO-FAN-2026-XXXX)                                - Explains acoustic signatures
                         - Urgency: PRIORITY_INSPECTION                                    - Advises bearing inspection tools
                         - ISO 10816-3 Zone C Severity                                     - Answers plant engineering Q&A
                         - Mandatory OSHA LOTO Checklist                                   - 300+ tokens/sec live streaming
                         - Required Tools & PPE
```

### Key Capabilities:
- **Groq LPU Engine:** Runs open-weights `llama-3.3-70b-versatile` (or `llama-3.1-8b-instant`) at ultra-high speeds (300–500 tokens/sec) for free.
- **Contextual Grounding:** Reads industrial SOPs directly from `data/knowledge_base/fan_maintenance_sop.md` and `data/knowledge_base/iso_10816_acoustic_standards.md`.
- **1-Click Digital Work Orders:** Automatically generates a complete mechanical maintenance ticket containing:
  - Work Order ID, Asset Tag (`fan_id_00`), Diagnostic Status
  - Severity Classification (ISO 10816-3 Zone A, B, C, or D)
  - Suspected mechanical failure mode (e.g. Bearing raceway spalling vs. V-belt slippage)
  - Mandatory OSHA 1910.147 **Lockout/Tagout (LOTO)** safety procedure
  - Required tools (Fluke IR thermometer, ISO vibration pen, dial indicator)
  - Required PPE (safety glasses, hearing protection, cut-resistant gloves)
  - Downloadable as formatted JSON for Computerized Maintenance Management Systems (CMMS).
- **Zero-Crash Offline Fallback:** If no API key is provided or the plant is offline, the system automatically uses a deterministic rule-based SOP engine matching the exact status.

---

## 4. All Implemented Dashboard Views (`app/dashboard.py`)

The application features 5 specialized operational modules accessible via the dark navy sidebar:

### 1. Acoustic Diagnostic Studio (Primary Workshop)
- **Guided 3-Stage Workflow:**
  - **Stage 1 (Select Recording):** Toggle between uploading custom 16-bit WAV files or testing 5 pre-calibrated MIMII scenarios.
  - **Stage 2 (Review Audio):** In-browser audio player with duration, sample rate, channel count, and file size metadata.
  - **Stage 3 (Run Diagnostics):** 1-Click CPU analysis button executing inference in under 150 ms.
- **Visual Evidence:** High-contrast decision card, 4 metric tiles (Reconstruction Error, Latent Familiarity, Peak Segment, Signal Quality), and dual-panel Waveform & Log-Mel Spectrogram plots.
- **Factory Brain Copilot:** Embedded tabs for 1-click Work Order generation and interactive chat with quick-prompt chips.

### 2. Plant Fleet Overview
- Comprehensive condition monitoring across multiple industrial ventilation lines (Lines 1 to 4).
- Real-time asset health registry, line status badges, and aggregate plant reliability KPIs.

### 3. Model Benchmarks & Validation
- Real, reproducible empirical evaluation data loaded from `results/evaluation_report.json`.
- Displays real ROC curve with **0.9375 AUROC** (outperforming the 0.9167 Isolation Forest baseline) and **0.9333 F1-Score**.
- Training loss curves and threshold derivation documentation.

### 4. Inspection History & Audit
- Embedded SQLite metadata history (`data/analysis_history.db`).
- Records timestamp, audio filename, anomaly score, processing latency, and decision status.
- Filter by status and export records directly to CSV.

### 5. System Architecture & ISO Standards
- Visual process flow diagram mapping the pipeline to **ISO 13374-1**.
- Formal dataset attribution for the MIMII dataset (Hitachi Ltd. / Ritsumeikan University, Creative Commons Attribution 4.0).

---

## 5. Verified ML Benchmark Results

Evaluated on the official **MIMII Fan Dataset (Model ID: `fan_id_00`)** under realistic industrial background noise (6 dB SNR):

| Model Architecture | Task Paradigm | AUROC Score | F1-Score | Precision | Recall | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PulseForge Deep Autoencoder** | Unsupervised Reconstruction | **0.9375** | **0.9333** | **0.8750** | **1.0000** | **< 150 ms** |
| **Baseline (Isolation Forest)** | Unsupervised Tree Density | 0.9167 | 0.7368 | 0.5833 | 1.0000 | < 120 ms |

> [!NOTE]
> The primary Autoencoder achieves **100% Recall** on mechanical fault recordings, ensuring no anomalous fan goes undetected.

---

## 6. Future Roadmap for Finals (December 26, 2026)

For the upcoming final round, the following advanced modules are planned:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FINALS PRODUCTION ROADMAP                              │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│ Milestone M9             │ Milestone M10            │ Milestone M11             │
│ Custom MCP Server        │ Modern Lovable React UI  │ Edge Hardware & Multi-Fan │
│                          │                          │                           │
│ Model Context Protocol   │ High-performance React / │ Multi-machine expansion   │
│ exposing diagnostic and  │ Vite frontend with rich  │ (pumps, valves, slides)   │
│ quality tools to AI      │ interactive charts and   │ with Raspberry Pi /       │
│ assistants (Claude, AGY) │ live WebSocket streaming │ NVIDIA Jetson edge tests  │
└──────────────────────────┴──────────────────────────┴───────────────────────────┘
```

1. **Milestone M9 — PulseForge Custom Model Context Protocol (MCP) Server (`src/mcp_server.py`):**
   - Implements a standardized MCP server exposing PulseForge tools (`analyze_fan_audio`, `inspect_signal_quality`, `get_machine_health_status`, `generate_maintenance_work_order`).
   - Allows external AI agents (like Claude Desktop, Google Antigravity, and Cursor) to query factory floor machines over Stdio and SSE transports.

2. **Milestone M10 — Lovable Modern React Industrial Dashboard:**
   - Full-fledged enterprise web client built on modern React, Vite, and Tailwind CSS.
   - Interactive zoomable spectrograms, real-time WebSocket audio streaming, and dark-mode industrial aesthetics.

3. **Milestone M11 — Multi-Machine Industrial Fleet & Edge Deployment:**
   - Extend beyond ventilation fans to industrial pumps, valves, and slide rails.
   - Package lightweight ONNX / TensorRT runtime for edge deployment on low-power hardware (Raspberry Pi 5 / NVIDIA Jetson Orin Nano).

---

## 7. How to Explain PulseForge AI to Judges (Cheat Sheet)

| Question | Winning Response |
| :--- | :--- |
| **"What makes PulseForge AI different from other audio classifiers?"** | *"Most models use supervised classifiers that require thousands of labelled failure sounds, which don't exist in real factories. PulseForge uses an unsupervised Autoencoder trained only on normal sounds, combined with signal quality gating and latent familiarity so it never hallucinates or triggers false alarms when plant noise shifts."* |
| **"Why is the Groq LPU integration a game changer?"** | *"It connects deterministic signal processing directly to executive maintenance action. Instead of giving an operator a confusing graph, Groq's Llama 3.3 engine instantly synthesizes a digital work order with OSHA Lockout/Tagout checklists, tool recommendations, and ISO 10816 classifications at over 300 tokens per second."* |
| **"Can this run on edge hardware?"** | *"Yes. The entire feature extraction and Autoencoder inference pipeline executes in under 150 milliseconds on a standard CPU with zero GPU requirement, making it ideal for edge industrial gateways."* |
| **"What dataset did you train on?"** | *"We trained on the MIMII benchmark industrial dataset published by Hitachi and Ritsumeikan University, specifically focusing on centrifugal ventilation fans recorded under realistic plant operational conditions."* |
