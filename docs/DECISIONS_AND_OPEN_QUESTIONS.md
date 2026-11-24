# Decisions, Assumptions, and Open Questions

## 1. Confirmed project decisions from the discussion

| Topic | Decision | Status |
|---|---|---|
| Project title | PulseForge AI | Confirmed in prior project recommendation |
| Tagline | Hear the Warning. Trust the Signal. Prevent the Downtime. | Confirmed in prior project recommendation |
| Domain | IndustrySphere AI | Confirmed |
| Primary sub-track | Predictive Plant | Confirmed |
| Supporting sub-track | Factory Brain | Confirmed as supporting / dashboard direction |
| Optional extension | Green Manufacturing | Optional future direction only |
| Core problem | Detect unusual machine behavior early from sound / vibration recordings | Confirmed concept |
| Initial model | Autoencoder anomaly detector | Recommended technical choice |
| Baseline | Isolation Forest | Recommended comparison |
| Main dataset | MIMII | Confirmed first dataset recommendation |
| First machine for implementation | Fan | Selected in the step-by-step implementation plan as the manageable first machine |
| Input mode | Uploaded audio recording for MVP | Confirmed implementation approach |
| Main UI | Streamlit | Recommended for hackathon speed |
| Backend | Python service functions first; FastAPI optional later | Scope resolution for implementation speed |
| Database | Not required for MVP; SQLite metadata history optional | Recommended to avoid overengineering |
| Authentication | Not required for local single-operator MVP | Recommended scope decision |
| Main innovation | Anomaly detection with explicit input-quality and uncertainty / unfamiliarity communication | Confirmed concept |
| Human control | Human reviews alerts; no automatic machine shutdown | Confirmed safety boundary |
| Build target | 48-hour hackathon plan | Planning target, not confirmed event rule |

## 2. Assumptions used to make the handover implementable

These are practical defaults and should be revisited if the team or event constraints require it.

1. A local laptop / workstation is available.
2. Python 3.11 is an acceptable starting runtime, subject to dependency compatibility.
3. WAV is the only upload format in the MVP.
4. Fan is the only enabled machine type in the MVP.
5. Initial analysis uses a manageable MIMII fan subset, not the complete dataset.
6. Streamlit calls the model pipeline in-process to reduce integration time.
7. The first version does not require accounts, cloud storage, external APIs, or hardware sensors.
8. The app processes uploaded audio locally and does not persist raw audio by default.
9. Model training and inference settings are configurable and saved with the checkpoint.
10. A model result is a screening signal for human review, not a certified diagnosis.

## 3. Scope resolution: earlier breadth vs. practical MVP

Earlier project explanations discussed motors, pumps, and fans as target machine families. The implementation plan later recommended beginning with a fan to keep the 48-hour scope realistic. This package resolves that by making **fan-only the MVP**. Pump and motor support remain future extensions and require separate data/model validation.

Earlier architecture discussions mentioned FastAPI and a database. The later implementation guidance recommended Streamlit direct-to-service first and optional FastAPI / SQLite after the core pipeline works. This package preserves the technologies as planned options but does not make them MVP dependencies.

The earlier dashboard sketch showed example status and numerical values for illustration. Those values are **not real model results** and must never be copied into the product as actual predictions.

## 4. Questions requiring project-owner confirmation & their resolution status

| Question | Why it matters | Initial Default | Resolution / Actual Outcome | Status |
|---|---|---|---|---|
| What is the actual team size and member availability? | Work allocation | 5-role split | Team "Asgardians" (Tamzidul Haque, Puja Chowdhury, MD. Adnan Islam Srabon) | Resolved |
| Which operating system will be used for the final demo? | Setup / audio dependencies | Linux or Windows | Linux (Ubuntu) + Streamlit Community Cloud (Debian Linux container) | Resolved |
| Is a GPU available? | Training time | CPU-compatible | PyTorch CPU-optimized training & inference (<150 ms) | Resolved |
| How much disk space and bandwidth are available? | MIMII archive size | Download subset | HTTP Range requests fetched targeted 50 files directly from Zenodo (~125 MB) | Resolved |
| Which exact MIMII noise condition / ID should be used? | Reproducible split | Fan id_00 | MIMII Fan `id_00` at 6 dB SNR factory ambient noise condition | Resolved |
| What upload duration / size limits are appropriate? | Runtime and resource safety | Conservative limits | Max upload 25 MB; 1.0s to 30.0s audio duration; 16 kHz mono downmixing | Resolved |
| What is the final deployment target? | Local demo vs hosted | Local Streamlit demo | Public hosted cloud: `https://pulse-forge-ai.streamlit.app/` + presentation site | Resolved |
| Does the event require a public URL or only local demo? | Hosting scope | Local demo | Public live URL required on submission portal; deployed on Streamlit Cloud | Resolved |
| Are there specific event rules for libraries, models, etc.? | Compliance | Follow rules | CloudCamp BD AI Builders Congress rubrics fully addressed | Resolved |
| Should the demo UI and pitch be English, Bangla, or both? | User experience | English UI | Professional English industrial UI with bilingual pitch readiness | Resolved |
| Is SQLite history desired? | Optional feature | Optional | Implemented in `src/persistence.py` with CSV export in dashboard | Resolved |
| Is a separate FastAPI service required? | Integration boundary | Service-first | Service-first for Phase 1; FastAPI REST & WebSocket scheduled for M10 | Resolved |

## 5. Decisions the coding agent may make without approval

- Exact Autoencoder layer dimensions and training hyperparameters, provided they are logged and chosen using training/validation data.
- Exact feature parameters (Mel bins, hop length, segment duration) based on inspection and experiments, provided training/inference remain consistent.
- Exact threshold rule, provided validation-based selection is explained and test data is untouched.
- Exact UI layout within the supplied UX principles.
- Internal class/function names and test fixture design.
- Whether to use a dense or convolutional Autoencoder, based on feature shape and practical performance.

## 6. Decisions requiring explicit approval before changing scope

- Supporting pumps/motors in MVP.
- Adding real hardware or streaming sensor input.
- Introducing accounts or public network access.
- Storing raw uploaded audio.
- Adding an LLM or external AI API to make fault decisions.
- Claiming exact component diagnosis or remaining useful life.
- Switching the primary dataset away from MIMII.
- Replacing the Autoencoder as the main approach without documenting why.
- Adding cloud infrastructure, a separate frontend, PostgreSQL, or a microservice architecture.

## 7. Decision log template

```text
Decision ID:
Date:
Question / change:
Options considered:
Chosen option:
Reason / evidence:
Impact on scope:
Documents updated:
Approved by:
```

## 8. Recorded Decisions

### Decision DEC-001: Python 3.12 Runtime
- **Date:** 2026-11-17
- **Question / change:** Host environment has Python 3.12.3 installed; Python 3.11 is not available.
- **Options considered:** (A) Attempt custom build/install of Python 3.11, (B) Use system Python 3.12.3.
- **Chosen option:** Option B (Python 3.12.3).
- **Reason / evidence:** Python 3.12.3 is officially supported by PyTorch 2.15+cpu, Librosa 1.0.1, Scikit-learn 1.9.1, and Streamlit 1.67. All 19 smoke tests passed without issue.
- **Impact on scope:** None. Zero architectural deviation.
- **Documents updated:** `docs/TECH_STACK.md`, `docs/IMPLEMENTATION_PLAN.md`.
- **Approved by:** Antigravity (Implementation Agent).

### Decision DEC-002: MIMII Fan Archive and Subset Acquisition Strategy
- **Date:** 2026-11-17
- **Question / change:** Which MIMII fan archive, noise condition, and acquisition method should be used for the manageable MVP subset?
- **Options considered:** (A) Download the full ~10 GB `6_dB_fan.zip`, (B) Use HTTP Range requests to extract a targeted 45-file subset directly from Zenodo, (C) Synthesize mock audio.
- **Chosen option:** Option B (`6_dB_fan.zip`, machine `id_00`, 35 normal + 15 abnormal = 50 genuine files extracted via range requests).
- **Reason / evidence:** Range requests allow fetching authentic 16 kHz 8-channel 10s recordings directly from the official Zenodo archive in ~2 minutes without downloading the full 10 GB. 50 genuine files provide a complete train/val/test split.
- **Impact on scope:** Fully adheres to MVP rules; real industrial data, manageable size (~125 MB on disk).
- **Documents updated:** `docs/DATASET_AND_MODEL_GUIDE.md`, `docs/IMPLEMENTATION_PLAN.md`.
- **Approved by:** Antigravity (Implementation Agent).

### Decision DEC-003: Baseline Isolation Forest Feature Representation and Thresholding Policy
- **Date:** 2026-11-18
- **Question / change:** What feature representation, score orientation, and threshold selection method should be used for the M3 Isolation Forest baseline?
- **Options considered:** (A) Flatten raw 4096-dim Mel frames, (B) Pool mean and std per Mel band across time frames ($2 \times 128 = 256$ dimensions per segment), (C) Single 128-dim mean vector per recording.
- **Chosen option:** Option B (256-dimensional `mean_std` pooling across time frames per segment, mean score aggregation across segments, and quantile threshold derived on validation normal data at target FPR=0.10).
- **Reason / evidence:** Inverted sign ensures higher score indicates higher anomaly (matching the upcoming Autoencoder convention). On the held-out test split, the baseline achieved 0.9167 AUROC, 0.9583 PR-AUC, and 0.875 Recall, establishing a solid, reproducible benchmark.
- **Impact on scope:** Establishes the required benchmark model for M3 without any data leakage.
- **Documents updated:** `docs/DATASET_AND_MODEL_GUIDE.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`, `src/config.py`.
- **Approved by:** Antigravity (Implementation Agent).

### Decision DEC-004: Primary Autoencoder Architecture, Training Strategy, and Comparative Benchmark
- **Date:** 2026-11-18
- **Question / change:** What PyTorch Autoencoder architecture, loss, and training strategy should be implemented for Milestone M4?
- **Options considered:** (A) Conv2D Autoencoder, (B) Fully-connected symmetric dense Autoencoder (4096 -> 512 -> 128 -> 64 -> 32 -> 64 -> 128 -> 512 -> 4096) with BatchNorm and LeakyReLU, (C) Linear PCA / shallow projection.
- **Chosen option:** Option B (Fully-connected symmetric dense Autoencoder with MSE loss, Adam optimizer with weight decay $1e-5$, and early stopping on validation loss).
- **Reason / evidence:** The symmetric dense Autoencoder converges smoothly in 8 seconds on CPU, achieving 0.9375 AUROC and 0.8571 F1-Score on the held-out test split, significantly outperforming the Isolation Forest baseline (F1 +0.1203, with 0 false positives). The 32-dimensional latent bottleneck also provides compact embeddings for M5 unfamiliarity checks.
- **Impact on scope:** Completes the core primary ML model for the hackathon MVP.
- **Documents updated:** `docs/DATASET_AND_MODEL_GUIDE.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`, `src/config.py`.
- **Approved by:** Antigravity (Implementation Agent).


### Decision DEC-005: Signal Quality, Latent Embedding Familiarity, and 4-State Industrial Decision Policy
- **Date:** 2026-11-19
- **Question / change:** How should audio signal quality, latent domain familiarity, and model anomaly scores be combined to produce trustworthy, actionable industrial recommendations without making unfounded failure probability claims?
- **Options considered:**
  - (A) Single binary threshold on raw anomaly score, ignoring signal quality and out-of-domain inputs.
  - (B) Soft heuristic probability estimate ("95% chance of bearing failure").
  - (C) Deterministic multi-stage policy: Pre-inference Signal Quality verification (silence, clipping, duration, NaN) -> Latent Standardized Euclidean Familiarity assessment against normal training distribution -> Model Anomaly Detection -> 4 Explicit States (`normal`, `possible_anomaly`, `needs_review`, `invalid_input`).
- **Chosen option:** Option C.
- **Reason / evidence:** Satisfies all requirements of `AGENTS.md` and `docs/IMPLEMENTATION_PLAN.md` (M5). Prevents silent failures on silent, digitally clipped, or domain-shifted audio by routing them to `needs_review`. Tested on 112 unit and integration tests with 100% pass rate.
- **Impact on scope:** Establishes the core differentiator for PulseForge AI: trustworthy industrial screening with explicit quality/unfamiliarity communication.
- **Documents updated:** `src/schemas.py`, `src/quality.py`, `src/familiarity.py`, `src/thresholding.py`, `src/decision.py`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`, `docs/IMPLEMENTATION_PLAN.md`.
- **Approved by:** Tamzidul & Antigravity (Implementation Team).

### Decision DEC-006: CloudCamp BD 2026 Hackathon Strategy Alignment & Architecture Expansion
- **Date:** 2026-11-19
- **Question / change:** How should the project architecture expand beyond the baseline MVP to maximize score on the official CloudCamp BD International AI Builders Congress 2026 Pre-Judgment Auto-Scoring Rubric without compromising ML acoustic integrity?
- **Options considered:**
  - (A) Strict minimal MVP: Keep Streamlit-only, no database, no API, no MCP server, no LLM copilot (risks losing 60+ pre-judgment rubric points).
  - (B) Replace ML pipeline with an LLM: Use an LLM for acoustic diagnosis (violates physical acoustic principles and AGENTS.md rules).
  - (C) Layered Architecture Expansion: Preserve the core acoustic ML pipeline (Autoencoder + Baseline + Quality + Familiarity + Decision Engine) as the deterministic **Predictive Plant** engine. Build on top of it:
    1. Reusable Service Layer (`src/service.py`) and SQLite audit persistence (`src/persistence.py`).
    2. FastAPI adapter (`src/api.py`) powering dual frontends: local Streamlit (`app/dashboard.py`) and modern React dashboard via Lovable MCP.
    3. Custom PulseForge MCP Server (`src/mcp_server.py`) exposing acoustic diagnosis tools (capturing 12 MCP points).
    4. Factory Brain Copilot (`src/copilot.py`) utilizing semantic RAG on fan maintenance SOPs and local Ollama execution (capturing 26+ points for RAG & Local Models).
    5. Cloudflare Tunneling (`scripts/run_tunnel.sh`) for live internet demo access (capturing 6 points).
    6. Dedicated Submission Dossier & Zero to Launch Single Master Prompt (`docs/CLOUDCAMP_SUBMISSION_DOSSIER.md` & `docs/ZERO_TO_LAUNCH_MASTER_PROMPT.md`).
- **Chosen option:** Option C.
- **Reason / evidence:** Directly addresses every scoring category in the CloudCamp knowledge base (`cloudcamp://knowledge-base`) while preserving complete scientific integrity, determinism, and reproducibility of the fan acoustic anomaly detection model.
- **Impact on scope:** Expands Milestones M6–M10 with structured, modular components.
- **Documents updated:** `docs/IMPLEMENTATION_PLAN.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/TECH_STACK.md`, `docs/API_DOCUMENTATION.md`, `docs/DATABASE_SCHEMA.md`.
- **Approved by:** Tamzidul & Antigravity (Implementation Team).

### Decision DEC-008: Model Weights Bundling and Streamlit Community Cloud Live Deployment
- **Date:** 2026-11-19
- **Question / change:** How should the working application be hosted for preliminary evaluation, public demonstration, and remote judging without requiring judges to set up a local Python environment?
- **Options considered:**
  - (A) Require judges to clone repo and install virtual environment locally.
  - (B) Deploy full AWS/GCP Kubernetes stack (expensive and complex for hackathon timeline).
  - (C) Streamlit Community Cloud with tracked model weights in `model_artifacts/` and OS audio packages in `packages.txt`.
- **Chosen option:** Option C (`packages.txt` containing `libsndfile1` & `ffmpeg`, tracked PyTorch/Scikit-learn model weights, hosted at `https://pulse-forge-ai.streamlit.app/`).
- **Reason / evidence:** Provides an instant, zero-setup, highly available public demonstration for judges and portal submission. The CPU container executes PyTorch autoencoder inference in <150 ms with zero external server dependencies.
- **Impact on scope:** Exceeded initial local-only MVP scope early to secure maximum submission rubric points.
- **Documents updated:** `README.md`, `docs/DEMO_GUIDE.md`, `packages.txt`, `.gitignore`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-009: Factory Brain Copilot with Groq LPU and Deterministic SOP Fallback
- **Date:** 2026-11-20
- **Question / change:** How should the Factory Brain AI Copilot be implemented so that it functions both on the hosted cloud URL (where local Ollama cannot run) and in completely offline/air-gapped edge environments?
- **Options considered:**
  - (A) Ollama only (fails on Streamlit Community Cloud container).
  - (B) OpenAI API only (requires paid credits and crashes when API key is missing).
  - (C) Hybrid Architecture: Groq LPU API (`llama-3.1-8b-instant`, `llama3-8b-8192`, `llama-3.3-70b-versatile`) as high-speed cloud engine (500+ tok/s), paired with a deterministic, zero-crash offline SOP fallback grounded in `data/knowledge_base/fan_maintenance_sop.md`.
- **Chosen option:** Option C.
- **Reason / evidence:** Guarantees that the app never crashes when an API key is absent. When an API key is supplied (via environment, secrets, or manual UI input), operators enjoy blazing-fast Llama 3.1/3.3 inference; when offline or without a key, the system still outputs formatted, actionable ISO 10816 / OSHA LOTO maintenance tickets. Tested across 9 unit tests.
- **Impact on scope:** Completes Milestone M8 early with 138/138 passing unit tests.
- **Documents updated:** `src/copilot.py`, `tests/test_copilot.py`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/TECH_STACK.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-010: Executive Light Industrial HTML Presentation Website
- **Date:** 2026-11-20
- **Question / change:** How should the team present the project opening in the video recording and to executive judges who want a high-level architectural walkthrough before seeing raw telemetry?
- **Options considered:**
  - (A) Start video recording directly in the Streamlit analysis studio.
  - (B) Use standard PowerPoint slides.
  - (C) Build a dedicated, interactive Executive HTML Presentation Website (`presentation/index.html`) hosted on GitHub Pages.
- **Chosen option:** Option C (`presentation/index.html` with vector SVG mechanical machinery blueprints, interactive spectrogram comparisons, and keynote walkthrough).
- **Reason / evidence:** Elevated the presentation quality to professional industrial standards; provides visual mechanical context (motor, bearings, impeller, microphone probe placement) before demonstrating acoustic waveform anomalies.
- **Impact on scope:** Added presentation asset under `presentation/`.
- **Documents updated:** `docs/PROJECT_MASTER_GUIDE.md`, `docs/PRELIMINARY_VIDEO_SCRIPT.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-011: Multi-View Industrial Operations Dashboard Redesign
- **Date:** 2026-11-20
- **Question / change:** How should the Streamlit interface be structured to satisfy both deep acoustic engineering analysis and plant-wide management overview?
- **Options considered:**
  - (A) Single crowded dashboard with vertical scrolling.
  - (B) Multi-page Streamlit app with multiple `.py` files (risks state synchronization issues).
  - (C) Multi-view unified dashboard with clean dark navy sidebar radio navigation across 5 specialized views:
    1. `Acoustic Diagnostic Studio` (3-stage diagnostic workflow: Ingestion, Quality Gate, Inference).
    2. `Plant Fleet Overview` (Multi-line machinery health status board).
    3. `Model Benchmarks & Validation` (Empirical ROC/PR curves and confusion matrix).
    4. `Inspection History & Audit` (SQLite run logs with 1-click CSV export).
    5. `System Architecture & ISO Standards` (ISO 13374-1 diagram, tech stack, dataset citation).
- **Chosen option:** Option C.
- **Reason / evidence:** Created a cohesive, executive-grade industrial software experience that impressed preliminary judges and cleanly separates diagnostic depth from fleet management.
- **Impact on scope:** Redesigned `app/dashboard.py` (1,134 LOC).
- **Documents updated:** `app/dashboard.py`, `docs/PROJECT_MASTER_GUIDE.md`, `docs/UI_UX_GUIDELINES.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-007: Two-Phase Hackathon Execution Strategy (Preliminary Video Sprint vs Final Round Expansion)
- **Date:** 2026-11-19
- **Question / change:** How should the engineering schedule be prioritized given that the Preliminary Submission deadline is tomorrow (Nov 20), requiring a video demonstration of working software, while the final round occurs two weeks later?
- **Options considered:**
  - (A) Attempt full cloud deployment, Ollama setup, and Lovable React build all in one night (high risk of partial implementations or broken recording setup before tomorrow morning).
  - (B) Submit slides only without running software (fails live software demonstration requirement).
  - (C) Two-Phase Staged Sprint:
    - **Phase 1 (Tonight / Tomorrow Morning):** Build and lock the local vertical slice: unified application service (`src/service.py`), SQLite inspection history (`src/persistence.py`), 5 curated demo audio scenarios (`demo_audio/`), and an interactive Streamlit operator UI (`app/dashboard.py`). Deliver an exact 2-minute video script (`docs/PRELIMINARY_VIDEO_SCRIPT.md`) and pre-filled submission dossier (`docs/PRELIMINARY_SUBMISSION_DOSSIER.md`).
    - **Phase 2 (Next 2 Weeks for Finals):** Expand with Factory Brain Copilot (Ollama/RAG), custom MCP server, Lovable React UI, public cloud hosting, and multi-machine fleet extensions (Milestones M8–M13).
- **Chosen option:** Option C.
- **Reason / evidence:** Guarantees an impressive, zero-risk, working demonstration for the preliminary video tomorrow morning, while creating an expanded roadmap for the 2-week final round.
- **Impact on scope:** Establishes Phase 1 (M6–M7) for preliminary submission and Phase 2 (M8–M13) for finals.
- **Documents updated:** `docs/IMPLEMENTATION_PLAN.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`, `docs/DEMO_GUIDE.md`.
- **Approved by:** Tamzidul & Antigravity (Implementation Team).

### Decision DEC-008: Model Weights Bundling and Streamlit Community Cloud Live Deployment
- **Date:** 2026-11-19
- **Question / change:** How should the working application be hosted for preliminary evaluation, public demonstration, and remote judging without requiring judges to set up a local Python environment?
- **Options considered:**
  - (A) Require judges to clone repo and install virtual environment locally.
  - (B) Deploy full AWS/GCP Kubernetes stack (expensive and complex for hackathon timeline).
  - (C) Streamlit Community Cloud with tracked model weights in `model_artifacts/` and OS audio packages in `packages.txt`.
- **Chosen option:** Option C (`packages.txt` containing `libsndfile1` & `ffmpeg`, tracked PyTorch/Scikit-learn model weights, hosted at `https://pulse-forge-ai.streamlit.app/`).
- **Reason / evidence:** Provides an instant, zero-setup, highly available public demonstration for judges and portal submission. The CPU container executes PyTorch autoencoder inference in <150 ms with zero external server dependencies.
- **Impact on scope:** Exceeded initial local-only MVP scope early to secure maximum submission rubric points.
- **Documents updated:** `README.md`, `docs/DEMO_GUIDE.md`, `packages.txt`, `.gitignore`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-009: Factory Brain Copilot with Groq LPU and Deterministic SOP Fallback
- **Date:** 2026-11-20
- **Question / change:** How should the Factory Brain AI Copilot be implemented so that it functions both on the hosted cloud URL (where local Ollama cannot run) and in completely offline/air-gapped edge environments?
- **Options considered:**
  - (A) Ollama only (fails on Streamlit Community Cloud container).
  - (B) OpenAI API only (requires paid credits and crashes when API key is missing).
  - (C) Hybrid Architecture: Groq LPU API (`llama-3.1-8b-instant`, `llama3-8b-8192`, `llama-3.3-70b-versatile`) as high-speed cloud engine (500+ tok/s), paired with a deterministic, zero-crash offline SOP fallback grounded in `data/knowledge_base/fan_maintenance_sop.md`.
- **Chosen option:** Option C.
- **Reason / evidence:** Guarantees that the app never crashes when an API key is absent. When an API key is supplied (via environment, secrets, or manual UI input), operators enjoy blazing-fast Llama 3.1/3.3 inference; when offline or without a key, the system still outputs formatted, actionable ISO 10816 / OSHA LOTO maintenance tickets. Tested across 9 unit tests.
- **Impact on scope:** Completes Milestone M8 early with 138/138 passing unit tests.
- **Documents updated:** `src/copilot.py`, `tests/test_copilot.py`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/TECH_STACK.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-010: Executive Light Industrial HTML Presentation Website
- **Date:** 2026-11-20
- **Question / change:** How should the team present the project opening in the video recording and to executive judges who want a high-level architectural walkthrough before seeing raw telemetry?
- **Options considered:**
  - (A) Start video recording directly in the Streamlit analysis studio.
  - (B) Use standard PowerPoint slides.
  - (C) Build a dedicated, interactive Executive HTML Presentation Website (`presentation/index.html`) hosted on GitHub Pages.
- **Chosen option:** Option C (`presentation/index.html` with vector SVG mechanical machinery blueprints, interactive spectrogram comparisons, and keynote walkthrough).
- **Reason / evidence:** Elevated the presentation quality to professional industrial standards; provides visual mechanical context (motor, bearings, impeller, microphone probe placement) before demonstrating acoustic waveform anomalies.
- **Impact on scope:** Added presentation asset under `presentation/`.
- **Documents updated:** `docs/PROJECT_MASTER_GUIDE.md`, `docs/PRELIMINARY_VIDEO_SCRIPT.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-011: Multi-View Industrial Operations Dashboard Redesign
- **Date:** 2026-11-20
- **Question / change:** How should the Streamlit interface be structured to satisfy both deep acoustic engineering analysis and plant-wide management overview?
- **Options considered:**
  - (A) Single crowded dashboard with vertical scrolling.
  - (B) Multi-page Streamlit app with multiple `.py` files (risks state synchronization issues).
  - (C) Multi-view unified dashboard with clean dark navy sidebar radio navigation across 5 specialized views:
    1. `Acoustic Diagnostic Studio` (3-stage diagnostic workflow: Ingestion, Quality Gate, Inference).
    2. `Plant Fleet Overview` (Multi-line machinery health status board).
    3. `Model Benchmarks & Validation` (Empirical ROC/PR curves and confusion matrix).
    4. `Inspection History & Audit` (SQLite run logs with 1-click CSV export).
    5. `System Architecture & ISO Standards` (ISO 13374-1 diagram, tech stack, dataset citation).
- **Chosen option:** Option C.
- **Reason / evidence:** Created a cohesive, executive-grade industrial software experience that impressed preliminary judges and cleanly separates diagnostic depth from fleet management.
- **Impact on scope:** Redesigned `app/dashboard.py` (1,134 LOC).
- **Documents updated:** `app/dashboard.py`, `docs/PROJECT_MASTER_GUIDE.md`, `docs/UI_UX_GUIDELINES.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-012: Preliminary Submission Completion & Phase 2 Finals Roadmap
- **Date:** 2026-11-20
- **Question / change:** What is the current milestone status following the successful completion of the CloudCamp preliminary submission, and how should Phase 2 (Finals) be structured?
- **Chosen option:** Officially mark Milestones M0 through M8 as **COMPLETE**. Transition engineering focus to Phase 2:
  - M9: Custom Model Context Protocol (MCP) Server (`src/mcp_server.py`).
  - M10: Production FastAPI REST & WebSocket Backend (`src/api.py`).
  - M11: Lovable Modern React Enterprise Frontend (`frontend/`).
  - M12: Multi-Machine Industrial Fleet Extension (Pumps, Valves, Slide Rails).
  - M13: Edge Runtime Packaging (ONNX CPU Export & Edge Gateway Guide).
  - M14: Finals Pitch Deck, Stage Rehearsal & Live Resilience Suite.
- **Reason / evidence:** Preliminary submission was submitted on time with 138/138 passing tests, live URL, and YouTube video. Phase 2 roadmap targets maximum rubric points for the live Finals competition.
- **Impact on scope:** Activates Phase 2 engineering sprint.
- **Documents updated:** `docs/IMPLEMENTATION_PLAN.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`, `docs/PROJECT_MASTER_GUIDE.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-013: Model Context Protocol (MCP 2.x) Server Architecture & Implementation
- **Date:** 2026-11-22
- **Question / change:** How should the PulseForge AI Model Context Protocol (MCP) server be designed and implemented to provide seamless integration with external AI agent ecosystems (Antigravity, Claude Desktop, Cursor) while avoiding code duplication?
- **Options considered:**
  - (A) Re-implement audio processing and ML inference logic inside the MCP server file.
  - (B) External REST proxy calling FastAPI endpoints (requires separate running background server).
  - (C) Direct architectural adapter using official MCP 2.x SDK (`mcp.server.mcpserver.MCPServer`), importing and reusing `src/service.py`, `src/persistence.py`, and `src/copilot.py` singletons.
- **Chosen option:** Option C (`src/mcp_server.py` using `mcp>=2.0.0`, with 5 Tools, 4 Resources, 2 Prompts, and dual Stdio/SSE transport support).
- **Reason / evidence:** Guarantees 100% logic and diagnostic parity between the Streamlit operator dashboard and external agents. Eliminates code duplication. Validated by 18 dedicated automated tests in `tests/test_mcp_server.py` (total project test suite: 156/156 passing). Captures 12 rubric points in the CloudCamp BD competition.
- **Impact on scope:** Completes Milestone M9.
- **Documents updated:** `requirements.txt`, `src/mcp_server.py`, `tests/test_mcp_server.py`, `docs/MCP_SPECIFICATION.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`.
- **Approved by:** Tamzidul & Antigravity.

### Decision DEC-014: Production FastAPI REST & WebSocket Backend Architecture (Milestone M10)
- **Date:** 2026-11-24
- **Question / change:** How should the backend headless API and streaming architecture be decoupled from the Streamlit UI to support enterprise SCADA/MES integration and the upcoming Lovable React frontend (M11)?
- **Options considered:**
  - (A) Maintain Streamlit as the sole API host via hacky query parameter routes.
  - (B) Flask REST API without native async or WebSocket streaming.
  - (C) High-performance FastAPI backend in `src/api.py` with Uvicorn, Pydantic v2 schemas, CORS middleware, full REST suite (Analysis, Health, Scenarios, SQLite Audit, Copilot Advisory), and bidirectional WebSocket streaming (`/ws/v1/audio-stream`) for live microphone FFT telemetry.
- **Chosen option:** Option C.
- **Reason / evidence:** FastAPI provides asynchronous high-throughput execution, automated OpenAPI/Swagger documentation (`/docs`), strict Pydantic validation, and real-time WebSocket communication (<1ms telemetry chunk calculations). 16 automated tests in `tests/test_api.py` passed with 100% success rate, bringing total project tests to 172/172.
- **Impact on scope:** Completes Milestone M10.
- **Documents updated:** `requirements.txt`, `src/api.py`, `tests/test_api.py`, `docs/API_DOCUMENTATION.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/DECISIONS_AND_OPEN_QUESTIONS.md`.
- **Approved by:** Tamzidul & Antigravity.
