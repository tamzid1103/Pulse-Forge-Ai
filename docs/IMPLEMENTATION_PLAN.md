# Sequential Implementation Plan

## 1. Planning principles

- The first machine is a **fan**. Other machine types are future extensions.
- Complete a thin end-to-end vertical slice before adding extra features.
- Do not spend the first hours polishing the UI before confirming the dataset and model path.
- Do not require a full multi-gigabyte dataset download if a smaller representative subset can validate the pipeline.
- Record actual outputs and blockers at each milestone.
- The 48-hour schedule is a suggested hackathon allocation, not a guarantee.

## 2. Milestone overview

| Milestone | Objective | Priority | Status | Dependency |
|---|---|---|---|---|
| M0 | Environment and repository setup | Must | ✅ Complete (Python 3.12, PyTorch CPU, Librosa) | None |
| M1 | Dataset audit and fan subset | Must | ✅ Complete (MIMII id_00 6dB fan, 50 files) | M0 |
| M2 | Audio exploration and preprocessing | Must | ✅ Complete (128-band log-Mel, FeatureScaler) | M1 |
| M3 | Baseline anomaly detector | Must | ✅ Complete (Isolation Forest, 0.9167 AUROC) | M2 |
| M4 | Autoencoder training and threshold | Must | ✅ Complete (Dense AE, 0.9375 AUROC, 1.0 Recall) | M2 |
| M5 | Reliability and decision policy | Must | ✅ Complete (Quality + Familiarity + 4 States) | M3/M4 |
| M6 | Service layer, SQLite audit & Streamlit MVP | Must | ✅ Complete (5 demo files, multi-view dashboard) | M5 |
| M7 | Preliminary submission dossier & video script | Must | ✅ Complete (Submitted to CloudCamp portal) | M6 |
| M8 | Factory Brain AI Copilot (Groq LPU + SOP RAG) | Must | ✅ Complete (138/138 tests passing, zero-crash fallback) | M6 |
| M9 | PulseForge Custom Model Context Protocol (MCP) Server | Should | ✅ Complete (MCPServer 2.x, 5 tools, 4 resources, 2 prompts) | M6/M8 |
| M10 | FastAPI REST & WebSocket Backend | Should | ✅ Complete (18 tests, REST + Live WS streaming) | M6 |
| M11 | Lovable React Industrial Enterprise Frontend | Should | ✅ Complete (React 18 + Vite + Tailwind + Lovable Cloud) | M10 |
| M12 | Multi-Machine Industrial Fleet Extension (Pumps, Valves) | Could | ✅ Complete (Fan, Pump, Valve, Slide Rail MIMII models + Registry) | M4/M5 |
| M13 | Edge Runtime Packaging (ONNX CPU Export & Gateway) | Could | 📋 Phase 2 (Finals) | M4/M8 |
| M14 | Finals Pitch Deck, Stage Rehearsal & Live Resilience | Must | 📋 Phase 2 (Finals) | M9-M11 |

## 3. Milestone details

### M0 — Environment and repository setup

**Objective:** Create a reproducible Python project environment.

**Files / directories:**

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/config.py`
- `tests/`
- directories described in `TECH_STACK.md`

**Instructions:**

1. Check Python version, OS, free disk space, and available GPU.
2. Create and activate a virtual environment.
3. Install dependencies; resolve PyTorch build separately if required.
4. Initialize Git and verify large data/model files are ignored.
5. Add configuration defaults and logging setup.
6. Run a simple import check and `pytest` baseline.

**Testing:** clean environment install, import test, Git ignore test.

**Expected output:** repository skeleton and usable development environment.

**Acceptance criteria:** documented install works; app dependencies import; no secrets or dataset archives are tracked.

### M1 — Dataset audit and fan subset

**Objective:** Confirm actual archive format and create a small, traceable fan dataset subset.

**Files:**

- `scripts/inspect_dataset.py`
- `scripts/build_manifest.py`
- `data/manifests/fan_manifest.csv` (local/ignored or small sanitized manifest if permitted)
- `docs/DATASET_AND_MODEL_GUIDE.md` updates

**Instructions:**

1. Download only one selected fan archive / manageable subset from MIMII.
2. Record source URL, archive filename, noise condition, checksum if available, size, and license details.
3. Inspect file names and folder structure before coding dataset assumptions.
4. Decode a small sample from normal and anomalous classes.
5. Count files, inspect labels, sample rate, channels, and duration.
6. Create a manifest with recording ID, relative path, label, group/section, and condition.
7. Design group-aware splits to avoid leakage.

**Testing:** every manifest row resolves to a file; sample WAVs decode; labels are valid; no duplicate source IDs cross splits.

**Expected output:** documented dataset subset and manifest.

**Acceptance criteria:** one normal and one anomalous fan recording can be loaded; split membership is reproducible; data volume is manageable.

### M2 — Audio exploration and preprocessing

**Objective:** Make audio loading and feature generation deterministic.

**Files:**

- `src/audio_io.py`
- `src/preprocessing.py`
- `src/features.py`
- `notebooks/01_audio_exploration.ipynb`
- `tests/test_audio_io.py`
- `tests/test_features.py`

**Instructions:**

1. Load WAV using SoundFile/Librosa.
2. Decide and document channel selection/downmixing.
3. Standardize sample rate and segment length.
4. Generate waveform and log-Mel visualizations.
5. Extract fixed-shape features from normal and abnormal examples.
6. Fit normalization using training data only.
7. Save the preprocessing configuration.

**Testing:** supported WAV decode; invalid file rejection; expected feature dimensions; finite feature values; consistent preprocessing for train/inference.

**Expected output:** repeatable feature tensors and plots.

**Acceptance criteria:** at least two recordings are visualized and converted to expected-shape features without errors.

### M3 — Baseline anomaly detector

**Objective:** Establish a simple benchmark before tuning the Autoencoder.

**Files:**

- `src/models/baseline.py`
- `scripts/train_baseline.py`
- `tests/test_baseline.py`

**Instructions:**

1. Select an interpretable feature vector or pooled feature representation.
2. Fit Isolation Forest using normal training samples or a documented baseline setup.
3. Define score direction and aggregation.
4. Select threshold using validation data only.
5. Evaluate against held-out test labels.
6. Save model and metrics.

**Testing:** deterministic inference under fixed seed; finite scores; threshold behavior; metrics calculation.

**Expected output:** baseline model and measured results.

**Acceptance criteria:** baseline can score a new sample and results can be reproduced from the manifest/config.

### M4 — Autoencoder anomaly model

**Objective:** Train the selected main model on normal fan acoustic features.

**Files:**

- `src/models/autoencoder.py`
- `scripts/train_autoencoder.py`
- `scripts/evaluate.py`
- `model_artifacts/` (local/ignored)
- `results/` (local/ignored)
- `tests/test_autoencoder.py`

**Instructions:**

1. Use the M2 feature pipeline.
2. Train using normal training data.
3. Track train/validation loss and prevent unnecessary overfitting.
4. Compute recording-level reconstruction scores using a documented aggregation rule.
5. Select threshold based on validation data and a declared operating criterion.
6. Evaluate once on held-out test data.
7. Compare to M3 baseline.
8. Save checkpoint, model config, threshold, environment details, and metrics.

**Testing:** checkpoint save/load; same input/config yields consistent score; finite reconstruction; model input shape is checked.

**Expected output:** trained model artifact and first honest evaluation report.

**Acceptance criteria:** model loads in a fresh process and produces a score for a held-out recording; threshold and evaluation split are documented.

### M5 — Signal quality, unfamiliarity, and decision policy

**Status:** Completed (Verified across 112 passing unit & integration tests)

**Objective:** Implement the trustworthiness differentiator.

**Files:**

- `src/schemas.py`
- `src/quality.py`
- `src/familiarity.py`
- `src/thresholding.py`
- `src/decision.py`
- `scripts/calibrate_decision.py`
- `tests/test_schemas.py`
- `tests/test_quality.py`
- `tests/test_thresholding.py`
- `tests/test_familiarity.py`
- `tests/test_decision.py`
- `tests/test_end_to_end_decision.py`

**Instructions:**

1. Implement required decode, duration, silence/near-silence, and clipping checks.
2. Set configurable thresholds after inspecting representative data.
3. Add an initial feature/embedding distance familiarity check with a validation-derived limit, if feasible.
4. Define explicit status logic: normal, possible_anomaly, needs_review, invalid_input.
5. Build result explanations from actual measured flags/scores.
6. Ensure anomaly score is never labelled as failure probability or confidence.

**Testing:** valid normal; valid anomaly; silent; clipped; too short; unfamiliar; combined anomaly + unfamiliar case; threshold boundary cases.

**Expected output:** typed analysis result with independent anomaly / quality / unfamiliarity fields.

**Acceptance criteria:** all decision paths are deterministic and unit-tested; poor/unfamiliar inputs cannot silently appear as normal.

## 3. Two-phase sprint structure (Preliminary submission vs. Final round)

To accommodate the **Preliminary Submission Deadline (Nov 20)** while ensuring full-score alignment with the CloudCamp BD competition rubrics, the roadmap is organized into two distinct phases:
- **Phase 1 (Tonight / Tomorrow Morning):** Deliver a rock-solid, fully functional, beautiful local product slice (Streamlit + SQLite + 5 preloaded demo scenarios) and an airtight video recording script so Tamzid can record the 2-minute video demo and submit on the portal tomorrow morning.
- **Phase 2 (Next 2 Weeks for Finals):** Expand the platform with advanced agentic RAG (Ollama), custom MCP server, Lovable React UI, public cloud deployment, and multi-machine industrial support.

---

### PHASE 1: PRELIMINARY SUBMISSION SPRINT (TONIGHT & TOMORROW MORNING)

### M6 — End-to-end application service, SQLite audit history, & interactive Streamlit dashboard

**Status:** Completed & Verified (All 5 demo scenarios validated; Streamlit operational; 129/129 tests passing)

**Objective:** Create the unified application service, local embedded SQLite metadata history, pre-packaged demo scenarios, and an interactive, executive-grade Streamlit dashboard for tomorrow morning's video demo recording.

**Files:**

- `src/service.py`
- `src/persistence.py`
- `app/dashboard.py`
- `demo_audio/` (5 curated, representative audio files)
- `tests/test_service.py`
- `tests/test_persistence.py`

**Instructions:**

1. Implement `src/service.py`: centralized application entrypoint coordinating audio decoding, signal quality verification, feature extraction, autoencoder scoring, familiarity checking, and decision policy.
2. Implement `src/persistence.py`: local SQLite metadata history logging all `analysis_runs` according to `docs/DATABASE_SCHEMA.md` (metadata only; no raw audio stored).
3. Bundle 5 curated demo audio clips in `demo_audio/`:
   - `01_normal_fan.wav` (Expected: NORMAL — Green status)
   - `02_bearing_anomaly.wav` (Expected: POSSIBLE_ANOMALY — Red status)
   - `03_silent_sensor_failure.wav` (Expected: NEEDS_REVIEW — Amber status)
   - `04_clipped_sensor_saturation.wav` (Expected: NEEDS_REVIEW — Amber status)
   - `05_unfamiliar_operating_noise.wav` (Expected: NEEDS_REVIEW — Amber status)
4. Build `app/dashboard.py`:
   - Clean industrial aesthetics with plant equipment metadata (`fan_id_00`).
   - 1-Click Preloaded Demo Selector (select demo scenario instantly without browsing files during video recording).
   - Audio playback player (hear the fan sound directly in the browser).
   - Dynamic Waveform and Log-Mel Spectrogram visualizers.
   - Anomaly Score vs. Baseline Threshold bar comparison.
   - Latent Familiarity distance gauge.
   - Prominent 4-state outcome badge with actionable maintenance recommendations.
   - SQLite-backed Recent Inspection History table.

**Testing:** Unit tests for service pipeline, SQLite schema CRUD, and demo scenario verification.

**Expected output:** Fully runnable, visually impressive local Streamlit app with zero external network reliance.

**Acceptance criteria:** One-command launch (`streamlit run app/dashboard.py`); all 5 demo scenarios produce their exact expected status; SQLite logs inspection runs.

---

### M7 — Preliminary submission dossier & video recording script

**Status:** Completed & Delivered (`docs/PRELIMINARY_VIDEO_SCRIPT.md` & `docs/PRELIMINARY_SUBMISSION_DOSSIER.md` ready for tomorrow morning)

**Objective:** Deliver an exact, timed video recording script and complete, pre-filled submission text for the CloudCamp portal.

**Files:**

- `docs/PRELIMINARY_VIDEO_SCRIPT.md`
- `docs/PRELIMINARY_SUBMISSION_DOSSIER.md`
- `docs/DEMO_GUIDE.md` (updates)

**Instructions:**

1. Author `docs/PRELIMINARY_VIDEO_SCRIPT.md`:
   - Timed 2–3 minute click-by-click narration script.
   - Opening hook (problem of industrial downtime & noisy false alarms).
   - Live software walkthrough (clicking the preloaded scenarios in Streamlit).
   - Technical highlight (MIMII dataset, PyTorch Autoencoder, Signal Quality, Latent Familiarity, Trustworthiness Differentiator).
   - Closing pitch on the Factory Brain vision and roadmap.
2. Author `docs/PRELIMINARY_SUBMISSION_DOSSIER.md`:
   - Complete, audited text answers for every form field on the CloudCamp portal (Project Title, Elevator Pitch, Problem, Solution, AI Architecture, Dataset Attribution, GitHub URL).

**Expected output:** Ready-to-record video script and ready-to-paste submission dossier.

**Acceptance criteria:** Video script timed under 3 minutes; all submission answers exceed character minimums.

---

### PHASE 2: FINAL ROUND PRODUCTION & ADVANCED EXPANSION

### M8 — Factory Brain Copilot (Groq LPU + SOP Knowledge Base + Deterministic Fallback)

**Status:** ✅ Completed & Verified (9 tests in `tests/test_copilot.py`; 138/138 total passing tests)

**Objective:** Add an agentic advisory layer grounded in rotating machinery SOPs and ISO 10816 condition monitoring standards to auto-generate OSHA 1910.147 Lockout/Tagout maintenance work orders and answer technician queries at 500+ tok/s.

**Files:**
- `src/copilot.py`
- `data/knowledge_base/fan_maintenance_sop.md`
- `data/knowledge_base/iso_10816_acoustic_standards.md`
- `tests/test_copilot.py`

**Accomplished:**
1. Built `IndustrialCopilot` engine with Groq LPU API execution supporting automated model fallback cascade (`llama-3.1-8b-instant` -> `llama3-8b-8192` -> `llama-3.3-70b-versatile`).
2. Implemented deterministic, zero-crash offline SOP fallback that generates complete, structured maintenance tickets even without an API key or when offline.
3. Added 1-Click Work Order Generator in Streamlit UI rendering structured JSON and formatted Markdown.
4. Added interactive technician diagnostic chat Q&A assistant in the UI.

---

### M9 — PulseForge Custom Model Context Protocol (MCP) Server

**Status:** ✅ Completed & Verified (18 tests in `tests/test_mcp_server.py`; 156/156 total passing tests)

**Objective:** Expose PulseForge diagnostic and copilot capabilities as a standardized Model Context Protocol (MCP 2.x) server (`src/mcp_server.py`) over Stdio and SSE transports, allowing external AI coding and operations agents (Antigravity, Claude, Cursor) to directly query machine health and generate work orders.

**Files:**
- `src/mcp_server.py`
- `tests/test_mcp_server.py`
- `docs/MCP_SPECIFICATION.md`
- `docs/examples/claude_desktop_config.json`
- `docs/examples/antigravity_mcp_config.json`

**Accomplished:**
1. Built MCP 2.x server with 5 Tools: `analyze_fan_audio`, `inspect_signal_quality`, `get_machine_telemetry`, `generate_maintenance_work_order`, `list_demo_scenarios`.
2. Implemented 4 Resources: `pulseforge://fleet/status`, `pulseforge://standards/iso10816`, `pulseforge://standards/sop/fan_maintenance`, `pulseforge://scenarios/catalog`.
3. Implemented 2 Prompts: `triage_acoustic_anomaly`, `osha_loto_compliance_check`.
4. Fully tested with 18 automated tests in `tests/test_mcp_server.py` bringing total project tests to 156.
5. Authored client configuration examples for Claude Desktop and Antigravity IDE.

---

### M10 — FastAPI Production REST & WebSocket Backend

**Status:** ✅ Completed & Verified (16 tests in `tests/test_api.py`; 172/172 total passing tests)

**Objective:** Decouple backend inference and telemetry persistence into a high-performance FastAPI service (`src/api.py`), providing headless REST endpoints for third-party plant SCADA/MES systems and real-time WebSocket audio streaming for modern frontends.

**Files:**
- `src/api.py`
- `tests/test_api.py`
- `docs/API_DOCUMENTATION.md`
- `docs/DECISIONS_AND_OPEN_QUESTIONS.md` (DEC-014)

**Accomplished:**
1. Implemented full REST suite:
   - `GET /health` (lightweight service & model readiness check)
   - `GET /api/v1/system/status` (telemetry thresholds, 4096-dim features, fleet status)
   - `GET /api/v1/scenarios` (catalog of 5 curated demonstration audio scenarios)
   - `POST /api/v1/analyze` (multipart WAV upload with technical validation and 4-state classification)
   - `POST /api/v1/analyze/scenario/{scenario_id}` (1-click evaluation of demo scenarios with alias support)
   - `GET /api/v1/runs` & `GET /api/v1/runs/{id}` (paginated SQLite audit history)
   - `DELETE /api/v1/runs` (audit trail purge for demonstration resets)
   - `POST /api/v1/copilot/work-order` & `POST /api/v1/copilot/chat` (AI copilot advisory)
2. Implemented WebSocket endpoint:
   - `WS /ws/v1/audio-stream` for bidirectional real-time audio chunk streaming, computing instantaneous RMS amplitude, clipping ratio, and 3-band FFT spectral energy (<1ms execution).
3. Configured CORS middleware, Pydantic v2 schemas, and structured exception handlers.
4. Added 16 automated tests in `tests/test_api.py` verifying all endpoints and protocols (172/172 total project tests passing).

---

### M11 — Lovable React Industrial Enterprise Frontend

**Status:** ✅ Complete (Local TypeScript + React 18 + Vite + Tailwind + Lovable Cloud Project)

**Objective:** Build a high-performance modern React + Vite + Tailwind CSS industrial operations dashboard via Lovable MCP, offering enterprise-grade plant floor UX, multi-machine status boards, and animated spectrogram canvas visualization.

**Files:**
- `frontend/package.json` & `frontend/tsconfig.json` & `frontend/vite.config.ts`
- `frontend/src/types/index.ts` & `frontend/src/services/api.ts`
- `frontend/src/components/Navbar.tsx`
- `frontend/src/components/AudioVisualizer.tsx`
- `frontend/src/components/DiagnosticStudio.tsx`
- `frontend/src/components/LiveStreamStudio.tsx`
- `frontend/src/components/FleetHealthBoard.tsx`
- `frontend/src/components/CopilotDrawer.tsx`
- `frontend/src/components/AuditTrail.tsx`
- `frontend/src/App.tsx` & `frontend/src/main.tsx`

**Accomplishments:**
1. Built full-stack TypeScript + React 18 + Tailwind CSS dark-mode SCADA operations UI (`#0B0F17`) conforming to ISO 13374 and ISO 10816-3.
2. Implemented 4 operational views: Acoustic Diagnostic Studio (WAV upload + 5 demo scenarios), Live Stream Studio (WebSocket streaming + 3-band FFT bars), Fleet Health Board (multi-asset matrix), and Inspection Audit Trail (SQLite run logs + CSV export).
3. Built Factory Brain Copilot drawer with 1-click OSHA 1910.147 Lockout/Tagout work order generation and technician Q&A chat.
4. Created and linked Lovable cloud project via `create_project` tool (`projectId: 87ff379e-b3a5-47ed-ade8-f1345d92ab5d`, Live URL: `https://id-preview--87ff379e-b3a5-47ed-ade8-f1345d92ab5d.lovable.app`).
5. Verified production bundling with `npm run build` (built in 2.64s with zero TypeScript errors).

---

### M12 — Multi-Machine Industrial Fleet Extension (Pumps, Valves, Slide Rails)

**Status:** ✅ Complete (All 4 MIMII Machine Families Trained, Routed & Verified)

**Objective:** Extend acoustic diagnostic capabilities beyond centrifugal ventilation fans to include other core rotating and fluid machinery families from the MIMII benchmark (Centrifugal Pumps, Industrial Valves, and Linear Slide Rails).

**Files:**
- `src/models/registry.py` (`ModelRegistry` dynamic loader + `MachineModelBundle`)
- `data/manifests/pump_manifest.csv`, `data/manifests/valve_manifest.csv`, `data/manifests/slider_manifest.csv`
- `scripts/download_subset.py`, `scripts/build_manifest.py`, `scripts/train_multi_machine.py`
- `data/knowledge_base/multi_machine_maintenance_sop.md`
- `results/multi_machine_summary.json`
- `tests/test_multi_machine.py` (20 automated tests)

**Accomplishments:**
1. Downloaded genuine MIMII 6 dB SNR acoustic subsets (`id_00`, 34 files each: 24 normal / 10 abnormal) for Pump, Valve, and Slide Rail via HTTP Range requests and built leakage-aware manifests.
2. Trained and calibrated dedicated `FeatureScaler`, `AutoencoderDetector`, and `LatentFamiliarityEstimator` bundles (`pump` AUROC 1.0000 / Recall 1.0000; `valve` AUROC 0.7200 / Recall 0.8000 / F1 0.8000; `slider` AUROC 1.0000 / Recall 1.0000).
3. Implemented `ModelRegistry` in `src/models/registry.py` and refactored `PulseForgeService` for dynamic machine-type routing, plus non-destructive SQLite `CHECK` migration in `src/persistence.py`.
4. Extended Copilot SOPs (`WO-PUMP-`, `WO-VALVE-`, `WO-SLDR-`), FastAPI (`/api/v1/fleet/scenarios`, `machine_type` routing), MCP server, Streamlit dashboard, and React frontend (`frontend/`).
5. Exported 6 curated 16 kHz mono held-out test demo clips (`demo_audio/06..11_*.wav`) verified end-to-end across normal, anomaly, and OOD checks.

---

### M13 — Edge Runtime Packaging (ONNX CPU Export & Edge Gateway Guide)

**Status:** 📋 Phase 2 (Planned)

**Objective:** Export PyTorch Autoencoder models to optimized ONNX format for sub-25ms inference on resource-constrained industrial edge gateways (Raspberry Pi 5 / NVIDIA Jetson Orin Nano).

**Files:**
- `scripts/export_onnx.py`
- `src/models/onnx_runner.py`
- `docs/EDGE_GATEWAY_DEPLOYMENT_GUIDE.md`
- `deploy/Dockerfile.edge`

**Key Tasks:**
1. Export PyTorch dense Autoencoder to ONNX with dynamic batching.
2. Validate ONNX Runtime CPU inference matches PyTorch output within $\epsilon = 1e-5$.
3. Benchmark edge CPU latency (target: < 30 ms per 10s audio segment).
4. Provide standalone Docker edge container and systemd service recipe for 24/7 autonomous monitoring.

---

### M14 — Finals Pitch Deck, Stage Rehearsal & Live Resilience Suite

**Status:** 📋 Phase 2 (Planned)

**Objective:** Prepare world-class presentation materials, keynote slides, and an airtight offline live-demo rehearsal package for the CloudCamp BD Finals judging stage.

**Files:**
- `docs/FINALS_PITCH_DECK.md`
- `docs/FINALS_STAGE_SCRIPT.md`
- `scripts/run_offline_demo.sh`

**Key Tasks:**
1. Author comprehensive 5-minute keynote presentation script and slide deck.
2. Build 1-click offline resilience launcher (`scripts/run_offline_demo.sh`) that spins up the local Streamlit dashboard and mock API with zero external dependencies.
3. Conduct timed rehearsals covering the problem, live audio demo, ISO standards compliance, and business ROI.

## 4. Suggested team roles (adapt to actual team)

| Role | Responsibility |
|---|---|
| Data / audio engineer | Dataset audit, manifest, preprocessing, feature pipeline, multi-machine expansion |
| ML engineer | Baseline, Autoencoder, thresholding, ONNX optimization, evaluation |
| Reliability / QA engineer | Quality checks, familiarity flags, decision tests, ISO standards grounding |
| App / Fullstack engineer | Streamlit dashboard, FastAPI backend, Lovable React frontend, MCP server |
| Product / integration lead | Scope, merge/release, video script, slides, live stage presentation |

## 5. Implementation status summary

| Milestone | Status | Evidence | Blocker / next action |
|---|---|---|---|
| M0 | ✅ Complete | 19 smoke tests passed; `.venv` with Python 3.12.3, PyTorch CPU, Librosa, Streamlit, Scikit-learn; config & skeleton created; requirements pinned in `requirements-lock.txt` | None. Proceed to M1 |
| M1 | ✅ Complete | 8 dataset tests passed (27 tests total); 50 genuine MIMII fan recordings downloaded from Zenodo (35 normal, 15 abnormal); `fan_manifest.csv` built with leakage-aware splits | None. Ready for M2 |
| M2 | ✅ Complete | 22 audio I/O & feature tests passed (49 tests total); deterministic log-Mel extraction (128 Mel bins, 1.0s window, 0.5s hop); FeatureScaler fitted on 21 normal train recordings (399 segments) and saved to `scaler_params.json`; normal and abnormal diagnostic plots generated in `results/plots/` | None. Ready for M3 |
| M3 | ✅ Complete | 10 baseline tests passed (59 tests total); Isolation Forest fitted on 256-dim pooled Mel features; validation threshold derived at target FPR=0.10; test AUROC = 0.9167, PR-AUC = 0.9583, Recall = 0.8750 | None. Ready for M4 |
| M4 | ✅ Complete | 12 autoencoder tests passed (71 tests total); PyTorch symmetric dense Autoencoder trained on normal fan sounds; test AUROC = 0.9375, Recall = 1.0000 (100% mechanical fault detection), F1 = 0.9333; weights saved to `model_artifacts/autoencoder_fan_id_00.pt` | None. Ready for M5 |
| M5 | ✅ Complete | 41 quality, familiarity, and decision tests passed (112 tests total); 4-state deterministic decision policy (`normal`, `possible_anomaly`, `needs_review`, `invalid_input`); latent Mahalanobis familiarity engine with threshold = 9.25 | None. Ready for M6 |
| M6 | ✅ Complete | 17 service, persistence, and UI tests passed (129 tests total); `src/service.py` pipeline, SQLite audit trail (`src/persistence.py`), 5 curated demo audio scenarios in `demo_audio/`, and multi-view Streamlit dashboard (`app/dashboard.py`) | None. Ready for M7 |
| M7 | ✅ Complete | Authored `docs/PRELIMINARY_VIDEO_SCRIPT.md` (2:45 min walkthrough) and `docs/PRELIMINARY_SUBMISSION_DOSSIER.md`; recorded video demo; preliminary submission officially completed on CloudCamp portal | None. Ready for M8 |
| M8 | ✅ Complete | 9 copilot tests passed (138 tests total); `src/copilot.py` with Groq LPU API (`llama-3.1-8b-instant`), automated fallback cascade, domain-grounded fan SOPs, and deterministic offline fallback; 1-click work order generation in UI | None. Ready for Phase 2 (Finals) |
| M9 | ✅ Complete | 18 MCP tests passed (156 tests total); `src/mcp_server.py` with 5 tools, 4 resources, 2 prompts; Stdio/SSE transports; Claude & Antigravity client configs | None. Ready for M10 |
| M10 | ✅ Complete | 18 automated tests passed (178 tests total); FastAPI backend in `src/api.py` with REST endpoints, WebSocket streaming, Pydantic schemas, and security hardening (CORS + upload streaming limit) | None. Ready for M11 |
| M11 | ✅ Complete | TypeScript + React 18 + Vite + Tailwind SCADA dashboard (`frontend/`), 4 operational views, real-time WebSocket audio streaming visualizer, OSHA work order generator, and Lovable cloud deployment (`projectId: 87ff379e-b3a5-47ed-ade8-f1345d92ab5d`) | None. Ready for M12 |
| M12 | ✅ Complete | 20 multi-machine tests passed (198 tests total); authentic MIMII subsets downloaded for Pump (`1.0000` AUROC), Valve (`0.7200` AUROC / `0.8000` F1), and Slide Rail (`1.0000` AUROC); `ModelRegistry` routing, SQLite migration, and dual UI selectors live | None. Ready for M13 |
| M13 | 📋 Phase 2 | ONNX CPU runtime export & Edge Gateway deployment | Planned for Finals |
| M14 | 📋 Phase 2 | Finals Pitch Deck, Stage Rehearsal & Live Resilience Suite | Final delivery for competition stage |
