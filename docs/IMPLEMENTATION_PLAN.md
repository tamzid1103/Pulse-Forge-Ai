# Sequential Implementation Plan

## 1. Planning principles

- The first machine is a **fan**. Other machine types are future extensions.
- Complete a thin end-to-end vertical slice before adding extra features.
- Do not spend the first hours polishing the UI before confirming the dataset and model path.
- Do not require a full multi-gigabyte dataset download if a smaller representative subset can validate the pipeline.
- Record actual outputs and blockers at each milestone.
- The 48-hour schedule is a suggested hackathon allocation, not a guarantee.

## 2. Milestone overview

| Milestone | Objective | Priority | Dependency |
|---|---|---|---|
| M0 | Environment and repository setup | Must | None |
| M1 | Dataset audit and fan subset | Must | M0 |
| M2 | Audio exploration and preprocessing | Must | M1 |
| M3 | Baseline anomaly detector | Must | M2 |
| M4 | Autoencoder training and threshold | Must | M2 |
| M5 | Reliability and decision policy | Must | M3/M4 |
| M6 | Streamlit end-to-end MVP | Must | M5 |
| M7 | Evaluation and demo scenarios | Must | M4/M5/M6 |
| M8 | Polish, docs, pitch, and final rehearsal | Must | M7 |
| M9 | SQLite / FastAPI / additional explanations | Optional | M8 or stable M6 |
| M10 | Extra machine types / DUE-DG / sensor capture | Future | Separate data and validation |

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

### M6 — Service layer, SQLite persistence, dual frontend (Streamlit + Lovable), & public tunneling

**Status:** Ready to Implement

**Objective:** Create the unified application service, persistent audit history, FastAPI backend, local Streamlit dashboard, Lovable React UI, and secure public tunneling to fulfill the Live Demo requirement.

**Files:**

- `src/service.py`
- `src/persistence.py`
- `src/api.py`
- `app/dashboard.py`
- `scripts/run_tunnel.sh`
- `tests/test_service.py`
- `tests/test_persistence.py`
- `tests/test_api.py`

**Instructions:**

1. Implement `src/service.py`: centralized application entrypoint coordinating audio decoding, signal quality verification, feature extraction, autoencoder scoring, familiarity checking, and decision policy.
2. Implement `src/persistence.py`: local SQLite metadata history logging all `analysis_runs` according to `docs/DATABASE_SCHEMA.md` (no raw audio stored).
3. Implement `src/api.py`: FastAPI application providing `GET /health`, `POST /api/v1/analyze`, and `GET /api/v1/history`.
4. Build `app/dashboard.py`: Streamlit operator interface with waveform/spectrogram rendering, live audio playback, preloaded demo selectors, and 4-state visual indicators.
5. Connect Lovable MCP to generate the modern React industrial web app communicating with the FastAPI backend.
6. Provide `scripts/run_tunnel.sh` to publish the application securely to the internet via Cloudflare Tunnel (`cloudflared`) or ngrok (unlocking 6 rubric points for Live Demo URL and Tunneling notes).

**Testing:** Unit tests for service pipeline, SQLite schema CRUD, FastAPI endpoint status codes/validation, and tunnel scripts.

**Expected output:** Fully runnable local Streamlit app, active FastAPI backend, and public live tunnel URL.

**Acceptance criteria:** One-command startup for both Streamlit and FastAPI; SQLite logs metadata accurately; public tunnel renders the application over HTTPS.

---

### M7 — PulseForge custom MCP server & Factory Brain Copilot (Semantic RAG + Ollama)

**Status:** Planned

**Objective:** Build a custom Model Context Protocol (MCP) server and an agentic Factory Brain Copilot to maximize CloudCamp AI Stack & MCP category points (up to 38 points).

**Files:**

- `src/mcp_server.py`
- `src/copilot.py`
- `data/knowledge_base/fan_maintenance_sop.md`
- `data/knowledge_base/iso_10816_acoustic_standards.md`
- `tests/test_mcp_server.py`
- `tests/test_copilot.py`

**Instructions:**

1. Build `src/mcp_server.py`: Custom MCP Server exposing PulseForge AI tools (`analyze_fan_audio`, `inspect_signal_quality`, `get_machine_status`) and resources (`pulseforge://fan-manifest`, `pulseforge://latest-telemetry`) over Stdio and Streamable HTTP/SSE.
2. Build `src/copilot.py`: Contextual RAG copilot that loads industrial fan maintenance SOPs and ISO standards using semantic chunking.
3. Integrate local model execution with Ollama (e.g. `llama3.2`, `deepseek-r1`) and cloud fallback to generate structured work orders and operator action checklists when an anomaly or review status is detected.
4. Keep the core acoustic detector 100% deterministic: the copilot acts strictly as a downstream industrial advisory agent, never altering the measured acoustic score.

**Testing:** Verify MCP tool invocation via JSON-RPC; test RAG retrieval precision; test copilot work order generation across normal, anomalous, and review states.

**Expected output:** Live custom MCP server and working agentic Factory Brain copilot.

**Acceptance criteria:** MCP server passes JSON-RPC `tools/list` and `tools/call`; copilot retrieves relevant SOP sections within 2 seconds.

---

### M8 — Evaluation benchmark suite, offline demo package, & diagnostic plots

**Status:** Planned

**Objective:** Verify model behavior, latency, and prepare a reliable offline demonstration package.

**Files:**

- `results/evaluation_report.json`
- `results/plots/roc_comparison.png`
- `results/plots/reconstruction_error_distribution.png`
- `results/plots/autoencoder_loss_curve.png`
- `demo_audio/` (5 curated, representative audio files)
- `tests/test_demo_audio.py`

**Instructions:**

1. Execute `scripts/evaluate.py` to produce final benchmark metrics comparing Baseline vs Autoencoder.
2. Generate all diagnostic comparison curves and distribution plots in `results/plots/`.
3. Bundle 5 curated demo audio clips in `demo_audio/`:
   - `01_normal_baseline.wav` (Expected: normal)
   - `02_bearing_anomaly.wav` (Expected: possible_anomaly)
   - `03_silent_sensor_failure.wav` (Expected: needs_review)
   - `04_clipped_sensor_saturation.wav` (Expected: needs_review)
   - `05_out_of_domain_noise.wav` (Expected: needs_review)
4. Record CPU inference latency benchmarks (target: < 100 ms per 10s audio clip).

**Testing:** Automated test verifying that all 5 demo audio files produce their exact documented status without network access.

**Expected output:** Complete offline demo kit and benchmark report.

**Acceptance criteria:** Zero-network demo verification; reproducible evaluation numbers.

---

### M9 — CloudCamp submission dossier & Zero to Launch single master prompt

**Status:** Planned

**Objective:** Author the single master prompt and compile the complete submission dossier to achieve 100% points on the CloudCamp Pre-Judgment Auto-Scoring Rubric.

**Files:**

- `docs/ZERO_TO_LAUNCH_MASTER_PROMPT.md`
- `docs/CLOUDCAMP_SUBMISSION_DOSSIER.md`
- `docs/PITCH_DECK.md`

**Instructions:**

1. Author `docs/ZERO_TO_LAUNCH_MASTER_PROMPT.md`: A single comprehensive master prompt adhering to the "ONE PROMPT ONLY" challenge rule, covering all 8 required modules:
   - Core Product & AI Instructions
   - Data Architecture & Pipelines
   - Admin Module & Access Controls
   - Security & Guardrails
   - Documentation Module
   - Audit Reporting & Observability
   - Presentation & Pitch Deck Outline
   - Deployment & Infrastructure Requirements
2. Author `docs/CLOUDCAMP_SUBMISSION_DOSSIER.md`: Pre-filled text and link answers for every form field on the CloudCamp submission portal, audited against character minimums (80+ chars for notes, 50+ chars for token optimization, 100+ chars for AI-DLC).
3. Outline the 10-slide pitch deck structure in `docs/PITCH_DECK.md`.

**Expected output:** Complete, audit-ready submission dossier and single master prompt.

**Acceptance criteria:** All text fields exceed character minimums; master prompt covers all 8 modules.

---

### M10 — Production packaging, video demo guide, & final audit

**Status:** Planned

**Objective:** Final end-to-end rehearsal, containerized deployment, video demo recording guide, and clean project handover.

**Files:**

- `docs/VIDEO_DEMO_SCRIPT.md`
- `deploy/Dockerfile`
- `docker-compose.yml`
- root `README.md`

**Instructions:**

1. Prepare a timed 2–3 minute video demo script (`docs/VIDEO_DEMO_SCRIPT.md`) with click-by-click instructions, script voiceover, and screen capture cues.
2. Build `deploy/Dockerfile` and `docker-compose.yml` for unified one-command container deployment.
3. Update root `README.md` with badges, system architecture, installation steps, MCP configuration, and live demo links.
4. Run full regression test suite (target: 100% pass across all modules).

**Expected output:** Production-ready containerized repository and video demo script.

**Acceptance criteria:** `docker compose up` starts the service cleanly; all documentation links resolve.

## 4. Suggested 48-hour schedule

| Time | Focus | Exit condition |
|---|---|---|
| 0–2 h | Environment, repo, roles | All developers can run Python environment |
| 2–6 h | MIMII fan subset audit | Small normal/anomaly sample manifest exists |
| 6–10 h | Audio pipeline and plots | Audio decodes; features and plots work |
| 10–16 h | Baseline + first Autoencoder | Score produced on held-out sample |
| 16–21 h | Threshold and evaluation | First metrics and model artifact saved |
| 21–26 h | Quality / unfamiliarity / decision | Normal, anomaly, review paths work in tests |
| 26–34 h | Streamlit vertical slice | Upload → result works end to end |
| 34–39 h | Integration / regression testing | Three demo cases pass |
| 39–44 h | UI polish / pitch / evidence | Clear presentation and results |
| 44–48 h | Rehearsal / backup / freeze | Demo can run offline from clean launch |

If training or dataset access takes longer, reduce scope rather than sacrificing the complete vertical slice. At minimum, keep one real model path and be transparent about limitations.

## 5. Suggested team roles (adapt to actual team)

| Role | Responsibility |
|---|---|
| Data / audio engineer | Dataset audit, manifest, preprocessing, feature pipeline |
| ML engineer | Baseline, Autoencoder, thresholding, evaluation |
| Reliability / QA engineer | Quality checks, familiarity flags, decision tests |
| App engineer | Streamlit, result rendering, optional API |
| Product / integration lead | Scope, merge/release, demo script, slides, quality gate |

For a smaller team, combine roles. Ensure one person owns integration and the final runnable build.

## 6. Implementation status template

At the end of every milestone, update a small table in the repository:

| Milestone | Status | Evidence | Blocker / next action |
|---|---|---|---|
| M0 | Complete | 19 smoke tests passed; `.venv` with Python 3.12, PyTorch CPU, Librosa, Streamlit, Scikit-learn; config & skeleton created; requirements pinned in `requirements-lock.txt` | None. Proceed to M1 |
| M1 | Complete | 8 dataset tests passed (27 tests total); 45 real MIMII fan recordings downloaded from Zenodo (30 normal, 15 abnormal); `fan_manifest.csv` built with leakage-aware splits | None. Ready for M2 |
| M2 | Complete | 22 audio I/O & feature tests passed (49 tests total); deterministic log-Mel extraction (128 Mel bins, 1.0s window, 0.5s hop); FeatureScaler fitted on 21 normal train recordings (399 segments) and saved to `scaler_params.json`; normal and abnormal diagnostic plots generated in `results/plots/` | None. Ready for M3 (Baseline Anomaly Detector) |

Never mark a milestone complete based on code generation alone; run the tests / workflow and record evidence.
