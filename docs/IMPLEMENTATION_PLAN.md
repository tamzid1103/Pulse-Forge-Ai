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

### PHASE 2: FINAL ROUND PRODUCTION & ADVANCED EXPANSION (NEXT 2 WEEKS)

### M8 — Factory Brain Copilot (Contextual RAG & local Ollama integration)

**Status:** Planned (Final Round)

**Objective:** Add an agentic advisory layer using contextual RAG on industrial fan manuals and local Ollama execution (`llama3.2` / `deepseek-r1`) to auto-generate operator work orders and inspection tickets.

**Files:**

- `src/copilot.py`
- `data/knowledge_base/fan_maintenance_sop.md`
- `data/knowledge_base/iso_10816_acoustic_standards.md`
- `tests/test_copilot.py`

---

### M9 — PulseForge custom Model Context Protocol (MCP) server

**Status:** Planned (Final Round)

**Objective:** Build a custom MCP server (`src/mcp_server.py`) exposing PulseForge tools (`analyze_fan_audio`, `inspect_signal_quality`, `get_machine_health_status`) to AI assistants (Antigravity, Claude, Cursor) over Stdio and SSE transports.

**Files:**

- `src/mcp_server.py`
- `tests/test_mcp_server.py`

---

### M10 — Lovable modern React industrial web dashboard

**Status:** Planned (Final Round)

**Objective:** Export and polish the modern React + Vite + Tailwind CSS industrial dashboard from Lovable, integrating it with the FastAPI backend (`src/api.py`) for both localhost and web access.

**Files:**

- `src/api.py`
- `frontend/` (React/Vite app)
- `tests/test_api.py`

---

### M11 — Production cloud deployment & public live hosting

**Status:** Planned (Final Round)

**Objective:** Deploy the full-stack system online: FastAPI backend on Render/Railway, React frontend on Vercel/Netlify, and automated Cloudflare Tunneling (`scripts/run_tunnel.sh`).

**Files:**

- `deploy/Dockerfile`
- `docker-compose.yml`
- `scripts/run_tunnel.sh`

---

### M12 — Multi-machine industrial fleet expansion

**Status:** Planned (Final Round)

**Objective:** Expand dataset and model registry beyond fans to include MIMII pumps, valves, and slide rails with multi-machine fleet monitoring.

**Files:**

- `src/models/registry.py`
- `data/manifests/fleet_manifest.csv`

---

### M13 — Zero to Launch single master prompt & final pitch deck

**Status:** Planned (Final Round)

**Objective:** Author the canonical single master prompt covering all 8 competition modules and prepare the final presentation slide deck.

**Files:**

- `docs/ZERO_TO_LAUNCH_MASTER_PROMPT.md`
- `docs/PITCH_DECK.md`

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
