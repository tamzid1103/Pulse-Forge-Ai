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

## 4. Questions requiring project-owner confirmation

These questions do not all block starting implementation. Resolve the critical ones early; otherwise use the explicit defaults in this document.

| Question | Why it matters | Default until confirmed | Blocks MVP? |
|---|---|---|---|
| What is the actual team size and member availability? | Work allocation | Use the five-role split as an adaptable template | No |
| Which operating system will be used for the final demo? | Setup / audio dependencies | Linux or Windows documented | No |
| Is a GPU available? | Training time | CPU-compatible small experiment | No |
| How much disk space and internet bandwidth are available? | MIMII archive size | Download one fan subset only | Dataset acquisition may be blocked |
| Which exact MIMII noise condition / product ID should be used? | Reproducible split and evaluation | Inspect archive and choose a manageable representative subset | Yes, before training |
| What upload duration / size limits are appropriate? | Runtime and resource safety | Start with configurable conservative limits, test chosen clips | No |
| What is the final deployment target? | Local demo vs hosted access | Local Streamlit demo | No |
| Does the event require a public URL or only a live local demo? | Hosting / security scope | Local demo unless rules say otherwise | Possibly |
| Are there specific event rules for libraries, model usage, datasets, team size, or internet? | Compliance | Follow official event rules when supplied | Must confirm before submission |
| Should the demo UI and pitch be English, Bangla, or both? | Judge/user experience | English UI labels with team-preferred pitch language | No |
| Is SQLite history desired? | Optional feature effort | Disable for MVP | No |
| Is a separate FastAPI service required by team integration? | Integration boundary | No; use direct service call | No |

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
