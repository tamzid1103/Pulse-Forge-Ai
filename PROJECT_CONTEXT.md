# PROJECT_CONTEXT.md — Canonical Handover Brief

**Project:** PulseForge AI  
**Tagline:** Hear the Warning. Trust the Signal. Prevent the Downtime.  
**Package version:** 1.0  
**Prepared:** 2026-11-16  
**Project state:** Proposed / not yet implemented

This file is the canonical source of truth for a fresh AI coding agent. It replaces the need to read the prior ChatGPT discussion. When a requirement is ambiguous, consult `docs/DECISIONS_AND_OPEN_QUESTIONS.md`; do not silently invent an answer.

## 1. Final selected concept

PulseForge AI is a trustworthy acoustic predictive-maintenance prototype for industrial rotating equipment. Its first iteration analyzes fan sound recordings from the MIMII industrial-machine sound dataset. It learns patterns from normal-operation recordings, produces an anomaly score for a new recording, checks input quality and familiarity, and presents a clear result with evidence and an inspection recommendation.

It is positioned under:

- **Domain:** IndustrySphere AI
- **Primary sub-track:** Predictive Plant — predictive maintenance to prevent downtime.
- **Supporting sub-track:** Factory Brain — a limited operational dashboard for machine status, alerts, and review; not a full factory-management platform.
- **Optional future extension:** Green Manufacturing, only if later work can credibly relate reliable maintenance to reduced waste/energy; it is not an MVP requirement.

The domain/sub-track labels and descriptions were selected from the user's `AllDomainList.pdf` competition document. The project idea and technical direction were finalized in the previous conversation.

## 2. Problem and target users

Industrial machines such as fans, pumps, and motors can develop unusual operating behavior before a breakdown. Unexpected downtime can interrupt production and increase inspection or repair costs. Periodic maintenance may miss developing issues between inspection intervals, while reactive maintenance discovers issues only after a failure or obvious symptom.

**Initial target user:** a factory maintenance operator or maintenance engineer reviewing a machine recording.  
**Secondary stakeholder:** a factory supervisor who needs a simple overview of machine health and pending inspections.  
**Hackathon audience:** judges evaluating innovation, technical execution, practical impact, business potential, scalability, and presentation.

PulseForge AI is initially a demonstrator using public data and uploaded audio. It is not yet a validated industrial product and must not be marketed as a replacement for qualified maintenance personnel.

## 3. Product objective

Deliver a small but complete end-to-end demo:

`WAV upload → audio validation → feature extraction → anomaly model → signal-quality / unfamiliarity checks → decision policy → visual result and inspection recommendation`

The prototype should help answer:

1. Does this recording differ from the learned normal acoustic pattern?
2. Is the audio recording suitable for analysis?
3. Does the recording appear unfamiliar relative to the training conditions?
4. Should the result be treated as normal, a possible anomaly, or inconclusive / requiring human review?
5. What evidence and practical next step should the operator see?

## 4. Scope decisions

### MVP (must deliver)

- Fan-only acoustic analysis using a manageable MIMII fan subset.
- WAV upload and input validation.
- Audio loading, waveform display, and log-Mel spectrogram or equivalent feature visualization.
- Anomaly detection using an Autoencoder trained primarily on normal fan recordings.
- A simple Isolation Forest or other transparent baseline for comparison.
- Anomaly score and a validation-selected threshold.
- Basic audio-quality checks: duration, silence/near-silence, clipping, and malformed/unsupported input.
- A clearly labelled uncertainty/review flag based on poor input quality and/or out-of-distribution / domain-shift evidence.
- A Streamlit dashboard showing the input, analysis, status, evidence, and recommended action.
- Reproducible evaluation on held-out recordings, with metrics reported only after they are measured.
- A README, dataset attribution, and reliable preloaded demo recordings where licensing permits.

### Important, only after core flow works

- Persist analysis metadata in a local SQLite database.
- A minimal FastAPI adapter for the analysis service.
- More detailed explanation of influential frequency/time regions.
- A comparison view for multiple test recordings.

### Optional / future

- Pump, motor, gearbox, or other machine types, each requiring suitable data and independent model validation.
- MIMII DUE or MIMII DG experiments for operational/environmental shift and domain generalization.
- Sensor / microphone live capture or edge gateway.
- Factory-level multi-machine dashboard and role-based accounts.
- Business deployment, industrial integrations, and pilot with a consenting factory.
- Green Manufacturing impact measurements.
- LLM/RAG for retrieving approved maintenance-manual passages; never use an LLM as the core fault detector.

## 5. Original product features and innovation

### Core predictive-maintenance behavior

- Learn normal machine sound patterns.
- Compare a new recording with learned patterns.
- Calculate an anomaly score.
- Flag possible abnormal operation.

### Trustworthiness layer — distinguishing feature

- Check the recording before inference.
- Detect obvious low-quality or invalid audio.
- Compare new input characteristics with the training distribution.
- Clearly separate anomaly score from probability / confidence.
- Flag poor-quality or unfamiliar cases for human review.
- Show the reason for a warning rather than only a red/green status.

### Factory Brain supporting experience

- One-screen machine analysis dashboard.
- Status and alert cards.
- Analysis history only if optional metadata persistence is implemented.
- Clear recommended action and human confirmation.

## 6. Recommended technical architecture

**Primary implementation:** Python modular monolith + Streamlit UI. This is chosen for hackathon speed and because audio processing, ML inference, and data visualization can remain in one Python environment.

**Core modules:**

- `audio_io`: secure file validation and audio loading.
- `preprocessing`: resampling and consistent input preparation.
- `features`: log-Mel spectrogram and any derived features.
- `models`: Autoencoder and baseline inference.
- `thresholding`: validation-derived anomaly threshold.
- `reliability`: quality checks and unfamiliarity/domain-shift flags.
- `decision`: final status logic and user-facing explanations.
- `visualization`: waveform, spectrogram, score and status rendering.
- `persistence`: optional SQLite analysis metadata.

**Optional service boundary:** a FastAPI wrapper can later expose the same application service; do not duplicate business logic between API and Streamlit.

**Database:** no database is required for the first working MVP. If history is useful and time permits, use SQLite for metadata only. Do not persist raw user audio by default.

**Authentication:** no login is required for a local, single-operator hackathon demo. Before public multi-user deployment, add authentication, authorization, rate limits, and secure storage.

**External services:** none required. MIMII is downloaded before use. No live sensor, cloud, LLM, or paid API is required for the MVP.

## 7. AI/ML decisions

- **Primary dataset:** MIMII fan audio. Original MIMII contains sounds from fans, pumps, valves, and slide rails, with normal and anomalous conditions. Review the official record and file sizes before choosing a subset.
- **Primary model:** Autoencoder trained on normal fan recordings. Use reconstruction error as an anomaly score.
- **Baseline:** Isolation Forest on suitable extracted feature vectors or aggregated segment features.
- **Input features:** start with log-Mel spectrograms; exact segment length, number of Mel bins, normalization, and model dimensions must be chosen by experiment and recorded in model metadata.
- **Threshold:** derive from a validation split of normal recordings and document the chosen false-positive operating point. Never choose a threshold by looking at the final test labels.
- **Reliability:** use explicit quality checks and an unfamiliarity / domain-shift detector. A flag is not a calibrated probability. Avoid presenting an unvalidated "confidence %".
- **Evaluation:** AUROC, precision, recall, F1, PR-AUC where helpful, false-positive rate, inference latency, and performance under selected noise/shift conditions. The test split must be independent and leakage-aware.

## 8. Safety and claim boundaries

- A model anomaly is not proof of mechanical failure.
- Do not diagnose a specific bearing, shaft, or component unless the data and experiment genuinely support it.
- Do not forecast an exact failure date or remaining useful life in the MVP.
- Do not claim an anomaly score is a failure probability.
- A poor-quality or unfamiliar recording should result in a review / inconclusive message.
- A human operator or qualified engineer makes final maintenance decisions.
- Never automatically stop, start, or control a real industrial machine.
- Do not claim real factory deployment, customer validation, partnerships, NRB collaboration, or performance values until evidence exists.

## 9. User experience

A demo operator should be able to:

1. Open the local dashboard without account creation.
2. See that the MVP is configured for fan recordings.
3. Upload a WAV file or select a bundled demo sample.
4. See an upload/analysis state and validation errors if input is unsuitable.
5. View waveform and spectrogram.
6. View anomaly score relative to the threshold and a readable status.
7. View signal quality and unfamiliarity flags separately from anomaly detection.
8. Read the reason for the flag and a non-prescriptive inspection recommendation.
9. Understand that the result is decision support, not a confirmed diagnosis.

## 10. Expected final hackathon deliverables

- Running local Streamlit app.
- Reproducible training / evaluation notebook or script.
- Trained fan Autoencoder checkpoint and model configuration.
- Baseline model result.
- Dataset subset manifest and attribution.
- Metrics and plots from held-out testing.
- Three prepared demo scenarios: normal, anomaly, and poor-quality / unfamiliar input.
- Technical README and this documentation set.
- Short presentation and pitch, with claims traceable to measured results.

## 11. Hackathon execution constraints

Previous discussion used a 48-hour build target and a five-person team as an example. Actual team size, hardware, event rules, and deployment requirements are not confirmed. The schedule must be adapted to the actual team.

Priority order:

1. Valid dataset subset + reproducible evaluation split.
2. Baseline detector and Autoencoder inference.
3. End-to-end upload → result path.
4. Quality and uncertainty flags.
5. Dashboard polish and presentation.
6. Optional persistence / separate API / extensions only after the end-to-end MVP is stable.

## 12. Current status

All content in this package is a plan and set of implementation requirements. No implementation, training, measured evaluation, deployment, or pilot is asserted to exist. Antigravity should update this status as evidence is produced, including dates, commands, test results, and model metrics.

## 13. Source references

- Competition domain/sub-track: user's `AllDomainList.pdf` from the previous discussion.
- [MIMII official dataset record](https://zenodo.org/records/3384388)
- [MIMII baseline repository](https://github.com/MIMII-hitachi/mimii_baseline)
- [MIMII DUE dataset](https://zenodo.org/records/4740355)
- [MIMII DG dataset](https://zenodo.org/records/6529888)

See `docs/DATASET_AND_MODEL_GUIDE.md` for fuller notes.
