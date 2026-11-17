# AGENTS.md — Instructions for Google Antigravity

You are the implementation agent for **PulseForge AI**. Read `PROJECT_CONTEXT.md` first, then the relevant files under `docs/`. This repository currently contains a handover package, not an existing application. Do not assume source code, data, model weights, test results, or deployed services already exist.

## 1. Mission and priority

Build a reliable, demonstrable hackathon MVP for fan acoustic anomaly detection.

The required end-to-end path is:

`WAV upload → validation → preprocessing/features → anomaly model → quality/unfamiliarity checks → decision policy → dashboard result`

Optimize for a working, understandable demo, reproducibility, and honest communication—not feature count or architectural complexity.

## 2. Non-negotiable product decisions

- Project name: PulseForge AI.
- Domain: IndustrySphere AI.
- Primary sub-track: Predictive Plant.
- Supporting sub-track: Factory Brain (limited dashboard and alert workflow).
- MVP machine type: fan only.
- Primary dataset: MIMII fan recordings.
- Primary detector: Autoencoder trained primarily on normal recordings; reconstruction error is an anomaly score.
- Baseline: Isolation Forest or another documented simple baseline.
- UI: Streamlit first.
- FastAPI: optional adapter after the local application pipeline works or if integration genuinely requires it.
- Persistence: no database required for initial MVP; SQLite metadata history is optional.
- No login for local, single-operator MVP.
- No live hardware, automatic machine control, LLM diagnosis, exact failure-date prediction, or multi-machine support in MVP.

Do not silently change these decisions. Propose deviations in a short decision note and explain the trade-off before implementing them.

## 3. Work method

1. Inspect the existing repository and environment before creating files.
2. Read `docs/DECISIONS_AND_OPEN_QUESTIONS.md` and record answers to blocking questions.
3. Work milestone-by-milestone using `docs/IMPLEMENTATION_PLAN.md`.
4. Do not advance to a later milestone until the current acceptance criteria pass or a blocker is explicitly documented.
5. Keep changes small and runnable. After each meaningful change, run the relevant tests.
6. Prefer a small number of clear modules over microservices or elaborate abstractions.
7. Keep a concise development log: date, changed files, commands, test results, known limitations.
8. Update documentation when implementation decisions change. Keep documentation and code consistent.

## 4. Coding conventions

- Python 3.11 is the recommended starting runtime; verify dependencies on the actual environment.
- Use type hints for public functions and meaningful domain objects.
- Use descriptive names and short functions with one responsibility.
- Add docstrings for non-obvious data/model/decision logic.
- Avoid global mutable state.
- Keep configuration (sample rate, feature settings, thresholds, upload limits) in one documented configuration module or JSON/YAML config, not duplicated across files.
- Use `pathlib` for paths.
- Use structured logging via Python `logging`; never rely only on `print` for application diagnostics.
- Catch expected user-input errors and show actionable messages. Do not swallow unexpected exceptions.
- Do not include secrets, local absolute paths, large datasets, audio uploads, or model binaries in Git unless explicitly approved.
- Keep UI code thin; call reusable application/service functions.

## 5. ML and data rules

- Inspect the exact MIMII archive structure before writing a dataset loader.
- Do not assume a specific folder layout or filename convention without confirming it from the downloaded archive.
- Begin with one fan subset and record the exact archive, noise condition, machine/product identifier, number of files, and labels used.
- Maintain a reproducible manifest for every recording used.
- Split by recording identity / machine section / session where possible. Avoid putting near-duplicate clips or segments from the same source recording in both train and test.
- Train the primary Autoencoder on normal recordings for the initial anomaly-detection experiment.
- Derive the anomaly threshold from validation data, never by optimizing against the final test set.
- Keep preprocessing identical at train and inference time. Save feature configuration and normalization statistics with the model.
- Compare with a baseline using the same data split.
- Report metrics only from executed evaluation code. Include the split definition and limitations.
- Treat anomaly score, uncertainty, signal quality, and decision status as separate concepts.
- Do not call anomaly score a probability or show a confidence percentage unless calibration is implemented and validated.
- A domain-shift / unfamiliarity score is a warning signal, not proof that the model is wrong.

## 6. Decision logic and user-facing language

Use explicit states, for example:

- `normal`: input passed quality checks and score is below the selected threshold.
- `possible_anomaly`: input passed quality checks and score exceeds threshold.
- `needs_review`: quality is poor, input is unsupported, or unfamiliarity is high enough that an automatic normal/anomaly conclusion should be withheld.
- `invalid_input`: file cannot be safely decoded or does not meet minimum requirements.

If both anomaly score and uncertainty are high, show a possible anomaly with a prominent review warning, or `needs_review` according to the documented decision policy. Do not hide either signal.

Preferred language:
- “Unusual acoustic pattern detected.”
- “Recording quality is poor; result may be unreliable.”
- “Input differs from the training conditions; human review is recommended.”
- “Please inspect the machine and verify the recording.”

Forbidden without strong evidence:
- “The machine will fail in X days.”
- “Bearing failure confirmed.”
- “95% probability of failure.”
- “Maintenance guaranteed.”

## 7. Security and privacy

- Accept only explicitly supported audio formats (WAV for MVP).
- Enforce file-size and duration limits before expensive processing.
- Validate actual decoded file content; do not trust filename or MIME header alone.
- Use generated temporary names; never concatenate user filenames into file paths.
- Remove temporary uploads after analysis unless the user explicitly exports/saves them.
- Do not execute uploaded files or parse arbitrary archive uploads in the app.
- Do not expose the app publicly without considering authentication, rate limiting, resource limits, and secure transport.
- Do not store raw audio by default. If optional history is added, persist metadata only and document retention.

## 8. Testing expectations

- Add unit tests for validation, feature extraction shape/finite values, thresholding, quality checks, decision policy, and serialization.
- Add integration tests for a valid WAV through the complete pipeline.
- Test malformed, empty, oversized, silent, clipped, and too-short inputs.
- Test normal, anomalous, and unfamiliar/noisy examples.
- Test missing/corrupt model artifacts and clear recovery messages.
- Tests must not require downloading the full MIMII dataset. Use small synthetic fixtures or a tiny explicitly licensed sample for CI; full-data tests may be marked separately.
- Before each handover, report tests run and whether they passed.

## 9. UX requirements

- Keep the app readable and usable on a laptop screen.
- Show the model/machine type and evaluation limitations.
- Display score and threshold together; do not rely on color alone.
- Distinguish status, signal quality, and uncertainty visually and in text.
- Show waveform/spectrogram where useful.
- Include an inspection recommendation and a human-review disclaimer.
- Provide preloaded demo examples so the live presentation does not depend on a network download.
- Handle app errors with actionable messages rather than a blank page or raw traceback.

## 10. Do not overbuild

Do not add accounts, roles, PostgreSQL, cloud storage, Docker, Kubernetes, live sensors, agent workflows, LLM/RAG, microservices, payment, or mobile apps unless the project owner explicitly approves and the MVP is already stable. Local SQLite is the only proposed persistence option, and it is optional.

## 11. Completion definition

The MVP is complete only when:

1. A clean environment can be installed using documented steps.
2. A user can launch Streamlit locally.
3. A valid fan WAV can be uploaded or selected from a bundled sample.
4. The entire pipeline produces a result without manual notebook execution.
5. A normal, anomalous, and poor-quality/unfamiliar scenario produce the expected distinct behavior.
6. Test-set evaluation and threshold selection are reproducible and documented.
7. No unsupported fault or probability claims appear in the UI or pitch.
8. The README, model card/metadata, dataset attribution, and demo guide are updated.

## 12. Output format for agent reports

At the end of each milestone, report:

- Milestone and status: complete / partial / blocked.
- Files created or changed.
- Commands executed.
- Tests and measured results.
- Dataset subset and model artifact details, if relevant.
- Known issues and limitations.
- Next milestone and any decision needed from the project owner.
