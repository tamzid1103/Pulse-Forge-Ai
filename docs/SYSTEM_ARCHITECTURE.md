# System Architecture

## 1. Architecture decision

Use a **modular Python monolith with Streamlit** for the hackathon MVP. It keeps the data path, model inference, and UI in one environment and avoids unnecessary service deployment work. Keep core logic independent of Streamlit so FastAPI can call the same application service later.

Do not start with microservices, PostgreSQL, cloud storage, Kubernetes, or a separate React frontend. The MVP has one operator and local audio uploads.

## 2. Logical architecture

```mermaid
flowchart TD
    U[Maintenance Operator / Demo User] --> UI[Streamlit Dashboard]
    UI --> UP[Upload and Input Validation]
    UP -->|Valid WAV| APP[Analysis Application Service]
    UP -->|Invalid| ERR[Actionable Validation Message]
    APP --> PRE[Audio Preprocessing]
    PRE --> FE[Feature Extraction: Log-Mel]
    FE --> MODEL[Fan Anomaly Model: Autoencoder]
    FE --> BASE[Optional Baseline: Isolation Forest]
    PRE --> Q[Signal Quality Checks]
    FE --> SHIFT[Input Familiarity / Domain-Shift Check]
    MODEL --> DEC[Decision Policy]
    Q --> DEC
    SHIFT --> DEC
    DEC --> RESULT[Analysis Result + Evidence + Recommendation]
    RESULT --> UI
    RESULT -. optional metadata .-> DB[(SQLite Analysis Metadata)]
    ART[Model Checkpoint + Config + Threshold] --> MODEL
    ART --> SHIFT
```

## 3. Component responsibilities

| Component | Responsibility | Input | Output |
|---|---|---|---|
| Streamlit dashboard | Collect machine selection and WAV; present progress and results | User actions | Application service calls and rendered result |
| Upload validator | Validate size, extension, format, decode, duration, channel count | Uploaded bytes / file handle | Safe audio object or validation error |
| Audio I/O | Decode using SoundFile/Librosa, retain sample rate/channels metadata | WAV | Waveform array and metadata |
| Preprocessor | Normalize sample rate, channels, segment length, and scaling consistently | Waveform | Model-ready signal |
| Feature extractor | Compute log-Mel or selected feature representation | Preprocessed signal | Feature tensor / feature summary |
| Autoencoder | Reconstruct normal acoustic features | Feature tensor | Reconstruction and anomaly score |
| Baseline | Provide benchmark detector | Aggregated features | Baseline score / prediction |
| Quality checker | Evaluate duration, silence, clipping, and configurable checks | Raw/preprocessed signal | Quality status and reasons |
| Familiarity checker | Compare feature / embedding representation with training distribution | Features or embedding | Familiarity flag / distance |
| Decision policy | Combine model score with quality and familiarity flags | Scores and flags | `normal`, `possible_anomaly`, `needs_review`, or `invalid_input` |
| Explanation builder | Convert measurable signals into cautious text | Result fields | Evidence summary and recommendation |
| Persistence (optional) | Store analysis metadata only | Result metadata | History record ID |

## 4. Application workflow

```mermaid
flowchart TD
    A[Open PulseForge] --> B[Select Fan]
    B --> C[Upload WAV or Choose Demo Sample]
    C --> D{File Valid?}
    D -- No --> E[Show Error and Stop]
    D -- Yes --> F[Load Audio]
    F --> G[Quality Checks]
    G --> H[Preprocess and Extract Features]
    H --> I[Run Autoencoder]
    H --> J[Check Familiarity / Shift]
    I --> K[Apply Validation Threshold]
    J --> L[Build Reliability Flags]
    G --> L
    K --> M[Decision Policy]
    L --> M
    M --> N[Render Status, Score, Plots, Reasons, Recommendation]
    N --> O[Human Reviews Machine / Recording]
```

## 5. Decision policy

Use separate values for anomaly detection and reliability. A proposed initial status policy:

1. **Invalid input:** unable to safely decode or fails mandatory file constraints. Stop before inference.
2. **Needs review:** recording fails an essential quality check or is sufficiently unfamiliar that the model result should not be interpreted as a reliable normal/anomaly decision.
3. **Possible anomaly:** input passes quality checks and anomaly score exceeds the validation-derived threshold. If there is also a mild unfamiliarity flag, show both the anomaly and the review warning prominently.
4. **Normal:** input passes quality checks, no severe unfamiliarity flag is present, and score is below threshold.

Exact thresholds and precedence must be written in tests and model metadata. Do not blend signal quality into the anomaly score without a separately validated method.

## 6. Data flow

```mermaid
flowchart LR
    RAW[MIMII WAV or User WAV] --> V[Validation]
    V --> DEC[Decode]
    DEC --> SIG[Waveform]
    SIG --> QC[Quality Metrics]
    SIG --> PRE[Preprocessing]
    PRE --> FEAT[Log-Mel Features]
    FEAT --> MODEL[Autoencoder]
    FEAT --> OOD[Unfamiliarity Check]
    MODEL --> SCORE[Anomaly Score]
    SCORE --> TH[Validation Threshold]
    QC --> POLICY[Decision Policy]
    OOD --> POLICY
    TH --> POLICY
    POLICY --> OUT[Status + Evidence + Recommendation]
    OUT --> DASH[Streamlit UI]
    OUT -. optional .-> HIST[(SQLite Metadata)]
```

## 7. Data / model lifecycle

### Offline training path

1. Inspect and inventory selected MIMII fan files.
2. Build a CSV manifest with file ID, relative path, machine/model section, label, noise condition, recording group, and split.
3. Create leakage-aware train / validation / test partitions.
4. Fit feature normalization on training data only.
5. Train Autoencoder on normal training samples.
6. Select anomaly threshold using validation data and a declared operating criterion.
7. Fit or configure unfamiliarity detector using training/validation data only.
8. Evaluate once on held-out test data.
9. Save checkpoint, preprocessing config, threshold, training metadata, and metrics.

### Online inference path

1. Load validated model artifact/config.
2. Decode and validate uploaded audio.
3. Apply exactly the same preprocessing and feature transformation as training.
4. Compute anomaly score and independent reliability flags.
5. Apply decision policy.
6. Render result and optionally persist metadata.

## 8. Component interaction contract

The core application service should expose one reusable function, conceptually:

`analyze_audio(audio_bytes, filename, model_bundle, config) -> AnalysisResult`

It should not depend on Streamlit or FastAPI. It should:

- validate and decode;
- preprocess and extract features;
- run detector(s);
- calculate quality / familiarity flags;
- apply decision rules;
- return a typed result or a typed validation error.

Streamlit calls this directly in the MVP. FastAPI, if added, calls the same service.

## 9. Deployment architecture

### MVP

- Local laptop or hackathon workstation.
- Streamlit process.
- Local model artifact and configuration.
- Local demo audio and dataset (not served for download).
- No required internet access after setup.

### Optional later deployment

A small server can host Streamlit or FastAPI behind HTTPS and authentication. Before public access, introduce access control, request/file limits, secure temporary storage, logging policy, and retention controls. No production deployment is claimed by this package.

## 10. Scalability and performance notes

- Use chunked / windowed inference if recordings exceed model input length.
- Batch segments carefully to avoid memory spikes.
- Limit file size and duration.
- Cache model loading and immutable assets, not user-specific results across users.
- Keep CPU fallback working; GPU is optional.
- Measure processing latency on the actual demo machine rather than promising a fixed time.
- For additional machine types, train and validate a distinct model/config per type or supported machine family. Do not reuse fan thresholds blindly.

## 11. Logging and error handling

Log timestamps, app version, model version, stage, duration, and sanitized error categories. Avoid logging raw audio, secrets, or unnecessary personally identifiable information.

Expected error categories:

- unsupported file type;
- oversized file;
- corrupt / undecodable WAV;
- too short / silent / clipped input;
- model artifact missing or incompatible;
- feature extraction failure;
- inference failure;
- optional persistence failure.

Show a helpful message to the user and preserve technical stack traces in local developer logs only.
