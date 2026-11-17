# Product and System Requirements

Requirement IDs are stable references for implementation and testing.

## 1. Functional requirements

### MVP — must-have

| ID | Requirement | Priority | Acceptance summary |
|---|---|---|---|
| FR-01 | The app shall support fan acoustic analysis only in the MVP. | Must | UI and API/config reject or clearly mark other machine types unsupported. |
| FR-02 | The user shall upload a WAV recording. | Must | Valid WAV can be selected and decoded. |
| FR-03 | The app shall validate file extension/content, size, duration, and decode success. | Must | Invalid inputs receive a clear message and do not crash. |
| FR-04 | The pipeline shall preprocess audio using the same configuration used in training. | Must | Inference uses saved sample-rate/feature/normalization configuration. |
| FR-05 | The app shall display waveform and spectrogram or equivalent feature view. | Must | Visuals correspond to the uploaded recording. |
| FR-06 | The app shall produce an anomaly score. | Must | Score is finite and documented; higher/lower meaning is clear. |
| FR-07 | The app shall use a threshold selected from validation data. | Must | Threshold is stored with model metadata and not tuned on test data. |
| FR-08 | The app shall display a distinct normal, possible-anomaly, or needs-review status. | Must | Decision policy is deterministic and unit-tested. |
| FR-09 | The app shall check basic signal quality. | Must | At minimum handles silence/near-silence, clipping, and duration. |
| FR-10 | The app shall flag poor-quality or unfamiliar input as needing review. | Must | Warning is visible and not represented as a calibrated probability. |
| FR-11 | The result shall include a short evidence statement and human inspection recommendation. | Must | User sees reason and next step, with no unsupported fault diagnosis. |
| FR-12 | The model shall be evaluated on a held-out test set. | Must | Metrics, split procedure, and limitations are saved. |
| FR-13 | The app shall provide at least three prepared scenarios for demo. | Must | Normal, anomalous, and poor-quality/unfamiliar scenario can be replayed. |
| FR-14 | The app shall display a decision-support disclaimer. | Must | UI states that an engineer/operator must verify the machine. |

### Important, after core flow

| ID | Requirement | Priority | Acceptance summary |
|---|---|---|---|
| FR-15 | The app may persist analysis metadata in SQLite. | Should | History can be viewed; raw audio is not retained by default. |
| FR-16 | The system may expose the analysis service through FastAPI. | Should | API and Streamlit call the same service functions. |
| FR-17 | The app may show a comparison of multiple recordings. | Should | Results can be compared without implying time-series health trends unless supported. |
| FR-18 | The system may provide a more detailed feature explanation. | Should | Explanation is tied to measurable signal features, not invented causes. |

### Future, not MVP

| ID | Requirement | Priority |
|---|---|---|
| FR-19 | Support pump / motor / additional machine-specific models. | Future |
| FR-20 | Accept live sensor or microphone streams. | Future |
| FR-21 | Integrate with factory monitoring systems. | Future |
| FR-22 | Add multi-user authentication and role-based access. | Future |
| FR-23 | Provide a factory fleet monitoring dashboard. | Future |

## 2. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Usability | A first-time demo operator should be able to run an analysis with minimal instructions. |
| NFR-02 | Reliability | Invalid input or missing model artifacts must produce actionable messages, not an unhandled crash. |
| NFR-03 | Reproducibility | Dataset manifest, split, preprocessing config, threshold, seed, and model version must be recorded. |
| NFR-04 | Performance | A representative short recording should complete inference within a practical laptop-demo time; measure and record actual latency. |
| NFR-05 | Safety | Outputs must not claim confirmed fault diagnosis, failure probability, or exact failure date. |
| NFR-06 | Privacy | Uploaded audio should be processed locally and discarded after analysis by default. |
| NFR-07 | Security | Restrict file types, file sizes, duration, temp-file paths, and public exposure. |
| NFR-08 | Maintainability | Audio processing, model inference, reliability, decision logic, and UI should be separate modules. |
| NFR-09 | Portability | Setup must be documented for Linux and Windows. |
| NFR-10 | Demonstrability | The app must work offline with preloaded demo audio and model artifacts. |
| NFR-11 | Accessibility | Status must use text and labels, not color alone; controls and error messages should be clear. |
| NFR-12 | Scalability | Additional machine types must use separate validated model configurations, not reuse fan weights blindly. |

## 3. User roles and access

MVP has one unauthenticated local demo operator. No user registration, password reset, or role system is required. If deployed to a public network or used by multiple factories, authentication, authorization, audit policy, and data governance become prerequisites and must be approved before implementation.

## 4. Input constraints (initial defaults to confirm during implementation)

- Accepted format: WAV only.
- Supported machine type: fan.
- Maximum upload size: configurable; choose a conservative demo limit, e.g. 25 MB, then test.
- Minimum duration: configurable and aligned to model segmentation; do not invent a value before inspecting training clips.
- Sample rate: normalize to the model's configured rate; MIMII source audio is described as 16 kHz in dataset references, but inference configuration must be verified from the exact selected files.
- Audio channels: inspect and define a consistent channel-selection/downmix strategy.

These defaults are implementation proposals, not fixed user-specified constraints.

## 5. Acceptance gates

The MVP cannot be considered complete based on a mocked dashboard alone. At least one real model pipeline using the selected MIMII subset must run end to end, and test-set evaluation must be documented. If the model is not reliable enough, the app must show that limitation instead of hiding it.
