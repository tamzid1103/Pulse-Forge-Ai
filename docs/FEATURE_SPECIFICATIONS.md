# Detailed Feature Specifications

This document defines the expected behavior of each feature. All items are **planned requirements**, not claims that the feature is already implemented. Priority labels:

- **MVP / Must:** required for a credible end-to-end demonstration.
- **Important / Should:** implement only after the MVP vertical slice works.
- **Optional / Could:** useful if time remains.
- **Future:** requires new data, validation, or scope approval.

## F-01 — Fan machine selection

**Priority:** MVP / Must  
**Purpose:** Make the supported model scope explicit and prevent the user from applying a fan model to other machine types.

**User workflow:** User opens the dashboard and sees `Fan` as the supported machine. Other machine families may be mentioned as planned, but cannot be run through the fan model.

**Frontend requirements:**
- Show a machine selector or fixed machine badge.
- Only enable Fan in the MVP.
- Clearly label Pump / Motor as planned / unsupported if displayed.

**Backend requirements:**
- Validate `machine_type == "fan"` before inference.
- Load the fan-specific model configuration.
- Return an unsupported-machine error for other values.

**Database requirements:** None in MVP. Optional history records `machine_type="fan"`.

**API requirements:** Internal service takes `machine_type`; only `fan` is accepted in MVP. Optional API returns 400/422 for unsupported type.

**Dependencies:** Configuration, model registry/path, dashboard.

**Expected behavior:** The user cannot accidentally interpret fan-model output as valid pump or motor analysis.

**Acceptance criteria:**
- Fan can be selected / is visibly active.
- Unsupported type is rejected before model inference.
- UI copy states the current fan-only limitation.

## F-02 — WAV upload and input validation

**Priority:** MVP / Must  
**Purpose:** Accept a user recording safely and reject unsuitable files before expensive processing.

**User workflow:** User uploads a WAV file or selects a prepared demo sample. The app displays its safe filename/label and basic metadata.

**Frontend requirements:**
- WAV uploader with supported-format note.
- Upload limit and duration note where configured.
- Loading / validation state.
- Clear errors for unsupported or invalid inputs.

**Backend requirements:**
- Inspect decoded content, not extension alone.
- Enforce configurable byte-size and duration limits.
- Decode using a supported audio library.
- Normalize filename for display only; never use it as a filesystem path.
- Use a generated temporary path if required and remove it after analysis.

**Database requirements:** None. Optional metadata history may save `input_label`, `source_kind`, and duration; do not save raw audio by default.

**API requirements:** Core service accepts bytes or a safe validated temporary handle. Optional FastAPI endpoint uses multipart upload and documented 400/413/415/422 errors.

**Dependencies:** Audio I/O, configuration, error schema.

**Expected behavior:** Valid WAV continues to preprocessing. Corrupt, unsupported, oversized, empty, or too-short inputs receive an actionable message and do not crash the app.

**Acceptance criteria:**
- Valid test WAV is accepted.
- A renamed non-WAV file is rejected after content validation.
- Empty/corrupt and oversized files are rejected.
- Path traversal characters in filename do not affect storage paths.
- Temporary data is cleaned up.

## F-03 — Audio metadata and visualization

**Priority:** MVP / Must  
**Purpose:** Help users and judges understand the signal being analyzed.

**User workflow:** After a valid upload, the user sees the recording duration, sample rate, channel count, waveform, and spectrogram.

**Frontend requirements:**
- Display decoded metadata.
- Render waveform with time and amplitude labels.
- Render log-Mel spectrogram or the configured feature representation with time/frequency labels.
- Keep plots readable on a laptop / projector.

**Backend requirements:**
- Produce visualization data from the validated recording.
- Ensure visualization and model pipeline refer to the same input.

**Database requirements:** None. Optional metadata can retain duration and sample rate only.

**API requirements:** Internal result can expose plot arrays / summary data to the UI. Do not return large raw arrays through an optional external API unless needed.

**Dependencies:** Librosa, NumPy, Matplotlib or Plotly, validated audio object.

**Expected behavior:** User can visually inspect the selected recording before interpreting the model output.

**Acceptance criteria:**
- Both plots render for a known valid sample.
- Axis labels and titles are present.
- Invalid files do not cause plotting errors.
- The plot is not described as a mechanical diagnosis.

## F-04 — Audio preprocessing and feature extraction

**Priority:** MVP / Must  
**Purpose:** Convert raw audio into a consistent model-ready representation.

**User workflow:** Automatic after upload; no advanced settings required from the demo user.

**Frontend requirements:**
- Show a short processing state.
- Advanced preprocessing parameters should not clutter the operator UI.

**Backend requirements:**
- Apply fixed sample-rate policy.
- Apply documented channel selection/downmixing.
- Segment / pad / trim according to training configuration.
- Compute log-Mel spectrogram or chosen feature representation.
- Apply normalization fitted on training data only.
- Validate output shape and finite values.

**Database requirements:** None. Save preprocessing config alongside model artifact, not per-user rows.

**API requirements:** Internal feature service called by training and inference. No separate endpoint required.

**Dependencies:** Librosa, SoundFile, SciPy, NumPy, model config.

**Expected behavior:** Training and inference use the same deterministic transformations.

**Acceptance criteria:**
- Feature shape matches model expectation.
- Same recording/config produces consistent features.
- NaN / infinite values are rejected or handled explicitly.
- Config is versioned with model artifact.

## F-05 — Autoencoder anomaly detection

**Priority:** MVP / Must  
**Purpose:** Learn normal fan acoustic patterns and score recordings that differ from them.

**User workflow:** Runs automatically after feature extraction. User sees an anomaly score and status, not neural-network internals.

**Backend requirements:**
- Train the Autoencoder primarily on normal fan training recordings.
- Save checkpoint and model configuration.
- At inference, reconstruct input features and calculate a documented reconstruction error.
- Aggregate segment-level errors into one recording-level score using a declared method.
- Check checkpoint and input dimensions before inference.

**Frontend requirements:**
- Display score with label “Anomaly score”.
- Display the threshold and score direction.
- Do not label the score as failure probability or confidence.

**Database requirements:** Optional history stores score, threshold, and model version.

**API requirements:** Internal inference returns score, segment summary if used, and model version.

**Dependencies:** PyTorch, feature extractor, saved model artifact/config.

**Expected behavior:** Score indicates how unusual the recording is relative to learned normal patterns. A score above threshold is a possible anomaly, not confirmed failure.

**Acceptance criteria:**
- Model loads from saved artifact in a fresh process.
- Model returns finite score for valid input.
- Score direction is documented.
- Threshold is selected using validation data, not final test labels.
- A held-out evaluation report exists.

## F-06 — Baseline detector and comparison

**Priority:** MVP / Must for technical evidence; UI exposure optional  
**Purpose:** Establish whether the Autoencoder improves over a simpler method.

**User workflow:** Usually not visible in the operator UI. Results appear in technical evaluation / presentation.

**Backend requirements:**
- Implement Isolation Forest or another documented baseline on a suitable feature vector.
- Use the same split as the Autoencoder.
- Select any threshold on validation data.
- Save scores and metrics.

**Frontend requirements:** Optional technical comparison section; not required for the main operator screen.

**Database requirements:** None; save experiment outputs to results files.

**API requirements:** None required.

**Dependencies:** Scikit-learn, manifest, feature pipeline.

**Expected behavior:** Provide a fair, reproducible baseline comparison.

**Acceptance criteria:**
- Baseline trains and scores without error.
- Evaluation uses the same held-out data as primary model.
- Metrics are traceable to code/config/split.

## F-07 — Validation threshold and status classification

**Priority:** MVP / Must  
**Purpose:** Convert a continuous anomaly score into a readable decision state.

**User workflow:** User sees one of: Normal, Possible anomaly, Needs review, or Invalid input.

**Backend requirements:**
- Choose threshold from validation data and document operating criterion.
- Store threshold with model artifact.
- Implement deterministic decision policy.
- Preserve anomaly score even when an uncertainty warning is present.

**Frontend requirements:**
- Show status text and label.
- Show score and threshold together.
- Use color only as a secondary cue.

**Database requirements:** Optional history stores score, threshold, status.

**API requirements:** Result field `decision_status` uses the documented enum.

**Dependencies:** Model score, threshold config, quality and unfamiliarity outputs.

**Expected behavior:** A familiar, valid recording below threshold is normal; above threshold is possible anomaly; poor/unfamiliar recordings require review; invalid files are rejected.

**Acceptance criteria:**
- Threshold boundary tests pass.
- All status branches have unit tests.
- Status and warning copy are consistent between API and UI.

## F-08 — Signal-quality assessment

**Priority:** MVP / Must  
**Purpose:** Prevent obviously unsuitable recordings from creating misleading automatic results.

**User workflow:** User sees Good / Warning / Poor plus plain-language reasons.

**Backend requirements:**
- Check successful decode.
- Check duration against model requirements.
- Detect silence / near-silence.
- Detect clipping / excessive full-scale samples.
- Optionally add other quality metrics only if defined and tested.
- Keep checks independent from anomaly score.

**Frontend requirements:**
- Dedicated signal-quality panel.
- Show a reason such as “recording is nearly silent” or “audio clipping detected”.
- For poor quality, recommend re-recording or checking microphone setup.

**Database requirements:** Optional quality status and reason list.

**API requirements:** `quality.status` and `quality.reasons` in result.

**Dependencies:** Raw waveform, configurable quality thresholds.

**Expected behavior:** Bad quality results in a warning or needs-review status rather than a misleading Normal label.

**Acceptance criteria:**
- Silent and clipped fixtures trigger expected flags.
- Good known sample passes checks.
- Thresholds are configurable and documented.

## F-09 — Unfamiliarity / domain-shift assessment

**Priority:** MVP / Must in basic form; advanced domain-shift research is optional  
**Purpose:** Warn when new audio differs substantially from the conditions represented in training.

**User workflow:** User sees “Within known training conditions” or “Unfamiliar input — human review recommended.”

**Backend requirements:**
- Implement a simple feature/embedding distribution comparison.
- Determine cutoff from training/validation data.
- Return flag, optional distance score, and reason.
- Do not call the score a probability.
- Treat this as an early warning mechanism, not a definitive detector of all distribution shifts.

**Frontend requirements:**
- Separate familiarity panel from anomaly score and quality.
- Explain that unfamiliarity means the model has less evidence for this input condition.

**Database requirements:** Optional unfamiliarity score and reasons.

**API requirements:** `unfamiliarity.flagged`, `score`, and `reasons` fields if implemented.

**Dependencies:** Saved training distribution / embedding config and validation data.

**Expected behavior:** Strongly unfamiliar input triggers a review warning. It must not be silently marked Normal based on a low anomaly score alone.

**Acceptance criteria:**
- Familiar validation examples mostly pass the configured gate.
- A designed unfamiliar test is flagged as expected.
- Method and limitations are described in model notes.

## F-10 — Evidence and inspection recommendation

**Priority:** MVP / Must  
**Purpose:** Translate model outputs into cautious, understandable next steps.

**User workflow:** After analysis, the user reads what was detected, why a warning appeared, and what to check.

**Backend requirements:**
- Build text from measured score/threshold and actual flags.
- Use rule-based templates for the MVP.
- Do not infer specific mechanical components or failure causes without a separate validated model.

**Frontend requirements:**
- Show evidence and one practical recommendation.
- Always show a human-review disclaimer.

**Database requirements:** Optional metadata may store the decision and reason codes; full narrative storage is not necessary.

**API requirements:** `evidence[]` and `recommendation` fields.

**Dependencies:** Decision result, quality and familiarity flags.

**Expected behavior:** Message matches the result. Examples include “reconstruction error is above the selected threshold” and “recording quality is poor; repeat the recording.”

**Acceptance criteria:**
- Normal, anomaly, poor-quality and unfamiliar result strings are distinct.
- No unsupported fault diagnosis / failure date claims appear.

## F-11 — Streamlit dashboard

**Priority:** MVP / Must  
**Purpose:** Provide the end-to-end user interface and Factory Brain supporting layer.

**User workflow:** User opens app, uploads/selects recording, runs analysis, reviews charts and result.

**Frontend requirements:**
- Header / tagline.
- Fan-only selection.
- WAV upload and demo selector.
- Progress / validation states.
- Waveform and spectrogram.
- Score + threshold.
- Quality / familiarity panels.
- Decision status, evidence, recommendation, disclaimer.
- Clear/reset action.

**Backend requirements:**
- Call the shared application service.
- Load model once and cache safely.
- Handle errors and avoid showing stale previous result as current.

**Database requirements:** None in MVP.

**API requirements:** None in MVP; optional FastAPI is separate.

**Dependencies:** Streamlit, Plotly/Matplotlib, service, model artifacts.

**Expected behavior:** One command launches a complete local product demo.

**Acceptance criteria:**
- App runs offline after setup.
- Core workflow completes for all three scenarios.
- UI is readable on laptop/projector.
- No dead controls or fake metrics.

## F-12 — Demo sample selection

**Priority:** MVP / Must for demonstration reliability  
**Purpose:** Make the live demo repeatable and independent of browsing for files.

**User workflow:** User chooses a prepared normal, anomaly, or review-needed recording.

**Frontend requirements:**
- Demo selector with descriptive labels.
- Clearly distinguish demo samples from uploaded data.

**Backend requirements:**
- Load only files from a configured demo directory.
- Validate demo files through the same pipeline as uploads.

**Database requirements:** Optional `source_kind=demo_sample`.

**API requirements:** None required.

**Dependencies:** Small legally usable demo audio files and manifest.

**Expected behavior:** The demo can run without network access or live dataset access.

**Acceptance criteria:**
- Three scenarios load consistently.
- Demo labels correspond to source ground truth / documented reason.
- Files are legally included or remain local and not redistributed.

## F-13 — Optional analysis history

**Priority:** Important / Should  
**Purpose:** Let the user review recent analyses in the same local demo session.

**User workflow:** User opens a history list and selects a previous result.

**Frontend requirements:**
- Recent history list with time, status, and score.
- Clearly indicate if history is disabled.

**Backend requirements:**
- Persist metadata only.
- Do not persist raw audio by default.
- Handle database unavailability without corrupting an analysis result.

**Database requirements:** Optional SQLite `analysis_runs` table in `DATABASE_SCHEMA.md`.

**API requirements:** None required; optional GET history endpoint only after explicit approval.

**Dependencies:** SQLite, persistence module.

**Expected behavior:** History survives local app restarts without storing audio.

**Acceptance criteria:**
- Records can be inserted/read.
- Schema initializes automatically.
- Database failure leaves analysis functional.
- History can be disabled using configuration.

## F-14 — Optional FastAPI service

**Priority:** Important / Should only if needed  
**Purpose:** Expose analysis to another frontend/client without duplicating model logic.

**User workflow:** Not visible to the Streamlit user; consumed by a separate client or test script.

**Frontend requirements:** None.

**Backend requirements:**
- `GET /health`.
- `POST /api/v1/analyze` with multipart WAV.
- Use the same application service and decision policy as Streamlit.
- Return documented error schema and status codes.

**Database requirements:** None required.

**API requirements:** Defined in `API_DOCUMENTATION.md`.

**Dependencies:** FastAPI, Uvicorn, python-multipart.

**Expected behavior:** HTTP client gets the same result semantics as Streamlit.

**Acceptance criteria:** API integration tests pass; upload limits and errors are handled; no duplicate model code.

## F-15 — Machine-family expansion

**Priority:** Future  
**Purpose:** Extend the product to pump, motor, or other machines.

**User workflow:** User selects a machine type that has its own validated model and configuration.

**Frontend requirements:** Enable only machine types with complete model/data support.

**Backend requirements:** Machine-specific model/config registry, preprocessing compatibility checks, separate threshold and evaluation evidence.

**Database requirements:** Future schema may store supported model family/version.

**API requirements:** Machine type must map to an available validated model; otherwise reject.

**Dependencies:** New datasets, model training, test protocol, domain expertise.

**Expected behavior:** Each new machine type is independently validated; fan model is never reused blindly.

**Acceptance criteria:** A new machine type is enabled only after independent held-out evaluation and documented limitations.
