# Testing and Quality Assurance Plan

## 1. QA principles

- Test the pipeline in layers, then as one user journey.
- Use small synthetic fixtures for CI; full MIMII tests are separate because the dataset is large.
- Test normal, anomalous, and uncertain / poor-quality cases.
- Record thresholds, model version, data split, and environment with every model evaluation.
- Do not accept a dashboard that displays hardcoded demo predictions as a functioning AI system.

## 2. Unit testing

| Area | Test | Expected result |
|---|---|---|
| Audio validation | Valid WAV | Accepted and decoded |
| Audio validation | Wrong file content / renamed extension | Rejected with unsupported/invalid message |
| Audio validation | Oversized file | Rejected before expensive processing |
| Audio validation | Empty/corrupt file | Clear validation error |
| Audio validation | Too-short audio | Needs review or validation rejection, per configured rule |
| Quality | Silent / near-silent audio | Poor quality flag |
| Quality | Clipped audio | Clipping warning or poor quality flag |
| Preprocessing | Known sample rate / channel input | Consistent configured output shape |
| Features | Valid waveform | Correct dimensions and all finite values |
| Model | Checkpoint load/save | Round-trip works |
| Model | Valid feature tensor | Finite score and reconstructed output |
| Thresholding | Score below/above threshold | Expected binary comparison |
| Decision | Quality failure | `needs_review` or `invalid_input` as specified |
| Decision | Familiar score below threshold | `normal` |
| Decision | Familiar score above threshold | `possible_anomaly` |
| Decision | Unfamiliar sample | Review warning / `needs_review` |
| Result schema | Serialization | API/UI result contains required fields |

## 3. Dataset and ML evaluation

### Data checks

- Every manifest path resolves.
- Every file decodes.
- Labels and machine identifiers are valid.
- Split membership is reproducible.
- No source recording is divided across train and test.
- No test data influences normalization, threshold selection, or model tuning.

### Model checks

- Autoencoder trains on normal training samples as specified.
- Validation loss is tracked.
- Model artifact and preprocessing configuration are saved together.
- Baseline and main model use the same evaluation split.
- The held-out test set is used only after design/threshold selection.

### Metrics

Report at least:

- AUROC;
- precision, recall, F1 at the selected threshold;
- confusion matrix;
- false-positive rate;
- inference latency on the target demo machine.

Where useful, add PR-AUC and breakdown by condition/section. Include class counts and split protocol next to metrics.

## 4. Reliability testing

| Scenario | Expected behavior |
|---|---|
| Clean normal fan recording | Normal if score is below threshold and familiarity checks pass |
| Clean anomalous fan recording | Possible anomaly if score exceeds threshold |
| Silent recording | Poor quality / needs review; do not call it normal |
| Clipped recording | Warning / needs review if clipping exceeds configured limit |
| Unsupported/corrupt audio | Invalid input; inference not run |
| Very short recording | Rejected or needs review according to model input requirement |
| Unfamiliar operating condition | Unfamiliarity flag and human review |
| Anomaly + unfamiliarity | Show both facts and require human review; do not suppress anomaly score |
| Missing model weights | Clear model unavailable message |
| Corrupt model config | Fail safely with developer-facing diagnostics |

## 5. Integration testing

Test the service pipeline:

`bytes → validation → decode → preprocessing → features → model → quality/familiarity → decision → result`

Verify:

- the same preprocessing configuration is used in training and inference;
- score and threshold appear in the returned result;
- decision status matches the explicit policy;
- evidence/recommendation fields are non-empty and grounded in measured flags;
- temporary files are cleaned up;
- optional database failure does not corrupt the model result.

## 6. API testing (only if FastAPI is implemented)

- `GET /health` reports whether the model is loaded.
- Valid multipart WAV returns the documented result schema.
- Invalid file returns 400/415 as appropriate.
- Oversized file returns 413.
- Invalid duration returns 422 or a structured `needs_review` result according to the chosen contract.
- Missing model returns 503.
- No stack trace or server path leaks in client response.
- Requests are bounded by file size, processing time, and concurrency settings.

## 7. User workflow tests

1. Start the app from a clean environment.
2. Confirm the model-ready state.
3. Load a normal demo sample.
4. Confirm normal status and visualizations.
5. Load an anomalous demo sample.
6. Confirm possible-anomaly status and evidence.
7. Load a poor-quality/unfamiliar sample.
8. Confirm review warning and reason.
9. Upload a malformed file and confirm an actionable error.
10. Clear result and repeat without stale state.
11. Disconnect internet and repeat all core demos.

## 8. Security and privacy tests

- Verify file type by decoding, not only extension.
- Verify upload-size and duration limits.
- Try filenames with path separators and unusual characters.
- Ensure no user filename becomes a file path.
- Verify temp files are removed after processing.
- Verify audio is not written to history/logs by default.
- Verify `.env`, dataset, model weights, databases, and uploads are ignored by Git.
- If exposed outside localhost, do not proceed without an approved authentication / TLS / rate-limit plan.

## 9. Performance tests

On the actual demo laptop, record:

- app cold-start time;
- model load time;
- feature extraction time;
- model inference time;
- end-to-end time for the chosen demo clip;
- peak memory use if practical.

Do not publish a speed claim without a recorded device/environment. If latency is too high, reduce segment batch size, cache the model, or use a smaller model while preserving evaluation correctness.

## 10. Regression gate before demo freeze

All of the following must pass:

- unit tests;
- one valid end-to-end normal case;
- one valid end-to-end anomaly case;
- one needs-review case;
- one invalid-input case;
- model loads after app restart;
- no required internet access at demo time;
- no unsubstantiated claims in UI or slides.

## 11. Test report template

```text
Date:
Git commit:
OS / Python version:
CPU / GPU:
Dataset archive and subset:
Train / validation / test grouping:
Model version:
Threshold selection rule:
Tests executed:
Unit tests:
Integration tests:
Demo workflow:
AUROC / Precision / Recall / F1:
False-positive rate:
Latency:
Known limitations:
Pass / fail decision:
```
