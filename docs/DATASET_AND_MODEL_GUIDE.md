# Dataset and Model Implementation Guide

## 1. Primary dataset: MIMII

Use the official [MIMII dataset record on Zenodo](https://zenodo.org/records/3384388). The dataset was released for investigation/inspection of malfunctioning industrial machine sounds. Its documented machine types include fans, pumps, valves, and slide rails; each type includes product models and normal/anomalous operating sounds. The dataset description identifies example abnormal conditions such as contamination, leakage, rotating unbalance, and rail damage.

The MIMII record lists a large total download size, and the individual fan archives can be around 10 GB depending on the recorded noise condition. Do not assume the complete dataset is practical to download during a short hackathon.

### Required initial selection

- Machine type: fan.
- Start with one selected archive/noise condition and one product/model section.
- Inspect the exact file structure after download.
- Record the exact archive filename, file size, source URL, checksum if available, machine/model section, noise condition, and license/attribution details.
- Use a manageable subset for the first end-to-end pipeline.
- Do not claim the model supports pumps or motors merely because they are included in the wider dataset.

### Audit findings and selected MVP subset (Recorded: 2026-11-17)

| Attribute | Verified Value |
|---|---|
| Dataset Name | MIMII (Malfunctioning Industrial Machine Investigation and Inspection) |
| Official Record | Zenodo Record 3384388 (DOI: 10.5281/zenodo.3384388) |
| Selected Archive | `6_dB_fan.zip` (Cleanest noise condition: +6 dB SNR) |
| Archive Size | 10,158,673,161 bytes (~9.68 GB) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Machine Type | Fan (`fan`) |
| Machine Model / ID | `id_00` (First primary fan model) |
| Folder Layout in Zip | `fan/{machine_id}/{condition}/{filename}.wav` (e.g. `fan/id_00/normal/00000000.wav`) |
| Total Files in Archive | 5,563 total entries across 4 machine IDs (`id_00`, `id_02`, `id_04`, `id_06`) |
| `id_00` File Counts | 1,011 normal files, 407 abnormal files |
| Selected MVP Subset | 50 normal + 20 abnormal = 70 files (~175 MB) |
| Audio Format | WAV (WAVEX header, 16-bit PCM) |
| Sample Rate | 16,000 Hz |
| Channels | 8 channels (circular microphone array) |
| Duration | Exactly 10.0 seconds per clip (160,000 samples per channel) |

### Dataset sources

- [MIMII public 1.0 — Zenodo](https://zenodo.org/records/3384388)
- [Official MIMII baseline repository](https://github.com/MIMII-hitachi/mimii_baseline)
- [MIMII paper on arXiv](https://arxiv.org/abs/1909.09347)

The official baseline repository is useful for understanding the dataset conventions and a reference anomaly-detection workflow. It is a reference, not a requirement to copy its implementation.

## 2. Optional datasets for later work

### MIMII DUE

[MIMII DUE on Zenodo](https://zenodo.org/records/4740355) was designed to investigate anomaly detection under domain shifts related to operational and environmental condition changes. It can be useful for a later experiment showing that a model may encounter conditions different from those seen in training.

### MIMII DG

[MIMII DG on Zenodo](https://zenodo.org/records/6529888) is intended for a domain-generalization task and includes a fan subset. It may be useful for an optional unfamiliarity / robustness evaluation.

These are extensions, not prerequisites for the first MVP. Do not switch to them as primary data without recording a decision and explaining the impact.

## 3. Data inspection checklist

Before model work, inspect:

- archive file listing and folder nesting;
- file extension and decode compatibility;
- sample rate, bit depth, channel count;
- recording duration;
- machine / model / section identifiers;
- normal and anomalous labels;
- noise condition / recording condition;
- number of files and total extracted size;
- duplicate or near-duplicate recordings.

Create a dataset manifest such as `data/manifests/fan_manifest.csv` with at least:

`recording_id, relative_path, machine_type, machine_id_or_section, label, noise_condition, duration_s, sample_rate_hz, channels, group_id, split`

The manifest should never contain absolute paths specific to one developer's computer. Keep raw audio outside Git.

## 4. Split strategy and leakage prevention

A random split can overestimate performance if fragments from the same recording, machine instance, or closely related recording session appear in both train and test.

- Keep all windows from a source recording in one split.
- Prefer grouping by machine/model/section/session where the dataset structure permits.
- Train the normal-only Autoencoder on normal training samples.
- Use validation normal samples to choose the anomaly threshold and any operating point.
- Keep a held-out test set untouched until model and threshold choices are fixed.
- Report exactly which group IDs / sections were used in each split.
- If the available subset is too small for a credible split, label the result as a prototype experiment and do not overstate generalization.

## 5. Audio preprocessing

The training and inference pipelines must be identical and versioned.

Recommended initial process:

1. Decode WAV using SoundFile or Librosa.
2. Confirm sample rate and channel layout.
3. Define a documented channel strategy: select a fixed microphone channel or downmix consistently. Inspect the original archive before choosing.
4. Convert to the configured sample rate if needed.
5. Apply a consistent amplitude normalization strategy, if used.
6. Segment long audio using a documented window length and hop/overlap.
7. Extract log-Mel spectrogram features.
8. Apply normalization statistics fitted on training data only.
9. Validate tensor shape and finite values.

Avoid silent changes to sample rate, channel selection, segment duration, Mel bins, hop length, or normalization after training. They must be part of the model bundle.

## 6. Primary detector: Autoencoder

### Rationale

The hackathon concept is anomaly detection from normal operation patterns. An Autoencoder can be trained mainly on normal samples and use reconstruction error as a measure of how different a new feature representation is from learned normal patterns.

### Workflow

1. Extract fixed-shape log-Mel segments.
2. Fit preprocessing/normalization on normal training examples.
3. Train an Autoencoder to reconstruct normal features.
4. Calculate reconstruction error per segment.
5. Aggregate segment errors into a recording-level score using a declared aggregation rule (e.g. mean, percentile, or maximum; compare empirically).
6. Select a threshold from validation data.
7. Evaluate on the held-out test set.

### Implementation choices to decide experimentally

- model dimensions / convolutional or dense design;
- segment length and overlap;
- Mel-bin count and hop length;
- loss function;
- optimizer, learning rate, batch size, epochs, early stopping;
- score aggregation;
- threshold selection criterion.

Do not overfit the design to the test set. Record the chosen configuration in `model_config.json` and training logs.

## 7. Baseline: Isolation Forest

Train a simpler anomaly detector on a documented feature vector or pooled representation. Possible inputs include aggregated spectral statistics or flattened/reduced log-Mel features. Choose a dimensionally appropriate representation and keep preprocessing consistent.

Purpose:

- establish a baseline;
- show whether the Autoencoder provides value;
- provide a simpler fallback if the neural model is unstable.

Compare both models on the same split. Do not report the better model without also reporting the comparison method.

## 8. Threshold selection

The anomaly score is model-specific and is not a probability.

- Calculate scores on normal validation data.
- Choose a threshold using a declared criterion, such as a target validation false-positive rate or a selected validation quantile.
- If anomalous validation data is available, it can support selection of an operating point, but keep the final test set untouched.
- Store threshold, score aggregation, validation split ID, and selection rule with the model.
- Explain that different model versions or machine types require their own threshold.

## 9. Reliability and uncertainty checks

The project must not equate high reconstruction error with high confidence. Keep the following independent:

### A. Signal quality

Implement measurable checks, such as:

- duration within configured limits;
- near-silence / low RMS;
- clipping ratio or fraction of samples near digital full scale;
- file decode validity;
- sample rate and channel handling;
- optional signal-to-noise proxy only if a valid method can be defined for the data.

Use thresholds derived from data inspection and representative examples. A generic RMS value should not be treated as a universal quality standard.

### B. Input familiarity / domain shift

A first approach can compare an embedding or normalized feature summary against the distribution of normal training features. For example, use a distance metric and a threshold set from validation data. The exact method should be selected and documented. Do not present the distance as a probability that the model is wrong.

### C. Decision policy

- Poor mandatory quality → `needs_review`.
- Strong unfamiliarity → `needs_review` or prominent uncertainty warning.
- Valid, familiar input with score above threshold → `possible_anomaly`.
- Valid, familiar input with score below threshold → `normal`.

If a possible anomaly is also unfamiliar, do not suppress either fact; show both and recommend human review.

### D. Calibrated confidence is not an MVP requirement

Do not show “95% confidence” unless an explicit calibration method is implemented, evaluated, and documented. The initial product should use plain-language quality and uncertainty flags.

## 10. Evaluation plan

Use held-out data and report:

- AUROC;
- precision, recall, F1 at the selected threshold;
- PR-AUC where class imbalance makes it useful;
- false-positive rate / false alarms per recording;
- inference latency;
- results by relevant machine section/noise condition when sample sizes permit;
- quality/uncertainty flag behavior on poor-quality or shifted cases.

A useful robustness experiment may add or use existing noisy recordings, compare performance across noise conditions, or use a domain-shift dataset later. Clearly distinguish synthetic corruption from naturally recorded noisy samples.

Do not create metric values for a pitch until a script has actually produced them. Save metric JSON/CSV, plots, split manifest, and model configuration.

## 11. Reproducibility artifacts

For each trained model, save:

- model checkpoint;
- preprocessing config;
- feature config;
- threshold config;
- training seed;
- train/validation/test manifest or split IDs;
- dataset archive name/version and citation;
- package versions;
- training history;
- evaluation metrics;
- model version and creation timestamp.

A model card should state supported machine type, training data, intended use, limitations, known failure cases, and evaluation protocol.

## 12. Data rights and attribution

The public Zenodo record provides dataset-specific metadata and a license. Review the current license terms on the source record before redistribution, publishing derived files, or including dataset audio in a public repository. Prefer not to redistribute the raw dataset. Include a citation/attribution in the app, README, and presentation when using the dataset.

Suggested citation from the dataset record:

Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., & Kawaguchi, Y. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*. DCASE 2019 Workshop / Zenodo dataset record.

Official links:

- https://zenodo.org/records/3384388
- https://github.com/MIMII-hitachi/mimii_baseline
- https://arxiv.org/abs/1909.09347
