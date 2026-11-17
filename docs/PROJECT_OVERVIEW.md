# Project Overview — PulseForge AI

## 1. Product identity

**Name:** PulseForge AI  
**Tagline:** Hear the Warning. Trust the Signal. Prevent the Downtime.  
**Domain:** IndustrySphere AI  
**Primary sub-track:** Predictive Plant  
**Supporting sub-track:** Factory Brain

PulseForge AI is a trustworthy acoustic predictive-maintenance prototype. It analyzes a recording of an industrial fan, identifies unusual acoustic patterns relative to learned normal operation, evaluates whether the input is suitable/familiar enough to support a result, and gives a human operator a clear, cautious recommendation.

## 2. Vision

Help maintenance personnel notice unusual machine behavior earlier and make better-informed inspection decisions by combining anomaly detection with visible signal-quality and uncertainty information.

## 3. Problem in brief

Industrial machines may exhibit changes in sound or vibration before a breakdown. If the change is not noticed, the machine may continue operating until downtime or damage occurs. Human listening and fixed maintenance schedules can be insufficient for continuous, consistent monitoring. An AI anomaly score alone is also not enough: poor recordings and unfamiliar operating conditions can make a result unreliable.

## 4. Proposed solution

PulseForge AI provides a local web dashboard that accepts a fan WAV recording, processes its acoustic features, runs a model trained on normal examples, and displays:

- normal / possible anomaly / needs review status;
- anomaly score and selected threshold;
- waveform and spectrogram;
- audio-quality indicators;
- unfamiliarity / domain-shift flag where implemented;
- reason(s) for the flag;
- a practical inspection recommendation.

The system does not claim to identify an exact mechanical fault or predict a guaranteed failure date.

## 5. Objectives

1. Build a reproducible audio-processing pipeline for one industrial machine type.
2. Train and evaluate a normal-only Autoencoder anomaly detector.
3. Compare with a simple baseline such as Isolation Forest.
4. Add explicit checks for malformed, poor-quality, and unfamiliar input.
5. Present model output and limitations in a simple dashboard.
6. Demonstrate normal, anomalous, and review-needed cases.
7. Produce evidence-based results and a credible hackathon pitch.

## 6. Target users and stakeholders

| Stakeholder | Need | How PulseForge helps |
|---|---|---|
| Maintenance operator | Know which recording / machine needs attention | Clear status, evidence, and inspection suggestion |
| Maintenance engineer | Understand why a warning appeared | Score, threshold, waveform/spectrogram, quality flags |
| Factory supervisor | See a simple operational status | Dashboard cards and optional history |
| Hackathon judge | Understand the problem and see a functioning product | Repeatable end-to-end demo and measured evaluation |

The first version is designed for a single demo operator, not a deployed multi-role factory environment.

## 7. Core value proposition

**Not just “an anomaly detected,” but “an unusual pattern was detected, here is the evidence, here is whether the input is trustworthy, and here is what a human should check next.”**

This trustworthiness layer is the central differentiator proposed in the project discussion.

## 8. Scope

### MVP

- Fan acoustic recordings only.
- MIMII fan subset.
- WAV upload and validation.
- Log-Mel or equivalent feature extraction.
- Autoencoder and a documented baseline.
- Validation-derived anomaly threshold.
- Signal-quality checks and uncertainty/review flag.
- Streamlit dashboard.
- Reproducible test evaluation and three-case demo.

### Not MVP

- Pump or motor model; real-time sensor stream; industrial controller integration; automatic shutdown; exact fault localization; remaining useful life; public multi-tenant SaaS; account/role management; payments; LLM fault diagnosis.

## 9. Success criteria

The product is successful as a hackathon MVP when:

- it runs from documented setup steps;
- an operator can upload or choose a fan WAV;
- the complete processing pipeline runs without notebook intervention;
- it shows distinct behavior for normal, anomalous, and poor-quality/unfamiliar cases;
- the threshold and model are selected without test-set leakage;
- evaluation metrics and limitations are documented;
- the UI does not confuse anomaly score with failure probability;
- the live demo can run offline with prepared samples.

## 10. Commercial direction (hypothesis, not validated)

Potential later options include a per-machine or per-factory subscription, sensor gateway integration, and enterprise integration with existing monitoring tools. A Bangladesh-first pilot could be explored with one consenting factory and an engineer reviewing alerts in shadow mode. No customer, partner, pilot, or NRB collaboration currently exists as a confirmed fact.

## 11. Expected deliverables

- Local Streamlit application.
- Audio preprocessing and inference modules.
- Autoencoder checkpoint and configuration.
- Baseline comparison.
- Dataset manifest and attribution.
- Evaluation results and visualizations.
- Three prepared demo cases.
- README, model notes, and presentation.
