# Problem Statement and Use Cases

## 1. Context

Industrial fans and other rotating machines operate under changing loads and environmental conditions. An acoustic recording can contain information about machine behavior. Changes in acoustic patterns may indicate that operation has become unusual, but a recording can also be affected by background noise, microphone placement, recording quality, or operating-condition changes.

## 2. Problem statement

Maintenance teams need a practical way to flag potentially unusual machine sound recordings before a problem becomes an unexpected interruption. A basic anomaly detector may raise a warning but can be misleading when the recording is poor or the operating condition differs from the training data. A usable system should therefore expose evidence and communicate when its assessment should not be trusted without human review.

## 3. Pain points

- A developing abnormal condition may not be obvious to a person during routine work.
- Periodic checks do not continuously inspect every machine.
- Unexpected downtime can interrupt production and create repair costs.
- Background noise can make acoustic recordings difficult to interpret.
- A model can encounter unfamiliar conditions not represented in training data.
- A bare anomaly label does not tell an operator whether the input is usable or what to inspect.

These are the motivation for the prototype, not claims that PulseForge has already reduced downtime.

## 4. Target users

### Primary: maintenance operator

Uploads/selects a recording and needs to know whether it appears normal, unusual, or inconclusive.

### Secondary: maintenance engineer

Reviews the visual evidence and decides whether to inspect the physical machine, repeat the recording, or disregard an invalid input.

### Tertiary: supervisor / judge

Needs an at-a-glance operational view and a clear explanation of the product's limits.

## 5. Primary use cases

### UC-01 — Analyze a fan recording

1. User opens the dashboard.
2. User selects the supported machine type, fan.
3. User uploads a WAV recording or selects a bundled sample.
4. App validates and loads the audio.
5. App extracts the configured features and runs the model.
6. App displays status, score, threshold, waveform/spectrogram, quality, uncertainty flags, and recommendation.

**Expected outcome:** a clear, reproducible assessment with no unsupported diagnosis.

### UC-02 — Reject invalid audio

1. User uploads an unsupported, corrupt, empty, too-short, or oversized file.
2. App validates the file.
3. App rejects it with a clear reason and does not run inference.

**Expected outcome:** actionable error message, no crash.

### UC-03 — Flag poor-quality recording

1. User uploads a decodable recording.
2. Quality checks detect silence, clipping, too-short duration, or another configured problem.
3. App displays “Needs review” / “Recording quality poor” and explains the reason.

**Expected outcome:** system avoids an overconfident result.

### UC-04 — Flag unfamiliar recording

1. User uploads a technically valid recording.
2. Model inference runs.
3. Unfamiliarity check identifies a significant difference from training conditions.
4. App shows an uncertainty / human-review warning alongside the anomaly score.

**Expected outcome:** the model's limitation is visible. An unfamiliarity flag is not treated as proof of malfunction.

### UC-05 — Review an analysis

User reads the result, inspects the visual evidence, and follows the recommendation to check the machine or obtain a better recording.

**Expected outcome:** human remains responsible for maintenance decisions.

## 6. Out of scope

- Autonomous diagnosis of a specific mechanical component.
- Remaining-useful-life or exact failure-time prediction.
- Direct live sensor ingestion or PLC/SCADA control.
- Automatic machine shutdown.
- Guaranteed prevention of downtime.
- Multi-machine fleet monitoring in the MVP.
- Real factory deployment claims.

## 7. Impact hypothesis

If validated in a real operational environment, earlier identification of unusual acoustic patterns may help maintenance teams prioritize inspections and reduce some avoidable downtime. A factory pilot and expert assessment would be required to test this hypothesis. The hackathon prototype can demonstrate technical feasibility on public data, not proven industrial impact.

## 8. Core problem-to-solution mapping

| Problem | Proposed PulseForge response |
|---|---|
| Subtle acoustic change | Model-based anomaly score |
| Poor audio | Signal-quality checks |
| Conditions unfamiliar to training | Unfamiliarity / domain-shift flag |
| Bare unexplained model output | Visual evidence and reason text |
| Operator needs action | Cautious inspection recommendation |
| Risk of overreliance | Human-review language and no diagnosis claims |
