# UI / UX Guidelines

## 1. UX objective

A maintenance operator or judge should understand what was analyzed, what the model observed, how trustworthy the input/result appears, and what should happen next—without needing to understand machine-learning terminology.

## 2. MVP page structure

Use one Streamlit page with clear sections.

### A. Header

- Product name: PulseForge AI.
- Tagline: “Hear the Warning. Trust the Signal. Prevent the Downtime.”
- Subtitle: “Acoustic anomaly screening for industrial fans.”
- Persistent note: “Prototype decision support; not a confirmed mechanical diagnosis.”

### B. Machine and recording input

- Machine selector: show `Fan` as the only enabled MVP machine.
- Other machine types may be shown as “Planned” but must not be selectable for inference.
- WAV uploader.
- Optional “Load demo sample” selector for preloaded normal, anomalous, and review-needed recordings.
- Display file name as a safe label, duration, sample rate, channel count, and upload validation state.

### C. Audio visualization

- Waveform plot.
- Log-Mel spectrogram or the exact feature representation used by the model.
- Axes, titles, and units where meaningful.
- Do not imply that a frequency region maps to a specific mechanical fault unless supported by evidence.

### D. Result summary

Show distinct, text-labelled state:

- **Normal pattern** — passed the quality/familiarity gate and score is below the selected threshold.
- **Possible anomaly** — score exceeds the validated threshold.
- **Needs review** — input quality is poor or the recording is too unfamiliar for a reliable automatic conclusion.
- **Invalid input** — decoding or required constraints failed.

Display anomaly score and threshold together. If a normalized visual bar is used, label it “anomaly score”, not “failure probability” or “confidence”.

### E. Trustworthiness panel

Show three separate concepts:

1. **Signal quality:** Good / Warning / Poor, plus reasons.
2. **Input familiarity:** Within known training conditions / Unfamiliar / Not evaluated.
3. **Anomaly assessment:** Score, threshold, and status.

Do not collapse these into a single confidence percentage.

### F. Explanation and recommendation

Use concise statements generated from measured signals and flags. Examples:

- “The reconstruction error is above the validation-selected threshold.”
- “The recording is clipped; repeat the recording with a suitable microphone level.”
- “The input differs from the training feature distribution; ask an engineer to review it.”
- “Inspect the machine and verify the recording before maintenance action.”

Do not state that a specific bearing, shaft, or motor component has failed unless a separate validated fault-classification model supports it. The MVP does not have one.

## 3. Interaction states

| State | UI behavior |
|---|---|
| Idle | Explain supported input and show upload / demo selection |
| Uploading | Show progress spinner and prevent duplicate submission |
| Validating | Show validation progress |
| Invalid | Show field-level reason and how to fix it |
| Analyzing | Show progress; avoid presenting stale previous result as the new result |
| Complete | Render result and plots |
| Needs review | Prominent warning with the reasons and suggested next step |
| Model unavailable | Explain that the model artifact is missing; show developer guidance without stack trace |
| Unexpected error | Friendly error ID/message; technical details in local logs |

## 4. Visual design guidance

- Use a clean industrial dashboard style.
- Neutral background with restrained status colors.
- Green can indicate normal; amber can indicate review; red can indicate possible anomaly, but every state must also use text and an icon/label.
- Avoid excessive cards, animations, or decorative gauges.
- Keep critical status, score, and warning visible without scrolling where practical.
- Keep charts large enough to read on a laptop projector.
- Use consistent number formatting and units.

## 5. Accessibility and clarity

- Do not rely on color alone.
- Use descriptive button and upload labels.
- Provide clear validation text.
- Use readable font sizes and contrast.
- Give plots titles and axis labels.
- Avoid acronyms without a short explanation at first use.
- Explain “anomaly score” as a model-specific measure of difference, not a failure probability.

## 6. Proposed dashboard wireframe

```text
┌────────────────────────────────────────────────────────────────┐
│ PulseForge AI                          Acoustic Fan Monitoring  │
│ Hear the Warning. Trust the Signal. Prevent the Downtime.       │
├────────────────────────────────────────────────────────────────┤
│ Machine: [Fan]       Audio: [Upload WAV] [Load Demo ▼]          │
│ File: fan_recording.wav | Duration | Sample rate | Channels     │
├────────────────────────────────────────────────────────────────┤
│ Waveform                         Log-Mel Spectrogram             │
│ [chart]                          [chart]                         │
├────────────────────────────────────────────────────────────────┤
│ RESULT: POSSIBLE ANOMALY                                         │
│ Anomaly score: <measured>       Threshold: <validated>           │
│ Signal quality: GOOD            Input familiarity: WITHIN RANGE  │
├────────────────────────────────────────────────────────────────┤
│ WHY: reconstruction error exceeds selected threshold.          │
│ NEXT: inspect the machine and verify the recording.             │
│ This is decision support, not a confirmed diagnosis.            │
└────────────────────────────────────────────────────────────────┘
```

## 7. Demo UX requirements

- Include a clearly marked demo selector so judges do not need to browse files.
- Preload the model at app startup and display a model-ready status.
- Keep the demo samples small and local.
- Have a reset / clear-analysis action.
- Show the exact demo scenario label, but do not expose ground-truth labels before the prediction if doing a blind evaluation demonstration.
