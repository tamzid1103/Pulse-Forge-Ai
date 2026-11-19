# PulseForge AI — Preliminary Submission Video Script (Milestone M7)

**Event:** CloudCamp BD "International AI Builders Congress 2026"  
**Track:** IndustrySphere AI — Predictive Plant (Supporting: Factory Brain)  
**Target Duration:** 2 minutes 30 seconds (Maximum: 3 minutes)  
**Presenter:** Single or Duo Presenter  
**Format:** Screen Recording with Voiceover (and optional webcam overlay)

---

## Technical Prep Before Recording

1. Open a terminal and run the Streamlit dashboard:
   ```bash
   source .venv/bin/activate
   streamlit run app/dashboard.py
   ```
2. Open your web browser to `http://localhost:8501`.
3. Set your browser zoom to **100% or 110%** for clear typography on laptop recordings.
4. Have your microphone ready and clear background noise.
5. In the dashboard sidebar, click **"Clear Audit Log"** so the session starts fresh.

---

## Video Script & Cue Sheet

```
+---------------------------------------------------------------------------------------+
| 0:00 - 0:30 | SECTION 1: THE INDUSTRIAL PROBLEM & HOOK                                |
+---------------------------------------------------------------------------------------+
```

**[VISUAL]**: Show title screen or camera on presenter, then switch to the PulseForge AI dashboard header (`PulseForge AI | v1.0-preliminary`).

**[VOICEOVER / NARRATION]**:
> "In industrial manufacturing, unexpected equipment failures cost factories over $50 billion annually in unplanned downtime.
> 
> While vibration sensors and temperature probes are common, acoustic sound signatures contain the earliest harmonic indicators of mechanical failure—like bearing wear and blade imbalance.
> 
> But existing acoustic AI tools fail in real plants. Why? Because they hallucinate failure probabilities, trigger false alarms when ambient factory noise shifts, and cannot tell when a physical sensor is broken or saturated.
> 
> Today, we present **PulseForge AI**—a self-validating edge acoustic anomaly detection system built specifically for predictive plant operations."

---

```
+---------------------------------------------------------------------------------------+
| 0:30 - 1:00 | SECTION 2: SYSTEM ARCHITECTURE & SELF-VALIDATION                       |
+---------------------------------------------------------------------------------------+
```

**[VISUAL]**: Hover over sidebar showing: Monitored Asset (`fan_id_00`), PyTorch Deep Autoencoder (16-latent bottleneck), Calibrated Anomaly Threshold (`0.8904`), and Latent Familiarity Limit (`9.2549`).

**[VOICEOVER / NARRATION]**:
> "PulseForge AI is built on a four-stage deterministic edge pipeline:
> 
> 1. **Signal Quality Gating:** Validates physical acoustic telemetry—detecting dead microphones, digital clipping, and DC offset before AI inference.
> 2. **Deep Acoustic Autoencoder:** Trained purely on normal industrial fan recordings to compute reconstruction error without requiring rare labelled fault data.
> 3. **Latent Familiarity Assessment:** Measures Mahalanobis distance in the model's bottleneck space. If factory background noise shifts, the system explicitly knows it is out of distribution.
> 4. **4-State Operational Decision Policy:** Outputs actionable, ISO 13374-compliant states—never misleading confidence percentages."

---

```
+---------------------------------------------------------------------------------------+
| 1:00 - 2:15 | SECTION 3: LIVE OPERATIONAL DEMONSTRATION (THE 5 SCENARIOS)            |
+---------------------------------------------------------------------------------------+
```

#### Demo 1: Healthy Operating Baseline (Normal)
**[ACTION]**: In the dropdown, select **"🟢 Scenario 1: Normal Operating Fan (Healthy Baseline)"**.
**[VISUAL]**: Show the green decision card, audio waveform, and spectrogram.

**[VOICEOVER / NARRATION]**:
> "Let's test Scenario 1: A healthy industrial fan operating under normal load.
> 
> Within 150 milliseconds on CPU, PulseForge analyzes the 10-second recording.
> Notice the **Green Status**: 'NORMAL OPERATION CONFIRMED'.
> The reconstruction error is 0.5245—well below our calibrated decision threshold of 0.8904.
> The latent familiarity is within known limits, and the spectrogram shows a consistent, harmonic baseline."

---

#### Demo 2: Genuine Mechanical Anomaly (Bearing Rattle)
**[ACTION]**: In the dropdown, select **"🔴 Scenario 2: Bearing Anomaly (Harmonic Rattling Fault)"**.
**[VISUAL]**: Show the red decision card, elevated reconstruction error, and spectrogram transient spikes.

**[VOICEOVER / NARRATION]**:
> "Now, Scenario 2: A fan with internal mechanical degradation from bearing rattle.
> 
> Instantly, the system triggers a **Red Status**: 'POSSIBLE ACOUSTIC ANOMALY DETECTED'.
> The reconstruction error surges to 1.2245, sharply exceeding our 0.8904 threshold.
> Notice our clear, responsible recommendation: 'Schedule physical inspection of fan bearings within 24 hours.'
> We give operators an actionable decision—not a fabricated probability."

---

#### Demo 3 & 4: Sensor Hardware Failures (Self-Validation in Action!)
**[ACTION]**: Select **"🟡 Scenario 3: Silent Sensor Failure"**, then **"🟡 Scenario 4: Clipped Sensor Saturation"**.
**[VISUAL]**: Show amber decision card with clear hardware diagnostic reasons.

**[VOICEOVER / NARRATION]**:
> "Here is where PulseForge AI truly shines over conventional AI:
> When we simulate a disconnected microphone in Scenario 3, or a saturated ADC in Scenario 4, typical models hallucinate or crash.
> 
> PulseForge intercepts this at the signal layer!
> It flags **'HUMAN OPERATOR REVIEW REQUIRED'** and tells the technician exactly: 'Microphone signal silent—check cable connection and sensor gain.'
> The AI refuses to make blind guesses on invalid data."

---

#### Demo 5: Ambient Factory Noise / Domain Shift (Latent Familiarity)
**[ACTION]**: Select **"🟡 Scenario 5: Unfamiliar Acoustic Noise (Out-of-Distribution Shift)"**.
**[VISUAL]**: Show amber decision card with Latent Familiarity Distance of 29.03 vs limit of 9.2549.

**[VOICEOVER / NARRATION]**:
> "Finally, Scenario 5: What happens when nearby machinery creates loud background acoustic noise?
> 
> Our latent familiarity estimator detects a Mahalanobis distance of 29.03—way above our in-distribution threshold of 9.2549.
> Instead of falsely crying wolf, PulseForge flags this for human engineering review as an unfamiliar acoustic environment.
> Every single run is automatically logged into our SQLite audit store for compliance."

---

```
+---------------------------------------------------------------------------------------+
| 2:15 - 2:45 | SECTION 4: THE PHASE 2 EXPANSION (FACTORY BRAIN & MCP)                 |
+---------------------------------------------------------------------------------------+
```

**[VISUAL]**: Show terminal showing 129 passing unit tests (`pytest tests/`), or switch back to presenter/slides highlighting Phase 2 architecture.

**[VOICEOVER / NARRATION]**:
> "For this preliminary milestone, we have delivered a 100% functional, rigorously tested system with 129 passing unit tests, full dataset manifests, and an offline Streamlit operator console.
> 
> In Phase 2 for the final congress, we are expanding PulseForge AI into the **Factory Brain** track:
> - Integrating a local **Ollama-powered AI Maintenance Copilot** with Retrieval-Augmented Generation over OEM machine manuals.
> - Exposing a dedicated **PulseForge MCP Server** so enterprise AI agents can autonomously query plant acoustic telemetry.
> - And deploying a dual **Lovable React Dashboard** for multi-machine fleet monitoring."

---

```
+---------------------------------------------------------------------------------------+
| 2:45 - 3:00 | SECTION 5: CONCLUSION & CALL TO ACTION                                 |
+---------------------------------------------------------------------------------------+
```

**[VISUAL]**: Presenter on camera or closing slide with GitHub URL and CloudCamp Congress branding.

**[VOICEOVER / NARRATION]**:
> "PulseForge AI bridges the gap between deep acoustic signal processing and trusted industrial decision-making.
> 
> Thank you to CloudCamp Bangladesh and the International AI Builders Congress 2026.
> We look forward to seeing you in the finals!"

---

## Operator Tips for Demo Recording

- **Pacing:** Speak clearly and with measured confidence. Do not rush through the numbers.
- **Mouse Movement:** Move the cursor deliberately to the dropdown, then to the decision banner, then to the metric cards.
- **Audio:** Ensure your recording captures the system sound so viewers can hear the healthy fan versus the rattling bearing when playing the audio!
