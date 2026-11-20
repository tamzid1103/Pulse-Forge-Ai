# PulseForge AI — Preliminary Submission Video Script (2 min 45 sec)

**Competition:** CloudCamp BD "International AI Builders Congress 2026"  
**Domain:** IndustrySphere AI  
**Primary Sub-Track:** Predictive Plant (Edge Acoustic Anomaly Detection)  
**Supporting Sub-Track:** Factory Brain (Autonomous AI Maintenance Copilot)  
**Target Duration:** 2 minutes 45 seconds (Hard Limit: 3 minutes / 180s)  
**Format:** Screen recording with voiceover (optional presenter picture-in-picture)  
**Live Demo URL:** `https://pulseforge-ai.streamlit.app`

---

## Technical Prep Before Recording (30 Seconds)

1. **Local or Cloud:** You can record either on your live cloud link (`https://pulseforge-ai.streamlit.app`) or locally:
   ```bash
   streamlit run app/dashboard.py
   ```
2. **Browser Setup:**
   - Use Google Chrome or Brave.
   - Set browser zoom to **100% or 110%** for clear fonts.
   - Sidebar visible on the left; clean workspace on the right.
3. **Reset Audit Log:** In the sidebar, click **"Clear Audit Log"** so the demo begins clean.
4. **Groq API Key:** Ensure your Groq API key is active (shows `● Groq LPU Active (llama-3.3-70b)` in the sidebar expander).

---

## Complete Timed Video Script & Cue Sheet

```
+--------------------------------------------------------------------------------------------------+
| 0:00 - 0:35 (35s) | SECTION 1: THE INDUSTRIAL PROBLEM & DUAL-TRACK HOOK                          |
+--------------------------------------------------------------------------------------------------+
```

**[VISUAL]**: Start on the presentation title or on the PulseForge AI dashboard header (`Acoustic Diagnostic Studio — Industrial Acoustic Intelligence`).

**[NARRATION]**:
> "Unplanned equipment downtime costs global manufacturing over **$50 billion** every year. In heavy industry, ventilation fans, blowers, and pumps run continuously—and when a bearing seizes, the entire production line stops.
> 
> Acoustic sound signatures contain the earliest physical indicators of mechanical distress—weeks before thermal cameras or broadband vibration sensors detect a problem.
> 
> But traditional AI fails on the factory floor because it hallucinates, triggers false alarms when ambient factory noise shifts, and cannot tell if a microphone is broken.
> 
> Welcome to **PulseForge AI**—an end-to-end industrial acoustic intelligence platform addressing two sub-tracks: **Predictive Plant** for self-validating anomaly detection, and **Factory Brain** for autonomous, ISO-compliant maintenance work orders."

---

```
+--------------------------------------------------------------------------------------------------+
| 0:35 - 1:15 (40s) | SECTION 2: ACOUSTIC DIAGNOSTIC STUDIO — HEALTHY VS FAULT                     |
+--------------------------------------------------------------------------------------------------+
```

**[ACTION]**: 
1. In Stage 1, select **"Try demo scenarios"**.
2. Click **"Scenario 1: Normal baseline fan"**.
3. Point cursor to Stage 2 audio player and play the fan sound for 2 seconds.
4. Click **"Run acoustic analysis"**.

**[NARRATION]**:
> "Let’s see it in action in our **Acoustic Diagnostic Studio**.
> 
> In Stage 1, we select Scenario 1—a healthy centrifugal fan under steady load. We preview the 16 kHz audio and click 'Run Diagnostics'.
> 
> Within 150 milliseconds on CPU, the pipeline evaluates the sound.
> 
> Notice the result: **NORMAL OPERATION CONFIRMED**. 
> The Autoencoder reconstruction error is 0.0021—well below our calibrated decision threshold of 0.0032. The latent familiarity is safely within distribution, and the dual-panel waveform and log-mel spectrogram show healthy harmonic continuity."

**[ACTION]**:
1. Switch to **"Scenario 2: Bearing rattle anomaly"**.
2. Click **"Run acoustic analysis"**.

**[NARRATION]**:
> "Now, let’s test Scenario 2—a developing bearing defect.
> 
> Instantly, the state machine shifts to **POSSIBLE ACOUSTIC ANOMALY**. 
> Reconstruction error surges to 0.0048, clearly exceeding our threshold. 
> Notice the spectrogram: high-frequency impact bursts between 2 kHz and 8 kHz indicate micro-spalling in the bearing raceway."

---

```
+--------------------------------------------------------------------------------------------------+
| 1:15 - 1:45 (30s) | SECTION 3: SIGNAL QUALITY & DOMAIN SHIFT (TRUSTWORTHINESS DIFFERENTIATOR)     |
+--------------------------------------------------------------------------------------------------+
```

**[ACTION]**: 
1. Switch to **"Scenario 3: Silent sensor failure"** or **"Scenario 5: Unfamiliar acoustic noise"**.
2. Click **"Run acoustic analysis"**.

**[NARRATION]**:
> "Here is what makes PulseForge truly industrial-grade: **Trustworthiness Gating**.
> 
> When we feed a disconnected sensor or sudden forklift background noise, PulseForge **does not guess** or trigger a false alarm.
> 
> Instead, it triggers **NEEDS REVIEW**. 
> It tells the operator: 'RMS energy is below physical thresholds—verify sensor cabling before condemning the machine.' 
> We never output uncalibrated failure probabilities or false positives."

---

```
+--------------------------------------------------------------------------------------------------+
| 1:45 - 2:25 (40s) | SECTION 4: FACTORY BRAIN — GROQ LPU AI MAINTENANCE COPILOT                   |
+--------------------------------------------------------------------------------------------------+
```

**[ACTION]**: 
1. Switch back to **"Scenario 2: Bearing rattle anomaly"**.
2. Scroll down to the **🤖 Industrial AI Maintenance Copilot** section.
3. Click **"⚡ Generate / Refresh Work Order"**.
4. Show the structured Work Order card (ID, Urgency, LOTO checklist, Tools, PPE).
5. Switch to the **💬 Technician Copilot Chat** tab and click the prompt chip: **"🛠️ Bearing inspection protocol"**.

**[NARRATION]**:
> "For our **Factory Brain** track, we integrated an autonomous maintenance copilot powered by **Groq LPU** running **Llama 3.3 (70B)**.
> 
> Grounded in real industrial Standard Operating Procedures and ISO 10816 vibration standards, the copilot automatically generates a **Digital Work Order**.
> 
> Look at the generated ticket: Asset ID, **PRIORITY INSPECTION** urgency, ISO Zone C classification, a mandatory OSHA Lockout/Tagout safety checklist, and required tools including infrared pyrometers and vibration pens. Operators can export this directly to JSON or CMMS.
> 
> In the interactive chat, technicians can ask engineering questions in real time, receiving instant, telemetry-grounded inspection advice at over 300 tokens per second."

---

```
+--------------------------------------------------------------------------------------------------+
| 2:25 - 2:45 (20s) | SECTION 5: PLANT FLEET, BENCHMARKS & ROADMAP TO FINALS                        |
+--------------------------------------------------------------------------------------------------+
```

**[ACTION]**: 
1. Click the sidebar navigation to **"Plant Fleet Overview"** (show the fleet line telemetry).
2. Click **"Model Benchmarks & Validation"** (show the AUROC 0.9375 curve).

**[NARRATION]**:
> "Our platform also provides a **Plant Fleet Overview** across multiple production lines, an **Inspection Audit Log** backed by SQLite, and verified empirical benchmarks—achieving **0.9375 AUROC** on real MIMII industrial recordings.
> 
> For the December 26th finals, we are expanding to multi-machine fleets, local edge Ollama execution, and a custom Model Context Protocol (MCP) server.
> 
> PulseForge AI brings acoustic intelligence and automated reasoning to the factory floor. Thank you!"

---

## Timing Check & Checklist

| Section | Content | Start Time | End Time | Target Duration |
| :--- | :--- | :--- | :--- | :--- |
| **Section 1** | Industrial Problem, $50B downtime, Dual Tracks | 0:00 | 0:35 | 35 sec |
| **Section 2** | Acoustic Diagnostic Studio (Normal vs Bearing Anomaly) | 0:35 | 1:15 | 40 sec |
| **Section 3** | Signal Quality Gating & Trustworthiness (Needs Review) | 1:15 | 1:45 | 30 sec |
| **Section 4** | Factory Brain Groq LPU Copilot & 1-Click Work Order | 1:45 | 2:25 | 40 sec |
| **Section 5** | Fleet Overview, AUROC 0.9375 Benchmarks, Finals Pitch | 2:25 | 2:45 | 20 sec |
| **Total** | **Complete Video Pitch** | **0:00** | **2:45** | **165 seconds (< 180s)** |
