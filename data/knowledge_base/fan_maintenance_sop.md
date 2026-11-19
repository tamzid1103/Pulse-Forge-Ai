# Standard Operating Procedure: Industrial Centrifugal Fan Acoustic Maintenance & Inspection
**Document ID:** SOP-PF-FAN-001  
**Target Equipment:** Centrifugal / Axial Industrial Ventilation Fans (MIMII Benchmark Class, Models 00, 02, 04, 06)  
**Applicability:** Predictive Maintenance, Acoustic Anomaly Triaging, and Work Order Issuance  
**Governing Standards:** ISO 13374 (Condition Monitoring & Diagnostics), ISO 10816-3 (Mechanical Vibration Severity), OSHA 1910.147 (Control of Hazardous Energy / LOTO)

---

## 1. Machine Architecture & Critical Mechanical Subsystems

Industrial fans monitored by PulseForge AI consist of five primary mechanical and electrical subsystems susceptible to degradation:

1. **Rolling Element Bearings (Inboard & Outboard):**
   - Typically deep groove ball bearings (e.g., SKF 6308-2Z) or spherical roller bearings.
   - Primary failure modes: Inner race flaking, outer race spalling, ball cage fracturing, lubricant breakdown/starvation.
   - Acoustic signature: High-frequency transient impact peaks (2 kHz – 8 kHz) detectable via log-mel spectral anomalies before macroscopic vibration rise.

2. **Rotational Impeller / Fan Wheel:**
   - Backward-curved or aerofoil centrifugal blades operating between 900 and 1800 RPM.
   - Primary failure modes: Dust/debris accumulation leading to mass unbalance, blade erosion from particulate abrasion, aerodynamic stall/surge.
   - Acoustic signature: 1X rotational speed fundamental frequency spike and low-frequency aerodynamic duct rumbling (< 500 Hz).

3. **V-Belt Transmission & Pulleys (Belt-Driven Units):**
   - Matched V-belt sets transferring torque from electric motor to fan shaft.
   - Primary failure modes: Belt slippage, slackening/excessive tension, pulley groove wear, angular or parallel shaft misalignment.
   - Acoustic signature: High-pitch belt squeal, harmonic belt pass frequencies (2X, 3X).

4. **Shaft Seals & Mechanical Housing:**
   - Labyrinth seals and housing hold-down fasteners.
   - Primary failure modes: Structural looseness, mounting foot bolt fatigue, rubbing of shaft against seal.
   - Acoustic signature: Non-harmonic impact bursts and broadband frictional hissing.

5. **Induction Drive Motor (3-Phase, 480V):**
   - Primary failure modes: Stator winding asymmetry, eccentric air-gap, loose rotor bars.
   - Acoustic signature: 2X line frequency (100 Hz / 120 Hz) magnetic hum.

---

## 2. Acoustic Diagnostic Decision Logic & Triage Protocols

When PulseForge AI executes an acoustic evaluation, the telemetry maps to specific physical actions:

### State A: NORMAL (Anomaly Score < Validation Threshold)
- **Acoustic Indicator:** Reconstruction error is within statistical baseline (e.g., < 0.0032 MSE). Latent distance is in-distribution.
- **Physical Meaning:** Machine is operating in nominal dynamic balance. Bearing raceways show hydrodynamic film lubrication with no metal-to-metal contact.
- **Action Required:**
  - No physical intervention required.
  - Continue scheduled automated acoustic monitoring at standard 4-hour polling intervals.
  - Record baseline acoustic profile in SQLite audit history.

### State B: POSSIBLE_ANOMALY (Anomaly Score ≥ Validation Threshold)
- **Acoustic Indicator:** Reconstruction error exceeds operational threshold; latent familiarity confirms the recording is in-distribution fan noise (valid machine sound with anomalous frequency content).
- **Physical Meaning:** Developing mechanical fault. Transient mechanical friction, bearing micro-spalling, blade fouling, or belt slippage detected.
- **Action Required:**
  - Issue **Priority Inspection Work Order** (within 24–48 hours).
  - Verify bearing housing surface temperature using calibrated infrared pyrometer.
  - Perform physical stethoscope or high-frequency accelerometer sweep on inboard/outboard bearing pillows.
  - Check dynamic belt tension using digital tension gauge.
  - Inspect intake air filters and impeller cleanliness.

### State C: NEEDS_REVIEW (Signal Degraded OR Out-of-Distribution Shift)
- **Condition 1 (Signal Degradation):** Silent sensor, clipping saturation (> 1.0%), low RMS energy (< 0.005), or truncated recording (< 2.0s).
  - *Action:* Inspect acoustic sensor / microphone mounting, wiring integrity, and preamplifier gain. Do not condemn the machine; verify the instrumentation.
- **Condition 2 (Domain Shift / Unfamiliar Acoustic Context):**
  - *Action:* High background plant noise detected (forklifts, impact hammers, adjacent compressors). Re-record during steady plant operational state.

### State D: INVALID_INPUT (File Corrupted / Decode Failure)
- **Action:** Reject file safely, log file header error, prompt operator to resubmit raw 16 kHz / 16-bit uncompressed WAV.

---

## 3. Mandatory Safety Protocol: Lockout / Tagout (LOTO)

**WARNING:** Never open fan inspection access doors or touch rotational components while the drive motor is energized.

Prior to performing any hands-on mechanical inspection triggered by PulseForge AI:
1. **Notify Production Supervisor:** Confirm temporary line isolation.
2. **De-energize Drive Motor:** Switch off the primary 480V circuit breaker at the local Motor Control Center (MCC).
3. **Apply LOTO Hardware:** Attach personal safety padlock, hasp, and dated warning tag to the electrical disconnect handle.
4. **Zero-Energy Verification:** Attempt to start fan via local start button to confirm zero electrical energization.
5. **Coast-Down & Wheel Stoppage:** Visually confirm through sight glass that the impeller has achieved complete standstill (0 RPM).
6. **Dissipate Residual Energy:** In high-temperature or pressurized ductwork, vent duct pressure and allow cooling below 45°C.

---

## 4. Maintenance Tooling & PPE Matrix

| Inspection Level | Mandatory PPE | Required Mechanical Tools |
| :--- | :--- | :--- |
| **Routine Acoustic Check** | Safety glasses (ANSI Z87.1), Steel-toe boots, Class 1 earplugs | Directional shotgun acoustic probe, calibrated calibrator |
| **Priority Anomaly Inspection** | Safety glasses, Class 2 hearing protection, Cut-resistant gloves | Handheld vibration pen (ISO 10816), Fluke IR thermometer, ultrasonic leak detector |
| **Full Mechanical Teardown** | Hard hat, Steel-toe boots, Heavy leather work gloves, Arc-flash rated gear (for MCC) | Torque wrench, dial indicator with magnetic base (shaft runout), bearing induction heater, grease gun with calibrated stroke meter |

---

## 5. Bearing Re-Lubrication Protocol

- **Grease Specification:** NLGI Grade 2 Synthetic Polyurea or Lithium Complex with ISO VG 220 synthetic base oil.
- **Relubrication Volume Formula:**  
  $$G = 0.005 \times D \times B$$  
  *(where $G$ is grease weight in grams, $D$ is bearing outer diameter in mm, and $B$ is bearing width in mm)*.
- **Application Note:** Clean zerk fitting thoroughly before connecting grease gun. Pump grease slowly while rotating shaft manually to distribute grease evenly across rolling elements. Avoid over-lubrication (causes churning and overheating).
