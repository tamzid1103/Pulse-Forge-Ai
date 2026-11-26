# Multi-Machine Industrial Fleet Maintenance SOPs (ISO 10816 / OSHA 1910.147)

This knowledge base extends PulseForge AI Factory Brain advisory across Centrifugal Pumps, Industrial Solenoid/Control Valves, and Linear Slide Rail Actuators alongside Centrifugal Fans.

---

## 1. Centrifugal Water & Process Pumps (`pump`)

### Acoustic Failure Signatures
- **Cavitation & Air Entrainment:** Broadband crackling/popping impulse energy across 2 kHz – 8 kHz resembling gravel inside the volute casing.
- **Mechanical Seal Leakage / Dry Running:** High-frequency squeal (> 4 kHz) from dry carbon-ceramic seal faces.
- **Bearing Raceway Spalling / Impeller Unbalance:** Periodic 1X rotational harmonics combined with cage fault frequencies.

### Mandatory Inspection Steps
1. Execute OSHA 1910.147 Lockout/Tagout on pump motor MCC breaker AND close suction/discharge isolation valves.
2. Verify Net Positive Suction Head Available (NPSHa) and inspect suction strainer for blockage.
3. Inspect mechanical seal gland for weeping or overheating via IR pyrometer (limit < 80°C).
4. Check coupling alignment with laser alignment tool and verify ISO 10816-3 Group 2 velocity.

### Required Tools & PPE
- **Tools:** Contact acoustic stethoscope, laser shaft alignment kit, Fluke IR thermometer, differential pressure gauge.
- **PPE:** ANSI Z87.1 splash goggles/face shield, chemical-resistant gloves, steel-toe boots, hearing protection.

---

## 2. Industrial Solenoid & Pneumatic Valves (`valve`)

### Acoustic Failure Signatures
- **Internal Seat Leakage / Wire-Drawing:** Continuous ultrasonic/high-frequency hissing (4 kHz – 8 kHz) while actuator command is closed.
- **Solenoid Coil Chatter / Plunger Sticking:** Irregular metallic clicking bursts during valve cycling.
- **Packing Gland Leak:** Escape of pressurized medium around valve stem.

### Mandatory Inspection Steps
1. Depressurize upstream/downstream piping and Lockout/Tagout pneumatic air supply & solenoid control voltage (24V DC / 120V AC).
2. Cycle valve manually/pneumatically to verify full stroke travel without stem binding.
3. Inspect elastomeric valve seat, O-rings, and spring return assembly for debris or erosion.
4. Tighten stem packing follower per OEM torque specification and leak-test with soap solution.

### Required Tools & PPE
- **Tools:** Digital multimeter, ultrasonic leak detector, torque wrench, calibrated pressure regulator.
- **PPE:** Face shield (pressurized line hazard), ANSI safety glasses, impact gloves, hearing protection.

---

## 3. Linear Slide Rail Actuators (`slider`)

### Acoustic Failure Signatures
- **Ball-Screw / Linear Bearing Pitting:** Rhythmic ticking or grinding bursts synchronized with carriage traverse reversals.
- **Rail Misalignment / Parallelism Drift:** Elevated friction hiss (> 3 kHz) concentrated near stroke endpoints.
- **Lubrication Starvation:** Dry metallic resonance during high-velocity travel.

### Mandatory Inspection Steps
1. Execute OSHA 1910.147 LOTO on servo/stepper drive power and verify zero kinetic/gravitational drop risk (engage Z-axis brake if vertical).
2. Wipe hardened steel guide rails clean and inspect ball-screw raceways under inspection light for brinelling or spalling.
3. Measure carriage parallelism and backlash with a dial test indicator.
4. Purge and re-grease linear runner blocks with NLGI Grade 2 lithium-complex EP grease.

### Required Tools & PPE
- **Tools:** Dial indicator with magnetic base, precision straightedge, NLGI 2 grease gun, feeler gauges.
- **PPE:** ANSI Z87.1 safety glasses, pinch-point protective gloves, steel-toe safety footwear.
