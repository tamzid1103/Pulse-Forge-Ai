# Model Context Protocol (MCP) Specification

**Product:** PulseForge AI  
**Protocol Version:** MCP 2.x (`mcp>=2.0.0`)  
**Server Implementation:** `src/mcp_server.py`  
**Test Suite:** `tests/test_mcp_server.py` (18 passing tests)  
**Standard Transports:** Stdio (standard input/output for local IDE agents) & SSE (Server-Sent Events)

---

## 1. Overview & Purpose

PulseForge AI exposes its industrial acoustic intelligence and condition monitoring engine to external AI agents (Google Antigravity, Claude Desktop, Cursor, and autonomous plant orchestrators) via the official Model Context Protocol (MCP).

External agents can:
1. **Analyze machine audio:** Ingest WAV recordings, detect micro-surface bearing distress via PyTorch Autoencoder reconstruction MSE, evaluate latent Mahalanobis familiarity, and obtain 4-state decisions.
2. **Inspect physical sensor health:** Evaluate RMS amplitude (sensor disconnection), digital clipping ratio (pre-amplifier saturation), and duration validity in <10 ms.
3. **Query SQLite plant telemetry:** Retrieve historical inspection records, baseline operating parameters, and defect trends.
4. **Generate OSHA/ISO work orders:** Trigger the Factory Brain Copilot to produce structured Lockout/Tagout (LOTO) maintenance tickets.
5. **Read industrial standards:** Access ISO 10816 vibration severity tables and ventilation fan maintenance SOPs.

---

## 2. Tools Catalog

### `analyze_fan_audio`
Executes the complete 6-stage acoustic diagnostic pipeline on a WAV file.

- **Parameters:**
  - `audio_path` (string, required): Filesystem path to the WAV recording.
  - `machine_id` (string, optional, default `"fan_id_00"`): Target asset identifier.
- **Return Payload:**
  ```json
  {
    "run_id": "c1b2f489-091a-4d43-85f2-9d32b50871ea",
    "machine_id": "fan_00_a",
    "filename": "01_normal_fan.wav",
    "status": "normal",
    "status_label": "Normal",
    "decision_explanation": "Normal acoustic operation confirmed. Reconstruction error within healthy baseline.",
    "recommended_action": "Continue standard routine monitoring schedule.",
    "quality": {
      "passed": true,
      "rms_level": 0.03451,
      "is_silent": false,
      "clipping_ratio": 0.0,
      "is_clipped": false,
      "duration_seconds": 10.0,
      "warnings": [],
      "errors": []
    },
    "anomaly_detection": {
      "anomaly_score": 0.00192,
      "threshold": 0.00320,
      "is_anomaly": false,
      "model_type": "autoencoder"
    },
    "domain_familiarity": {
      "score": 4.120,
      "threshold": 9.250,
      "is_familiar": true
    }
  }
  ```

---

### `inspect_signal_quality`
Lightweight pre-inference hardware screening of audio sensor suitability.

- **Parameters:**
  - `audio_path` (string, required): Filesystem path to the WAV recording.
- **Return Payload:**
  ```json
  {
    "filename": "03_silent_sensor_failure.wav",
    "suitable_for_ml": false,
    "duration_seconds": 10.0,
    "rms_amplitude": 0.00002,
    "clipping_ratio": 0.0,
    "is_silent": true,
    "is_clipped": false,
    "is_duration_valid": true,
    "diagnostic_flags": ["Audio signal is silent or near-silent (RMS: 0.00002 < 0.00100)."],
    "recommendation": "Hardware malfunction or acoustic interference detected. Inspect microphone probe."
  }
  ```

---

### `get_machine_telemetry`
Retrieves historical inspection records and summary metrics from SQLite audit store.

- **Parameters:**
  - `machine_id` (string, optional, default `"fan_id_00"`): Machine identifier or `"all"`.
  - `limit` (integer, optional, default `10`): Maximum number of recent records.
- **Return Payload:**
  ```json
  {
    "machine_id": "fan_id_00",
    "monitored_fleet_summary": {
      "total_runs": 24,
      "normal_runs": 18,
      "anomaly_runs": 3,
      "review_runs": 3
    },
    "retrieved_runs_count": 5,
    "recent_inspections": [...]
  }
  ```

---

### `generate_maintenance_work_order`
Generates a formal, actionable OSHA 1910.147 and ISO 10816 compliant maintenance ticket.

- **Parameters:**
  - `machine_id` (string, optional, default `"fan_id_00"`): Target machine identifier.
  - `decision_status` (string, optional, default `"possible_anomaly"`): Current diagnostic status.
  - `operator_note` (string, optional): Operator field observations.
- **Return Payload:**
  ```json
  {
    "work_order_id": "WO-FAN00-20261122-8A91",
    "urgency": "HIGH",
    "suspected_mechanism": "Rolling element bearing fatigue / cage degradation",
    "iso_severity_zone": "Zone C",
    "safety_checklist": [
      "Comply with OSHA 1910.147 Lockout/Tagout",
      "De-energize 480V motor supply switch and apply personal safety padlock",
      "Verify zero kinetic energy / wait for blower impeller coast-down",
      "Depress local start button to confirm zero energy state"
    ],
    "required_ppe": ["Steel-toe boots", "ANSI Z87.1 safety glasses", "Cut-resistant gloves"],
    "recommended_steps": [
      "Remove fan belt guard",
      "Check SKF 6308 pillow block bearings for thermal discoloration",
      "Measure broadband vibration velocity (ISO 10816-3)",
      "Replenish synthetic polyurea grease if required"
    ],
    "ai_engine": "llama-3.1-8b-instant (Groq LPU) [or offline SOP fallback]"
  }
  ```

---

### `list_demo_scenarios`
Returns metadata catalog of pre-packaged demonstration audio recordings.

- **Parameters:** None
- **Returns:** List of 5 scenario objects with `id`, `label`, `file_path`, `expected_status`, and `description`.

---

## 3. Resources Catalog

| Resource URI | Description | MIME Type |
|:---|:---|:---|
| `pulseforge://fleet/status` | Operational health table of all monitored plant ventilation units | `text/plain` (Markdown) |
| `pulseforge://standards/iso10816` | ISO 10816-3 mechanical vibration severity zones (A, B, C, D) | `text/plain` (Markdown) |
| `pulseforge://standards/sop/fan_maintenance` | Standard Operating Procedure text from domain knowledge base | `text/plain` (Markdown) |
| `pulseforge://scenarios/catalog` | JSON catalog of benchmark audio scenario files | `application/json` |

---

## 4. Prompts Catalog

| Prompt Name | Arguments | Purpose |
|:---|:---|:---|
| `triage_acoustic_anomaly` | `machine_id`, `anomaly_score`, `threshold` | Guides AI agent through ISO 10816 severity evaluation and shift log authoring |
| `osha_loto_compliance_check` | `machine_id`, `work_order_id` | Verifies 6 mandatory Lockout/Tagout safety steps prior to technician dispatch |

---

## 5. Client Configuration & Quick Start

### Claude Desktop Configuration
Add to `~/.config/Claude/claude_desktop_config.json` (Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "pulseforge-ai": {
      "command": "/path/to/Pulse-Forge-Ai/.venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/Pulse-Forge-Ai",
      "env": {
        "PYTHONPATH": "/path/to/Pulse-Forge-Ai"
      }
    }
  }
}
```

### Stdio Verification Command
Verify the server manually from the terminal:
```bash
.venv/bin/python -m src.mcp_server stdio
```

### Automated Testing
Run the MCP test suite:
```bash
.venv/bin/pytest tests/test_mcp_server.py -v
# 18 passed in ~7s
```
