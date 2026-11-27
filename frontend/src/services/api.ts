/**
 * PulseForge AI API Service Client.
 * Connects to FastAPI REST & WebSocket endpoints with resilient fallback handling.
 */

import {
  AnalysisResponse,
  AnalysisResult,
  AuditRun,
  ChatMessage,
  DemoScenario,
  PersistenceStatus,
  WebSocketMessage,
  WorkOrder,
} from "../types";

// Base API URL: uses Vite proxy '/api' or direct environment variable
const API_BASE = import.meta.env.VITE_API_URL || "";

// Default fallback scenarios matching backend catalog exactly
export const FALLBACK_SCENARIOS: DemoScenario[] = [
  {
    id: "scenario_01",
    filename: "01_normal_fan.wav",
    name: "Baseline Healthy Centrifugal Fan",
    machine_id: "fan_id_00",
    condition: "normal",
    description: "Standard operational acoustic baseline. Balanced aerodynamic flow.",
    expected_status: "normal",
    file_path: "demo_audio/01_normal_fan.wav",
  },
  {
    id: "scenario_02",
    filename: "02_bearing_outer_race_fault.wav",
    name: "Impeller Bearing Fault",
    machine_id: "fan_id_00",
    condition: "anomalous",
    description: "Outer-race mechanical flaking. High-frequency acoustic impact pulses.",
    expected_status: "possible_anomaly",
    file_path: "demo_audio/02_bearing_outer_race_fault.wav",
  },
  {
    id: "scenario_03",
    filename: "03_silent_sensor_failure.wav",
    name: "Dead Sensor / Probe Disconnect",
    machine_id: "fan_id_00",
    condition: "sensor_failure",
    description: "Acoustic line disconnected or dead sensor. High DC bias, zero RMS signal.",
    expected_status: "needs_review",
    file_path: "demo_audio/03_silent_sensor_failure.wav",
  },
  {
    id: "scenario_04",
    filename: "04_clipped_preamp_saturation.wav",
    name: "Pre-amp Acoustic Saturation",
    machine_id: "fan_id_00",
    condition: "clipping",
    description: "Microphone pre-amplifier saturation. Severe peak clipping (>15%).",
    expected_status: "needs_review",
    file_path: "demo_audio/04_clipped_preamp_saturation.wav",
  },
  {
    id: "scenario_05",
    filename: "05_unfamiliar_operating_noise.wav",
    name: "Ambient Acoustic Domain Shift",
    machine_id: "fan_id_00",
    machine_type: "fan",
    condition: "unfamiliar_shift",
    description: "External crane & forklift interference. High latent Mahalanobis distance.",
    expected_status: "needs_review",
    file_path: "demo_audio/05_unfamiliar_operating_noise.wav",
  },
  {
    id: "scenario_06",
    filename: "06_pump_normal.wav",
    name: "Centrifugal Pump Normal Flow",
    machine_id: "pump_id_00",
    machine_type: "pump",
    condition: "normal",
    description: "MIMII Centrifugal Pump ID 00 healthy steady-flow acoustic signature.",
    expected_status: "normal",
    file_path: "demo_audio/06_pump_normal.wav",
  },
  {
    id: "scenario_07",
    filename: "07_pump_anomaly.wav",
    name: "Pump Cavitation / Seal Defect",
    machine_id: "pump_id_00",
    machine_type: "pump",
    condition: "anomalous",
    description: "MIMII Centrifugal Pump ID 00 impeller cavitation and seal wear signature.",
    expected_status: "possible_anomaly",
    file_path: "demo_audio/07_pump_anomaly.wav",
  },
  {
    id: "scenario_08",
    filename: "08_valve_normal.wav",
    name: "Control Valve Normal Cycle",
    machine_id: "valve_id_00",
    machine_type: "valve",
    condition: "normal",
    description: "MIMII Industrial Valve ID 00 clean actuation acoustic baseline.",
    expected_status: "normal",
    file_path: "demo_audio/08_valve_normal.wav",
  },
  {
    id: "scenario_09",
    filename: "09_valve_anomaly.wav",
    name: "Valve Internal Seat Leak",
    machine_id: "valve_id_00",
    machine_type: "valve",
    condition: "anomalous",
    description: "MIMII Industrial Valve ID 00 internal seat leakage / solenoid chatter.",
    expected_status: "possible_anomaly",
    file_path: "demo_audio/09_valve_anomaly.wav",
  },
  {
    id: "scenario_10",
    filename: "10_slider_normal.wav",
    name: "Slide Rail Smooth Traverse",
    machine_id: "slider_id_00",
    machine_type: "slider",
    condition: "normal",
    description: "MIMII Linear Slide Rail ID 00 lubricated ball-screw nominal motion.",
    expected_status: "normal",
    file_path: "demo_audio/10_slider_normal.wav",
  },
  {
    id: "scenario_11",
    filename: "11_slider_anomaly.wav",
    name: "Slide Rail Ball-Screw Pitting",
    machine_id: "slider_id_00",
    machine_type: "slider",
    condition: "anomalous",
    description: "MIMII Linear Slide Rail ID 00 abnormal ball-screw impact impulses.",
    expected_status: "possible_anomaly",
    file_path: "demo_audio/11_slider_anomaly.wav",
  },
];

export async function checkBackendHealth(): Promise<{
  online: boolean;
  latencyMs: number;
  persistence?: PersistenceStatus;
}> {
  const start = performance.now();
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const latency = Math.round(performance.now() - start);
    if (!res.ok) return { online: false, latencyMs: latency };
    const data = await res.json().catch(() => ({}));
    return { online: true, latencyMs: latency, persistence: data.persistence };
  } catch {
    return { online: false, latencyMs: 0 };
  }
}

export async function fetchDemoScenarios(machineType: string = "all"): Promise<DemoScenario[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/scenarios?machine_type=${encodeURIComponent(machineType)}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) throw new Error("Failed to load scenarios");
    const data = await res.json();
    const list: DemoScenario[] = Array.isArray(data) ? data : data.scenarios || FALLBACK_SCENARIOS;
    return list.map((s) => ({
      ...s,
      name: s.name || s.label || s.filename,
      machine_id: s.machine_id || `${s.machine_type || "fan"}_id_00`,
      machine_type: s.machine_type || "fan",
    }));
  } catch {
    return FALLBACK_SCENARIOS;
  }
}

export async function evaluateScenario(
  scenarioId: string,
  machineId?: string,
  machineType?: string
): Promise<AnalysisResponse> {
  try {
    const body = new FormData();
    if (machineId) body.append("machine_id", machineId);
    if (machineType) body.append("machine_type", machineType);
    const res = await fetch(`${API_BASE}/api/v1/analyze/scenario/${scenarioId}`, {
      method: "POST",
      body,
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail?.message || `Server returned ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    // Deterministic mock fallback in case backend is not running
    return createSyntheticFallback(scenarioId);
  }
}

export async function uploadAudioAnalysis(
  file: File,
  machineId: string = "fan_id_00",
  machineType: string = "fan",
  operatorNote: string = ""
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("machine_id", machineId);
  formData.append("machine_type", machineType);
  formData.append("source_kind", "uploaded");
  formData.append("record_to_db", "true");
  formData.append("operator_note", operatorNote);

  const res = await fetch(`${API_BASE}/api/v1/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.message || `HTTP ${res.status}: Upload failed`);
  }
  return await res.json();
}

export async function fetchRecentRunsWithStatus(limit: number = 50): Promise<{
  runs: AuditRun[];
  persistence?: PersistenceStatus;
}> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/runs?limit=${limit}`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error("Failed to fetch runs");
    const data = await res.json();
    return { runs: data.runs || [], persistence: data.persistence };
  } catch {
    return { runs: [] };
  }
}

export async function fetchRecentRuns(limit: number = 20): Promise<AuditRun[]> {
  const res = await fetchRecentRunsWithStatus(limit);
  return res.runs;
}

export async function triggerCloudSync(): Promise<PersistenceStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/runs/sync`, {
      method: "POST",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.persistence || null;
  } catch {
    return null;
  }
}

export async function generateCopilotWorkOrder(
  result: AnalysisResult,
  machineId: string = "fan_id_00"
): Promise<WorkOrder> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/copilot/work-order`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        machine_id: machineId,
        machine_type: result.machine_type || "fan",
        decision_status: result.status,
      }),
      signal: AbortSignal.timeout(12000),
    });
    if (!res.ok) throw new Error("Failed to generate work order");
    const data = await res.json();
    return data.work_order || data;
  } catch (err) {
    return createSyntheticWorkOrder(result, machineId);
  }
}

export async function sendCopilotChat(
  messages: ChatMessage[],
  result?: AnalysisResult,
  machineId: string = "fan_id_00"
): Promise<string> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/copilot/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        machine_id: machineId,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
        analysis_result: result,
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error("Chat failed");
    const data = await res.json();
    return data.reply;
  } catch {
    return (
      `**[Offline SOP Mode]** Autonomous reliability advisory for ${machineId}:\n\n` +
      `Diagnostic status: **${result?.status.toUpperCase() || "NORMAL"}**. ` +
      `Please observe ISO 10816-3 Class II vibration guidelines and execute OSHA 1910.147 Lockout/Tagout prior to mechanical inspection.`
    );
  }
}

export function connectWebSocket(
  machineId: string,
  onMessage: (msg: WebSocketMessage) => void,
  onError?: (err: Event) => void
): { socket: WebSocket; sendChunk: (chunk: Int16Array) => void; close: () => void } {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  // If Vite proxy is active, '/ws' proxies to ws://127.0.0.1:8000/ws/v1/audio-stream
  const wsUrl = `${protocol}//${host}/ws/v1/audio-stream?machine_id=${encodeURIComponent(machineId)}`;

  const socket = new WebSocket(wsUrl);
  socket.binaryType = "arraybuffer";

  socket.onmessage = (event) => {
    try {
      const data: WebSocketMessage = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.warn("Non-JSON WebSocket message:", e);
    }
  };

  if (onError) socket.onerror = onError;

  return {
    socket,
    sendChunk: (chunk: Int16Array) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(chunk.buffer);
      }
    },
    close: () => {
      socket.close();
    },
  };
}

// ---------------------------------------------------------------------------
// Resilient Synthetic Fallback Generators
// ---------------------------------------------------------------------------

function createSyntheticFallback(scenarioId: string): AnalysisResponse {
  const match = FALLBACK_SCENARIOS.find((s) => s.id === scenarioId) || FALLBACK_SCENARIOS[0];
  const isNormal = match.expected_status === "normal";
  const isAnomaly = match.expected_status === "possible_anomaly";

  const mType = match.machine_type || "fan";
  return {
    run_id: `offline-${Date.now()}`,
    machine_id: match.machine_id || `${mType}_id_00`,
    filename: match.filename,
    processing_ms: 18,
    analysis: {
      status: match.expected_status as any,
      status_label: isNormal
        ? "Normal Acoustic Operation"
        : isAnomaly
        ? "Possible Acoustic Anomaly Detected"
        : "Operational Review Required",
      explanation: match.description,
      recommended_action: isNormal
        ? "Continue regular scheduled monitoring."
        : isAnomaly
        ? `Schedule mechanical inspection for ${mType} assembly.`
        : "Check acoustic sensor coupling and ambient factory noise.",
      machine_id: match.machine_id || `${mType}_id_00`,
      machine_type: mType,
      quality: {
        passed: match.expected_status !== "needs_review",
        rms_level: match.id === "scenario_03" ? 0.00001 : 0.038,
        is_silent: match.id === "scenario_03",
        clipping_ratio: match.id === "scenario_04" ? 0.22 : 0.0,
        is_clipped: match.id === "scenario_04",
        duration_seconds: 10.0,
        is_duration_valid: true,
        warnings: match.id === "scenario_05" ? ["Latent domain shift detected"] : [],
        errors: match.id === "scenario_03" ? ["Silent sensor failure"] : [],
      },
      model_prediction: {
        model_type: "AcousticAutoencoder",
        anomaly_score: isAnomaly ? 0.0068 : 0.0019,
        threshold: 0.0032,
        is_anomaly: isAnomaly,
        segment_scores: [0.002, 0.004, 0.005],
      },
      familiarity: {
        score: match.id === "scenario_05" ? 14.8 : 3.2,
        threshold: 9.25,
        is_familiar: match.id !== "scenario_05",
      },
      metadata: {
        filename: match.filename,
        sample_rate: 16000,
        channels: 1,
        duration_seconds: 10.0,
      },
    },
  };
}

function createSyntheticWorkOrder(result: AnalysisResult, machineId: string): WorkOrder {
  const isAnomaly = result.status === "possible_anomaly";
  return {
    work_order_id: `WO-FAN-2026-${Math.floor(Math.random() * 8999 + 1000)}`,
    asset_id: machineId,
    diagnostic_status: result.status,
    urgency: isAnomaly ? "PRIORITY_INSPECTION" : "ROUTINE",
    suspected_mechanism: isAnomaly
      ? "Bearing outer race micro-spalling or mechanical imbalance"
      : "Nominal mechanical baseline condition",
    iso_severity_zone: isAnomaly ? "Zone C (Unsatisfactory)" : "Zone A (Good)",
    recommended_steps: [
      "Initiate standard OSHA 1910.147 Lockout/Tagout on 480V motor supply.",
      "Verify zero-mechanical-speed and install shaft rotation block.",
      "Inspect drive-end and non-drive-end bearing housings with dial indicator.",
      "Perform high-frequency acoustic strobe check on fan belt tension.",
    ],
    safety_checklist: [
      "Lockout/Tagout 480V Disconnect",
      "Dissipate residual kinetic energy",
      "Safety Glasses & Steel Toe PPE",
      "Hearing Protection (Grade 4)",
    ],
    required_tools: [
      "Contact acoustic stethoscope probe",
      "Laser tachometer",
      "IR thermal pyrometer",
      "Torque wrench set (10-120 Nm)",
    ],
    required_ppe: [
      "OSHA NFPA 70E Arc Flash face shield",
      "ANSI Z87.1 safety glasses",
      "Class 1 ear defenders (NRR 27dB)",
    ],
    engineering_summary:
      "Acoustic reconstruction error indicates mechanical irregularity. Precautionary bearing inspection is required to prevent secondary motor fatigue.",
    ai_engine: "PulseForge Deterministic SOP Engine",
    created_at: new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC",
  };
}
