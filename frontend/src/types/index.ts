/**
 * Domain Type Definitions for PulseForge AI Frontend.
 * Strictly aligned with backend schemas in src/schemas.py and src/api.py.
 */

export type DecisionStatus = "normal" | "possible_anomaly" | "needs_review" | "invalid_input";

export interface QualityReport {
  passed: boolean;
  rms_level: number;
  is_silent: boolean;
  clipping_ratio: number;
  is_clipped: boolean;
  duration_seconds: number;
  is_duration_valid: boolean;
  warnings: string[];
  errors: string[];
}

export interface ModelPrediction {
  model_type: string;
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  segment_scores?: number[];
}

export interface FamiliarityReport {
  score: number;
  threshold: number;
  is_familiar: boolean;
}

export interface AudioMetadata {
  filename?: string;
  sample_rate: number;
  channels: number;
  duration_seconds: number;
  size_bytes?: number;
}

export interface AnalysisResult {
  status: DecisionStatus;
  status_label: string;
  quality: QualityReport;
  model_prediction?: ModelPrediction | null;
  familiarity?: FamiliarityReport | null;
  metadata?: AudioMetadata | null;
  explanation: string;
  recommended_action: string;
  machine_id: string;
  machine_type: string;
  timestamp?: string;
}

export interface AnalysisResponse {
  run_id: string;
  machine_id: string;
  filename: string;
  analysis: AnalysisResult;
  processing_ms: number;
}

export interface DemoScenario {
  id: string;
  filename: string;
  name: string;
  machine_id: string;
  condition: string;
  description: string;
  expected_status: string;
  file_path: string;
}

export interface TelemetryPacket {
  rms_level: number;
  clipping_ratio: number;
  spectral_bands: {
    low_0_2khz: number;
    mid_2_5khz: number;
    high_5_8khz: number;
  };
  quality_pass: boolean;
}

export interface WebSocketMessage {
  type: "welcome" | "telemetry" | "analysis_result" | "pong" | "error";
  client_id?: string;
  machine_id?: string;
  frame_index?: number;
  telemetry?: TelemetryPacket;
  analysis?: AnalysisResult;
  scenario?: string;
  detail?: string;
}

export interface WorkOrder {
  work_order_id: string;
  asset_id: string;
  diagnostic_status: string;
  urgency: "ROUTINE" | "PRIORITY_INSPECTION" | "IMMEDIATE_ACTION";
  suspected_mechanism: string;
  iso_severity_zone: string;
  recommended_steps: string[];
  safety_checklist: string[];
  required_tools: string[];
  required_ppe: string[];
  engineering_summary: string;
  ai_engine?: string;
  created_at?: string;
  warning?: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
}

export interface AuditRun {
  id: string;
  created_at: string;
  machine_type: string;
  machine_id: string;
  input_label: string;
  source_kind: string;
  audio_duration_seconds: number;
  sample_rate_hz: number;
  anomaly_score: number;
  threshold: number;
  decision_status: DecisionStatus;
  quality_status: string;
  quality_reasons: string[];
  unfamiliarity_score?: number | null;
  processing_ms: number;
  operator_note?: string;
}

export interface FleetMachine {
  id: string;
  name: string;
  type: string;
  ratedRpm: number;
  powerKw: number;
  vibrationZone: "Zone A" | "Zone B" | "Zone C" | "Zone D";
  status: DecisionStatus;
  lastInspection: string;
  score: number;
  threshold: number;
}
