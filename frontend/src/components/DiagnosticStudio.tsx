import React, { useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  FileAudio,
  HelpCircle,
  Play,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { AnalysisResult, DemoScenario, MachineType } from "../types";
import { AudioVisualizer } from "./AudioVisualizer";

interface DiagnosticStudioProps {
  scenarios: DemoScenario[];
  activeScenarioId: string;
  machineType: MachineType;
  onSelectScenario: (scenarioId: string) => void;
  onUploadFile: (file: File) => void;
  analysis: AnalysisResult | null;
  loading: boolean;
  onOpenWorkOrder: () => void;
}

export const DiagnosticStudio: React.FC<DiagnosticStudioProps> = ({
  scenarios,
  activeScenarioId,
  machineType,
  onSelectScenario,
  onUploadFile,
  analysis,
  loading,
  onOpenWorkOrder,
}) => {
  const [dragActive, setDragActive] = useState<boolean>(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
  };

  // Status Styling & Badges
  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "normal":
        return {
          bg: "bg-emerald-950/80 border-emerald-500/50 text-emerald-300",
          icon: <CheckCircle2 className="h-5 w-5 text-emerald-400" />,
          title: "NORMAL ACOUSTIC BASELINE",
          desc: "Passed physical quality gates and Autoencoder reconstruction threshold.",
        };
      case "possible_anomaly":
        return {
          bg: "bg-red-950/80 border-red-500/60 text-red-200",
          icon: <AlertTriangle className="h-5 w-5 text-red-400" />,
          title: "POSSIBLE ACOUSTIC ANOMALY DETECTED",
          desc: "Reconstruction error exceeds validation baseline. Unusual spectral pattern detected.",
        };
      case "needs_review":
        return {
          bg: "bg-amber-950/80 border-amber-500/60 text-amber-200",
          icon: <AlertCircle className="h-5 w-5 text-amber-400" />,
          title: "OPERATIONAL REVIEW REQUIRED",
          desc: "Degraded signal quality or acoustic domain shift detected. Human review required.",
        };
      default:
        return {
          bg: "bg-gray-900 border-gray-700 text-gray-300",
          icon: <HelpCircle className="h-5 w-5 text-gray-400" />,
          title: "INPUT VERIFICATION REQUIRED",
          desc: "Upload standard 16-bit PCM WAV audio or select a demonstration recording.",
        };
    }
  };

  const statusConfig = getStatusBadge(analysis?.status);

  return (
    <div className="space-y-6">
      {/* Top Banner / Heading */}
      <div className="bg-gradient-to-r from-industrial-card via-[#161F30] to-industrial-card border border-industrial-border rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-blue-950/80 text-blue-300 border border-blue-500/30">
                Predictive Plant Track
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-500/30 uppercase">
                Active Engine: {machineType}
              </span>
            </div>
            <h1 className="text-xl font-bold text-white mt-1">
              Acoustic Diagnostic Studio
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Multi-stage acoustic anomaly screening: Signal Quality Gates → 128-band Log-Mel → PyTorch Autoencoder → Mahalanobis Familiarity.
            </p>
          </div>

          {analysis && analysis.status === "possible_anomaly" && (
            <button
              onClick={onOpenWorkOrder}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-red-600/30 hover:bg-red-600/40 border border-red-500 text-red-200 text-xs font-semibold transition shadow-lg shadow-red-950/40"
            >
              <AlertTriangle className="h-4 w-4 text-red-400" />
              <span>Generate OSHA Work Order</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Grid: Demo Scenarios (Left) & Upload Box (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Scenario Quick Selector (2 Columns) */}
        <div className="lg:col-span-2 bg-industrial-card border border-industrial-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-gray-300 font-mono flex items-center space-x-1.5">
              <span>Curated Industrial Scenarios (MIMII Benchmark)</span>
            </h2>
            <span className="text-[11px] text-gray-400 font-mono">1-Click Evaluation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {scenarios.map((s) => {
              const active = s.id === activeScenarioId;
              const isNormal = s.expected_status === "normal";
              const isAnomaly = s.expected_status === "possible_anomaly";

              return (
                <button
                  key={s.id}
                  onClick={() => onSelectScenario(s.id)}
                  disabled={loading}
                  className={`text-left p-3 rounded-lg border transition flex flex-col justify-between ${
                    active
                      ? "bg-industrial-cardHover border-emerald-500 shadow-md shadow-emerald-950/20 ring-1 ring-emerald-500"
                      : "bg-[#0E1420] border-industrial-border hover:border-gray-600 hover:bg-industrial-cardHover/50"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <span className="font-semibold text-xs text-white line-clamp-1">
                      {s.name}
                    </span>
                    <span
                      className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ml-2 whitespace-nowrap ${
                        isNormal
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                          : isAnomaly
                          ? "bg-red-950 text-red-300 border border-red-500/40"
                          : "bg-amber-950 text-amber-300 border border-amber-500/40"
                      }`}
                    >
                      {s.expected_status.replace("_", " ")}
                    </span>
                  </div>

                  <p className="text-[11px] text-gray-400 line-clamp-2 mt-1.5">
                    {s.description}
                  </p>

                  <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-800/80 text-[10px] font-mono text-gray-400">
                    <span>{s.filename}</span>
                    <span className="text-emerald-400 font-medium flex items-center space-x-1">
                      <Play className="h-2.5 w-2.5" />
                      <span>Evaluate</span>
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* WAV Upload Box (1 Column) */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`bg-industrial-card border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center text-center transition ${
            dragActive
              ? "border-emerald-500 bg-emerald-950/10"
              : "border-industrial-border hover:border-gray-600"
          }`}
        >
          <div className="h-12 w-12 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-3">
            <Upload className="h-5 w-5" />
          </div>
          <h3 className="font-semibold text-sm text-white">Upload Fan Recording</h3>
          <p className="text-xs text-gray-400 mt-1 max-w-[200px]">
            Drag & drop standard 16-bit PCM WAV (10s @ 16 kHz recommended)
          </p>

          <label className="mt-4 px-4 py-2 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40 text-blue-200 text-xs font-medium cursor-pointer transition">
            <span>Browse Computer</span>
            <input
              type="file"
              accept=".wav,audio/wav"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {/* Primary Result & Telemetry Gauges */}
      {analysis && (
        <div className="space-y-4">
          {/* Decision Status Card */}
          <div className={`p-4 rounded-xl border flex items-start space-x-3.5 ${statusConfig.bg}`}>
            <div className="mt-0.5">{statusConfig.icon}</div>
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <h3 className="font-bold text-sm tracking-wide font-mono">
                  {statusConfig.title}
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-black/40 border border-white/10 uppercase">
                  Asset: {analysis.machine_id}
                </span>
              </div>
              <p className="text-xs mt-1 text-gray-300 leading-relaxed">
                {analysis.explanation}
              </p>
              <div className="mt-2 text-xs font-mono font-medium text-white/90 bg-black/30 p-2 rounded border border-white/5 flex items-center space-x-2">
                <span className="text-emerald-400 font-bold uppercase">Action:</span>
                <span>{analysis.recommended_action}</span>
              </div>
            </div>
          </div>

          {/* Oscilloscope Waveform */}
          <AudioVisualizer
            rmsLevel={analysis.quality.rms_level}
            clippingRatio={analysis.quality.clipping_ratio}
          />

          {/* Metric Telemetry Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Autoencoder Reconstruction Anomaly Score */}
            <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
              <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold block">
                Autoencoder Reconstruction Error (MSE)
              </span>
              <div className="flex items-baseline space-x-2 mt-2">
                <span className="text-2xl font-bold font-mono text-white">
                  {analysis.model_prediction?.anomaly_score !== undefined
                    ? analysis.model_prediction.anomaly_score.toFixed(4)
                    : "0.0000"}
                </span>
                <span className="text-xs font-mono text-gray-400">
                  / thr: {analysis.model_prediction?.threshold.toFixed(4) || "0.0032"}
                </span>
              </div>
              {/* Score bar */}
              <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    (analysis.model_prediction?.anomaly_score || 0) >
                    (analysis.model_prediction?.threshold || 0.0032)
                      ? "bg-red-500"
                      : "bg-emerald-500"
                  }`}
                  style={{
                    width: `${Math.min(
                      100,
                      ((analysis.model_prediction?.anomaly_score || 0) /
                        ((analysis.model_prediction?.threshold || 0.0032) * 2)) *
                        100
                    )}%`,
                  }}
                />
              </div>
              <p className="text-[11px] text-gray-400 mt-2">
                Trained on normal fan baseline; elevated MSE indicates acoustic anomaly.
              </p>
            </div>

            {/* Latent Mahalanobis Familiarity Distance */}
            <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
              <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold block">
                Latent Familiarity (Mahalanobis Distance)
              </span>
              <div className="flex items-baseline space-x-2 mt-2">
                <span className="text-2xl font-bold font-mono text-white">
                  {analysis.familiarity?.score !== undefined
                    ? analysis.familiarity.score.toFixed(2)
                    : "0.00"}
                </span>
                <span className="text-xs font-mono text-gray-400">
                  / thr: {analysis.familiarity?.threshold.toFixed(2) || "9.25"}
                </span>
              </div>
              {/* Familiarity bar */}
              <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    (analysis.familiarity?.score || 0) >
                    (analysis.familiarity?.threshold || 9.25)
                      ? "bg-amber-500"
                      : "bg-blue-500"
                  }`}
                  style={{
                    width: `${Math.min(
                      100,
                      ((analysis.familiarity?.score || 0) /
                        ((analysis.familiarity?.threshold || 9.25) * 1.5)) *
                        100
                    )}%`,
                  }}
                />
              </div>
              <p className="text-[11px] text-gray-400 mt-2">
                Gates against out-of-domain plant background noise and forklift horns.
              </p>
            </div>

            {/* Signal Quality Gate Report */}
            <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold">
                  Physical Quality Screening
                </span>
                <span
                  className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                    analysis.quality.passed
                      ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                      : "bg-red-950 text-red-300 border border-red-500/40"
                  }`}
                >
                  {analysis.quality.passed ? "PASSED" : "FAILED"}
                </span>
              </div>
              <div className="mt-2.5 space-y-1.5 text-xs font-mono">
                <div className="flex justify-between text-gray-300">
                  <span className="text-gray-400">RMS Amplitude:</span>
                  <span>{analysis.quality.rms_level.toFixed(4)}</span>
                </div>
                <div className="flex justify-between text-gray-300">
                  <span className="text-gray-400">Peak Clipping:</span>
                  <span>{(analysis.quality.clipping_ratio * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-gray-300">
                  <span className="text-gray-400">Duration:</span>
                  <span>{analysis.quality.duration_seconds.toFixed(1)}s (Valid)</span>
                </div>
              </div>
              <p className="text-[11px] text-gray-400 mt-2">
                Screens for dead sensors, pre-amp saturation, and duration boundary violations.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
