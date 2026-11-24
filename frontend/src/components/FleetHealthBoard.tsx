import React from "react";
import { Activity, AlertTriangle, CheckCircle2, ChevronRight, Layers, ShieldCheck } from "lucide-react";
import { FleetMachine } from "../types";

interface FleetHealthBoardProps {
  onSelectMachine: (machineId: string) => void;
}

export const FleetHealthBoard: React.FC<FleetHealthBoardProps> = ({ onSelectMachine }) => {
  const fleet: FleetMachine[] = [
    {
      id: "fan_id_00",
      name: "Centrifugal Ventilation Fan #00",
      type: "Centrifugal Industrial Fan",
      ratedRpm: 1750,
      powerKw: 45.0,
      vibrationZone: "Zone A",
      status: "normal",
      lastInspection: "10 mins ago",
      score: 0.0019,
      threshold: 0.0032,
    },
    {
      id: "fan_id_02",
      name: "Line B Exhaust Blower #02",
      type: "High-Volume Exhaust Fan",
      ratedRpm: 1450,
      powerKw: 30.0,
      vibrationZone: "Zone C",
      status: "possible_anomaly",
      lastInspection: "25 mins ago",
      score: 0.0068,
      threshold: 0.0032,
    },
    {
      id: "fan_id_04",
      name: "Boiler Induced Draft Fan #04",
      type: "Induced Draft Fan",
      ratedRpm: 1800,
      powerKw: 75.0,
      vibrationZone: "Zone B",
      status: "normal",
      lastInspection: "1 hour ago",
      score: 0.0022,
      threshold: 0.0032,
    },
    {
      id: "fan_id_06",
      name: "Secondary Cooling Tower Fan #06",
      type: "Axial Cooling Fan",
      ratedRpm: 900,
      powerKw: 22.0,
      vibrationZone: "Zone B",
      status: "needs_review",
      lastInspection: "3 hours ago",
      score: 0.0028,
      threshold: 0.0032,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-industrial-card border border-industrial-border rounded-xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
                Factory Brain Multi-Asset
              </span>
              <span className="text-xs font-mono text-gray-400">
                Standards: <strong className="text-white">ISO 10816-3 Class II</strong>
              </span>
            </div>
            <h1 className="text-xl font-bold text-white mt-1">Plant Fleet Health Status Board</h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Comprehensive rotating equipment acoustic condition matrix across industrial production lines.
            </p>
          </div>

          <div className="flex items-center space-x-4 text-xs font-mono">
            <div className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500"></span>
              <span className="text-gray-300">Normal (2)</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500"></span>
              <span className="text-gray-300">Review (1)</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
              <span className="text-gray-300">Anomaly (1)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Fleet Asset Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {fleet.map((machine) => {
          const isNormal = machine.status === "normal";
          const isAnomaly = machine.status === "possible_anomaly";

          return (
            <div
              key={machine.id}
              className={`bg-industrial-card rounded-xl border p-5 transition flex flex-col justify-between ${
                isAnomaly
                  ? "border-red-500/60 shadow-lg shadow-red-950/20"
                  : isNormal
                  ? "border-emerald-500/40 shadow-sm"
                  : "border-amber-500/40"
              }`}
            >
              <div>
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider">
                      {machine.id} • {machine.type}
                    </span>
                    <h3 className="font-bold text-base text-white mt-0.5">{machine.name}</h3>
                  </div>

                  <span
                    className={`text-[10px] font-mono font-bold px-2.5 py-1 rounded-full uppercase border ${
                      isNormal
                        ? "bg-emerald-950/80 text-emerald-300 border-emerald-500/40"
                        : isAnomaly
                        ? "bg-red-950/80 text-red-200 border-red-500/50 animate-pulse"
                        : "bg-amber-950/80 text-amber-300 border-amber-500/40"
                    }`}
                  >
                    {machine.status.replace("_", " ")}
                  </span>
                </div>

                {/* Machine Specs & ISO Severity */}
                <div className="grid grid-cols-3 gap-2 mt-4 p-2.5 rounded-lg bg-[#0E1420] border border-gray-800 text-xs font-mono">
                  <div>
                    <span className="text-gray-400 block text-[10px]">Rated RPM</span>
                    <span className="font-semibold text-white">{machine.ratedRpm} RPM</span>
                  </div>
                  <div>
                    <span className="text-gray-400 block text-[10px]">Rated Power</span>
                    <span className="font-semibold text-white">{machine.powerKw} kW</span>
                  </div>
                  <div>
                    <span className="text-gray-400 block text-[10px]">ISO 10816 Zone</span>
                    <span
                      className={`font-bold ${
                        machine.vibrationZone === "Zone A"
                          ? "text-emerald-400"
                          : machine.vibrationZone === "Zone C"
                          ? "text-red-400"
                          : "text-amber-400"
                      }`}
                    >
                      {machine.vibrationZone}
                    </span>
                  </div>
                </div>

                {/* Score vs Threshold */}
                <div className="mt-4">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-gray-400">Reconstruction MSE Score:</span>
                    <span className="text-white font-bold">
                      {machine.score.toFixed(4)}{" "}
                      <span className="text-gray-400 font-normal">/ {machine.threshold.toFixed(4)}</span>
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 h-2 rounded-full mt-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        machine.score > machine.threshold ? "bg-red-500" : "bg-emerald-500"
                      }`}
                      style={{
                        width: `${Math.min(100, (machine.score / (machine.threshold * 2)) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Card Footer / Action */}
              <div className="mt-5 pt-3 border-t border-gray-800 flex items-center justify-between text-xs font-mono">
                <span className="text-gray-400 text-[11px]">Inspected: {machine.lastInspection}</span>
                <button
                  onClick={() => onSelectMachine(machine.id)}
                  className="flex items-center space-x-1 text-emerald-400 hover:text-emerald-300 font-medium transition"
                >
                  <span>Open Diagnostic Studio</span>
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
