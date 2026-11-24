import React from "react";
import {
  Activity,
  Cpu,
  Database,
  Radio,
  Wrench,
  Wifi,
  WifiOff,
} from "lucide-react";

interface NavbarProps {
  activeTab: "diagnostic" | "streaming" | "fleet" | "audit";
  setActiveTab: (tab: "diagnostic" | "streaming" | "fleet" | "audit") => void;
  machineId: string;
  setMachineId: (id: string) => void;
  backendOnline: boolean;
  latencyMs: number;
  onOpenCopilot: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  machineId,
  setMachineId,
  backendOnline,
  latencyMs,
  onOpenCopilot,
}) => {
  const machines = [
    { id: "fan_id_00", label: "Centrifugal Fan #00 (MIMII Primary)" },
    { id: "fan_id_02", label: "Centrifugal Fan #02 (Line B)" },
    { id: "fan_id_04", label: "Exhaust Blower #04 (Line C)" },
    { id: "fan_id_06", label: "Cooling Fan #06 (Line D)" },
  ];

  return (
    <header className="border-b border-industrial-border bg-[#0B0F17]/90 backdrop-blur sticky top-0 z-40 px-4 lg:px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand & Subtitle */}
        <div className="flex items-center space-x-3 w-full md:w-auto justify-between md:justify-start">
          <div className="flex items-center space-x-2.5">
            <div className="h-9 w-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <Activity className="h-5 w-5 animate-pulse-subtle" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold tracking-tight text-white text-lg">
                  PulseForge<span className="text-emerald-400">.AI</span>
                </span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-semibold">
                  ISO 13374
                </span>
              </div>
              <p className="text-xs text-gray-400 hidden sm:block">
                Predictive Plant Acoustic Anomaly Engine
              </p>
            </div>
          </div>

          {/* Mobile Copilot Trigger */}
          <button
            onClick={onOpenCopilot}
            className="md:hidden flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-600/20 border border-purple-500/40 text-purple-300 hover:bg-purple-600/30 transition"
          >
            <Wrench className="h-3.5 w-3.5 text-purple-400" />
            <span>Copilot</span>
          </button>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-industrial-card/80 p-1 rounded-xl border border-industrial-border text-xs w-full md:w-auto justify-center overflow-x-auto">
          <button
            onClick={() => setActiveTab("diagnostic")}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg font-medium transition ${
              activeTab === "diagnostic"
                ? "bg-industrial-cardHover text-white shadow-sm border border-gray-700"
                : "text-gray-400 hover:text-white hover:bg-industrial-cardHover/50"
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-blue-400" />
            <span>Diagnostic Studio</span>
          </button>

          <button
            onClick={() => setActiveTab("streaming")}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg font-medium transition ${
              activeTab === "streaming"
                ? "bg-industrial-cardHover text-white shadow-sm border border-gray-700"
                : "text-gray-400 hover:text-white hover:bg-industrial-cardHover/50"
            }`}
          >
            <Radio className="h-3.5 w-3.5 text-amber-400" />
            <span>Live Stream</span>
          </button>

          <button
            onClick={() => setActiveTab("fleet")}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg font-medium transition ${
              activeTab === "fleet"
                ? "bg-industrial-cardHover text-white shadow-sm border border-gray-700"
                : "text-gray-400 hover:text-white hover:bg-industrial-cardHover/50"
            }`}
          >
            <Activity className="h-3.5 w-3.5 text-emerald-400" />
            <span>Fleet Health</span>
          </button>

          <button
            onClick={() => setActiveTab("audit")}
            className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg font-medium transition ${
              activeTab === "audit"
                ? "bg-industrial-cardHover text-white shadow-sm border border-gray-700"
                : "text-gray-400 hover:text-white hover:bg-industrial-cardHover/50"
            }`}
          >
            <Database className="h-3.5 w-3.5 text-gray-400" />
            <span>Audit Trail</span>
          </button>
        </nav>

        {/* Right Section: Machine Selector & Status & Copilot */}
        <div className="flex items-center space-x-2.5 w-full md:w-auto justify-end">
          {/* Asset Select */}
          <select
            value={machineId}
            onChange={(e) => setMachineId(e.target.value)}
            className="bg-industrial-card border border-industrial-border text-xs rounded-lg px-2.5 py-1.5 text-gray-200 focus:outline-none focus:border-emerald-500 font-mono"
          >
            {machines.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>

          {/* Backend Health Pill */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-mono ${
              backendOnline
                ? "bg-emerald-950/60 border-emerald-500/40 text-emerald-300"
                : "bg-amber-950/60 border-amber-500/40 text-amber-300"
            }`}
            title={backendOnline ? `FastAPI Backend online (${latencyMs}ms)` : "Running standalone demo mode"}
          >
            {backendOnline ? (
              <Wifi className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <WifiOff className="h-3.5 w-3.5 text-amber-400" />
            )}
            <span className="hidden sm:inline">
              {backendOnline ? `${latencyMs}ms` : "Offline SOP"}
            </span>
          </div>

          {/* Desktop Copilot Button */}
          <button
            onClick={onOpenCopilot}
            className="hidden md:flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-purple-600/20 border border-purple-500/40 text-purple-200 hover:bg-purple-600/30 transition shadow-sm hover:shadow-purple-500/10"
          >
            <Wrench className="h-3.5 w-3.5 text-purple-400" />
            <span>Copilot</span>
          </button>
        </div>
      </div>
    </header>
  );
};
