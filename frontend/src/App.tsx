import React, { useEffect, useState } from "react";
import { Navbar } from "./components/Navbar";
import { DiagnosticStudio } from "./components/DiagnosticStudio";
import { LiveStreamStudio } from "./components/LiveStreamStudio";
import { FleetHealthBoard } from "./components/FleetHealthBoard";
import { AuditTrail } from "./components/AuditTrail";
import { CopilotDrawer } from "./components/CopilotDrawer";
import { AnalysisResult, DemoScenario } from "./types";
import {
  checkBackendHealth,
  evaluateScenario,
  FALLBACK_SCENARIOS,
  fetchDemoScenarios,
  uploadAudioAnalysis,
} from "./services/api";

export function App() {
  const [activeTab, setActiveTab] = useState<"diagnostic" | "streaming" | "fleet" | "audit">("diagnostic");
  const [machineId, setMachineId] = useState<string>("fan_id_00");
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [scenarios, setScenarios] = useState<DemoScenario[]>(FALLBACK_SCENARIOS);
  const [activeScenarioId, setActiveScenarioId] = useState<string>("scenario_01");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [copilotOpen, setCopilotOpen] = useState<boolean>(false);

  // Poll Backend Health & Scenarios on mount
  useEffect(() => {
    const init = async () => {
      const health = await checkBackendHealth();
      setBackendOnline(health.online);
      setLatencyMs(health.latencyMs);

      const scs = await fetchDemoScenarios();
      setScenarios(scs);

      // Auto-evaluate first scenario for immediate dashboard populated state
      try {
        const res = await evaluateScenario("scenario_01");
        setAnalysis(res.analysis);
      } catch (e) {
        console.error("Initial scenario evaluation failed:", e);
      }
    };

    init();
    const interval = setInterval(async () => {
      const health = await checkBackendHealth();
      setBackendOnline(health.online);
      setLatencyMs(health.latencyMs);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleSelectScenario = async (scenarioId: string) => {
    setActiveScenarioId(scenarioId);
    setLoading(true);
    try {
      const res = await evaluateScenario(scenarioId);
      setAnalysis(res.analysis);
      setMachineId(res.machine_id || "fan_id_00");
    } catch (e) {
      console.error("Scenario evaluation failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadFile = async (file: File) => {
    setLoading(true);
    try {
      const res = await uploadAudioAnalysis(file, machineId);
      setAnalysis(res.analysis);
    } catch (e: any) {
      alert(e.message || "File upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] flex flex-col selection:bg-emerald-500 selection:text-black">
      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        machineId={machineId}
        setMachineId={setMachineId}
        backendOnline={backendOnline}
        latencyMs={latencyMs}
        onOpenCopilot={() => setCopilotOpen(true)}
      />

      {/* Main Workspace View */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6">
        {activeTab === "diagnostic" && (
          <DiagnosticStudio
            scenarios={scenarios}
            activeScenarioId={activeScenarioId}
            onSelectScenario={handleSelectScenario}
            onUploadFile={handleUploadFile}
            analysis={analysis}
            loading={loading}
            onOpenWorkOrder={() => setCopilotOpen(true)}
          />
        )}

        {activeTab === "streaming" && (
          <LiveStreamStudio machineId={machineId} />
        )}

        {activeTab === "fleet" && (
          <FleetHealthBoard
            onSelectMachine={(id) => {
              setMachineId(id);
              setActiveTab("diagnostic");
            }}
          />
        )}

        {activeTab === "audit" && <AuditTrail />}
      </main>

      {/* AI Maintenance Copilot Drawer */}
      <CopilotDrawer
        isOpen={copilotOpen}
        onClose={() => setCopilotOpen(false)}
        analysis={analysis}
        machineId={machineId}
      />

      {/* Bottom Industrial Status Bar */}
      <footer className="border-t border-industrial-border bg-[#0B0F17] py-2.5 px-4 lg:px-6 text-[11px] font-mono text-gray-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center space-x-3">
          <span>PulseForge AI v2.0 (Finals Build)</span>
          <span>•</span>
          <span>ISO 13374 Condition Monitoring</span>
          <span>•</span>
          <span>MIMII Fan Benchmark</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
          <span>Autoencoder Edge Engine Ready</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
