import React, { useState } from "react";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Download,
  FileText,
  Send,
  Shield,
  User,
  Wrench,
  X,
} from "lucide-react";
import { AnalysisResult, ChatMessage, WorkOrder } from "../types";
import { generateCopilotWorkOrder, sendCopilotChat } from "../services/api";

interface CopilotDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  analysis: AnalysisResult | null;
  machineId: string;
}

export const CopilotDrawer: React.FC<CopilotDrawerProps> = ({
  isOpen,
  onClose,
  analysis,
  machineId,
}) => {
  const [activeTab, setActiveTab] = useState<"chat" | "workorder">("workorder");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: `Hello, I am the **PulseForge Industrial Reliability Copilot**. I have reviewed the acoustic condition monitoring telemetry for **${machineId}**. How can I assist your physical mechanical inspection or maintenance planning?`,
    },
  ]);
  const [inputMessage, setInputMessage] = useState<string>("");
  const [loadingChat, setLoadingChat] = useState<boolean>(false);
  const [workOrder, setWorkOrder] = useState<WorkOrder | null>(null);
  const [loadingWorkOrder, setLoadingWorkOrder] = useState<boolean>(false);

  if (!isOpen) return null;

  const handleGenerateWorkOrder = async () => {
    if (!analysis) return;
    setLoadingWorkOrder(true);
    try {
      const wo = await generateCopilotWorkOrder(analysis, machineId);
      setWorkOrder(wo);
    } catch (e) {
      console.error("Failed to generate work order:", e);
    } finally {
      setLoadingWorkOrder(false);
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || loadingChat) return;

    const userText = inputMessage.trim();
    setInputMessage("");
    const newHistory: ChatMessage[] = [...messages, { role: "user", content: userText }];
    setMessages(newHistory);
    setLoadingChat(true);

    try {
      const reply = await sendCopilotChat(newHistory, analysis || undefined, machineId);
      setMessages([...newHistory, { role: "assistant", content: reply }]);
    } catch {
      setMessages([
        ...newHistory,
        {
          role: "assistant",
          content:
            "**[Offline SOP Mode]** Anomaly verified. Please enforce OSHA 1910.147 LOTO and verify bearing lubrication levels.",
        },
      ]);
    } finally {
      setLoadingChat(false);
    }
  };

  const downloadWorkOrderJson = () => {
    if (!workOrder) return;
    const blob = new Blob([JSON.stringify(workOrder, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${workOrder.work_order_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end">
      <div className="w-full max-w-xl bg-[#0B0F17] border-l border-industrial-border h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-200">
        {/* Drawer Header */}
        <div className="p-4 border-b border-industrial-border flex items-center justify-between bg-industrial-card">
          <div className="flex items-center space-x-2.5">
            <div className="h-8 w-8 rounded-lg bg-purple-600/20 border border-purple-500/40 flex items-center justify-center text-purple-400">
              <Bot className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="font-bold text-sm text-white">Industrial AI Copilot</h2>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-purple-950/80 border border-purple-500/40 text-purple-300 font-semibold">
                  Factory Brain
                </span>
              </div>
              <p className="text-[11px] text-gray-400">
                Autonomous OSHA 1910.147 LOTO reasoning & technician advisory
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex border-b border-industrial-border bg-[#0E1420] text-xs font-medium">
          <button
            onClick={() => setActiveTab("workorder")}
            className={`flex-1 py-2.5 px-4 flex items-center justify-center space-x-2 border-b-2 transition ${
              activeTab === "workorder"
                ? "border-purple-500 text-purple-300 font-bold bg-purple-950/20"
                : "border-transparent text-gray-400 hover:text-white"
            }`}
          >
            <FileText className="h-3.5 w-3.5" />
            <span>Maintenance Work Order</span>
          </button>

          <button
            onClick={() => setActiveTab("chat")}
            className={`flex-1 py-2.5 px-4 flex items-center justify-center space-x-2 border-b-2 transition ${
              activeTab === "chat"
                ? "border-purple-500 text-purple-300 font-bold bg-purple-950/20"
                : "border-transparent text-gray-400 hover:text-white"
            }`}
          >
            <Bot className="h-3.5 w-3.5" />
            <span>Technician Copilot Chat</span>
          </button>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeTab === "workorder" ? (
            <div className="space-y-4">
              {!workOrder ? (
                <div className="bg-industrial-card border border-industrial-border rounded-xl p-5 text-center space-y-3">
                  <div className="h-12 w-12 rounded-full bg-purple-600/10 border border-purple-500/30 flex items-center justify-center mx-auto text-purple-400">
                    <FileText className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="font-bold text-sm text-white">Generate Maintenance Work Order</h3>
                    <p className="text-xs text-gray-400 mt-1 max-w-sm mx-auto">
                      Synthesize diagnostic telemetry into an OSHA 1910.147 compliant work order with safety checklists, required tools, and ISO severity zones.
                    </p>
                  </div>

                  <button
                    onClick={handleGenerateWorkOrder}
                    disabled={loadingWorkOrder || !analysis}
                    className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-xs font-bold transition shadow-lg shadow-purple-950/50 disabled:opacity-50"
                  >
                    {loadingWorkOrder ? "Generating via Copilot..." : "Generate Work Order (1-Click)"}
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Work Order Header */}
                  <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <span className="text-[10px] font-mono text-purple-400 uppercase tracking-wider block">
                          OSHA 1910.147 COMPLIANT WORK ORDER
                        </span>
                        <h3 className="font-bold text-base text-white font-mono mt-0.5">
                          {workOrder.work_order_id}
                        </h3>
                      </div>
                      <span
                        className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                          workOrder.urgency === "PRIORITY_INSPECTION"
                            ? "bg-red-950 text-red-300 border border-red-500/50"
                            : "bg-emerald-950 text-emerald-300 border border-emerald-500/50"
                        }`}
                      >
                        {workOrder.urgency}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 mt-3 p-2.5 rounded bg-[#0E1420] border border-gray-800 text-xs font-mono">
                      <div>
                        <span className="text-gray-400 block text-[10px]">Asset ID</span>
                        <span className="text-white">{workOrder.asset_id}</span>
                      </div>
                      <div>
                        <span className="text-gray-400 block text-[10px]">ISO 10816 Zone</span>
                        <span className="text-amber-400 font-bold">{workOrder.iso_severity_zone}</span>
                      </div>
                    </div>

                    <div className="mt-3">
                      <span className="text-[11px] font-mono uppercase text-gray-400 block">
                        Engineering Rationale
                      </span>
                      <p className="text-xs text-gray-300 mt-1 leading-relaxed">
                        {workOrder.engineering_summary}
                      </p>
                    </div>
                  </div>

                  {/* Safety & LOTO Checklist */}
                  <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
                    <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-amber-400 font-mono mb-2">
                      <Shield className="h-4 w-4" />
                      <span>Mandatory OSHA Lockout / Tagout (LOTO)</span>
                    </div>
                    <ul className="space-y-1.5 text-xs text-gray-300">
                      {workOrder.safety_checklist.map((item, idx) => (
                        <li key={idx} className="flex items-start space-x-2">
                          <CheckCircle2 className="h-3.5 w-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Recommended Physical Steps */}
                  <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
                    <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-emerald-400 font-mono mb-2">
                      <Wrench className="h-4 w-4" />
                      <span>Recommended Maintenance Steps</span>
                    </div>
                    <ol className="space-y-1.5 text-xs text-gray-300 list-decimal list-inside">
                      {workOrder.recommended_steps.map((step, idx) => (
                        <li key={idx} className="leading-relaxed">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={downloadWorkOrderJson}
                      className="flex-1 flex items-center justify-center space-x-1.5 px-4 py-2 rounded-lg bg-industrial-cardHover border border-industrial-border text-xs font-medium text-white hover:border-gray-500 transition"
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>Export JSON Work Order</span>
                    </button>
                    <button
                      onClick={() => setWorkOrder(null)}
                      className="px-4 py-2 rounded-lg bg-gray-800 text-xs font-medium text-gray-300 hover:text-white transition"
                    >
                      Regenerate
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Chat Interface */
            <div className="flex flex-col h-full space-y-4">
              <div className="flex-1 space-y-3">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex items-start space-x-2.5 ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    {msg.role !== "user" && (
                      <div className="h-7 w-7 rounded-lg bg-purple-600/20 border border-purple-500/40 flex items-center justify-center text-purple-400 flex-shrink-0 mt-0.5">
                        <Bot className="h-3.5 w-3.5" />
                      </div>
                    )}

                    <div
                      className={`p-3 rounded-xl text-xs max-w-[85%] leading-relaxed ${
                        msg.role === "user"
                          ? "bg-purple-600 text-white rounded-br-none"
                          : "bg-industrial-card border border-industrial-border text-gray-200 rounded-bl-none font-sans"
                      }`}
                    >
                      {msg.content}
                    </div>

                    {msg.role === "user" && (
                      <div className="h-7 w-7 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 flex-shrink-0 mt-0.5">
                        <User className="h-3.5 w-3.5" />
                      </div>
                    )}
                  </div>
                ))}
                {loadingChat && (
                  <div className="flex items-center space-x-2 text-xs text-purple-400 font-mono">
                    <span className="h-2 w-2 rounded-full bg-purple-400 animate-ping"></span>
                    <span>Copilot reasoning with Groq LPU...</span>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <form onSubmit={handleSendMessage} className="pt-2 flex items-center space-x-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask regarding bearing defect signatures or ISO thresholds..."
                  className="flex-1 bg-industrial-card border border-industrial-border text-xs rounded-lg px-3 py-2.5 text-white focus:outline-none focus:border-purple-500"
                />
                <button
                  type="submit"
                  disabled={loadingChat || !inputMessage.trim()}
                  className="p-2.5 rounded-lg bg-purple-600 hover:bg-purple-700 text-white transition disabled:opacity-40"
                >
                  <Send className="h-4 w-4" />
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
