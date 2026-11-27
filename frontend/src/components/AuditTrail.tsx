import React, { useEffect, useState } from "react";
import { Database, Download, RefreshCw, Search } from "lucide-react";
import { AuditRun, PersistenceStatus } from "../types";
import { fetchRecentRunsWithStatus, triggerCloudSync } from "../services/api";

export const AuditTrail: React.FC = () => {
  const [runs, setRuns] = useState<AuditRun[]>([]);
  const [persistence, setPersistence] = useState<PersistenceStatus | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const loadRuns = async () => {
    setLoading(true);
    try {
      const data = await fetchRecentRunsWithStatus(50);
      setRuns(data.runs);
      setPersistence(data.persistence);
    } catch (e) {
      console.error("Failed to load audit history:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleManualSync = async () => {
    setLoading(true);
    try {
      const synced = await triggerCloudSync();
      if (synced) setPersistence(synced);
      const data = await fetchRecentRunsWithStatus(50);
      setRuns(data.runs);
      if (data.persistence) setPersistence(data.persistence);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, []);

  const filteredRuns = runs.filter((r) => {
    const matchesStatus = filterStatus === "all" || r.decision_status === filterStatus;
    const matchesSearch =
      r.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.machine_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.input_label.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStatus && matchesSearch;
  });

  const exportCsv = () => {
    if (runs.length === 0) return;
    const headers = [
      "ID",
      "Timestamp",
      "Machine ID",
      "Label",
      "Source",
      "Anomaly Score",
      "Threshold",
      "Decision Status",
      "Quality",
      "Latency (ms)",
    ];
    const rows = runs.map((r) => [
      r.id,
      r.created_at,
      r.machine_id,
      r.input_label,
      r.source_kind,
      r.anomaly_score.toFixed(4),
      r.threshold.toFixed(4),
      r.decision_status,
      r.quality_status,
      r.processing_ms,
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `pulseforge_audit_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-industrial-card border border-industrial-border rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-mono font-semibold border ${
                persistence?.mode === "supabase"
                  ? "bg-emerald-950/70 text-emerald-300 border-emerald-500/40"
                  : persistence?.mode === "offline_sqlite"
                  ? "bg-amber-950/70 text-amber-300 border-amber-500/40"
                  : "bg-gray-800 text-gray-300 border-gray-700"
              }`}
            >
              {persistence?.label || "SQLite Audit Engine"}
            </span>
            {persistence?.pending_sync_count ? (
              <span className="text-[11px] font-mono text-amber-300">
                ({persistence.pending_sync_count} pending sync)
              </span>
            ) : null}
            <span className="text-xs font-mono text-gray-400">
              Records: <strong className="text-white">{runs.length}</strong>
            </span>
          </div>
          <h1 className="text-xl font-bold text-white mt-1">Inspection Audit Trail</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Immutable inspection log recording technical scores, duration, quality gates, and technician actions.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {persistence?.cloud_configured && (
            <button
              onClick={handleManualSync}
              disabled={loading}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-blue-600/20 border border-blue-500/40 text-xs font-medium text-blue-200 hover:bg-blue-600/30 transition"
            >
              <Database className="h-3.5 w-3.5" />
              <span>Sync Supabase</span>
            </button>
          )}
          <button
            onClick={loadRuns}
            disabled={loading}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-industrial-cardHover border border-industrial-border text-xs font-medium text-gray-300 hover:text-white transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={exportCsv}
            disabled={runs.length === 0}
            className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold transition shadow-sm"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-72">
          <Search className="h-4 w-4 text-gray-400 absolute left-3 top-3" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Run ID or Asset..."
            className="w-full bg-industrial-card border border-industrial-border text-xs rounded-lg pl-9 pr-3 py-2 text-white focus:outline-none focus:border-emerald-500 font-mono"
          />
        </div>

        <div className="flex items-center space-x-1.5 text-xs font-mono self-end sm:self-auto">
          {["all", "normal", "possible_anomaly", "needs_review"].map((status) => (
            <button
              key={status}
              onClick={() => setFilterStatus(status)}
              className={`px-3 py-1.5 rounded-lg border uppercase transition ${
                filterStatus === status
                  ? "bg-industrial-cardHover text-white border-gray-600 font-bold"
                  : "bg-transparent text-gray-400 border-industrial-border hover:text-white"
              }`}
            >
              {status.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Audit Table */}
      <div className="bg-industrial-card border border-industrial-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#0E1420] border-b border-industrial-border text-gray-400 text-[10px] uppercase">
              <tr>
                <th className="py-3 px-4">Run ID</th>
                <th className="py-3 px-4">Timestamp (UTC)</th>
                <th className="py-3 px-4">Asset ID</th>
                <th className="py-3 px-4">Input Label</th>
                <th className="py-3 px-4">Score / Thr</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Quality</th>
                <th className="py-3 px-4 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-industrial-border text-gray-300">
              {filteredRuns.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-gray-400">
                    No inspection runs recorded yet. Evaluate a demo scenario to generate audit records.
                  </td>
                </tr>
              ) : (
                filteredRuns.map((r) => {
                  const isNormal = r.decision_status === "normal";
                  const isAnomaly = r.decision_status === "possible_anomaly";

                  return (
                    <tr key={r.id} className="hover:bg-industrial-cardHover/40 transition">
                      <td className="py-3 px-4 font-semibold text-white">
                        {r.id.substring(0, 8)}...
                      </td>
                      <td className="py-3 px-4 text-gray-400">{r.created_at.substring(0, 19)}</td>
                      <td className="py-3 px-4">{r.machine_id}</td>
                      <td className="py-3 px-4 text-gray-200 max-w-[150px] truncate">
                        {r.input_label}
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-white font-bold">{r.anomaly_score.toFixed(4)}</span>
                        <span className="text-gray-400 text-[10px]"> / {r.threshold.toFixed(4)}</span>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-[9px] px-2 py-0.5 rounded font-bold uppercase ${
                            isNormal
                              ? "bg-emerald-950 text-emerald-300 border border-emerald-500/40"
                              : isAnomaly
                              ? "bg-red-950 text-red-300 border border-red-500/40"
                              : "bg-amber-950 text-amber-300 border border-amber-500/40"
                          }`}
                        >
                          {r.decision_status.replace("_", " ")}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-[9px] px-1.5 py-0.5 rounded ${
                            r.quality_status === "good"
                              ? "text-emerald-400"
                              : r.quality_status === "warning"
                              ? "text-amber-400"
                              : "text-red-400"
                          }`}
                        >
                          {r.quality_status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-medium text-gray-400">
                        {r.processing_ms} ms
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
