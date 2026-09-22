"use client";

import { useEffect, useState } from "react";
import { apiGetAllAlerts, apiAcknowledgeAlert, AlertRecord } from "@/lib/api";
import AlertCard from "@/components/AlertCard";
import { AlertResult } from "@/lib/api";

function recordToResult(r: AlertRecord): AlertResult {
  return {
    drug_a:            r.drug_a,
    drug_b:            r.drug_b,
    severity:          r.severity,
    final_severity:    r.adjusted_severity ?? r.severity,
    severity_label:    r.severity_label ?? "",
    severity_emoji:    ["🟢","🟡","🟠","🔴"][Math.min(r.severity, 3)],
    confidence:        0,
    mechanism:         "",
    adjustment_reason: "",
    clinician_report:  r.clinician_report ?? "",
    patient_report:    r.patient_report ?? "",
    urgency:           "",
    routed_to:         r.routed_to ?? [],
    citations:         [],
    react_iterations:  0,
    error:             "",
  };
}

export default function AlertsPage() {
  const [alerts,  setAlerts]  = useState<AlertRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");
  const [filter,  setFilter]  = useState<"all" | "unread">("unread");

  useEffect(() => {
    apiGetAllAlerts()
      .then((d) => setAlerts(d.alerts))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function acknowledge(id: string) {
    try {
      await apiAcknowledgeAlert(id);
      setAlerts((prev) => prev.map((a) => a.id === id ? { ...a, was_acknowledged: true } : a));
    } catch {
      // silent
    }
  }

  const shown = filter === "unread"
    ? alerts.filter((a) => !a.was_acknowledged)
    : alerts;

  const unreadCount = alerts.filter((a) => !a.was_acknowledged).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-5 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Alert history</h1>
        <p className="text-sm text-gray-500 mt-0.5">All drug interaction alerts from your sessions.</p>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700 text-sm">{error}</div>
      )}

      {/* Filter toggle */}
      <div className="flex gap-2">
        {(["unread", "all"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`text-sm px-4 py-1.5 rounded-full border transition font-medium ${
              filter === f
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"
            }`}
          >
            {f === "unread" ? `Unread (${unreadCount})` : `All (${alerts.length})`}
          </button>
        ))}
      </div>

      {shown.length === 0 ? (
        <div className="rounded-xl border border-gray-200 bg-white p-10 text-center text-gray-400">
          <p className="text-3xl mb-2">🟢</p>
          <p className="font-medium">
            {filter === "unread" ? "No unread alerts" : "No alerts yet"}
          </p>
          <p className="text-sm mt-1">
            {filter === "unread"
              ? "All alerts have been acknowledged."
              : "Run a drug check to see results here."}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {shown.map((a) => (
            <div key={a.id} className="relative">
              {!a.was_acknowledged && (
                <div className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full z-10" />
              )}
              <AlertCard
                alert={recordToResult(a)}
                onAcknowledge={!a.was_acknowledged ? () => acknowledge(a.id) : undefined}
              />
              {a.created_at && (
                <p className="text-xs text-gray-400 mt-1 pl-1">
                  {new Date(a.created_at).toLocaleString()}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
