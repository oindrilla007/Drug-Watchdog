"use client";

import { useState } from "react";
import { AlertResult } from "@/lib/api";
import ReportViewer from "./ReportViewer";

const SEV_CLASSES: Record<number, string> = {
  0: "severity-none",
  1: "severity-mild",
  2: "severity-moderate",
  3: "severity-severe",
};

const SEV_LABELS: Record<number, string> = {
  0: "No interaction",
  1: "Mild",
  2: "Moderate",
  3: "Severe",
};

const SEV_EMOJIS: Record<number, string> = {
  0: "🟢",
  1: "🟡",
  2: "🟠",
  3: "🔴",
};

interface Props {
  alert: AlertResult;
  defaultMode?: "patient" | "clinician";
  onAcknowledge?: () => void;
}

export default function AlertCard({ alert, defaultMode = "patient", onAcknowledge }: Props) {
  const [expanded, setExpanded] = useState(false);
  const sev = Math.min(alert.final_severity ?? alert.severity ?? 0, 3);
  const cls = SEV_CLASSES[sev] ?? SEV_CLASSES[0];

  return (
    <div className={`rounded-xl border-2 p-4 ${cls} transition-all`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xl leading-none">{SEV_EMOJIS[sev]}</span>
          <span className="font-semibold text-sm sm:text-base">
            {alert.drug_a}
          </span>
          <span className="text-xs opacity-60">+</span>
          <span className="font-semibold text-sm sm:text-base">
            {alert.drug_b}
          </span>
        </div>
        <span className="text-xs font-bold uppercase tracking-wide shrink-0 px-2 py-1 rounded-full bg-white/40">
          {alert.severity_label || SEV_LABELS[sev]}
        </span>
      </div>

      {/* Mechanism brief */}
      {alert.mechanism && (
        <p className="mt-2 text-sm opacity-80 leading-snug line-clamp-2">
          {alert.mechanism}
        </p>
      )}

      {/* Confidence + urgency row */}
      <div className="mt-2 flex items-center gap-4 text-xs opacity-70 flex-wrap">
        <span>Confidence: {(alert.confidence * 100).toFixed(0)}%</span>
        {alert.urgency && <span>Urgency: {alert.urgency}</span>}
        {alert.adjustment_reason && (
          <span className="italic">Adj: {alert.adjustment_reason}</span>
        )}
      </div>

      {/* Expand / collapse */}
      <div className="mt-3 flex items-center gap-3 flex-wrap">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs underline underline-offset-2 opacity-70 hover:opacity-100 transition"
        >
          {expanded ? "Hide details" : "Show details"}
        </button>
        {onAcknowledge && (
          <button
            onClick={onAcknowledge}
            className="text-xs px-3 py-1 rounded-full bg-white/50 hover:bg-white/70 transition font-medium"
          >
            Acknowledge
          </button>
        )}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="mt-3 pt-3 border-t border-current/20">
          <ReportViewer
            clinicianReport={alert.clinician_report}
            patientReport={alert.patient_report}
            defaultMode={defaultMode}
          />
          {alert.citations.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-semibold opacity-60 uppercase tracking-wide mb-1">
                Citations
              </p>
              <ul className="text-xs space-y-0.5 opacity-70">
                {alert.citations.map((c) => (
                  <li key={c} className="font-mono">
                    [{c}]
                  </li>
                ))}
              </ul>
            </div>
          )}
          {alert.routed_to.length > 0 && (
            <div className="mt-2 text-xs opacity-60">
              Routed to: {alert.routed_to.join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
