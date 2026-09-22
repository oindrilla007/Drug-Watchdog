"use client";

import { useState } from "react";

interface Props {
  clinicianReport?: string;
  patientReport?: string;
  defaultMode?: "patient" | "clinician";
}

export default function ReportViewer({ clinicianReport, patientReport, defaultMode = "patient" }: Props) {
  const [mode, setMode] = useState<"patient" | "clinician">(defaultMode);

  const text = mode === "clinician" ? clinicianReport : patientReport;

  if (!clinicianReport && !patientReport) return null;

  return (
    <div>
      {/* Toggle */}
      <div className="flex gap-1 mb-2">
        {(["patient", "clinician"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`text-xs px-3 py-1 rounded-full transition font-medium capitalize ${
              mode === m
                ? "bg-white/70 shadow-sm"
                : "bg-white/20 hover:bg-white/40"
            }`}
          >
            {m === "patient" ? "For me" : "Clinical"}
          </button>
        ))}
      </div>

      {/* Report text */}
      {text ? (
        <p className="text-sm leading-relaxed opacity-90 whitespace-pre-wrap">
          {text}
        </p>
      ) : (
        <p className="text-sm opacity-50 italic">No {mode} report available.</p>
      )}
    </div>
  );
}
