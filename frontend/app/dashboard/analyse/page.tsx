"use client";

import { useState } from "react";
import { apiAnalyse, apiIntakeImage, AnalyseResponse, IntakeResponse, AlertResult } from "@/lib/api";
import DrugInputPanel from "@/components/DrugInputPanel";
import ImageUploader from "@/components/ImageUploader";
import AlertCard from "@/components/AlertCard";

type InputTab = "text" | "image";

export default function AnalysePage() {
  const [tab,        setTab]        = useState<InputTab>("text");
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState<AnalyseResponse | null>(null);
  const [intake,     setIntake]     = useState<IntakeResponse | null>(null);
  const [confirmed,  setConfirmed]  = useState(false);
  const [error,      setError]      = useState("");

  async function handleTextAnalyse(drugs: string[]) {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await apiAnalyse(drugs);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleImageUpload(b64: string, mode: string) {
    setLoading(true);
    setError("");
    setIntake(null);
    setResult(null);
    setConfirmed(false);
    try {
      const res = await apiIntakeImage(b64, mode);
      setIntake(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Image scan failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleConfirmAndAnalyse() {
    if (!intake?.drugs?.length) return;
    setLoading(true);
    setError("");
    setConfirmed(true);
    try {
      const res = await apiAnalyse(intake.drugs);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  const hasAlerts  = (result?.alerts?.length ?? 0) > 0;
  const maxSev     = result ? Math.max(0, ...result.alerts.map((a) => a.final_severity ?? 0)) : 0;
  const severeBadge = ["🟢", "🟡", "🟠", "🔴"][Math.min(maxSev, 3)];

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Drug interaction check</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Type drug names or upload a prescription photo.
        </p>
      </div>

      {/* Input mode toggle */}
      <div className="flex gap-1 rounded-xl bg-gray-100 p-1 w-fit">
        {(["text", "image"] as InputTab[]).map((t) => (
          <button
            key={t}
            onClick={() => { setTab(t); setResult(null); setIntake(null); setError(""); }}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
              tab === t ? "bg-white shadow-sm text-gray-900" : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {t === "text" ? "✏️ Type drugs" : "📷 Scan image"}
          </button>
        ))}
      </div>

      {/* Input panel */}
      <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
        {tab === "text" ? (
          <DrugInputPanel onSubmit={handleTextAnalyse} loading={loading} />
        ) : (
          <ImageUploader onImage={handleImageUpload} loading={loading} />
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Image intake confirmation */}
      {intake && !confirmed && (
        <div className="bg-white rounded-2xl border border-blue-200 p-5 shadow-sm space-y-3">
          <h2 className="font-semibold text-gray-900">
            {intake.mode === "lab_report" ? "Lab report scanned" : "Drugs found in image"}
          </h2>

          {intake.drugs.length > 0 ? (
            <>
              <div className="flex flex-wrap gap-2">
                {intake.drugs.map((d) => (
                  <span key={d} className="text-xs bg-blue-100 text-blue-800 rounded-full px-3 py-1 font-medium">
                    {d}
                  </span>
                ))}
              </div>
              <p className="text-xs text-gray-400">
                Confidence: {(intake.confidence * 100).toFixed(0)}%
                {intake.prescriber && ` · Prescriber: ${intake.prescriber}`}
              </p>
              <button
                onClick={handleConfirmAndAnalyse}
                disabled={loading}
                className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-semibold text-sm hover:bg-blue-700 disabled:opacity-40 transition"
              >
                {loading ? "Analysing…" : "Analyse these drugs"}
              </button>
            </>
          ) : (
            <p className="text-sm text-gray-500">No drugs were extracted from this image. Try a clearer photo or use text input.</p>
          )}

          {intake.lab_values && Object.keys(intake.lab_values).some((k) => k !== "other" && intake.lab_values![k] != null) && (
            <div className="mt-2 pt-2 border-t border-gray-100">
              <p className="text-xs text-gray-500 font-medium mb-1">Lab values found:</p>
              <div className="flex flex-wrap gap-2 text-xs text-gray-600">
                {Object.entries(intake.lab_values)
                  .filter(([k, v]) => k !== "other" && v != null)
                  .map(([k, v]) => (
                    <span key={k} className="bg-gray-100 rounded px-2 py-0.5">{k}: {String(v)}</span>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-2xl">{severeBadge}</span>
              <div>
                <h2 className="font-semibold text-gray-900">
                  {result.pairs_analysed} pair{result.pairs_analysed !== 1 ? "s" : ""} checked —{" "}
                  {hasAlerts
                    ? `${result.alerts.filter((a) => (a.final_severity ?? 0) >= 2).length} significant interaction${result.alerts.filter((a) => (a.final_severity ?? 0) >= 2).length !== 1 ? "s" : ""} found`
                    : "no significant interactions"}
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  {result.drugs_analysed.join(", ")} ·{" "}
                  {(result.total_latency_ms / 1000).toFixed(1)}s
                </p>
              </div>
            </div>
          </div>

          {/* Alert cards */}
          {result.alerts.length > 0 ? (
            <div className="space-y-3">
              {result.alerts.map((alert: AlertResult, i: number) => (
                <AlertCard key={i} alert={alert} />
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-green-200 bg-green-50 p-4 text-center text-green-800">
              <p className="text-lg font-bold mb-1">🟢 All clear</p>
              <p className="text-sm">No known interactions found between these drugs.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
