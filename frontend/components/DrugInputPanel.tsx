"use client";

import { useState, KeyboardEvent } from "react";

interface Props {
  onSubmit: (drugs: string[]) => void;
  loading?: boolean;
  placeholder?: string;
}

export default function DrugInputPanel({ onSubmit, loading, placeholder }: Props) {
  const [input, setInput]   = useState("");
  const [drugs, setDrugs]   = useState<string[]>([]);
  const [error, setError]   = useState("");

  function addDrug() {
    const name = input.trim();
    if (!name) return;
    if (drugs.includes(name.toLowerCase())) {
      setError(`"${name}" is already in the list`);
      return;
    }
    setDrugs((prev) => [...prev, name]);
    setInput("");
    setError("");
  }

  function removeDrug(idx: number) {
    setDrugs((prev) => prev.filter((_, i) => i !== idx));
  }

  function handleKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addDrug();
    }
  }

  function handleSubmit() {
    if (drugs.length === 0) {
      setError("Add at least one drug to check.");
      return;
    }
    if (drugs.length < 2) {
      setError("Add at least two drugs to check for interactions.");
      return;
    }
    onSubmit(drugs);
  }

  return (
    <div className="space-y-3">
      {/* Input row */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => { setInput(e.target.value); setError(""); }}
          onKeyDown={handleKey}
          placeholder={placeholder || "e.g. warfarin 5mg — press Enter to add"}
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="button"
          onClick={addDrug}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition"
        >
          Add
        </button>
      </div>

      {/* Error */}
      {error && <p className="text-red-500 text-xs">{error}</p>}

      {/* Drug chips */}
      {drugs.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {drugs.map((d, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1.5 rounded-full bg-blue-100 text-blue-800 text-xs px-3 py-1 font-medium"
            >
              {d}
              <button
                onClick={() => removeDrug(i)}
                className="hover:text-red-500 transition text-base leading-none"
                aria-label={`Remove ${d}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Hint */}
      {drugs.length > 0 && drugs.length < 2 && (
        <p className="text-xs text-gray-400">Add one more drug to enable analysis.</p>
      )}

      {/* Submit */}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={loading || drugs.length < 2}
        className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-semibold text-sm
                   hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
      >
        {loading ? "Analysing…" : `Check ${drugs.length} drug${drugs.length !== 1 ? "s" : ""}`}
      </button>
    </div>
  );
}
