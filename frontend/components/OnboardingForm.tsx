"use client";

import { useState } from "react";
import { apiCreateProfile, Medication, Condition, Allergy } from "@/lib/api";

type Step = 1 | 2 | 3 | 4;

const STEPS: Record<Step, string> = {
  1: "Demographics",
  2: "Conditions & Allergies",
  3: "Medications",
  4: "Lab Values",
};

interface FormState {
  name: string;
  date_of_birth: string;
  sex: string;
  weight_kg: string;
  height_cm: string;
  is_pregnant: boolean;
  is_breastfeeding: boolean;
  is_dialysis: boolean;
  smoker: boolean;
  alcohol_use: string;
  preferred_language: string;
  report_mode: string;
  conditions: Condition[];
  allergies: Allergy[];
  medications: Medication[];
  egfr: string;
  creatinine: string;
  ast: string;
  alt: string;
  inr: string;
  hba1c: string;
  potassium: string;
  hemoglobin: string;
}

const empty: FormState = {
  name: "", date_of_birth: "", sex: "", weight_kg: "", height_cm: "",
  is_pregnant: false, is_breastfeeding: false, is_dialysis: false,
  smoker: false, alcohol_use: "none", preferred_language: "en", report_mode: "patient",
  conditions: [], allergies: [], medications: [],
  egfr: "", creatinine: "", ast: "", alt: "", inr: "", hba1c: "", potassium: "", hemoglobin: "",
};

interface Props {
  onComplete: () => void;
}

export default function OnboardingForm({ onComplete }: Props) {
  const [step, setStep]   = useState<Step>(1);
  const [form, setForm]   = useState<FormState>(empty);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Generic field setter
  function set<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  // Condition helpers
  function addCondition() {
    set("conditions", [...form.conditions, { name: "" }]);
  }
  function updateCondition(i: number, val: string) {
    const updated = [...form.conditions];
    updated[i] = { ...updated[i], name: val };
    set("conditions", updated);
  }
  function removeCondition(i: number) {
    set("conditions", form.conditions.filter((_, j) => j !== i));
  }

  // Allergy helpers
  function addAllergy() {
    set("allergies", [...form.allergies, { drug_name: "" }]);
  }
  function updateAllergy(i: number, field: keyof Allergy, val: string) {
    const updated = [...form.allergies];
    updated[i] = { ...updated[i], [field]: val };
    set("allergies", updated);
  }
  function removeAllergy(i: number) {
    set("allergies", form.allergies.filter((_, j) => j !== i));
  }

  // Medication helpers
  function addMed() {
    set("medications", [...form.medications, { drug_name: "" }]);
  }
  function updateMed(i: number, field: keyof Medication, val: string) {
    const updated = [...form.medications];
    updated[i] = { ...updated[i], [field]: val };
    set("medications", updated);
  }
  function removeMed(i: number) {
    set("medications", form.medications.filter((_, j) => j !== i));
  }

  function numOrNull(s: string) {
    const n = parseFloat(s);
    return isNaN(n) ? undefined : n;
  }

  async function handleSubmit() {
    if (!form.name.trim()) { setError("Name is required."); return; }
    setLoading(true);
    setError("");
    try {
      await apiCreateProfile({
        name:               form.name,
        date_of_birth:      form.date_of_birth || undefined,
        sex:                form.sex || undefined,
        weight_kg:          numOrNull(form.weight_kg),
        height_cm:          numOrNull(form.height_cm),
        is_pregnant:        form.is_pregnant,
        is_breastfeeding:   form.is_breastfeeding,
        is_dialysis:        form.is_dialysis,
        smoker:             form.smoker,
        alcohol_use:        form.alcohol_use,
        preferred_language: form.preferred_language,
        report_mode:        form.report_mode,
        conditions:         form.conditions.filter((c) => c.name.trim()),
        allergies:          form.allergies.filter((a) => a.drug_name.trim()),
        medications:        form.medications.filter((m) => m.drug_name.trim()),
        egfr:               numOrNull(form.egfr),
        creatinine:         numOrNull(form.creatinine),
        ast:                numOrNull(form.ast),
        alt:                numOrNull(form.alt),
        inr:                numOrNull(form.inr),
        hba1c:              numOrNull(form.hba1c),
        potassium:          numOrNull(form.potassium),
        hemoglobin:         numOrNull(form.hemoglobin),
      });
      onComplete();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save profile.");
    } finally {
      setLoading(false);
    }
  }

  const inputCls = "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500";
  const labelCls = "block text-xs font-medium text-gray-600 mb-1";

  return (
    <div className="max-w-lg mx-auto">
      {/* Step indicators */}
      <div className="flex gap-1 mb-6">
        {([1, 2, 3, 4] as Step[]).map((s) => (
          <div
            key={s}
            className={`flex-1 h-1.5 rounded-full transition ${
              s <= step ? "bg-blue-600" : "bg-gray-200"
            }`}
          />
        ))}
      </div>
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        Step {step} of 4 — {STEPS[step]}
      </h2>

      {/* Step 1: Demographics */}
      {step === 1 && (
        <div className="space-y-4">
          <div>
            <label className={labelCls}>Full name *</label>
            <input className={inputCls} value={form.name} onChange={(e) => set("name", e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>Date of birth</label>
              <input type="date" className={inputCls} value={form.date_of_birth} onChange={(e) => set("date_of_birth", e.target.value)} />
            </div>
            <div>
              <label className={labelCls}>Sex</label>
              <select className={inputCls} value={form.sex} onChange={(e) => set("sex", e.target.value)}>
                <option value="">—</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
                <option value="O">Other</option>
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>Weight (kg)</label>
              <input type="number" className={inputCls} value={form.weight_kg} onChange={(e) => set("weight_kg", e.target.value)} />
            </div>
            <div>
              <label className={labelCls}>Height (cm)</label>
              <input type="number" className={inputCls} value={form.height_cm} onChange={(e) => set("height_cm", e.target.value)} />
            </div>
          </div>
          <div>
            <label className={labelCls}>Alcohol use</label>
            <select className={inputCls} value={form.alcohol_use} onChange={(e) => set("alcohol_use", e.target.value)}>
              <option value="none">None</option>
              <option value="social">Social</option>
              <option value="heavy">Heavy</option>
            </select>
          </div>
          <div>
            <label className={labelCls}>Preferred report language</label>
            <select className={inputCls} value={form.report_mode} onChange={(e) => set("report_mode", e.target.value)}>
              <option value="patient">Patient (plain language)</option>
              <option value="clinician">Clinician (technical)</option>
            </select>
          </div>
          <div className="flex flex-wrap gap-4 text-sm">
            {(["smoker", "is_pregnant", "is_breastfeeding", "is_dialysis"] as const).map((k) => (
              <label key={k} className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={form[k] as boolean} onChange={(e) => set(k, e.target.checked)} className="w-4 h-4 rounded" />
                {k.replace(/_/g, " ").replace(/^is /, "").replace(/\b\w/g, (c) => c.toUpperCase())}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Step 2: Conditions & Allergies */}
      {step === 2 && (
        <div className="space-y-5">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Medical conditions</span>
              <button onClick={addCondition} className="text-xs text-blue-600 underline">+ Add</button>
            </div>
            {form.conditions.map((c, i) => (
              <div key={i} className="flex gap-2 mb-2">
                <input
                  className={`${inputCls} flex-1`}
                  placeholder="e.g. Atrial fibrillation"
                  value={c.name}
                  onChange={(e) => updateCondition(i, e.target.value)}
                />
                <button onClick={() => removeCondition(i)} className="text-gray-400 hover:text-red-500">×</button>
              </div>
            ))}
            {form.conditions.length === 0 && <p className="text-xs text-gray-400">No conditions added.</p>}
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Drug allergies</span>
              <button onClick={addAllergy} className="text-xs text-blue-600 underline">+ Add</button>
            </div>
            {form.allergies.map((a, i) => (
              <div key={i} className="flex gap-2 mb-2 flex-wrap">
                <input
                  className={`${inputCls} flex-1 min-w-32`}
                  placeholder="Drug name"
                  value={a.drug_name}
                  onChange={(e) => updateAllergy(i, "drug_name", e.target.value)}
                />
                <input
                  className={`${inputCls} flex-1 min-w-32`}
                  placeholder="Reaction (e.g. rash)"
                  value={a.reaction_type || ""}
                  onChange={(e) => updateAllergy(i, "reaction_type", e.target.value)}
                />
                <button onClick={() => removeAllergy(i)} className="text-gray-400 hover:text-red-500">×</button>
              </div>
            ))}
            {form.allergies.length === 0 && <p className="text-xs text-gray-400">No allergies added.</p>}
          </div>
        </div>
      )}

      {/* Step 3: Medications */}
      {step === 3 && (
        <div>
          <div className="flex justify-between items-center mb-3">
            <span className="text-sm font-medium text-gray-700">Current medications</span>
            <button onClick={addMed} className="text-xs text-blue-600 underline">+ Add medication</button>
          </div>
          {form.medications.map((m, i) => (
            <div key={i} className="rounded-lg border border-gray-200 p-3 mb-3 space-y-2">
              <div className="flex gap-2">
                <input
                  className={`${inputCls} flex-1`}
                  placeholder="Drug name *"
                  value={m.drug_name}
                  onChange={(e) => updateMed(i, "drug_name", e.target.value)}
                />
                <button onClick={() => removeMed(i)} className="text-gray-400 hover:text-red-500 text-lg leading-none">×</button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input className={inputCls} placeholder="Dose (e.g. 5mg)" value={m.dose || ""} onChange={(e) => updateMed(i, "dose", e.target.value)} />
                <input className={inputCls} placeholder="Frequency (e.g. once daily)" value={m.frequency || ""} onChange={(e) => updateMed(i, "frequency", e.target.value)} />
              </div>
            </div>
          ))}
          {form.medications.length === 0 && (
            <p className="text-xs text-gray-400">No medications added. You can add them here or by scanning a prescription later.</p>
          )}
        </div>
      )}

      {/* Step 4: Lab Values */}
      {step === 4 && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500 mb-2">Optional — used to personalise severity adjustments. Leave blank if unknown.</p>
          <div className="grid grid-cols-2 gap-3">
            {(["egfr", "creatinine", "ast", "alt", "inr", "hba1c", "potassium", "hemoglobin"] as const).map((k) => (
              <div key={k}>
                <label className={labelCls}>{k.toUpperCase()}</label>
                <input type="number" step="0.01" className={inputCls} value={form[k]} onChange={(e) => set(k, e.target.value)} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && <p className="mt-3 text-red-500 text-sm">{error}</p>}

      {/* Navigation */}
      <div className="mt-6 flex gap-3">
        {step > 1 && (
          <button
            onClick={() => setStep((s) => (s - 1) as Step)}
            className="flex-1 py-2.5 rounded-lg border border-gray-300 text-sm font-medium hover:bg-gray-50 transition"
          >
            Back
          </button>
        )}
        {step < 4 ? (
          <button
            onClick={() => {
              if (step === 1 && !form.name.trim()) { setError("Name is required."); return; }
              setError("");
              setStep((s) => (s + 1) as Step);
            }}
            className="flex-1 py-2.5 rounded-lg bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 transition"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="flex-1 py-2.5 rounded-lg bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 disabled:opacity-40 transition"
          >
            {loading ? "Saving…" : "Complete setup"}
          </button>
        )}
      </div>
    </div>
  );
}
