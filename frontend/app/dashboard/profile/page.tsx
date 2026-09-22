"use client";

import { useEffect, useState } from "react";
import { apiGetProfile, apiUpdateProfile, PatientProfile, Medication, Condition, Allergy } from "@/lib/api";
import LabValuesDisplay from "@/components/LabValuesDisplay";

type Tab = "overview" | "meds" | "labs" | "conditions";

export default function ProfilePage() {
  const [profile,  setProfile]  = useState<PatientProfile | null>(null);
  const [loading,  setLoading]  = useState(true);
  const [saving,   setSaving]   = useState(false);
  const [error,    setError]    = useState("");
  const [success,  setSuccess]  = useState("");
  const [tab,      setTab]      = useState<Tab>("overview");

  // Editable fields
  const [meds,       setMeds]       = useState<Medication[]>([]);
  const [labs,       setLabs]       = useState<Record<string, string>>({});
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [allergies,  setAllergies]  = useState<Allergy[]>([]);
  const [reportMode, setReportMode] = useState("patient");

  useEffect(() => {
    apiGetProfile()
      .then((p) => {
        setProfile(p);
        setMeds(p.medications || []);
        setConditions(p.conditions || []);
        setAllergies(p.allergies || []);
        setReportMode(p.report_mode || "patient");
        const labKeys = ["egfr","creatinine","ast","alt","inr","hba1c","potassium","hemoglobin"] as const;
        const l: Record<string, string> = {};
        for (const k of labKeys) {
          l[k] = p[k] != null ? String(p[k]) : "";
        }
        setLabs(l);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  function numOrUndef(s: string) {
    const n = parseFloat(s);
    return isNaN(n) ? undefined : n;
  }

  async function handleSave() {
    if (!profile) return;
    setSaving(true);
    setError("");
    setSuccess("");
    try {
      await apiUpdateProfile({
        name:               profile.name,
        date_of_birth:      undefined,
        sex:                profile.sex,
        weight_kg:          profile.weight_kg,
        height_cm:          profile.height_cm,
        preferred_language: profile.preferred_language,
        report_mode:        reportMode,
        is_pregnant:        profile.is_pregnant,
        is_breastfeeding:   profile.is_breastfeeding,
        is_dialysis:        profile.is_dialysis,
        smoker:             profile.smoker,
        alcohol_use:        profile.alcohol_use,
        egfr:        numOrUndef(labs.egfr),
        creatinine:  numOrUndef(labs.creatinine),
        ast:         numOrUndef(labs.ast),
        alt:         numOrUndef(labs.alt),
        inr:         numOrUndef(labs.inr),
        hba1c:       numOrUndef(labs.hba1c),
        potassium:   numOrUndef(labs.potassium),
        hemoglobin:  numOrUndef(labs.hemoglobin),
        conditions:  conditions.filter((c) => c.name.trim()),
        allergies:   allergies.filter((a) => a.drug_name.trim()),
        medications: meds.filter((m) => m.drug_name.trim()),
      });
      setSuccess("Profile saved.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const inputCls = "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500";
  const labelCls = "block text-xs font-medium text-gray-500 mb-1";

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!profile) {
    return <p className="text-red-500">{error || "Could not load profile."}</p>;
  }

  return (
    <div className="max-w-2xl space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">{profile.name}</h1>
        <p className="text-sm text-gray-400 mt-0.5">
          {profile.age && `Age ${profile.age}`}
          {profile.sex && ` · ${profile.sex}`}
          {profile.sessions_count > 0 && ` · ${profile.sessions_count} sessions`}
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {(["overview","meds","labs","conditions"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm font-medium capitalize border-b-2 transition -mb-px ${
              tab === t ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab: overview */}
      {tab === "overview" && (
        <div className="space-y-3">
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-3">
            <div>
              <label className={labelCls}>Report style</label>
              <select className={inputCls} value={reportMode} onChange={(e) => setReportMode(e.target.value)}>
                <option value="patient">Patient (plain language)</option>
                <option value="clinician">Clinician (technical)</option>
              </select>
            </div>
          </div>

          {/* Clinical flags */}
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">Clinical flags</p>
            <div className="flex flex-wrap gap-2">
              {profile.renal_impaired  && <span className="severity-moderate text-xs rounded-full px-3 py-1 border">Renal impairment</span>}
              {profile.hepatic_impaired && <span className="severity-moderate text-xs rounded-full px-3 py-1 border">Hepatic impairment</span>}
              {profile.elderly         && <span className="severity-mild text-xs rounded-full px-3 py-1 border">Elderly</span>}
              {profile.high_bleed_risk  && <span className="severity-severe text-xs rounded-full px-3 py-1 border">High bleed risk</span>}
              {!profile.renal_impaired && !profile.hepatic_impaired && !profile.elderly && !profile.high_bleed_risk && (
                <span className="severity-none text-xs rounded-full px-3 py-1 border">No active flags</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab: medications */}
      {tab === "meds" && (
        <div className="space-y-3">
          {meds.map((m, i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-4 space-y-2">
              <div className="flex gap-2">
                <input className={`${inputCls} flex-1`} placeholder="Drug name" value={m.drug_name} onChange={(e) => { const u=[...meds]; u[i]={...u[i],drug_name:e.target.value}; setMeds(u); }} />
                <button onClick={() => setMeds(meds.filter((_,j)=>j!==i))} className="text-gray-300 hover:text-red-500 text-xl">×</button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input className={inputCls} placeholder="Dose" value={m.dose||""} onChange={(e) => { const u=[...meds]; u[i]={...u[i],dose:e.target.value}; setMeds(u); }} />
                <input className={inputCls} placeholder="Frequency" value={m.frequency||""} onChange={(e) => { const u=[...meds]; u[i]={...u[i],frequency:e.target.value}; setMeds(u); }} />
              </div>
            </div>
          ))}
          <button onClick={() => setMeds([...meds,{drug_name:""}])} className="text-sm text-blue-600 underline">+ Add medication</button>
        </div>
      )}

      {/* Tab: labs */}
      {tab === "labs" && (
        <div className="space-y-4">
          <LabValuesDisplay profile={profile} />
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">Update values</p>
            <div className="grid grid-cols-2 gap-3">
              {(["egfr","creatinine","ast","alt","inr","hba1c","potassium","hemoglobin"] as const).map((k) => (
                <div key={k}>
                  <label className={labelCls}>{k.toUpperCase()}</label>
                  <input type="number" step="0.01" className={inputCls} value={labs[k]} onChange={(e) => setLabs({...labs,[k]:e.target.value})} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab: conditions */}
      {tab === "conditions" && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-2">
            <div className="flex justify-between items-center mb-1">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Conditions</p>
              <button onClick={() => setConditions([...conditions,{name:""}])} className="text-xs text-blue-600 underline">+ Add</button>
            </div>
            {conditions.map((c,i) => (
              <div key={i} className="flex gap-2">
                <input className={`${inputCls} flex-1`} placeholder="Condition name" value={c.name} onChange={(e) => { const u=[...conditions]; u[i]={...u[i],name:e.target.value}; setConditions(u); }} />
                <button onClick={() => setConditions(conditions.filter((_,j)=>j!==i))} className="text-gray-300 hover:text-red-500">×</button>
              </div>
            ))}
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-2">
            <div className="flex justify-between items-center mb-1">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Allergies</p>
              <button onClick={() => setAllergies([...allergies,{drug_name:""}])} className="text-xs text-blue-600 underline">+ Add</button>
            </div>
            {allergies.map((a,i) => (
              <div key={i} className="flex gap-2">
                <input className={`${inputCls} flex-1`} placeholder="Drug name" value={a.drug_name} onChange={(e) => { const u=[...allergies]; u[i]={...u[i],drug_name:e.target.value}; setAllergies(u); }} />
                <input className={`${inputCls} flex-1`} placeholder="Reaction" value={a.reaction_type||""} onChange={(e) => { const u=[...allergies]; u[i]={...u[i],reaction_type:e.target.value}; setAllergies(u); }} />
                <button onClick={() => setAllergies(allergies.filter((_,j)=>j!==i))} className="text-gray-300 hover:text-red-500">×</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error / Success */}
      {error   && <p className="text-red-500 text-sm">{error}</p>}
      {success && <p className="text-green-600 text-sm">{success}</p>}

      {/* Save */}
      <button
        onClick={handleSave}
        disabled={saving}
        className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-semibold text-sm hover:bg-blue-700 disabled:opacity-40 transition"
      >
        {saving ? "Saving…" : "Save changes"}
      </button>
    </div>
  );
}
