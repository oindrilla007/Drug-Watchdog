import { PatientProfile } from "@/lib/api";

type LabKey = "egfr" | "creatinine" | "ast" | "alt" | "inr" | "hba1c" | "potassium" | "hemoglobin";

interface LabMeta {
  label: string;
  unit: string;
  low?: number;
  high?: number;
  critHigh?: number;
  critLow?: number;
}

const LAB_META: Record<LabKey, LabMeta> = {
  egfr:       { label: "eGFR",        unit: "mL/min/1.73m²", critLow: 15, low: 60 },
  creatinine: { label: "Creatinine",  unit: "mg/dL",          high: 1.2,  critHigh: 3.0 },
  ast:        { label: "AST",         unit: "U/L",            high: 40,   critHigh: 120 },
  alt:        { label: "ALT",         unit: "U/L",            high: 40,   critHigh: 120 },
  inr:        { label: "INR",         unit: "",               high: 1.1,  critHigh: 3.0 },
  hba1c:      { label: "HbA1c",       unit: "%",              high: 5.7,  critHigh: 8.0 },
  potassium:  { label: "Potassium",   unit: "mEq/L",          critLow: 3.5, critHigh: 5.5 },
  hemoglobin: { label: "Hemoglobin",  unit: "g/dL",           critLow: 8.0 },
};

function flagColor(key: LabKey, value: number): string {
  const m = LAB_META[key];
  if (m.critLow !== undefined && value < m.critLow) return "text-red-600 font-bold";
  if (m.critHigh !== undefined && value > m.critHigh) return "text-red-600 font-bold";
  if (m.low !== undefined && value < m.low)   return "text-orange-500 font-semibold";
  if (m.high !== undefined && value > m.high) return "text-orange-500 font-semibold";
  return "text-green-700";
}

function flagLabel(key: LabKey, value: number): string {
  const m = LAB_META[key];
  if (m.critLow !== undefined && value < m.critLow) return "CRIT LOW";
  if (m.critHigh !== undefined && value > m.critHigh) return "CRIT HIGH";
  if (m.low !== undefined && value < m.low)  return "LOW";
  if (m.high !== undefined && value > m.high) return "HIGH";
  return "normal";
}

interface Props {
  profile: Pick<PatientProfile, LabKey>;
}

export default function LabValuesDisplay({ profile }: Props) {
  const keys = Object.keys(LAB_META) as LabKey[];
  const present = keys.filter((k) => profile[k] != null);

  if (present.length === 0) {
    return <p className="text-sm text-gray-400 italic">No lab values on file.</p>;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {present.map((key) => {
        const val = profile[key] as number;
        const m   = LAB_META[key];
        const col = flagColor(key, val);
        const lbl = flagLabel(key, val);
        return (
          <div key={key} className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
            <p className="text-xs text-gray-400 uppercase tracking-wide font-medium">{m.label}</p>
            <p className={`text-lg font-bold mt-0.5 ${col}`}>
              {val.toFixed(key === "inr" ? 1 : 0)}
              {m.unit && <span className="text-xs font-normal ml-1">{m.unit}</span>}
            </p>
            <span className={`text-xs mt-0.5 inline-block px-1.5 py-0.5 rounded-full ${
              lbl === "normal"
                ? "bg-green-100 text-green-700"
                : lbl.startsWith("CRIT")
                ? "bg-red-100 text-red-700"
                : "bg-orange-100 text-orange-700"
            }`}>
              {lbl}
            </span>
          </div>
        );
      })}
    </div>
  );
}
