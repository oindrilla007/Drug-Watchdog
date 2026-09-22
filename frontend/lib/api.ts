import { getAccessToken } from "./supabase";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function authHeaders(): Promise<HeadersInit> {
  const token = await getAccessToken();
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = await authHeaders();
  const res = await fetch(`${BASE}${path}`, { ...options, headers: { ...headers, ...options.headers } });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  user_id: string;
  email: string;
}

export function apiLogin(email: string, password: string) {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function apiSignup(email: string, password: string, name: string) {
  return request<LoginResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
}

// ── Patient profile ───────────────────────────────────────────────────────────

export interface Medication {
  drug_name: string;
  dose?: string;
  frequency?: string;
  start_date?: string;
  prescriber?: string;
}

export interface Condition {
  name: string;
  icd10_code?: string;
  diagnosed_year?: number;
  severity?: string;
}

export interface Allergy {
  drug_name: string;
  reaction_type?: string;
  severity?: string;
}

export interface PatientProfile {
  patient_id: string;
  name: string;
  age?: number;
  sex?: string;
  weight_kg?: number;
  height_cm?: number;
  preferred_language: string;
  report_mode: string;
  is_pregnant: boolean;
  is_breastfeeding: boolean;
  is_dialysis: boolean;
  smoker: boolean;
  alcohol_use: string;
  egfr?: number;
  creatinine?: number;
  ast?: number;
  alt?: number;
  inr?: number;
  hba1c?: number;
  potassium?: number;
  hemoglobin?: number;
  conditions: Condition[];
  allergies: Allergy[];
  medications: Medication[];
  renal_impaired: boolean;
  hepatic_impaired: boolean;
  elderly: boolean;
  high_bleed_risk: boolean;
  prior_alerts: AlertRecord[];
  sessions_count: number;
  last_session_at?: string;
}

export function apiGetProfile() {
  return request<PatientProfile>("/patients/me");
}

export function apiCreateProfile(profile: Omit<PatientProfile, "patient_id" | "age" | "renal_impaired" | "hepatic_impaired" | "elderly" | "high_bleed_risk" | "prior_alerts" | "sessions_count" | "last_session_at"> & { date_of_birth?: string }) {
  return request<{ message: string; patient_id: string }>("/patients/me", {
    method: "POST",
    body: JSON.stringify(profile),
  });
}

export function apiUpdateProfile(profile: Parameters<typeof apiCreateProfile>[0]) {
  return request<{ message: string; patient_id: string }>("/patients/me", {
    method: "PUT",
    body: JSON.stringify(profile),
  });
}

// ── Analysis ──────────────────────────────────────────────────────────────────

export interface AlertResult {
  drug_a: string;
  drug_b: string;
  severity: number;
  final_severity: number;
  severity_label: string;
  severity_emoji: string;
  confidence: number;
  mechanism: string;
  adjustment_reason: string;
  clinician_report: string;
  patient_report: string;
  urgency: string;
  routed_to: string[];
  citations: string[];
  react_iterations: number;
  error: string;
}

export interface AnalyseResponse {
  patient_id: string;
  patient_name: string;
  input_method: string;
  drugs_analysed: string[];
  pairs_analysed: number;
  alerts: AlertResult[];
  total_latency_ms: number;
  error: string;
}

export function apiAnalyse(drug_list: string[], new_medications?: string[]) {
  return request<AnalyseResponse>("/analyse", {
    method: "POST",
    body: JSON.stringify({ drug_list, new_medications }),
  });
}

// ── Image intake ──────────────────────────────────────────────────────────────

export interface IntakeResponse {
  mode: string;
  success: boolean;
  confidence: number;
  drugs: string[];
  medications: Medication[];
  lab_values?: Record<string, unknown>;
  prescriber?: string;
  prescription_date?: string;
  patient_name_on_rx?: string;
  report_type?: string;
  patient_info?: Record<string, string>;
  error: string;
}

export function apiIntakeImage(image_b64: string, mode = "auto") {
  return request<IntakeResponse>("/intake/image", {
    method: "POST",
    body: JSON.stringify({ image_b64, mode }),
  });
}

// ── Alerts ────────────────────────────────────────────────────────────────────

export interface AlertRecord {
  id: string;
  drug_a: string;
  drug_b: string;
  severity: number;
  severity_label?: string;
  adjusted_severity?: number;
  clinician_report?: string;
  patient_report?: string;
  routed_to?: string[];
  was_acknowledged: boolean;
  created_at?: string;
}

export function apiGetAlerts() {
  return request<{ alerts: AlertRecord[]; count: number }>("/alerts");
}

export function apiGetAllAlerts() {
  return request<{ alerts: AlertRecord[]; count: number }>("/alerts/all");
}

export function apiAcknowledgeAlert(alert_id: string) {
  return request<{ message: string }>(`/alerts/${alert_id}/acknowledge`, {
    method: "PUT",
  });
}
