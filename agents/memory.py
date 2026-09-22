"""
memory.py  ·  Drug Watchdog Phase 4
=====================================
Supabase-backed patient memory layer.

Three responsibilities:
  1. Patient profile CRUD  — demographics, labs, medications, allergies
  2. Session history       — every interaction appended, read at session start
  3. Alert persistence     — every generated alert saved with full context

Supabase is used via the REST API (postgrest) — no supabase-py dependency
required, just the anon key and project URL from environment variables.

Tables created by this module (run create_tables() once on first deploy):
  patients          — full patient profile (one row per patient)
  patient_sessions  — one row per analysis session
  patient_alerts    — one row per drug-pair alert generated
  patient_meds      — current medication list (replaced on update)

Usage
-----
  mem = PatientMemory()

  # Onboarding / update profile
  pid = mem.upsert_patient(profile_dict)

  # Session start: load everything the agents need
  ctx = mem.load_patient_context(patient_id)
  print(ctx.medications)
  print(ctx.prior_alerts[-3:])

  # Session end: save session + alerts
  mem.save_session(patient_id, session_dict)
  mem.save_alert(patient_id, alert_dict)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ──────────────────────────────────────────

load_project_env()

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")           # e.g. https://xyz.supabase.co
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")      # anon/public key
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") # service role key (for writes)

REQUEST_TIMEOUT = 15


def _headers(service: bool = False) -> dict:
    key = SUPABASE_SERVICE_KEY if (service and SUPABASE_SERVICE_KEY) else SUPABASE_ANON_KEY
    return {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def _raise_for_status(resp: requests.Response, action: str) -> None:
    """Raise HTTP errors with the Supabase response body attached."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text.strip()
        if detail:
            raise requests.HTTPError(f"{action} failed: {exc} | Supabase: {detail}", response=resp) from exc
        raise requests.HTTPError(f"{action} failed: {exc}", response=resp) from exc


# ─────────────────────────── Patient context dataclass ───────────────────────

@dataclass
class PatientContext:
    """
    Everything the agents need about a patient — loaded from Supabase
    at the start of every session by the orchestrator.
    """
    # Identity
    patient_id:       str
    name:             str
    age:              int | None       = None
    sex:              str | None       = None
    weight_kg:        float | None     = None
    height_cm:        float | None     = None
    preferred_language: str            = "en"
    report_mode:      str              = "patient"   # "patient" | "clinician"

    # Clinical flags
    is_pregnant:      bool             = False
    is_breastfeeding: bool             = False
    is_dialysis:      bool             = False
    smoker:           bool             = False
    alcohol_use:      str              = "none"      # "none"|"social"|"heavy"

    # Organ function (latest values)
    egfr:             float | None     = None   # mL/min/1.73m²
    creatinine:       float | None     = None   # mg/dL
    ast:              float | None     = None   # U/L
    alt:              float | None     = None   # U/L
    inr:              float | None     = None
    hba1c:            float | None     = None   # %
    potassium:        float | None     = None   # mEq/L
    hemoglobin:       float | None     = None   # g/dL

    # Lists
    conditions:       list[dict]       = field(default_factory=list)
    # [{icd10_code, name, diagnosed_year, severity}]

    allergies:        list[dict]       = field(default_factory=list)
    # [{drug_name, reaction_type, severity, onset_year}]

    medications:      list[dict]       = field(default_factory=list)
    # [{drug_name, dose, frequency, start_date, prescriber, rxcui}]

    # History (populated from patient_sessions + patient_alerts)
    prior_alerts:     list[dict]       = field(default_factory=list)
    # Last N alerts: [{drug_a, drug_b, severity, severity_label,
    #                  was_acknowledged, created_at}]

    sessions_count:   int              = 0
    last_session_at:  str | None       = None

    # Derived clinical risk flags (computed by load_patient_context)
    renal_impaired:   bool             = False   # eGFR < 60
    renal_severe:     bool             = False   # eGFR < 30
    hepatic_impaired: bool             = False   # AST or ALT > 3x ULN
    elderly:          bool             = False   # age >= 75
    high_bleed_risk:  bool             = False   # INR > 3.0 or anticoagulant in meds

    def organ_function_summary(self) -> str:
        """Plain-text summary for injection into agent prompts."""
        parts = []
        if self.egfr is not None:
            flag = " ⚠ SEVERE" if self.egfr < 30 else (" ⚠ MODERATE" if self.egfr < 60 else "")
            parts.append(f"eGFR={self.egfr:.0f} mL/min{flag}")
        if self.ast is not None:
            flag = " ⚠ ELEVATED" if self.ast > 120 else ""
            parts.append(f"AST={self.ast:.0f}{flag}")
        if self.alt is not None:
            flag = " ⚠ ELEVATED" if self.alt > 120 else ""
            parts.append(f"ALT={self.alt:.0f}{flag}")
        if self.inr is not None:
            flag = " ⚠ HIGH" if self.inr > 3.0 else ""
            parts.append(f"INR={self.inr:.1f}{flag}")
        if self.creatinine is not None:
            parts.append(f"Cr={self.creatinine:.1f} mg/dL")
        if self.hba1c is not None:
            parts.append(f"HbA1c={self.hba1c:.1f}%")
        if self.potassium is not None:
            flag = " ⚠ HIGH" if self.potassium > 5.5 else (" ⚠ LOW" if self.potassium < 3.5 else "")
            parts.append(f"K⁺={self.potassium:.1f}{flag}")
        return "  ".join(parts) if parts else "No lab values on file"

    def prior_alert_summary(self) -> str:
        """Summary of prior alerts for agent context — flags ignored warnings."""
        if not self.prior_alerts:
            return "No prior interaction alerts on file."
        lines = []
        for a in self.prior_alerts[-5:]:  # last 5
            ack = "✓ acknowledged" if a.get("was_acknowledged") else "✗ NOT acknowledged"
            lines.append(
                f"  {a.get('drug_a','?')} + {a.get('drug_b','?')} — "
                f"{a.get('severity_label','?')} — {ack} — {a.get('created_at','')[:10]}"
            )
        return "\n".join(lines)

    def drug_name_list(self) -> list[str]:
        """Flat list of drug names from the medication list."""
        return [m["drug_name"] for m in self.medications if m.get("drug_name")]


# ─────────────────────────── PatientMemory ───────────────────────────────────

class PatientMemory:
    """
    Supabase-backed memory for patient profiles, sessions, and alerts.

    Falls back to an in-memory dict store when Supabase is not configured
    (useful for local dev / testing without a Supabase project).
    """

    def __init__(self):
        self._use_supabase = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
        self._local: dict[str, Any] = {}   # fallback in-memory store

        if self._use_supabase:
            log.info("PatientMemory: using Supabase at %s", SUPABASE_URL)
        else:
            log.warning(
                "PatientMemory: SUPABASE_URL / SUPABASE_ANON_KEY not set — "
                "using in-memory fallback (data will not persist)"
            )

    # ── Schema creation ───────────────────────────────────────────────────────

    def create_tables(self):
        """
        Print the SQL to create all required tables in Supabase.
        Run this once in the Supabase SQL editor.
        """
        sql = """
-- Patients (one row per user)
CREATE TABLE IF NOT EXISTS patients (
    patient_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email               TEXT UNIQUE,
    name                TEXT,
    date_of_birth       DATE,
    sex                 TEXT,
    weight_kg           FLOAT,
    height_cm           FLOAT,
    ethnicity           TEXT,
    preferred_language  TEXT DEFAULT 'en',
    report_mode         TEXT DEFAULT 'patient',
    is_pregnant         BOOLEAN DEFAULT FALSE,
    is_breastfeeding    BOOLEAN DEFAULT FALSE,
    is_dialysis         BOOLEAN DEFAULT FALSE,
    smoker              BOOLEAN DEFAULT FALSE,
    alcohol_use         TEXT DEFAULT 'none',
    egfr                FLOAT,
    creatinine          FLOAT,
    ast                 FLOAT,
    alt                 FLOAT,
    inr                 FLOAT,
    hba1c               FLOAT,
    potassium           FLOAT,
    hemoglobin          FLOAT,
    conditions          JSONB DEFAULT '[]',
    allergies           JSONB DEFAULT '[]',
    notification_channel TEXT DEFAULT 'in-app',
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

-- Current medications (replace on every update)
CREATE TABLE IF NOT EXISTS patient_meds (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id  UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    drug_name   TEXT NOT NULL,
    dose        TEXT,
    frequency   TEXT,
    start_date  DATE,
    prescriber  TEXT,
    rxcui       TEXT,
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Session log
CREATE TABLE IF NOT EXISTS patient_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    drugs_checked   JSONB,       -- list of drug names checked this session
    pairs_analysed  INT DEFAULT 0,
    input_method    TEXT,        -- "text" | "prescription_scan" | "pill_photo" | "lab_scan"
    session_data    JSONB,       -- full session payload
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Alert log (one row per drug pair interaction found)
CREATE TABLE IF NOT EXISTS patient_alerts (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id         UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    drug_a             TEXT NOT NULL,
    drug_b             TEXT NOT NULL,
    severity           INT NOT NULL,          -- 0-3
    severity_label     TEXT,
    adjusted_severity  INT,                   -- after patient context adjustment
    adjustment_reason  TEXT,
    cyp_pathway        TEXT,
    clinician_report   TEXT,
    patient_report     TEXT,
    citations          JSONB DEFAULT '[]',
    shap_features      JSONB DEFAULT '[]',
    routed_to          TEXT[],                -- ["physician", "pharmacist", ...]
    was_acknowledged   BOOLEAN DEFAULT FALSE,
    acknowledged_at    TIMESTAMPTZ,
    context_hash       TEXT,
    created_at         TIMESTAMPTZ DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE patients         ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_meds     ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_alerts   ENABLE ROW LEVEL SECURITY;

-- RLS: patients can only see their own rows (link to Supabase Auth)
CREATE POLICY "patients_own" ON patients         FOR ALL USING (auth.uid()::text = patient_id::text);
CREATE POLICY "meds_own"     ON patient_meds     FOR ALL USING (auth.uid()::text = patient_id::text);
CREATE POLICY "sessions_own" ON patient_sessions FOR ALL USING (auth.uid()::text = patient_id::text);
CREATE POLICY "alerts_own"   ON patient_alerts   FOR ALL USING (auth.uid()::text = patient_id::text);
"""
        print(sql)
        return sql

    # ── Patient CRUD ──────────────────────────────────────────────────────────

    def upsert_patient(self, profile: dict) -> str:
        """
        Create or update a patient profile.
        Returns the patient_id (UUID string).

        profile dict keys match the `patients` table columns.
        """
        if not self._use_supabase:
            pid = profile.get("patient_id", f"local-{int(time.time())}")
            self._local.setdefault("patients", {})[pid] = profile
            log.info("(local) Upserted patient %s", pid)
            return pid

        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        resp = requests.post(
            _url("patients"),
            headers={**_headers(service=True), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=profile,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "upsert patient")
        rows = resp.json()
        pid = rows[0]["patient_id"] if rows else profile.get("patient_id", "")
        log.info("Upserted patient %s", pid)
        return pid

    def save_medications(self, patient_id: str, medications: list[dict]):
        """
        Replace the patient's current medication list.
        Deactivates old records, inserts new ones.
        """
        if not self._use_supabase:
            self._local.setdefault("meds", {})[patient_id] = medications
            return

        # Deactivate existing active meds
        requests.patch(
            _url("patient_meds") + f"?patient_id=eq.{patient_id}&active=eq.true",
            headers=_headers(service=True),
            json={"active": False},
            timeout=REQUEST_TIMEOUT,
        )
        # Insert new meds
        if medications:
            rows = [{"patient_id": patient_id, **m} for m in medications]
            resp = requests.post(
                _url("patient_meds"),
                headers=_headers(service=True),
                json=rows,
                timeout=REQUEST_TIMEOUT,
            )
            _raise_for_status(resp, "insert patient medications")
        log.info("Saved %d medications for patient %s", len(medications), patient_id)

    # ── Load patient context ──────────────────────────────────────────────────

    def load_patient_context(self, patient_id: str) -> PatientContext:
        """
        Load the full PatientContext for a patient.
        Called by the orchestrator at the start of every session.
        """
        if not self._use_supabase:
            return self._load_local_context(patient_id)

        # 1. Patient profile
        resp = requests.get(
            _url("patients") + f"?patient_id=eq.{patient_id}&limit=1",
            headers=_headers(service=True),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "load patient")
        rows = resp.json()
        if not rows:
            raise ValueError(f"Patient {patient_id} not found")
        p = rows[0]

        # 2. Active medications
        resp2 = requests.get(
            _url("patient_meds") + f"?patient_id=eq.{patient_id}&active=eq.true",
            headers=_headers(service=True),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp2, "load medications")
        meds = resp2.json()

        # 3. Prior alerts (last 10)
        resp3 = requests.get(
            _url("patient_alerts") + (
                f"?patient_id=eq.{patient_id}"
                "&order=created_at.desc&limit=10"
                "&select=drug_a,drug_b,severity,severity_label,was_acknowledged,created_at"
            ),
            headers=_headers(service=True),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp3, "load prior alerts")
        prior_alerts = resp3.json()

        # 4. Sessions count
        resp4 = requests.get(
            _url("patient_sessions") + f"?patient_id=eq.{patient_id}&select=created_at&order=created_at.desc&limit=1",
            headers=_headers(service=True),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp4, "load sessions")
        sessions_rows = resp4.json()

        return self._build_context(p, meds, prior_alerts, sessions_rows)

    def _build_context(
        self,
        p:             dict,
        meds:          list[dict],
        prior_alerts:  list[dict],
        sessions_rows: list[dict],
    ) -> PatientContext:
        # Compute age from DOB
        age = None
        if p.get("date_of_birth"):
            try:
                dob = datetime.fromisoformat(p["date_of_birth"])
                age = (datetime.now() - dob).days // 365
            except Exception:
                pass

        ctx = PatientContext(
            patient_id        = str(p.get("patient_id", "")),
            name              = p.get("name", "Unknown"),
            age               = age,
            sex               = p.get("sex"),
            weight_kg         = p.get("weight_kg"),
            height_cm         = p.get("height_cm"),
            preferred_language= p.get("preferred_language", "en"),
            report_mode       = p.get("report_mode", "patient"),
            is_pregnant       = bool(p.get("is_pregnant", False)),
            is_breastfeeding  = bool(p.get("is_breastfeeding", False)),
            is_dialysis       = bool(p.get("is_dialysis", False)),
            smoker            = bool(p.get("smoker", False)),
            alcohol_use       = p.get("alcohol_use", "none"),
            egfr              = p.get("egfr"),
            creatinine        = p.get("creatinine"),
            ast               = p.get("ast"),
            alt               = p.get("alt"),
            inr               = p.get("inr"),
            hba1c             = p.get("hba1c"),
            potassium         = p.get("potassium"),
            hemoglobin        = p.get("hemoglobin"),
            conditions        = p.get("conditions") or [],
            allergies         = p.get("allergies") or [],
            medications       = meds,
            prior_alerts      = prior_alerts,
            sessions_count    = len(sessions_rows),
            last_session_at   = sessions_rows[0].get("created_at") if sessions_rows else None,
        )

        # Derived clinical flags
        ctx.renal_impaired   = ctx.egfr is not None and ctx.egfr < 60
        ctx.renal_severe     = ctx.egfr is not None and ctx.egfr < 30
        ctx.hepatic_impaired = (
            (ctx.ast is not None and ctx.ast > 120) or
            (ctx.alt is not None and ctx.alt > 120)
        )
        ctx.elderly        = age is not None and age >= 75
        ctx.high_bleed_risk = (ctx.inr is not None and ctx.inr > 3.0)

        return ctx

    def _load_local_context(self, patient_id: str) -> PatientContext:
        """In-memory fallback for local testing."""
        p    = self._local.get("patients", {}).get(patient_id, {})
        meds = self._local.get("meds", {}).get(patient_id, [])
        alts = self._local.get("alerts", {}).get(patient_id, [])
        return self._build_context(p, meds, alts, [])

    # ── Session + alert persistence ───────────────────────────────────────────

    def save_session(self, patient_id: str, session_data: dict):
        """Append a session record to patient_sessions."""
        if not self._use_supabase:
            self._local.setdefault("sessions", {}).setdefault(patient_id, []).append(session_data)
            return

        row = {
            "patient_id":     patient_id,
            "drugs_checked":  session_data.get("drugs_checked", []),
            "pairs_analysed": session_data.get("pairs_analysed", 0),
            "input_method":   session_data.get("input_method", "text"),
            "session_data":   session_data,
        }
        resp = requests.post(
            _url("patient_sessions"),
            headers=_headers(service=True),
            json=row,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "save session")
        log.info("Saved session for patient %s", patient_id)

    def save_alert(self, patient_id: str, alert: dict):
        """Persist a drug interaction alert to patient_alerts."""
        if not self._use_supabase:
            self._local.setdefault("alerts", {}).setdefault(patient_id, []).append(alert)
            return

        row = {
            "patient_id":        patient_id,
            "drug_a":            alert.get("drug_a", ""),
            "drug_b":            alert.get("drug_b", ""),
            "severity":          alert.get("severity", 0),
            "severity_label":    alert.get("severity_label", ""),
            "adjusted_severity": alert.get("adjusted_severity"),
            "adjustment_reason": alert.get("adjustment_reason"),
            "cyp_pathway":       alert.get("cyp_pathway"),
            "clinician_report":  alert.get("clinician_report"),
            "patient_report":    alert.get("patient_report"),
            "citations":         alert.get("citations", []),
            "shap_features":     alert.get("shap_features", []),
            "routed_to":         alert.get("routed_to", []),
            "context_hash":      alert.get("context_hash"),
        }
        resp = requests.post(
            _url("patient_alerts"),
            headers=_headers(service=True),
            json=row,
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "save alert")
        log.info("Saved alert: %s + %s (sev=%d)", row["drug_a"], row["drug_b"], row["severity"])

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged (patient/clinician confirmed they saw it)."""
        if not self._use_supabase:
            return
        resp = requests.patch(
            _url("patient_alerts") + f"?id=eq.{alert_id}",
            headers=_headers(service=True),
            json={"was_acknowledged": True, "acknowledged_at": datetime.now(timezone.utc).isoformat()},
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "acknowledge alert")

    def get_unacknowledged_alerts(self, patient_id: str) -> list[dict]:
        """Return all unacknowledged alerts for a patient (for escalation logic)."""
        if not self._use_supabase:
            return [a for a in self._local.get("alerts", {}).get(patient_id, [])
                    if not a.get("was_acknowledged")]

        resp = requests.get(
            _url("patient_alerts") + (
                f"?patient_id=eq.{patient_id}"
                "&was_acknowledged=eq.false"
                "&order=created_at.desc"
            ),
            headers=_headers(service=True),
            timeout=REQUEST_TIMEOUT,
        )
        _raise_for_status(resp, "load unacknowledged alerts")
        return resp.json()


# ─────────────────────────── Singleton ───────────────────────────────────────

_memory: PatientMemory | None = None

def get_memory() -> PatientMemory:
    global _memory
    if _memory is None:
        _memory = PatientMemory()
    return _memory


# ─────────────────────────── CLI / setup helper ───────────────────────────────

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    mem = PatientMemory()

    print("── Supabase SQL (run in Supabase SQL editor to create tables) ──")
    mem.create_tables()

    print("\n── Local smoke test ──")
    pid = mem.upsert_patient({
        "patient_id":   "00000000-0000-4000-8000-000000000001",
        "name":         "Jane Doe",
        "date_of_birth":"1950-03-15",
        "sex":          "F",
        "weight_kg":    62.0,
        "egfr":         28.0,
        "ast":          145.0,
        "inr":          2.8,
        "conditions":   [{"icd10_code": "I48", "name": "Atrial fibrillation", "severity": "moderate"}],
        "allergies":    [{"drug_name": "penicillin", "reaction_type": "anaphylaxis", "severity": "severe"}],
    })
    mem.save_medications(pid, [
        {"drug_name": "warfarin",  "dose": "5mg", "frequency": "daily"},
        {"drug_name": "aspirin",   "dose": "81mg", "frequency": "daily"},
        {"drug_name": "lisinopril","dose": "10mg", "frequency": "daily"},
    ])

    ctx = mem.load_patient_context(pid)
    print(f"Patient   : {ctx.name}  age={ctx.age}  sex={ctx.sex}")
    print(f"Labs      : {ctx.organ_function_summary()}")
    print(f"Meds      : {ctx.drug_name_list()}")
    print(f"Renal     : impaired={ctx.renal_impaired}  severe={ctx.renal_severe}")
    print(f"Hepatic   : {ctx.hepatic_impaired}")
    print(f"Elderly   : {ctx.elderly}")
    print(f"Bleed risk: {ctx.high_bleed_risk}")
    print(f"Alerts    : {ctx.prior_alert_summary()}")
