"""
alert_routing_agent.py  ·  Drug Watchdog Phase 4
==================================================
LangGraph node: classifies the final alert and routes it to the
correct destination(s) based on severity, patient context, and
prior acknowledgement history.

Routing table
-------------
  Severity 0 → LOG ONLY           (audit trail, no notification)
  Severity 1 → PATIENT            (app/SMS notification to patient)
  Severity 2 → PHARMACIST         (pharmacist review queue)
  Severity 3 → PHYSICIAN + EHR    (urgent physician alert + EHR flag)

Escalation overrides (any severity can be bumped up a routing tier):
  • Prior ignored alert for same pair          → escalate one tier
  • Patient on dialysis + renally-cleared drug → escalate to PHYSICIAN
  • Pregnancy + contraindicated drug           → escalate to PHYSICIAN CRITICAL
  • INR > 3.5 + anticoagulant interaction      → escalate to PHYSICIAN CRITICAL

Delivery channels
-----------------
  • Supabase   — alert row written via memory.save_alert (always)
  • Email      — SMTP via ALERT_SMTP_* env vars
  • Webhook    — POST to ALERT_WEBHOOK_URL (EHR integration / Slack)
  • Console    — always printed (good for CLI demo / CI)

  If env vars are absent, notifications are logged to console only.
  The pipeline never crashes due to missing notification config.

LangGraph state keys
--------------------
  Reads  : state["dual_report"]      → DualReport
            state["prediction"]      → PredictionResult
            state["patient_context"] → PatientContext
            state["citations"]       → list[Citation]
  Writes : state["routing_decision"] → RoutingDecision
            state["alert_record"]    → dict  (for memory.save_alert)

Usage (standalone)
------------------
  agent = AlertRoutingAgent()
  decision = agent.route(dual_report, prediction, patient_context, citations)
  print(decision.destinations)
  print(decision.urgency_label())
"""

import json
import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import requests as http_requests

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

log = logging.getLogger(__name__)

# ─────────────────────────── Config from env ─────────────────────────────────

load_project_env()

SMTP_HOST         = os.getenv("ALERT_SMTP_HOST",     "")
SMTP_PORT         = int(os.getenv("ALERT_SMTP_PORT", "587"))
SMTP_USER         = os.getenv("ALERT_SMTP_USER",     "")
SMTP_PASSWORD     = os.getenv("ALERT_SMTP_PASSWORD", "")
SMTP_FROM         = os.getenv("ALERT_SMTP_FROM",     "noreply@drugwatchdog.app")
WEBHOOK_URL       = os.getenv("ALERT_WEBHOOK_URL",   "")
PHARMACIST_EMAILS = [e.strip() for e in os.getenv("PHARMACIST_EMAILS", "").split(",") if e.strip()]
PHYSICIAN_EMAILS  = [e.strip() for e in os.getenv("PHYSICIAN_EMAILS",  "").split(",") if e.strip()]
REQUEST_TIMEOUT   = 10


# ─────────────────────────── Enums / data classes ────────────────────────────

class Destination(str, Enum):
    LOG        = "log"
    PATIENT    = "patient"
    PHARMACIST = "pharmacist"
    PHYSICIAN  = "physician"
    EHR        = "ehr"


class Urgency(str, Enum):
    ROUTINE  = "routine"
    PRIORITY = "priority"
    URGENT   = "urgent"
    CRITICAL = "critical"


URGENCY_EMOJI = {
    Urgency.ROUTINE:  "🟢",
    Urgency.PRIORITY: "🟡",
    Urgency.URGENT:   "🔴",
    Urgency.CRITICAL: "🚨",
}

SEVERITY_BASE_ROUTING: dict[int, list[Destination]] = {
    0: [Destination.LOG],
    1: [Destination.LOG, Destination.PATIENT],
    2: [Destination.LOG, Destination.PATIENT, Destination.PHARMACIST],
    3: [Destination.LOG, Destination.PATIENT, Destination.PHARMACIST,
        Destination.PHYSICIAN, Destination.EHR],
}


@dataclass
class RoutingDecision:
    drug_a:          str
    drug_b:          str
    severity:        int
    urgency:         Urgency
    destinations:    list[Destination]
    override_reason: str       = ""
    latency_ms:      float     = 0.0
    delivery_log:    list[dict]= field(default_factory=list)

    def urgency_label(self) -> str:
        return f"{URGENCY_EMOJI[self.urgency]} {self.urgency.value.upper()}"

    def delivered_to(self) -> list[str]:
        return [d.value for d in self.destinations]

    def any_failed(self) -> bool:
        return any(not d.get("success", True) for d in self.delivery_log)

    def to_dict(self) -> dict:
        return {
            "drug_a":          self.drug_a,
            "drug_b":          self.drug_b,
            "severity":        self.severity,
            "urgency":         self.urgency.value,
            "destinations":    self.delivered_to(),
            "override_reason": self.override_reason,
            "latency_ms":      self.latency_ms,
            "delivery_log":    self.delivery_log,
        }


# ─────────────────────────── Agent ───────────────────────────────────────────

class AlertRoutingAgent:
    """
    LangGraph node: determines alert destinations and dispatches notifications.

    Routing is purely rule-based — fast, deterministic, fully auditable.
    Notification delivery is best-effort: failures logged, never raised.
    """

    def __init__(self):
        channels = ["console"]
        if SMTP_HOST and SMTP_USER:
            channels.append("email")
        if WEBHOOK_URL:
            channels.append("webhook")
        log.info("AlertRoutingAgent ready — channels: %s", ", ".join(channels))

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        dual_report:     Any,
        prediction:      Any,
        patient_context: Any,
        citations:       list,
    ) -> RoutingDecision:
        """
        Determine routing destinations and dispatch all notifications.
        Returns RoutingDecision with per-channel delivery results.
        """
        t0 = time.perf_counter()

        drug_a    = getattr(prediction, "drug_a", "drug_a")
        drug_b    = getattr(prediction, "drug_b", "drug_b")
        final_sev = (
            getattr(prediction, "adjusted_severity", None)
            or getattr(prediction, "severity", 0)
        )

        # Base routing from severity table
        destinations = list(SEVERITY_BASE_ROUTING.get(final_sev, [Destination.LOG]))

        # Urgency classification + override check
        urgency, override_reason = self._classify_urgency(
            final_sev, prediction, patient_context
        )

        # Critical escalation: ensure full routing regardless of base severity
        if urgency == Urgency.CRITICAL:
            for dest in [Destination.PHYSICIAN, Destination.EHR,
                         Destination.PHARMACIST, Destination.PATIENT]:
                if dest not in destinations:
                    destinations.append(dest)

        decision = RoutingDecision(
            drug_a          = drug_a,
            drug_b          = drug_b,
            severity        = final_sev,
            urgency         = urgency,
            destinations    = destinations,
            override_reason = override_reason,
        )

        self._dispatch(decision, dual_report, prediction, patient_context, citations)
        decision.latency_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "Alert routed: %s + %s → %s (sev=%d) | to: %s (%.0f ms)",
            drug_a, drug_b, decision.urgency_label(), final_sev,
            ", ".join(decision.delivered_to()), decision.latency_ms,
        )
        return decision

    # ── LangGraph node entrypoint ─────────────────────────────────────────────

    def run(self, state: dict) -> dict:
        """
        LangGraph node function.

        Reads  : state["dual_report"], state["prediction"],
                 state["patient_context"], state["citations"]
        Writes : state["routing_decision"], state["alert_record"]
        """
        dual_report = state.get("dual_report")
        prediction  = state.get("prediction")
        patient_ctx = state.get("patient_context")
        citations   = state.get("citations", [])

        if not dual_report or not prediction:
            log.error("AlertRoutingAgent.run: dual_report or prediction missing in state")
            return state

        decision     = self.route(dual_report, prediction, patient_ctx, citations)
        alert_record = self._build_alert_record(decision, dual_report, prediction, citations)

        state["routing_decision"] = decision
        state["alert_record"]     = alert_record
        return state

    # ── Urgency classifier ────────────────────────────────────────────────────

    @staticmethod
    def _classify_urgency(
        severity:        int,
        prediction:      Any,
        patient_context: Any,
    ) -> tuple[Urgency, str]:
        """
        Classify urgency and apply escalation overrides.
        Returns (Urgency, comma-joined override reasons).
        """
        base_map = {0: Urgency.ROUTINE, 1: Urgency.ROUTINE,
                    2: Urgency.PRIORITY, 3: Urgency.URGENT}
        urgency  = base_map.get(severity, Urgency.ROUTINE)

        if patient_context is None:
            return urgency, ""

        overrides: list[str] = []
        drugs = {
            getattr(prediction, "drug_a", "").lower(),
            getattr(prediction, "drug_b", "").lower(),
        }

        # Override 1: pregnancy + contraindicated drug
        if getattr(patient_context, "is_pregnant", False):
            try:
                try:
                    from .patient_context_agent import PREGNANCY_CONTRAINDICATED
                except ImportError:
                    from patient_context_agent import PREGNANCY_CONTRAINDICATED
                if drugs & PREGNANCY_CONTRAINDICATED:
                    urgency = Urgency.CRITICAL
                    overrides.append("Pregnancy + contraindicated drug")
            except ImportError:
                pass

        # Override 2: dialysis + renally-cleared drug
        if getattr(patient_context, "is_dialysis", False):
            try:
                try:
                    from .patient_context_agent import RENAL_CLEARED
                except ImportError:
                    from patient_context_agent import RENAL_CLEARED
                if drugs & RENAL_CLEARED:
                    urgency = Urgency.CRITICAL
                    overrides.append("Dialysis + renally-cleared drug")
            except ImportError:
                pass

        # Override 3: INR > 3.5 + anticoagulant
        inr = getattr(patient_context, "inr", None)
        if inr and inr > 3.5:
            try:
                try:
                    from .patient_context_agent import ANTICOAGULANTS
                except ImportError:
                    from patient_context_agent import ANTICOAGULANTS
                if drugs & ANTICOAGULANTS:
                    urgency = Urgency.CRITICAL
                    overrides.append(f"INR={inr:.1f} + anticoagulant interaction")
            except ImportError:
                pass

        # Override 4: prior ignored alert for same pair → bump one tier
        drug_a = getattr(prediction, "drug_a", "").lower()
        drug_b = getattr(prediction, "drug_b", "").lower()
        for alert in getattr(patient_context, "prior_alerts", []):
            a = str(alert.get("drug_a", "")).lower()
            b = str(alert.get("drug_b", "")).lower()
            if {a, b} == {drug_a, drug_b} and not alert.get("was_acknowledged"):
                if urgency == Urgency.ROUTINE:
                    urgency = Urgency.PRIORITY
                elif urgency == Urgency.PRIORITY:
                    urgency = Urgency.URGENT
                overrides.append("Prior alert not acknowledged")
                break

        return urgency, " | ".join(overrides)

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        decision:        RoutingDecision,
        dual_report:     Any,
        prediction:      Any,
        patient_context: Any,
        citations:       list,
    ):
        """Dispatch to every destination. Each failure is logged, never raised."""
        patient_name     = getattr(patient_context, "name",       "Patient") if patient_context else "Patient"
        patient_id       = getattr(patient_context, "patient_id", "unknown") if patient_context else "unknown"
        clinician_report = getattr(dual_report, "clinician_report", "")
        patient_report   = getattr(dual_report, "patient_report",   "")

        for dest in decision.destinations:

            if dest == Destination.LOG:
                self._log_to_console(decision, patient_name, clinician_report)
                decision.delivery_log.append({"channel": "log", "success": True})

            elif dest == Destination.PATIENT:
                ok = self._notify_patient(
                    patient_id, patient_name,
                    decision.drug_a, decision.drug_b,
                    decision.severity, patient_report,
                )
                decision.delivery_log.append({"channel": "patient", "success": ok})

            elif dest == Destination.PHARMACIST:
                ok = self._notify_pharmacist(
                    patient_name, decision.drug_a, decision.drug_b,
                    decision.severity, decision.urgency_label(),
                    clinician_report, citations,
                )
                decision.delivery_log.append({"channel": "pharmacist", "success": ok})

            elif dest == Destination.PHYSICIAN:
                ok = self._notify_physician(
                    patient_name, decision.drug_a, decision.drug_b,
                    decision.severity, decision.urgency_label(),
                    clinician_report, decision.override_reason,
                )
                decision.delivery_log.append({"channel": "physician", "success": ok})

            elif dest == Destination.EHR:
                ok = self._flag_ehr(
                    patient_id, decision.drug_a, decision.drug_b,
                    decision.severity, decision.urgency_label(),
                    decision.override_reason,
                )
                decision.delivery_log.append({"channel": "ehr", "success": ok})

    # ── Channel implementations ───────────────────────────────────────────────

    @staticmethod
    def _log_to_console(decision: RoutingDecision, patient_name: str, clinician_report: str):
        sep = "═" * 68
        log.info(
            "\n%s\n%s  DRUG INTERACTION ALERT\n"
            "Patient : %s\n"
            "Pair    : %s + %s\n"
            "Severity: %d/3  Urgency: %s\n"
            "%s\n"
            "%s...\n%s",
            sep, URGENCY_EMOJI.get(decision.urgency, ""),
            patient_name, decision.drug_a, decision.drug_b,
            decision.severity, decision.urgency_label(),
            decision.override_reason or "(no escalation override)",
            clinician_report[:500], sep,
        )

    def _notify_patient(
        self, patient_id: str, patient_name: str,
        drug_a: str, drug_b: str, severity: int, patient_report: str,
    ) -> bool:
        payload = {
            "type": "patient_drug_alert", "patient_id": patient_id,
            "patient_name": patient_name, "drug_a": drug_a, "drug_b": drug_b,
            "severity": severity, "message": patient_report,
        }
        return self._post_webhook(payload, label=f"patient-{patient_id}")

    def _notify_pharmacist(
        self, patient_name: str, drug_a: str, drug_b: str,
        severity: int, urgency_str: str, clinician_report: str, citations: list,
    ) -> bool:
        citation_keys = [getattr(c, "key", str(c)) for c in citations]
        subject = f"[DrugWatchdog] {urgency_str} — {drug_a} + {drug_b} ({patient_name})"
        body = (
            f"Pharmacist Review Required\n\n"
            f"Patient  : {patient_name}\n"
            f"Pair     : {drug_a} + {drug_b}\n"
            f"Severity : {severity}/3  Urgency: {urgency_str}\n\n"
            f"--- CLINICIAN REPORT ---\n{clinician_report}\n\n"
            f"Citations: {', '.join(citation_keys)}\n"
        )
        return self._send_email(PHARMACIST_EMAILS, subject, body)

    def _notify_physician(
        self, patient_name: str, drug_a: str, drug_b: str,
        severity: int, urgency_str: str, clinician_report: str, override_reason: str,
    ) -> bool:
        subject = f"[DrugWatchdog] ⚠ URGENT — {drug_a} + {drug_b} ({patient_name})"
        body = (
            f"URGENT PHYSICIAN REVIEW REQUIRED\n\n"
            f"Patient  : {patient_name}\n"
            f"Pair     : {drug_a} + {drug_b}\n"
            f"Severity : {severity}/3  Urgency: {urgency_str}\n"
            f"Override : {override_reason or 'N/A'}\n\n"
            f"--- CLINICIAN REPORT ---\n{clinician_report}\n"
        )
        ok_email = self._send_email(PHYSICIAN_EMAILS, subject, body)
        payload  = {
            "type": "physician_urgent_alert", "patient_name": patient_name,
            "drug_a": drug_a, "drug_b": drug_b, "severity": severity,
            "urgency": urgency_str, "override_reason": override_reason,
            "clinician_report": clinician_report[:1000],
        }
        ok_hook = self._post_webhook(payload, label="physician-alert")
        return ok_email or ok_hook

    def _flag_ehr(
        self, patient_id: str, drug_a: str, drug_b: str,
        severity: int, urgency_str: str, override_reason: str,
    ) -> bool:
        payload = {
            "type": "ehr_flag", "patient_id": patient_id,
            "drug_a": drug_a, "drug_b": drug_b, "severity": severity,
            "urgency": urgency_str, "flag": "DRUG_INTERACTION_ALERT",
            "override_reason": override_reason,
        }
        return self._post_webhook(payload, label=f"ehr-flag-{patient_id}")

    # ── Delivery primitives ───────────────────────────────────────────────────

    def _send_email(self, recipients: list[str], subject: str, body: str) -> bool:
        if not recipients or not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
            log.debug("Email not configured — skipping: %s", subject)
            return True   # not a failure
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SMTP_FROM
            msg["To"]      = ", ".join(recipients)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
                srv.starttls()
                srv.login(SMTP_USER, SMTP_PASSWORD)
                srv.sendmail(SMTP_FROM, recipients, msg.as_string())
            log.info("Email → %s: %s", recipients, subject)
            return True
        except Exception as exc:
            log.warning("Email failed (%s): %s", subject, exc)
            return False

    @staticmethod
    def _post_webhook(payload: dict, label: str = "") -> bool:
        if not WEBHOOK_URL:
            log.debug("Webhook not configured — skipping: %s", label)
            return True
        try:
            resp = http_requests.post(
                WEBHOOK_URL, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            log.info("Webhook ✓ (%s): %d", label, resp.status_code)
            return True
        except Exception as exc:
            log.warning("Webhook failed (%s): %s", label, exc)
            return False

    # ── Alert record builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_alert_record(
        decision: RoutingDecision, dual_report: Any,
        prediction: Any, citations: list,
    ) -> dict:
        """Complete alert dict ready for memory.save_alert()."""
        return {
            "drug_a":            decision.drug_a,
            "drug_b":            decision.drug_b,
            "severity":          getattr(prediction, "severity", 0),
            "severity_label":    getattr(prediction, "severity_label", ""),
            "adjusted_severity": getattr(prediction, "adjusted_severity", None),
            "adjustment_reason": getattr(prediction, "adjustment_reason", ""),
            "cyp_pathway":       getattr(prediction, "mechanism", ""),
            "clinician_report":  getattr(dual_report, "clinician_report", ""),
            "patient_report":    getattr(dual_report, "patient_report",   ""),
            "citations":         [getattr(c, "key", str(c)) for c in citations],
            "shap_features":     list(getattr(prediction, "shap_top", {}).keys()),
            "routed_to":         decision.delivered_to(),
            "urgency":           decision.urgency.value,
            "override_reason":   decision.override_reason,
            "context_hash":      str(hash(f"{decision.drug_a}:{decision.drug_b}")),
        }


# ─────────────────────────── Singleton ───────────────────────────────────────

_agent: AlertRoutingAgent | None = None

def get_routing_agent() -> AlertRoutingAgent:
    global _agent
    if _agent is None:
        _agent = AlertRoutingAgent()
    return _agent


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as dc, field as f

    @dc
    class MockPrediction:
        drug_a            : str   = "warfarin"
        drug_b            : str   = "aspirin"
        severity          : int   = 2
        severity_label    : str   = "moderate"
        adjusted_severity : int   = 3
        adjustment_reason : str   = "High INR + anticoagulant interaction"
        mechanism         : str   = "CYP2C9 inhibition + additive bleed risk"
        shap_top          : dict  = f(default_factory=lambda: {"anticoagulant_flag": 0.42})

    @dc
    class MockReport:
        clinician_report : str = (
            "WARFARIN + ASPIRIN — SEVERE\nAdditive anticoagulation. INR monitoring required."
        )
        patient_report   : str = (
            "These two medications together raise your bleeding risk. "
            "Watch for bruising. Call your doctor before stopping either drug."
        )

    @dc
    class MockPatient:
        name         : str   = "Jane Doe"
        patient_id   : str   = "test-001"
        inr          : float = 3.7
        is_pregnant  : bool  = False
        is_dialysis  : bool  = False
        prior_alerts : list  = f(default_factory=list)

    @dc
    class MockCitation:
        key: str = "FDA-1"

    agent   = AlertRoutingAgent()
    pred    = MockPrediction()
    report  = MockReport()
    patient = MockPatient()
    cites   = [MockCitation("FDA-1"), MockCitation("FAERS-2")]

    print("\n── Alert Routing Smoke Test ──")
    decision = agent.route(report, pred, patient, cites)

    print(f"Urgency     : {decision.urgency_label()}")
    print(f"Destinations: {decision.delivered_to()}")
    print(f"Override    : {decision.override_reason or 'none'}")
    print(f"Latency     : {decision.latency_ms:.0f} ms")
    print(f"Any failed  : {decision.any_failed()}")
