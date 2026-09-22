"""
patient_context_agent.py  ·  Drug Watchdog Phase 4
====================================================
LangGraph node that adjusts ML-predicted severity using patient-specific
clinical factors.

Why this exists
---------------
The Phase 2 XGBoost model predicts interaction severity for a "generic"
patient — it knows nothing about the specific person's eGFR, liver function,
age, weight, or comorbidities.  This agent upgrades (or rarely downgrades)
that score using real clinical pharmacology rules.

Key adjustments
---------------
  • Renal impairment   — renally-cleared drugs accumulate; severity +1 if moderate,
                         severity escalated to 3 if severe eGFR <15
  • Hepatic impairment — hepatically-metabolised drugs accumulate; severity +1
  • Elderly (age ≥ 75) — narrowed therapeutic window, polypharmacy risk; +1 cap
  • High INR (>3.0)    — automatic escalation for any anticoagulant interaction
  • Pregnancy          — contraindicated combinations → force severity 3
  • Dialysis           — renally-cleared drugs cannot be excreted; severity +1 or +2
  • Prior ignored alerts— patient did not acknowledge a past warning → escalate

Rules are cumulative but capped at 3.  Every adjustment is logged in the
adjustment_reason field so the explanation agent can cite the logic.

LangGraph state keys
--------------------
  Reads  : state["prediction"]      → PredictionResult
            state["patient_context"]→ PatientContext
            state["drug_pair"]      → (drug_a, drug_b)
  Writes : state["prediction"]      → updated (adjusted_severity filled in)
            state["context_summary"]→ str (for LLM prompt injection)

Usage (standalone)
------------------
  agent = PatientContextAgent()
  adjusted = agent.adjust(prediction_result, patient_context)
  print(adjusted.adjusted_severity, adjusted.adjustment_reason)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────── Drug class dictionaries ─────────────────────────
# Used to detect which pharmacological class a drug belongs to.

# Renally cleared drugs: accumulate when eGFR is low
RENAL_CLEARED: set[str] = {
    "metformin", "gabapentin", "pregabalin", "digoxin", "lithium",
    "atenolol", "sotalol", "methotrexate", "vancomycin", "aminoglycosides",
    "gentamicin", "tobramycin", "amikacin", "acyclovir", "valacyclovir",
    "ciprofloxacin", "levofloxacin", "dabigatran", "rivaroxaban", "apixaban",
    "allopurinol", "colchicine", "ranitidine", "famotidine", "cefazolin",
}

# Hepatically metabolised drugs (CYP2C9, CYP3A4, CYP2D6)
HEPATIC_METABOLISED: set[str] = {
    "warfarin", "simvastatin", "atorvastatin", "lovastatin", "rosuvastatin",
    "metoprolol", "carvedilol", "propranolol", "amitriptyline", "nortriptyline",
    "haloperidol", "risperidone", "codeine", "tramadol", "oxycodone",
    "diazepam", "alprazolam", "midazolam", "ciclosporin", "tacrolimus",
    "erythromycin", "clarithromycin", "ketoconazole", "fluconazole",
    "rifampicin", "phenytoin", "carbamazepine", "phenobarbital",
}

# Anticoagulants — any INR-raising interaction is elevated
ANTICOAGULANTS: set[str] = {
    "warfarin", "heparin", "enoxaparin", "dabigatran", "rivaroxaban",
    "apixaban", "edoxaban", "fondaparinux", "acenocoumarol",
}

# NSAIDs — relevant for pregnancy and bleeding-risk escalation
NSAIDS: set[str] = {
    "ibuprofen", "naproxen", "diclofenac", "celecoxib", "indomethacin",
    "ketorolac", "meloxicam", "piroxicam", "aspirin",
}

# Drugs contraindicated in pregnancy (categories D/X)
PREGNANCY_CONTRAINDICATED: set[str] = {
    "warfarin", "methotrexate", "thalidomide", "isotretinoin",
    "valproate", "valproic acid", "lithium", "tetracycline",
    "doxycycline", "fluoroquinolones", "ciprofloxacin", "levofloxacin",
    "enalapril", "lisinopril", "ramipril",  # ACE inhibitors teratogenic in 2nd/3rd trimester
    "losartan", "valsartan",                 # ARBs similarly contraindicated
    "atorvastatin", "simvastatin", "rosuvastatin",  # statins contraindicated
}

# ─────────────────────────── Adjustment result ───────────────────────────────

@dataclass
class AdjustmentRecord:
    """Tracks each individual severity adjustment made by the agent."""
    rule:        str          # short rule name, e.g. "renal_severe"
    delta:       int          # +1, +2, or 0 (capped)
    reason:      str          # human-readable explanation
    evidence:    str          = ""  # optional clinical reference


# ─────────────────────────── Agent ───────────────────────────────────────────

class PatientContextAgent:
    """
    LangGraph node: applies patient-specific severity adjustments.

    Mutates the PredictionResult.adjusted_severity and .adjustment_reason
    fields in place, and adds a context_summary to the LangGraph state.
    """

    def adjust(
        self,
        prediction: Any,           # PredictionResult from ml_prediction_agent
        patient_context: Any,      # PatientContext from memory.py
    ) -> Any:
        """
        Apply all clinical adjustment rules and update prediction in place.

        Returns the same prediction object with .adjusted_severity and
        .adjustment_reason filled in.
        """
        if patient_context is None:
            log.warning("PatientContextAgent: no patient_context provided — skipping adjustment")
            prediction.adjusted_severity = prediction.severity
            prediction.adjustment_reason = "No patient context — using raw ML prediction."
            return prediction

        base  = prediction.severity
        delta = 0
        adjustments: list[AdjustmentRecord] = []

        drug_a = prediction.drug_a.lower()
        drug_b = prediction.drug_b.lower()
        drugs  = {drug_a, drug_b}

        # ── Rule 1: Renal impairment ──────────────────────────────────────────
        if patient_context.renal_severe and drugs & RENAL_CLEARED:
            # eGFR < 30 → force severity to at least 3 for renally-cleared drugs
            forced = max(base, 3) - base
            adjustments.append(AdjustmentRecord(
                rule    = "renal_severe",
                delta   = forced,
                reason  = (
                    f"Patient has severe renal impairment (eGFR={patient_context.egfr:.0f} mL/min). "
                    f"Drug(s) in this pair are renally cleared and will accumulate, "
                    f"dramatically increasing toxicity risk. Severity escalated to SEVERE."
                ),
                evidence= "Lexicomp Renal Dosing Guidelines; FDA renal impairment guidance",
            ))
            delta = max(delta, forced)

        elif patient_context.renal_impaired and drugs & RENAL_CLEARED:
            # eGFR 30–59 → +1
            adjustments.append(AdjustmentRecord(
                rule    = "renal_moderate",
                delta   = 1,
                reason  = (
                    f"Patient has moderate renal impairment (eGFR={patient_context.egfr:.0f} mL/min). "
                    f"Renally-cleared drug(s) will have reduced clearance, increasing exposure."
                ),
                evidence= "FDA renal impairment labeling guidance",
            ))
            delta += 1

        # ── Rule 2: Hepatic impairment ────────────────────────────────────────
        if patient_context.hepatic_impaired and drugs & HEPATIC_METABOLISED:
            adjustments.append(AdjustmentRecord(
                rule    = "hepatic_impaired",
                delta   = 1,
                reason  = (
                    f"Patient has elevated liver enzymes "
                    f"(AST={patient_context.ast}, ALT={patient_context.alt}), "
                    f"suggesting hepatic impairment. CYP-metabolised drug(s) in this pair "
                    f"will have reduced first-pass metabolism and elevated plasma levels."
                ),
                evidence= "FDA hepatic impairment labeling guidance",
            ))
            delta += 1

        # ── Rule 3: Elderly patient ───────────────────────────────────────────
        if patient_context.elderly:
            adjustments.append(AdjustmentRecord(
                rule    = "elderly",
                delta   = 1,
                reason  = (
                    f"Patient is elderly (age={patient_context.age}). "
                    f"Reduced physiological reserve, polypharmacy, and altered drug PK/PD "
                    f"narrow the therapeutic window for most drug interactions."
                ),
                evidence= "Beers Criteria 2023; AGS Pharmacotherapy Principles",
            ))
            delta += 1

        # ── Rule 4: High INR / anticoagulant interaction ──────────────────────
        if patient_context.high_bleed_risk and drugs & ANTICOAGULANTS:
            escalate = max(0, 3 - (base + delta))  # bump to at least 3
            if escalate > 0:
                adjustments.append(AdjustmentRecord(
                    rule    = "high_inr_anticoagulant",
                    delta   = escalate,
                    reason  = (
                        f"Patient has INR={patient_context.inr:.1f} (>3.0), indicating "
                        f"supratherapeutic anticoagulation. Any interaction involving an "
                        f"anticoagulant at this INR level carries immediate haemorrhage risk."
                    ),
                    evidence= "ISTH Bleeding Risk Guidelines 2021",
                ))
                delta += escalate

        # ── Rule 5: Pregnancy ─────────────────────────────────────────────────
        if patient_context.is_pregnant and drugs & PREGNANCY_CONTRAINDICATED:
            escalate = max(0, 3 - (base + delta))
            adjustments.append(AdjustmentRecord(
                rule    = "pregnancy_contraindicated",
                delta   = escalate,
                reason  = (
                    f"Patient is pregnant. Drug(s) in this pair ({', '.join(drugs & PREGNANCY_CONTRAINDICATED)}) "
                    f"are contraindicated in pregnancy (FDA Category D/X or equivalent). "
                    f"Severity escalated to SEVERE — urgent prescriber review required."
                ),
                evidence= "FDA Drug Safety Labeling; Briggs Drugs in Pregnancy & Lactation",
            ))
            delta += escalate

        # ── Rule 6: Dialysis ──────────────────────────────────────────────────
        if patient_context.is_dialysis and drugs & RENAL_CLEARED:
            # Dialysis patients have effectively zero renal clearance
            escalate = max(0, 3 - (base + delta))
            adjustments.append(AdjustmentRecord(
                rule    = "dialysis",
                delta   = escalate,
                reason  = (
                    f"Patient is on dialysis (eGFR ≈ 0). Renally-cleared drug(s) cannot "
                    f"be excreted and will accumulate to toxic levels. "
                    f"Dose adjustment or drug substitution is required — escalated to SEVERE."
                ),
                evidence= "Lexicomp Dialysis Dosing; KDIGO CKD-MBD Guideline 2017",
            ))
            delta += escalate

        # ── Rule 7: Prior ignored alert (same drug pair) ──────────────────────
        if self._has_unacknowledged_alert(patient_context, drug_a, drug_b):
            adjustments.append(AdjustmentRecord(
                rule    = "prior_ignored_alert",
                delta   = 1,
                reason  = (
                    f"A previous interaction alert for {drug_a} + {drug_b} was generated "
                    f"but not acknowledged by the patient or clinician. "
                    f"Escalating severity to ensure this is not silently repeated."
                ),
            ))
            delta += 1

        # ── Clamp and apply ───────────────────────────────────────────────────
        adjusted = min(3, base + delta)   # never exceed severity 3
        actual_delta = adjusted - base    # how much we actually moved

        prediction.adjusted_severity = adjusted
        prediction.adjustment_reason = self._build_reason_text(
            base, adjusted, actual_delta, adjustments, patient_context
        )

        if actual_delta > 0:
            log.info(
                "Context adjusted %s + %s: %d → %d (+%d) — %d rule(s) applied",
                drug_a, drug_b, base, adjusted, actual_delta, len(adjustments),
            )
        else:
            log.info(
                "Context check %s + %s: severity unchanged at %d — no modifying factors",
                drug_a, drug_b, base,
            )

        return prediction

    # ── LangGraph node entrypoint ─────────────────────────────────────────────

    def run(self, state: dict) -> dict:
        """
        LangGraph node function.

        Reads  : state["prediction"]       → PredictionResult
                 state["patient_context"]  → PatientContext
        Writes : state["prediction"]       → updated with adjusted_severity
                 state["context_summary"]  → str for LLM prompts
        """
        prediction = state.get("prediction")
        patient_ctx = state.get("patient_context")

        if prediction is None:
            log.error("PatientContextAgent.run: state['prediction'] is None")
            return state

        self.adjust(prediction, patient_ctx)

        # Build a short context summary for injection into LLM prompts
        if patient_ctx:
            state["context_summary"] = self._build_context_summary(patient_ctx)
        else:
            state["context_summary"] = "No patient context available."

        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _has_unacknowledged_alert(
        patient_context: Any, drug_a: str, drug_b: str
    ) -> bool:
        """Check if a prior alert for this exact pair was never acknowledged."""
        for alert in getattr(patient_context, "prior_alerts", []):
            alert_a = str(alert.get("drug_a", "")).lower()
            alert_b = str(alert.get("drug_b", "")).lower()
            if (
                {alert_a, alert_b} == {drug_a, drug_b}
                and not alert.get("was_acknowledged", True)
            ):
                return True
        return False

    @staticmethod
    def _build_reason_text(
        base:        int,
        adjusted:    int,
        actual_delta:int,
        adjustments: list[AdjustmentRecord],
        ctx:         Any,
    ) -> str:
        """Format the full adjustment reason for alert storage and LLM context."""
        if actual_delta == 0:
            return "No patient-specific factors warranted severity adjustment."

        parts = [
            f"Severity adjusted from {base} → {adjusted} "
            f"(+{actual_delta}) based on patient-specific clinical factors:\n"
        ]
        for adj in adjustments:
            parts.append(f"  [{adj.rule}] {adj.reason}")
            if adj.evidence:
                parts.append(f"    Evidence: {adj.evidence}")

        parts.append(
            f"\nPatient context: {ctx.organ_function_summary() if hasattr(ctx, 'organ_function_summary') else 'N/A'}"
        )
        return "\n".join(parts)

    @staticmethod
    def _build_context_summary(ctx: Any) -> str:
        """
        Short multi-line summary injected into explanation_agent prompts.
        Covers the most clinically relevant fields.
        """
        lines = [
            f"Patient: {ctx.name}, age {ctx.age or 'unknown'}, sex {ctx.sex or 'unknown'}",
        ]

        if hasattr(ctx, "organ_function_summary"):
            labs = ctx.organ_function_summary()
            if labs != "No lab values on file":
                lines.append(f"Labs: {labs}")

        flags = []
        if getattr(ctx, "is_pregnant",    False): flags.append("PREGNANT")
        if getattr(ctx, "is_breastfeeding",False): flags.append("BREASTFEEDING")
        if getattr(ctx, "is_dialysis",    False): flags.append("ON DIALYSIS")
        if getattr(ctx, "elderly",        False): flags.append("ELDERLY (≥75)")
        if getattr(ctx, "renal_severe",   False): flags.append("SEVERE RENAL IMPAIRMENT")
        elif getattr(ctx, "renal_impaired",False): flags.append("MODERATE RENAL IMPAIRMENT")
        if getattr(ctx, "hepatic_impaired",False): flags.append("HEPATIC IMPAIRMENT")
        if getattr(ctx, "high_bleed_risk",False): flags.append("HIGH BLEED RISK")
        if flags:
            lines.append(f"Risk flags: {', '.join(flags)}")

        conditions = getattr(ctx, "conditions", [])
        if conditions:
            cond_names = [c.get("name", "") for c in conditions[:3]]
            lines.append(f"Conditions: {', '.join(cond_names)}")

        allergies = getattr(ctx, "allergies", [])
        if allergies:
            allergy_drugs = [a.get("drug_name", "") for a in allergies[:3]]
            lines.append(f"Allergies: {', '.join(allergy_drugs)}")

        if hasattr(ctx, "prior_alert_summary"):
            prior = ctx.prior_alert_summary()
            if prior != "No prior interaction alerts on file.":
                lines.append(f"Prior alerts:\n{prior}")

        return "\n".join(lines)


# ─────────────────────────── Singleton ───────────────────────────────────────

_agent: PatientContextAgent | None = None

def get_context_agent() -> PatientContextAgent:
    global _agent
    if _agent is None:
        _agent = PatientContextAgent()
    return _agent


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    # We need PatientContext and PredictionResult for the smoke test.
    # Import lazily to avoid circular dep issues in standalone run.
    import sys

    # ── Mock PatientContext ───────────────────────────────────────────────────
    from dataclasses import dataclass as dc, field as f

    @dc
    class MockPatient:
        name             : str   = "Jane Doe"
        age              : int   = 78
        sex              : str   = "F"
        egfr             : float = 24.0
        ast              : float = 155.0
        alt              : float = 130.0
        inr              : float = 3.4
        creatinine       : float = 2.1
        hba1c            : float = None
        potassium        : float = None
        hemoglobin       : float = None
        is_pregnant      : bool  = False
        is_breastfeeding : bool  = False
        is_dialysis      : bool  = False
        smoker           : bool  = False
        alcohol_use      : str   = "none"
        renal_impaired   : bool  = True
        renal_severe     : bool  = True    # eGFR=24 < 30
        hepatic_impaired : bool  = True    # AST/ALT elevated
        elderly          : bool  = True    # age=78 ≥ 75
        high_bleed_risk  : bool  = True    # INR=3.4 > 3.0
        conditions       : list  = f(default_factory=lambda: [
            {"name": "Atrial fibrillation"},
            {"name": "CKD Stage 4"},
        ])
        allergies        : list  = f(default_factory=lambda: [
            {"drug_name": "penicillin", "reaction_type": "anaphylaxis"}
        ])
        prior_alerts     : list  = f(default_factory=list)
        sessions_count   : int   = 3

        def organ_function_summary(self):
            return (f"eGFR={self.egfr:.0f} ⚠ SEVERE  AST={self.ast:.0f} ⚠ ELEVATED  "
                    f"ALT={self.alt:.0f} ⚠ ELEVATED  INR={self.inr:.1f} ⚠ HIGH  "
                    f"Cr={self.creatinine:.1f} mg/dL")

        def prior_alert_summary(self):
            return "No prior interaction alerts on file."

    # ── Mock PredictionResult ─────────────────────────────────────────────────
    @dc
    class MockPrediction:
        drug_a            : str   = "warfarin"
        drug_b            : str   = "aspirin"
        severity          : int   = 2    # Phase 2 said moderate
        severity_label    : str   = "moderate"
        severity_emoji    : str   = "🟠"
        confidence        : float = 0.88
        mechanism         : str   = "Additive anticoagulation + GI bleed risk"
        shap_top          : dict  = f(default_factory=dict)
        adjusted_severity : int   = None
        adjustment_reason : str   = ""

    patient = MockPatient()
    prediction = MockPrediction()

    agent = PatientContextAgent()
    agent.adjust(prediction, patient)

    print(f"\n── Patient Context Adjustment ──")
    print(f"Drug pair       : {prediction.drug_a} + {prediction.drug_b}")
    print(f"Base severity   : {prediction.severity} (moderate)")
    print(f"Adjusted        : {prediction.adjusted_severity} (max=3=severe)")
    print(f"\nAdjustment reason:\n{prediction.adjustment_reason}")
    print(f"\nContext summary:\n{agent._build_context_summary(patient)}")