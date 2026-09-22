"""
explanation_agent.py  ·  Drug Watchdog Phase 4
================================================
LangGraph node: generates dual clinical + patient reports for each
drug interaction alert using the LLM router.

Two report modes, generated in a single LLM call
-------------------------------------------------
  CLINICIAN report:
    • CYP pathway, pharmacokinetic mechanism, severity justification
    • SHAP feature explanations ("The model weighted X most heavily because…")
    • Evidence citations by key ([FDA-1], [PUB-2])
    • Adjusted severity + patient-specific risk factors
    • Recommended clinical action (monitor / adjust dose / contraindicate)

  PATIENT report:
    • Plain English — no jargon
    • What the interaction means in everyday terms
    • What symptoms to watch for
    • What to do (not panic, call your doctor, what NOT to do)
    • Reassuring tone — clear but not alarming

Both reports are generated from a single prompt that produces JSON with two
keys: "clinician_report" and "patient_report".  This halves LLM calls vs
generating reports separately.

LangGraph state keys
--------------------
  Reads  : state["prediction"]      → PredictionResult
            state["citations"]      → list[Citation]
            state["evidence_text"]  → str
            state["context_summary"]→ str
            state["patient_context"]→ PatientContext
  Writes : state["clinician_report"]→ str
            state["patient_report"] → str
            state["report_metadata"]→ dict (provider, latency, tokens)

Usage (standalone)
------------------
  agent = ExplanationAgent(router)
  reports = agent.generate(prediction, citations, evidence_text, context_summary)
  print(reports.clinician_report)
  print(reports.patient_report)
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .llm_router import LLMRouter, CompletionMode, RouterExhaustedError, get_router
except ImportError:
    from llm_router import LLMRouter, CompletionMode, RouterExhaustedError, get_router

log = logging.getLogger(__name__)

# ─────────────────────────── Constants ───────────────────────────────────────

SEVERITY_LABEL = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
SEVERITY_ACTION = {
    0: "No immediate action required. Document for awareness.",
    1: "Monitor for symptoms. Patient education recommended.",
    2: "Review with pharmacist. Consider dose adjustment or monitoring plan.",
    3: "Urgent prescriber review required. Consider discontinuation or substitution.",
}

# Temperature settings per mode
TEMPERATURE_MAP = {
    CompletionMode.CLINICAL:  0.1,   # very deterministic for clinical reports
    CompletionMode.PATIENT:   0.3,   # slightly more natural for patient text
    CompletionMode.REASONING: 0.2,
}

# ─────────────────────────── Report dataclass ────────────────────────────────

@dataclass
class DualReport:
    drug_a:            str
    drug_b:            str
    clinician_report:  str
    patient_report:    str
    severity:          int
    adjusted_severity: int | None

    provider_used:     str   = ""
    model_used:        str   = ""
    latency_ms:        float = 0.0
    tokens_in:         int   = 0
    tokens_out:        int   = 0
    prompt_version:    str   = "v1"

    def final_severity(self) -> int:
        return self.adjusted_severity if self.adjusted_severity is not None else self.severity

    def severity_label(self) -> str:
        return SEVERITY_LABEL.get(self.final_severity(), "Unknown")

    def recommended_action(self) -> str:
        return SEVERITY_ACTION.get(self.final_severity(), "Review with clinical team.")

    def to_alert_dict(self) -> dict:
        """Serialisable dict for saving to patient_alerts table."""
        return {
            "drug_a":            self.drug_a,
            "drug_b":            self.drug_b,
            "severity":          self.severity,
            "adjusted_severity": self.adjusted_severity,
            "clinician_report":  self.clinician_report,
            "patient_report":    self.patient_report,
            "severity_label":    self.severity_label(),
            "recommended_action":self.recommended_action(),
            "provider_used":     self.provider_used,
            "model_used":        self.model_used,
        }


# ─────────────────────────── Prompt builder ──────────────────────────────────

def _build_prompt(
    drug_a:           str,
    drug_b:           str,
    severity:         int,
    adjusted_severity:int | None,
    mechanism:        str,
    shap_summary:     str,
    evidence_text:    str,
    context_summary:  str,
    adjustment_reason:str,
    patient_language: str = "en",
) -> str:
    """
    Builds the master prompt sent to the LLM.
    Returns a JSON with "clinician_report" and "patient_report".
    """
    final_sev = adjusted_severity if adjusted_severity is not None else severity
    sev_label = SEVERITY_LABEL.get(final_sev, "Unknown").upper()
    action    = SEVERITY_ACTION.get(final_sev, "Review with clinical team.")

    prompt = f"""You are Drug Watchdog, generating interaction reports for the drug pair:
  Drug A: {drug_a}
  Drug B: {drug_b}

SEVERITY: {sev_label} (score={final_sev}/3)
  Raw ML prediction : {severity}/3
  Adjusted for patient: {adjusted_severity if adjusted_severity is not None else 'N/A'}

MECHANISM: {mechanism}

PATIENT CONTEXT:
{context_summary}

SEVERITY ADJUSTMENT REASON:
{adjustment_reason if adjustment_reason else 'No patient-specific adjustments.'}

ML MODEL SHAP FEATURES:
{shap_summary}

CLINICAL EVIDENCE:
{evidence_text}

RECOMMENDED ACTION: {action}

---

Generate two reports in this exact JSON format (no markdown fences, no preamble):
{{
  "clinician_report": "<full technical report for pharmacist/physician>",
  "patient_report": "<plain English report for the patient>"
}}

CLINICIAN REPORT requirements:
- Start with: "{drug_a.upper()} + {drug_b.upper()} — {sev_label} INTERACTION"
- Include: CYP enzyme pathway(s) involved, pharmacokinetic/pharmacodynamic mechanism
- Explain SHAP features: which model features most drove the severity prediction and why
- Cite evidence using the provided keys (e.g. [FDA-1], [PUB-2]) — never fabricate citations
- Include patient-specific risk factors from the context summary
- State the recommended clinical action clearly
- If adjusted_severity > raw severity, explain the clinical rationale for escalation
- Length: 200–350 words
- Tone: precise, clinical, evidence-based

PATIENT REPORT requirements:
- Start with a one-sentence plain-English summary of the interaction
- Explain what could happen in everyday language (no jargon)
- List 2–3 symptoms to watch for
- Give 2–3 clear "what to do" steps (including "tell your doctor or pharmacist")
- Do NOT tell patient to stop taking medications on their own
- Be reassuring but honest about the risk level
- If language="{patient_language}" and not "en": respond in that language
- Length: 120–200 words
- Tone: warm, clear, non-alarmist
"""
    return prompt


# ─────────────────────────── Agent ───────────────────────────────────────────

class ExplanationAgent:
    """
    LangGraph node: generates dual clinician + patient reports via LLM.

    Uses llm_router for automatic Groq → OpenRouter → Ollama fallback.
    For high-severity interactions (≥3) with low RAG confidence, uses
    CompletionMode.REASONING to trigger DeepSeek-R1 chain-of-thought.
    """

    def __init__(self, router: LLMRouter | None = None):
        self._router = router or get_router()
        log.info("ExplanationAgent ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prediction:        Any,        # PredictionResult
        citations:         list,       # list[Citation]
        evidence_text:     str,
        context_summary:   str,
        adjustment_reason: str  = "",
        patient_language:  str  = "en",
        rag_confident:     bool = True,
    ) -> DualReport:
        """
        Generate clinician and patient reports for a single drug-pair interaction.

        Parameters
        ----------
        prediction        : PredictionResult from ml_prediction_agent (adjusted)
        citations         : Citation objects from rag_retrieval_agent
        evidence_text     : Pre-formatted evidence block for prompt injection
        context_summary   : Patient context string from patient_context_agent
        adjustment_reason : Why severity was adjusted (from patient_context_agent)
        patient_language  : ISO 639-1 language code for patient report
        rag_confident     : Whether RAG found strong evidence (score ≥ 0.75)

        Returns
        -------
        DualReport with both reports and metadata
        """
        t0 = time.perf_counter()

        drug_a = prediction.drug_a
        drug_b = prediction.drug_b
        final_sev = (
            prediction.adjusted_severity
            if prediction.adjusted_severity is not None
            else prediction.severity
        )

        # Choose LLM mode:
        # - Severity 3 + low RAG confidence → REASONING (triggers DeepSeek first)
        # - Otherwise → CLINICAL (Groq 70B primary)
        use_reasoning = (final_sev >= 3 and not rag_confident)
        mode = CompletionMode.REASONING if use_reasoning else CompletionMode.CLINICAL

        if use_reasoning:
            log.info(
                "ExplanationAgent: using REASONING mode for %s + %s "
                "(sev=%d, rag_confident=%s)",
                drug_a, drug_b, final_sev, rag_confident,
            )

        # Build SHAP summary string
        shap_summary = (
            prediction.shap_summary()
            if hasattr(prediction, "shap_summary")
            else "No SHAP data available."
        )

        prompt = _build_prompt(
            drug_a            = drug_a,
            drug_b            = drug_b,
            severity          = prediction.severity,
            adjusted_severity = prediction.adjusted_severity,
            mechanism         = prediction.mechanism,
            shap_summary      = shap_summary,
            evidence_text     = evidence_text,
            context_summary   = context_summary,
            adjustment_reason = adjustment_reason,
            patient_language  = patient_language,
        )

        try:
            resp = self._router.complete(
                messages    = [{"role": "user", "content": prompt}],
                mode        = mode,
                max_tokens  = 1500,
                temperature = TEMPERATURE_MAP.get(mode, 0.2),
            )
        except RouterExhaustedError as exc:
            log.error("ExplanationAgent: all LLM providers failed: %s", exc)
            return self._fallback_report(prediction, str(exc))

        clinician_report, patient_report = self._parse_dual_report(resp.text, drug_a, drug_b, final_sev)

        total_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "ExplanationAgent: reports generated for %s + %s via %s (%.0f ms)",
            drug_a, drug_b, resp.provider_used, total_ms,
        )

        return DualReport(
            drug_a            = drug_a,
            drug_b            = drug_b,
            clinician_report  = clinician_report,
            patient_report    = patient_report,
            severity          = prediction.severity,
            adjusted_severity = prediction.adjusted_severity,
            provider_used     = resp.provider_used,
            model_used        = resp.model_used,
            latency_ms        = total_ms,
            tokens_in         = resp.tokens_in,
            tokens_out        = resp.tokens_out,
        )

    # ── LangGraph node entrypoint ─────────────────────────────────────────────

    def run(self, state: dict) -> dict:
        """
        LangGraph node function.

        Reads  : state["prediction"], state["citations"], state["evidence_text"],
                 state["context_summary"], state["patient_context"], state["rag_confident"]
        Writes : state["clinician_report"], state["patient_report"],
                 state["report_metadata"], state["dual_report"]
        """
        prediction   = state.get("prediction")
        citations    = state.get("citations", [])
        evidence_text= state.get("evidence_text", "No evidence available.")
        ctx_summary  = state.get("context_summary", "No patient context.")
        patient_ctx  = state.get("patient_context")
        rag_confident= state.get("rag_confident", True)

        if prediction is None:
            log.error("ExplanationAgent.run: state['prediction'] is None")
            return state

        adj_reason = getattr(prediction, "adjustment_reason", "")
        language   = getattr(patient_ctx, "preferred_language", "en") if patient_ctx else "en"

        dual_report = self.generate(
            prediction        = prediction,
            citations         = citations,
            evidence_text     = evidence_text,
            context_summary   = ctx_summary,
            adjustment_reason = adj_reason,
            patient_language  = language,
            rag_confident     = rag_confident,
        )

        state["clinician_report"] = dual_report.clinician_report
        state["patient_report"]   = dual_report.patient_report
        state["dual_report"]      = dual_report
        state["report_metadata"]  = {
            "provider_used": dual_report.provider_used,
            "model_used":    dual_report.model_used,
            "latency_ms":    dual_report.latency_ms,
            "tokens_in":     dual_report.tokens_in,
            "tokens_out":    dual_report.tokens_out,
        }
        return state

    # ── Parsers + fallbacks ───────────────────────────────────────────────────

    def _parse_dual_report(
        self,
        text:     str,
        drug_a:   str,
        drug_b:   str,
        severity: int,
    ) -> tuple[str, str]:
        """
        Extract clinician_report and patient_report from LLM JSON response.
        Falls back to splitting the raw text if JSON parse fails.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

        # Find JSON block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                clinician = data.get("clinician_report", "").strip()
                patient   = data.get("patient_report",   "").strip()
                if clinician and patient:
                    return clinician, patient
            except json.JSONDecodeError:
                pass

        log.warning("ExplanationAgent: JSON parse failed — using raw text split")
        return self._split_raw_text(text, drug_a, drug_b, severity)

    @staticmethod
    def _split_raw_text(
        text:     str,
        drug_a:   str,
        drug_b:   str,
        severity: int,
    ) -> tuple[str, str]:
        """
        Last-resort parser: if LLM didn't return clean JSON, split at
        "patient report" boundary or return full text as clinician report.
        """
        lower = text.lower()
        splits = ["patient report:", "for the patient:", "patient-friendly:"]
        for split_marker in splits:
            idx = lower.find(split_marker)
            if idx > 0:
                clinician = text[:idx].strip()
                patient   = text[idx + len(split_marker):].strip()
                return clinician, patient

        # Absolute fallback: use full text for clinician, generate minimal patient note
        sev_label = SEVERITY_LABEL.get(severity, "Unknown")
        action    = SEVERITY_ACTION.get(severity, "Contact your healthcare provider.")
        patient_fallback = (
            f"Your medications {drug_a} and {drug_b} may interact. "
            f"This interaction is rated {sev_label.lower()}. "
            f"{action} Please speak with your doctor or pharmacist."
        )
        return text, patient_fallback

    @staticmethod
    def _fallback_report(prediction: Any, error_msg: str) -> DualReport:
        """
        Generates a minimal safe report when all LLM providers fail.
        Ensures the pipeline never returns empty-handed.
        """
        drug_a  = getattr(prediction, "drug_a", "Drug A")
        drug_b  = getattr(prediction, "drug_b", "Drug B")
        sev     = getattr(prediction, "adjusted_severity", None) or getattr(prediction, "severity", 0)
        label   = SEVERITY_LABEL.get(sev, "Unknown")
        action  = SEVERITY_ACTION.get(sev, "Review with clinical team.")

        clinician = (
            f"{drug_a.upper()} + {drug_b.upper()} — {label.upper()} INTERACTION\n\n"
            f"SYSTEM NOTE: LLM report generation failed ({error_msg}). "
            f"Raw ML severity: {sev}/3. Mechanism: {getattr(prediction, 'mechanism', 'Unknown')}.\n\n"
            f"Recommended action: {action}\n\n"
            f"Manual clinical review required — consult Lexicomp/Micromedex."
        )
        patient = (
            f"There may be an interaction between {drug_a} and {drug_b}. "
            f"Our automated report couldn't be generated right now. "
            f"Please contact your doctor or pharmacist for guidance."
        )
        return DualReport(
            drug_a           = drug_a,
            drug_b           = drug_b,
            clinician_report = clinician,
            patient_report   = patient,
            severity         = getattr(prediction, "severity", 0),
            adjusted_severity= getattr(prediction, "adjusted_severity", None),
            provider_used    = "fallback",
            model_used       = "none",
        )


# ─────────────────────────── Singleton ───────────────────────────────────────

_agent: ExplanationAgent | None = None

def get_explanation_agent() -> ExplanationAgent:
    global _agent
    if _agent is None:
        _agent = ExplanationAgent()
    return _agent


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as dc, field as f

    @dc
    class MockPrediction:
        drug_a            : str   = "warfarin"
        drug_b            : str   = "aspirin"
        severity          : int   = 2
        adjusted_severity : int   = 3
        mechanism         : str   = "Additive anticoagulation + COX-1 inhibition"
        shap_top          : dict  = f(default_factory=lambda: {
            "anticoagulant_flag": 0.42, "bleeding_risk_score": 0.31
        })
        adjustment_reason : str   = (
            "Elevated INR=3.4 → anticoagulant interaction escalated to SEVERE. "
            "Elderly (age 78) adds further risk."
        )

        def shap_summary(self):
            lines = [f"  {k}: {v:+.3f}" for k, v in self.shap_top.items()]
            return "Top predictive features:\n" + "\n".join(lines)

    MOCK_EVIDENCE = """[FDA-1] (FDA, relevance=0.93)
Aspirin inhibits platelet aggregation via irreversible COX-1 inhibition and can displace
warfarin from plasma protein binding sites, elevating free warfarin concentration.

[FAERS-2] (FAERS, relevance=0.89)
4,821 serious bleeding reports involving warfarin + aspirin combination (2013–2022).
GI haemorrhage most common (62%). Median INR at event: 3.8."""

    MOCK_CONTEXT = """Patient: Jane Doe, age 78, sex F
Labs: eGFR=24 ⚠ SEVERE  AST=155 ⚠ ELEVATED  INR=3.4 ⚠ HIGH
Risk flags: ELDERLY (≥75), SEVERE RENAL IMPAIRMENT, HEPATIC IMPAIRMENT, HIGH BLEED RISK
Conditions: Atrial fibrillation, CKD Stage 4"""

    agent = ExplanationAgent()
    pred  = MockPrediction()

    print("Generating dual report…")
    report = agent.generate(
        prediction        = pred,
        citations         = [],
        evidence_text     = MOCK_EVIDENCE,
        context_summary   = MOCK_CONTEXT,
        adjustment_reason = pred.adjustment_reason,
        patient_language  = "en",
        rag_confident     = True,
    )

    print(f"\n── Provider: {report.provider_used}  Model: {report.model_used}  "
          f"Latency: {report.latency_ms:.0f}ms ──\n")

    print("═══ CLINICIAN REPORT ═══")
    print(report.clinician_report)
    print("\n═══ PATIENT REPORT ═══")
    print(report.patient_report)
    print(f"\nRecommended action: {report.recommended_action()}")
