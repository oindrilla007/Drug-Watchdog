"""
ml_prediction_agent.py  ·  Drug Watchdog Phase 4
==================================================
LangGraph node that wraps the Phase 2 XGBoost predictor.

Responsibilities
----------------
  • Accept a (drug_a, drug_b, patient_context) tuple from the orchestrator
  • Call the Phase 2 predictor to get severity score (0–3) + confidence
  • Return SHAP feature importances so the explanation agent can cite reasons
  • Normalise drug names before prediction (handle brand→generic, case)
  • Emit a structured PredictionResult that downstream nodes consume

Phase 2 integration contract
-----------------------------
  Phase 2 exposes:  predictor.predict(drug_a, drug_b) -> dict
  Expected return:
    {
      "severity":    int,          # 0=none 1=mild 2=moderate 3=severe
      "confidence":  float,        # 0.0–1.0
      "shap_values": {str: float}, # top feature → contribution
      "mechanism":   str,          # short text e.g. "CYP2C9 inhibition"
    }
  If Phase 2 is not importable (demo/test mode), this agent uses a
  lookup-table fallback so the Phase 4 pipeline still runs end-to-end.

LangGraph state key produced
-----------------------------
  state["prediction"]  →  PredictionResult (one per drug pair)
  The orchestrator collects these into state["predictions"] (list).

Usage (standalone)
------------------
  agent = MLPredictionAgent()
  result = agent.predict("warfarin", "aspirin", patient_ctx)
  print(result.severity, result.confidence, result.shap_top)
"""

import importlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────── Severity labels ─────────────────────────────────

SEVERITY_LABEL = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
}

SEVERITY_EMOJI = {
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴",
}

# ─────────────────────────── Fallback lookup table ───────────────────────────
# Used when Phase 2 predictor.py is not importable (demo / CI / eval mode).
# Keys are frozensets so order doesn't matter.

_FALLBACK_TABLE: dict[frozenset, dict] = {
    frozenset({"warfarin", "aspirin"}): {
        "severity": 3, "confidence": 0.91,
        "mechanism": "Additive anticoagulation + GI bleed risk via COX-1 inhibition",
        "shap_values": {"anticoagulant_flag": 0.42, "bleeding_risk_score": 0.31, "cyp2c9_substrate": 0.18},
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": 3, "confidence": 0.88,
        "mechanism": "CYP2C9 inhibition elevates warfarin plasma levels; additive bleed risk",
        "shap_values": {"cyp2c9_inhibitor": 0.45, "anticoagulant_flag": 0.38, "nsaid_flag": 0.22},
    },
    frozenset({"metformin", "contrast_dye"}): {
        "severity": 2, "confidence": 0.85,
        "mechanism": "Lactic acidosis risk in renally impaired patients",
        "shap_values": {"renal_clearance_flag": 0.51, "lactic_acidosis_risk": 0.29},
    },
    frozenset({"simvastatin", "clarithromycin"}): {
        "severity": 3, "confidence": 0.93,
        "mechanism": "CYP3A4 inhibition dramatically increases simvastatin AUC → myopathy/rhabdomyolysis",
        "shap_values": {"cyp3a4_inhibitor": 0.55, "statin_flag": 0.34, "myopathy_risk": 0.21},
    },
    frozenset({"ssri", "tramadol"}): {
        "severity": 3, "confidence": 0.87,
        "mechanism": "Serotonin syndrome risk; tramadol also lowers seizure threshold",
        "shap_values": {"serotonergic_flag": 0.48, "seizure_risk": 0.27, "cyp2d6_inhibitor": 0.19},
    },
    frozenset({"lisinopril", "potassium"}): {
        "severity": 2, "confidence": 0.82,
        "mechanism": "ACE inhibitor reduces aldosterone → potassium retention → hyperkalemia",
        "shap_values": {"ace_inhibitor_flag": 0.44, "hyperkalemia_risk": 0.36},
    },
    frozenset({"metoprolol", "verapamil"}): {
        "severity": 2, "confidence": 0.84,
        "mechanism": "Additive AV nodal blockade; bradycardia and heart block risk",
        "shap_values": {"av_node_effect": 0.52, "beta_blocker_flag": 0.31, "calcium_channel_flag": 0.28},
    },
    frozenset({"ciprofloxacin", "antacid"}): {
        "severity": 1, "confidence": 0.79,
        "mechanism": "Chelation reduces ciprofloxacin absorption by up to 90%",
        "shap_values": {"chelation_risk": 0.61, "bioavailability_flag": 0.25},
    },
}

# ─────────────────────────── Drug name normalisation ─────────────────────────

# Brand → generic mapping (extend as needed)
_BRAND_TO_GENERIC: dict[str, str] = {
    "tylenol":     "acetaminophen",
    "advil":       "ibuprofen",
    "motrin":      "ibuprofen",
    "aleve":       "naproxen",
    "coumadin":    "warfarin",
    "zocor":       "simvastatin",
    "lipitor":     "atorvastatin",
    "prozac":      "fluoxetine",
    "zoloft":      "sertraline",
    "paxil":       "paroxetine",
    "xanax":       "alprazolam",
    "valium":      "diazepam",
    "lasix":       "furosemide",
    "glucophage":  "metformin",
    "glucophage xr": "metformin",
    "crestor":     "rosuvastatin",
    "norvasc":     "amlodipine",
    "zithromax":   "azithromycin",
    "biaxin":      "clarithromycin",
    "cipro":       "ciprofloxacin",
    "augmentin":   "amoxicillin-clavulanate",
    "bactrim":     "trimethoprim-sulfamethoxazole",
    "prinivil":    "lisinopril",
    "zestril":     "lisinopril",
    "toprol":      "metoprolol",
    "lopressor":   "metoprolol",
}


def normalise_drug_name(name: str) -> str:
    """
    Lowercase, strip dose/frequency, convert brand → generic.
    e.g. "Warfarin 5mg daily" → "warfarin"
         "Advil 400mg" → "ibuprofen"
    """
    # Lowercase and strip dose info
    cleaned = re.sub(r"\d+\.?\d*\s*(mg|mcg|g|ml|units?|iu|%)\b.*$", "", name.lower()).strip()
    # Remove trailing frequency words
    cleaned = re.sub(r"\s+(daily|twice|tid|qid|bid|once|weekly|prn|as needed).*$", "", cleaned).strip()
    # Brand → generic
    return _BRAND_TO_GENERIC.get(cleaned, cleaned)


# ─────────────────────────── Result dataclass ────────────────────────────────

@dataclass
class PredictionResult:
    drug_a:       str
    drug_b:       str
    severity:     int            # 0–3
    severity_label: str          # "none"|"mild"|"moderate"|"severe"
    severity_emoji: str
    confidence:   float          # 0.0–1.0
    mechanism:    str            # short pharmacological mechanism
    shap_top:     dict[str, float] = field(default_factory=dict)
    # Top SHAP features → contribution scores

    source:       str            = "phase2_xgboost"  # or "fallback_table"
    latency_ms:   float          = 0.0

    # Will be adjusted later by patient_context_agent
    adjusted_severity: int | None = None
    adjustment_reason: str        = ""

    def needs_react_loop(self, threshold: float = 0.6) -> bool:
        """
        Returns True if confidence is below threshold, signalling the
        orchestrator to trigger a ReAct loop for more evidence retrieval.
        """
        return self.confidence < threshold

    def to_agent_context(self) -> str:
        """
        One-line summary injected into downstream agent prompts.
        e.g. "warfarin + aspirin: SEVERE (conf=0.91) — CYP2C9 inhibition..."
        """
        sev = self.adjusted_severity if self.adjusted_severity is not None else self.severity
        label = SEVERITY_LABEL.get(sev, "unknown").upper()
        return (
            f"{self.drug_a} + {self.drug_b}: {label} "
            f"(conf={self.confidence:.2f}) — {self.mechanism}"
        )

    def shap_summary(self) -> str:
        """Formatted SHAP feature summary for clinician report."""
        if not self.shap_top:
            return "No SHAP data available."
        lines = [f"  {feat}: {val:+.3f}" for feat, val in
                 sorted(self.shap_top.items(), key=lambda x: -abs(x[1]))]
        return "Top predictive features:\n" + "\n".join(lines)


# ─────────────────────────── Agent ───────────────────────────────────────────

class MLPredictionAgent:
    """
    LangGraph node: runs XGBoost severity prediction for a drug pair.

    Attempts to import Phase 2 predictor.py at init time.
    Falls back to the lookup table if import fails.
    """

    def __init__(self):
        self._predictor = self._load_phase2_predictor()
        if self._predictor:
            log.info("MLPredictionAgent: Phase 2 predictor loaded ✓")
        else:
            log.warning(
                "MLPredictionAgent: Phase 2 predictor not found — "
                "using built-in fallback table for demo mode"
            )

    # ── Phase 2 loader ────────────────────────────────────────────────────────

    @staticmethod
    def _load_phase2_predictor() -> Any | None:
        """
        Try to import predictor from Phase 2.
        Expected interface: predictor.predict(drug_a: str, drug_b: str) -> dict
        Searches common relative paths used in the project structure.
        """
        search_paths = [
            "predictor",                      # same dir / PYTHONPATH
            "phase2.predictor",
            "src.phase2.predictor",
            "drug_watchdog.phase2.predictor",
        ]
        for module_path in search_paths:
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, "predict"):
                    return mod
                log.debug("Module %s found but has no predict() function", module_path)
            except ImportError:
                continue
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        patient_context: Any | None = None,  # PatientContext from memory.py
    ) -> PredictionResult:
        """
        Run severity prediction for a drug pair.

        Parameters
        ----------
        drug_a, drug_b    : Drug names (brand or generic, with/without dose)
        patient_context   : Optional PatientContext — used to log context
                            hash for provenance; NOT used to adjust severity
                            here (that's patient_context_agent's job)

        Returns
        -------
        PredictionResult
        """
        t0 = time.perf_counter()

        norm_a = normalise_drug_name(drug_a)
        norm_b = normalise_drug_name(drug_b)

        raw: dict
        source: str

        if self._predictor:
            raw, source = self._call_phase2(norm_a, norm_b)
        else:
            raw, source = self._call_fallback(norm_a, norm_b)

        severity = int(raw.get("severity", 0))
        severity = max(0, min(3, severity))  # clamp to 0–3

        result = PredictionResult(
            drug_a         = norm_a,
            drug_b         = norm_b,
            severity       = severity,
            severity_label = SEVERITY_LABEL[severity],
            severity_emoji = SEVERITY_EMOJI[severity],
            confidence     = float(raw.get("confidence", 0.5)),
            mechanism      = raw.get("mechanism", "Unknown mechanism"),
            shap_top       = raw.get("shap_values", {}),
            source         = source,
            latency_ms     = (time.perf_counter() - t0) * 1000,
        )

        log.info(
            "Predicted %s + %s → %s %s (conf=%.2f, src=%s, %.0f ms)",
            norm_a, norm_b,
            result.severity_emoji, result.severity_label.upper(),
            result.confidence, source, result.latency_ms,
        )

        return result

    def predict_all_pairs(
        self,
        drug_list: list[str],
        patient_context: Any | None = None,
    ) -> list[PredictionResult]:
        """
        Run predictions for all unique pairs in drug_list.
        Returns list sorted by severity desc, then confidence desc.

        e.g. for ["warfarin", "aspirin", "lisinopril"] → 3 pairs predicted.
        """
        results: list[PredictionResult] = []
        drugs = [normalise_drug_name(d) for d in drug_list]

        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                try:
                    r = self.predict(drugs[i], drugs[j], patient_context)
                    results.append(r)
                except Exception as exc:
                    log.error("Prediction failed for %s + %s: %s", drugs[i], drugs[j], exc)

        return sorted(results, key=lambda r: (-r.severity, -r.confidence))

    # ── LangGraph node entrypoint ─────────────────────────────────────────────

    def run(self, state: dict) -> dict:
        """
        LangGraph node function.

        Reads  : state["drug_pair"]       → (drug_a, drug_b) tuple
                 state["patient_context"] → PatientContext or None
        Writes : state["prediction"]      → PredictionResult
                 state["needs_react"]     → bool (confidence < 0.6)
        """
        pair = state.get("drug_pair")
        if not pair or len(pair) < 2:
            log.error("MLPredictionAgent.run: state['drug_pair'] missing or malformed")
            return state

        drug_a, drug_b = pair[0], pair[1]
        ctx = state.get("patient_context")

        prediction = self.predict(drug_a, drug_b, ctx)

        state["prediction"]   = prediction
        state["needs_react"]  = prediction.needs_react_loop()
        return state

    # ── Internal callers ──────────────────────────────────────────────────────

    def _call_phase2(self, drug_a: str, drug_b: str) -> tuple[dict, str]:
        """Call the real Phase 2 XGBoost predictor."""
        try:
            result = self._predictor.predict(drug_a, drug_b)
            if not isinstance(result, dict):
                raise ValueError(f"predict() returned {type(result).__name__}, expected dict")
            return result, "phase2_xgboost"
        except Exception as exc:
            log.warning("Phase 2 predictor raised: %s — falling back", exc)
            return self._call_fallback(drug_a, drug_b)

    @staticmethod
    def _call_fallback(drug_a: str, drug_b: str) -> tuple[dict, str]:
        """
        Lookup-table fallback for demo / test mode.
        If the pair is not in the table, returns severity=0 with low confidence.
        """
        key = frozenset({drug_a, drug_b})
        if key in _FALLBACK_TABLE:
            return _FALLBACK_TABLE[key].copy(), "fallback_table"

        # Generic low-confidence zero result for unknown pairs
        return {
            "severity":    0,
            "confidence":  0.35,
            "mechanism":   "No known direct interaction — verify with clinical resources",
            "shap_values": {},
        }, "fallback_table_miss"


# ─────────────────────────── Singleton ───────────────────────────────────────

_agent: MLPredictionAgent | None = None

def get_ml_agent() -> MLPredictionAgent:
    global _agent
    if _agent is None:
        _agent = MLPredictionAgent()
    return _agent


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    agent = MLPredictionAgent()

    test_pairs = [
        ("Warfarin 5mg daily", "Aspirin 81mg"),
        ("simvastatin 40mg", "Clarithromycin 500mg"),
        ("metoprolol", "Verapamil"),
        ("lisinopril", "potassium chloride"),
        ("amoxicillin", "ibuprofen"),     # likely not in fallback → miss
    ]

    print(f"\n{'Drug A':<28} {'Drug B':<24} {'Sev':>4}  {'Conf':>5}  Mechanism")
    print("─" * 90)
    for a, b in test_pairs:
        r = agent.predict(a, b)
        print(
            f"{r.drug_a:<28} {r.drug_b:<24} "
            f"{r.severity_emoji} {r.severity_label:<8} "
            f"{r.confidence:.2f}  {r.mechanism[:45]}"
        )

    print("\n── Pair sweep for ['warfarin', 'aspirin', 'lisinopril', 'simvastatin'] ──")
    results = agent.predict_all_pairs(["warfarin", "aspirin", "lisinopril", "simvastatin"])
    for r in results:
        print(f"  {r.drug_a} + {r.drug_b}: {r.severity_emoji} {r.severity_label} (conf={r.confidence:.2f})")
        if r.shap_top:
            for feat, val in list(r.shap_top.items())[:2]:
                print(f"    SHAP: {feat} = {val:+.3f}")