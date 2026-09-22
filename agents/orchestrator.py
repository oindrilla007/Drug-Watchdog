"""
orchestrator.py  ·  Drug Watchdog Phase 4
==========================================
The top-level entry point for a complete Drug Watchdog analysis session.

What it does
------------
  1. Load patient history from Supabase via memory.py
  2. Accept input (text drug list OR image via vision_intake_agent)
  3. Generate all unique drug pairs from the medication list
  4. For each pair: run the full LangGraph pipeline (graph.py)
  5. Collect all alerts, sort by severity
  6. Proactively sweep new medications against the existing med list
  7. Return a structured SessionResult

The orchestrator owns the session-level logic.
The graph owns the per-pair pipeline logic.

ReAct loop (per pair)
---------------------
  The graph.py handles ReAct internally via a conditional edge.
  The orchestrator simply calls graph.invoke() per pair and reads results.

Proactive med-change sweep
--------------------------
  If new_medications is passed (different from patient's stored list),
  all new drugs are cross-checked against every existing medication —
  not just against each other. This is the feature that sets Drug Watchdog
  apart from reactive checkers.

Usage
-----
  orch = Orchestrator()

  # Text input
  result = orch.run(
      patient_id  = "patient-uuid",
      drug_list   = ["warfarin 5mg", "aspirin 81mg", "lisinopril 10mg"],
      input_method= "text",
  )

  # Image input
  result = orch.run(
      patient_id   = "patient-uuid",
      image_b64    = "<base64>",
      image_mode   = "prescription",
  )

  for alert in result.alerts:
      print(alert["drug_a"], alert["drug_b"], alert["adjusted_severity"])
      print(alert["clinician_report"])

  print(result.summary())
"""
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

load_project_env()

log = logging.getLogger(__name__)

# ─────────────────────────── Session result ──────────────────────────────────

@dataclass
class SessionResult:
    patient_id:      str
    patient_name:    str
    input_method:    str
    drugs_analysed:  list[str]
    pairs_analysed:  int
    alerts:          list[dict]          # one dict per pair, sorted sev desc
    total_latency_ms:float
    error:           str                 = ""

    @property
    def high_severity_count(self) -> int:
        return sum(1 for a in self.alerts if a.get("final_severity", 0) >= 2)

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.alerts if a.get("urgency") == "critical")

    def summary(self) -> str:
        lines = [
            f"╔══ Drug Watchdog Session ══════════════════════════════════╗",
            f"║ Patient   : {self.patient_name:<46}║",
            f"║ Input     : {self.input_method:<46}║",
            f"║ Drugs     : {len(self.drugs_analysed):<46}║",
            f"║ Pairs     : {self.pairs_analysed:<46}║",
            f"║ Alerts ≥2 : {self.high_severity_count:<46}║",
            f"║ Critical  : {self.critical_count:<46}║",
            f"║ Latency   : {self.total_latency_ms/1000:.1f}s{'':<43}║",
            f"╚══════════════════════════════════════════════════════════╝",
        ]
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        for a in self.alerts:
            sev = a.get("final_severity", 0)
            emoji = ["🟢", "🟡", "🟠", "🔴"][min(sev, 3)]
            lines.append(
                f"  {emoji} {a.get('drug_a','?'):>18} + {a.get('drug_b','?'):<18}"
                f"  sev={sev}  → {', '.join(a.get('routed_to', []))}"
            )
        return "\n".join(lines)


# ─────────────────────────── Orchestrator ────────────────────────────────────

class Orchestrator:
    """
    Session-level coordinator.

    Loads patient context → resolves drug list → fans out to graph per pair
    → collects results → triggers proactive sweep on med changes.
    """

    def __init__(
        self,
        use_vision: bool = True,
        use_memory: bool = True,
        min_severity_to_report: int = 0,    # 0 = report all pairs
    ):
        self._use_vision = use_vision
        self._use_memory = use_memory
        self._min_sev    = min_severity_to_report

        # Lazy-load graph and memory to avoid import cycles at module level
        self._graph = None
        self._mem   = None

        log.info(
            "Orchestrator initialised — vision=%s  memory=%s  min_sev=%d",
            use_vision, use_memory, min_severity_to_report,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        patient_id:       str,
        drug_list:        list[str] | None = None,
        image_b64:        str | None       = None,
        image_mode:       str              = "auto",
        new_medications:  list[str] | None = None,
        input_method:     str              = "text",
    ) -> SessionResult:
        """
        Run a complete analysis session for one patient.

        Parameters
        ----------
        patient_id       : Supabase patient UUID (required)
        drug_list        : Explicit drug list (text input mode)
        image_b64        : Base64-encoded image (vision input mode)
        image_mode       : "prescription" | "pill_photo" | "lab_report" | "auto"
        new_medications  : If provided, these are cross-checked against ALL existing
                           patient meds (proactive sweep on med change)
        input_method     : Logged to session record

        Returns
        -------
        SessionResult with all alerts sorted by severity descending
        """
        t0 = time.perf_counter()

        # ── 1. Load patient context ───────────────────────────────────────────
        patient_ctx = self._load_patient_context(patient_id)
        patient_name = getattr(patient_ctx, "name", "Unknown") if patient_ctx else "Unknown"

        # ── 2. Resolve drug list ──────────────────────────────────────────────
        resolved_drugs, intake_result, method = self._resolve_drug_list(
            drug_list    = drug_list,
            image_b64    = image_b64,
            image_mode   = image_mode,
            patient_ctx  = patient_ctx,
            input_method = input_method,
        )

        if not resolved_drugs:
            return SessionResult(
                patient_id       = patient_id,
                patient_name     = patient_name,
                input_method     = method,
                drugs_analysed   = [],
                pairs_analysed   = 0,
                alerts           = [],
                total_latency_ms = (time.perf_counter() - t0) * 1000,
                error            = "No drugs could be resolved from the input.",
            )

        log.info(
            "Session for %s (%s): %d drugs → %d pairs",
            patient_name, patient_id, len(resolved_drugs),
            len(list(itertools.combinations(resolved_drugs, 2))),
        )

        # ── 3. Proactive sweep: cross new meds with ALL existing meds ─────────
        all_drug_pairs = self._build_pairs(resolved_drugs, new_medications, patient_ctx)

        # ── 4. Fan out: run graph per pair ────────────────────────────────────
        app    = self._get_graph()
        alerts = []

        for pair in all_drug_pairs:
            try:
                alert = self._run_pair(app, pair, patient_ctx, intake_result, method)
                if alert and alert.get("final_severity", 0) >= self._min_sev:
                    alerts.append(alert)
            except Exception as exc:
                log.error("Pair %s + %s failed: %s", pair[0], pair[1], exc)
                alerts.append({
                    "drug_a": pair[0], "drug_b": pair[1],
                    "final_severity": 0, "error": str(exc),
                })

        # Sort by final_severity desc, then confidence desc
        alerts.sort(key=lambda a: (-a.get("final_severity", 0), -a.get("confidence", 0)))

        # ── 5. Update patient med list if vision intake added new meds ───────
        if intake_result and self._use_memory:
            self._update_patient_meds(patient_id, intake_result)

        total_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Session complete: %d pairs analysed, %d alerts (≥sev2: %d) in %.1fs",
            len(all_drug_pairs), len(alerts),
            sum(1 for a in alerts if a.get("final_severity", 0) >= 2),
            total_ms / 1000,
        )

        return SessionResult(
            patient_id       = patient_id,
            patient_name     = patient_name,
            input_method     = method,
            drugs_analysed   = resolved_drugs,
            pairs_analysed   = len(all_drug_pairs),
            alerts           = alerts,
            total_latency_ms = total_ms,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_patient_context(self, patient_id: str) -> Any | None:
        """Load PatientContext from Supabase. Returns None on failure."""
        if not self._use_memory:
            return None
        try:
            mem = self._get_memory()
            ctx = mem.load_patient_context(patient_id)
            log.info(
                "Loaded context for %s — %d meds, %d prior alerts",
                getattr(ctx, "name", patient_id),
                len(getattr(ctx, "medications", [])),
                len(getattr(ctx, "prior_alerts", [])),
            )
            return ctx
        except Exception as exc:
            log.warning("Could not load patient context for %s: %s", patient_id, exc)
            return None

    def _resolve_drug_list(
        self,
        drug_list:   list[str] | None,
        image_b64:   str | None,
        image_mode:  str,
        patient_ctx: Any,
        input_method:str,
    ) -> tuple[list[str], Any, str]:
        """
        Resolve drugs from text input, image input, or patient's stored med list.
        Returns (drug_names, intake_result_or_None, method_used).
        """
        intake_result = None

        # Priority 1: explicit drug list (text input)
        if drug_list:
            return list(drug_list), None, input_method

        # Priority 2: image input via vision_intake_agent
        if image_b64 and self._use_vision:
            try:
                try:
                    from .vision_intake_agent import VisionIntakeAgent
                except ImportError:
                    from vision_intake_agent import VisionIntakeAgent
                agent  = VisionIntakeAgent()
                result = agent.intake_from_b64(image_b64, mode=image_mode)
                intake_result = result
                if result.success and result.drugs:
                    log.info(
                        "Vision intake: %d drugs from %s (conf=%.2f)",
                        len(result.drugs), image_mode, result.confidence,
                    )
                    return result.drugs, result, result.mode.value
                else:
                    log.warning("Vision intake returned no drugs: %s", result.error)
            except Exception as exc:
                log.error("Vision intake failed: %s", exc)

        # Priority 3: patient's stored medication list
        if patient_ctx:
            stored = getattr(patient_ctx, "drug_name_list", lambda: [])()
            if stored:
                log.info("Using stored medication list: %d drugs", len(stored))
                return stored, None, "stored_profile"

        return [], None, input_method

    @staticmethod
    def _build_pairs(
        resolved_drugs:  list[str],
        new_medications: list[str] | None,
        patient_ctx:     Any,
    ) -> list[tuple[str, str]]:
        """
        Build all unique drug pairs to analyse.

        Standard mode:   all combinations within resolved_drugs
        Proactive mode:  new_medications × (resolved_drugs + stored meds)
        """
        try:
            from .ml_prediction_agent import normalise_drug_name
        except ImportError:
            from ml_prediction_agent import normalise_drug_name

        if new_medications:
            # Proactive sweep: new drugs vs entire medication universe
            existing = [normalise_drug_name(d) for d in resolved_drugs]
            stored   = []
            if patient_ctx:
                stored = getattr(patient_ctx, "drug_name_list", lambda: [])()
            all_existing = list(dict.fromkeys(existing + [normalise_drug_name(d) for d in stored]))
            new_normed   = [normalise_drug_name(d) for d in new_medications]

            pairs = set()
            for new in new_normed:
                for existing_drug in all_existing:
                    if new != existing_drug:
                        pairs.add(tuple(sorted([new, existing_drug])))
            # Also check new vs new
            for a, b in itertools.combinations(new_normed, 2):
                pairs.add(tuple(sorted([a, b])))

            log.info(
                "Proactive sweep: %d new × %d existing → %d pairs",
                len(new_normed), len(all_existing), len(pairs),
            )
            return [tuple(p) for p in pairs]  # type: ignore

        # Standard: all combinations within the resolved list
        normed = [normalise_drug_name(d) for d in resolved_drugs]
        return list(itertools.combinations(normed, 2))

    def _run_pair(
        self,
        app:          Any,
        pair:         tuple[str, str],
        patient_ctx:  Any,
        intake_result:Any,
        input_method: str,
    ) -> dict:
        """
        Invoke the LangGraph for a single drug pair.
        Returns a flat alert dict for SessionResult.alerts.
        """
        try:
            from .graph import WatchdogState
        except ImportError:
            from graph import WatchdogState

        initial_state: WatchdogState = {
            "drug_pair":        pair,
            "patient_context":  patient_ctx,
            "input_method":     input_method,
            "react_iterations": 0,
            "intake_result":    intake_result,
        }

        result = app.invoke(initial_state)

        prediction = result.get("prediction")
        routing    = result.get("routing_decision")
        dual       = result.get("dual_report")

        final_sev = (
            getattr(prediction, "adjusted_severity", None)
            or getattr(prediction, "severity", 0)
        ) if prediction else 0

        return {
            "drug_a":            pair[0],
            "drug_b":            pair[1],
            "severity":          getattr(prediction, "severity", 0) if prediction else 0,
            "final_severity":    final_sev,
            "severity_label":    getattr(prediction, "severity_label", "") if prediction else "",
            "severity_emoji":    getattr(prediction, "severity_emoji", "") if prediction else "",
            "confidence":        getattr(prediction, "confidence", 0.0) if prediction else 0.0,
            "mechanism":         getattr(prediction, "mechanism", "") if prediction else "",
            "adjustment_reason": getattr(prediction, "adjustment_reason", "") if prediction else "",
            "clinician_report":  getattr(dual, "clinician_report", "") if dual else "",
            "patient_report":    getattr(dual, "patient_report", "") if dual else "",
            "urgency":           routing.urgency.value if routing else "",
            "routed_to":         routing.delivered_to() if routing else [],
            "override_reason":   getattr(routing, "override_reason", "") if routing else "",
            "provider_used":     result.get("report_metadata", {}).get("provider_used", ""),
            "react_iterations":  result.get("react_iterations", 0),
            "citations":         [
                getattr(c, "key", str(c)) for c in result.get("citations", [])
            ],
        }

    def _update_patient_meds(self, patient_id: str, intake_result: Any):
        """Save medications extracted from vision intake to patient profile."""
        try:
            mem  = self._get_memory()
            meds = intake_result.to_medication_list()
            if meds:
                mem.save_medications(patient_id, meds)
                log.info("Updated medication list for %s: %d meds", patient_id, len(meds))
        except Exception as exc:
            log.warning("Could not update patient meds: %s", exc)

    # ── Lazy singletons ───────────────────────────────────────────────────────

    def _get_graph(self):
        if self._graph is None:
            try:
                from .graph import build_graph
            except ImportError:
                from graph import build_graph
            self._graph = build_graph(
                use_vision = self._use_vision,
                use_memory = self._use_memory,
            )
        return self._graph

    def _get_memory(self):
        if self._mem is None:
            try:
                from .memory import get_memory
            except ImportError:
                from memory import get_memory
            self._mem = get_memory()
        return self._mem


# ─────────────────────────── Singleton ───────────────────────────────────────

_orchestrator: Orchestrator | None = None

def get_orchestrator(**kwargs) -> Orchestrator:
    """Return a shared Orchestrator singleton (lazy init)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(**kwargs)
    return _orchestrator


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    print("\n── Drug Watchdog Orchestrator Smoke Test ──\n")

    orch = Orchestrator(use_vision=False, use_memory=False)

    result = orch.run(
        patient_id  = "demo-patient-001",
        drug_list   = ["warfarin 5mg", "aspirin 81mg", "simvastatin 40mg", "lisinopril 10mg"],
        input_method= "text",
    )

    print(result.summary())

    print("\n── Alert details ──")
    for alert in result.alerts:
        emoji = ["🟢","🟡","🟠","🔴"][min(alert.get("final_severity", 0), 3)]
        print(
            f"\n{emoji} {alert['drug_a']} + {alert['drug_b']}"
            f"  sev={alert.get('final_severity',0)}  conf={alert.get('confidence',0):.2f}"
            f"  urgency={alert.get('urgency','?')}"
        )
        if alert.get("clinician_report"):
            print(f"  Clinician (first 200): {alert['clinician_report'][:200]}")
        if alert.get("patient_report"):
            print(f"  Patient  (first 200): {alert['patient_report'][:200]}")

    print("\n── Proactive sweep test (adding metformin) ──")
    result2 = orch.run(
        patient_id      = "demo-patient-001",
        drug_list       = ["warfarin", "lisinopril"],
        new_medications = ["metformin"],
        input_method    = "text",
    )
    print(f"New pairs checked: {result2.pairs_analysed}")
    for alert in result2.alerts:
        print(f"  {alert['drug_a']} + {alert['drug_b']} → sev={alert.get('final_severity',0)}")
