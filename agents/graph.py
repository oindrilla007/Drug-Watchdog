"""
graph.py  ·  Drug Watchdog Phase 4
=====================================
LangGraph state graph definition.

This file owns:
  1. WatchdogState  — the TypedDict that flows through every node
  2. build_graph()  — wires all agent nodes into a compiled LangGraph

Node sequence (per drug pair)
------------------------------
  vision_intake  →  ml_prediction  →  rag_retrieval
                                            ↓
                                    [react_loop?]
                                            ↓
                                  patient_context  →  explanation
                                                           ↓
                                                    alert_routing
                                                           ↓
                                                    memory_write
                                                           ↓
                                                         END

ReAct loop
----------
  After rag_retrieval, if state["needs_react"] is True AND
  state["react_iterations"] < MAX_REACT_ITERATIONS, the graph
  routes back to rag_retrieval for a deeper search pass.
  This handles low-confidence cases (e.g. novel drug combinations
  not well-covered in the vector store).

Usage
-----
  from graph import build_graph

  app = build_graph()

  # Run for a single drug pair
  result = app.invoke({
      "drug_pair": ("warfarin", "aspirin"),
      "patient_context": ctx,
  })
  print(result["clinician_report"])
  print(result["routing_decision"].urgency_label())
"""

import logging
from typing import Any, Literal

from typing_extensions import TypedDict

try:
    from .env_loader import load_project_env
except ImportError:
    from env_loader import load_project_env

load_project_env()

log = logging.getLogger(__name__)

# ─────────────────────────── LangGraph import ────────────────────────────────

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    log.warning(
        "langgraph not installed — build_graph() will return a LinearFallbackGraph. "
        "Install with: pip install langgraph"
    )

MAX_REACT_ITERATIONS = 2  # max extra RAG retrieval passes before forcing forward


# ─────────────────────────── State definition ────────────────────────────────

class WatchdogState(TypedDict, total=False):
    """
    The single shared state dict that flows through every LangGraph node.

    Fields are grouped by the node that writes them.
    All fields are optional (total=False) — each node adds its own slice.
    """

    # ── Inputs (set by orchestrator before graph.invoke) ─────────────────────
    drug_pair:         tuple          # (drug_a, drug_b) — normalised strings
    patient_context:   Any            # PatientContext from memory.py
    input_method:      str            # "text" | "prescription" | "pill_photo" | "lab_report"

    # ── vision_intake_agent ───────────────────────────────────────────────────
    intake_result:     Any            # IntakeResult (if image input was used)

    # ── ml_prediction_agent ──────────────────────────────────────────────────
    prediction:        Any            # PredictionResult
    needs_react:       bool           # True if confidence < 0.6

    # ── rag_retrieval_agent ──────────────────────────────────────────────────
    citations:         list           # list[Citation]
    evidence_text:     str            # formatted evidence block for LLM prompt
    rag_confident:     bool           # True if top citation score >= 0.75
    react_iterations:  int            # how many ReAct loops have run

    # ── patient_context_agent ─────────────────────────────────────────────────
    context_summary:   str            # formatted context string for LLM prompt

    # ── explanation_agent ─────────────────────────────────────────────────────
    clinician_report:  str
    patient_report:    str
    dual_report:       Any            # DualReport dataclass
    report_metadata:   dict           # provider, latency, tokens

    # ── alert_routing_agent ───────────────────────────────────────────────────
    routing_decision:  Any            # RoutingDecision dataclass
    alert_record:      dict           # ready for memory.save_alert()

    # ── memory_write (final node) ─────────────────────────────────────────────
    session_saved:     bool
    alert_saved:       bool


# ─────────────────────────── ReAct router ────────────────────────────────────

def _should_react(state: WatchdogState) -> Literal["rag_retrieval", "patient_context"]:
    """
    Conditional edge: after ml_prediction, decide whether to loop RAG
    for more evidence or proceed directly to patient_context.

    Triggers ReAct loop when:
      - needs_react is True  (confidence < 0.6)
      - AND react_iterations < MAX_REACT_ITERATIONS
    """
    needs_react      = state.get("needs_react", False)
    react_iterations = state.get("react_iterations", 0)

    if needs_react and react_iterations < MAX_REACT_ITERATIONS:
        log.info(
            "ReAct loop triggered (iteration %d/%d) — re-running RAG for deeper evidence",
            react_iterations + 1, MAX_REACT_ITERATIONS,
        )
        return "rag_retrieval"
    return "patient_context"


def _increment_react(state: WatchdogState) -> WatchdogState:
    """
    Thin node inserted before looping back to rag_retrieval.
    Increments react_iterations and clears needs_react.
    Without this, the graph would loop infinitely on the same state.
    """
    state["react_iterations"] = state.get("react_iterations", 0) + 1
    state["needs_react"]      = False   # reset so next RAG pass can re-evaluate
    return state


# ─────────────────────────── Memory-write node ───────────────────────────────

def _memory_write_node(state: WatchdogState) -> WatchdogState:
    """
    Final node: persists session + alert to Supabase via memory.py.
    Always runs — failures are caught and logged, pipeline never crashes here.
    """
    try:
        try:
            from .memory import get_memory
        except ImportError:
            from memory import get_memory

        mem        = get_memory()
        patient_ctx = state.get("patient_context")
        patient_id  = getattr(patient_ctx, "patient_id", "unknown") if patient_ctx else "unknown"

        # Build session record
        pair = state.get("drug_pair", ("?", "?"))
        session_data = {
            "drugs_checked":  list(pair),
            "pairs_analysed": 1,
            "input_method":   state.get("input_method", "text"),
            "severity":       getattr(state.get("prediction"), "severity", 0),
            "adjusted_severity": getattr(state.get("prediction"), "adjusted_severity", None),
            "routed_to":      state.get("routing_decision").delivered_to()
                              if state.get("routing_decision") else [],
            "provider_used":  state.get("report_metadata", {}).get("provider_used", ""),
        }
        mem.save_session(patient_id, session_data)
        state["session_saved"] = True
        log.info("Session saved for patient %s", patient_id)

        # Save alert record
        alert_record = state.get("alert_record")
        if alert_record:
            mem.save_alert(patient_id, alert_record)
            state["alert_saved"] = True
            log.info("Alert saved: %s + %s", pair[0], pair[1])

    except Exception as exc:
        log.error("memory_write_node failed: %s", exc)
        state["session_saved"] = False
        state["alert_saved"]   = False

    return state


# ─────────────────────────── Graph builder ───────────────────────────────────

def build_graph(
    use_vision:  bool = True,
    use_memory:  bool = True,
):
    """
    Build and compile the LangGraph state machine.

    Parameters
    ----------
    use_vision : Include vision_intake_agent node (set False for text-only input)
    use_memory : Include memory_write final node (set False for stateless eval runs)

    Returns
    -------
    Compiled LangGraph app (supports .invoke, .stream, .astream)
    OR a LinearFallbackGraph if langgraph is not installed.
    """
    # ── Import all agent node functions ──────────────────────────────────────
    try:
        from .ml_prediction_agent  import get_ml_agent
        from .rag_retrieval_agent  import get_rag_agent
        from .patient_context_agent import get_context_agent
        from .explanation_agent    import get_explanation_agent
        from .alert_routing_agent  import get_routing_agent
    except ImportError:
        from ml_prediction_agent  import get_ml_agent
        from rag_retrieval_agent  import get_rag_agent
        from patient_context_agent import get_context_agent
        from explanation_agent    import get_explanation_agent
        from alert_routing_agent  import get_routing_agent

    ml_agent       = get_ml_agent()
    rag_agent      = get_rag_agent()
    ctx_agent      = get_context_agent()
    explain_agent  = get_explanation_agent()
    routing_agent  = get_routing_agent()

    if not LANGGRAPH_AVAILABLE:
        log.warning("Returning LinearFallbackGraph — install langgraph for full graph features")
        return _LinearFallbackGraph(
            ml_agent, rag_agent, ctx_agent, explain_agent, routing_agent, use_memory
        )

    # ── Build the StateGraph ──────────────────────────────────────────────────
    builder = StateGraph(WatchdogState)

    # ── Add nodes ─────────────────────────────────────────────────────────────

    if use_vision:
        try:
            from .vision_intake_agent import VisionIntakeAgent
        except ImportError:
            from vision_intake_agent import VisionIntakeAgent
        vision_agent = VisionIntakeAgent()
        builder.add_node("vision_intake",   vision_agent.run)

    builder.add_node("ml_prediction",   ml_agent.run)
    builder.add_node("rag_retrieval",   rag_agent.run)
    builder.add_node("react_increment", _increment_react)
    builder.add_node("patient_context", ctx_agent.run)
    builder.add_node("explanation",     explain_agent.run)
    builder.add_node("alert_routing",   routing_agent.run)

    if use_memory:
        builder.add_node("memory_write", _memory_write_node)

    # ── Wire edges ────────────────────────────────────────────────────────────

    # Entry point
    if use_vision:
        builder.set_entry_point("vision_intake")
        builder.add_edge("vision_intake", "ml_prediction")
    else:
        builder.set_entry_point("ml_prediction")

    # ml_prediction → rag_retrieval (always — need evidence before ReAct decision)
    builder.add_edge("ml_prediction", "rag_retrieval")

    # rag_retrieval → conditional: react loop OR patient_context
    builder.add_conditional_edges(
        "rag_retrieval",
        _should_react,
        {
            "rag_retrieval":  "react_increment",   # loop: increment first, then re-run RAG
            "patient_context": "patient_context",
        },
    )

    # react_increment loops back to rag_retrieval
    builder.add_edge("react_increment", "rag_retrieval")

    # Linear: patient_context → explanation → alert_routing
    builder.add_edge("patient_context", "explanation")
    builder.add_edge("explanation",     "alert_routing")

    if use_memory:
        builder.add_edge("alert_routing", "memory_write")
        builder.add_edge("memory_write",  END)
    else:
        builder.add_edge("alert_routing", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    app = builder.compile()
    log.info(
        "LangGraph compiled — nodes: %s  use_vision=%s  use_memory=%s",
        list(builder.nodes.keys()) if hasattr(builder, "nodes") else "?",
        use_vision, use_memory,
    )
    return app


# ─────────────────────────── Linear fallback ─────────────────────────────────

class _LinearFallbackGraph:
    """
    Used when langgraph is not installed.
    Runs all nodes in linear order with no conditional branching.
    Supports the same .invoke() interface as the compiled LangGraph.
    """

    def __init__(self, ml_agent, rag_agent, ctx_agent, explain_agent, routing_agent, use_memory):
        self._nodes = [
            ("ml_prediction",   ml_agent.run),
            ("rag_retrieval",   rag_agent.run),
            ("patient_context", ctx_agent.run),
            ("explanation",     explain_agent.run),
            ("alert_routing",   routing_agent.run),
        ]
        if use_memory:
            self._nodes.append(("memory_write", _memory_write_node))

    def invoke(self, state: dict) -> dict:
        state.setdefault("react_iterations", 0)
        for name, fn in self._nodes:
            log.info("LinearFallbackGraph: running node '%s'", name)
            try:
                state = fn(state)
            except Exception as exc:
                log.error("Node '%s' failed: %s", name, exc)
        return state

    def stream(self, state: dict):
        """Yield (node_name, state) after each node for progress streaming."""
        state.setdefault("react_iterations", 0)
        for name, fn in self._nodes:
            log.info("LinearFallbackGraph: running node '%s'", name)
            try:
                state = fn(state)
            except Exception as exc:
                log.error("Node '%s' failed: %s", name, exc)
            yield name, state


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Building graph (no-vision, no-memory for smoke test)…")
    app = build_graph(use_vision=False, use_memory=False)

    # Minimal state: just a drug pair, no patient context
    initial_state: WatchdogState = {
        "drug_pair":        ("warfarin", "aspirin"),
        "patient_context":  None,
        "input_method":     "text",
        "react_iterations": 0,
    }

    print("Invoking graph…")
    if hasattr(app, "stream"):
        # LangGraph stream mode — print progress
        for step in app.stream(initial_state):
            if isinstance(step, tuple):
                node, state = step
                print(f"  ✓ {node}")
            else:
                print(f"  ✓ step: {type(step).__name__}")
    else:
        result = app.invoke(initial_state)

        print(f"\n── Result ──")
        prediction = result.get("prediction")
        if prediction:
            print(f"Severity   : {prediction.severity_emoji} {prediction.severity_label}")
            print(f"Confidence : {prediction.confidence:.2f}")
            print(f"Mechanism  : {prediction.mechanism}")

        routing = result.get("routing_decision")
        if routing:
            print(f"Urgency    : {routing.urgency_label()}")
            print(f"Routed to  : {routing.delivered_to()}")

        clinician = result.get("clinician_report", "")
        if clinician:
            print(f"\n── Clinician report (first 300 chars) ──")
            print(clinician[:300])

        patient = result.get("patient_report", "")
        if patient:
            print(f"\n── Patient report ──")
            print(patient[:300])
