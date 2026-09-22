"""
__init__.py  ·  Drug Watchdog — agents package
================================================
Public API for the Phase 4 multi-agent system.

Import the orchestrator for a full analysis session:

    from agents import Orchestrator, get_orchestrator

    orch   = get_orchestrator()
    result = orch.run(patient_id="uuid", drug_list=["warfarin", "aspirin"])
    print(result.summary())

Or import individual agents if you need finer control:

    from agents import (
        MLPredictionAgent,   get_ml_agent,
        RAGRetrievalAgent,   get_rag_agent,
        PatientContextAgent, get_context_agent,
        ExplanationAgent,    get_explanation_agent,
        AlertRoutingAgent,   get_routing_agent,
        VisionIntakeAgent,
        LLMRouter,           get_router,
        PatientMemory,       get_memory,
        build_graph,
    )

Version
-------
Phase 4 · Drug Watchdog · All models free/open-source.
"""

# ── Orchestrator (primary entry point) ───────────────────────────────────────
from .orchestrator import Orchestrator, SessionResult, get_orchestrator

# ── Graph ─────────────────────────────────────────────────────────────────────
from .graph import build_graph, WatchdogState

# ── Individual agents ─────────────────────────────────────────────────────────
from .ml_prediction_agent   import (
    MLPredictionAgent,
    PredictionResult,
    get_ml_agent,
    normalise_drug_name,
    SEVERITY_LABEL,
    SEVERITY_EMOJI,
)
from .rag_retrieval_agent   import (
    RAGRetrievalAgent,
    RetrievalResult,
    Citation,
    get_rag_agent,
)
from .patient_context_agent import (
    PatientContextAgent,
    AdjustmentRecord,
    get_context_agent,
    # Drug class sets (useful for external rule customisation)
    RENAL_CLEARED,
    HEPATIC_METABOLISED,
    ANTICOAGULANTS,
    NSAIDS,
    PREGNANCY_CONTRAINDICATED,
)
from .explanation_agent     import (
    ExplanationAgent,
    DualReport,
    get_explanation_agent,
)
from .alert_routing_agent   import (
    AlertRoutingAgent,
    RoutingDecision,
    Destination,
    Urgency,
    get_routing_agent,
)

# ── Vision ────────────────────────────────────────────────────────────────────
from .vision_intake_agent   import (
    VisionIntakeAgent,
    IntakeResult,
    IntakeMode,
    PillIdentification,
)

# ── LLM router ────────────────────────────────────────────────────────────────
from .llm_router            import (
    LLMRouter,
    RouterResponse,
    RouterExhaustedError,
    CompletionMode,
    get_router,
)

# ── Memory ────────────────────────────────────────────────────────────────────
from .memory                import (
    PatientMemory,
    PatientContext,
    get_memory,
)


# ─────────────────────────── Package metadata ────────────────────────────────

__version__   = "4.0.0"
__phase__     = "Phase 4 — Multi-Agent Orchestration"
__authors__   = ["Drug Watchdog Team"]

__all__ = [
    # Orchestrator
    "Orchestrator", "SessionResult", "get_orchestrator",

    # Graph
    "build_graph", "WatchdogState",

    # ML Prediction
    "MLPredictionAgent", "PredictionResult", "get_ml_agent",
    "normalise_drug_name", "SEVERITY_LABEL", "SEVERITY_EMOJI",

    # RAG Retrieval
    "RAGRetrievalAgent", "RetrievalResult", "Citation", "get_rag_agent",

    # Patient Context
    "PatientContextAgent", "AdjustmentRecord", "get_context_agent",
    "RENAL_CLEARED", "HEPATIC_METABOLISED", "ANTICOAGULANTS",
    "NSAIDS", "PREGNANCY_CONTRAINDICATED",

    # Explanation
    "ExplanationAgent", "DualReport", "get_explanation_agent",

    # Alert Routing
    "AlertRoutingAgent", "RoutingDecision", "Destination", "Urgency",
    "get_routing_agent",

    # Vision
    "VisionIntakeAgent", "IntakeResult", "IntakeMode", "PillIdentification",

    # LLM Router
    "LLMRouter", "RouterResponse", "RouterExhaustedError",
    "CompletionMode", "get_router",

    # Memory
    "PatientMemory", "PatientContext", "get_memory",
]


# ─────────────────────────── Quick-start helper ───────────────────────────────

def analyse(
    patient_id:  str,
    drugs:       list[str],
    image_b64:   str | None = None,
    image_mode:  str        = "auto",
) -> SessionResult:
    """
    One-liner entry point for a complete analysis session.

    Examples
    --------
    # Text input
    result = analyse("patient-uuid", ["warfarin", "aspirin", "lisinopril"])

    # Image input (prescription photo)
    with open("rx.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    result = analyse("patient-uuid", [], image_b64=b64, image_mode="prescription")

    print(result.summary())
    """
    orch = get_orchestrator()
    return orch.run(
        patient_id   = patient_id,
        drug_list    = drugs if drugs else None,
        image_b64    = image_b64,
        image_mode   = image_mode,
        input_method = "image" if image_b64 else "text",
    )
