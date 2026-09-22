import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas.drug import AnalyseRequest, AnalyseResponse, AlertResult

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse(
    req: AnalyseRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Run a full drug interaction analysis session.

    Pass `drug_list` for a text-based check.
    Pass `new_medications` to trigger a proactive sweep of new drugs
    against the patient's stored medication list.

    Returns all drug pair alerts sorted by severity (highest first).
    """
    patient_id = current_user["patient_id"]
    try:
        from agents.orchestrator import get_orchestrator

        orch = get_orchestrator()
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(
            None,
            lambda: orch.run(
                patient_id      = patient_id,
                drug_list       = req.drug_list,
                new_medications = req.new_medications,
                input_method    = "text",
            ),
        )

        alerts = [
            AlertResult(
                drug_a            = a.get("drug_a", ""),
                drug_b            = a.get("drug_b", ""),
                severity          = a.get("severity", 0),
                final_severity    = a.get("final_severity", 0),
                severity_label    = a.get("severity_label", ""),
                severity_emoji    = a.get("severity_emoji", ""),
                confidence        = a.get("confidence", 0.0),
                mechanism         = a.get("mechanism", ""),
                adjustment_reason = a.get("adjustment_reason", ""),
                clinician_report  = a.get("clinician_report", ""),
                patient_report    = a.get("patient_report", ""),
                urgency           = a.get("urgency", ""),
                routed_to         = a.get("routed_to", []),
                citations         = a.get("citations", []),
                react_iterations  = a.get("react_iterations", 0),
                error             = a.get("error", ""),
            )
            for a in session.alerts
        ]

        return AnalyseResponse(
            patient_id       = session.patient_id,
            patient_name     = session.patient_name,
            input_method     = session.input_method,
            drugs_analysed   = session.drugs_analysed,
            pairs_analysed   = session.pairs_analysed,
            alerts           = alerts,
            total_latency_ms = session.total_latency_ms,
            error            = session.error,
        )

    except Exception as exc:
        log.error("Analysis failed for patient %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
