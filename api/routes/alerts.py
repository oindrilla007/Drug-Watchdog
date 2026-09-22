import logging
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def get_alerts(current_user: dict = Depends(get_current_user)):
    """Return all unacknowledged alerts for the authenticated patient."""
    patient_id = current_user["patient_id"]
    try:
        from agents.memory import get_memory
        mem = get_memory()
        alerts = mem.get_unacknowledged_alerts(patient_id)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as exc:
        log.error("Error loading alerts for %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/all")
async def get_all_alerts(current_user: dict = Depends(get_current_user)):
    """Return last 50 alerts (acknowledged + unacknowledged) for history view."""
    patient_id = current_user["patient_id"]
    try:
        import os
        import requests as req_lib

        supabase_url = os.getenv("SUPABASE_URL", "")
        service_key  = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not supabase_url or not service_key:
            return {"alerts": [], "count": 0}

        headers = {
            "apikey":        service_key,
            "Authorization": f"Bearer {service_key}",
        }
        url = (
            f"{supabase_url}/rest/v1/patient_alerts"
            f"?patient_id=eq.{patient_id}"
            "&order=created_at.desc&limit=50"
        )
        resp = req_lib.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        alerts = resp.json()
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as exc:
        log.error("Error loading all alerts for %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Mark a specific alert as acknowledged."""
    try:
        from agents.memory import get_memory
        mem = get_memory()
        mem.acknowledge_alert(alert_id)
        return {"message": "Alert acknowledged", "alert_id": alert_id}
    except Exception as exc:
        log.error("Error acknowledging alert %s: %s", alert_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
