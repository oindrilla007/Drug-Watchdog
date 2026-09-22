import logging
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user
from api.schemas.patient import PatientProfileRequest

log = logging.getLogger(__name__)
router = APIRouter()


def _profile_to_dict(profile: PatientProfileRequest, patient_id: str) -> dict:
    d = profile.model_dump(exclude={"medications", "conditions", "allergies"})
    d["patient_id"] = patient_id
    if d.get("date_of_birth"):
        d["date_of_birth"] = str(d["date_of_birth"])
    d["conditions"] = [c.model_dump() for c in (profile.conditions or [])]
    d["allergies"]  = [a.model_dump() for a in (profile.allergies or [])]
    return d


@router.get("/me")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Return the full PatientContext for the authenticated patient."""
    patient_id = current_user["patient_id"]
    try:
        from agents.memory import get_memory
        mem = get_memory()
        ctx = mem.load_patient_context(patient_id)
        return {
            "patient_id":          ctx.patient_id,
            "name":                ctx.name,
            "age":                 ctx.age,
            "sex":                 ctx.sex,
            "weight_kg":           ctx.weight_kg,
            "height_cm":           ctx.height_cm,
            "preferred_language":  ctx.preferred_language,
            "report_mode":         ctx.report_mode,
            "is_pregnant":         ctx.is_pregnant,
            "is_breastfeeding":    ctx.is_breastfeeding,
            "is_dialysis":         ctx.is_dialysis,
            "smoker":              ctx.smoker,
            "alcohol_use":         ctx.alcohol_use,
            "egfr":                ctx.egfr,
            "creatinine":          ctx.creatinine,
            "ast":                 ctx.ast,
            "alt":                 ctx.alt,
            "inr":                 ctx.inr,
            "hba1c":               ctx.hba1c,
            "potassium":           ctx.potassium,
            "hemoglobin":          ctx.hemoglobin,
            "conditions":          ctx.conditions,
            "allergies":           ctx.allergies,
            "medications":         ctx.medications,
            "renal_impaired":      ctx.renal_impaired,
            "hepatic_impaired":    ctx.hepatic_impaired,
            "elderly":             ctx.elderly,
            "high_bleed_risk":     ctx.high_bleed_risk,
            "prior_alerts":        ctx.prior_alerts,
            "sessions_count":      ctx.sessions_count,
            "last_session_at":     ctx.last_session_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.error("Error loading patient %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail="Could not load patient profile")


@router.post("/me")
async def create_profile(
    profile: PatientProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create the initial patient profile after Supabase Auth signup."""
    patient_id = current_user["patient_id"]
    try:
        from agents.memory import get_memory
        mem = get_memory()
        mem.upsert_patient(_profile_to_dict(profile, patient_id))
        if profile.medications:
            mem.save_medications(patient_id, [m.model_dump() for m in profile.medications])
        return {"message": "Profile created", "patient_id": patient_id}
    except Exception as exc:
        log.error("Error creating patient %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/me")
async def update_profile(
    profile: PatientProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update patient demographics, labs, conditions, allergies, and medications."""
    patient_id = current_user["patient_id"]
    try:
        from agents.memory import get_memory
        mem = get_memory()
        mem.upsert_patient(_profile_to_dict(profile, patient_id))
        if profile.medications:
            mem.save_medications(patient_id, [m.model_dump() for m in profile.medications])
        return {"message": "Profile updated", "patient_id": patient_id}
    except Exception as exc:
        log.error("Error updating patient %s: %s", patient_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
