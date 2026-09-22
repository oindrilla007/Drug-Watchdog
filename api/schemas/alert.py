from pydantic import BaseModel
from typing import Optional


class AlertResponse(BaseModel):
    id: str
    patient_id: str
    drug_a: str
    drug_b: str
    severity: int
    severity_label: Optional[str] = None
    adjusted_severity: Optional[int] = None
    adjustment_reason: Optional[str] = None
    clinician_report: Optional[str] = None
    patient_report: Optional[str] = None
    citations: list = []
    routed_to: list = []
    was_acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    created_at: Optional[str] = None


class AcknowledgeRequest(BaseModel):
    alert_id: str
