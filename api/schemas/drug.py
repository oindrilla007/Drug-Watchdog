from pydantic import BaseModel
from typing import Optional


class AnalyseRequest(BaseModel):
    drug_list: list[str]
    new_medications: Optional[list[str]] = None


class IntakeImageRequest(BaseModel):
    image_b64: str
    mode: str = "auto"


class AlertResult(BaseModel):
    drug_a: str
    drug_b: str
    severity: int = 0
    final_severity: int = 0
    severity_label: str = ""
    severity_emoji: str = ""
    confidence: float = 0.0
    mechanism: str = ""
    adjustment_reason: str = ""
    clinician_report: str = ""
    patient_report: str = ""
    urgency: str = ""
    routed_to: list[str] = []
    citations: list[str] = []
    react_iterations: int = 0
    error: str = ""


class AnalyseResponse(BaseModel):
    patient_id: str
    patient_name: str
    input_method: str
    drugs_analysed: list[str]
    pairs_analysed: int
    alerts: list[AlertResult]
    total_latency_ms: float
    error: str = ""


class IntakeResponse(BaseModel):
    mode: str
    success: bool
    confidence: float = 0.0
    drugs: list[str] = []
    medications: list[dict] = []
    lab_values: Optional[dict] = None
    prescriber: Optional[str] = None
    prescription_date: Optional[str] = None
    patient_name_on_rx: Optional[str] = None
    report_type: Optional[str] = None
    patient_info: dict = {}
    error: str = ""
