from pydantic import BaseModel
from typing import Optional
from datetime import date


class ConditionSchema(BaseModel):
    icd10_code: Optional[str] = None
    name: str
    diagnosed_year: Optional[int] = None
    severity: Optional[str] = None


class AllergySchema(BaseModel):
    drug_name: str
    reaction_type: Optional[str] = None
    severity: Optional[str] = None
    onset_year: Optional[int] = None


class MedicationSchema(BaseModel):
    drug_name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[str] = None
    prescriber: Optional[str] = None
    rxcui: Optional[str] = None


class PatientProfileRequest(BaseModel):
    name: str
    date_of_birth: Optional[date] = None
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    ethnicity: Optional[str] = None
    preferred_language: str = "en"
    report_mode: str = "patient"
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    is_dialysis: bool = False
    smoker: bool = False
    alcohol_use: str = "none"
    egfr: Optional[float] = None
    creatinine: Optional[float] = None
    ast: Optional[float] = None
    alt: Optional[float] = None
    inr: Optional[float] = None
    hba1c: Optional[float] = None
    potassium: Optional[float] = None
    hemoglobin: Optional[float] = None
    conditions: list[ConditionSchema] = []
    allergies: list[AllergySchema] = []
    medications: list[MedicationSchema] = []
