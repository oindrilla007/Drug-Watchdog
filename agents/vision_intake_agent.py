"""
vision_intake_agent.py  ·  Drug Watchdog Phase 4
==================================================
Multimodal intake agent — the differentiating feature of Drug Watchdog.

Three intake modes, all powered by Groq Llama 4 Scout (17B, free tier):

  Mode 1 — PRESCRIPTION SCAN
  Mode 2 — PILL PHOTO (reads packaging text + imprints; NIH lookup optional)
  Mode 3 — LAB REPORT SCAN (generalized — handles any lab report type)
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import requests

try:
    from .llm_router import LLMRouter, get_router
    from .env_loader import PROJECT_ROOT
except ImportError:
    from llm_router import LLMRouter, get_router
    from env_loader import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ──────────────────────────────────────────

NIH_PILLBOX_API  = "https://rximage.nlm.nih.gov/api/rximage/1/rxnav"
REQUEST_TIMEOUT  = 8
NIH_TIMEOUT      = 5    # shorter — NIH is optional, don't block on it


# ─────────────────────────── Enums / data classes ────────────────────────────

class IntakeMode(str, Enum):
    PRESCRIPTION = "prescription"
    PILL_PHOTO   = "pill_photo"
    LAB_REPORT   = "lab_report"
    AUTO         = "auto"


@dataclass
class PillIdentification:
    shape:           str | None  = None
    color:           str | None  = None
    imprint:         str | None  = None
    brand_name:      str | None  = None   # NEW — read from blister pack text
    size_mm:         str | None  = None
    identified_drug: str | None  = None
    dose:            str | None  = None
    confidence:      float       = 0.0
    nih_confirmed:   bool        = False
    nih_name:        str | None  = None
    nih_rxcui:       str | None  = None


@dataclass
class IntakeResult:
    mode:             IntakeMode
    success:          bool
    confidence:       float              = 0.0
    provider_used:    str                = ""
    latency_ms:       float              = 0.0
    error:            str                = ""

    # Prescription / pill fields
    drugs:            list[str]          = field(default_factory=list)
    medications:      list[dict]         = field(default_factory=list)
    prescriber:       str | None         = None
    prescription_date: str | None        = None
    patient_name_on_rx: str | None       = None
    pharmacy_notes:   str | None         = None

    # Pill fields
    pills:            list[PillIdentification] = field(default_factory=list)
    pill_warnings:    list[str]          = field(default_factory=list)

    # Lab report fields — now generalized
    lab_values:       dict[str, Any]     = field(default_factory=dict)
    # Standard keys where present: egfr, creatinine, ast, alt, inr, hba1c, potassium, hemoglobin
    # Plus any other test results found

    drugs_listed_on_report: list[str]    = field(default_factory=list)
    report_date:      str | None         = None
    lab_name:         str | None         = None
    raw_lab_lines:    list[str]          = field(default_factory=list)
    report_type:      str | None         = None   # NEW — e.g. "Widal", "CBC", "LFT"
    patient_info:     dict               = field(default_factory=dict)  # NEW — name, age, UHID

    detected_mode:    str | None         = None

    def to_patient_profile_patch(self) -> dict:
        """
        Returns a dict for patching PatientContext / Supabase patients table.
        Only includes standard lab fields that have actual values.
        """
        patch = {}
        lv = self.lab_values
        for key in ["egfr", "creatinine", "ast", "alt", "inr", "hba1c", "potassium", "hemoglobin"]:
            val = lv.get(key)
            if val is not None:
                patch[key] = val
        return patch

    def to_medication_list(self) -> list[dict]:
        if self.medications:
            return self.medications
        result = []
        for d in self.drugs:
            parts = str(d).strip().split()
            name = parts[0] if parts else d
            dose = parts[1] if len(parts) > 1 else None
            freq = " ".join(parts[2:]) if len(parts) > 2 else None
            result.append({"drug_name": name, "dose": dose, "frequency": freq})
        return result


# ─────────────────────────── Prompts ─────────────────────────────────────────

PRESCRIPTION_PROMPT = """
You are extracting medication information from a prescription image.
This may be handwritten or printed. Read carefully.

Return ONLY valid JSON with this exact schema — no markdown, no extra text:
{
  "drugs": ["drug name + dose + frequency as a single string"],
  "medications": [
    {
      "drug_name": "generic name lowercase",
      "dose": "e.g. 5mg or 1 tab",
      "frequency": "e.g. once daily, morning and night",
      "prescriber": "doctor name or null",
      "start_date": null
    }
  ],
  "prescriber": "string or null",
  "prescription_date": "YYYY-MM-DD or null",
  "patient_name_on_rx": "string or null",
  "pharmacy_notes": "string or null",
  "confidence": 0.85
}

Rules:
- Never use the string "null" — use JSON null (no quotes)
- Use generic drug names (e.g. "lorazepam" not "Ativan")
- confidence 0.0-1.0 based on legibility
- Include every medication visible, even if dose is unclear
"""

PILL_PHOTO_PROMPT = """
You are a pharmaceutical identification expert analyzing a photo of medications.

STEP 1 — READ ALL TEXT FIRST (most reliable method):
Scan the entire image for ANY printed, embossed, or stamped text:
- Blister pack foil text (often repeated across the strip, e.g. "DOMPAN DOMPAN DOMPAN")
- Box or carton text visible in the background
- Text on individual pill surfaces (imprint codes like "M357", "L484")
- Sticker labels on bottles or packets
- Manufacturer name, batch numbers (these help identify the drug)

STEP 2 — DESCRIBE PHYSICAL APPEARANCE:
For each visually distinct medication type:
- Shape: round, oval, oblong, capsule, caplet, triangle, diamond, other
- Color: be specific (light pink, pale yellow, white, silver-grey, gold, orange)
- Size: small (<8mm), medium (8-14mm), large (>14mm)
- Surface: scored (has a line), plain, coated, gelatin capsule

STEP 3 — IDENTIFY THE DRUG:
Use ALL available clues in this priority order:
1. Blister pack foil text → look up generic name
2. Imprint code → identify by shape+color+imprint combination  
3. Box/label text visible nearby
4. Physical appearance alone (last resort, low confidence)

Return ONLY valid JSON — no markdown, no extra text:
{
  "packaging_text_found": ["list every piece of text you can read from packaging/foil/labels"],
  "pills": [
    {
      "shape": "round|oval|oblong|capsule|caplet|other",
      "color": "specific color description",
      "imprint": "exact text on pill surface or null",
      "brand_name": "brand name from packaging or null",
      "manufacturer": "manufacturer name if visible or null",
      "size": "small|medium|large",
      "surface": "scored|plain|coated|gelatin",
      "identified_drug": "generic drug name or null",
      "drug_class": "e.g. antiemetic, antibiotic, antihypertensive or null",
      "dose": "dose if visible or null",
      "confidence": 0.0,
      "identification_method": "foil_text|imprint|label|appearance"
    }
  ],
  "warnings": ["any safety concerns — mixed pills, broken seals, unusual appearance"],
  "packaging_text": ["verbatim text from any packaging visible"]
}

Known brand mappings (use these if you see the text):
- DOMPAN, VOMITAB, DOMSTAL → domperidone (antiemetic)
- CROCIN, CALPOL, METACIN → paracetamol / acetaminophen
- COMBIFLAM → ibuprofen + paracetamol
- AUGMENTIN → amoxicillin + clavulanate
- TAXIM, ZIFI → cefixime
- AZITHRAL, AZEE, ZITHROMAX → azithromycin
- PAN, PANTOP, PANTOCID → pantoprazole
- RANTAC, ZINETAC → ranitidine
- MONTAIR, MONTEK → montelukast
- ALLEGRA → fexofenadine
- TELMA, TELMIKIND → telmisartan
- ECOSPRIN → aspirin
- GLYCOMET, GLUCOPHAGE → metformin
- THYRONORM, ELTROXIN → levothyroxine
- ASTHALIN, VENTOLIN → salbutamol
- WYSOLONE, OMNACORTIL → prednisolone
- ATIVAN → lorazepam
- RIVOTRIL → clonazepam
- SIZODON → risperidone
- QUETIN, QUTAN → quetiapine

Rules:
- NEVER use the string "null" — always use JSON null (no quotes)
- If blister pack text is partially visible/repeated, still extract it
- A gold/yellow blister pack with repeated text is almost always the brand name
- Set confidence based on: foil_text=0.85+, imprint=0.80+, label=0.75+, appearance_only=0.30-0.50
- identification_method must reflect HOW you identified it
- If truly unidentifiable, set identified_drug to null but still describe appearance fully
"""

LAB_REPORT_PROMPT = """
You are extracting ALL information from a medical lab report or pathology report image.
This could be any type of test — blood panel, serology, urine, culture, etc.

Return ONLY valid JSON — no markdown, no extra text:
{
  "report_type": "name of the test e.g. Widal Test, CBC, LFT, RFT, Urine Analysis",
  "lab_name": "name of the lab/hospital or null",
  "report_date": "YYYY-MM-DD or null",
  "patient_info": {
    "name": "patient name or null",
    "age": "age or null",
    "sex": "M/F or null",
    "uhid": "patient ID or null"
  },
  "lab_values": {
    "egfr": null,
    "creatinine": null,
    "ast": null,
    "alt": null,
    "inr": null,
    "hba1c": null,
    "potassium": null,
    "hemoglobin": null,
    "other": {
      "test_name": {"result": "value", "reference": "normal range", "unit": "unit", "flag": "normal|high|low|reactive|non-reactive"}
    }
  },
  "drugs_listed_on_report": [],
  "raw_lab_lines": ["verbatim result lines from the report"],
  "interpretation": "overall interpretation or impression if stated",
  "confidence": 0.9
}

Rules:
- Never use the string "null" — use JSON null (no quotes)
- For standard blood tests, fill egfr/creatinine/ast/alt/inr/hba1c/potassium/hemoglobin
- For OTHER test types (Widal, culture, serology, etc.), put ALL results in the "other" dict
- For Widal test: put each agglutination result (Salmonella typhi O, H, etc.) in "other"
- raw_lab_lines: copy key result lines verbatim from the report
- confidence 0.0-1.0
"""

AUTO_DETECT_PROMPT = """
Analyze this medical image and identify what type it is.

Return ONLY valid JSON:
{
  "detected_mode": "prescription|pill_photo|lab_report|other",
  "confidence": 0.9,
  "description": "one sentence describing what you see"
}
"""


# ─────────────────────────── NIH Pillbox helper ──────────────────────────────

class NIHPillboxClient:
    """
    Optional NIH NLM RxImageAPI lookup.
    Skips silently if network is unavailable — never blocks the pipeline.
    """

    def __init__(self):
        self._available: bool | None = None  # None = not yet tested

    def _check_available(self) -> bool:
        """Test connectivity once per session."""
        if self._available is not None:
            return self._available
        try:
            requests.get("https://rximage.nlm.nih.gov", timeout=3)
            self._available = True
        except Exception:
            self._available = False
            log.info("NIH Pillbox API unreachable — skipping imprint lookups.")
        return self._available

    def lookup_by_imprint(self, imprint: str, color: str = "", shape: str = "") -> dict | None:
        # Skip if imprint is None, empty, or the literal string "null"
        if not imprint or imprint.strip().lower() in ("null", "none", ""):
            return None
        if not self._check_available():
            return None

        params: dict[str, str] = {"imprint": imprint.strip(), "resolution": "thumbnail"}
        if color:
            params["color"] = color.lower()
        if shape:
            params["shape"] = shape.lower()

        try:
            resp = requests.get(NIH_PILLBOX_API, params=params, timeout=NIH_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("nlmRxImages", [])
            if results:
                top = results[0]
                return {
                    "name":     top.get("name", ""),
                    "rxcui":    top.get("rxcui", ""),
                    "ndc11":    top.get("ndc11", ""),
                    "imageUrl": top.get("imageUrl", ""),
                }
        except Exception as exc:
            log.debug("NIH Pillbox lookup failed for '%s': %s", imprint, exc)
        return None


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _clean_null(value: Any) -> Any:
    """
    Convert the string "null" or "None" (model mistake) to actual None.
    Recursively cleans dicts and lists.
    """
    if isinstance(value, str) and value.strip().lower() in ("null", "none", "n/a", ""):
        return None
    if isinstance(value, dict):
        return {k: _clean_null(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_null(v) for v in value]
    return value


# ─────────────────────────── Vision Intake Agent ─────────────────────────────

class VisionIntakeAgent:
    """
    Processes photos of prescriptions, pills, and lab reports.
    Uses Groq Llama 4 Scout. Returns a structured IntakeResult.
    """

    def __init__(self, router: LLMRouter | None = None):
        self._router  = router or get_router()
        self._pillbox = NIHPillboxClient()

    # ── Public interface ──────────────────────────────────────────────────────

    def intake_from_file(self, path: str | Path, mode: str = "auto") -> IntakeResult:
        requested_path = Path(path).expanduser()
        resolved_path = self._resolve_image_path(requested_path)
        if resolved_path is None:
            checked = self._format_checked_paths(requested_path)
            return IntakeResult(mode=IntakeMode(mode), success=False,
                                error=f"File not found: {requested_path} (checked: {checked})")
        with open(resolved_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return self.intake_from_b64(b64, mode=mode)

    @staticmethod
    def _image_path_candidates(path: Path) -> list[Path]:
        if path.is_absolute():
            return [path]
        return [
            Path.cwd() / path,
            PROJECT_ROOT / path,
            Path(__file__).resolve().parent / path,
        ]

    @classmethod
    def _resolve_image_path(cls, path: str | Path) -> Path | None:
        for candidate in cls._image_path_candidates(Path(path).expanduser()):
            if candidate.exists():
                return candidate.resolve()
        return None

    @classmethod
    def _format_checked_paths(cls, path: Path) -> str:
        return ", ".join(str(p) for p in cls._image_path_candidates(path))

    def intake_from_b64(self, image_b64: str, mode: str = "auto") -> IntakeResult:
        intake_mode = IntakeMode(mode)
        t0 = time.perf_counter()

        if intake_mode == IntakeMode.AUTO:
            detected = self._detect_image_type(image_b64)
            intake_mode = IntakeMode(detected) if detected and detected != "other" \
                          else IntakeMode.PRESCRIPTION

        try:
            if intake_mode == IntakeMode.PRESCRIPTION:
                result = self._process_prescription(image_b64)
            elif intake_mode == IntakeMode.PILL_PHOTO:
                result = self._process_pill_photo(image_b64)
            elif intake_mode == IntakeMode.LAB_REPORT:
                result = self._process_lab_report(image_b64)
            else:
                result = self._process_prescription(image_b64)

            result.latency_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "Vision intake (%s): %.0f ms  confidence=%.2f  drugs=%s",
                result.mode.value, result.latency_ms, result.confidence,
                result.drugs[:3] if result.drugs else (result.lab_values or {}),
            )
            return result

        except Exception as exc:
            log.error("Vision intake failed: %s", exc, exc_info=True)
            return IntakeResult(
                mode=intake_mode, success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    def intake_auto(self, image_b64: str) -> IntakeResult:
        return self.intake_from_b64(image_b64, mode="auto")

    # ── Auto-detect ───────────────────────────────────────────────────────────

    def _detect_image_type(self, image_b64: str) -> str | None:
        try:
            resp = self._router.vision_complete(
                prompt=AUTO_DETECT_PROMPT, image_b64=image_b64, max_tokens=200)
            return self._parse_json(resp.text).get("detected_mode")
        except Exception as exc:
            log.warning("Auto-detect failed: %s", exc)
            return None

    # ── Prescription ──────────────────────────────────────────────────────────

    def _process_prescription(self, image_b64: str) -> IntakeResult:
        resp = self._router.vision_complete(
            prompt=PRESCRIPTION_PROMPT, image_b64=image_b64, max_tokens=1000)
        data = _clean_null(self._parse_json(resp.text))

        medications = data.get("medications") or []
        drugs_raw   = data.get("drugs") or []

        # Filter out any None/null drug names
        medications = [m for m in medications if m.get("drug_name")]
        drugs_raw   = [d for d in drugs_raw if d]

        if not medications and drugs_raw:
            for d in drugs_raw:
                parts = str(d).strip().split()
                medications.append({
                    "drug_name":  parts[0],
                    "dose":       parts[1] if len(parts) > 1 else None,
                    "frequency":  " ".join(parts[2:]) if len(parts) > 2 else None,
                    "prescriber": data.get("prescriber"),
                    "start_date": data.get("prescription_date"),
                })

        return IntakeResult(
            mode               = IntakeMode.PRESCRIPTION,
            success            = True,
            confidence         = float(data.get("confidence") or 0.7),
            provider_used      = resp.provider_used,
            drugs              = drugs_raw or [m["drug_name"] for m in medications],
            medications        = medications,
            prescriber         = data.get("prescriber"),
            prescription_date  = data.get("prescription_date"),
            patient_name_on_rx = data.get("patient_name_on_rx"),
            pharmacy_notes     = data.get("pharmacy_notes"),
        )

    # ── Pill photo ────────────────────────────────────────────────────────────

    def _process_pill_photo(self, image_b64: str) -> IntakeResult:
        resp = self._router.vision_complete(
            prompt=PILL_PHOTO_PROMPT, image_b64=image_b64, max_tokens=1000)
        data      = _clean_null(self._parse_json(resp.text))
        pills_raw = data.get("pills") or []

        pills: list[PillIdentification] = []
        for p in pills_raw:
            # Skip entirely null/empty entries
            if not any([p.get("color"), p.get("shape"), p.get("identified_drug"),
                        p.get("brand_name"), p.get("imprint")]):
                continue

            pill = PillIdentification(
                shape           = p.get("shape"),
                color           = p.get("color"),
                imprint         = p.get("imprint"),        # already cleaned to None if "null"
                brand_name      = p.get("brand_name"),
                size_mm         = p.get("size_mm"),
                identified_drug = p.get("identified_drug"),
                dose            = p.get("dose"),
                confidence      = float(p.get("confidence") or 0.5),
            )

            # If brand name found but no identified_drug, try brand→generic mapping
            if pill.brand_name and not pill.identified_drug:
                pill.identified_drug = self._brand_to_generic(pill.brand_name)
                if pill.identified_drug:
                    pill.confidence = max(pill.confidence, 0.75)

            # NIH lookup only if we have a real imprint
            if pill.imprint:
                nih = self._pillbox.lookup_by_imprint(
                    pill.imprint, pill.color or "", pill.shape or "")
                if nih and nih.get("name"):
                    pill.nih_confirmed  = True
                    pill.nih_name       = nih["name"]
                    pill.nih_rxcui      = nih.get("rxcui")
                    pill.identified_drug = nih["name"]
                    pill.confidence     = min(1.0, pill.confidence + 0.2)

            pills.append(pill)

        identified_drugs = [p.identified_drug for p in pills if p.identified_drug]

        return IntakeResult(
            mode          = IntakeMode.PILL_PHOTO,
            success       = True,
            confidence    = (sum(p.confidence for p in pills) / len(pills)) if pills else 0.0,
            provider_used = resp.provider_used,
            drugs         = identified_drugs,
            medications   = [
                {"drug_name": p.identified_drug,
                 "dose": p.dose,
                 "brand_name": p.brand_name,
                 "frequency": None}
                for p in pills if p.identified_drug
            ],
            pills         = pills,
            pill_warnings = data.get("warnings") or [],
        )

    # ── Lab report ────────────────────────────────────────────────────────────

    def _process_lab_report(self, image_b64: str) -> IntakeResult:
        resp = self._router.vision_complete(
            prompt=LAB_REPORT_PROMPT, image_b64=image_b64, max_tokens=1200)
        data  = _clean_null(self._parse_json(resp.text))
        raw_lv = data.get("lab_values") or {}

        # Standard fields — float or None
        lab_values: dict[str, Any] = {}
        for key in ["egfr", "creatinine", "ast", "alt", "inr", "hba1c", "potassium", "hemoglobin"]:
            val = raw_lv.get(key)
            if val is not None:
                try:
                    lab_values[key] = float(val)
                except (TypeError, ValueError):
                    lab_values[key] = None
            else:
                lab_values[key] = None

        # Preserve ALL other results (Widal, culture, serology, etc.)
        lab_values["other"] = raw_lv.get("other") or {}

        drugs_on_report = [d for d in (data.get("drugs_listed_on_report") or []) if d]

        return IntakeResult(
            mode                    = IntakeMode.LAB_REPORT,
            success                 = True,
            confidence              = float(data.get("confidence") or 0.8),
            provider_used           = resp.provider_used,
            lab_values              = lab_values,
            drugs_listed_on_report  = drugs_on_report,
            drugs                   = drugs_on_report,
            report_date             = data.get("report_date"),
            lab_name                = data.get("lab_name"),
            raw_lab_lines           = data.get("raw_lab_lines") or [],
            report_type             = data.get("report_type"),
            patient_info            = data.get("patient_info") or {},
        )

    # ── Brand → generic mapping ───────────────────────────────────────────────

    BRAND_GENERIC: dict[str, str] = {
        # From your pill image
        "dompan":        "domperidone",
        "nulife":        "domperidone",   # NuLife visible on box in image
        # Common others
        "crocin":        "paracetamol",
        "combiflam":     "ibuprofen + paracetamol",
        "augmentin":     "amoxicillin-clavulanate",
        "taxim":         "cefixime",
        "montair":       "montelukast",
        "telma":         "telmisartan",
        "ecosprin":      "aspirin",
        "pan":           "pantoprazole",
        "rantac":        "ranitidine",
        "allegra":       "fexofenadine",
        "cetrizine":     "cetirizine",
        "wysolone":      "prednisolone",
        "practin":       "cyproheptadine",
        "asthalin":      "salbutamol",
    }

    def _brand_to_generic(self, brand: str) -> str | None:
        if not brand:
            return None
        return self.BRAND_GENERIC.get(brand.lower().strip())

    # ── JSON parser ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("Could not parse JSON from vision response:\n%s", text[:300])
            return {}


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision_intake_agent.py <image_path> [prescription|pill_photo|lab_report|auto]")
        sys.exit(1)

    image_path = sys.argv[1]
    mode       = sys.argv[2] if len(sys.argv) > 2 else "auto"

    agent  = VisionIntakeAgent()
    result = agent.intake_from_file(image_path, mode=mode)

    print(f"\n── Vision Intake Result ──────────────────────────────")
    print(f"Mode       : {result.mode.value}")
    print(f"Success    : {result.success}")
    print(f"Confidence : {result.confidence:.2f}")
    print(f"Provider   : {result.provider_used}")
    print(f"Latency    : {result.latency_ms:.0f} ms")

    if result.error:
        print(f"Error      : {result.error}")

    if result.report_type:
        print(f"Report type: {result.report_type}")

    if result.patient_info:
        print(f"Patient    : {result.patient_info}")

    if result.drugs:
        print(f"\nDrugs identified ({len(result.drugs)}):")
        for d in result.drugs:
            print(f"  • {d}")

    if result.medications:
        print(f"\nStructured medications:")
        for m in result.medications:
            print(f"  • {m}")

    if result.pills:
        print(f"\nPill identifications ({len(result.pills)}):")
        for p in result.pills:
            brand = f" [{p.brand_name}]" if p.brand_name else ""
            nih   = f"  NIH=✓ {p.nih_name}" if p.nih_confirmed else "  NIH=✗"
            print(f"  • {p.identified_drug or '?'}{brand}  "
                  f"color={p.color}  imprint={p.imprint}{nih}  conf={p.confidence:.2f}")
        if result.pill_warnings:
            print(f"\nWarnings:")
            for w in result.pill_warnings:
                print(f"  ⚠ {w}")

    if result.lab_values:
        print(f"\nStandard lab values:")
        for k, v in result.lab_values.items():
            if k != "other" and v is not None:
                print(f"  {k:12s}: {v}")
        if result.lab_values.get("other"):
            print(f"\nOther test results:")
            for test, vals in result.lab_values["other"].items():
                if isinstance(vals, dict):
                    flag = f"  [{vals.get('flag','').upper()}]" if vals.get('flag') else ""
                    print(f"  {test:35s}: {vals.get('result','?')} "
                          f"{vals.get('unit','')}  (ref: {vals.get('reference','?')}){flag}")
                else:
                    print(f"  {test}: {vals}")

    if result.raw_lab_lines:
        print(f"\nRaw lab lines:")
        for line in result.raw_lab_lines:
            print(f"  {line}")

    if result.drugs_listed_on_report:
        print(f"\nDrugs on report: {result.drugs_listed_on_report}")

    patch = result.to_patient_profile_patch()
    print(f"\nPatient profile patch: {patch if patch else '(no standard lab values — see other results above)'}")
