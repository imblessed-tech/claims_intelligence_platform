import re
import spacy
import logging
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EntityResult:
    """What the API returns for a medical entity extraction."""
    medications: list[str]
    conditions: list[str]
    procedures: list[str]
    vitals: dict
    clinical_summary: str

# Regex patterns 
MEDICATION_PATTERN = r'\b(?:metformin|amlodipine|atorvastatin|aspirin|ramipril|bisoprolol|insulin|omeprazole|paracetamol|amoxicillin|salbutamol|tiotropium|sertraline|nitrofurantoin|empagliflozin|rosuvastatin|losartan|prednisolone)\b'
CONDITION_PATTERN = r'\b(?:diabetes|hypertension|COPD|obesity|asthma|hyperlipidemia|pneumonia|malaria|typhoid|anaemia|depression|anxiety|ischaemic)\b'
PROCEDURE_PATTERN = r'\b(?:ECG|X-ray|MRI|CT scan|ultrasound|spirometry|echocardiogram|blood test|urinalysis|colonoscopy)\b'

# VITALS_PATTERNS
BP_PATTERN = r'(\d{2,3})[/\\](\d{2,3})\s*mmHg'
BMI_PATTERN = r'BMI\s*(?:of\s*)?(?:is\s*)?(\d{2,3}(?:\.\d)?)'
HBA1C_PATTERN = r'HbA1c\s*(?:of\s*)?(\d{1,2}(?:\.\d)?)\s*%'
HR_PATTERN = r'HR\s*(\d{2,3})\s*bpm'

class MedicalEntityExtractor:
    """Extracts medical entities and codes from clinical notes."""

    def __init__(self) -> None:
        # Compile all regex patterns once using re.compile(pattern, re.IGNORECASE)
        self.med_regex = re.compile(MEDICATION_PATTERN, re.IGNORECASE)
        self.cond_regex = re.compile(CONDITION_PATTERN, re.IGNORECASE)
        self.proc_regex = re.compile(PROCEDURE_PATTERN, re.IGNORECASE)
        self.bp_regex = re.compile(BP_PATTERN, re.IGNORECASE)
        self.bmi_regex = re.compile(BMI_PATTERN, re.IGNORECASE)
        self.hba1c_regex = re.compile(HBA1C_PATTERN, re.IGNORECASE)
        self.hr_regex = re.compile(HR_PATTERN, re.IGNORECASE)

        # Standard casings mapping for normalization
        self.medication_map = {m.lower(): m for m in [
            "metformin", "amlodipine", "atorvastatin", "aspirin", "ramipril",
            "bisoprolol", "insulin", "omeprazole", "paracetamol", "amoxicillin",
            "salbutamol", "tiotropium", "sertraline", "nitrofurantoin",
            "empagliflozin", "rosuvastatin", "losartan", "prednisolone"
        ]}
        
        self.condition_map = {c.lower(): c for c in [
            "diabetes", "hypertension", "COPD", "obesity", "asthma",
            "hyperlipidemia", "pneumonia", "malaria", "typhoid",
            "anaemia", "depression", "anxiety", "ischaemic"
        ]}
        self.condition_map["ischemic"] = "ischaemic"

        self.procedure_map = {p.lower(): p for p in [
            "ECG", "X-ray", "MRI", "CT scan", "ultrasound", "spirometry", "echocardiogram",
            "blood test", "urinalysis", "colonoscopy"
        ]}

        # Load spaCy model
        self.nlp: Optional[Any] = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(
                f"Could not load spaCy model 'en_core_web_sm': {e}. "
                "Falling back to regex-only extraction."
            )
            self.nlp = None

    def extract(self, note: str) -> EntityResult:
        """Extract medical entities from a clinical note."""
        # Lowercase a copy for matching
        note_lower = note.lower()

        # Extract medications: use re.findall, deduplicate with list(set(...))
        meds_found = re.findall(self.med_regex, note_lower)
        medications = list(set(self.medication_map.get(m, m) for m in meds_found))

        # Extract conditions
        conds_found = re.findall(self.cond_regex, note_lower)
        conditions = list(set(self.condition_map.get(c, c) for c in conds_found))

        # Extract procedures
        procs_found = re.findall(self.proc_regex, note_lower)
        procedures = list(set(self.procedure_map.get(p, p) for p in procs_found))

        # Extract vitals: parse captured groups into a dict
        bp_match = self.bp_regex.search(note_lower)
        if bp_match:
            bp = {
                "systolic": int(bp_match.group(1)),
                "diastolic": int(bp_match.group(2))
            }
        else:
            bp = None

        bmi_match = self.bmi_regex.search(note_lower)
        bmi = float(bmi_match.group(1)) if bmi_match else None

        hba1c_match = self.hba1c_regex.search(note_lower)
        hba1c = float(hba1c_match.group(1)) if hba1c_match else None

        hr_match = self.hr_regex.search(note_lower)
        hr = int(hr_match.group(1)) if hr_match else None

        vitals = {
            "blood_pressure": bp,
            "bmi": bmi,
            "hba1c": hba1c,
            "heart_rate": hr
        }

        # If self.nlp is available, catch additional named entities
        if self.nlp:
            doc = self.nlp(note)
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent.label_ == "PRODUCT":
                    if ent_text.lower() not in [m.lower() for m in medications]:
                        medications.append(ent_text)
                elif ent.label_ in ["ORG", "GPE"]:
                    if ent.label_ == "ORG" and ent_text.lower() not in [p.lower() for p in procedures]:
                        procedures.append(ent_text)

        # Build clinical summary
        clinical_summary = self._build_summary(medications, conditions, vitals)

        return EntityResult(
            medications=medications,
            conditions=conditions,
            procedures=procedures,
            vitals=vitals,
            clinical_summary=clinical_summary
        )

    def _build_summary(self, medications: list[str], conditions: list[str], vitals: dict) -> str:
        """Build a readable sentence summary of the clinical note."""
        parts = []

        if conditions:
            conds_str = ", ".join(conditions)
            parts.append(f"Patient with {conds_str}")
        else:
            parts.append("Patient")

        if medications:
            meds_str = ", ".join(medications)
            parts.append(f"prescribed {meds_str}")

        bp = vitals.get("blood_pressure")
        if bp:
            parts.append(f"Vitals: BP {bp['systolic']}/{bp['diastolic']} mmHg")

        summary = " ".join(parts).strip()
        if not summary.endswith("."):
            summary += "."
        return summary

