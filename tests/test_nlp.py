import pytest
from claims.nlp.entity_extractor import MedicalEntityExtractor
from claims.nlp.note_processor import ClinicalNoteProcessor
from claims.nlp.icd_classifier import ICDClassifier

def test_entity_extractor_finds_medication():
    extractor = MedicalEntityExtractor()
    note = "Patient prescribed Metformin 500mg for hyperglycemia."
    result = extractor.extract(note)
    assert "metformin" in result.medications

def test_entity_extractor_finds_blood_pressure():
    extractor = MedicalEntityExtractor()
    note = "Patient vitals: BP 130/85 mmHg, temp 98.6."
    result = extractor.extract(note)
    assert "blood_pressure" in result.vitals
    assert result.vitals["blood_pressure"]["systolic"] == 130
    assert result.vitals["blood_pressure"]["diastolic"] == 85

def test_entity_extractor_finds_condition():
    extractor = MedicalEntityExtractor()
    note = "Subject has known Type 2 Diabetes and asthma."
    result = extractor.extract(note)
    assert "diabetes" in result.conditions
    assert "asthma" in result.conditions

def test_note_processor_clean_expands_abbreviations():
    processor = ClinicalNoteProcessor()
    note = "BP is elevated, BMI recorded at 32."
    cleaned = processor.clean(note)
    assert "blood pressure" in cleaned.lower()
    assert "body mass index" in cleaned.lower()

def test_note_processor_urgency_urgent():
    processor = ClinicalNoteProcessor()
    note = "Patient presenting with acute emergency, critical condition chest pains."
    urgency = processor.get_urgency(note)
    assert urgency.lower() == "urgent"

def test_note_processor_urgency_routine():
    processor = ClinicalNoteProcessor()
    note = "Annual wellness review, normal vitals, routine checkup."
    urgency = processor.get_urgency(note)
    assert urgency.lower() == "routine"

def test_icd_classifier_returns_valid_code():
    classifier = ICDClassifier()
    # Simple training notes/labels
    notes = [
        "Patient has diabetes and needs insulin.",
        "Experiencing breathlessness and severe COPD symptoms.",
        "Urgent ECG required for chest pain.",
        "Routine physical checkup with normal vitals."
    ] * 5
    labels = ["E11", "J44", "I21", "Z00"] * 5
    
    classifier.train(notes, labels)
    
    # Predict on test string
    test_note = "Subject taking insulin for diabetes control."
    result = classifier.predict(test_note)
    
    assert result.icd_code in labels
    assert len(result.alternative_codes) >= 1
