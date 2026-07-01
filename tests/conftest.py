import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from claims.api.dependencies import ModelRegistry
from claims.data.preprocessor import ClaimsPreprocessor

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Create a small mock DataFrame with 20 rows matching the CSV schema."""
    np.random.seed(42)
    data = {
        "age": [19, 62, 48, 53, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30, 60],
        "sex": ["female", "female", "male", "female", "female", "female", "male", "female", "female", "male", "female", "male", "female", "female", "male", "male", "male", "female", "female", "female"],
        "bmi": [27.9, 26.2, 35.3, 36.6, 28.5, 33.4, 29.8, 27.7, 25.8, 26.2, 26.3, 34.4, 39.8, 42.1, 24.6, 30.2, 28.0, 31.8, 35.3, 32.4],
        "children": [0, 0, 1, 3, 0, 1, 0, 3, 0, 0, 0, 0, 3, 0, 1, 1, 0, 2, 0, 0],
        "smoker": ["yes", "no", "no", "no", "no", "no", "no", "no", "no", "no", "yes", "no", "no", "yes", "no", "no", "no", "no", "yes", "no"],
        "region": ["southwest", "southeast", "southeast", "northwest", "northeast", "southeast", "northwest", "northwest", "northeast", "northeast", "southeast", "southwest", "southeast", "southeast", "southwest", "northeast", "southwest", "southeast", "southwest", "northwest"],
        "charges": [16884.92, 13143.34, 7277.91, 10797.34, 3857.94, 8240.59, 4449.46, 7281.51, 12629.17, 2721.32, 27808.73, 2395.17, 11674.23, 39611.76, 1725.55, 9748.91, 2240.20, 11090.72, 36837.47, 12622.28],
        "clinical_note": [
            "Patient is an active smoker with acute emergency, critical condition chest pain.",
            "Routine annual physical exam, normal vitals, blood pressure 120/80.",
            "Diabetes monitoring. HbA1c of 7.2% and BMI is 35.3. Patient taking Metformin 500mg.",
            "History of severe hypertension. BP is 150/95 mmHg. Urgency is normal.",
            "Patient presenting with cough, chest congestion. Urinalysis and spirometry ordered.",
            "Routine checkup. BMI of 33.4. Normal heart rate HR 72 bpm.",
            "Cardiology referral. Normal ECG scheduled.",
            "Annual wellness review, normal vitals.",
            "Follow-up for depression and anxiety. Prescribed Sertraline 500mg.",
            "Pneumonia diagnosis, starting Amoxicillin.",
            "Acute asthma attack, prescribed Salbutamol.",
            "COPD exacerbation, using Tiotropium inhaler.",
            "Urinary tract infection, nitrofurantoin prescribed.",
            "Rosuvastatin prescribed for hyperlipidemia.",
            "Diabetes mellitus type 2, starting Empagliflozin.",
            "Knee pain, taking Paracetamol as needed.",
            "Joint pain, taking Aspirin 81mg daily.",
            "Hypertension control, Losartan 50mg.",
            "Anxiety, taking Omeprazole for acid reflux.",
            "Routine blood test shows anaemia, starting iron supplements."
        ],
        "icd_code": ["I21", "Z00", "E11", "I10", "J44", "Z00", "I25", "Z00", "F32", "J18", "J45", "J44", "N39", "E78", "E11", "M25", "M25", "I10", "F41", "D64"]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_notes() -> list[str]:
    """A list of 5 clinical note strings for patient categories."""
    return [
        "Patient has Type 2 Diabetes and taking Metformin 500mg. HbA1c of 7.5% and BMI is 32.",
        "Severe hypertension. BP is 155/95 mmHg, heart rate HR 85 bpm.",
        "Acute emergency, critical condition with chest pain, possible myocardial infarction.",
        "Routine wellness review, normal vitals.",
        "Respiratory issues, COPD diagnosed, spirometry ordered, X-ray scheduled."
    ]

@pytest.fixture(scope="session")
def trained_preprocessor(sample_df) -> ClaimsPreprocessor:
    """Return a fitted ClaimsPreprocessor instance."""
    preprocessor = ClaimsPreprocessor()
    preprocessor.fit_transform(sample_df)
    return preprocessor

@pytest.fixture(scope="module")
def mock_registry() -> ModelRegistry:
    """Create a ModelRegistry with MagicMocks to bypass real model loading."""
    registry = ModelRegistry()
    registry.is_loaded = True
    
    registry.preprocessor = MagicMock()
    registry.cost_predictor = MagicMock()
    registry.anomaly_detector = MagicMock()
    registry.risk_segmentation = MagicMock()
    registry.approval_classifier = MagicMock()
    registry.note_processor = MagicMock()
    registry.icd_classifier = MagicMock()
    registry.entity_extractor = MagicMock()
    
    return registry
