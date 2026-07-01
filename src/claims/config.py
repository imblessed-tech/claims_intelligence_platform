# src/claims/config.py

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Project paths 
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # ── Data files 
    SOURCE_DATA_URL: str = "https://www.kaggle.com/datasets/easonlai/sample-insurance-claim-prediction-dataset?resource=download&select=insurance2.csv"
    RAW_DATA_FILE: Path = DATA_DIR / "insurance.csv"
    ENRICHED_DATA_FILE: Path = DATA_DIR / "claims_with_notes.csv"

    # ── Data Validation
    MIN_ROW: int = 100

    # ── Model files 
    COST_PREDICTOR_FILE: Path = MODELS_DIR / "cost_predictor.joblib"
    ANOMALY_DETECTOR_FILE: Path = MODELS_DIR / "anomaly_detector.joblib"
    RISK_SEGMENTATION_FILE: Path = MODELS_DIR / "risk_segmentation.joblib"
    APPROVAL_CLASSIFIER_FILE: Path = MODELS_DIR / "approval_classifier.joblib"
    NLP_VECTORIZER_FILE: Path = MODELS_DIR / "nlp_vectorizer.joblib"
    ICD_CLASSIFIER_FILE: Path = MODELS_DIR / "icd_classifier.joblib"

    # ── ML settings 
    TEST_SIZE: float = 0.2       
    RANDOM_STATE: int = 42           
    N_CLUSTERS: int = 4              
    ANOMALY_CONTAMINATION: float = 0.05  
    HIGH_COST_PERCENTILE: float = 0.75   

    # ── API settings 
    API_TITLE: str = "Claims Intelligence Platform"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"         
        extra = "ignore"

settings = Settings()


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.REPORTS_DIR,
        settings.REPORTS_DIR / "dashboards",
    ]:
        directory.mkdir(parents=True, exist_ok=True)