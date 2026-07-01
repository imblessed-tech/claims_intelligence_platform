import time
import logging
from typing import Optional
from fastapi import HTTPException
from claims.config import settings

from claims.data.preprocessor import ClaimsPreprocessor
from claims.ml.cost_predictor import CostPredictor
from claims.ml.anomaly_detector import AnomalyDetector
from claims.ml.risk_segmentation import RiskSegmentation
from claims.ml.approval_classifier import ApprovalClassifier
from claims.nlp.icd_classifier import ICDClassifier
from claims.nlp.note_processor import ClinicalNoteProcessor
from claims.nlp.entity_extractor import MedicalEntityExtractor

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.preprocessor: Optional[ClaimsPreprocessor] = None
        self.cost_predictor: Optional[CostPredictor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.risk_segmentation: Optional[RiskSegmentation] = None
        self.approval_classifier: Optional[ApprovalClassifier] = None
        self.note_processor: Optional[ClinicalNoteProcessor] = None
        self.icd_classifier: Optional[ICDClassifier] = None
        self.entity_extractor: Optional[MedicalEntityExtractor] = None
        self.is_loaded: bool = False

    def load_all_models(self) -> None:
        start_time = time.time()
        logger.info("Loading all ML and NLP models...")

        try:
            self.preprocessor = ClaimsPreprocessor.load(settings.MODELS_DIR / "preprocessor.joblib")
            self.cost_predictor = CostPredictor.load(settings.COST_PREDICTOR_FILE)
            self.anomaly_detector = AnomalyDetector.load(settings.ANOMALY_DETECTOR_FILE)
            self.risk_segmentation = RiskSegmentation.load(settings.RISK_SEGMENTATION_FILE)
            self.approval_classifier = ApprovalClassifier.load(settings.APPROVAL_CLASSIFIER_FILE)
            self.note_processor = ClinicalNoteProcessor.load(settings.NLP_VECTORIZER_FILE)
            self.icd_classifier = ICDClassifier.load(settings.ICD_CLASSIFIER_FILE)
            self.entity_extractor = MedicalEntityExtractor()
            
            self.is_loaded = True
            end_time = time.time()
            logger.info(f"All models loaded in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise


registry = ModelRegistry()


def load_all_models() -> None:
    """Load all models into the global registry."""
    registry.load_all_models()


def get_registry() -> ModelRegistry:
    """Get the global model registry, raising a 503 if not loaded."""
    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    return registry