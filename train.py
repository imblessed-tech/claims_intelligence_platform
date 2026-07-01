# train.py

import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from claims.config import settings, ensure_directories
from claims.data.loader import load_data
from claims.data.synthesizer import ClinicalNoteSynthesizer
from claims.data.preprocessor import ClaimsPreprocessor
from claims.ml.cost_predictor import CostPredictor
from claims.ml.anomaly_detector import AnomalyDetector
from claims.ml.risk_segmentation import RiskSegmentation
from claims.ml.approval_classifier import ApprovalClassifier
from claims.nlp.icd_classifier import ICDClassifier
from claims.nlp.note_processor import ClinicalNoteProcessor
from claims.visualization.dashboard import DashboardGenerator

#  Logging setup ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("CLAIMS INTELLIGENCE PLATFORM - TRAINING PIPELINE")
    logger.info("=" * 60)

    ensure_directories()

    #  Step 1: Load raw data ─
    logger.info("\n[1/9] Loading raw insurance data...")
    df = load_data(settings.RAW_DATA_FILE)
    logger.info(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

    #  Step 2: Generate clinical notes 
    logger.info("\n[2/9] Generating synthetic clinical notes...")
    if settings.ENRICHED_DATA_FILE.exists():
        logger.info("      Enriched file already exists — skipping generation")
        import pandas as pd
        df = pd.read_csv(settings.ENRICHED_DATA_FILE)
    else:
        synthesizer = ClinicalNoteSynthesizer()
        df = synthesizer.generate(df)
        synthesizer.save(df, settings.ENRICHED_DATA_FILE)
    logger.info(f"      Dataset now has {len(df.columns)} columns")

    #  Step 3: Preprocess features ─
    logger.info("\n[3/9] Engineering features and preprocessing...")
    preprocessor = ClaimsPreprocessor()
    processed = preprocessor.fit_transform(df)
    preprocessor.save(settings.MODELS_DIR / "preprocessor.joblib")
    logger.info(f"      Feature matrix shape: {processed.X_regression.shape}")

    #  Step 4: Train Cost Predictor 
    logger.info("\n[4/9] Training Cost Predictor (regression)...")
    cost_predictor = CostPredictor()
    cost_eval = cost_predictor.train(
        X=processed.X_regression,
        y_log=processed.y_cost,
        y_raw=processed.y_cost_raw,
        feature_names=processed.feature_names or [],
    )
    cost_predictor.save(settings.COST_PREDICTOR_FILE)
    logger.info(f"      [OK] MAE: ${cost_eval.mae:,.2f} | R2: {cost_eval.r2:.4f}")

    #  Step 5: Train Anomaly Detector 
    logger.info("\n[5/9] Training Anomaly Detector...")
    anomaly_detector = AnomalyDetector(
        contamination=settings.ANOMALY_CONTAMINATION
    )
    anomaly_summary = anomaly_detector.train(processed.X_clustering)
    anomaly_detector.save(settings.ANOMALY_DETECTOR_FILE)
    logger.info(
        f"      [OK] Anomaly rate: {anomaly_summary['anomaly_rate']:.1%}"
    )

    #  Step 6: Train Risk Segmentation 
    logger.info("\n[6/9] Training Risk Segmentation (clustering)...")
    risk_model = RiskSegmentation(n_clusters=settings.N_CLUSTERS)
    risk_model.find_optimal_k(
        processed.X_clustering,
        output_dir=settings.REPORTS_DIR
    )
    seg_summary = risk_model.train(
        processed.X_clustering, processed.df
    )
    risk_model.save(settings.RISK_SEGMENTATION_FILE)
    logger.info(
        f"      [OK] Silhouette score: {seg_summary['silhouette_score']}"
    )

    #  Step 7: Train Approval Classifier 
    logger.info("\n[7/9] Training Approval Classifier...")
    approval_clf = ApprovalClassifier()
    approval_eval = approval_clf.train(
        X=processed.X_classification,
        y=processed.y_approved,
        feature_names=processed.feature_names or [],
        output_dir=settings.REPORTS_DIR,
    )
    approval_clf.save(settings.APPROVAL_CLASSIFIER_FILE)
    logger.info(
        f"      [OK] ROC-AUC: {approval_eval['roc_auc']} | "
        f"F1: {approval_eval['f1']}"
    )

    #  Step 7b: Note Processor
    logger.info("\n[7b/9] Training Note Processor...")
    note_processor = ClinicalNoteProcessor()
    note_processor.fit(notes=df["clinical_note"].tolist())
    note_processor.save(settings.NLP_VECTORIZER_FILE)
    logger.info(
        f"      [OK] Note Processor trained and saved to {settings.NLP_VECTORIZER_FILE}"
    )

    #  Step 8: Train ICD Classifier 
    logger.info("\n[8/9] Training ICD Classifier...")
    icd_clf = ICDClassifier()
    icd_eval = icd_clf.train(
        notes=df["clinical_note"].tolist(),
        labels=df["icd_code"].tolist(),
    )
    icd_clf.save(settings.ICD_CLASSIFIER_FILE)

    logger.info(
        f"      [OK] ROC-AUC: {icd_eval['roc_auc']} | "
        f"F1: {icd_eval['f1']}"
    )

    #  Step 9: Generate Static Dashboard Report
    logger.info("\n[9/9] Generating static HTML dashboard report...")
    models = {
        "preprocessor": preprocessor,
        "cost_predictor": cost_predictor,
        "anomaly_detector": anomaly_detector,
        "risk_segmentation": risk_model,
        "approval_classifier": approval_clf,
        "icd_classifier": icd_clf,
    }
    dashboard_path = settings.REPORTS_DIR / "dashboards" / "claims_intelligence_report.html"
    dashboard = DashboardGenerator(df, models)
    dashboard.generate(dashboard_path)
    logger.info(f"      [OK] Dashboard generated at: {dashboard_path.resolve()}")

    #  Final Summary 
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - All models saved to models/")
    logger.info("=" * 60)
    logger.info(f"\n  Cost Predictor   -> MAE=${cost_eval.mae:,.2f}, R2={cost_eval.r2:.4f}")
    logger.info(f"  Anomaly Detector -> {anomaly_summary['n_anomalies']} anomalies flagged")
    logger.info(f"  Risk Segments    -> Silhouette={seg_summary['silhouette_score']}")
    logger.info(f"  Approval Model   -> ROC-AUC={approval_eval['roc_auc']}, F1={approval_eval['f1']}")
    logger.info(f"  Dashboard        -> Generated at {dashboard_path.resolve()}")
    logger.info(f"\n  Run next: uvicorn src.claims.main:app --reload")


if __name__ == "__main__":
    main()