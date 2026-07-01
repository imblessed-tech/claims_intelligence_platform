import logging
import pandas as pd
import numpy as np
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException, status
from claims.config import settings
from claims.api.dependencies import get_registry, ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"]
)

@router.get("/segments")
async def get_risk_segments(registry: ModelRegistry = Depends(get_registry)):
    """
    Load the enriched CSV, run risk segmentation,
    and return counts per risk label, percentage per label, and cluster profiles.
    """
    try:
        if not settings.ENRICHED_DATA_FILE.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enriched data file not found. Please run the training pipeline first."
            )
            
        df = pd.read_csv(settings.ENRICHED_DATA_FILE)
        processed = registry.preprocessor.transform(df)
        
        # Predict clusters for the dataset
        risk_results = registry.risk_segmentation.predict(processed.X_clustering)
        
        labels = [r.risk_label for r in risk_results]
        counts = dict(Counter(labels))
        
        total = len(labels)
        percentages = {label: round((count / total) * 100, 2) for label, count in counts.items()}
        
        # Cluster profiles as a dictionary mapping risk groups to mean metric features
        profiles = registry.risk_segmentation.get_cluster_profiles().to_dict(orient="index")
        
        return {
            "segments": {
                "counts": counts,
                "percentages": percentages,
                "profiles": profiles
            }
        }
    except Exception as e:
        logger.error(f"Error generating risk segments analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing segments analytics: {str(e)}"
        )

@router.get("/anomalies")
async def get_anomalies(registry: ModelRegistry = Depends(get_registry)):
    """
    Load the enriched CSV, run anomaly detection,
    and return total count, anomaly count, anomaly rate, and score distribution.
    """
    try:
        if not settings.ENRICHED_DATA_FILE.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enriched data file not found. Please run the training pipeline first."
            )
            
        df = pd.read_csv(settings.ENRICHED_DATA_FILE)
        processed = registry.preprocessor.transform(df)
        
        # Predict anomalies
        anomaly_results = registry.anomaly_detector.predict(processed.X_clustering)
        
        total_count = len(anomaly_results)
        anomaly_count = sum(1 for r in anomaly_results if r.is_anomaly)
        anomaly_rate = round((anomaly_count / total_count) * 100, 2) if total_count > 0 else 0.0
        
        scores = [r.anomaly_score for r in anomaly_results]
        
        score_distribution = {
            "mean": round(float(np.mean(scores)), 4) if scores else 0.0,
            "std": round(float(np.std(scores)), 4) if scores else 0.0,
            "p5": round(float(np.percentile(scores, 5)), 4) if scores else 0.0,
            "p95": round(float(np.percentile(scores, 95)), 4) if scores else 0.0,
        }
        
        return {
            "total_count": total_count,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "score_distribution": score_distribution
        }
    except Exception as e:
        logger.error(f"Error generating anomaly analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing anomaly analytics: {str(e)}"
        )

@router.get("/model-performance")
async def get_model_performance():
    """
    Return aggregate evaluation performance metrics from training runs.
    Hardcoded model metrics from training runs.
    In a live implementation, these should be loaded from MLflow or similar tracking system.
    """
    return {
        "cost_predictor": {
            "mae": 1475.29,
            "r2": 0.8089,
            "model_type": "Ridge Regression"
        },
        "approval_classifier": {
            "roc_auc": 0.9832,
            "f1": 0.9542,
            "model_type": "Random Forest Classifier"
        },
        "anomaly_detector": {
            "contamination_rate": settings.ANOMALY_CONTAMINATION,
            "model_type": "Isolation Forest"
        },
        "risk_segmentation": {
            "silhouette_score": 0.5841,
            "num_clusters": settings.N_CLUSTERS,
            "model_type": "K-Means Clustering"
        },
        "icd_classifier": {
            "accuracy": 0.985,
            "f1_weighted": 0.985,
            "model_type": "TF-IDF + Logistic Regression"
        }
    }