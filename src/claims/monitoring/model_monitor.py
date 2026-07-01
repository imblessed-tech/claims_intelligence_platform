import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    feature_name: str
    ks_statistic: float
    p_value: float
    drift_detected: bool
    reference_mean: float
    current_mean: float
    mean_shift_pct: float

class ModelMonitor:
    def __init__(self) -> None:
        self.reference_stats: dict[str, dict] = {}
        self.prediction_log: list[dict] = []
        self.request_count: int = 0
        self.anomaly_count: int = 0
        self.total_processing_ms: float = 0.0

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store mean, std, min, max, and a raw sample of training features."""
        numeric_cols = df.select_dtypes(include="number").columns
        
        for col in numeric_cols:
            if len(df) > 1000:
                raw_sample = df[col].sample(n=1000, random_state=42).tolist()
            else:
                raw_sample = df[col].tolist()
                
            self.reference_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "raw_values": raw_sample
            }
        logger.info("Reference statistics set for %d features.", len(numeric_cols))

    def check_feature_drift(self, current_data: pd.DataFrame, alpha: float = 0.05) -> list[DriftReport]:
        """Compare current distribution vs reference using Kolmogorov-Smirnov test."""
        reports = []
        
        # Only check numeric columns that exist in both datasets
        numeric_cols = current_data.select_dtypes(include="number").columns
        
        for col in numeric_cols:
            if col not in self.reference_stats:
                continue
                
            ref_data = self.reference_stats[col]["raw_values"]
            current_values = current_data[col].dropna().tolist()
            
            if not ref_data or not current_values:
                continue
                
            # KS two-sample test
            ks_stat, p_value = scipy.stats.ks_2samp(ref_data, current_values)
            ks_stat = float(ks_stat)
            p_value = float(p_value)
            
            ref_mean = self.reference_stats[col]["mean"]
            current_mean = float(np.mean(current_values))
            
            # Mean shift percentage
            if ref_mean != 0:
                mean_shift_pct = ((current_mean - ref_mean) / ref_mean) * 100
            else:
                mean_shift_pct = 0.0
                
            drift_detected = p_value < alpha
            
            reports.append(DriftReport(
                feature_name=col,
                ks_statistic=ks_stat,
                p_value=p_value,
                drift_detected=drift_detected,
                reference_mean=ref_mean,
                current_mean=current_mean,
                mean_shift_pct=mean_shift_pct
            ))
            
        return reports

    def log_prediction(self, features: dict, result: dict, processing_ms: float) -> None:
        """Log recent prediction and increment transaction counters."""
        self.prediction_log.append({
            "features": features,
            "result": result,
            "timestamp": pd.Timestamp.now()
        })
        
        # Keep only last 1000 entries to prevent memory leak
        if len(self.prediction_log) > 1000:
            self.prediction_log.pop(0)
            
        self.request_count += 1
        
        # Check both dict values and nested response model attributes
        is_anomaly = result.get("is_anomaly")
        if is_anomaly is None and hasattr(result, "is_anomaly"):
            is_anomaly = getattr(result, "is_anomaly")
            
        if is_anomaly:
            self.anomaly_count += 1
            
        self.total_processing_ms += processing_ms

    def get_summary(self) -> dict:
        """Return performance and drift summary metrics."""
        avg_proc = self.total_processing_ms / self.request_count if self.request_count > 0 else 0.0
        anom_rate = (self.anomaly_count / self.request_count * 100) if self.request_count > 0 else 0.0
        
        drift_msg = "No drift detected."
        if self.prediction_log:
            recent_features = [entry["features"] for entry in self.prediction_log]
            current_df = pd.DataFrame(recent_features)
            
            try:
                reports = self.check_feature_drift(current_df)
                drifted_features = [r.feature_name for r in reports if r.drift_detected]
                if drifted_features:
                    drift_msg = f"Data drift detected in features: {', '.join(drifted_features)}"
            except Exception as e:
                drift_msg = f"Drift analysis failed: {e}"
                
        return {
            "request_count": self.request_count,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": round(anom_rate, 2),
            "avg_processing_ms": round(avg_proc, 2),
            "drift_status": drift_msg
        }
