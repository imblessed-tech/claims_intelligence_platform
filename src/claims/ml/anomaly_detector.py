from pathlib import Path
from typing import cast
from numpy import percentile
import numpy as np
import logging
import joblib
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Result of a single anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    percentile_rank: float
    flag_reason: str


class AnomalyDetector:
    """Detects anomalies in claim data."""
    def __init__(self, contamination:float=0.05) -> None:
        """Initialize AnomalyDetector with IsolationForest."""
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs = -1
        )
        self._all_scores: np.ndarray = np.array([])

    def train(self, X: np.ndarray) -> dict:
        """Train the anomaly detector on the given data.
        
        Call self.model.fit(X). Then call self.model.decision_function(X) 
        and store the result in self._all_scores. 
        Call self.model.predict(X) — count how many are -1 (anomalies). 
        Log the anomaly rate. Return a summary dict with n_anomalies, 
        anomaly_rate, score_mean, score_std.
        """
        logger.info("Training AnomalyDetector on %d samples...", len(X))

        self.model.fit(X)
        raw_scores = self.model.decision_function(X).reshape(-1)
        self._all_scores = cast(np.ndarray, raw_scores)

        n_anomalies = (self.model.predict(X) == -1).sum()

        score_mean = self._all_scores.mean()
        score_std = self._all_scores.std()
        anomaly_rate = n_anomalies / len(X) * 100

        logger.info(
            "Anomaly detection complete. "
            "Anomalies: %d (%.1f%%), "
            "Score range: %.3f to %.3f, "
            "Mean score: %.3f"
            % (n_anomalies, anomaly_rate, self._all_scores.min(), self._all_scores.max(), score_mean)
        )

        return {
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(anomaly_rate),
            "score_mean": float(score_mean),
            "score_std": float(score_std),
        }      
        
    def predict(self, X: np.ndarray) -> list[AnomalyResult]:
        """Predict anomalies in new data."""
        X = np.asarray(X)
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)  

        results = []
        for score, pred in zip(scores, predictions):
            # Percentile rank: what % of training claims had a lower anomaly score
            percentile = float(
                np.mean(self._all_scores <= score) * 100
            )
            is_anomaly = pred == -1

            reason = "Normal claim pattern"
            if is_anomaly:
                if score < np.percentile(self._all_scores, 2):
                    reason = "Severely anomalous — multiple KPIs far from population norm"
                else:
                    reason = "Unusual claim pattern — review recommended"

            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=round(float(score), 4),
                percentile_rank=round(percentile, 1),
                flag_reason=reason,
            ))

        return results

    def get_anomaly_summary(self, X: np.ndarray) -> dict:
        """Run predict on full X. 
        Return counts of anomalies, percentage, and 
        distribution of scores."""
        results = self.predict(X)

        n_anomalies = sum(1 for r in results if r.is_anomaly)
        anomaly_rate = n_anomalies / len(results) * 100

        # Calculate score quartiles
        all_scores = [r.anomaly_score for r in results]
        q1 = float(np.percentile(all_scores, 25))
        median = float(np.percentile(all_scores, 50))
        q3 = float(np.percentile(all_scores, 75))

        return {
            "n_claims": len(results),
            "n_anomalies": n_anomalies,
            "anomaly_rate_pct": round(anomaly_rate, 1),
            "score_min": round(min(all_scores), 4),
            "score_q1": round(q1, 4),
            "score_median": round(median, 4),
            "score_q3": round(q3, 4),
            "score_max": round(max(all_scores), 4),
        }

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        return joblib.load(path)




        
        

        
        

    