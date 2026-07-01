from typing import Any, cast
from numpy.typing import NDArray

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

@dataclass
class ApprovalResult:
    approved: bool
    probability: float
    confidence: str
    decision_factors: list[str]

class ApprovalClassifier:
    def __init__(self, random_state: int = 42, n_estimators: int = 300, min_samples_split: int = 10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )
        self.feature_names: list[str] = []
        self.threshold: float = 0.5

    def train(self, X:np.ndarray, y:np.ndarray, feature_names:list[str], output_dir:Path|None = None) -> dict:
        self.feature_names = feature_names
        logger.info(f"Training ApprovalClassifier with {len(X)} samples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report = cast(dict, class_report)

        if output_dir:
            self._save_roc_curve(y_test, y_proba, float(auc), output_dir)
            self._save_confusion_matrix(y_test, y_pred, output_dir)

        logger.info(f"""
        ApprovalClassifier trained. AUC: {auc:.4f},
        Threshold: {self.threshold:.4f},
        Approval Rate: {y_pred.mean():.4f}
        """)

        return {
            "roc_auc": float(auc),
            "threshold": self.threshold,
            "accuracy": float(class_report["accuracy"]),
            "precision": float(class_report["1"]["precision"]),
            "recall": float(class_report["1"]["recall"]),
            "f1": float(class_report["1"]["f1-score"]),
            "approval_rate": float(y_pred.mean()),
        }

    def _save_roc_curve(self, y_true: Any, y_proba: Any, auc: float, output_dir: Path) -> None:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=cast(Any, ax))
        ax.set_title(f"ROC Curve (AUC = {auc:.4f})")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "roc_curve.png")
        plt.close()
        logger.info("ROC curve saved")

    def _save_confusion_matrix(self, y_true: Any, y_pred: Any, output_dir: Path) -> None:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, 
            display_labels=["Rejected", "Approved"],
            ax=cast(Any, ax), 
            colorbar=False
        )
        ax.set_title("Approval Classifier - Confusion Matrix")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
        logger.info("Confusion matrix saved")

    def predict(self, X: np.ndarray) -> list[ApprovalResult]:
        probas = self.model.predict_proba(X)[:, 1]
        importances = self.model.feature_importances_
        top_importances = np.argsort(importances)[-3:][::-1]
        factors = [
            self.feature_names[i] for i in top_importances
            if i < len(self.feature_names)
        ]
        
        results = []
        for proba in probas:
            is_approved = proba >= self.threshold
            if abs(proba - self.threshold) > 0.3:
                confidence = "high"
            elif abs(proba - 0.5) > 0.15:
                confidence = "medium"
            else:
                confidence = "Low"
            results.append(
                ApprovalResult(
                    approved=bool(is_approved),
                    probability=round(float(proba), 4),
                    confidence=confidence,
                    decision_factors=factors,
                )
            )
        return results

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ApprovalClassifier":
        return joblib.load(path)