from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from typing import Optional, cast
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class ICDResult:
    icd_code: str
    description: str
    confidence: float
    alternative_codes: list[str]
    all_probabilities: dict

class ICDClassifier:
    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.classes_ = []
        self.code_descriptions  = {}

    def train(self, notes: list[str], labels: list[str]) -> dict:
        """Train the ICD classifier."""
        logger.info("Training ICD classifier on %d samples...", len(notes))

        X_train, X_val, y_train, y_val = train_test_split(
            notes, labels, test_size=0.2, stratify=labels
        )

        self.classes_ = sorted(list(set(labels)))

        # Build pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver="lbfgs",
            )),
        ])

        # Train
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred_val = self.pipeline.predict(X_val)
        report = cast(dict, classification_report(y_val, y_pred_val, output_dict=True))

        # Description mapping - Description will be loaded from ICD dataset, in real world
        if not self.code_descriptions:
            for label in self.classes_:
                if " " not in label:  # simple check: code should not contain space
                    self.code_descriptions[label] = "Description coming soon"

        # Sample predictions
        samples = []
        for i in range(3):
            text = X_val[i]
            pred = y_pred_val[i]
            probs = self.pipeline.predict_proba([text])[0]
            top_idx = probs.argsort()[-1]
            top_prob = float(probs[top_idx])

            samples.append({
                "input": text[:100] + "...",
                "predicted_code": pred,
                "predicted_desc": self.code_descriptions.get(pred, "Unknown"),
                "confidence": round(top_prob, 3),
            })

        logger.info("Training complete. Sample predictions:")
        for s in samples:
            logger.info(f"  - {s['input']} → {s['predicted_code']} ({s['confidence']:.3f})")

        y_prob = self.pipeline.predict_proba(X_val)
        try:
            roc_auc = float(roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = 0.0

        return {
            "num_samples": len(notes),
            "num_classes": len(self.classes_),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "accuracy": float(report["accuracy"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
            "roc_auc": roc_auc,
            "f1": float(report["weighted avg"]["f1-score"]),
            "samples": samples,
        }
        

    def predict(self, note: str) -> ICDResult:
        """
        Make a prediction for a single note.
        """
        if self.pipeline is None:
            raise RuntimeError("ICDClassifier must be trained before using predict()")

        # Predict class
        predicted_class = cast(str, self.pipeline.predict([note])[0])

        # Predict probabilities
        probs = self.pipeline.predict_proba([note])[0]
        
        # Build all probabilities dict
        all_probabilities = dict(
            zip(self.classes_, map(float, probs))
        )

        # Sort by probability descending
        sorted_probs = sorted(
            all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Extract top 3 alternatives
        top_3 = []
        for code, prob in sorted_probs[1:4]:
            top_3.append(code)

        return ICDResult(
            icd_code=predicted_class,
            description=self.code_descriptions.get(predicted_class, "Unknown"),
            confidence=sorted_probs[0][1],
            alternative_codes=top_3,
            all_probabilities=all_probabilities,
        )

    def evaluate(self, notes_test: list[str], labels_test: list[str]) -> dict[str, float]:
        """Evaluate the classifier on test data."""
        if self.pipeline is None:
            raise RuntimeError("ICDClassifier must be trained before using evaluate()")

        y_pred = self.pipeline.predict(notes_test)
        report = cast(dict[str, float], classification_report(labels_test, y_pred, output_dict=True))

        return report
    
    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "classes_": self.classes_,
                "code_descriptions": self.code_descriptions,
            },
            path,
        )
        logger.info("ICDClassifier saved to %s", path)
    
    @classmethod
    def load(cls, path: Path) -> "ICDClassifier":
        data = joblib.load(path)
        if isinstance(data, dict):
            obj = cls()
            obj.pipeline = data.get("pipeline")
            obj.classes_ = data.get("classes_", [])
            obj.code_descriptions = data.get("code_descriptions", {})
            return obj
        return data