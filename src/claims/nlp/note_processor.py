import logging
import re
import numpy as np
from pathlib import Path
from typing import Any, cast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

ABBREVIATIONS = {
    "BP": "blood pressure",
    "BMI": "body mass index",
    "Hx": "history",
    "Rx": "prescription",
    "Dx": "diagnosis",
    "Tx": "treatment",
    "SOB": "shortness of breath",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "MI": "myocardial infarction",
    "ECG": "electrocardiogram",
    "FBC": "full blood count",
    "OD": "once daily",
    "BD": "twice daily",
    "TDS": "three times daily",
    "QDS": "four times daily",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ClinicalNoteProcessor:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self.lda = LatentDirichletAllocation(n_components=5, random_state=42)
        self.is_fitted = False
        self.topic_labels = {}
    
    def clean(self, note: str) -> str:
        # Replace abbreviations using word boundaries and case-insensitivity
        for abbrev, full in ABBREVIATIONS.items():
            note = re.sub(rf"\b{abbrev}\b", full, note, flags=re.IGNORECASE)
        note = note.lower()
        # Remove special characters
        note = re.sub(r'[^a-z0-9\s]', ' ', note)
        # Remove extra whitespace
        note = re.sub(r'\s+', ' ', note).strip()
        return note        

    def fit(self, notes: list[str]) -> None:
        logger.info("Fitting ClinicalNoteProcessor...")
        cleaned_notes = [self.clean(n) for n in notes]
        self.vectorizer.fit(cleaned_notes)
        X = self.vectorizer.transform(cleaned_notes)
        self.lda.fit(X)
        self._assign_topic_labels(X)
        self.is_fitted = True
    
    def extract_keywords(self, note: str, top_n: int = 10) -> list[str]:
        """Extract top n keywords from a note using TF-IDF."""
        if not self.is_fitted:
            raise RuntimeError("ClinicalNoteProcessor must be fit before using extract_keywords()")
        
        cleaned = self.clean(note)
        X = self.vectorizer.transform([cleaned])

        # Get the TF-IDF scores for this single document
        tfidf_scores = cast(Any, X).toarray().flatten()

        # Return the top N feature names by score
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
        top_words = [self.vectorizer.get_feature_names_out()[i] for i in top_indices]
        return top_words        

    def _assign_topic_labels(self, X: Any) -> None:
        """Auto-label topics based on highest-weight terms in each topic."""
        feature_names = self.vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(self.lda.components_):
            top_10_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_10_indices]
            logger.info(f"Topic {topic_idx} top words: {top_words}")
            
            topic_str = " ".join(top_words).lower()
            
            if any(word in topic_str for word in ["diabetes", "insulin", "glucose", "hba1c", "metformin", "lipid", "hyperlipidemia", "cholesterol", "ldl"]):
                label = "Metabolic Disorders"
            elif any(word in topic_str for word in ["hypertension", "blood pressure", "cardiac", "heart", "ischemic", "ischaemic", "angina", "ecg", "amlodipine", "systolic", "diastolic"]):
                label = "Cardiovascular Disease"
            elif any(word in topic_str for word in ["cough", "respiratory", "bronchial", "copd", "asthma", "wheeze", "shortness", "dyspnoea", "inhaler"]):
                label = "Respiratory Conditions"
            elif any(word in topic_str for word in ["wellness", "annual", "routine", "examination", "screening", "preventive"]):
                label = "Preventive Care"
            else:
                label = "General Medicine"
                
            self.topic_labels[topic_idx] = label

    def get_urgency(self, note: str) -> str:
        """Retrieve Urgency Level of the claim note"""

        cleaned_note = self.clean(note)
        
        URGENT_WORDS = {
            "critical", "emergency", "severe", "severely", "uncontrolled", "acute", "urgent", 
            "urgently", "admitted", "admission", "troponin", "arrest", "immediate", "unstable", 
            "sepsis", "collapse", "worsening", "progressive", "decline", "complication", 
            "complicated", "exacerbation", "angina", "dyspnoea", "dyspnea", "infarction"
        }

        ROUTINE_WORDS = {
            "stable", "controlled", "elective", "routine", "annual", "wellness", "preventive", 
            "screening", "normal", "lifestyle", "reassured", "mild", "regular", "check", 
            "review", "monitoring", "follow", "scheduled"
        }

        # Count matches in each set
        words = cleaned_note.split()
        urgent_count = sum(1 for word in words if word in URGENT_WORDS)
        routine_count = sum(1 for word in words if word in ROUTINE_WORDS)

        # Determine urgency level
        if urgent_count > routine_count:
            return "urgent"
        elif routine_count > urgent_count:
            return "routine"
        else:
            return "moderate"

    def get_topic(self, note: str) -> dict:
        """Retrieve Topic Information"""

        if not self.is_fitted:
            raise RuntimeError("ClinicalNoteProcessor must be fit before using get_topic()")
        
        cleaned = self.clean(note)
        X = self.vectorizer.transform([cleaned])
        probs = self.lda.transform(X)[0]
        
        dominant_topic = int(np.argmax(probs))
        
        return {
            "dominant_topic": dominant_topic,
            "topic_label": self.topic_labels.get(dominant_topic, "General Medicine"),
            "topic_probabilities": probs.tolist(),
        }

    def save(self, path: Path) -> None:
        """Save the NoteProcessor to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ClinicalNoteProcessor":
        """Load the NoteProcessor from a file."""
        return joblib.load(path)

