# src/claims/ml/risk_segmentation.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

# Maps cluster ID to human-readable risk label.
# You will assign these manually AFTER looking at cluster profiles.
# The mapping below is our expectation — verify after training.
RISK_LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Critical Risk",
}


@dataclass
class SegmentResult:
    cluster_id: int
    risk_label: str
    risk_score: float
    similar_members_count: int


class RiskSegmentation:
    """
    Segments members into 4 risk tiers using K-Means clustering.
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        self.cluster_sizes: dict = {}
        self.cluster_profiles: pd.DataFrame = pd.DataFrame()
        self._label_mapping: dict = RISK_LABELS.copy()

    def find_optimal_k(
        self, X: np.ndarray, k_range: range = range(2, 9),
        output_dir: Path | None = None
    ) -> dict:
        """
        Test K values from 2 to 8.
        Plots elbow curve and silhouette scores.
        Returns the optimal K.
        """
        inertias = []       # Elbow method metric
        silhouettes = []    # Silhouette score metric

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, labels, sample_size=500)) # type: ignore
            logger.info(f"  K={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

        if output_dir:
            self._plot_k_selection(
                list(k_range), inertias, silhouettes, output_dir
            )

        optimal_k = list(k_range)[np.argmax(silhouettes)]
        logger.info(f"Optimal K by silhouette score: {optimal_k}")
        return {"optimal_k": optimal_k, "silhouettes": silhouettes, "inertias": inertias}

    def _plot_k_selection(self, k_values, inertias, silhouettes, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(k_values, inertias, "bo-")
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia (lower = tighter clusters)")
        ax1.set_title("Elbow Method")
        ax1.grid(True)

        ax2.plot(k_values, silhouettes, "ro-")
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score (higher = better)")
        ax2.set_title("Silhouette Score")
        ax2.grid(True)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "k_selection.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"K-selection plot saved")

    def train(self, X: np.ndarray, df: pd.DataFrame) -> dict:
        """Fit K-Means and profile each cluster."""
        logger.info(f"Training K-Means with K={self.n_clusters}...")
        labels = self.model.fit_predict(X)
        df = df.copy()
        df["cluster"] = labels

        # Count members per cluster
        self.cluster_sizes = df["cluster"].value_counts().to_dict()

        # Profile: mean feature values per cluster
        numeric_cols = ["age", "bmi", "children", "charges", "risk_score"]
        available = [c for c in numeric_cols if c in df.columns]
        self.cluster_profiles = df.groupby("cluster")[available].mean().round(2)

        logger.info("Cluster profiles:")
        logger.info(f"\n{self.cluster_profiles}")

        # Auto-assign risk labels based on mean charges per cluster
        # Highest mean charges = Critical Risk
        charge_order = self.cluster_profiles["charges"].rank().astype(int) - 1
        label_list = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
        self._label_mapping = {
            cluster: label_list[rank]
            for cluster, rank in charge_order.items()
        }
        logger.info(f"Cluster label mapping: {self._label_mapping}")

        silhouette = silhouette_score(X, labels, sample_size=500) # type: ignore
        return {
            "silhouette_score": round(silhouette, 4),
            "cluster_sizes": self.cluster_sizes,
            "label_mapping": self._label_mapping,
        }

    def predict(self, X: np.ndarray) -> list[SegmentResult]:
        labels = self.model.predict(X)
        results = []
        for label in labels:
            results.append(SegmentResult(
                cluster_id=int(label),
                risk_label=self._label_mapping.get(label, "Unknown"),
                risk_score=round(float(label / (self.n_clusters - 1) * 10), 1),
                similar_members_count=self.cluster_sizes.get(label, 0),
            ))
        return results

    def get_cluster_profiles(self) -> pd.DataFrame:
        return self.cluster_profiles

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RiskSegmentation":
        return joblib.load(path)