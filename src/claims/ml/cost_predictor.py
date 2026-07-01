import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostPrediction:
    """What the API returns for a cost prediction."""
    predicted_cost: float          # In actual currency (not log)
    log_prediction: float          # Raw model output (log scale)
    ci_lower: float                # 95% confidence interval lower bound
    ci_upper: float                # 95% confidence interval upper bound
    cost_tier: str                 # "low" / "medium" / "high" / "very_high"
    model_used: str                # Which model made this prediction


@dataclass
class ModelEvaluation:
    """Stores evaluation metrics for a trained model."""
    model_name: str
    mae: float        # Mean Absolute Error: average ₦ off per prediction
    rmse: float       # Root Mean Squared Error: punishes big errors more
    r2: float         # R-squared: 1.0 = perfect, 0.0 = no better than mean
    cv_r2_mean: float # Cross-validation R² (more reliable than single split)
    cv_r2_std: float  # Standard deviation of CV scores (lower = more stable)


class CostPredictor:
    """Predicts insurance claim cost from member features."""
    def __init__(self):
        self.model: Optional[
            GradientBoostingRegressor
            | RandomForestRegressor
            | Ridge
        ] = None
        self.model_name: str = ""
        self.feature_names: list[str] = []
        self.evaluations: list[ModelEvaluation] = []
        self._cost_percentiles: dict = {}

    # Three candidate models to compare
    CANDIDATE_MODELS = {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Ridge": Ridge(alpha=1.0),
    }


    def _tune_model(
        self, model_name: str, X: Any, y: Any
    ):
        """Run GridSearchCV for hyperparameter tuning on the winning model."""

        param_grids = {
            "GradientBoosting": {
                "n_estimators": [100, 200],
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.1],
            },
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0],
            },
        }

        base_model = self.CANDIDATE_MODELS[model_name]
        param_grid = param_grids[model_name]

        logger.info(f"Tuning {model_name} with GridSearchCV...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="r2",
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        logger.info(f"Best params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def train(self,
              X: Any,
              y_log: Any,
              y_raw: Any,
              feature_names: list[str],) -> ModelEvaluation:
        """Train all models, select best one, return its evaluation on validation split."""
        self.feature_names = feature_names

        # Calculate Cost Percentiles (needed for cost tier bucketing)
        self._cost_percentiles = {}
        quantiles = [0.25, 0.5, 0.75, 0.9]

        for q in quantiles:
            self._cost_percentiles[f"q{int(q*100)}"] = np.percentile(y_raw, q*100)
        
        # Split into training and validation sets for unbiased metrics evaluation
        X_train, X_val, y_train_log, y_val_log, y_train_raw, y_val_raw = train_test_split(
            X, y_log, y_raw, test_size=0.2, random_state=42
        )

        logger.info("Comparing candidate models with 5-fold cross-validation on training split...")

        best_score = -np.inf
        best_name = ""

        for name, model in self.CANDIDATE_MODELS.items():
            cv_scores = cross_val_score(
                model, X_train, y_train_log, cv=5, scoring="r2", n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            logger.info(
                f"  {name}: CV R² = {cv_mean:.4f} ± {cv_std:.4f}"
            )

            if cv_mean > best_score:
                best_score = cv_mean
                best_name = name
        
        logger.info(f"Training best model: {best_name} (R²: {best_score:.4f})")
        
        # Use GridSearchCV to find optimal hyperparameters and tune winning model on training split
        best_model = self._tune_model(best_name, X_train, y_train_log)

        # Final Evaluation on the held-out validation split
        y_pred_log_val = best_model.predict(X_val)
        y_pred_raw_val = np.expm1(y_pred_log_val)
        
        metrics = {
            "model_name": best_name,
            "mae": mean_absolute_error(y_val_raw, y_pred_raw_val),
            "rmse": np.sqrt(mean_squared_error(y_val_raw, y_pred_raw_val)),
            "r2": r2_score(y_val_raw, y_pred_raw_val),
            "cv_r2_mean": best_score,
            "cv_r2_std": cv_std,
        }
        evaluation_result = ModelEvaluation(**metrics)
        logger.info(
            f"Final validation metrics — MAE: ${evaluation_result.mae:,.2f} | "
            f"RMSE: ${evaluation_result.rmse:,.2f} | R²: {evaluation_result.r2:.4f}"
        )

        # Refit the tuned model on the entire dataset X so it's optimized for inference
        logger.info(f"Refitting tuned {best_name} on the full dataset...")
        best_model.fit(X, y_log)

        self.model = best_model
        self.model_name = best_name

        return evaluation_result

    def predict(self, X: Any) -> list[CostPrediction]:
        """Predict cost and confidence interval for one or more samples."""

        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict()")
        
        # Predict log cost
        y_pred_log = self.model.predict(X)
        y_pred_raw = np.expm1(y_pred_log)

         # ── Bootstrap confidence interval ─────────────────────────────────────
        # Bootstrap = make many random subsets of training data, predict on each,
        # the spread of predictions gives us the confidence interval.
        # Here we approximate with a fixed percentage band for speed.
        # In production, we would store bootstrapped predictions during training.
        ci_width = y_pred_raw * 0.18  # ±18% band (approximation)

        predictions = []
        for i, (raw, log_pred) in enumerate(zip(y_pred_raw, y_pred_log)):
            predictions.append(CostPrediction(
                predicted_cost=round(float(raw), 2),
                log_prediction=round(float(log_pred), 4),
                ci_lower=round(float(raw - ci_width[i]), 2),
                ci_upper=round(float(raw + ci_width[i]), 2),
                cost_tier=self._assign_cost_tier(raw),
                model_used=self.model_name,
            ))

        return predictions

    def _assign_cost_tier(self, cost: float) -> str:
        """Assign a human-readable cost tier based on percentile thresholds."""
        if cost <= self._cost_percentiles["q25"]:
            return "low"
        elif cost <= self._cost_percentiles["q75"]:
            return "medium"
        elif cost <= self._cost_percentiles["q90"]:
            return "high"
        else:
            return "very_high"

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importance scores.
        Higher score = more influence on the prediction.
        """
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame({"feature": [], "importance": []})

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance_df

    def save(self, path: Path) -> None:
        """Save trained model and artifacts to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(
            self,
            path,
        )
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: Path) -> "CostPredictor":
        return joblib.load(path)