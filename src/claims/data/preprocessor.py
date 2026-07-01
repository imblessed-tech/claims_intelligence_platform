import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
import joblib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class ProcessedData:
    df: pd.DataFrame
    X_regression: Any | np.ndarray = field(default=None)
    X_clustering: Any | np.ndarray = field(default=None)
    X_classification: Any | np.ndarray = field(default=None)
    y_cost: Any | np.ndarray = field(default=None)
    y_cost_raw: Any | np.ndarray = field(default=None)
    y_approved: Any | np.ndarray = field(default=None)
    feature_names: Optional[list[str]] = field(default=None)
    scaler: Optional[StandardScaler] = None


class ClaimsPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Track min/max values for features to prevent data leakage and zero division at inference
        self.age_min: Optional[float] = None
        self.age_max: Optional[float] = None
        self.bmi_min: Optional[float] = None
        self.bmi_max: Optional[float] = None
        self.children_min: Optional[float] = None
        self.children_max: Optional[float] = None

    def fit_transform(self, df: pd.DataFrame) -> ProcessedData:
        """Fit scaler and transform features - Called ONLY during training"""
        df = self._engineer_features(df)
        df = self._create_target_variables(df)
        result = self._build_feature_matrices(df, fit=True)

        self.is_fitted = True
        return result
    
    def transform(self, df: pd.DataFrame) -> ProcessedData:
        """Transform features using fitted scaler - Called at prediction time"""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming.")

        df = self._engineer_features(df)
        result = self._build_feature_matrices(df, fit=False)
        return result

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core feature engineering step - Create new columns from existing ones"""

        _df = df.copy()

        # Convert categorical columns to numeric representation
        if "smoker" in _df.columns:
            if _df["smoker"].dtype == bool:
                _df["smoker"] = _df["smoker"].astype(int)
            else:
                _df["smoker"] = (_df["smoker"].astype(str).str.lower().str.strip().isin(["yes", "true", "1"])).astype(int)
        
        if "sex" in _df.columns:
            if _df["sex"].dtype == bool:
                _df["sex"] = _df["sex"].astype(int)
            else:
                _df["sex"] = (_df["sex"].astype(str).str.lower().str.strip().isin(["male", "true", "1"])).astype(int)

        if "region" in _df.columns:
            if not pd.api.types.is_numeric_dtype(_df["region"]):
                region_map = {
                    "northeast": 0,
                    "northwest": 1,
                    "southeast": 2,
                    "southwest": 3
                }
                _df["region"] = _df["region"].astype(str).str.lower().str.strip().map(region_map).fillna(0).astype(int)

        # Age Group
        # Medical risk increases with age in distinct jumps, so grouping
        _df["age_group"] = pd.cut(
            _df["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["youth", "young_adult", "adult", "middle_aged", "senior"],
        )

        # BMI Categories (Categorical Feature)
        # WHO standard BMI categories. These have clinical meaning.
        # underweight:<18.5, normal:18.5-25, overweight:25-30, obese:>30
        _df["bmi_category"] = pd.cut(
            _df["bmi"],
            bins=[0, 18.5, 25.0, 30.0, 100],
            labels=["underweight", "normal", "overweight", "obese"],
        )
           
        # Fit or use min/max values to prevent data leakage and zero division at inference
        if not self.is_fitted:
            self.age_min, self.age_max = float(_df["age"].min()), float(_df["age"].max())
            self.bmi_min, self.bmi_max = float(_df["bmi"].min()), float(_df["bmi"].max())
            self.children_min, self.children_max = float(_df["children"].min()), float(_df["children"].max())

        if (
            self.age_min is None or self.age_max is None or
            self.bmi_min is None or self.bmi_max is None or
            self.children_min is None or self.children_max is None
        ):
            raise RuntimeError("Preprocessor has not been fitted or is missing scale parameters.")

        # Safeguard division by zero if max == min (e.g. single row prediction)
        age_range = self.age_max - self.age_min if self.age_max != self.age_min else 1.0
        bmi_range = self.bmi_max - self.bmi_min if self.bmi_max != self.bmi_min else 1.0
        children_range = self.children_max - self.children_min if self.children_max != self.children_min else 1.0

        age_norm = (_df["age"] - self.age_min) / age_range
        bmi_norm = (_df["bmi"] - self.bmi_min) / bmi_range
        children_norm = (_df["children"] - self.children_min) / children_range

        # Health Risk: interaction of age, bmi, children and smoker status
        # Smoking (weight 3), BMI (weight 2), age (weight 1.5), children (weight 0.5)
        _df["risk_score"] = (
            _df["smoker"] * 3 +
            bmi_norm * 2 +
            age_norm * 1.5 +
            children_norm * 0.5
        )
        
        # Interaction features
        # To discover hidden patterns; 60 year old smoker (higher risk), obese smoker (higher risk), etc
        _df["age_smoker"] = _df["age"] * _df["smoker"]
        _df["bmi_smoker"] = _df["bmi"] * _df["smoker"]
        _df["age_bmi"] = _df["age"] * _df["bmi"]

        # Log transform charges (creating a separate column to keep raw charges intact)
        if "charges" in _df.columns:
            _df["log_charges"] = np.log1p(_df["charges"])

        return _df

    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the labels (y values) the models learn to predict.
        This runs only during training — test data has no labels at API time."""

        _df = df.copy()
        
        # High cost flag for classification
        # Top 25% of claims = "high cost". This becomes our approval target.
        cost_threshold = _df["charges"].quantile(0.75)
        _df["high_cost_flag"] = (_df["charges"] > cost_threshold).astype(int)

        # Approval label
        # Simulate business approval logic:
        # 1. High cost claims = approved
        # 2. Very obese (BMI>45) = rejected
        # 3. Elderly smokers (age>60) = rejected
        # 4. Everything else = premium (approved)
        _df["approved"] = (
            (_df["charges"] < _df["charges"].quantile(0.90)) &
            (_df["bmi"] < 45) &
            ~((_df["smoker"] == 1) & (_df["age"] > 60))
        ).astype(int)

        logger.info(
            f"Approval rate: {_df['approved'].mean():.1%} "
            f"({_df['approved'].sum()} approved of {len(_df)})"
        )

        return _df

    def _build_feature_matrices(self, df: pd.DataFrame, fit: bool = False) -> ProcessedData:
        """Select and scale features for each model type.
        fit=True during training, fit=False during prediction."""
        
        numeric_features = [
            "age", "bmi", "children", "risk_score",
            "age_smoker", "bmi_smoker", "age_bmi",
            "smoker", "sex"
        ]

        # One-hot encode categorical features
        region_dummies = pd.get_dummies(df["region"], prefix="region")
        age_group_dummies = pd.get_dummies(df["age_group"], prefix="age_group")
        bmi_cat_dummies = pd.get_dummies(df["bmi_category"], prefix="bmi_cat")

        # Combine all features into a single DataFrame
        feature_df = pd.concat(
            [
                df[numeric_features],
                region_dummies,
                age_group_dummies,
                bmi_cat_dummies,
            ],
            axis=1
        ).fillna(0)

        features_names = feature_df.columns.tolist()

        if not fit and hasattr(self.scaler, "feature_names_in_"):
            # Ensure the columns exactly match the features seen during fit
            feature_df = feature_df.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            features_names = feature_df.columns.tolist()

        if fit:
            X_scaled = self.scaler.fit_transform(feature_df)
        else:
            X_scaled = self.scaler.transform(feature_df)

        # Build target arrays (only available during training)
        y_cost = df["log_charges"].values if "log_charges" in df.columns else None
        y_cost_raw = df["charges"].values if "charges" in df.columns else None
        y_approved = df["approved"].values if "approved" in df.columns else None
            
        return ProcessedData(
            df=df,
            X_regression=X_scaled,
            X_clustering=X_scaled,
            X_classification=X_scaled,
            y_cost=y_cost,      
            y_cost_raw=y_cost_raw,
            y_approved=y_approved,
            feature_names=features_names,
            scaler=self.scaler,
        )

    def save(self, path: Path) -> None:
        """Save the fitted preprocessor so it can be loaded at API time."""
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ClaimsPreprocessor":
        """Load a previously fitted preprocessor."""
        return joblib.load(path)