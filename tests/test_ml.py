import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from claims.config import settings
from claims.ml.cost_predictor import CostPredictor, CostPrediction
from claims.ml.anomaly_detector import AnomalyDetector
from claims.ml.risk_segmentation import RiskSegmentation
from claims.ml.approval_classifier import ApprovalClassifier

def test_cost_predictor_returns_correct_type():
    predictor = CostPredictor()
    # Fake minimal training state so we can call predict
    predictor.model = MagicMock()
    predictor.model.predict = MagicMock(return_value=np.array([9.5, 10.2]))
    predictor.model_name = "MockRidge"
    predictor._cost_percentiles = {"q25": 1000, "q75": 5000, "q90": 10000}
    
    # 2D dummy array
    dummy_X = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = predictor.predict(dummy_X)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for p in predictions:
        assert isinstance(p, CostPrediction)
        assert p.model_used == "MockRidge"

@pytest.mark.parametrize("cost,expected_tier", [
    (500.0, "low"),
    (3000.0, "medium"),
    (8000.0, "high"),
    (15000.0, "very_high")
])
def test_cost_predictor_cost_tier_assignment(cost, expected_tier):
    predictor = CostPredictor()
    predictor._cost_percentiles = {"q25": 1000.0, "q75": 5000.0, "q90": 10000.0}
    assert predictor._assign_cost_tier(cost) == expected_tier

def test_anomaly_detector_flags_outlier():
    # Normal training data
    np.random.seed(42)
    normal_data = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    
    detector = AnomalyDetector(contamination=0.05)
    detector.train(normal_data)
    
    # Extreme outlier
    outlier = np.array([[150.0, 90.0, 500000.0]])
    results = detector.predict(outlier)
    
    assert len(results) == 1
    assert bool(results[0].is_anomaly) is True

def test_anomaly_result_percentile_in_range():
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    detector = AnomalyDetector()
    detector.train(data)
    
    results = detector.predict(data)
    for r in results:
        assert 0.0 <= r.percentile_rank <= 100.0

def test_risk_segmentation_n_clusters():
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    model = RiskSegmentation(n_clusters=settings.N_CLUSTERS)
    
    df = pd.DataFrame(data, columns=["age", "bmi", "charges"])
    df["risk_score"] = 1.0
    df["children"] = 0
    
    model.train(data, df)
    
    assert len(model.cluster_sizes) == settings.N_CLUSTERS

def test_risk_segmentation_all_rows_assigned():
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))
    model = RiskSegmentation(n_clusters=settings.N_CLUSTERS)
    
    df = pd.DataFrame(data, columns=["age", "bmi", "charges"])
    df["risk_score"] = 1.0
    df["children"] = 0
    
    model.train(data, df)
    
    results = model.predict(data)
    for r in results:
        assert 0 <= r.cluster_id < settings.N_CLUSTERS

def test_approval_classifier_probability_range():
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    y = np.random.randint(0, 2, size=100)
    
    clf = ApprovalClassifier()
    clf.train(X, y, feature_names=["f1", "f2", "f3", "f4", "f5"])
    
    results = clf.predict(X)
    for r in results:
        assert 0.0 <= r.probability <= 1.0

def test_approval_classifier_returns_factors():
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    y = np.random.randint(0, 2, size=100)
    
    clf = ApprovalClassifier()
    clf.train(X, y, feature_names=["f1", "f2", "f3", "f4", "f5"])
    
    results = clf.predict(X[0:1])
    assert isinstance(results[0].decision_factors, list)
    assert len(results[0].decision_factors) >= 1
    for factor in results[0].decision_factors:
        assert isinstance(factor, str)
