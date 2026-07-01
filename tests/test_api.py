import pytest
import httpx
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from fastapi import status
from claims.main import app
from claims.api.dependencies import get_registry

@pytest.fixture
def test_app(mock_registry):
    # Override registry dependency
    app.dependency_overrides[get_registry] = lambda: mock_registry
    yield app
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_health_endpoint_returns_200(test_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]

@pytest.mark.asyncio
async def test_predict_valid_request_returns_200(test_app, mock_registry):
    # Setup mocks for predict call
    mock_cost_info = MagicMock()
    mock_cost_info.predicted_cost = 5000.0
    mock_cost_info.ci_lower = 4000.0
    mock_cost_info.ci_upper = 6000.0
    mock_cost_info.cost_tier = "medium"
    mock_cost_info.model_used = "MockRegressor"
    mock_registry.cost_predictor.predict.return_value = [mock_cost_info]
    
    mock_anomaly_info = MagicMock()
    mock_anomaly_info.is_anomaly = False
    mock_anomaly_info.anomaly_score = 0.12
    mock_anomaly_info.percentile_rank = 15.0
    mock_anomaly_info.flag_reason = "Normal"
    mock_registry.anomaly_detector.predict.return_value = [mock_anomaly_info]
    
    mock_risk_info = MagicMock()
    mock_risk_info.cluster_id = 1
    mock_risk_info.risk_label = "Moderate Risk"
    mock_risk_info.risk_score = 4.5
    mock_risk_info.similar_members_count = 200
    mock_registry.risk_segmentation.predict.return_value = [mock_risk_info]
    
    mock_approval_info = MagicMock()
    mock_approval_info.approved = True
    mock_approval_info.probability = 0.85
    mock_approval_info.confidence = "high"
    mock_approval_info.decision_factors = ["smoker", "bmi"]
    mock_registry.approval_classifier.predict.return_value = [mock_approval_info]
    
    payload = {
        "age": 45,
        "gender": "Male",
        "bmi": 28.5,
        "children": 1,
        "smoker": False,
        "region": "Southeast"
    }
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post("/api/claims/predict", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    assert "claim_id" in data
    assert "cost_prediction" in data
    assert "anomaly_detection" in data
    assert "risk_segmentation" in data
    assert "approval_status" in data

@pytest.mark.asyncio
async def test_predict_invalid_age_returns_422(test_app):
    payload = {
        "age": -5,
        "gender": "Male",
        "bmi": 28.5,
        "children": 1,
        "smoker": False,
        "region": "Southeast"
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post("/api/claims/predict", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_invalid_bmi_returns_422(test_app):
    payload = {
        "age": 45,
        "gender": "Male",
        "bmi": 200.0,
        "children": 1,
        "smoker": False,
        "region": "Southeast"
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post("/api/claims/predict", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_missing_field_returns_422(test_app):
    # Missing 'gender'
    payload = {
        "age": 45,
        "bmi": 28.5,
        "children": 1,
        "smoker": False,
        "region": "Southeast"
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post("/api/claims/predict", json=payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_with_clinical_note(test_app, mock_registry):
    # Setup mocks
    mock_cost_info = MagicMock()
    mock_cost_info.predicted_cost = 5000.0
    mock_cost_info.ci_lower = 4000.0
    mock_cost_info.ci_upper = 6000.0
    mock_cost_info.cost_tier = "medium"
    mock_cost_info.model_used = "MockRegressor"
    mock_registry.cost_predictor.predict.return_value = [mock_cost_info]
    
    mock_anomaly_info = MagicMock()
    mock_anomaly_info.is_anomaly = False
    mock_anomaly_info.anomaly_score = 0.12
    mock_anomaly_info.percentile_rank = 15.0
    mock_anomaly_info.flag_reason = "Normal"
    mock_registry.anomaly_detector.predict.return_value = [mock_anomaly_info]
    
    mock_risk_info = MagicMock()
    mock_risk_info.cluster_id = 1
    mock_risk_info.risk_label = "Moderate Risk"
    mock_risk_info.risk_score = 4.5
    mock_risk_info.similar_members_count = 200
    mock_registry.risk_segmentation.predict.return_value = [mock_risk_info]
    
    mock_approval_info = MagicMock()
    mock_approval_info.approved = True
    mock_approval_info.probability = 0.85
    mock_approval_info.confidence = "high"
    mock_approval_info.decision_factors = ["smoker", "bmi"]
    mock_registry.approval_classifier.predict.return_value = [mock_approval_info]
    
    mock_registry.entity_extractor.extract.return_value = MagicMock(
        conditions=["diabetes"], medications=["metformin"], procedures=["ecg"], vitals={}, clinical_summary="Summary"
    )
    mock_registry.icd_classifier.predict.return_value = MagicMock(
        icd_code="E11", description="Diabetes mellitus", confidence=0.92
    )
    mock_registry.note_processor.get_urgency.return_value = "routine"
    mock_registry.note_processor.extract_keywords.return_value = ["diabetes", "metformin"]
    
    payload = {
        "age": 45,
        "gender": "Male",
        "bmi": 28.5,
        "children": 1,
        "smoker": False,
        "region": "Southeast",
        "clinical_note": "Patient has diabetes and is taking Metformin."
    }
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post("/api/claims/predict", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    assert data["nlp_analysis"] is not None

@pytest.mark.asyncio
async def test_analytics_segments_returns_dict(test_app, mock_registry):
    # Setup mocks for segments endpoint
    mock_registry.preprocessor.transform.return_value = MagicMock(
        X_clustering=np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    mock_registry.risk_segmentation.predict.return_value = [
        MagicMock(cluster_id=0, risk_label="Low Risk"),
        MagicMock(cluster_id=1, risk_label="Moderate Risk")
    ]
    mock_registry.risk_segmentation.get_cluster_profiles.return_value = pd.DataFrame(
        [[45.0, 28.0, 5000.0]], columns=["age", "bmi", "charges"]
    )
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/api/analytics/segments")
        
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert isinstance(data["segments"], dict)

@pytest.mark.asyncio
async def test_model_performance_endpoint(test_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/api/analytics/model-performance")
    assert response.status_code == 200
    data = response.json()
    assert "cost_predictor" in data
