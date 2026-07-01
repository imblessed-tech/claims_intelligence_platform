import uuid
import time
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from claims.api.dependencies import get_registry, ModelRegistry
from claims.monitoring.metrics import increment
from claims.model.schemas import (
    ClaimRequest,
    FullClaimResponse,
    CostPredictionResponse,
    AnomalyResponse,
    RiskSegmentResponse,
    ApprovalResponse,
    NLPResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/claims",
    tags=["claims"]
)

@router.post("/predict", response_model=FullClaimResponse)
async def predict_claim(
    claim_request: ClaimRequest,
    registry: ModelRegistry = Depends(get_registry)
):
    start = time.perf_counter()
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    claim_request_dict = claim_request.model_dump()
    claim_request_dict['smoker'] = "yes" if claim_request_dict['smoker'] else "no"
    claim_request_dict['region'] = claim_request_dict['region'].strip()
    if 'gender' in claim_request_dict:
        claim_request_dict['sex'] = claim_request_dict.pop('gender')

    # Convert to DataFrame for consistent preprocessing (requires 2D input)
    input_df = pd.DataFrame([claim_request_dict])

    try:
        processed_data = registry.preprocessor.transform(input_df)

        # Step 1: Cost Prediction (regression)
        cost_results = registry.cost_predictor.predict(processed_data.X_regression[0:1])
        cost_info = cost_results[0]

        # Step 2: Anomaly Detection
        anomaly_results = registry.anomaly_detector.predict(processed_data.X_clustering[0:1])
        anomaly_info = anomaly_results[0]

        # Step 3: Risk Segmentation
        risk_results = registry.risk_segmentation.predict(processed_data.X_clustering[0:1])
        risk_info = risk_results[0]

        # Step 4: Approval Classification
        approval_results = registry.approval_classifier.predict(processed_data.X_classification[0:1])
        approval_info = approval_results[0]

        # Step 5: NLP Analysis (ICD + Note Processing)
        nlp_analysis = None
        if claim_request.clinical_note:
            entity_extract_info = registry.entity_extractor.extract(claim_request.clinical_note)
            icd_pred_info = registry.icd_classifier.predict(claim_request.clinical_note)
            urgency = registry.note_processor.get_urgency(claim_request.clinical_note)
            keywords = registry.note_processor.extract_keywords(claim_request.clinical_note)

            nlp_analysis = NLPResponse(
                extracted_conditions=entity_extract_info.conditions,
                extracted_medications=entity_extract_info.medications,
                extracted_procedures=entity_extract_info.procedures,
                vitals=entity_extract_info.vitals,
                predicted_icd=icd_pred_info.icd_code,
                icd_description=icd_pred_info.description,
                icd_confidence=icd_pred_info.confidence,
                note_urgency=urgency,
                clinical_summary=entity_extract_info.clinical_summary,
                keywords=keywords,
            )

        # Step 6: Calculate processing time
        processing_time_ms = (time.perf_counter() - start) * 1000

        # Step 7: Construct full response
        response = FullClaimResponse(
            claim_id=claim_id,
            cost_prediction=CostPredictionResponse(
                predicted_cost=cost_info.predicted_cost,
                ci_lower=cost_info.ci_lower,
                ci_upper=cost_info.ci_upper,
                cost_tier=cost_info.cost_tier,
                model_used=cost_info.model_used,
            ),
            anomaly_detection=AnomalyResponse(
                is_anomaly=anomaly_info.is_anomaly,
                anomaly_score=anomaly_info.anomaly_score,
                percentile_rank=anomaly_info.percentile_rank,
                flag_reason=anomaly_info.flag_reason,
            ),
            risk_segmentation=RiskSegmentResponse(
                cluster_id=risk_info.cluster_id,
                risk_label=risk_info.risk_label,
                risk_score=risk_info.risk_score,
                similar_members_count=risk_info.similar_members_count,
            ),
            approval_status=ApprovalResponse(
                approved=approval_info.approved,
                probability=approval_info.probability,
                confidence=approval_info.confidence,
                decision_factors=approval_info.decision_factors,
            ),
            nlp_analysis=nlp_analysis,
            processing_time_ms=processing_time_ms,
            timestamp=pd.Timestamp.now().isoformat(),
        )

        # Update central metrics
        increment("predictions_total", 1)
        increment("total_processing_ms", processing_time_ms)
        if claim_request.clinical_note:
            increment("nlp_analyses_total", 1)
        if anomaly_info.is_anomaly:
            increment("anomalies_flagged", 1)

        logger.info(f"[{claim_id}] Claim processed in {processing_time_ms:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Error processing prediction for claim: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model prediction pipeline: {str(e)}"
        )
