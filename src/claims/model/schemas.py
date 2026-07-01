from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

#  Request Models
class ClaimRequest(BaseModel):
    age: int = Field(..., gt=0, lt=120, description = "Patient age")
    gender: Literal["Male", "Female"] = Field(..., description = "Patient gender")
    bmi: float = Field(..., ge=10.0, le=70.0, description = "Patient BMI")
    children: int = Field(..., ge=0, le=10, description = "Number of children")
    smoker: bool = Field(..., description = "Patient smoker")
    region: Literal["Northeast", "Southeast", "Northwest", "Southwest"] = Field(..., description = "Patient region")
    clinical_note: Optional[str] = Field(None, description = "Patient clinical note")

    @field_validator("region", mode="before")
    @classmethod
    def clean_region(cls, v: str) -> str:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            mapping = {
                "northeast": "Northeast",
                "northwest": "Northwest",
                "southeast": "Southeast",
                "southwest": "Southwest"
            }
            return mapping.get(cleaned, v)
        return v

    @field_validator("gender", mode="before")
    @classmethod
    def clean_gender(cls, v: str) -> str:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            mapping = {
                "male": "Male",
                "female": "Female"
            }
            return mapping.get(cleaned, v)
        return v
    
    
#  Response Models
class CostPredictionResponse(BaseModel):
    predicted_cost: float = Field(..., ge=0, description = "Predicted medical cost")
    ci_lower: float = Field(..., ge=0, description = "Confidence interval lower bound")
    ci_upper: float = Field(..., ge=0, description = "Confidence interval upper bound")
    cost_tier: str = Field(..., description = "Cost tier")
    model_used: str = Field(..., description = "Model used for prediction")
    
class AnomalyResponse(BaseModel):
    is_anomaly: bool = Field(..., description = "Is the claim an anomaly")
    anomaly_score: float = Field(..., description = "Anomaly score")
    percentile_rank: float = Field(..., description = "Percentile rank")
    flag_reason: str = Field(..., description = "Reason for anomaly flagging")
    
class RiskSegmentResponse(BaseModel):
    cluster_id: int = Field(..., description = "Cluster ID")
    risk_label: str = Field(..., description = "Risk label")
    risk_score: float = Field(..., description = "Risk score")
    similar_members_count: int = Field(..., description = "Number of similar members")

class ApprovalResponse(BaseModel):
    approved: bool = Field(..., description = "Approval status")
    probability: float = Field(..., description = "Approval probability")
    confidence: str = Field(..., description = "Approval confidence")
    decision_factors: list[str] = Field(..., description = "Decision factors")

class NLPResponse(BaseModel):
    extracted_conditions: list[str] = Field(..., description = "")
    extracted_medications: list[str] = Field(..., description = "")
    extracted_procedures: list[str] = Field(..., description = "")
    vitals: dict = Field(..., description = "")
    predicted_icd: str = Field(..., description = "")
    icd_description: str = Field(..., description = "")
    icd_confidence: float = Field(..., description = "")
    note_urgency: str = Field(..., description = "")
    clinical_summary: str = Field(..., description = "")
    keywords: list[str] = Field(..., description = "")

class FullClaimResponse(BaseModel):
    claim_id: str = Field(..., description = "Unique claim identifier")
    cost_prediction: CostPredictionResponse
    anomaly_detection: AnomalyResponse
    risk_segmentation: RiskSegmentResponse
    approval_status: ApprovalResponse
    nlp_analysis: Optional[NLPResponse] = None
    processing_time_ms: float = Field(..., description = "Processing time in ms")
    timestamp: str = Field(..., description = "Timestamp of the analysis")