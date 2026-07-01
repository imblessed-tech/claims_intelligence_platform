# Health Insurance Claims Intelligence Platform

Health insurance companies process thousands of claims daily. Each claim involves a member's medical profile, a doctor's clinical note, and a charged amount. Reviewing these manually is slow, expensive, and inconsistent.

This platform automates the intelligence layer of that process. Given a claim submission, it:
* Predicts the expected claim cost and flags if the actual amount is anomalous.
* Decides whether the claim meets approval criteria based on learned patterns.
* Segments the member into a risk tier (Low / Moderate / High / Critical).
* Reads the doctor's clinical note and extracts conditions, medications, vitals, and a predicted ICD-10 diagnosis code.
* Exposes everything through a single REST API call that returns a unified JSON response in under 200ms.

The platform is designed to the standards expected in a production health-tech environment — typed, tested, documented, and deployable.

---

## Table of Contents
* [Overview](#overview)
* [Architecture](#architecture)
* [Tech Stack](#tech-stack)
* [Features](#features)
* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Training the Models](#training-the-models)
* [Running the API](#running-the-api)
* [API Reference](#api-reference)
* [Running Tests](#running-tests)
* [Dashboard](#dashboard)
* [Deployment](#deployment)
* [Model Performance](#model-performance)
* [Methodology](#methodology)
* [Known Limitations](#known-limitations)
* [Author](#author)

---

## Overview
(covered above)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT / DASHBOARD                      │
│              (Browser, curl, Postman, HTML report)          │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI REST API                          │
│         Request logging │ Timing │ CORS │ Validation        │
│              (src/claims/main.py + middleware)               │
└──────┬──────────────┬───────────────┬───────────────────────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐ ┌────────────┐ ┌─────────────────┐
│  /predict  │ │ /analytics │ │    /health       │
│  /batch    │ │ /segments  │ │    /metrics      │
└─────┬──────┘ └────────────┘ └─────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                  DEPENDENCY LAYER                           │
│          Models loaded ONCE at startup, injected            │
│              into every route via Depends()                 │
└──────┬────────────┬────────────┬────────────┬──────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   Cost   │ │ Anomaly  │ │  Risk    │ │ Approval │
│Predictor │ │Detector  │ │ Segment  │ │Classifier│
│  Ridge   │ │Iso Forest│ │ K-Means  │ │  RF + BW │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │         NLP PIPELINE         │
       │  (only if clinical_note      │
       │   is provided in request)    │
       └───────┬──────────┬───────────┘
               │          │
               ▼          ▼
       ┌──────────┐ ┌──────────────┐
       │ Medical  │ │ICD-10 Code   │
       │ Entity   │ │Classifier    │
       │Extractor │ │TF-IDF + LR   │
       └──────────┘ └──────────────┘
```

### Data Flow for a Single Prediction Request:
1. Client makes a `POST /api/claims/predict` call.
2. FastAPI runs Pydantic validation (returns 422 if inputs violate ranges or schema).
3. Preprocessor transforms inputs (standardizing numeric variables and mapping categoricals).
4. Models run in parallel:
   * `CostPredictor` predicts continuous USD charges and assigns a cost tier.
   * `AnomalyDetector` scores feature outlier likelihood.
   * `RiskSegmentation` clusters demographic profiles.
   * `ApprovalClassifier` yields approval decisions.
5. If `clinical_note` is present:
   * Note text is normalized and cleaned.
   * `EntityExtractor` parses entities (medications, conditions, vitals, procedures) using compiled regex patterns and spaCy.
   * `ICDClassifier` extracts TF-IDF bigram features and runs multinomial Logistic Regression to suggest the most appropriate ICD-10 code.
6. The router returns a unified `FullClaimResponse` JSON.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.110+ | Async REST API with auto-generated docs |
| **Data** | pandas 2.0, numpy 1.26 | Data manipulation and feature engineering |
| **ML Models** | scikit-learn 1.4 | Ridge, Random Forest, K-Means, Isolation Forest, Logistic Regression |
| **Model Persistence** | joblib | Serialise and load trained models |
| **NLP** | spaCy 3.7, TF-IDF | Entity extraction, text classification |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **Configuration** | pydantic-settings | Environment-based config management |
| **Testing** | pytest, httpx | Unit and integration testing |
| **Statistics** | scipy | A/B testing, drift detection |
| **Visualisation** | matplotlib | Dashboard chart generation |
| **Deployment** | Docker, Render | Containerisation and cloud hosting |
| **Language** | Python 3.11 | Core runtime |

---

## Features

### Machine Learning Pipeline
* **Cost Predictor:** Ridge Regression model to predict claim costs in USD (MAE, $R^2$).
* **Anomaly Detector:** Unsupervised Isolation Forest model flagging suspicious claims (Anomaly rate).
* **Risk Segmentation:** K-Means clustering ($K=4$) grouping members by health profile (Silhouette score).
* **Approval Classifier:** Random Forest classification predicting claim approvals (ROC-AUC, F1).

### NLP Pipeline
* **Note Cleaner:** Regex-based abbreviation expansions and normalization.
* **Entity Extractor:** Structured parsing of medications, conditions, vitals, and procedures.
* **ICD Classifier:** TF-IDF + Logistic Regression predicting ICD-10 codes with confidence scores.
* **Topic Modelling:** Latent Dirichlet Allocation (5 topics) identifying dominant clinical themes.
* **Urgency Scorer:** Pattern matching for "urgent", "moderate", or "routine" clinical states.

### API Capabilities
* Single and batch prediction endpoints.
* Analytics endpoints — segment distribution, anomaly summary, model performance.
* Service health checks and in-memory Prometheus-style metrics tracking.
* Auto-generated Swagger documentation at `/docs` and `/redoc`.

### Supporting Infrastructure
* A/B testing framework with t-test and Cohen's d effect size calculations.
* Data drift detection using Kolmogorov-Smirnov test.
* Request logging and timing tracking via middleware.

---

## Project Structure

```
claims_intelligence/
│
├── src/claims/
│   ├── config.py                  ← all settings and paths
│   ├── main.py                    ← FastAPI app entry point
│   │
│   ├── data/
│   │   ├── loader.py              ← CSV loading and validation
│   │   ├── synthesizer.py         ← clinical note generation
│   │   └── preprocessor.py       ← feature engineering and scaling
│   │
│   ├── ml/
│   │   ├── cost_predictor.py      ← Ridge regression
│   │   ├── anomaly_detector.py    ← Isolation Forest
│   │   ├── risk_segmentation.py   ← K-Means clustering
│   │   └── approval_classifier.py ← Random Forest classification
│   │
│   ├── nlp/
│   │   ├── note_processor.py      ← text cleaning, TF-IDF, LDA
│   │   ├── entity_extractor.py    ← regex + spaCy NER
│   │   └── icd_classifier.py      ← ICD-10 text classification
│   │
│   ├── api/
│   │   ├── dependencies.py        ← model registry + DI
│   │   ├── middleware.py          ← request logging + timing
│   │   └── routes/
│   │       ├── claims.py          ← predict endpoints
│   │       ├── analytics.py       ← aggregate stats endpoints
│   │       └── health.py          ← health + metrics endpoints
│   │
│   ├── experimentation/
│   │   └── ab_testing.py          ← statistical A/B framework
│   │
│   ├── monitoring/
│   │   ├── model_monitor.py       ← drift detection
│   │   └── metrics.py             ← in-memory counters
│   │
│   └── visualization/
│       └── dashboard.py           ← HTML report generator
│
├── data/
│   └── insurance.csv              ← source dataset
│
├── models/                        ← generated by train.py
├── reports/dashboards/            ← generated HTML report
├── notebooks/
│   └── 01_full_analysis.ipynb    ← stakeholder EDA notebook
├── tests/
│   ├── conftest.py
│   ├── test_ml.py
│   ├── test_nlp.py
│   └── test_api.py
│
├── train.py                       ← full training pipeline
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── METHODOLOGY.md
└── README.md
```

---

## Setup & Installation

### Prerequisites
* Python 3.10 or higher
* pip
* Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/claims-intelligence.git
cd claims-intelligence
```

### Step 2 — Create and activate a virtual environment
```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download the spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Download the dataset
Download the insurance dataset from either source:

**Option A — Direct download (no login required):**
```bash
curl -o data/insurance.csv https://raw.githubusercontent.com/dsrscientist/dataset1/master/medical_cost.csv
```

**Option B — Kaggle:**
Visit https://www.kaggle.com/datasets/easonlai/sample-insurance-claim-prediction-dataset and download insurance.csv. Place it in the `data/` folder.

### Step 6 — Configure environment
```bash
copy .env.example .env
```
The default `.env` values work for local development. No changes needed unless deploying.

### Step 7 — Verify setup
```bash
python -c "
from src.claims.config import settings
print('BASE_DIR:', settings.BASE_DIR)
print('DATA_FILE exists:', settings.RAW_DATA_FILE.exists())
"
```
You should see `DATA_FILE exists: True`.

---

## Training the Models

Run the full training pipeline with one command:
```bash
python train.py
```
This will:
* Load and validate `data/insurance.csv`.
* Generate synthetic clinical notes → saves `data/claims_with_notes.csv`.
* Engineer features and fit the preprocessor.
* Train and tune the Cost Predictor (GBM vs Random Forest vs Ridge comparison).
* Train the Anomaly Detector.
* Find optimal K and train Risk Segmentation.
* Train the Approval Classifier.
* Train the NLP Note Processor and ICD Classifier.
* Generate the HTML dashboard report.
* Save all models to `models/`.

---

## Running the API

### Local development
```bash
uvicorn src.claims.main:app --reload --host 0.0.0.0 --port 8000
```
The `--reload` flag restarts the server automatically when you change code. Remove it in production.

### Verify it is running
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-06-06T10:30:00"
}
```

### Interactive API documentation
Open in your browser:
* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

---

## API Reference

### `POST /api/claims/predict`
Accepts a claim and returns a full AI analysis.

**Testing via CLI:**
* **Linux / macOS (Bash / curl):**
  ```bash
  curl -X POST "https://claims-intelligence.onrender.com/claims/predict" \
    -H "Content-Type: application/json" \
    -d '{"age": 40, "gender": "Male", "bmi": 28.5, "children": 2, "smoker": true, "region": "Southwest", "clinical_note": "Patient presents with chest tightness."}'
  ```
* **Windows (PowerShell):**
  ```powershell
  curl -UseBasicParsing -Uri "https://claims-intelligence.onrender.com/claims/predict" `
    -Method POST `
    -Headers @{"Content-Type" = "application/json"} `
    -Body '{"age": 40, "gender": "Male", "bmi": 28.5, "children": 2, "smoker": true, "region": "Southwest", "clinical_note": "Patient presents with chest tightness."}'
  ```

**Request body:**
```json
{
  "age": 45,
  "sex": "male",
  "bmi": 32.5,
  "children": 2,
  "smoker": true,
  "region": "southwest",
  "clinical_note": "Patient presents with Type 2 Diabetes, poorly controlled. BMI 32.5. HbA1c 9.2%. Metformin dosage adjusted."
}
```

**Field validation:**

| Field | Type | Constraints |
|---|---|---|
| age | integer | 0 – 120 |
| sex | string | "male" or "female" |
| bmi | float | 10.0 – 70.0 |
| children | integer | 0 – 10 |
| smoker | boolean | true / false |
| region | string | "northeast", "northwest", "southeast", "southwest" |
| clinical_note | string | Optional |

**Response:**
```json
{
  "claim_id": "CLM-A3F9C2D1",
  "cost_prediction": {
    "predicted_cost": 18420.50,
    "ci_lower": 15104.81,
    "ci_upper": 21736.19,
    "cost_tier": "high",
    "model_used": "Ridge"
  },
  "anomaly": {
    "is_anomaly": false,
    "anomaly_score": 0.0842,
    "percentile_rank": 73.2,
    "flag_reason": "Normal claim pattern"
  },
  "risk_segment": {
    "cluster_id": 2,
    "risk_label": "High Risk",
    "risk_score": 6.7,
    "similar_members_count": 312
  },
  "approval": {
    "approved": true,
    "probability": 0.8876,
    "confidence": "high",
    "decision_factors": ["risk_score", "smoker", "age"]
  },
  "nlp_analysis": {
    "extracted_conditions": ["diabetes"],
    "extracted_medications": ["metformin"],
    "extracted_procedures": [],
    "vitals": {
      "bmi": 32.5,
      "hba1c": 9.2
    },
    "predicted_icd": "E11",
    "icd_description": "Type 2 Diabetes Mellitus",
    "icd_confidence": 0.9341,
    "note_urgency": "moderate",
    "clinical_summary": "Patient with diabetes prescribed metformin. BMI 32.5, HbA1c 9.2%.",
    "keywords": ["diabetes", "hba1c", "metformin", "bmi", "blood sugar"]
  },
  "processing_time_ms": 43.7,
  "timestamp": "2026-06-06T10:31:22"
}
```

### `POST /api/claims/predict/batch`
Batch prediction for up to 100 claims.

**Request body:** Array of `ClaimRequest` objects (same schema as above).

```bash
curl -X POST http://localhost:8000/api/claims/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": true, "region": "southwest"},
    {"age": 28, "sex": "female", "bmi": 22.1, "children": 0, "smoker": false, "region": "northeast"}
  ]'
```

**Response:** Array of `FullClaimResponse` objects.

### `GET /api/analytics/segments`
Returns current member risk segment distribution.

```json
{
  "segments": {
    "counts": {
      "Low Risk": 412,
      "Moderate Risk": 389,
      "High Risk": 298,
      "Critical Risk": 239
    },
    "percentages": {
      "Low Risk": 30.8,
      "Moderate Risk": 29.1,
      "High Risk": 22.3,
      "Critical Risk": 17.9
    },
    "profiles": {
      "0": {"age": 26.1, "bmi": 23.4, "charges": 3241.50},
      "1": {"age": 38.7, "bmi": 27.8, "charges": 8102.30},
      "2": {"age": 45.2, "bmi": 33.1, "charges": 14820.70},
      "3": {"age": 52.8, "bmi": 36.4, "charges": 32104.80}
    }
  }
}
```

### `GET /api/analytics/anomalies`
Returns summary of anomalous claims in the dataset.

```json
{
  "total_claims": 1338,
  "anomalies_detected": 67,
  "anomaly_rate": 0.05,
  "score_distribution": {
    "mean": -0.0412,
    "std": 0.1823,
    "p5": -0.3241,
    "p95": 0.1204
  }
}
```

### `GET /api/analytics/model-performance`
Returns model evaluation metrics from the last training run.

```json
{
  "cost_predictor": {
    "model": "Ridge Regression",
    "mae": 1475.29,
    "r2": 0.8089
  },
  "approval_classifier": {
    "roc_auc": 0.9848,
    "f1": 0.9851
  },
  "anomaly_detector": {
    "contamination": 0.05,
    "anomalies_flagged": 67
  },
  "risk_segmentation": {
    "n_clusters": 4,
    "silhouette_score": 0.1550
  },
  "icd_classifier": {
    "accuracy": 0.9850,
    "f1_weighted": 0.9850
  }
}
```

### `GET /health`
Service health check. Always returns 200.

### `GET /metrics`
In-memory request metrics since startup.

---

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage report:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Dashboard

After running `train.py`, open the generated HTML report in any browser:
```bash
# Windows
start reports/dashboards/claims_intelligence_report.html
```

The report contains 6 sections:

| Section | Content |
|---|---|
| **Key Metrics** | Total claims, approval rate, avg cost, anomaly rate |
| **Cost Analysis** | Claim cost distribution, top cost drivers (feature weights) |
| **Risk Segmentation** | Cluster sizes, member profiles per risk tier (heatmap) |
| **Approval Analysis** | Approval rate by region, approved vs rejected breakdown |
| **Clinical Intelligence** | ICD-10 code distribution from NLP analysis |
| **Model Performance** | All model evaluation metrics in one view |

The file is entirely self-contained — charts are embedded as base64 images.

---

## Deployment & Live Links

The platform is configured for production-grade deployments and is live for user testing:

* **Production Backend (Render):** The FastAPI REST API is containerized using Docker and hosted on Render. Render automatically manages build hooks and handles traffic routing to the internal Uvicorn interfaces.
* **Interactive Analytics Dashboard (GitHub Pages):** The self-contained HTML analytics dashboard is compiled dynamically by the model training pipeline and served as a static site via GitHub Pages.

### Live URLs for Testing

> [!NOTE]
> **Free Tier Spin-Down Delay:** The API is hosted on Render's free tier. If the service is inactive for 15 minutes, it automatically spins down. The first request after a sleep period will experience a **50–60 second "cold start" delay** while the container spins up and loads ML/NLP models into memory. Subsequent requests will execute at normal sub-200ms latency.

* **Interactive Swagger API Documentation:** [claims-intelligence.onrender.com/docs](https://claims-intelligence.onrender.com/docs)
* **API Telemetry Metrics:** [claims-intelligence.onrender.com/metrics](https://claims-intelligence.onrender.com/metrics)
* **Static Visual Analytics Dashboard:** [imblessed-tech.github.io/claims_intelligence_platform/](https://imblessed-tech.github.io/claims_intelligence_platform/)
* **Source Code Repository:** [github.com/imblessed-tech/claims_intelligence_platform](https://github.com/imblessed-tech/claims_intelligence_platform)

---

## Model Performance

| Model | Algorithm | Metric | Value |
|---|---|---|---|
| **Cost Predictor** | Ridge Regression | MAE | `$1,475.29` |
| **Cost Predictor** | Ridge Regression | $R^2$ | `0.8089` |
| **Cost Predictor** | Ridge Regression | CV $R^2$ (5-fold) | `0.8089 ± 0.0355` |
| **Anomaly Detector** | Isolation Forest | Anomaly Rate | `5.0%` (67 claims) |
| **Risk Segmentation** | K-Means (K=4) | Silhouette Score | `0.1550` |
| **Approval Classifier** | Random Forest | ROC-AUC | `0.9848` |
| **Approval Classifier** | Random Forest | F1 Score | `0.9851` |
| **ICD Classifier** | TF-IDF + Logistic Regression | Accuracy | `0.9850` |
| **ICD Classifier** | TF-IDF + Logistic Regression | Weighted F1 | `0.9850` |

---

## Methodology

* **Why log-transform charges?** Claim costs are right-skewed. Log transform compresses high-cost outliers and produces a normally distributed target, which helps gradient-based and linear models converge.
* **Why Ridge Regression for cost prediction?** Ridge handles multi-collinear dummy features via L2 regularization, providing a highly stable linear estimator that outperformed standard boosting/bagging ensembles during cross-validation on this dataset.
* **Why Isolation Forest for anomaly detection?** Isolation Forest acts as an unsupervised classifier, partition-isolating outliers on tabular data without requiring pre-existing anomaly labels.
* **Why K=4 for clustering?** Validated by average silhouette score analysis (inflection at K=4) and maps naturally to clinically meaningful risk tiers.
* **Why TF-IDF + Logistic Regression for ICD classification?** TF-IDF bigrams capture context (like `"blood pressure"`) and Logistic Regression yields sub-millisecond CPU inference latency, achieving `98.5%` accuracy on this sample size.

---

## Known Limitations
* **Small dataset (1,338 rows)** — models may not generalise perfectly to real-world claim distributions.
* **Synthetic clinical notes** — trained on template-generated text. Real notes contain transcriber shorthand and complex negation.
* **Rule-based targets** — approval labels are derived from hard rules rather than capturing manual auditor variances.
* **In-memory metrics** — request metrics reset on container restart.
* **No authentication** — the endpoints have no API key or OAuth protection.

---

## Author
* **GitHub:** [github.com/imblessed-tech](https://github.com/imblessed-tech)

---
Built with Python 3.11 · FastAPI · scikit-learn · spaCy · pandas · matplotlib
