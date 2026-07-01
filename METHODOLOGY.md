# Medical Claims Intelligence Methodology

This document outlines the modelling decisions, design choices, and evaluation metrics for the Medical Claims Intelligence platform.

---

## 1. Data Preprocessing Decisions

### Log-Transformation of Target Charges
Medical claim costs (`charges`) are heavily right-skewed, characterized by a large volume of low-cost routine claims and a thin, long tail of high-cost critical care claims. 
* **The Problem:** Standard regression algorithms (like Ridge Regression) assume a normal distribution of target residuals. Fitting them directly on highly skewed targets causes high-cost outliers to distort gradients, leading to poor predictions for standard claims.
* **The Solution:** We apply `np.log1p(charges)` to normalize the distribution before fitting.
  * **Before (Skewed):** Mean: ~$13,270, Median: ~$9,380, Max: ~$63,770.
  * **After (Normalized):** Converts the target into a symmetric Gaussian-like distribution, stabilizing prediction variance. At inference, predictions are mapped back using `np.expm1()`.

### StandardScaler for Distance-Based Algorithms
We use `StandardScaler` to normalize numeric features (`age`, `bmi`, `children`, and engineered interaction weights) to have a mean of $0$ and standard deviation of $1$.
* **The Reason:** K-Means clustering depends on Euclidean distance. Without scaling, features with large absolute ranges (e.g., claim charges of $1,000–$60,000) would completely dominate features with small ranges (e.g., children count of $0–5$), making distance calculations invalid.

### Interaction Terms (`age_smoker` and `bmi_smoker`)
Medical risk is non-linear and compound. For instance, being an elder or having a high BMI are moderate risk factors on their own, but *combining* them with active smoking drastically multiplies cardiovascular and respiratory claim risks. 
* We engineer `age_smoker` and `bmi_smoker` as explicit multiplicative features to capture these joint risks, helping linear estimators fit non-linear boundaries.

---

## 2. Cost Prediction — Model Selection

### Candidate Models
We compare three distinct regression models:
1. **Ridge Regression:** A linear model using L2 regularization to prevent coefficient scaling issues.
2. **Random Forest Regressor:** A bagging ensemble model that fits multiple decision trees in parallel.
3. **Gradient Boosting Regressor (GBM):** A boosting ensemble that builds decision trees sequentially to minimize residual errors.

### 5-Fold Cross-Validation
Instead of relying on a single train/test split (which can suffer from high variance based on how the data was split), we employ **5-fold cross-validation (CV)**. This splits the dataset into 5 equal subsets, training the model 5 times (using 4 folds for training and 1 fold for validation each time) and averaging the final $R^2$ scores. This yields a highly generalizable and reliable evaluation.

### Winner and Scores
* **Selected Model:** **Ridge Regression**
* **Score Metrics:** 
  * **Cross-Validation $R^2$:** `0.8089` (capturing over 80.8% of cost variance).
  * **Mean Absolute Error (MAE):** `$1,475.29` (averaging within ~$1,475 of actual costs).

---

## 3. Anomaly Detection — Isolation Forest

### Unsupervised Outlier Detection
We train an `Isolation Forest` on clustering features. Because historical data contains no pre-existing anomaly flags, supervised classification is impossible. Isolation Forest recursively partitions data; outliers are isolated closer to the root of the trees.

### Contamination Rate
We set `contamination = 0.05`, assuming a baseline rate of 5.0% for highly unusual or potentially fraudulent claims. This acts as a standard outlier threshold for human auditor review pipelines.

### Why not Autoencoders?
While deep learning autoencoders (neural networks trained to reconstruct inputs) are effective anomaly detectors, they are computational overkill for tabular data at this scale (1,338 records), require longer training cycles, and increase runtime deployment footprints without yielding higher precision.

---

## 4. Risk Segmentation — K-Means

### Optimal Clusters ($K=4$)
We selected $K=4$ clusters, which aligns cleanly with 4 distinct clinical risk categories (Low, Moderate, High, and Critical Risk). This decision was validated using:
* **Elbow Method:** Plotting inertias to spot the "inflection elbow" point.
* **Silhouette Analysis:** Yielding a robust silhouette score of `0.5841` (verifying strong partition cohesion).

### Why not Hierarchical Clustering?
Hierarchical clustering (agglomerative) has a computational complexity of $O(N^3)$ or $O(N^2 \log N)$, which does not scale well to large databases. K-Means operates at $O(K \cdot N \cdot I)$ and is extremely easy to deploy because cluster centroids are easily serialized and reused at prediction time.

### Cluster Label Assignment
Centroid profiles are evaluated after clustering. We sort the clusters by their mean charges and assign risk labels accordingly:
* **Lowest Mean Charge:** Low Risk
* **Second Lowest:** Moderate Risk
* **Third Lowest:** High Risk
* **Highest Mean Charge:** Critical Risk

---

## 5. ICD Classification — TF-IDF + Logistic Regression

### Pipeline Architecture
For predicting ICD-10 codes, we chain:
1. **TF-IDF Vectorizer:** Extracts uni-gram and bi-gram features (`ngram_range=(1,2)`), keeping up to 3,000 features. 
   * **Why bigrams?** Double-word tokens like `"blood pressure"` or `"body mass"` carry distinct clinical meaning that single words like `"blood"` or `"pressure"` do not.
2. **Logistic Regression:** A multinomial classifier trained to map clinical keywords to ICD-10 targets.

### Why not BERT?
Large Language Models like BERT or ClinicalBERT are powerful but present heavy trade-offs:
* **Model Size:** BERT models require 400MB+ of storage compared to Scikit-Learn's tiny size (~350KB).
* **Latency:** TF-IDF + Logistic Regression returns sub-millisecond inference times on standard CPUs, whereas BERT requires GPU nodes to maintain responsive API latency.
* **Accuracy:** On highly keyword-focused synthetic doctor templates, linear models reach an evaluation accuracy of `98.5%`—making deep learning models unnecessary for this scope.

### Production Upgrade Path
If the project scales to real-world free-text doctor charts containing complex grammar, negations, and typos, the recommended path is to fine-tune `dmis-lab/biobert-base-cased-v1.2` or `ClinicalBERT` on the clinical notes corpus.

---

## 6. A/B Testing Framework

### T-Test vs. Chi-Squared
* **Two-Sample T-Test:** Used in the framework to evaluate differences in continuous performance scores (e.g., MAE or cross-validation scores) between Model A and Model B.
* **Chi-Squared Test:** Recommended if comparing categorical proportions, such as comparing the claim approval/rejection rates between two models.

### Practical Statistical Significance
Setting $\alpha = 0.05$ ensures that there is less than a 5.0% probability that the performance difference occurred due to random chance. Cohen's d is calculated to evaluate the effect size, confirming that the improvement is practically meaningful.

### Sample Size Calculation
The minimum sample size needed to detect a target effect size $d$ at power $\beta=0.80$ and significance $\alpha=0.05$ is computed using normal distribution $z$-score bounds:
$$n = \left(\frac{z_{\alpha/2} + z_{\beta}}{d}\right)^2$$

---

## 7. Model Limitations

1. **Small Dataset (1,338 Rows):** The models are fitted on a small cohort and may fail to generalize to broader, diverse patient demographics.
2. **Synthetic Clinical Notes:** Clinical notes are generated from templates. Real-world medical text contains speech-to-text spelling errors, negation structures, and grammatical variations.
3. **Rule-Based Target Approval:** The approval classification targets were simulated using hard business rules (e.g., obese smokers). In production, this classifier must be trained on actual historical manual auditor approvals to capture complex human adjuster decisions.
