# Scholarship Eligibility Prediction System

[![Course](https://img.shields.io/badge/Course-AD4V71%20%2F%20AD5302%20Data%20Science-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

---

## Academic Metadata

- **Project Title:** Scholarship Eligibility Prediction
- **Project Type:** Academic machine-learning classification project
- **Course Code:** AD4V71 / AD5302 – Data Science (PBL Reviews 1 & 2)
- **Project Team:**
  - **Kirutick Siddhesh V** (Regd No.: `210425243120`, Section: C)
  - **Krishna H** (Regd No.: `210425243124`, Section: C)

---

## 1. Project Overview & Scope Definition

### The Academic Problem
Scholarship screening across higher education institutions often requires evaluating student applications against multi-tiered eligibility criteria (such as income limits, reservation categories, first-generation graduate status, and academic cutoffs). In a manual workflow, evaluating multiple criteria across large applicant pools can be repetitive and time-consuming.

### Project Objective
To develop an academic machine-learning classification pipeline that learns patterns from student attribute data to predict whether a student application is predicted as **Eligible** or **Not Eligible** for scholarship screening.

### Explicit Scope Boundaries
- **IN SCOPE:**
  - Data preprocessing, exploratory data analysis, and feature engineering.
  - Multi-model classification benchmarking (Logistic Regression, Decision Tree, Random Forest, Naïve Bayes).
  - Model diagnostic evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Gini Feature Importance).
  - Decision-support prediction interface with probability estimation.
- **OUT OF SCOPE:**
  - **Fund Disbursement:** The system does not transfer money or manage bank disbursements.
  - **Document Authentication:** The system does not inspect or verify physical certificates.
  - **Official Legal Determination:** The system is an academic machine-learning demonstration and does NOT determine actual statutory or governmental scholarship eligibility.

---

## 2. Dataset Transparency & Provenance

### Dataset Origin Statement
- **Dataset Nature:** **Synthetic reference dataset** generated specifically for academic machine-learning experimentation.
- **Reference Inspiration:** General Indian higher-education demographics and published scholarship guidelines (e.g., AISHE demographic categories, affirmative reservation classifications, means-testing thresholds) served as conceptual inspiration.
- **Generation Logic:** Synthesized deterministically using `scripts/generate_reference_dataset.py` (seed = 42).
- **Label Generation:** Labels were assigned using multi-variable threshold logic. Random label variation was intentionally introduced during synthetic generation to avoid creating a perfectly deterministic, trivial target.
- **Privacy & Ground Truth:** No actual government student records, confidential citizen data, or live institutional files were downloaded, leaked, or used.

---

## 3. Dataset Schema & Attributes

The benchmark dataset ([data/raw/scholarship_data.csv](file:///e:/DS/data/raw/scholarship_data.csv)) contains **1,000 records** and **10 columns**:

| Column Name | Data Type | Kind | Description & Valid Values | Preprocessing Handling |
| :--- | :--- | :--- | :--- | :--- |
| `StudentID` | String | Identifier | Unique applicant ID (`STU1001` to `STU2000`) | Dropped prior to modeling |
| `Gender` | String | Categorical | `Male`, `Female` | One-Hot Encoded |
| `Community` | String | Categorical | `SC`, `ST`, `MBC`, `BC`, `OC` | One-Hot Encoded |
| `FamilyIncome` | Integer | Numerical | Annual household income in INR (₹45,000 to ₹750,000) | StandardScaled |
| `12thMarks` | Float | Numerical | Higher Secondary Board percentage (48.0% to 99.0%) | StandardScaled |
| `FirstGraduate` | String | Categorical | First graduate in family (`Yes`, `No`) | One-Hot Encoded |
| `District` | String | Categorical | Native district in Tamil Nadu (10 districts) | One-Hot Encoded |
| `CollegeType` | String | Categorical | `Government`, `Government Aided`, `Private` | One-Hot Encoded |
| `Course` | String | Categorical | Degree stream (`Engineering`, `Arts & Science`, etc.) | One-Hot Encoded |
| **`Eligibility`** | String | **Target** | Target class: **`Eligible`** or **`Not Eligible`** | Binary Classification Target |

### Verified Target Distribution:
- **Eligible:** 638 records (**63.8%**)
- **Not Eligible:** 362 records (**36.2%**)
- **Target Characteristic:** Moderately imbalanced (handled via stratified train/test split and F1-score evaluation).

---

## 4. Architecture & Pipeline

```
+-------------------------------------------------------------------------+
|                  SYNTHETIC REFERENCE STUDENT DATASET                    |
|                        (1,000 Student Records)                          |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|               DATA PREPROCESSING & STRATIFIED SPLIT                     |
|        Deduplication • Drop StudentID • 80/20 Stratified Split          |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                 EXPLORATORY DATA ANALYSIS (EDA)                         |
|     Distribution Plots • Pearson Correlation • Bivariate Violins        |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|               ZERO-LEAKAGE FEATURE ENGINEERING PIPELINE                 |
|   Num: SimpleImputer(median) -> StandardScaler                          |
|   Cat: SimpleImputer(mode)   -> OneHotEncoder(ignore_unknown)           |
|            *Fitted STRICTLY on X_train inside Pipeline*                 |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                   MACHINE LEARNING CLASSIFIERS                          |
|   Logistic Regression • Decision Tree • Random Forest • Naïve Bayes     |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                    EVALUATION & INFERENCE API                           |
|       Metrics (Acc/P/R/F1) • Confusion Matrices • Gini Importance       |
|            Predicted: Eligible / Not Eligible + Confidence              |
+-------------------------------------------------------------------------+
```

---

## 5. Exploratory Data Analysis & Correlation Analysis

All visual plots are saved under [outputs/plots/](file:///e:/DS/outputs/plots).

### Verified Correlation Measurements:
- **`FamilyIncome` vs Target ($r$):** **$-0.5739$** (Rounded for presentation: **$r \approx -0.57$**)
- **`12thMarks` vs Target ($r$):** **$+0.2226$** (Rounded for presentation: **$r \approx +0.22$**)

### Interpretation and Causality Caveats:
1. **Negative Correlation ($r \approx -0.57$):** Students with higher household incomes show a lower probability of being labeled Eligible due to means-testing criteria.
2. **Positive Correlation ($r \approx +0.22$):** Higher 12th marks moderately correlate with higher eligibility labels under merit criteria.
3. **Correlation Does NOT Imply Causation:** Statistical correlation reflects linear association within the dataset, not direct causation. High marks do not "cause" an award if income exceeds thresholds.

---

## 6. Machine Learning Model Results & Comparison

Models were trained on 800 training samples and evaluated on 200 held-out test samples using stratified sampling (`random_state=42`).

### Verified Model Performance Table

| Model | Accuracy | Precision (Eligible) | Recall (Eligible) | F1-Score (Eligible) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 86.00% | 0.8906 | 0.8906 | 0.8906 |
| **Decision Tree** | 90.50% | 0.9504 | 0.8984 | 0.9237 |
| **Random Forest** | **93.00%** | **0.9672** | **0.9219** | **0.9440** |
| **Naïve Bayes** | 84.00% | 0.8810 | 0.8672 | 0.8740 |

- **Best Measured Model:** **Random Forest** (Accuracy = 93.00%, F1-Score = 0.9440).
- **Confusion Matrix (Random Forest):** True Ineligible = 68, False Eligible = 4, False Ineligible = 10, True Eligible = 118.

---

## 7. Random Forest Feature Importance

Calculated directly from the fitted Random Forest classifier (`feature_importances_` based on mean Gini impurity reduction):

| Rank | Feature Name | Model Feature Importance |
| :---: | :--- | :---: |
| **1** | **`FamilyIncome`** | **46.94%** |
| **2** | **`12thMarks`** | **21.28%** |
| 3 | `Community_OC` | 3.88% |
| 4 | `Community_SC` | 2.62% |
| 5 | `CollegeType_Private` | 1.77% |
| 6 | `FirstGraduate_No` | 1.56% |

- **Scientific Disclaimer:** Feature importance reflects how much the decision trees relied on each feature to partition this dataset. It indicates **predictive contribution**, not real-world causality.

---

## 8. Presentation Reconciliation (Slide Updates)

| Topic | PPT Slide Claim | Verified Implementation | Recommended PPT Update |
| :--- | :---: | :---: | :--- |
| **Random Forest Accuracy** | 95.5% | **93.00%** | Update PPT: *"Random Forest achieved 93.0% accuracy on held-out test set"* |
| **Random Forest F1-Score** | 0.966 | **0.9440** | Update PPT: *"Random Forest achieved 0.944 F1-score"* |
| **Target Distribution** | 63.3% / 36.7% | **63.8% / 36.2%** | Update PPT: *"63.8% Eligible vs 36.2% Not Eligible"* |
| **FamilyIncome Correlation** | $r = -0.58$ | **$r \approx -0.57$** | Update PPT: *"r ≈ -0.57"* |
| **12thMarks Correlation** | $r = 0.30$ | **$r \approx +0.22$** | Update PPT: *"r ≈ +0.22"* |
| **Top Predictors** | Income & Marks | **Income (46.9%) & Marks (21.3%)** | Exact rank match confirmed |

---

## 9. Limitations & Ethical Considerations

1. **Synthetic Dataset Limitation:** The model was trained on a synthetic reference dataset. Results demonstrate machine learning workflow and algorithm comparison, but may not generalize to real-world administrative schemes without training on certified, authentic administrative data.
2. **Dataset Size:** 1,000 records provide an academic proof-of-concept; production systems require significantly larger and longitudinally tracked cohorts.
3. **No Causality:** Predictions reflect statistical associations in the data. The model does not determine who is genuinely "deserving" or legally entitled to funds.
4. **Document Verification:** The pipeline evaluates submitted tabular data; it does not detect fraudulent income declarations or forged certificates.

---

## 10. Saved Models & Artifacts

- **Production Random Forest Pipeline:** `models/random_forest_pipeline.joblib` (and `models/best_model.joblib`)
  - Contains complete fitted `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder) + `RandomForestClassifier`.
  - Ingests raw 8-feature dictionaries or DataFrames directly.
- **Reports:**
  - `outputs/reports/eda_report.md` (Comprehensive EDA findings and distribution metrics)
  - `outputs/reports/project_verification_report.md` (Formal pipeline verification audit)
  - `outputs/consolidated_viva_guide.md` (Topics A to Z, 20 likely viva Q&A, 10 trap questions, and elevator pitches)
- **Presentation:**
  - `outputs/presentation/Scholarship_Eligibility_Prediction.pptx` (10-slide PowerPoint presentation)
  - `outputs/presentation/presentation_slides_content.md` (Slide text and visual breakdown)
  - `outputs/presentation_speaking_script.md` (Word-for-word spoken script for each slide)

---

## 11. Execution Instructions

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing & stratified train/test split
python src/data_preprocessing.py

# 3. Run exploratory data analysis and generate plots
python src/eda.py

# 4. Train, benchmark, and evaluate all 4 models
python src/train.py

# 5. Run inference tests on sample student profiles
python src/predict.py

# 6. Run inference using the explicit pipeline path
python src/predict.py --model models/random_forest_pipeline.joblib

# 7. Run interactive student screening CLI
python src/predict.py --interactive
```
