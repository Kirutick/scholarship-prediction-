# Project Verification & Execution Audit Report

**Project Title:** Scholarship Eligibility Prediction  
**Course Code:** AD4V71 / AD5302 – Data Science  
**Authors:** Kirutick Siddhesh V (`210425243120`) & Krishna H (`210425243124`)  
**Execution Timestamp:** 2026-09-28T23:40:00+05:30  
**Environment:** Python 3.12 (Windows PowerShell) | Scikit-learn 1.4+ | Pandas 2.2+  
**Execution Status:** **ALL PIPELINE MODULES EXECUTED & EMPIRICALLY VERIFIED (EXIT CODE 0)**

---

## 1. Dataset & Provenance Verification

- **Raw Data File:** `data/raw/scholarship_data.csv`
- **Schema Specification:** `data/raw/dataset_schema.json`
- **Total Records:** `1,000` rows
- **Total Raw Columns:** `10` columns
- **Missing Values:** `0` across all columns (verified via `df.isnull().sum()`)
- **Duplicate Records:** `0` (verified via `df.duplicated().sum()`)
- **Data Provenance:** 
  - **100% Synthetic Reference Dataset** generated programmatically for academic machine learning experimentation.
  - Realistic attribute distributions and criteria inspired by the **Government of Tamil Nadu Post-Matric Scholarship Scheme** and **Ministry of Education AISHE demographic statistics**.
  - **Zero confidential real-world student records** or live OGD datasets were accessed or downloaded.

### Raw Column List & Data Types:
| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `StudentID` | `object` | Identifier | Unique student identifier (`STU1001` - `STU2000`) — **Dropped before modeling** |
| `Gender` | `object` | Feature (Nominal) | Student gender (`Female`, `Male`) |
| `Community` | `object` | Feature (Nominal) | Reservation group (`BC`, `MBC`, `SC`, `ST`, `OC`) |
| `FamilyIncome` | `int64` | Feature (Continuous) | Annual parental/family income in INR |
| `12thMarks` | `float64` | Feature (Continuous) | Higher secondary aggregate board percentage |
| `FirstGraduate` | `object` | Feature (Binary Nominal) | First-generation degree status (`Yes`, `No`) |
| `District` | `object` | Feature (Nominal) | Home district across 10 Tamil Nadu districts |
| `CollegeType` | `object` | Feature (Nominal) | Institution type (`Government`, `Government Aided`, `Private`) |
| `Course` | `object` | Feature (Nominal) | Enrolled academic stream (`Engineering`, `Medical`, `Arts & Science`, etc.) |
| `Eligibility` | `object` | **Target (Binary)** | Scholarship qualification label (`Eligible`, `Not Eligible`) |

---

## 2. Target Variable & Stratified Partition Verification

- **Target Variable:** `Eligibility`
- **Class Distribution (Full Dataset, 1,000 records):**
  - **`Eligible`:** `638` records (**63.8%**)
  - **`Not Eligible`:** `362` records (**36.2%**)
  - Class Ratio: Moderate imbalance ($\approx 1.76 : 1$)
- **Train/Test Splitting Protocol:**
  - Method: `train_test_split(test_size=0.2, random_state=42, stratify=y)`
  - **Training Set (`data/processed/train.csv`):** `800` records
    - `Eligible`: 510 records (63.75%)
    - `Not Eligible`: 290 records (36.25%)
  - **Testing Set (`data/processed/test.csv`):** `200` records
    - `Eligible`: 128 records (64.0%)
    - `Not Eligible`: 72 records (36.0%)

---

## 3. Preprocessing & Feature Engineering Configuration

- **Module:** `src/feature_engineering.py`
- **Zero-Leakage Architecture:** Encapsulated inside `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`.
  - All scalers, imputers, and encoders are fitted **strictly on `X_train`** (`800` samples).
  - Unseen `X_test` (`200` samples) is transformed strictly via `.transform()` using the learned training statistics.
- **Numerical Pipeline:**
  - Features: `FamilyIncome`, `12thMarks` (2 features)
  - Imputation: `SimpleImputer(strategy='median')`
  - Scaling: `StandardScaler()` (Centers to $\mu=0, \sigma=1$)
- **Categorical Pipeline:**
  - Features: `Community`, `FirstGraduate`, `District`, `CollegeType`, `Course`, `Gender` (6 features)
  - Imputation: `SimpleImputer(strategy='most_frequent')`
  - Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Post-Transformation Feature Matrix:**
  - Transformed output dimension: **`29` numerical columns** (2 standardized continuous + 27 one-hot binary indicator columns).

---

## 4. Exploratory Data Analysis (EDA) Verified Findings

- **Execution Script:** `src/eda.py`
- **Calculated Pearson Linear Correlations ($r$):**
  - **`FamilyIncome` vs. Eligibility:** **$r = -0.5739$ ($r \approx -0.57$)**  
    *Interpretation:* Strongest negative linear association with eligibility, reflecting means-tested income ceilings.
  - **`12thMarks` vs. Eligibility:** **$r = +0.2226$ ($r \approx +0.22$)**  
    *Interpretation:* Moderate positive linear association, reflecting merit-based qualifying criteria.
- **Scientific Causation Warning:** Correlation denotes linear co-occurrence within the dataset, **not physical or legal causation**. Low income does not automatically cause an award if academic or quota requirements fail; high marks do not cause an award if parental income exceeds limits.
- **Generated Plots (`outputs/plots/`):**
  - `target_distribution.png`: Bar and donut charts displaying 63.8% vs 36.2% split.
  - `family_income_distribution.png`: Density histogram and boxplots showing eligible concentration below ₹2.5L.
  - `12th_marks_distribution.png`: Normal-like distribution showing higher marks shifting probability toward eligibility.
  - `community_distribution.png`: Count plot showing demographic coverage across reservation categories.
  - `correlation_heatmap.png`: Heatmap quantifying feature-to-target correlations.

---

## 5. Model Benchmarking & Empirical Results

All four classifiers were trained on the identical 800 training observations and evaluated on the identical 200 held-out test observations.

### Verified Performance Table (Exact Measured Test Scores):
| Model Family | Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear** | Logistic Regression | 86.00% (0.8600) | 0.8906 | 0.8906 | 0.8906 |
| **Rule-Based** | Decision Tree | 90.50% (0.9050) | 0.9504 | 0.8984 | 0.9237 |
| **Ensemble** | **Random Forest (Best)** | **93.00% (0.9300)** | **0.9672** | **0.9219** | **0.9440** |
| **Probabilistic** | Naïve Bayes | 84.00% (0.8400) | 0.8810 | 0.8672 | 0.8740 |

### Verified Confusion Matrices (200 Test Records):
*(Rows = Actual Label [`Not Eligible`, `Eligible`]; Columns = Predicted Label [`Not Eligible`, `Eligible`])*

1. **Random Forest (Best Model):**
   ```
   Actual \ Pred     Not Eligible    Eligible    Total
   Not Eligible           68 (TN)      4 (FP)       72
   Eligible               10 (FN)    118 (TP)      128
   Total                  78         122           200
   ```
   - Correctly Classified: $68 + 118 = 186$ ($93.0\%$)
   - False Approvals ($FP$): $4$ ($3.28\%$ of positive predictions)
   - Wrongful Rejections ($FN$): $10$ ($7.81\%$ of eligible candidates)

2. **Decision Tree:**
   ```
   Actual \ Pred     Not Eligible    Eligible    Total
   Not Eligible           66 (TN)      6 (FP)       72
   Eligible               13 (FN)    115 (TP)      128
   Total                  79         121           200
   ```
   - Correctly Classified: $66 + 115 = 181$ ($90.5\%$)

3. **Logistic Regression:**
   ```
   Actual \ Pred     Not Eligible    Eligible    Total
   Not Eligible           58 (TN)     14 (FP)       72
   Eligible               14 (FN)    114 (TP)      128
   Total                  72         128           200
   ```
   - Correctly Classified: $58 + 114 = 172$ ($86.0\%$)

4. **Naïve Bayes:**
   ```
   Actual \ Pred     Not Eligible    Eligible    Total
   Not Eligible           57 (TN)     15 (FP)       72
   Eligible               17 (FN)    111 (TP)      128
   Total                  74         126           200
   ```
   - Correctly Classified: $57 + 111 = 168$ ($84.0\%$)

---

## 6. Random Forest Feature Importance Verification

- **Method:** Mean Decrease in Impurity (Gini Impurity Reduction) averaged across all 100 trees.
- **Top Predictors Extracted from Fitted Pipeline:**
  1. **`FamilyIncome`:** **`0.469358` (46.94%)** — Rank #1 dominant predictor
  2. **`12thMarks`:** **`0.212762` (21.28%)** — Rank #2 predictor
  3. `Community_OC`: `0.038811` (3.88%)
  4. `Community_SC`: `0.026247` (2.62%)
  5. `CollegeType_Private`: `0.017694` (1.77%)
  6. `FirstGraduate_No`: `0.015552` (1.56%)
  7. `CollegeType_Government`: `0.014585` (1.46%)
  8. `Community_BC`: `0.013222` (1.32%)
- **Cumulative Contribution:** `FamilyIncome` + `12thMarks` account for **`68.22%`** of the total split decisions in the forest.

---

## 7. Model Serialization & Live Inference Verification

- **Exported Pipeline Path:** `models/best_model.joblib` (`1.56 MB`)
- **Serialized Object:** Full scikit-learn `Pipeline` containing fitted `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder) and fitted `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)`.
- **Inference Verification Test:**
  - Input Student Profile:
    ```python
    {
        'Gender': 'Female',
        'Community': 'SC',
        'FamilyIncome': 120000,
        '12thMarks': 85.0,
        'FirstGraduate': 'Yes',
        'District': 'Chennai',
        'CollegeType': 'Government',
        'Course': 'Engineering'
    }
    ```
  - Output:
    - **Predicted Label:** **`Eligible`**
    - **Confidence Score:** **`96.88%`**
    - **Eligibility Probability:** **`96.88%`**
    - **Disclaimer Appended:** Confirmed mandatory screening disclaimer present.

---

## 8. Discrepancy & Historical Alignment Audit

| Parameter | Old / Initial Presentation Claim | Actual Measured Run | Status & Resolution |
| :--- | :---: | :---: | :--- |
| **RF Accuracy** | 95.5% | **93.00%** | Updated in presentation and speaking script to match verified measurement. |
| **RF F1-Score** | 0.966 | **0.9440** | Updated in presentation and speaking script to match verified measurement. |
| **Dataset Origin** | Implied OGD / live data | **Synthetic reference dataset** | Explicitly documented as synthetic academic benchmark inspired by TN schemes. |
| **Top Features** | FamilyIncome, 12thMarks | **FamilyIncome (46.94%), 12thMarks (21.28%)** | Confirmed 100% consistent with empirical feature importance ranking. |
| **Data Leakage** | Preprocessing outside pipeline | **Zero leakage ColumnTransformer** | Formally validated inside scikit-learn Pipeline fitted strictly on `X_train`. |

---

## 9. Conclusion

The Scholarship Eligibility Prediction pipeline has been executed, logged, and audited. All figures reported in `outputs/metrics/model_comparison.json`, `outputs/plots/`, and this report represent the single empirical source of truth.
