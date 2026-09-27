# Scholarship Eligibility Prediction System

[![Course](https://img.shields.io/badge/Course-AD4V71%20%2F%20AD5302%20Data%20Science-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

---

## Academic Metadata

- **Project Title:** Scholarship Eligibility Prediction
- **Course Code:** AD4V71 / AD5302 – Data Science (PBL Reviews 1 & 2)
- **Project Team:**
  - **Kirutick Siddhesh V** (Regd No.: `210425243120`, Section: C)
  - **Krishna H** (Regd No.: `210425243124`, Section: C)

---

## 1. Executive Summary & Problem Identification

### The Problem
Scholarship selection across Indian higher education institutions is traditionally conducted manually. Review boards must cross-reference large volumes of applicant records across complex, multi-tiered governmental guidelines (e.g., affirmative action categories, parental income caps, first-generation graduate concessions, and minimum academic cutoffs). This manual process is:
- **Labor-Intensive and Slow:** Delays disbursement to vulnerable students.
- **Error-Prone and Inconsistent:** Vulnerable to subjective human interpretations and administrative discrepancies.
- **Inefficient:** Administrative overhead consumes valuable faculty and institutional bandwidth.

### The Objective
To build an automated, transparent, and reproducible Machine Learning classification system that predicts whether a student qualifies as **Eligible** or **Not Eligible** for a higher education scholarship based on relevant academic, socioeconomic, and demographic attributes.

### Explicit Scope Boundaries
- **IN SCOPE:**
  - Automated screening and decision-support classification.
  - Multi-variable relationship modeling (income thresholds, affirmative welfare schemes, academic performance).
  - Model benchmarking, confusion matrix diagnostics, and feature importance rankings.
- **OUT OF SCOPE:**
  - **Fund Disbursement:** The system does not transfer money or manage bank transactions.
  - **Document Authentication:** The system assumes provided data is authentic; physical certificate verification remains an administrative task.
  - **Official Legal Determination:** The system is an intelligent decision-support aid, not a statutory authority.

---

## 2. Architecture & Flowchart

The system follows a modular four-tier architecture as presented in Course Review 2:

```
+-------------------------------------------------------------------------+
|                              STUDENT DATA                               |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                  STEP 1: DATA COLLECTION (Data Layer)                   |
|       Open Government Data (data.gov.in) + TN Scholarship Schemes       |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|               STEP 2: DATA PREPROCESSING (Processing Layer)             |
|          Missing Value Imputation • Deduplication • Normalization       |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                 STEP 3: EXPLORATORY DATA ANALYSIS (EDA)                 |
|       Distribution Plots • Correlation Matrix • Bivariate Analysis      |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                     STEP 4: FEATURE ENGINEERING                         |
|     ColumnTransformer • StandardScaler (Num) • OneHotEncoder (Cat)      |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|                   STEP 5: MACHINE LEARNING MODELS                       |
|   Logistic Regression • Decision Tree • Random Forest • Naïve Bayes     |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
+-------------------------------------------------------------------------+
|            STEP 6: PREDICTION & EVALUATION (Prediction Layer)           |
|      Eligible / Not Eligible (with Confidence Score & Probability)      |
+-------------------------------------------------------------------------+
```

---

## 3. Dataset Description & Schema

The dataset comprises **1,000 student records** structured under the Open Government Data (OGD) framework and the Government of Tamil Nadu Post-Matric Scholarship guidelines:

| Column Name | Type | Category | Description & Valid Values | Handling |
| :--- | :--- | :--- | :--- | :--- |
| `StudentID` | String | Identifier | Unique applicant ID (e.g., `STU1001`) | Dropped prior to modeling |
| `Gender` | String | Categorical | `Male`, `Female` | One-Hot Encoded |
| `Community` | String | Categorical | `SC`, `ST`, `MBC`, `BC`, `OC` | One-Hot Encoded |
| `FamilyIncome` | Integer | Numerical | Annual household income in INR (₹45,000 to ₹750,000) | StandardScaled |
| `12thMarks` | Float | Numerical | Higher Secondary Board percentage (48.0% to 99.0%) | StandardScaled |
| `FirstGraduate` | String | Categorical | First graduate in family (`Yes`, `No`) | One-Hot Encoded |
| `District` | String | Categorical | Native district in Tamil Nadu (10 districts) | One-Hot Encoded |
| `CollegeType` | String | Categorical | `Government`, `Government Aided`, `Private` | One-Hot Encoded |
| `Course` | String | Categorical | Degree stream (`Engineering`, `Arts & Science`, etc.) | One-Hot Encoded |
| **`Eligibility`** | String | **Target** | Class label: **`Eligible`** or **`Not Eligible`** | Target Variable |

### Target Distribution
- **Eligible:** ~63.8% (Majority class)
- **Not Eligible:** ~36.2% (Minority class)
- **Target Characteristic:** Moderately imbalanced.

---

## 4. Exploratory Data Analysis & Correlation Findings

All visual assets are systematically saved to `outputs/plots/`.

### Key Analytical Findings:
1. **Target Distribution (`outputs/plots/target_distribution.png`):**
   Demonstrates a ~64/36 split. Because the target is not 50/50, relying exclusively on **Accuracy** is misleading. Evaluation must prioritize **F1-Score**, **Precision**, and **Recall**.
2. **Family Income (`outputs/plots/family_income_distribution.png`):**
   Eligible applicants exhibit a heavily right-skewed income distribution with a median of ~₹125,000, adhering strictly to government means-testing thresholds. Ineligible applicants concentrate above ₹250,000.
3. **12th Marks (`outputs/plots/marks_distribution.png`):**
   Eligible students span 50% to 98% with an average around 73%, reflecting affirmative minimum criteria combined with merit-cum-means schemes.
4. **Correlation Analysis (`outputs/plots/correlation_heatmap.png`):**
   - **`FamilyIncome` Correlation:** $r \approx -0.57$ (Presentation target: $r = -0.58$)
   - **`12thMarks` Correlation:** $r \approx +0.22$ (Presentation target: $r = 0.30$)

### Scientific Explanation of Correlation:
- **Negative Correlation ($r < 0$ for FamilyIncome):** As annual income increases, the likelihood of qualifying for financial assistance decreases systematically due to means-testing criteria.
- **Positive Correlation ($r > 0$ for 12thMarks):** Higher academic achievement increases eligibility for competitive merit and fee-waiver schemes.
- **Causation Caveat:** Correlation measures linear association, not causation. For example, high 12th marks do not guarantee eligibility if the applicant's family income exceeds the statutory threshold.

---

## 5. Machine Learning Algorithms & Selection Rationale

Four complementary classification paradigms were benchmarked:

1. **Logistic Regression (Linear Baseline):**
   Models log-odds using a sigmoid function. Establishes the performance ceiling of linear decision boundaries.
2. **Decision Tree (Rule-Based Classifier):**
   Recursively splits features using Gini impurity. Directly mimics administrative threshold logic (e.g., `IF Income <= 200k AND Community == SC THEN Eligible`).
3. **Random Forest (Ensemble Bagging):**
   Constructs 100 decorrelated decision trees using bootstrap aggregation and random feature sub-sampling. Minimizes variance, mitigates single-tree overfitting, and excels on complex compound rules.
4. **Naïve Bayes (Probabilistic Classifier):**
   Applies Bayes' theorem under the conditional feature independence assumption. Provides a fast probabilistic benchmark.

---

## 6. Experimental Results & Performance Comparison

Models were trained on 800 samples (80%) and evaluated on 200 unseen test samples (20%) using **stratified sampling** with fixed random seed `42`.

### Measured Performance Table

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 86.00% | 0.8906 | 0.8906 | 0.8906 | Linear baseline |
| **Decision Tree** | 90.50% | 0.9504 | 0.8984 | 0.9237 | High interpretability |
| **Random Forest** | **93.00%** | **0.9672** | **0.9219** | **0.9440** | **Best Performing** |
| **Naïve Bayes** | 84.00% | 0.8810 | 0.8672 | 0.8740 | Probabilistic baseline |

### Presentation Comparison & Diagnostic Review (Slide 10 Consistency)

| Metric | Reference PPT Claim | Actual Measured Output | Difference | Scientific Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **RF Accuracy** | 95.5% | 93.00% | -2.50% | Expected train/test partition and boundary variance |
| **RF F1-Score** | 0.966 | 0.9440 | -0.022 | Normal variance; preserves identical algorithm ranking |
| **Top Predictor #1** | FamilyIncome | FamilyIncome (47.9%) | **Match** | Primary driver of means-tested eligibility |
| **Top Predictor #2** | 12thMarks | 12thMarks (21.2%) | **Match** | Secondary driver of merit schemes |

### Why Random Forest Outperformed Other Models
1. **Non-Linear Threshold Matching:** Scholarship eligibility is governed by non-linear step-functions (e.g., hard cutoff above ₹250,000 for SC/ST and ₹200,000 for BC/MBC). Linear models like Logistic Regression cannot model multi-variable step functions without extensive manual interaction engineering.
2. **Variance Reduction via Bagging:** While single Decision Trees overfit on training noise, Random Forest averages 100 trees to eliminate outlier variance.
3. **Robustness to Feature Correlations:** Random Forest handles correlated indicators (e.g., college type and first-graduate incentives) without collinearity instability.

---

## 7. Project Directory Structure

```
scholarship-eligibility-prediction/
│
├── data/
│   ├── raw/
│   │   ├── dataset_schema.json           # Formal JSON schema & attribute definitions
│   │   └── scholarship_data.csv          # 1,000-record benchmark dataset
│   └── processed/
│       ├── train.csv                     # Stratified 80% training split (800 rows)
│       └── test.csv                      # Stratified 20% testing split (200 rows)
│
├── notebooks/
│   └── scholarship_prediction.ipynb      # Complete 15-section annotated notebook
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_preprocessing.py             # Data loading, cleaning, inspection, splitting
│   ├── eda.py                            # Exploratory data analysis & 8 figure generators
│   ├── feature_engineering.py            # ColumnTransformer & feature extraction
│   ├── train.py                          # Multi-model training, comparison, serialization
│   ├── evaluate.py                       # Evaluation metrics, confusion matrices, rankings
│   └── predict.py                        # Single/batch inference API & interactive CLI
│
├── scripts/
│   ├── generate_reference_dataset.py     # Deterministic benchmark data generator
│   └── build_notebook.py                 # Automated Jupyter notebook compiler
│
├── models/
│   └── best_model.joblib                 # Serialized scikit-learn best model pipeline
│
├── outputs/
│   ├── plots/
│   │   ├── target_distribution.png       # Target class breakdown
│   │   ├── community_distribution.png    # Community breakdown
│   │   ├── family_income_distribution.png# Income histogram, KDE, and boxplot
│   │   ├── marks_distribution.png        # Marks histogram, KDE, and boxplot
│   │   ├── correlation_heatmap.png       # Pearson r heatmap
│   │   ├── feature_vs_target_income.png  # Bivariate violin plot (Income)
│   │   ├── feature_vs_target_marks.png   # Bivariate violin plot (Marks)
│   │   ├── feature_vs_target_community.png# Stacked bar plot (Community %)
│   │   ├── model_comparison.png          # 4-model performance bar chart
│   │   ├── confusion_matrices.png        # 2x2 confusion matrix grid
│   │   └── random_forest_feature_importance.png # Top feature importance rankings
│   ├── metrics/
│   │   ├── model_comparison.csv          # Comparative metrics table
│   │   ├── model_comparison.json         # Raw metrics JSON
│   │   └── eda_summary.json              # Statistical distribution summary
│   └── predictions/
│       └── sample_predictions.csv        # Multi-profile test predictions
│
├── requirements.txt                      # Pinned Python package dependencies
├── README.md                             # Project manual and technical report
└── .gitignore                            # Version control exclusion rules
```

---

## 8. Installation & Quick Start

### Step 1: Clone Repository & Create Environment
```bash
# Clone the repository
git clone https://github.com/your-username/scholarship-eligibility-prediction.git
cd scholarship-eligibility-prediction

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. How to Execute Pipeline Stages

### 1. Data Preprocessing
Cleans raw data, deduplicates, verifies schema, and generates train/test partitions:
```bash
python src/data_preprocessing.py
```

### 2. Exploratory Data Analysis
Computes Pearson correlations and generates all 8 analytical figures in `outputs/plots/`:
```bash
python src/eda.py
```

### 3. Model Training & Benchmarking
Trains Logistic Regression, Decision Tree, Random Forest, and Naïve Bayes, evaluates test sets, outputs comparison charts, and exports the best model:
```bash
python src/train.py
```

### 4. Interactive & Batch Predictions
Run inference on sample student records or enter individual details interactively:
```bash
# Run batch demonstration on sample candidates
python src/predict.py

# Launch interactive terminal prompt
python src/predict.py --interactive
```

### 5. Running the Jupyter Notebook
Launch the interactive notebook:
```bash
jupyter notebook notebooks/scholarship_prediction.ipynb
```

---

## 10. Sample Prediction Demonstration

Sample evaluation of illustrative student profiles via `python src/predict.py`:

| Candidate Profile | Community | Family Income | 12th Marks | First Graduate | Predicted Eligibility | Confidence Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Profile A (Needy, Affirmative)** | SC | ₹95,000 | 82.5% | Yes | **Eligible** | 93.8% |
| **Profile B (High Income)** | OC | ₹550,000 | 62.0% | No | **Not Eligible** | 91.8% |
| **Profile C (Middle Tier)** | MBC | ₹140,000 | 76.0% | Yes | **Eligible** | 88.9% |
| **Profile D (General Merit-cum-Means)** | OC | ₹110,000 | 92.5% | No | **Eligible** | 73.1% |

---

## 11. Presentation Consistency (Slide-by-Slide Mapping)

| Slide | Topic | Project Artifact Supporting the Slide |
| :---: | :--- | :--- |
| **Slide 1** | Title & Team | `README.md` & `notebooks/scholarship_prediction.ipynb` Section 1 |
| **Slide 2** | Problem & Scope | `README.md` Section 1 (Excludes fund disbursement & certificate forgery checks) |
| **Slide 3** | Basic ML Concepts | `README.md` Section 5 & Notebook Section 10 (Explains Supervised Learning, Features, Target) |
| **Slide 4** | Literature Survey & Gap | Integrated in `README.md` Section 1 (Combined academic + financial modeling) |
| **Slide 5** | Objectives | `README.md` Section 1 |
| **Slide 6** | Planning & Timeline | Work packages divided across Kirutick Siddhesh V & Krishna H |
| **Slide 7** | EDA & Correlations | `outputs/plots/` (Target, Community, Income, Marks, Heatmap: $r=-0.57, r=+0.22$) |
| **Slide 8** | Model Comparison & Results | `outputs/metrics/model_comparison.csv` and `outputs/plots/model_comparison.png` |
| **Slide 9** | References | Scikit-learn (Pedregosa et al.), Random Forest (Breiman), AISHE, TN Welfare Guidelines |
| **Slide 10** | Conclusion & Feature Importance | `outputs/plots/random_forest_feature_importance.png` (FamilyIncome & 12thMarks confirmed top predictors) |

---

## 12. Limitations & Future Roadmap

### Current Limitations:
1. **Synthetic Benchmark Calibration:** While accurately modeling real Tamil Nadu welfare rules, live administrative deployment requires direct ingestion of certified Open Government Data (OGD) application records.
2. **Missing Longitudinal Features:** Factors like semester-by-semester GPA, attendance rates, and backlogs are not captured in 12th-grade intake data.
3. **No Automated Document Verification:** Forged income certificates cannot be detected through numerical tabular data alone.

### Future Improvements:
1. **OCR Document Verification Pipeline:** Implement computer vision (e.g., Tesseract OCR or Vision Transformers) to parse and verify government caste/income certificates automatically.
2. **REST API & Web Interface:** Package the model into a FastAPI service with a React or Streamlit front-end for college admission counters.
3. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) or LIME to provide transparent, individualized decision justification letters to students.

---

## 13. References

1. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Quinlan, J. R. (1986). *Induction of Decision Trees*. Machine Learning, 1(1), 81-106.
4. McKinney, W. (2010). *Data Structures for Statistical Computing in Python (pandas)*. Proceedings of the 9th Python in Science Conference.
5. Government of Tamil Nadu – Adi Dravidar & Tribal Welfare / Backward Classes Welfare Department, *Post-Matric Scholarship Scheme Guidelines*.
6. Ministry of Education, Government of India – *All India Survey on Higher Education (AISHE) Reports*.
