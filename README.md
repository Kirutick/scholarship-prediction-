# Scholarship Eligibility Prediction

> An academic screening system that uses a synthetic-data Random Forest model and profile-based suggestions for scholarship categories to research. It does not determine official government scholarship eligibility.

**Live Demo:** https://scholarship-prediction-.vercel.app  
**GitHub:** https://github.com/Kirutick/scholarship-prediction-

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [How to Install](#how-to-install)
4. [How to Train](#how-to-train)
5. [How to Run Locally](#how-to-run-locally)
6. [How Prediction Works](#how-prediction-works)
7. [Model Results (Verified)](#model-results-verified)
8. [Deployment](#deployment)
9. [API Reference](#api-reference)

---

## Project Overview

This system trains and compares **four classification algorithms** on a synthetic dataset of 1,000 student records to classify a profile as `Eligible` / `Not Eligible`. The best-performing model (Random Forest) is serialized as a scikit-learn pipeline and served via Flask locally and on Vercel. The classifier and its training/evaluation artifacts remain unchanged by the recommendation layer.

**Key Features**
- 4 ML models trained with `random_state=42` for full reproducibility
- Pre-trained pipeline loaded at server start — **no retraining on deploy**
- Input validation on all 8 features before inference
- The API reports all eight submitted fields and classifier probabilities
- Transparent possible-category suggestions that do not assert official eligibility
- Estimated-amount configuration is explicitly empty until verified ranges are supplied
- CORS-enabled REST API consumable by any frontend

---

## Project Structure

```
scholarship-prediction/
├── data/
│   ├── raw/                      # Original dataset (scholarship_data.csv)
│   └── processed/
│       ├── train.csv             # 800 training observations
│       └── test.csv              # 200 test observations
│
├── src/
│   ├── data_preprocessing.py     # Data loading, cleaning, train/test split
│   ├── feature_engineering.py    # ColumnTransformer preprocessor pipeline
│   ├── scholarship_recommendations.py # Non-official category suggestions and empty amount config
│   ├── train.py                  # Trains all 4 models, saves best pipeline
│   ├── evaluate.py               # Metrics, confusion matrices, plots
│   ├── eda.py                    # Exploratory data analysis
│   └── predict.py                # Standalone prediction utility
│
├── models/
│   ├── best_model.joblib              # Best model by F1-Score (= Random Forest)
│   └── random_forest_pipeline.joblib  # Explicit RF pipeline (primary)
│
├── outputs/
│   ├── metrics/
│   │   ├── model_comparison.csv  # All 4 models - tabular metrics
│   │   └── model_comparison.json # All 4 models - JSON with confusion matrices
│   └── plots/
│       ├── confusion_matrices.png
│       ├── confusion_matrix_random_forest.png
│       ├── model_comparison.png
│       └── random_forest_feature_importance.png
│
├── web/
│   ├── app.py                    # Flask app (local server)
│   ├── templates/index.html      # Frontend UI
│   └── static/css/style.css      # Stylesheet
│
├── api/
│   └── index.py                  # Vercel serverless entrypoint (Flask WSGI)
│
├── public/                       # Static assets for Vercel frontend
├── notebooks/
│   └── scholarship_prediction.ipynb
├── scripts/
│   ├── verify_metrics.py         # Reproduces all 4 model metrics from scratch
│   └── verify_api.py             # Live API health + prediction tests
├── requirements.txt
├── vercel.json
└── README.md
```

---

## How to Install

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/Kirutick/scholarship-prediction-.git
cd scholarship-prediction-

# (Recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn>=1.3.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
flask>=3.0.0
```

---

## How to Train

All 4 models are trained from the pre-split processed data. The script:
1. Loads `data/processed/train.csv` and `test.csv`
2. Trains Logistic Regression, Decision Tree, Random Forest, Naïve Bayes
3. Evaluates on the held-out test set
4. Saves the best model (by F1-Score) to `models/best_model.joblib`
5. Always saves the Random Forest pipeline to `models/random_forest_pipeline.joblib`

```bash
python -m src.train
```

To reproduce raw train/test splits from the original dataset:

```bash
python -c "
from src.data_preprocessing import load_raw_data, clean_data, split_data, save_processed_splits
save_processed_splits(*split_data(clean_data(load_raw_data())))
"
```

To verify all 4 model metrics reproducibly:

```bash
python scripts/verify_metrics.py
```

---

## How to Run Locally

```bash
python web/app.py
```

The server starts at **http://127.0.0.1:5000**

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI (fill form → predict) |
| `/api/health` | GET | Server + model status |
| `/api/predict` | POST | Prediction endpoint |
| `/api/metadata` | GET | Schema categories + model benchmarks |

---

## How Prediction Works

The system uses a **scikit-learn Pipeline** that chains two steps:

```
Input DataFrame (8 features)
        │
        ▼
┌──────────────────────────────────┐
│  ColumnTransformer (Preprocessor)│
│  • Numeric: SimpleImputer(median)│
│            + StandardScaler      │
│  • Categorical: SimpleImputer +  │
│            OneHotEncoder         │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  RandomForestClassifier          │
│  n_estimators=100, max_depth=10  │
│  random_state=42                 │
└──────────────────────────────────┘
        │
        ▼
  "Eligible" / "Not Eligible"
  + probability scores
```

**Input features (all are passed to the existing classifier):**

| Feature | Type | Values |
|---|---|---|
| Gender | Categorical | Female, Male |
| Community | Categorical | BC, MBC, OC, SC, ST |
| FamilyIncome | Numeric | 0 – 10,000,000 (INR/year) |
| 12thMarks | Numeric | 0.0 – 100.0 (%) |
| FirstGraduate | Categorical | Yes, No |
| District | Categorical | Chennai, Coimbatore, Erode, Kanchipuram, Madurai, Salem, Thanjavur, Tiruchirappalli, Tirunelveli, Vellore |
| CollegeType | Categorical | Government, Government Aided, Private |
| Course | Categorical | Arts & Science, Commerce, Engineering, Management, Medical |

**Top Predictors (Gini Importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | FamilyIncome | 46.94% |
| 2 | 12thMarks | 21.28% |
| 3 | Community_OC | 3.88% |
| 4 | Community_SC | 2.62% |

The model is loaded **once at server start** from `models/random_forest_pipeline.joblib`. No training occurs during deployment or serving.

---

## Model Results (Verified)

> All metrics below are **empirically verified** by running `scripts/verify_metrics.py` against the held-out test set (200 samples, `random_state=42`).

### Why are there two sets of numbers? (93.0% vs 95.5%)

| Source | Accuracy | F1-Score | Origin |
|---|---|---|---|
| **Current code / saved model** | **93.00%** | **0.9440** | `model_comparison.json`, live API — ground truth |
| Presentation Slide 10 | 95.5% | 0.966 | Earlier experimental run / aspirational target |

**The 93.0% / 0.9440 numbers are the ground truth** — produced every time with `random_state=42` on the current dataset and pipeline. The 95.5% figure in the presentation was from an earlier experimental run and should be treated as a reference target, not the final measured result.

---

### All 4 Models — Test Set Performance (n=200, random_state=42)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** ✅ | **93.00%** | **0.9672** | **0.9219** | **0.9440** |
| Decision Tree | 90.50% | 0.9504 | 0.8984 | 0.9237 |
| Logistic Regression | 86.00% | 0.8906 | 0.8906 | 0.8906 |
| Naïve Bayes | 84.00% | 0.8810 | 0.8672 | 0.8740 |

Random Forest selected as best model by **F1-Score** (most reliable metric for moderately imbalanced classes).

---

### Random Forest Confusion Matrix (Exact)

Test set: 200 samples — 128 Eligible, 72 Not Eligible

```
                    Predicted: Not Eligible    Predicted: Eligible
True: Not Eligible       TN = 68                  FP = 4
True: Eligible           FN = 10                  TP = 118
```

| Metric | Value | Interpretation |
|---|---|---|
| True Positives (TP) | 118 | Correctly predicted Eligible |
| True Negatives (TN) | 68 | Correctly predicted Not Eligible |
| False Positives (FP) | 4 | Wrongly predicted Eligible (Type I) |
| False Negatives (FN) | 10 | Wrongly predicted Not Eligible (Type II) |
| **Total correct** | **186 / 200** | **93.00% accuracy** |

---

## Deployment

### Vercel (Production)

The project is deployed on Vercel using `api/index.py` as the serverless Flask WSGI entrypoint.

**URL:** https://scholarship-prediction-.vercel.app

`vercel.json` routes:
- `POST /api/predict` → `api/index.py`
- `GET /api/health` → `api/index.py`
- `GET /api/metadata` → `api/index.py`

`vercel.json` explicitly includes `models/random_forest_pipeline.joblib` in the Python function bundle so the runtime can load the pre-trained model. The public frontend calls the same-origin `/api/predict` route. No retraining occurs on Vercel.

The Vercel application must be redeployed after pushing these changes. This code change does not itself deploy the application.

### Local Flask Server

```bash
python web/app.py
# → http://127.0.0.1:5000
```

---

## API Reference

### `GET /api/health`

Returns server and model status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/random_forest_pipeline.joblib",
  "algorithm": "RandomForestClassifier",
  "test_accuracy": "93.00%",
  "test_f1_score": "0.9440"
}
```

---

### `POST /api/predict`

**Request body (JSON):**
```json
{
  "Gender": "Female",
  "Community": "SC",
  "FamilyIncome": 120000,
  "12thMarks": 88.5,
  "FirstGraduate": "Yes",
  "District": "Chennai",
  "CollegeType": "Government",
  "Course": "Engineering"
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Eligible",
  "eligible": true,
  "is_eligible": true,
  "eligible_probability": 95.88,
  "not_eligible_probability": 4.12,
  "confidence": 95.88,
  "model_used": "Random Forest Classifier (100 Estimators)",
  "potential_scholarships": [
    {
      "name": "First-generation student support category",
      "reason": "FirstGraduate = Yes. This profile may be worth checking against first-generation support programs; no official program criteria are configured.",
      "estimated_amount": null,
      "type": "Possible Scholarship Category"
    }
  ],
  "estimated_support": "Not configured: this project contains no verified award amounts.",
  "input_summary": { "...": "..." },
  "disclaimer": "Eligibility predictions use a synthetic academic dataset..."
}
```

`probability`, `confidence`, `eligible_probability`, and `not_eligible_probability` come from the saved classifier. `potential_scholarships` are generic review prompts based on the submitted profile, not named schemes or official matches. `src/scholarship_recommendations.py` contains an amount-range configuration with all entries unset. Configure a numeric range only after its source and conditions have been verified. Category estimates are not added together because categories may overlap.

**Error response (400):**
```json
{
  "status": "error",
  "message": "Missing required student attribute(s): FamilyIncome, 12thMarks"
}
```

---

### `GET /api/metadata`

Returns valid dropdown categories, model benchmarks, and feature importance data for UI rendering.

---

## Notes

- Dataset is **synthetically generated** for academic demonstration purposes. Its label heuristics are not official scholarship eligibility rules.
- All predictions carry a disclaimer — this system is for academic screening only, not statutory scholarship approval.
- The repository contains no verified scholarship program rules or scholarship amount targets. Amounts are not predicted by the Random Forest and remain unconfigured.
- `best_model.joblib` and `random_forest_pipeline.joblib` are **identical** (Random Forest won by F1-Score, verified by feature importance comparison).
- No API keys, secrets, or credentials exist anywhere in this repository.
- `.gitignore` covers `__pycache__`, `.env`, `venv/`, `.vscode/`, `.DS_Store`, and Jupyter checkpoints.
