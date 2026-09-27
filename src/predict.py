"""
Inference and Prediction Module for Scholarship Eligibility Prediction.

Provides:
1. Python API `predict_student_eligibility()`
2. CLI interactive mode for evaluating individual student profiles
3. Batch prediction generator exporting to outputs/predictions/sample_predictions.csv
4. Proper probability estimation and regulatory disclaimer labeling
"""

import os
import sys
import argparse
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_MODEL_PATH = os.path.join("models", "best_model.joblib")
PREDICTIONS_DIR = os.path.join("outputs", "predictions")

DISCLAIMER_TEXT = (
    "DISCLAIMER: This prediction is produced by a Machine Learning model for academic "
    "research and preliminary screening purposes only. It DOES NOT represent official "
    "government scholarship approval, legal eligibility certification, document verification, "
    "or fund disbursement."
)


def load_trained_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """Load the serialized scikit-learn best model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python src/train.py' first to train and export the model."
        )
    return joblib.load(model_path)


def predict_student_eligibility(
    student_data: Union[Dict[str, Any], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Predict scholarship eligibility and confidence score for a student or batch of students.

    Expected input attributes:
      - Gender: 'Male' / 'Female'
      - Community: 'SC', 'ST', 'MBC', 'BC', 'OC'
      - FamilyIncome: integer / float (e.g., 120000)
      - 12thMarks: float (e.g., 84.5)
      - FirstGraduate: 'Yes' / 'No'
      - District: District name (e.g., 'Salem')
      - CollegeType: 'Government', 'Government Aided', 'Private'
      - Course: 'Engineering', 'Arts & Science', 'Commerce', 'Medical', 'Management'
    """
    pipeline = load_trained_pipeline(model_path)

    if isinstance(student_data, dict):
        df_input = pd.DataFrame([student_data])
        is_single = True
    elif isinstance(student_data, pd.DataFrame):
        df_input = student_data.copy()
        is_single = False
    else:
        raise ValueError("student_data must be a dict or a pandas DataFrame.")

    # Remove identifier or target columns if inadvertently passed
    drop_cols = [c for c in ['StudentID', 'Eligibility'] if c in df_input.columns]
    if drop_cols:
        df_input = df_input.drop(columns=drop_cols)

    # Predict
    preds = pipeline.predict(df_input)

    # Probabilities
    has_proba = hasattr(pipeline, "predict_proba")
    if has_proba:
        probs = pipeline.predict_proba(df_input)
        classes = list(pipeline.classes_)
        eligible_idx = classes.index("Eligible") if "Eligible" in classes else 1
    else:
        probs = None
        eligible_idx = None

    results = []
    for i, pred in enumerate(preds):
        if probs is not None:
            conf = float(probs[i][classes.index(pred)]) * 100.0
            p_eligible = float(probs[i][eligible_idx]) * 100.0
        else:
            conf = None
            p_eligible = None

        record = {
            "Prediction": pred,
            "Confidence": round(conf, 2) if conf is not None else "N/A",
            "Eligible_Probability": round(p_eligible, 2) if p_eligible is not None else "N/A",
            "Disclaimer": DISCLAIMER_TEXT
        }
        results.append(record)

    if is_single:
        return results[0]
    return {"predictions": results}


def generate_sample_predictions(
    output_path: str = os.path.join(PREDICTIONS_DIR, "sample_predictions.csv"),
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """Generate predictions on illustrative test cases and export to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = [
        {
            "StudentID": "SAMPLE_01",
            "Gender": "Female",
            "Community": "SC",
            "FamilyIncome": 95000,
            "12thMarks": 82.5,
            "FirstGraduate": "Yes",
            "District": "Madurai",
            "CollegeType": "Government",
            "Course": "Engineering",
            "Profile_Description": "Low income, SC community, First Graduate"
        },
        {
            "StudentID": "SAMPLE_02",
            "Gender": "Male",
            "Community": "OC",
            "FamilyIncome": 550000,
            "12thMarks": 62.0,
            "FirstGraduate": "No",
            "District": "Chennai",
            "CollegeType": "Private",
            "Course": "Management",
            "Profile_Description": "High income, OC category, moderate marks"
        },
        {
            "StudentID": "SAMPLE_03",
            "Gender": "Female",
            "Community": "MBC",
            "FamilyIncome": 140000,
            "12thMarks": 76.0,
            "FirstGraduate": "Yes",
            "District": "Salem",
            "CollegeType": "Government Aided",
            "Course": "Arts & Science",
            "Profile_Description": "Moderate income, MBC category, First Graduate"
        },
        {
            "StudentID": "SAMPLE_04",
            "Gender": "Male",
            "Community": "BC",
            "FamilyIncome": 190000,
            "12thMarks": 88.0,
            "FirstGraduate": "No",
            "District": "Coimbatore",
            "CollegeType": "Government",
            "Course": "Engineering",
            "Profile_Description": "Moderate income, BC category, high academic merit"
        },
        {
            "StudentID": "SAMPLE_05",
            "Gender": "Male",
            "Community": "OC",
            "FamilyIncome": 110000,
            "12thMarks": 92.5,
            "FirstGraduate": "No",
            "District": "Tiruchirappalli",
            "CollegeType": "Government",
            "Course": "Engineering",
            "Profile_Description": "Low income, OC category, very high merit (Merit-cum-Means candidate)"
        }
    ]

    df_samples = pd.DataFrame(samples)
    pipeline = load_trained_pipeline(model_path)

    feature_df = df_samples.drop(columns=["StudentID", "Profile_Description"])
    preds = pipeline.predict(feature_df)
    probs = pipeline.predict_proba(feature_df)
    classes = list(pipeline.classes_)
    eligible_idx = classes.index("Eligible")

    df_samples["Predicted_Eligibility"] = preds
    df_samples["Eligibility_Probability_%"] = (probs[:, eligible_idx] * 100).round(2)
    df_samples["Confidence_%"] = (np.max(probs, axis=1) * 100).round(2)

    df_samples.to_csv(output_path, index=False)
    print(f"[SAVED] Exported sample predictions -> {output_path}")
    return df_samples


def interactive_prediction():
    """CLI prompt for interactive eligibility prediction."""
    print("\n" + "=" * 65)
    print("SCHOLARSHIP ELIGIBILITY PREDICTION SYSTEM (INTERACTIVE)")
    print("=" * 65)
    print("Enter candidate details:")

    gender = input("Gender (Male/Female) [Male]: ").strip() or "Male"
    community = input("Community (SC/ST/MBC/BC/OC) [BC]: ").strip() or "BC"
    income_str = input("Annual Family Income in INR [150000]: ").strip() or "150000"
    marks_str = input("12th Board Marks (%) [78.5]: ").strip() or "78.5"
    first_grad = input("First Graduate in Family? (Yes/No) [Yes]: ").strip() or "Yes"
    district = input("District [Salem]: ").strip() or "Salem"
    col_type = input("College Type (Government/Government Aided/Private) [Government]: ").strip() or "Government"
    course = input("Course (Engineering/Arts & Science/Commerce/Medical/Management) [Engineering]: ").strip() or "Engineering"

    student = {
        "Gender": gender,
        "Community": community,
        "FamilyIncome": float(income_str),
        "12thMarks": float(marks_str),
        "FirstGraduate": first_grad,
        "District": district,
        "CollegeType": col_type,
        "Course": course
    }

    result = predict_student_eligibility(student)

    print("\n" + "-" * 50)
    print("PREDICTION RESULT:")
    print("-" * 50)
    print(f"Scholarship Eligibility Prediction: {result['Prediction']}")
    print(f"Prediction Confidence:              {result['Confidence']}%")
    print(f"Eligible Probability:               {result['Eligible_Probability']}%")
    print("-" * 50)
    print(result["Disclaimer"])
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Predict scholarship eligibility")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--samples", action="store_true", default=True, help="Generate sample test predictions CSV")
    args = parser.parse_args()

    if args.interactive:
        interactive_prediction()
    else:
        print("=" * 65)
        print("RUNNING BATCH SAMPLE PREDICTIONS")
        print("=" * 65)
        df_results = generate_sample_predictions()
        cols = ["StudentID", "Community", "FamilyIncome", "12thMarks", "Predicted_Eligibility", "Eligibility_Probability_%"]
        print(df_results[cols].to_string(index=False))
        print("\n" + DISCLAIMER_TEXT)
        print("=" * 65)


if __name__ == "__main__":
    main()
