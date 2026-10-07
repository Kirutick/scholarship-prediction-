"""
Vercel Serverless Entrypoint for Scholarship Eligibility Prediction.

Exposes a Flask WSGI application instance `app` compatible with Vercel's
Python runtime (@vercel/python). Loads the pre-trained Random Forest pipeline
once from `models/random_forest_pipeline.joblib` without retraining.
Guarantees JSON responses on all routes and error handlers.
"""

import os
import sys
import logging
import math
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolve Project Root across Windows local and Vercel Linux Serverless (/var/task)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scholarship_recommendations import build_scholarship_assessment
from src.scholarship_catalog import CatalogError, recommend_scholarships
from src.scholarship_api import scholarship_api

PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

# Candidate paths for the serialized Random Forest pipeline
MODEL_CANDIDATE_PATHS = [
    os.path.join(PROJECT_ROOT, "models", "random_forest_pipeline.joblib"),
    os.path.join(CURRENT_DIR, "..", "models", "random_forest_pipeline.joblib"),
    os.path.join(os.getcwd(), "models", "random_forest_pipeline.joblib"),
    os.path.join(PROJECT_ROOT, "models", "best_model.joblib"),
    os.path.join(CURRENT_DIR, "..", "models", "best_model.joblib"),
    os.path.join(os.getcwd(), "models", "best_model.joblib"),
]

VALID_CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Community": ["BC", "MBC", "OC", "SC", "ST"],
    "FirstGraduate": ["No", "Yes"],
    "District": [
        "Chennai", "Coimbatore", "Erode", "Kanchipuram", "Madurai",
        "Salem", "Thanjavur", "Tiruchirappalli", "Tirunelveli", "Vellore"
    ],
    "CollegeType": ["Government", "Government Aided", "Private"],
    "Course": ["Arts & Science", "Commerce", "Engineering", "Management", "Medical"]
}

REQUIRED_FEATURES = [
    "Gender", "Community", "FamilyIncome", "12thMarks",
    "FirstGraduate", "District", "CollegeType", "Course"
]

DISCLAIMER_TEXT = (
    "DISCLAIMER: Eligibility predictions use a synthetic academic dataset and are for preliminary screening "
    "and scholarship matching assistance only. It does not guarantee scholarship eligibility, approval, award "
    "amount, or payment. Verify current requirements with official scholarship guidelines."
)

app = Flask(__name__, static_folder=None)
app.register_blueprint(scholarship_api)

_LOADED_MODEL = None
_RESOLVED_MODEL_PATH = None


def get_model():
    """Load and cache the pre-trained Random Forest pipeline."""
    global _LOADED_MODEL, _RESOLVED_MODEL_PATH
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    for path in MODEL_CANDIDATE_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            logger.info(f"Loading pre-trained pipeline from: {abs_path}")
            _LOADED_MODEL = joblib.load(abs_path)
            _RESOLVED_MODEL_PATH = abs_path
            logger.info(f"Successfully loaded model from {abs_path}")
            return _LOADED_MODEL

    raise FileNotFoundError("Random Forest model pipeline artifact was not found.")


# Pre-load model on module initialization
try:
    get_model()
except Exception as e:
    logger.warning(f"Deferred model loading until first request: {e}")


def validate_and_parse_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Validate input payload against the 8 required student features."""
    if not isinstance(data, dict):
        return False, "Payload must be a valid JSON object.", None, None

    cleaned = {}
    missing_fields = [f for f in REQUIRED_FEATURES if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing_fields:
        return False, f"Missing required student attribute(s): {', '.join(missing_fields)}", None, None

    # Validate Gender
    gender = str(data["Gender"]).strip()
    if gender not in VALID_CATEGORIES["Gender"]:
        return False, f"Invalid Gender '{gender}'. Allowed values: {VALID_CATEGORIES['Gender']}", None, None
    cleaned["Gender"] = gender

    # Validate Community
    community = str(data["Community"]).strip()
    if community not in VALID_CATEGORIES["Community"]:
        return False, f"Invalid Community '{community}'. Allowed values: {VALID_CATEGORIES['Community']}", None, None
    cleaned["Community"] = community

    # Validate FamilyIncome
    try:
        raw_income = str(data["FamilyIncome"]).replace(",", "").strip()
        income = float(raw_income)
        if not math.isfinite(income):
            return False, "Annual Family Income must be a finite number.", None, None
        if income < 0:
            return False, "Annual Family Income cannot be negative.", None, None
        if income > 10000000:
            return False, "Annual Family Income exceeds the supported input limit (Rs. 1 Crore).", None, None
        cleaned["FamilyIncome"] = income
    except (ValueError, TypeError):
        return False, "Annual Family Income must be a valid positive number in INR.", None, None

    # Validate 12thMarks
    try:
        raw_marks = str(data["12thMarks"]).replace("%", "").strip()
        marks = float(raw_marks)
        if not math.isfinite(marks):
            return False, "12th Board Marks must be a finite percentage.", None, None
        if marks < 0.0 or marks > 100.0:
            return False, "12th Board Marks must be between 0.0% and 100.0%.", None, None
        cleaned["12thMarks"] = round(marks, 2)
    except (ValueError, TypeError):
        return False, "12th Board Marks must be a valid percentage between 0 and 100.", None, None

    # Validate FirstGraduate
    first_grad = str(data["FirstGraduate"]).strip()
    if first_grad not in VALID_CATEGORIES["FirstGraduate"]:
        return False, f"Invalid FirstGraduate '{first_grad}'. Allowed values: {VALID_CATEGORIES['FirstGraduate']}", None, None
    cleaned["FirstGraduate"] = first_grad

    # Validate District
    district = str(data["District"]).strip()
    if district not in VALID_CATEGORIES["District"]:
        return False, f"Invalid District '{district}'. Supported districts: {VALID_CATEGORIES['District']}", None, None
    cleaned["District"] = district

    # Validate CollegeType
    col_type = str(data["CollegeType"]).strip()
    if col_type not in VALID_CATEGORIES["CollegeType"]:
        return False, f"Invalid CollegeType '{col_type}'. Allowed types: {VALID_CATEGORIES['CollegeType']}", None, None
    cleaned["CollegeType"] = col_type

    # Validate Course
    course = str(data["Course"]).strip()
    if course not in VALID_CATEGORIES["Course"]:
        return False, f"Invalid Course '{course}'. Allowed courses: {VALID_CATEGORIES['Course']}", None, None
    cleaned["Course"] = course

    application_type = data.get("ApplicationType")
    if application_type not in (None, ""):
        if application_type not in ["New", "Renewal", "Not sure"]:
            return False, "Invalid ApplicationType. Choose New, Renewal, or Not sure.", None, None
        cleaned["ApplicationType"] = application_type

    for optional_field, max_length in (("Domicile", 100), ("YearOfStudy", 40)):
        value = data.get(optional_field)
        if value not in (None, ""):
            if not isinstance(value, str) or len(value.strip()) > max_length:
                return False, f"{optional_field} must be text up to {max_length} characters.", None, None
            cleaned[optional_field] = value.strip()

    df_input = pd.DataFrame([cleaned])[REQUIRED_FEATURES]
    return True, None, df_input, cleaned


# CORS Support
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# Health Check Endpoints
@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/api/health.py", methods=["GET"])
@app.route("/api/index/health", methods=["GET"])
@app.route("/api/index.py/health", methods=["GET"])
def health_check():
    """Health check endpoint confirming server and model pipeline status."""
    try:
        model = get_model()
        is_ready = model is not None
    except Exception as error:
        logger.error("Model health check failed: %s", error, exc_info=True)
        is_ready = False

    return jsonify({
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "algorithm": "RandomForestClassifier",
        "test_accuracy": "93.00%",
        "test_f1_score": "0.9440",
        "features": REQUIRED_FEATURES,
        "disclaimer": DISCLAIMER_TEXT
    }), (200 if is_ready else 500)


# Metadata Endpoint
@app.route("/api/metadata", methods=["GET"])
@app.route("/metadata", methods=["GET"])
@app.route("/api/index/metadata", methods=["GET"])
def get_metadata():
    """Return schema categories, model metrics, and feature importance."""
    feature_importance = [
        {"feature": "FamilyIncome", "importance_pct": 46.94, "category": "Means-Testing Criterion"},
        {"feature": "12thMarks", "importance_pct": 21.28, "category": "Academic Merit Criterion"},
        {"feature": "Community_OC", "importance_pct": 3.88, "category": "Social Welfare Category"},
        {"feature": "Community_SC", "importance_pct": 2.62, "category": "Social Welfare Category"},
        {"feature": "CollegeType_Private", "importance_pct": 1.77, "category": "Institutional Affiliation"},
        {"feature": "FirstGraduate_No", "importance_pct": 1.56, "category": "First Generation Status"},
        {"feature": "CollegeType_Government", "importance_pct": 1.46, "category": "Institutional Affiliation"},
        {"feature": "Community_BC", "importance_pct": 1.32, "category": "Social Welfare Category"}
    ]

    model_benchmarks = [
        {"model": "Random Forest (Selected)", "accuracy": "93.00%", "precision": "0.9672", "recall": "0.9219", "f1_score": "0.9440", "status": "Optimal Best"},
        {"model": "Decision Tree", "accuracy": "90.50%", "precision": "0.9504", "recall": "0.8984", "f1_score": "0.9237", "status": "Benchmark"},
        {"model": "Logistic Regression", "accuracy": "86.00%", "precision": "0.8906", "recall": "0.8906", "f1_score": "0.8906", "status": "Baseline"},
        {"model": "Gaussian Naïve Bayes", "accuracy": "84.00%", "precision": "0.8810", "recall": "0.8672", "f1_score": "0.8740", "status": "Benchmark"}
    ]

    return jsonify({
        "categories": VALID_CATEGORIES,
        "features": REQUIRED_FEATURES,
        "benchmarks": model_benchmarks,
        "feature_importance": feature_importance,
        "test_samples": 200,
        "training_samples": 800
    })


# Prediction Endpoints - Catch all possible routing paths
@app.route("/api/predict", methods=["POST", "OPTIONS"])
@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api", methods=["POST", "OPTIONS"])
@app.route("/api/", methods=["POST", "OPTIONS"])
@app.route("/api/index", methods=["POST", "OPTIONS"])
@app.route("/api/index.py", methods=["POST", "OPTIONS"])
@app.route("/api/predict.py", methods=["POST", "OPTIONS"])
@app.route("/api/index.py/predict", methods=["POST", "OPTIONS"])
@app.route("/", methods=["POST", "OPTIONS"])
def predict():
    """
    Main prediction endpoint.
    Accepts 8 student features via JSON and performs inference using the pre-trained pipeline.
    Returns:
    {
      "prediction": "Eligible" or "Not Eligible",
      "probability": <number>,
      "confidence": <number>,
      "eligible_probability": <number>,
      "not_eligible_probability": <number>,
      ...
    }
    """
    if request.method == "OPTIONS":
        return "", 204

    try:
        model = get_model()
    except Exception as e:
        logger.error("Model failed to load for prediction: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Prediction service temporarily unavailable."
        }), 500

    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()

    is_valid, error_msg, df_input, cleaned_summary = validate_and_parse_input(payload)
    if not is_valid:
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400

    try:
        # Perform inference using serialized pipeline directly
        pred_label = model.predict(df_input)[0]
        probs = model.predict_proba(df_input)[0]

        classes = list(model.classes_)
        eligible_idx = classes.index("Eligible")
        not_eligible_idx = classes.index("Not Eligible")

        p_eligible = round(float(probs[eligible_idx]) * 100.0, 2)
        p_not_eligible = round(float(probs[not_eligible_idx]) * 100.0, 2)
        confidence = round(float(max(probs)) * 100.0, 2)

        # Standard decimal probability for programmatic consumers (e.g. 0.9688)
        prob_value = round((p_eligible if pred_label == "Eligible" else p_not_eligible) / 100.0, 4)

        assessment = build_scholarship_assessment(cleaned_summary)
        try:
            scholarship_matches = recommend_scholarships(cleaned_summary)
            catalog_status = "available"
        except CatalogError as catalog_error:
            logger.error("Scholarship catalog unavailable during prediction: %s", catalog_error, exc_info=True)
            scholarship_matches = []
            catalog_status = "unavailable"
        factors = [
            f"{feature} = {cleaned_summary[feature]} was included in the Random Forest input."
            for feature in REQUIRED_FEATURES
        ]

        return jsonify({
            "status": "success",
            "prediction": str(pred_label),
            "eligible": bool(pred_label == "Eligible"),
            "probability": prob_value,
            "confidence": confidence,
            "is_eligible": bool(pred_label == "Eligible"),
            "eligible_probability": p_eligible,
            "not_eligible_probability": p_not_eligible,
            "input_summary": cleaned_summary,
            "decision_factors": factors,
            **assessment,
            "scholarship_matches": scholarship_matches,
            "scholarship_catalog_status": catalog_status,
            "model_used": "Random Forest Classifier (100 Estimators)",
            "disclaimer": DISCLAIMER_TEXT
        })

    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Prediction service temporarily unavailable."
        }), 500


# JSON Error Handlers (Never return HTML for errors)
@app.errorhandler(404)
def handle_404(e):
    logger.info(f"404 handler triggered for {request.method} {request.path}")
    if request.method == "POST":
        return predict()
    if request.path in ["/api/health", "/health", "/api/index.py/health"]:
        return health_check()
    return jsonify({
        "status": "error",
        "message": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def handle_500(e):
    logger.error(f"500 Internal Error: {e}")
    return jsonify({
        "status": "error",
        "message": "Internal server error."
    }), 500


@app.errorhandler(Exception)
def handle_all_exceptions(e):
    logger.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({
        "status": "error",
        "message": "The request could not be completed."
    }), 500


# Static Frontend & Asset Routes (Serves UI when running locally or on serverless fallback)
@app.route("/", methods=["GET"])
def index():
    """Serve the main frontend HTML interface."""
    public_index = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(public_index):
        return send_from_directory(PUBLIC_DIR, "index.html")
    web_index = os.path.join(PROJECT_ROOT, "web", "templates", "index.html")
    if os.path.exists(web_index):
        return send_from_directory(os.path.dirname(web_index), "index.html")
    return "<h1>Scholarship Eligibility Prediction API</h1><p>API is active. Send POST requests to <code>/api/predict</code>.</p>"


@app.route("/static/<path:filename>", methods=["GET"])
def serve_static(filename):
    """Serve CSS, JS, and image assets."""
    for s_dir in [os.path.join(PUBLIC_DIR, "static"), os.path.join(PROJECT_ROOT, "web", "static")]:
        if os.path.exists(os.path.join(s_dir, filename)):
            return send_from_directory(s_dir, filename)
    return "", 404


@app.route("/plots/<path:filename>", methods=["GET"])
def serve_plots(filename):
    """Serve evaluation plots directly from outputs/plots."""
    for p_dir in [os.path.join(PUBLIC_DIR, "plots"), PLOTS_DIR]:
        if os.path.exists(os.path.join(p_dir, filename)):
            return send_from_directory(p_dir, filename)
    return "", 404


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Handle browser favicon cleanly."""
    return "", 204


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting server at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
