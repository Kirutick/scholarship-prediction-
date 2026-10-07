"""Shared Flask routes for the local scholarship catalog and rule evaluation."""

import logging

from flask import Blueprint, jsonify, request

from src.scholarship_catalog import (
    CatalogError,
    get_scholarship,
    list_scholarships,
    load_catalog,
    recommend_scholarships,
    validate_profile,
)


logger = logging.getLogger(__name__)
scholarship_api = Blueprint("scholarship_api", __name__)


def _catalog_error(error):
    logger.error("Scholarship catalog request failed: %s", error, exc_info=True)
    return jsonify({
        "status": "error",
        "message": "Scholarship information is temporarily unavailable.",
    }), 503


@scholarship_api.route("/api/scholarships", methods=["GET"])
def scholarships():
    try:
        records = list_scholarships({
            "q": request.args.get("q", ""),
            "state": request.args.get("state", ""),
            "category": request.args.get("category", ""),
            "scholarship_type": request.args.get("type", ""),
            "open": request.args.get("open", ""),
        })
        catalog = load_catalog()["catalog"]
        return jsonify({
            "status": "success",
            "database_last_updated": catalog.get("database_last_updated"),
            "count": len(records),
            "scholarships": records,
        })
    except CatalogError as error:
        return _catalog_error(error)


@scholarship_api.route("/api/scholarships/<scholarship_id>", methods=["GET"])
def scholarship_detail(scholarship_id):
    try:
        record = get_scholarship(scholarship_id)
    except CatalogError as error:
        return _catalog_error(error)
    if record is None:
        return jsonify({"status": "error", "message": "Scholarship record not found."}), 404
    return jsonify({"status": "success", "scholarship": record})


@scholarship_api.route("/api/scholarships/recommend", methods=["POST"])
def scholarship_recommendations():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "Payload must be a JSON object."}), 400
    try:
        profile = validate_profile(payload.get("profile", {}))
    except CatalogError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    try:
        records = recommend_scholarships(profile)
    except CatalogError as error:
        return _catalog_error(error)
    return jsonify({
        "status": "success",
        "profile_completeness": {
            "provided_ml_fields": sum(
                1 for field in (
                    "Gender", "Community", "FamilyIncome", "12thMarks",
                    "FirstGraduate", "District", "CollegeType", "Course",
                ) if field in profile
            ),
            "required_ml_fields": 8,
            "missing_ml_fields": [
                field for field in (
                    "Gender", "Community", "FamilyIncome", "12thMarks",
                    "FirstGraduate", "District", "CollegeType", "Course",
                ) if field not in profile
            ],
            "note": "Optional discovery fields are only required when a verified scheme rule uses them.",
        },
        "scholarship_matches": records,
        "disclaimer": (
            "Match scores compare only documented criteria; they are not award probabilities. "
            "Partial records require official verification."
        ),
    })


@scholarship_api.route("/api/profile/validate", methods=["POST"])
def validate_student_profile():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "Payload must be a JSON object."}), 400
    try:
        profile = validate_profile(payload)
    except CatalogError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    return jsonify({
        "status": "success",
        "valid": True,
        "provided_fields": sorted(profile),
        "disclaimer": "This validates supplied profile values; it does not establish scholarship eligibility.",
    })
