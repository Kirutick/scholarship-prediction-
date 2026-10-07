"""Load and evaluate the separately maintained scholarship knowledge base."""

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHOLARSHIPS_PATH = PROJECT_ROOT / "scholarships" / "scholarships.json"
SOURCES_PATH = PROJECT_ROOT / "scholarships" / "sources.json"
CATALOG_FIELDS = {
    "id", "name", "provider", "state", "category", "scholarship_type",
    "education_level", "eligible_communities", "income_limit", "minimum_marks",
    "eligible_courses", "eligible_college_types", "first_graduate_requirement",
    "gender_requirement", "domicile_requirement", "year_of_study_requirement", "disability_requirement",
    "minority_requirement", "eligibility_rules_status", "benefit_type",
    "benefit_amount", "benefit_description", "application_open_date", "deadline",
    "institute_verification_deadline", "level_2_verification_deadline",
    "application_window", "application_url", "payment_tracking_url", "official_source_id",
    "official_source_url", "required_documents", "renewal_information",
    "notes", "last_verified", "source_status",
}
PROFILE_RULES = {
    "eligible_communities": ("Community", "one_of"),
    "income_limit": ("FamilyIncome", "max"),
    "minimum_marks": ("12thMarks", "min"),
    "eligible_courses": ("Course", "one_of"),
    "eligible_college_types": ("CollegeType", "one_of"),
    "first_graduate_requirement": ("FirstGraduate", "equal"),
    "gender_requirement": ("Gender", "equal"),
    "domicile_requirement": ("Domicile", "equal"),
    "year_of_study_requirement": ("YearOfStudy", "equal"),
    "disability_requirement": ("Disability", "equal"),
    "minority_requirement": ("Minority", "equal"),
}
DEFAULT_OFFICIAL_HOSTS = {"scholarships.gov.in"}
PROFILE_ENUMS = {
    "Gender": {"Female", "Male"},
    "Community": {"BC", "MBC", "OC", "SC", "ST"},
    "FirstGraduate": {"Yes", "No"},
    "Course": {"Arts & Science", "Commerce", "Engineering", "Management", "Medical"},
    "CollegeType": {"Government", "Government Aided", "Private"},
    "ApplicationType": {"New", "Renewal", "Not sure"},
    "Disability": {"Yes", "No"},
    "Minority": {"Yes", "No"},
}
PROFILE_NUMBERS = {"FamilyIncome": (0, 10_000_000), "12thMarks": (0, 100)}
UNVERIFIED_RULE_FIELDS = (
    "eligible_communities", "income_limit", "minimum_marks", "eligible_courses",
    "eligible_college_types", "first_graduate_requirement", "gender_requirement",
    "domicile_requirement", "year_of_study_requirement", "disability_requirement", "minority_requirement",
)
CHECKLIST_LABELS = {
    "eligible_communities": "Community requirement",
    "income_limit": "Income requirement",
    "minimum_marks": "Academic requirement",
    "eligible_courses": "Course requirement",
    "eligible_college_types": "Institution requirement",
    "first_graduate_requirement": "First graduate requirement",
    "gender_requirement": "Gender requirement",
    "domicile_requirement": "Domicile requirement",
    "year_of_study_requirement": "Year of study requirement",
    "disability_requirement": "Disability requirement",
    "minority_requirement": "Minority requirement",
}
DISCLAIMER = (
    "A scholarship match is a profile-to-documented-criteria comparison, not "
    "the probability of receiving an award or an official eligibility decision."
)


class CatalogError(ValueError):
    """Raised when structured scholarship data is malformed or unsafe."""


def validate_profile(profile: Any) -> Dict[str, Any]:
    """Validate supplied profile values without requiring optional discovery data."""
    if not isinstance(profile, dict):
        raise CatalogError("Profile must be a JSON object.")
    allowed = set(PROFILE_ENUMS) | set(PROFILE_NUMBERS) | {"District", "Domicile", "YearOfStudy"}
    unknown = set(profile) - allowed
    if unknown:
        raise CatalogError(f"Unknown profile field(s): {', '.join(sorted(unknown))}")

    cleaned = {}
    for field, allowed_values in PROFILE_ENUMS.items():
        value = profile.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or value not in allowed_values:
            raise CatalogError(f"Invalid {field}; choose one of the supported values.")
        cleaned[field] = value
    for field, (minimum, maximum) in PROFILE_NUMBERS.items():
        value = profile.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            raise CatalogError(f"{field} must be numeric.")
        try:
            numeric_value = float(str(value).replace(",", "").replace("%", ""))
        except (TypeError, ValueError) as error:
            raise CatalogError(f"{field} must be numeric.") from error
        if not math.isfinite(numeric_value) or numeric_value < minimum or numeric_value > maximum:
            raise CatalogError(f"{field} must be between {minimum} and {maximum}.")
        cleaned[field] = numeric_value
    for field in ("District", "Domicile", "YearOfStudy"):
        value = profile.get(field)
        if value is not None and value != "":
            max_length = 40 if field == "YearOfStudy" else 100
            if not isinstance(value, str) or len(value.strip()) > max_length:
                raise CatalogError(f"{field} must be a short text value.")
            cleaned[field] = value.strip()
    return cleaned


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as data_file:
            result = json.load(data_file)
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Unable to load scholarship data file: {path.name}") from error
    if not isinstance(result, dict):
        raise CatalogError(f"Scholarship data file must contain an object: {path.name}")
    return result


def _is_official_url(value: Any, allowed_hosts: Optional[set] = None) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
        return (
            parsed.scheme == "https"
            and parsed.hostname in (allowed_hosts or DEFAULT_OFFICIAL_HOSTS)
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def _validate_iso_date(value: Any, label: str) -> None:
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise CatalogError(f"Invalid {label}; use ISO YYYY-MM-DD format.") from error


def load_catalog() -> Dict[str, Any]:
    """Load the scholarship and source snapshots and enforce safe data shape."""
    catalog = _load_json(SCHOLARSHIPS_PATH)
    sources_data = _load_json(SOURCES_PATH)
    if not isinstance(catalog.get("schema_version"), int) or not isinstance(catalog.get("database_last_updated"), str):
        raise CatalogError("Scholarship catalog metadata is invalid.")
    _validate_iso_date(catalog["database_last_updated"], "catalog update date")
    scholarships = catalog.get("scholarships")
    sources = sources_data.get("sources")
    configured_hosts = sources_data.get("official_hosts", [])
    if not isinstance(scholarships, list) or not isinstance(sources, list):
        raise CatalogError("Scholarship catalog and sources must be lists.")
    if not isinstance(configured_hosts, list) or any(not isinstance(host, str) for host in configured_hosts):
        raise CatalogError("Configured official hosts must be a list of host names.")
    allowed_hosts = set(configured_hosts) | DEFAULT_OFFICIAL_HOSTS

    source_by_id = {}
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("id"), str)
            or not source["id"]
            or not _is_official_url(source.get("url"), allowed_hosts)
        ):
            raise CatalogError("Every scholarship source must have an ID and an official HTTPS URL.")
        if (
            not isinstance(source.get("name"), str)
            or source.get("verification_status") not in {"official", "partial"}
            or not isinstance(source.get("verified_facts"), list)
            or any(not isinstance(fact, str) for fact in source["verified_facts"])
            or not isinstance(source.get("limitations"), str)
        ):
            raise CatalogError("Scholarship source metadata is invalid.")
        _validate_iso_date(source.get("last_verified"), "source verification date")
        if source["id"] in source_by_id:
            raise CatalogError("Scholarship source IDs must be unique.")
        source_by_id[source["id"]] = source

    seen_ids = set()
    for scholarship in scholarships:
        if not isinstance(scholarship, dict) or not CATALOG_FIELDS.issubset(scholarship):
            raise CatalogError("A scholarship record is missing required catalog fields.")
        if not isinstance(scholarship["id"], str) or not scholarship["id"]:
            raise CatalogError("Scholarship IDs must be non-empty strings.")
        if scholarship["id"] in seen_ids:
            raise CatalogError("Scholarship IDs must be unique.")
        seen_ids.add(scholarship["id"])
        for field in (
            "name", "provider", "state", "category", "scholarship_type",
            "education_level", "eligibility_rules_status", "source_status",
        ):
            if not isinstance(scholarship[field], str) or not scholarship[field]:
                raise CatalogError(f"Scholarship text field is invalid: {field}")
        for field in (
            "eligible_communities", "eligible_courses", "eligible_college_types",
            "required_documents",
        ):
            value = scholarship[field]
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            ):
                raise CatalogError(f"Scholarship list field is invalid: {field}")
            if value is not None and not value:
                raise CatalogError(f"Scholarship list field cannot be empty when documented: {field}")
        for field in ("income_limit", "minimum_marks"):
            value = scholarship[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise CatalogError(f"Scholarship numeric field is invalid: {field}")
        if scholarship["income_limit"] is not None and scholarship["income_limit"] < 0:
            raise CatalogError("Scholarship income limit must be non-negative.")
        if scholarship["eligible_communities"] is not None and not set(
            scholarship["eligible_communities"]
        ).issubset(PROFILE_ENUMS["Community"]):
            raise CatalogError(f"Scholarship community rule contains unsupported values: {scholarship['id']}")
        if scholarship["eligible_courses"] is not None and not set(
            scholarship["eligible_courses"]
        ).issubset(PROFILE_ENUMS["Course"]):
            raise CatalogError(f"Scholarship course rule contains unsupported values: {scholarship['id']}")
        if scholarship["eligible_college_types"] is not None and not set(
            scholarship["eligible_college_types"]
        ).issubset(PROFILE_ENUMS["CollegeType"]):
            raise CatalogError(f"Scholarship institution rule contains unsupported values: {scholarship['id']}")
        if scholarship["minimum_marks"] is not None and not 0 <= scholarship["minimum_marks"] <= 100:
            raise CatalogError("Scholarship minimum marks must be between 0 and 100.")
        for field in (
            "first_graduate_requirement", "gender_requirement", "domicile_requirement",
            "year_of_study_requirement",
            "disability_requirement", "minority_requirement", "benefit_type",
            "benefit_description", "application_window", "renewal_information", "notes",
        ):
            if scholarship[field] is not None and not isinstance(scholarship[field], str):
                raise CatalogError(f"Scholarship text field is invalid: {field}")
        if scholarship["eligibility_rules_status"] not in {"verified", "unknown"}:
            raise CatalogError(f"Invalid eligibility rules status: {scholarship['id']}")
        if (
            scholarship["eligibility_rules_status"] == "verified"
            and scholarship["source_status"] != "official"
        ):
            raise CatalogError(f"Verified eligibility rules require a fully verified source: {scholarship['id']}")
        if not isinstance(scholarship["official_source_id"], str):
            raise CatalogError(f"Scholarship source reference is invalid: {scholarship['id']}")
        source = source_by_id.get(scholarship["official_source_id"])
        if source is None or scholarship["official_source_url"] != source["url"]:
            raise CatalogError(f"Scholarship source reference is invalid: {scholarship['id']}")
        if scholarship["benefit_amount"] is not None and (
            isinstance(scholarship["benefit_amount"], bool)
            or not isinstance(scholarship["benefit_amount"], (int, float, str))
        ):
            raise CatalogError(f"Scholarship benefit amount is invalid: {scholarship['id']}")
        if scholarship["application_url"] is not None and not _is_official_url(scholarship["application_url"], allowed_hosts):
            raise CatalogError(f"Application URL must use the configured official host: {scholarship['id']}")
        if scholarship["payment_tracking_url"] is not None and not _is_official_url(
            scholarship["payment_tracking_url"], allowed_hosts
        ):
            raise CatalogError(f"Payment tracking URL must use the configured official host: {scholarship['id']}")
        if scholarship["source_status"] not in {"official", "partial"}:
            raise CatalogError(f"Invalid source status: {scholarship['id']}")
        for field in (
            "application_open_date", "deadline", "institute_verification_deadline",
            "level_2_verification_deadline", "last_verified",
        ):
            _validate_iso_date(scholarship[field], f"{field} for {scholarship['id']}")
    return {"catalog": catalog, "scholarships": scholarships, "sources": source_by_id}


def deadline_status(deadline: Optional[str], today: Optional[date] = None) -> str:
    """Return a display status derived only from a documented closing date."""
    if not deadline:
        return "DATE NOT AVAILABLE"
    try:
        closing_date = date.fromisoformat(deadline)
    except (TypeError, ValueError) as error:
        raise CatalogError("Scholarship deadline must use ISO YYYY-MM-DD format.") from error
    reference_day = today or date.today()
    days_remaining = (closing_date - reference_day).days
    if days_remaining < 0:
        return "CLOSED"
    if 0 <= days_remaining <= 7:
        return "CLOSING SOON"
    return "OPEN"


def _evaluate_eligibility(scholarship: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    matched_rules: List[str] = []
    failed_rules: List[str] = []
    missing_information: List[str] = []
    checked_criteria = 0
    checklist = []

    for rule_key, (profile_key, operator) in PROFILE_RULES.items():
        rule_value = scholarship.get(rule_key)
        if rule_value is None:
            checklist.append({
                "criterion": rule_key,
                "label": CHECKLIST_LABELS[rule_key],
                "status": "UNKNOWN",
                "reason": "This criterion is not documented in the available source.",
                "profile_field": profile_key,
            })
            continue
        if profile.get(profile_key) is None or profile.get(profile_key) == "":
            missing_information.append(profile_key)
            checklist.append({
                "criterion": rule_key,
                "label": CHECKLIST_LABELS[rule_key],
                "status": "UNKNOWN",
                "reason": f"{profile_key} is needed to check the documented rule.",
                "profile_field": profile_key,
            })
            continue
        checked_criteria += 1
        student_value = profile[profile_key]
        if operator == "one_of":
            matches = student_value in rule_value
            expected_text = ", ".join(str(value) for value in rule_value)
        elif operator == "max":
            matches = float(student_value) <= float(rule_value)
            expected_text = f"at most ₹{float(rule_value):,.0f} annual family income"
        elif operator == "min":
            matches = float(student_value) >= float(rule_value)
            expected_text = f"at least {rule_value}% marks"
        else:
            matches = student_value == rule_value
            expected_text = str(rule_value)

        label = f"{profile_key}: {student_value} (required: {expected_text})"
        (matched_rules if matches else failed_rules).append(label)
        checklist.append({
            "criterion": rule_key,
            "label": CHECKLIST_LABELS[rule_key],
            "status": "PASS" if matches else "FAIL",
            "reason": label,
            "profile_field": profile_key,
        })

    return {
        "matched_rules": matched_rules,
        "failed_rules": failed_rules,
        "missing_information": sorted(set(missing_information)),
        "checked_criteria": checked_criteria,
        "eligibility_checklist": checklist,
    }


def evaluate_scholarship(
    scholarship: Dict[str, Any],
    profile: Dict[str, Any],
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Evaluate only documented rules and preserve unknowns explicitly."""
    result = _evaluate_eligibility(scholarship, profile)
    deadline = scholarship.get("deadline")
    status_date = deadline_status(deadline, today)
    missing_information = list(result["missing_information"])
    reasons = [f"Matches documented criterion: {rule}" for rule in result["matched_rules"]]
    reasons.extend(f"Does not match documented criterion: {rule}" for rule in result["failed_rules"])
    application_window = scholarship.get("application_window") or ""
    if "renewal" in application_window.casefold():
        application_type = profile.get("ApplicationType")
        if application_type == "New":
            reasons.append(
                "The cited NSP notice describes an open renewal window; it does not establish a new-application window."
            )
        elif application_type == "Renewal":
            reasons.append(
                "Your profile says Renewal, matching the application type described in the cited NSP notice."
            )
        else:
            if "ApplicationType" not in missing_information:
                missing_information.append("ApplicationType")
            reasons.append("Confirm whether you are applying as a new student or renewing an existing award.")
            result["eligibility_checklist"].append({
                "criterion": "application_window",
                "label": "Application type",
                "status": "UNKNOWN",
                "reason": "Confirm whether this is a new application or renewal.",
                "profile_field": "ApplicationType",
            })
    missing_information = sorted(set(missing_information))

    unverified_criteria = [
        field for field in UNVERIFIED_RULE_FIELDS if scholarship.get(field) is None
    ]
    cannot_determine = bool(
        missing_information
        or unverified_criteria
        or scholarship.get("source_status") != "official"
        or scholarship.get("eligibility_rules_status") != "verified"
    )
    if cannot_determine:
        status = "Needs Verification"
        if unverified_criteria:
            reasons.append(
                "The available source does not document all eligibility criteria; review the official guidelines."
            )
        reasons.extend(f"Cannot verify because {field} was not provided." for field in missing_information)
    elif result["failed_rules"]:
        status = "Not a Match"
    else:
        status = "Strong Match" if result["checked_criteria"] else "Needs Verification"

    match_score = None
    if result["checked_criteria"] and not cannot_determine:
        match_score = round(100 * len(result["matched_rules"]) / result["checked_criteria"])

    if not reasons:
        reasons.append("No documented profile criteria were available to compare.")

    required_documents = scholarship.get("required_documents")
    return {
        "id": scholarship["id"],
        "name": scholarship["name"],
        "provider": scholarship["provider"],
        "state": scholarship["state"],
        "category": scholarship["category"],
        "scholarship_type": scholarship["scholarship_type"],
        "education_level": scholarship["education_level"],
        "eligibility_criteria": {
            field: scholarship.get(field)
            for field in (
                "eligible_communities", "income_limit", "minimum_marks",
                "eligible_courses", "eligible_college_types",
                "first_graduate_requirement", "gender_requirement",
                "domicile_requirement", "year_of_study_requirement", "disability_requirement",
                "minority_requirement",
            )
        },
        "match_score": match_score,
        "score_type": "Compatibility with documented criteria; not award probability.",
        "score_components": [
            {
                "criterion": rule,
                "result": "matched" if rule in result["matched_rules"] else "not matched",
            }
            for rule in result["matched_rules"] + result["failed_rules"]
        ],
        "status": status,
        "eligibility_assessment": (
            "Cannot fully determine eligibility."
            if cannot_determine
            else "Does not meet at least one documented criterion."
            if result["failed_rules"]
            else "Meets all documented profile criteria; confirm remaining official terms."
        ),
        "matched_rules": result["matched_rules"],
        "failed_rules": result["failed_rules"],
        "eligibility_checklist": result["eligibility_checklist"],
        "missing_information": missing_information,
        "unverified_criteria": unverified_criteria,
        "reasons": reasons,
        "benefits": {
            "type": scholarship.get("benefit_type"),
            "amount": scholarship.get("benefit_amount"),
            "description": scholarship.get("benefit_description"),
            "display": scholarship.get("benefit_description")
            or scholarship.get("benefit_amount")
            or "Not specified in the verified source.",
        },
        "deadline": deadline,
        "deadline_status": status_date,
        "days_until_deadline": (
            (date.fromisoformat(deadline) - (today or date.today())).days
            if deadline else None
        ),
        "application_open_date": scholarship.get("application_open_date"),
        "application_window": scholarship.get("application_window"),
        "verification_deadlines": {
            "defective_application": None,
            "institution": scholarship.get("institute_verification_deadline"),
            "department": None,
            "level_2": scholarship.get("level_2_verification_deadline"),
            "final_processing": None,
        },
        "required_documents": required_documents,
        "documents_display": required_documents or [],
        "documents_status": "verified" if required_documents is not None else "Not specified",
        "application_url": scholarship.get("application_url"),
        "payment_tracking_url": scholarship.get("payment_tracking_url"),
        "official_source_url": scholarship["official_source_url"],
        "source_name": scholarship["provider"],
        "source_status": scholarship["source_status"],
        "eligibility_rules_status": scholarship["eligibility_rules_status"],
        "last_verified": scholarship.get("last_verified"),
        "renewal_information": scholarship.get("renewal_information"),
        "notes": scholarship.get("notes"),
        "disclaimer": DISCLAIMER,
    }


def _with_source(record: Dict[str, Any], source_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = evaluate_scholarship(record, {})
    source = source_by_id[record["official_source_id"]]
    evaluated["source"] = {
        "name": source["name"],
        "url": source["url"],
        "verification_status": source["verification_status"],
        "last_verified": source["last_verified"],
        "verified_facts": source["verified_facts"],
        "limitations": source["limitations"],
    }
    return evaluated


def _matches_filters(
    scholarship: Dict[str, Any],
    filters: Dict[str, str],
    today: Optional[date] = None,
) -> bool:
    query = filters.get("q", "").strip().casefold()
    if query:
        searchable = " ".join(str(scholarship.get(key) or "") for key in (
            "name", "provider", "category", "state", "scholarship_type",
            "education_level", "eligible_courses", "eligible_communities",
            "benefit_type", "benefit_description", "application_window", "notes",
        )).casefold()
        if query not in searchable:
            return False
    for key in ("state", "category", "scholarship_type"):
        value = filters.get(key)
        if value and str(scholarship.get(key) or "").casefold() != value.casefold():
            return False
    if filters.get("course") and (
        not scholarship.get("eligible_courses")
        or filters["course"] not in scholarship["eligible_courses"]
    ):
        return False
    if filters.get("community") and (
        not scholarship.get("eligible_communities")
        or filters["community"] not in scholarship["eligible_communities"]
    ):
        return False
    if filters.get("income_based") == "true" and scholarship.get("income_limit") is None:
        return False
    if filters.get("merit_based") == "true" and not (
        scholarship.get("minimum_marks") is not None
        or "merit" in str(scholarship.get("category", "")).casefold()
        or "merit" in str(scholarship.get("scholarship_type", "")).casefold()
    ):
        return False
    status = deadline_status(scholarship.get("deadline"), today)
    if filters.get("open") == "true" and status not in {"OPEN", "CLOSING SOON"}:
        return False
    if filters.get("closing_soon") == "true" and status != "CLOSING SOON":
        return False
    return True


def catalog_filter_options() -> Dict[str, List[str]]:
    """Return only filter values present in the reviewed source records."""
    scholarships = load_catalog()["scholarships"]

    def distinct(field: str, *, include_list_values: bool = False) -> List[str]:
        values = set()
        for record in scholarships:
            value = record.get(field)
            candidates = value if include_list_values and isinstance(value, list) else [value]
            for item in candidates:
                if isinstance(item, str) and item and item.casefold() != "not specified":
                    values.add(item)
        return sorted(values, key=str.casefold)

    return {
        "states": distinct("state"),
        "categories": distinct("category"),
        "types": distinct("scholarship_type"),
        "courses": distinct("eligible_courses", include_list_values=True),
        "communities": distinct("eligible_communities", include_list_values=True),
    }


def list_scholarships(filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Search the local reviewed catalog; never performs live website scraping."""
    params = filters or {}
    data = load_catalog()
    return [
        _with_source(scholarship, data["sources"])
        for scholarship in data["scholarships"]
        if _matches_filters(scholarship, params)
    ]


def get_scholarship(scholarship_id: str) -> Optional[Dict[str, Any]]:
    data = load_catalog()
    record = next((item for item in data["scholarships"] if item["id"] == scholarship_id), None)
    return _with_source(record, data["sources"]) if record else None


def recommend_scholarships(
    profile: Dict[str, Any],
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate and rank records by rule status, score, and deadline urgency."""
    profile = validate_profile(profile)
    data = load_catalog()
    matches = []
    for record in data["scholarships"]:
        if not _matches_filters(record, filters or {}):
            continue
        match = evaluate_scholarship(record, profile)
        source = data["sources"][record["official_source_id"]]
        match["source"] = {
            "name": source["name"],
            "url": source["url"],
            "verification_status": source["verification_status"],
            "last_verified": source["last_verified"],
            "verified_facts": source["verified_facts"],
            "limitations": source["limitations"],
        }
        matches.append(match)
    status_order = {"Strong Match": 0, "Potential Match": 1, "Needs Verification": 2, "Not a Match": 3}

    def ranking(item: Dict[str, Any]):
        known_failures = len(item["failed_rules"])
        unresolved = len(item["missing_information"]) + len(item["unverified_criteria"])
        return (
            known_failures > 0,
            unresolved,
            -item["match_score"] if item["match_score"] is not None else 0,
            status_order.get(item["status"], 4),
            item["days_until_deadline"] if item["days_until_deadline"] is not None else float("inf"),
            item["benefits"]["amount"] is None and item["benefits"]["description"] is None,
            item["name"],
        )

    return sorted(
        matches,
        key=ranking,
    )


def catalog_quality_report() -> Dict[str, Any]:
    """Report incomplete but loadable records without promoting them to verified."""
    data = load_catalog()
    warnings = []
    for record in data["scholarships"]:
        prefix = f"{record['id']}: "
        if not record["name"].strip():
            warnings.append(prefix + "missing scholarship name.")
        if not record["provider"].strip():
            warnings.append(prefix + "missing provider.")
        if not record.get("official_source_url"):
            warnings.append(prefix + "missing official source.")
        if not record.get("last_verified"):
            warnings.append(prefix + "missing verification date.")
        if record["source_status"] != "official" or record["eligibility_rules_status"] != "verified":
            warnings.append(prefix + "eligibility information is partial or unverified.")
        if record["deadline"] is None:
            warnings.append(prefix + "application deadline is not available.")
        ordered_dates = [
            record.get("application_open_date"),
            record.get("deadline"),
            record.get("institute_verification_deadline"),
            record.get("level_2_verification_deadline"),
        ]
        known_dates = [date.fromisoformat(value) for value in ordered_dates if value]
        if any(later < earlier for earlier, later in zip(known_dates, known_dates[1:])):
            warnings.append(prefix + "documented application and verification dates are out of order.")
        if record.get("application_url") and record.get("official_source_url") == record.get("application_url"):
            warnings.append(prefix + "application link is only a general source URL, not a scheme-specific portal.")
        known_categories = {
            "state", "category", "scholarship_type", "eligible_communities",
            "income_limit", "minimum_marks", "eligible_courses",
            "eligible_college_types",
        }
        if not any(record.get(field) for field in known_categories):
            warnings.append(prefix + "no category or eligibility attributes are documented.")
    return {
        "record_count": len(data["scholarships"]),
        "warning_count": len(warnings),
        "warnings": warnings,
        "database_last_updated": data["catalog"]["database_last_updated"],
    }
