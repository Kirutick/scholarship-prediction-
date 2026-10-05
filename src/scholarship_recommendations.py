"""Transparent, non-official scholarship category suggestions and amount config."""

from typing import Any, Dict, List, Optional, Tuple


ESTIMATED_AMOUNT_RANGES: Dict[str, Optional[Tuple[int, int]]] = {
    "overall_support": None,
    "first_generation": None,
    "community_based": None,
    "merit_based": None,
    "need_based": None,
    "course_based": None,
}

AMOUNT_NOT_CONFIGURED = "Not configured: this project contains no verified award amounts."


def _format_amount_range(amount_range: Optional[Tuple[int, int]]) -> Optional[str]:
    if amount_range is None:
        return None
    minimum, maximum = amount_range
    if minimum < 0 or maximum < minimum:
        raise ValueError("Configured scholarship amount ranges must be non-negative and ordered.")
    return f"₹{minimum:,}–₹{maximum:,} / year"


def build_scholarship_assessment(student: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest broad categories for review without asserting official eligibility."""
    candidates = []

    if student["FirstGraduate"] == "Yes":
        candidates.append((
            "first_generation",
            "First-generation student support category",
            "FirstGraduate = Yes. This profile may be worth checking against first-generation support programs; no official program criteria are configured.",
        ))

    candidates.extend([
        (
            "community_based",
            "Community/category-based support review",
            f"Community = {student['Community']}. The profile includes a community category, but no scheme-specific official criteria are configured.",
        ),
        (
            "merit_based",
            "Academic-merit support review",
            f"12thMarks = {student['12thMarks']}%. Compare this result with any program's own verified cutoff; no cutoff is configured here.",
        ),
        (
            "need_based",
            "Need-based support review",
            f"FamilyIncome = ₹{student['FamilyIncome']:,.0f} per year. Check program-specific income criteria; no verified income limit is configured here.",
        ),
        (
            "course_based",
            "Course/institution-specific support review",
            f"Course = {student['Course']} and CollegeType = {student['CollegeType']}. Use these details to check program-specific study requirements; none are configured here.",
        ),
    ])

    potential_scholarships: List[Dict[str, Any]] = []
    for amount_key, name, reason in candidates:
        potential_scholarships.append({
            "name": name,
            "reason": reason,
            "estimated_amount": _format_amount_range(ESTIMATED_AMOUNT_RANGES[amount_key]),
            "type": "Possible Scholarship Category",
        })

    return {
        "potential_scholarships": potential_scholarships,
        "estimated_support": _format_amount_range(ESTIMATED_AMOUNT_RANGES["overall_support"])
        or AMOUNT_NOT_CONFIGURED,
    }
