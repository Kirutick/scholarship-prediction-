import unittest
import json
from datetime import date
from unittest.mock import patch

from api import index as api_module
from src.scholarship_catalog import (
    CatalogError,
    deadline_status,
    evaluate_scholarship,
    list_scholarships,
    load_catalog,
    recommend_scholarships,
    validate_profile,
)


def complete_record():
    return {
        "id": "verified-fixture",
        "name": "Verified fixture",
        "provider": "Test source",
        "state": "Test",
        "category": "Merit-based",
        "scholarship_type": "Merit-based",
        "education_level": "College",
        "eligible_communities": ["BC"],
        "income_limit": 200000,
        "minimum_marks": 70,
        "eligible_courses": ["Engineering"],
        "eligible_college_types": ["Government"],
        "first_graduate_requirement": "Yes",
        "gender_requirement": "Female",
        "domicile_requirement": "Tamil Nadu",
        "year_of_study_requirement": "2nd year",
        "disability_requirement": "No",
        "minority_requirement": "No",
        "eligibility_rules_status": "verified",
        "benefit_type": "Tuition reimbursement",
        "benefit_amount": None,
        "benefit_description": "Varies according to documented conditions.",
        "application_open_date": "2026-06-01",
        "deadline": "2026-10-31",
        "institute_verification_deadline": None,
        "level_2_verification_deadline": None,
        "application_window": "Open",
        "application_url": None,
        "payment_tracking_url": None,
        "official_source_id": "fixture-source",
        "official_source_url": "https://scholarships.gov.in/",
        "required_documents": ["Officially documented document"],
        "renewal_information": None,
        "notes": None,
        "last_verified": "2026-10-07",
        "source_status": "official",
    }


def complete_profile():
    return {
        "Community": "BC",
        "FamilyIncome": 150000,
        "12thMarks": 88,
        "Course": "Engineering",
        "CollegeType": "Government",
        "FirstGraduate": "Yes",
        "Gender": "Female",
        "Domicile": "Tamil Nadu",
        "YearOfStudy": "2nd year",
        "Disability": "No",
        "Minority": "No",
    }


class ScholarshipCatalogTests(unittest.TestCase):
    def test_verified_source_snapshot_loads_with_unknown_terms_intact(self):
        data = load_catalog()
        self.assertEqual(len(data["scholarships"]), 2)
        self.assertTrue(all(record["source_status"] == "partial" for record in data["scholarships"]))
        self.assertTrue(all(record["benefit_amount"] is None for record in data["scholarships"]))
        self.assertTrue(all(record["required_documents"] is None for record in data["scholarships"]))

    def test_partial_records_need_verification_and_have_no_score(self):
        record = load_catalog()["scholarships"][0]
        match = evaluate_scholarship(record, complete_profile(), date(2026, 10, 7))
        self.assertEqual(match["status"], "Needs Verification")
        self.assertIsNone(match["match_score"])
        self.assertIn("income_limit", match["unverified_criteria"])
        self.assertEqual(match["benefits"]["display"], "Not specified in the verified source.")
        self.assertEqual(match["eligibility_assessment"], "Cannot fully determine eligibility.")
        self.assertTrue(all(item["status"] == "UNKNOWN" for item in match["eligibility_checklist"]))

    def test_fully_documented_rules_get_transparent_score_and_reasons(self):
        match = evaluate_scholarship(complete_record(), complete_profile(), date(2026, 10, 7))
        self.assertEqual(match["status"], "Strong Match")
        self.assertEqual(match["match_score"], 100)
        self.assertEqual(len(match["score_components"]), 11)
        self.assertEqual(len(match["eligibility_checklist"]), 11)
        self.assertTrue(any("FamilyIncome" in reason for reason in match["matched_rules"]))
        self.assertEqual(match["benefits"]["display"], "Varies according to documented conditions.")

    def test_failed_known_rule_is_not_match_without_claiming_official_decision(self):
        profile = complete_profile()
        profile["Community"] = "OC"
        match = evaluate_scholarship(complete_record(), profile, date(2026, 10, 7))
        self.assertEqual(match["status"], "Not a Match")
        self.assertEqual(match["match_score"], 91)
        self.assertTrue(any("Community" in reason for reason in match["failed_rules"]))

    def test_missing_required_profile_input_prevents_score(self):
        profile = complete_profile()
        del profile["FamilyIncome"]
        match = evaluate_scholarship(complete_record(), profile, date(2026, 10, 7))
        self.assertEqual(match["status"], "Needs Verification")
        self.assertIsNone(match["match_score"])
        self.assertIn("FamilyIncome", match["missing_information"])

    def test_deadline_status_uses_dates_without_fabrication(self):
        self.assertEqual(deadline_status(None, date(2026, 10, 7)), "DATE NOT AVAILABLE")
        self.assertEqual(deadline_status("2026-10-31", date(2026, 10, 7)), "OPEN")
        self.assertEqual(deadline_status("2026-10-15", date(2026, 10, 7)), "OPEN")
        self.assertEqual(deadline_status("2026-10-14", date(2026, 10, 7)), "CLOSING SOON")
        self.assertEqual(deadline_status("2026-10-07", date(2026, 10, 7)), "CLOSING SOON")
        self.assertEqual(deadline_status("2026-12-31", date(2026, 10, 7)), "OPEN")
        self.assertEqual(deadline_status("2026-10-01", date(2026, 10, 7)), "CLOSED")
        with self.assertRaises(CatalogError):
            deadline_status("not-a-date", date(2026, 10, 7))

    def test_search_and_recommendation_include_source_and_deadline(self):
        results = list_scholarships({"q": "CSSS"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "pm-usp-csss")
        self.assertEqual(results[0]["source"]["url"], "https://scholarships.gov.in/")
        matches = recommend_scholarships({"ApplicationType": "Renewal"})
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(match["status"] == "Needs Verification" for match in matches))
        self.assertTrue(all(
            match["deadline_status"] == deadline_status("2026-10-31")
            for match in matches
        ))
        filters = list_scholarships({"state": "Central", "q": "CSSS", "open": "true"})
        self.assertEqual(len(filters), 1)
        self.assertEqual(list_scholarships({"community": "SC"}), [])
        self.assertEqual(list_scholarships({"income_based": "true"}), [])

    def test_profile_validation_rejects_unknown_or_out_of_range_values(self):
        with self.assertRaises(CatalogError):
            validate_profile({"Community": "Unknown"})
        with self.assertRaises(CatalogError):
            validate_profile({"12thMarks": 101})
        with self.assertRaises(CatalogError):
            validate_profile({"Aadhaar": "not accepted"})
        with self.assertRaises(CatalogError):
            validate_profile({"12thMarks": float("nan")})
        with self.assertRaises(CatalogError):
            validate_profile({"Disability": ["Yes"]})
        self.assertEqual(validate_profile({"12thMarks": 88.5}), {"12thMarks": 88.5})

    def test_malformed_catalog_shape_fails_explicitly(self):
        loaded = load_catalog()
        catalog = json.loads(json.dumps(loaded["catalog"]))
        sources = {"sources": list(loaded["sources"].values()), "official_hosts": ["scholarships.gov.in"]}
        catalog["scholarships"][0]["eligible_courses"] = "Engineering"
        with patch(
            "src.scholarship_catalog._load_json",
            side_effect=[catalog, sources],
        ):
            with self.assertRaises(CatalogError):
                load_catalog()

    def test_catalog_api_search_detail_and_invalid_profile(self):
        client = api_module.app.test_client()
        search = client.get("/api/scholarships?q=CSSS")
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.get_json()["count"], 1)
        detail = client.get("/api/scholarships/pm-usp-csss")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["scholarship"]["source_status"], "partial")
        missing = client.get("/api/scholarships/no-such-id")
        self.assertEqual(missing.status_code, 404)
        invalid = client.post("/api/scholarships/recommend", json={"profile": {"Community": "Unknown"}})
        self.assertEqual(invalid.status_code, 400)
        valid = client.post("/api/scholarships/recommend", json={"profile": {"ApplicationType": "Renewal"}})
        self.assertEqual(valid.status_code, 200)
        self.assertIsNone(valid.get_json()["scholarship_matches"][0]["match_score"])
        quality = client.get("/api/catalog/quality")
        self.assertEqual(quality.status_code, 200)
        self.assertGreater(quality.get_json()["warning_count"], 0)
        institute = client.get("/api/institutes?district=Chennai&course=Engineering")
        self.assertEqual(institute.status_code, 200)
        self.assertEqual(institute.get_json()["status"], "not_configured")
        self.assertEqual(institute.get_json()["count"], 0)
        self.assertIsNone(institute.get_json()["source"])

    def test_catalog_failure_does_not_interrupt_ml_prediction(self):
        client = api_module.app.test_client()
        payload = {
            "Gender": "Female", "Community": "BC", "FamilyIncome": 150000,
            "12thMarks": 88, "FirstGraduate": "Yes", "District": "Chennai",
            "CollegeType": "Government", "Course": "Engineering",
        }
        with patch("api.index.recommend_scholarships", side_effect=CatalogError("fixture failure")):
            response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], {"Eligible", "Not Eligible"})
        self.assertEqual(response.get_json()["scholarship_catalog_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
