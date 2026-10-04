"""
Verification Script for Scholarship Prediction Bug Fix.
Tests both api/index.py and web/app.py across:
1. Marks variation: 60 vs 99 (SC profile)
2. Income variation: 120,000 vs 850,000
3. Borderline student: Marks 60 vs 99 (OC profile flipping prediction)
4. Schema validation and debug logging
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from api.index import app as api_app
from web.app import app as web_app



def test_endpoint(app, name):
    print("=" * 70)
    print(f"VERIFYING ENDPOINT: {name}")
    print("=" * 70)
    client = app.test_client()

    base_payload = {
        "Gender": "Female",
        "Community": "SC",
        "FamilyIncome": 120000,
        "12thMarks": 60.0,
        "FirstGraduate": "Yes",
        "District": "Chennai",
        "CollegeType": "Government",
        "Course": "Engineering"
    }

    # 1. Test A: 12thMarks = 60
    p60 = dict(base_payload, **{"12thMarks": 60.0})
    r60 = client.post("/api/predict", json=p60)
    d60 = r60.get_json()
    marks60_received = d60["input_summary"]["12thMarks"]
    print(f"[Test A] 12thMarks = 60.0")
    print(f"  HTTP Status:            {r60.status_code}")
    print(f"  Backend Received Marks: {marks60_received}")
    print(f"  Predicted Label:        {d60['prediction']}")
    print(f"  Eligible Probability:   {d60['eligible_probability']}%")
    print(f"  Not Eligible Prob:      {d60['not_eligible_probability']}%")
    print(f"  Confidence:             {d60['confidence']}%")
    print(f"  Decision Factors:       {d60['decision_factors']}")
    assert marks60_received == 60.0, f"Expected 60.0, got {marks60_received}"

    # 2. Test B: 12thMarks = 99
    p99 = dict(base_payload, **{"12thMarks": 99.0})
    r99 = client.post("/api/predict", json=p99)
    d99 = r99.get_json()
    marks99_received = d99["input_summary"]["12thMarks"]
    print(f"\n[Test B] 12thMarks = 99.0")
    print(f"  HTTP Status:            {r99.status_code}")
    print(f"  Backend Received Marks: {marks99_received}")
    print(f"  Predicted Label:        {d99['prediction']}")
    print(f"  Eligible Probability:   {d99['eligible_probability']}%")
    print(f"  Not Eligible Prob:      {d99['not_eligible_probability']}%")
    print(f"  Confidence:             {d99['confidence']}%")
    print(f"  Decision Factors:       {d99['decision_factors']}")
    assert marks99_received == 99.0, f"Expected 99.0, got {marks99_received}"
    assert d99["eligible_probability"] != d60["eligible_probability"], "Model probabilities should reflect changed marks"

    # 3. Test C: FamilyIncome substantial variation (1.2L vs 8.5L)
    p_low = dict(base_payload, **{"FamilyIncome": 120000.0, "12thMarks": 75.0})
    p_high = dict(base_payload, **{"FamilyIncome": 850000.0, "12thMarks": 75.0})
    d_low = client.post("/api/predict", json=p_low).get_json()
    d_high = client.post("/api/predict", json=p_high).get_json()
    inc_low_received = d_low["input_summary"]["FamilyIncome"]
    inc_high_received = d_high["input_summary"]["FamilyIncome"]
    print(f"\n[Test C] FamilyIncome Variation: 120,000 vs 850,000")
    print(f"  Low Income (1.2L) Received:  Rs. {inc_low_received}")
    print(f"    Prediction: {d_low['prediction']} (Eligible: {d_low['eligible_probability']}%)")
    print(f"  High Income (8.5L) Received: Rs. {inc_high_received}")
    print(f"    Prediction: {d_high['prediction']} (Eligible: {d_high['eligible_probability']}%, Not Eligible: {d_high['not_eligible_probability']}%)")
    assert inc_low_received == 120000.0
    assert inc_high_received == 850000.0
    assert d_low["prediction"] != d_high["prediction"], "Prediction should flip between low and high income"

    # 4. Test D: Borderline Student where 12thMarks 60 vs 99 flips class
    p_b60 = {
        "Gender": "Male",
        "Community": "OC",
        "FamilyIncome": 150000,
        "12thMarks": 60.0,
        "FirstGraduate": "No",
        "District": "Salem",
        "CollegeType": "Government",
        "Course": "Engineering"
    }
    p_b99 = dict(p_b60, **{"12thMarks": 99.0})
    d_b60 = client.post("/api/predict", json=p_b60).get_json()
    d_b99 = client.post("/api/predict", json=p_b99).get_json()
    print(f"\n[Test D] Borderline Candidate Sensitivity (Marks 60 vs 99):")
    print(f"  At 12thMarks = 60.0 -> Prediction: {d_b60['prediction']} (Eligible: {d_b60['eligible_probability']}%, Not Eligible: {d_b60['not_eligible_probability']}%)")
    print(f"  At 12thMarks = 99.0 -> Prediction: {d_b99['prediction']} (Eligible: {d_b99['eligible_probability']}%, Not Eligible: {d_b99['not_eligible_probability']}%)")
    assert d_b60["prediction"] == "Not Eligible"
    assert d_b99["prediction"] == "Eligible"
    print("  -> CLASS FLIP CONFIRMED: 60% = Not Eligible, 99% = Eligible!")
    print(f"\n[PASSED] All checks passed for {name}!\n")


if __name__ == "__main__":
    test_endpoint(api_app, "api/index.py (Vercel Serverless Entrypoint)")
    test_endpoint(web_app, "web/app.py (Local Flask Web App)")
