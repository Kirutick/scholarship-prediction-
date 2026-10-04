"""Live API verification - tests health, predict eligible, predict not eligible."""
import requests
import json

BASE = "http://127.0.0.1:5000"

# 1. Health check
print("=== HEALTH CHECK ===")
r = requests.get(f"{BASE}/api/health", timeout=5)
print(f"Status: {r.status_code}")
h = r.json()
print(f"  model_loaded: {h['model_loaded']}")
print(f"  algorithm:    {h['algorithm']}")
print(f"  test_accuracy:{h['test_accuracy']}")
print(f"  test_f1:      {h['test_f1_score']}")
print(f"  model_path:   {h['model_path']}")

# 2. Eligible student
print("\n=== PREDICTION TEST 1 - Expected: ELIGIBLE ===")
p1 = {
    "Gender": "Female", "Community": "SC", "FamilyIncome": 120000,
    "12thMarks": 88.5, "FirstGraduate": "Yes", "District": "Chennai",
    "CollegeType": "Government", "Course": "Engineering"
}
r1 = requests.post(f"{BASE}/api/predict", json=p1, timeout=5)
d1 = r1.json()
print(f"  HTTP Status:     {r1.status_code}")
print(f"  prediction:      {d1['prediction']}")
print(f"  is_eligible:     {d1['is_eligible']}")
print(f"  eligible_prob:   {d1['eligible_probability']}%")
print(f"  confidence:      {d1['confidence']}%")
print(f"  model_used:      {d1['model_used']}")
print(f"  decision_factors:{d1['decision_factors']}")

# 3. Not eligible student
print("\n=== PREDICTION TEST 2 - Expected: NOT ELIGIBLE ===")
p2 = {
    "Gender": "Male", "Community": "OC", "FamilyIncome": 850000,
    "12thMarks": 55.0, "FirstGraduate": "No", "District": "Coimbatore",
    "CollegeType": "Private", "Course": "Management"
}
r2 = requests.post(f"{BASE}/api/predict", json=p2, timeout=5)
d2 = r2.json()
print(f"  HTTP Status:     {r2.status_code}")
print(f"  prediction:      {d2['prediction']}")
print(f"  is_eligible:     {d2['is_eligible']}")
print(f"  not_elig_prob:   {d2['not_eligible_probability']}%")
print(f"  confidence:      {d2['confidence']}%")

# 4. Invalid payload test
print("\n=== PREDICTION TEST 3 - Invalid Payload (should return 400) ===")
r3 = requests.post(f"{BASE}/api/predict", json={"Gender": "Female"}, timeout=5)
d3 = r3.json()
print(f"  HTTP Status: {r3.status_code}")
print(f"  Error msg:   {d3['message']}")
