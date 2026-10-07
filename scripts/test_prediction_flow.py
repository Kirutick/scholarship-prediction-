"""Exercise both Flask prediction endpoints and verify the model input frame."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import api.index as api_module
import web.app as web_module


FEATURES = [
    "Gender", "Community", "FamilyIncome", "12thMarks",
    "FirstGraduate", "District", "CollegeType", "Course",
]

BASE_PROFILE = {
    "Gender": "Female",
    "Community": "SC",
    "FamilyIncome": 50000,
    "12thMarks": 60.0,
    "FirstGraduate": "Yes",
    "District": "Chennai",
    "CollegeType": "Government",
    "Course": "Engineering",
}


class RecordingModel:
    """Record the exact frame passed to inference while delegating to the real model."""

    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        self.inputs = []

    def predict(self, frame):
        self.inputs.append(frame.copy())
        return self.model.predict(frame)

    def predict_proba(self, frame):
        return self.model.predict_proba(frame)


def test_endpoint(module, model_attribute, name):
    model = getattr(module, model_attribute)
    recorder = RecordingModel(model)
    setattr(module, model_attribute, recorder)
    try:
        client = module.app.test_client()
        results = {}
        profiles = {
            "marks_60": dict(BASE_PROFILE),
            "marks_99": dict(BASE_PROFILE, **{"12thMarks": 99.0}),
            "income_50000": dict(BASE_PROFILE, **{"12thMarks": 75.0}),
            "income_500000": dict(BASE_PROFILE, FamilyIncome=500000, **{"12thMarks": 75.0}),
            "community_BC": dict(BASE_PROFILE, Community="BC"),
            "first_graduate_no": dict(BASE_PROFILE, FirstGraduate="No"),
            "course_management": dict(BASE_PROFILE, Course="Management"),
        }

        for case_name, payload in profiles.items():
            response = client.post("/api/predict", json=payload)
            result = response.get_json()
            assert response.status_code == 200, (name, case_name, result)
            assert result["status"] == "success"
            assert result["input_summary"] == payload
            assert len(recorder.inputs) == len(results) + 1

            received = recorder.inputs[-1]
            assert list(received.columns) == FEATURES
            for feature in FEATURES:
                expected_value = payload[feature]
                actual_value = received.iloc[0][feature]
                if feature in ("FamilyIncome", "12thMarks"):
                    assert float(actual_value) == float(expected_value), (case_name, feature, actual_value)
                else:
                    assert actual_value == expected_value, (case_name, feature, actual_value)

            assert result["prediction"] == model.predict(received)[0]
            assert result["eligible"] is result["is_eligible"]
            assert isinstance(result["potential_scholarships"], list)
            assert result["estimated_support"].startswith("Not configured")
            assert all(item["estimated_amount"] is None for item in result["potential_scholarships"])
            results[case_name] = result
            print(
                f"{name} {case_name}: {result['prediction']}, "
                f"eligible probability {result['eligible_probability']}%, "
                f"all {len(FEATURES)} features received by model"
            )

        assert results["marks_60"]["eligible_probability"] != results["marks_99"]["eligible_probability"]
        assert (
            results["income_50000"]["eligible_probability"]
            != results["income_500000"]["eligible_probability"]
        )

        for field, value in (("FamilyIncome", float("inf")), ("12thMarks", float("nan"))):
            invalid_profile = dict(BASE_PROFILE, **{field: value})
            response = client.post("/api/predict", json=invalid_profile)
            assert response.status_code == 400, (name, field, response.get_json())
    finally:
        setattr(module, model_attribute, model)


if __name__ == "__main__":
    test_endpoint(api_module, "_LOADED_MODEL", "Vercel Flask API")
    test_endpoint(web_module, "LOADED_MODEL", "Local Flask API")
    print("All prediction flow checks passed.")
