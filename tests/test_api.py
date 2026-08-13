"""
test_api.py
-----------
pytest tests for src/app.py, the FastAPI churn prediction service.

Run from the project root:
    pytest tests/test_api.py -v

These tests use FastAPI's TestClient, which calls the app directly in
memory rather than over a real network connection, so no need to have
uvicorn running separately first. They do need a real, already-trained
model though: run 'dvc repro' or 'dvc pull' before running these tests,
the same as the API itself needs.
"""

import pytest
from fastapi.testclient import TestClient

from src.app import (
    MULTILINES_MAP,
    NO_INTERNET_MAP,
    TWO_VALUE_MAPS,
    CustomerRecord,
    app,
    encode_record,
)


@pytest.fixture(scope="module")
def client():
    """FastAPI's TestClient only fires @app.on_event("startup") (which loads
    the model, scaler, and feature_columns) when used as a context manager.
    A plain `TestClient(app)` with no `with` block skips startup entirely,
    leaving model/scaler/feature_columns as None, which is why every test
    below needs to take this fixture as an argument rather than relying on
    a module-level client."""
    with TestClient(app) as c:
        yield c


# A known-good customer record, reused across several tests so each test
# only needs to change the one or two fields it actually cares about.
BASE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1020.0,
}


# ---------------------------------------------------------------------------
# Health and root endpoints
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 26


def test_health_reports_model_type(client):
    """model_type should be read off the real loaded model, not hardcoded,
    so this stays correct if the team ever switches models."""
    response = client.get("/health")
    body = response.json()
    assert body["model_type"] in ("LogisticRegression", "RandomForestClassifier")


def test_root_lists_docs_and_predict(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "docs" in body
    assert "predict" in body


# ---------------------------------------------------------------------------
# /predict: valid requests
# ---------------------------------------------------------------------------

def test_predict_valid_customer_returns_200(client):
    response = client.post("/predict", json=BASE_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_has_expected_shape(client):
    response = client.post("/predict", json=BASE_CUSTOMER)
    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert "risk_level" in body
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_high_risk_customer(client):
    """Fiber optic, month-to-month, electronic check, no security add-ons.
    These are the exact patterns the EDA flagged as the strongest churn
    predictors (Contract_Month-to-month had r=0.41, the single strongest
    correlation in the dataset), so this customer should score as
    meaningfully likely to churn, not just technically over 0.5."""
    response = client.post("/predict", json=BASE_CUSTOMER)
    body = response.json()
    assert body["churn_probability"] > 0.5
    assert body["churn_prediction"] == "Yes"


def test_predict_low_risk_customer(client):
    """Long tenure, two-year contract, no internet service at all. This is
    close to the opposite profile of the high-risk case above, and should
    score with a low churn probability."""
    low_risk = dict(BASE_CUSTOMER)
    low_risk.update({
        "tenure": 60,
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": 1200.0,
    })
    response = client.post("/predict", json=low_risk)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] < 0.5
    assert body["churn_prediction"] == "No"


# ---------------------------------------------------------------------------
# /predict: validation failures
# ---------------------------------------------------------------------------

def test_predict_invalid_gender_rejected(client):
    bad_customer = dict(BASE_CUSTOMER)
    bad_customer["gender"] = "female"  # wrong case, Literal is case-sensitive
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_unknown_category_rejected(client):
    bad_customer = dict(BASE_CUSTOMER)
    bad_customer["InternetService"] = "Satellite"  # not a real category
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_missing_field_rejected(client):
    incomplete = dict(BASE_CUSTOMER)
    del incomplete["tenure"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_negative_tenure_rejected(client):
    bad_customer = dict(BASE_CUSTOMER)
    bad_customer["tenure"] = -5
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_negative_monthly_charges_rejected(client):
    bad_customer = dict(BASE_CUSTOMER)
    bad_customer["MonthlyCharges"] = -10.0
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_senior_citizen_must_be_zero_or_one(client):
    bad_customer = dict(BASE_CUSTOMER)
    bad_customer["SeniorCitizen"] = 2
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# encode_record: pins down the exact encoding mappings this file was
# verified against on July 20, 2026 using the team's real raw CSV. If
# prepare.py or the raw data ever changes, this is the test that should
# fail first and loudest.
# ---------------------------------------------------------------------------

def test_two_value_maps_match_verified_encoding(client):
    assert TWO_VALUE_MAPS["gender"] == {"Female": 0, "Male": 1}
    assert TWO_VALUE_MAPS["Partner"] == {"No": 0, "Yes": 1}
    assert TWO_VALUE_MAPS["Dependents"] == {"No": 0, "Yes": 1}
    assert TWO_VALUE_MAPS["PhoneService"] == {"No": 0, "Yes": 1}
    assert TWO_VALUE_MAPS["PaperlessBilling"] == {"No": 0, "Yes": 1}


def test_multilines_map_matches_verified_encoding(client):
    assert MULTILINES_MAP == {"No": 0, "No phone service": 1, "Yes": 2}


def test_no_internet_map_matches_verified_encoding(client):
    assert NO_INTERNET_MAP == {"No": 0, "No internet service": 1, "Yes": 2}


def test_encode_record_produces_correct_column_count(client):
    record = CustomerRecord(**BASE_CUSTOMER)
    from src.app import feature_columns
    encoded = encode_record(record, feature_columns)
    assert encoded.shape == (1, 26)


def test_encode_record_sets_correct_one_hot_columns(client):
    record = CustomerRecord(**BASE_CUSTOMER)
    from src.app import feature_columns
    encoded = encode_record(record, feature_columns)

    assert encoded["InternetService_Fiber optic"].iloc[0] == 1
    assert encoded["InternetService_DSL"].iloc[0] == 0
    assert encoded["InternetService_No"].iloc[0] == 0

    assert encoded["Contract_Month-to-month"].iloc[0] == 1
    assert encoded["Contract_One year"].iloc[0] == 0
    assert encoded["Contract_Two year"].iloc[0] == 0

    assert encoded["PaymentMethod_Electronic check"].iloc[0] == 1
    assert encoded["PaymentMethod_Mailed check"].iloc[0] == 0


def test_encode_record_label_encodes_binary_columns_correctly(client):
    record = CustomerRecord(**BASE_CUSTOMER)
    from src.app import feature_columns
    encoded = encode_record(record, feature_columns)

    assert encoded["gender"].iloc[0] == 0
    assert encoded["Partner"].iloc[0] == 1
    assert encoded["OnlineSecurity"].iloc[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])