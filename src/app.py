"""
app.py
------
FastAPI service that serves churn predictions from the trained model.

Loads models/model.pkl, models/scaler.pkl, and
data/processed/feature_columns.json at startup, then exposes:

    GET  /health   basic liveness and model info check
    POST /predict  takes one raw customer record, returns a churn prediction

Run from the project root:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

IMPORTANT ENCODING NOTE
------------------------
prepare.py label-encodes twelve columns as if they were all binary, but
seven of them actually have three raw values in the Telco dataset:
MultipleLines has "No", "No phone service", "Yes". OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and
StreamingMovies each have "No", "No internet service", "Yes".
scikit-learn's LabelEncoder sorts alphabetically, so these became 0, 1, 2
rather than clean 0/1. prepare.py never saved the fitted encoders or their
mappings to disk, only the final column names in feature_columns.json.

The mappings below are hardcoded to match what LabelEncoder would produce
on the standard Telco Customer Churn dataset, sorted alphabetically. This
is a documented assumption, not something read from a saved encoder. If
the raw data ever contains different category spellings, this will
silently produce wrong encodings. tests/test_api.py includes a check that
pins these exact values down.
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODELS_DIR = Path("models")
FEATURE_COLUMNS_PATH = Path("data/processed/feature_columns.json")
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns that are genuinely two-valued in the raw data.
# LabelEncoder sorts alphabetically: "No" < "Yes", "Female" < "Male".
TWO_VALUE_MAPS = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Columns that look binary but actually have a third "no service" value.
# Alphabetical order: "No" < "No phone service" < "Yes"
MULTILINES_MAP = {"No": 0, "No phone service": 1, "Yes": 2}

# Alphabetical order: "No" < "No internet service" < "Yes"
NO_INTERNET_MAP = {"No": 0, "No internet service": 1, "Yes": 2}
THREE_VALUE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# One-hot columns. Values must match the raw dataset's category strings
# exactly, since these build the feature_columns.json column names.
INTERNET_SERVICE_VALUES = ["DSL", "Fiber optic", "No"]
CONTRACT_VALUES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_VALUES = [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check",
]


class CustomerRecord(BaseModel):
    """One customer's raw attributes, matching the original Telco CSV
    columns (minus customerID and Churn, which don't apply to a new
    prediction request)."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High"]


def load_artifacts():
    """Load the trained model, scaler, and expected feature column order.
    Raises a clear error at startup if dvc repro hasn't been run yet."""
    model_path = MODELS_DIR / "model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Model or scaler not found in models/. Run 'dvc repro' or "
            "'dvc pull' first."
        )
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_COLUMNS_PATH} not found. Run 'dvc repro' first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


def encode_record(record: CustomerRecord, feature_columns: list) -> pd.DataFrame:
    """
    Turn one raw customer record into the exact same encoded row shape
    that prepare.py produces for training data. See the module docstring
    for why the two/three-value maps below are hardcoded rather than
    loaded from a saved encoder.
    """
    row = {}

    # Two-value columns
    for col, mapping in TWO_VALUE_MAPS.items():
        row[col] = mapping[getattr(record, col)]

    # SeniorCitizen is already 0/1 in the raw data
    row["SeniorCitizen"] = record.SeniorCitizen

    # Numeric passthrough columns (scaled below)
    row["tenure"] = record.tenure
    row["MonthlyCharges"] = record.MonthlyCharges
    row["TotalCharges"] = record.TotalCharges

    # Three-value "looks binary but isn't" columns
    row["MultipleLines"] = MULTILINES_MAP[record.MultipleLines]
    for col in THREE_VALUE_COLS:
        row[col] = NO_INTERNET_MAP[getattr(record, col)]

    # One-hot columns, start every possible dummy at 0
    for val in INTERNET_SERVICE_VALUES:
        row[f"InternetService_{val}"] = 0
    for val in CONTRACT_VALUES:
        row[f"Contract_{val}"] = 0
    for val in PAYMENT_METHOD_VALUES:
        row[f"PaymentMethod_{val}"] = 0

    # Then flip the one matching dummy to 1
    row[f"InternetService_{record.InternetService}"] = 1
    row[f"Contract_{record.Contract}"] = 1
    row[f"PaymentMethod_{record.PaymentMethod}"] = 1

    df = pd.DataFrame([row])

    # Reindex to the exact training column order. Any column the model
    # expects that we didn't build gets filled with 0, which should
    # never actually happen if the maps above are correct and complete.
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Encoding produced a row missing expected columns: {missing}",
        )
    df = df[feature_columns]
    return df


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

model = None
scaler = None
feature_columns = None
model_type = None


@app.on_event("startup")
def startup():
    global model, scaler, feature_columns, model_type
    model, scaler, feature_columns = load_artifacts()
    model_type = type(model).__name__


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": model_type,
        "n_features": len(feature_columns) if feature_columns else None,
    }


@app.get("/")
def root():
    return {
        "message": "Telco Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    X = encode_record(record, feature_columns)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    prob = float(model.predict_proba(X)[0, 1])
    pred = "Yes" if prob >= 0.5 else "No"

    if prob < 0.4:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
    )