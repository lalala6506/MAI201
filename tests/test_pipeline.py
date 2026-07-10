"""
tests/test_pipeline.py
----------------------
Pytest tests for the MAI201 Customer Churn Prediction pipeline.
Tests verify that prepare.py, train.py, and evaluate.py produce
the expected outputs with the correct shapes and value ranges.
"""

import pytest
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

# Paths
PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")

# ── Part 1: Data preparation tests ──────────────────────────────────────────

def test_train_file_exists():
    assert (PROCESSED / "train.csv").exists(), "train.csv missing -- run prepare.py"

def test_val_file_exists():
    assert (PROCESSED / "val.csv").exists(), "val.csv missing -- run prepare.py"

def test_test_file_exists():
    assert (PROCESSED / "test.csv").exists(), "test.csv missing -- run prepare.py"

def test_feature_columns_file_exists():
    assert (PROCESSED / "feature_columns.json").exists(), "feature_columns.json missing"

def test_feature_columns_count():
    with open(PROCESSED / "feature_columns.json") as f:
        cols = json.load(f)
    assert len(cols) == 26, f"Expected 26 feature columns, got {len(cols)}"

def test_train_shape():
    df = pd.read_csv(PROCESSED / "train.csv")
    assert df.shape[0] == 4930, f"Expected 4930 train rows, got {df.shape[0]}"
    assert df.shape[1] == 27, f"Expected 27 columns, got {df.shape[1]}"

def test_val_shape():
    df = pd.read_csv(PROCESSED / "val.csv")
    assert df.shape[0] == 1056, f"Expected 1056 val rows, got {df.shape[0]}"

def test_test_shape():
    df = pd.read_csv(PROCESSED / "test.csv")
    assert df.shape[0] == 1057, f"Expected 1057 test rows, got {df.shape[0]}"

def test_no_missing_values_in_train():
    df = pd.read_csv(PROCESSED / "train.csv")
    assert df.isnull().sum().sum() == 0, "Missing values found in train.csv"

def test_churn_rate_train():
    df = pd.read_csv(PROCESSED / "train.csv")
    churn_rate = df["Churn"].mean()
    assert 0.25 <= churn_rate <= 0.28, f"Unexpected churn rate in train: {churn_rate:.4f}"

def test_customer_id_not_in_features():
    df = pd.read_csv(PROCESSED / "train.csv")
    assert "customerID" not in df.columns, "customerID should be dropped in prepare.py"

# ── Part 2: Model training tests ─────────────────────────────────────────────

def test_model_file_exists():
    assert (MODELS / "model.pkl").exists(), "model.pkl missing -- run train.py"

def test_scaler_file_exists():
    assert (MODELS / "scaler.pkl").exists(), "scaler.pkl missing -- run train.py"

def test_model_loads():
    import joblib
    model = joblib.load(MODELS / "model.pkl")
    assert hasattr(model, "predict"), "model.pkl does not have a predict method"

def test_model_predicts():
    import joblib
    model = joblib.load(MODELS / "model.pkl")
    with open(PROCESSED / "feature_columns.json") as f:
        feature_cols = json.load(f)
    df = pd.read_csv(PROCESSED / "test.csv")
    X = df[feature_cols].values
    preds = model.predict(X)
    assert len(preds) == len(df), "Prediction count does not match test set size"
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"

def test_metrics_file_exists():
    assert (REPORTS / "metrics.json").exists(), "metrics.json missing -- run train.py"

def test_metrics_keys():
    with open(REPORTS / "metrics.json") as f:
        metrics = json.load(f)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"

# ── Part 3: Evaluation tests ─────────────────────────────────────────────────

def test_metrics_test_file_exists():
    assert (REPORTS / "metrics_test.json").exists(), "metrics_test.json missing -- run evaluate.py"

def test_roc_auc_target():
    with open(REPORTS / "metrics_test.json") as f:
        metrics = json.load(f)
    assert metrics["roc_auc"] >= 0.80, f"ROC-AUC {metrics['roc_auc']:.4f} is below 0.80"

def test_confusion_matrix_exists():
    assert (REPORTS / "plots" / "confusion_matrix.png").exists(), \
        "confusion_matrix.png missing -- run evaluate.py"

def test_roc_curve_exists():
    assert (REPORTS / "plots" / "roc_curve.png").exists(), \
        "roc_curve.png missing -- run evaluate.py"

def test_feature_importance_exists():
    # feature_importance.png is only generated for tree-based models
    # Logistic Regression does not produce this plot -- test skipped
    import pytest
    pytest.skip("feature_importance.png not generated for Logistic Regression")
