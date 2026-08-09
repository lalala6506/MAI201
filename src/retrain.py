"""
retrain.py
----------
Retrains the model on the latest training data and compares
performance against the current deployed model.

If the new model has a higher ROC-AUC AND recall, it replaces the current model.

Run:
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python src/retrain.py

Output:
    - MLflow run logged under experiment "churn-prediction-retrain"
    - If new model is better: models/model.pkl and models/scaler.pkl updated
    - reports/metrics/retrain_comparison.json saved
"""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH    = Path("data/processed/train.csv")
TEST_PATH     = Path("data/processed/test.csv")
FEATURES_PATH = Path("data/processed/feature_columns.json")
MODEL_PATH    = Path("models/model.pkl")
SCALER_PATH   = Path("models/scaler.pkl")
METRICS_PATH  = Path("reports/metrics_test.json")
RETRAIN_PATH  = Path("reports/metrics/retrain_comparison.json")


def load_data():
    """Load train and test splits."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)

    X_train = train[feature_cols]
    y_train = train["Churn"]
    X_test  = test[feature_cols]
    y_test  = test["Churn"]

    return X_train, y_train, X_test, y_test


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def load_current_metrics() -> dict:
    """Load metrics from the currently deployed model."""
    if not METRICS_PATH.exists():
        print("No existing metrics found. Treating current model as baseline 0.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0}

    with open(METRICS_PATH) as f:
        return json.load(f)


def retrain(X_train, y_train, X_test, y_test) -> dict:
    """Retrain the model and return new metrics."""
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    return model, scaler, metrics


def print_comparison(current: dict, new: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "="*55)
    print(f"{'Metric':<15} {'Current Model':>18} {'New Model':>18}")
    print("="*55)
    for key in ["roc_auc", "recall", "f1", "accuracy", "precision"]:
        curr_val = current.get(key, 0)
        new_val  = new.get(key, 0)
        delta    = new_val - curr_val
        arrow    = "+" if delta > 0 else ""
        print(f"{key:<15} {curr_val:>18.4f} {new_val:>14.4f} ({arrow}{delta:.4f})")
    print("="*55)


if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    print("Loading current model metrics...")
    current_metrics = load_current_metrics()

    print("\nRetraining model...")
    with mlflow.start_run(run_name="retrain"):
        mlflow.set_tag("trigger", "scheduled_retrain")

        new_model, new_scaler, new_metrics = retrain(X_train, y_train, X_test, y_test)

        mlflow.log_params({"model_type": "logistic", "C": 1.0, "max_iter": 200})
        mlflow.log_metrics(new_metrics)
        mlflow.sklearn.log_model(new_model, "model")

    print_comparison(current_metrics, new_metrics)

    # Decide whether to replace
    improved = (
        new_metrics["roc_auc"] > current_metrics.get("roc_auc", 0)
        and new_metrics["recall"] > current_metrics.get("recall", 0)
    )

    if improved:
        print("\nNew model is better on both ROC-AUC and Recall. Replacing current model.")
        joblib.dump(new_model,  MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        print(f"Saved new model to {MODEL_PATH}")
    else:
        print("\nNew model is not better. Keeping current model.")

    # Save comparison report
    RETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison = {"current": current_metrics, "new": new_metrics, "replaced": improved}
    with open(RETRAIN_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {RETRAIN_PATH}")
