"""
train.py
--------
Trains a churn prediction model and logs everything to MLflow.

Run from the project root:
    python src/train.py --model logistic       # baseline
    python src/train.py --model random_forest  # default

Outputs:
    models/model.pkl
    models/scaler.pkl
    reports/metrics.json
    mlruns/  (created automatically by MLflow)
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Paths
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
PARAMS_FILE   = Path("params.yaml")
REPORTS_DIR   = Path("reports")


def load_split(name: str):
    """Load a processed CSV split and return features and target separately."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    return df.drop(columns=["Churn"]), df["Churn"]


def load_params() -> dict:
    """
    Read hyperparameters from params.yaml.
    DVC tracks this file. Changing a value and running dvc repro
    automatically re-runs the train stage with the new values.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, params: dict):
    """
    Construct an unfitted scikit-learn model using params from params.yaml.
    class_weight='balanced' adjusts for the 26.5% class imbalance
    so the model does not just predict "no churn" for every row.
    """
    if model_name == "logistic":
        return LogisticRegression(
            C=params["logistic"]["C"],
            max_iter=params["logistic"]["max_iter"],
            class_weight=params["logistic"]["class_weight"],
            random_state=42,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["random_forest"]["n_estimators"],
            max_depth=params["random_forest"]["max_depth"],
            class_weight=params["random_forest"]["class_weight"],
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model '{model_name}'. Use 'logistic' or 'random_forest'.")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Return a dictionary of all five evaluation metrics.

    ROC-AUC is the primary metric for this project because it is
    robust to the class imbalance and uses predicted probabilities
    rather than hard labels.

    Recall on the churn class matters most from a business standpoint.
    A missed churner (false negative) costs more than a wasted
    retention offer (false positive).
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def main(model_name: str, experiment_name: str) -> None:
    params   = load_params()
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")

    print("=" * 48)
    print(f"  Model      : {model_name}")
    print(f"  Train rows : {len(X_train):,}")
    print(f"  Val rows   : {len(X_val):,}")
    print(f"  Params     : {params.get(model_name, {})}")
    print("=" * 48)

    # Scale numeric features.
    # fit_transform on training data learns the mean and std.
    # transform on val applies the same scale without re-fitting.
    # Re-fitting on val would be data leakage.
    scaler = StandardScaler()
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"]
                if c in X_train.columns]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])

    model = build_model(model_name, params)

    # Group all runs under one experiment name so they appear together
    # in the MLflow UI for easy comparison.
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        t0 = time.time()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        # predict_proba returns [[prob_class0, prob_class1], ...]
        # [:, 1] takes only the churn probability for every row

        duration = round(time.time() - t0, 2)
        metrics  = compute_metrics(y_val, y_pred, y_prob)

        # Log everything to MLflow
        mlflow.log_params({**params.get(model_name, {}), "model_type": model_name})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_sec", duration)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Print results to terminal
        print("\nValidation metrics:")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")
        print(f"  training time: {duration}s")

        # Save model and scaler for the API and evaluate.py
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,  MODELS_DIR / "model.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

        # Save metrics as JSON for DVC to track
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nModel saved to {MODELS_DIR / 'model.pkl'}")
        print("Run 'mlflow ui' and open http://localhost:5000 to compare experiments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest"],
        default="random_forest",
    )
    parser.add_argument("--experiment", default="churn-prediction")
    args = parser.parse_args()
    main(args.model, args.experiment)
