"""
monitor.py
----------
Drift detection using EvidentlyAI.

Compares the training data (reference) against the test data (current)
to detect if the feature distributions have shifted.

Run:
    python src/monitor.py

Output:
    reports/drift/data_drift_report.html
    reports/drift/drift_summary.json
"""

import json
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

# ── paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH   = Path("data/processed/train.csv")
TEST_PATH    = Path("data/processed/test.csv")
DRIFT_DIR    = Path("reports/drift")
REPORT_PATH  = DRIFT_DIR / "data_drift_report.html"
SUMMARY_PATH = DRIFT_DIR / "drift_summary.json"


def load_data():
    """Load reference (train) and current (test) datasets."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} not found. Run dvc pull first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"{TEST_PATH} not found. Run dvc pull first.")

    reference = pd.read_csv(TRAIN_PATH)
    current   = pd.read_csv(TEST_PATH)

    # Drop target column for drift analysis -- we compare feature distributions
    reference = reference.drop(columns=["Churn"], errors="ignore")
    current   = current.drop(columns=["Churn"], errors="ignore")

    print(f"Reference dataset: {reference.shape[0]:,} rows, {reference.shape[1]} features")
    print(f"Current dataset:   {current.shape[0]:,} rows, {current.shape[1]} features")
    return reference, current


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Generate HTML drift report using EvidentlyAI."""
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=current)
    report.save_html(str(REPORT_PATH))
    print(f"Drift report saved: {REPORT_PATH}")

    # Extract summary for logging
    result = report.as_dict()
    metrics = result.get("metrics", [])

    drift_detected = False
    drifted_features = 0
    total_features = 0
    found = False

    for metric in metrics:
        if metric.get("metric") == "DatasetDriftMetric":
            r = metric.get("result", {})
            drifted_features = r.get("number_of_drifted_columns", 0)
            total_features = r.get("number_of_columns", 0)
            drift_detected = r.get("dataset_drift", False)
            found = True
            break

    if not found:
        raise RuntimeError("DatasetDriftMetric not found in report output")

    summary = {
        "drift_detected":   drift_detected if "drift_detected" in dir() else False,
        "drifted_features": drifted_features,
        "total_features":   total_features,
        "report_path":      str(REPORT_PATH),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDrift Summary:")
    print(f"  Drift detected:   {summary['drift_detected']}")
    print(f"  Drifted features: {summary['drifted_features']} / {summary['total_features']}")
    print(f"  Summary saved:    {SUMMARY_PATH}")

    return summary


if __name__ == "__main__":
    reference, current = load_data()
    summary = generate_drift_report(reference, current)

    if summary["drift_detected"]:
        print("\nWARNING: Significant data drift detected. Consider retraining.")
    else:
        print("\nNo significant drift detected. Model is stable.")
