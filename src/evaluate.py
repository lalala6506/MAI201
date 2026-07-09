"""
evaluate.py
-----------
Runs final evaluation on the held-out test set.
Use the test set only once, here, for final reporting.

Run from the project root:
    python src/evaluate.py

Outputs:
    reports/metrics_test.json
    reports/plots/confusion_matrix.png
    reports/plots/roc_curve.png
    reports/plots/feature_importance.png
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
REPORTS_DIR   = Path("reports")
PLOTS_DIR     = REPORTS_DIR / "plots"


def load_test():
    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return df.drop(columns=["Churn"]), df["Churn"]


def plot_confusion_matrix(y_true, y_pred, out: Path) -> None:
    """
    A confusion matrix shows four outcomes in a 2x2 grid.

    True Negative  = predicted no churn, actually stayed (correct)
    False Positive = predicted churn, actually stayed (wasted offer)
    False Negative = predicted no churn, actually left (missed churner)
    True Positive  = predicted churn, actually left (correct, can intervene)

    False negatives are the most costly. Missing a churner means losing
    a customer we could have retained.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix -- Test Set", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_roc_curve(y_true, y_prob, out: Path) -> None:
    """
    The ROC curve plots true positive rate against false positive rate
    at every possible threshold. The area under the curve (AUC) is
    the primary metric for this project. A score of 0.5 is random
    guessing. The project target is 0.84 or higher.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve -- Test Set", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_importance(model, feature_names: list, out: Path) -> None:
    """
    Feature importance shows which input columns the model relied on most.
    This is only available for tree-based models (Random Forest).
    Logistic Regression produces a coefficient plot instead, but this
    script skips it to keep things simple.
    """
    if not hasattr(model, "feature_importances_"):
        print("Skipping feature importance (not a tree-based model).")
        return
    s = pd.Series(model.feature_importances_, index=feature_names).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    s.plot(kind="barh", ax=ax, color="#4f86c6")
    ax.set_title("Top 15 Feature Importances", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    print("=" * 48)
    print("  Telco Churn -- Final Evaluation (Test Set)")
    print("=" * 48)

    model_path = MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at '{model_path}'. Run train.py or dvc repro first."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    X_test, y_test = load_test()
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"]
                if c in X_test.columns]
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    # Print a formatted table
    header = f"{'Metric':<12} {'Value'}"
    print(f"\n{header}")
    print("-" * 20)
    for k, v in metrics.items():
        print(f"{k:<12} {v}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Save metrics JSON for DVC tracking
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to reports/metrics_test.json")

    # Save plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob,        PLOTS_DIR / "roc_curve.png")
    plot_feature_importance(model, list(X_test.columns), PLOTS_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
