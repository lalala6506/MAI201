"""
prepare.py
----------
Cleans the raw Telco Customer Churn CSV, encodes all features,
and produces a stratified 70/15/15 train/val/test split.

Run from the project root:
    python src/prepare.py

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/feature_columns.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
RAW_PATH      = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = Path("data/processed")


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV and print a quick shape check."""
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the two known data quality issues in this dataset.

    Issue 1: TotalCharges is stored as a string.
        pd.to_numeric converts it to float. The errors='coerce' argument
        turns any non-numeric value into NaN instead of raising an error.

    Issue 2: 11 rows have a blank TotalCharges value.
        These all belong to customers with tenure = 0 (brand new customers
        who have not been billed yet). Filling with 0.0 is correct.

    customerID is dropped because it is a row identifier, not a feature.
    The model would learn to memorise IDs rather than real patterns.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=["customerID"])

    remaining_nulls = df.isnull().sum().sum()
    print(f"After cleaning: {remaining_nulls} nulls remaining")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text columns to numbers so scikit-learn can use them.

    Binary columns (two possible values) use LabelEncoder, which maps
    each value to 0 or 1.

    Multi-category columns (three or more values) use one-hot encoding
    via pd.get_dummies, which creates one new binary column per value.
    For example, Contract becomes three columns:
        Contract_Month-to-month, Contract_One year, Contract_Two year.

    The target column (Churn) is mapped manually: Yes to 1, No to 0.
    """
    # Target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary columns
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Multi-category columns
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # get_dummies produces bool columns; convert to int for scikit-learn
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"After encoding: {df.shape[1]} columns total")
    return df


def split_and_save(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Split into train (70%), val (15%), test (15%) and save each as a CSV.

    Stratified splitting preserves the 26.5% churn rate in every subset.
    Without stratification, random chance could put most churners in one
    split and almost none in another.

    The test set is for final evaluation only. Nothing in the training
    process should ever touch it until the very last step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # First cut: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Second cut: split temp evenly into val and test (each 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    splits = [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]

    for name, X_split, y_split in splits:
        out = pd.concat([X_split, y_split], axis=1)
        out.to_csv(out_dir / f"{name}.csv", index=False)
        rate = y_split.mean() * 100
        print(f"  {name}: {len(out):,} rows, churn rate {rate:.1f}%")

    # Save the ordered list of feature columns.
    # train.py and app.py both read this file so the column order
    # is always consistent across scripts.
    feature_cols = list(X.columns)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Saved {len(feature_cols)} feature columns to feature_columns.json")


def main(raw_path: Path) -> None:
    print("=" * 48)
    print("  Telco Churn -- Data Preparation")
    print("=" * 48)
    df = load_data(raw_path)
    df = clean(df)
    df = encode(df)
    split_and_save(df, PROCESSED_DIR)
    print("\nDone. Check data/processed/ for outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    main(args.input)
