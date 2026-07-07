# MAI201 MLOps -- Customer Churn Prediction

Predicts which telecom customers are likely to cancel their subscription.
Phase 1 covers the dataset, architecture, DVC pipeline, and MLflow experiment tracking.

**Team:** Group 2 | Seneca Polytechnic | Summer 2026 | Instructor: Asma Azim

---

## Team

| Member | Role | Phase 1 Tasks |
|---|---|---|
| Devreet Kaur | ML Lead | EDA notebook, train.py, MLflow experiments 1 and 2, tech stack docs |
| Arushi Anand | Project + Docs Lead | Architecture diagram, dataset docs, evaluate.py, MLflow experiment 3 |
| Cha Li | Engineering Lead | prepare.py, DVC setup, dvc.yaml, reproducibility testing |

---

## Dataset

### Source and Overview

| Property | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | [Kaggle - IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Origin | IBM synthetic dataset. Publicly available, no privacy/licensing constraints. |
| Size | 7,043 rows × 21 columns (~955 KB) |
| Target variable | `Churn` — binary (`Yes` / `No`), encoded to `1` / `0` in `prepare.py` |
| Class balance | 73.5% No Churn / 26.5% Churned (moderate imbalance) |

### Features (20 total, + target)

| Category | Features |
|---|---|
| Demographic (4) | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account (4) | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Billing (2) | `MonthlyCharges`, `TotalCharges` |
| Services (10) | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Identifier (dropped) | `customerID` — unique per row, no predictive value |

12 of these are binary (Label-encoded) and 3 are multi-category (`InternetService`, `Contract`, `PaymentMethod` — one-hot encoded), giving 19 encoded feature columns after `prepare.py` runs.

### Data Quality Assessment

| Check | Finding | Action Taken |
|---|---|---|
| **Missing values** | `TotalCharges` is stored as a string; 11 rows have a blank value (all customers with `tenure = 0`, i.e. brand-new accounts not yet billed) | `pd.to_numeric(errors='coerce')` then `fillna(0.0)` in `prepare.py` |
| **Duplicates** | 0 duplicate rows found (`df.duplicated().sum() == 0`) | No action needed |
| **Outliers** | Numeric ranges: `tenure` 0–72 months, `MonthlyCharges` $18.25–$118.75, `TotalCharges` up to $8,684.80. Distributions are right-skewed but reflect real long-tenure/high-usage customers rather than data errors — no values fall outside plausible telecom billing ranges, so no rows were removed. `StandardScaler` is applied in `train.py` to control for the wide value ranges rather than treating them as outliers to drop. | None removed; scaled instead |
| **Type issues** | `TotalCharges` read as `object` instead of `float64` | Converted in `prepare.py` |
| **Identifier leakage risk** | `customerID` has no predictive signal and could act as a memorization shortcut | Dropped in `prepare.py` |

### Class Imbalance

26.5% churn rate is moderate, not severe. Rather than resampling (SMOTE), the pipeline uses `class_weight='balanced'` in both Logistic Regression and Random Forest, and prioritizes ROC-AUC and churn recall over raw accuracy — accuracy alone is misleading here since predicting "No Churn" for every row would already score ~73.5%.

### Train / Validation / Test Split

70% / 15% / 15%, stratified on `Churn` to preserve the 26.5% rate in every split (`prepare.py`, `random_state=42` for reproducibility). The test set is held out and touched only once, in `evaluate.py`, for final reporting.

### Data Versioning (DVC)

The raw CSV and every processed split are tracked with DVC rather than committed to Git directly:

```bash
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
dvc remote add -d storage <remote-url>
dvc push
```

This keeps the Git repo lightweight while still giving every team member and the CI pipeline an identical, reproducible copy of the data via `dvc pull`. The `prepare` stage in `dvc.yaml` regenerates `data/processed/{train,val,test}.csv` and `feature_columns.json` automatically whenever the raw file or `prepare.py` changes.

---
## Architecture

![Pipeline Diagram](docs/architecture.png)

The pipeline runs left to right: raw data is versioned and cleaned (`prepare.py`), trained with MLflow experiment tracking (`train.py`), scored on the held-out test set (`evaluate.py`), then served via FastAPI, containerized with Docker, and deployed to Render. A monitoring loop (EvidentlyAI drift checks) watches the deployed model in production and can trigger retraining. GitHub Actions runs the pipeline and tests on every push.