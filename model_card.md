# Model Card: Telco Customer Churn Prediction

## Model Details

- **Model type:** Logistic Regression (scikit-learn `LogisticRegression`)
- **Version:** 1.0.0
- **Date:** July 2026, Phase 2, MAI201 MLOps
- **Developed by:** Group 2, MAI201, Seneca Polytechnic. Devreet Kaur (ML Lead), Arushi Anand (Engineering Lead), Cha Li (Project and Docs Lead)
- **Framework:** scikit-learn
- **Hyperparameters:** C=1.0, max_iter=200, class_weight=balanced, random_state=42

**Why Logistic Regression over Random Forest.** Both models were trained and compared during Phase 1. Random Forest scored slightly higher on accuracy and precision, but Logistic Regression had meaningfully higher recall on the churn class (0.81 vs 0.74 on validation). In this business problem, a missed churner, a customer who leaves without any retention attempt, costs more than a wasted retention offer sent to someone who was never going to leave. Recall was treated as the deciding metric for that reason, not accuracy.

## Intended Use

**Primary use case.** Predicts the probability that an existing telecom customer will cancel their subscription, so a retention team can prioritize outreach toward customers most likely to churn.

**Intended users.** Internal customer retention and support staff, accessed through the `/predict` endpoint of the FastAPI service in this repository.

**Out of scope.**
- Not intended as the sole basis for automated account actions (automatic discounts, automatic cancellations, automatic account flags) without a human reviewing the recommendation first.
- Not validated on customer populations outside the training distribution, a different country's telecom market, a different pricing structure, or a customer base with different service offerings would need revalidation before this model is trusted on them.
- Not intended for use in any legally protected decision, credit, employment, insurance, or housing. This is a retention marketing tool, nothing more.

## Training Data

- **Source:** IBM Telco Customer Churn dataset, publicly available on Kaggle. Synthetic and anonymized, no real customer personal information.
- **Size:** 7,043 customers, 21 raw columns, 26 encoded feature columns after `prepare.py` runs.
- **Target:** `Churn`, Yes or No. Class balance is 73.5% No Churn, 26.5% Churned.
- **Split:** 70% train, 15% validation, 15% test, stratified on `Churn` so all three splits keep roughly the same 26.5% churn rate. `random_state=42` for reproducibility.

## Evaluation

The project's target thresholds, set during Phase 0, were ROC-AUC of at least 0.84 and churn recall of at least 0.70. Both are met on both the validation and test sets below.

**Validation set** (1,056 customers, used during model selection)

| Metric | Value |
|---|---|
| Accuracy | 0.7415 |
| Precision | 0.5078 |
| Recall | 0.8143 |
| F1 | 0.6255 |
| ROC-AUC | 0.8449 |

**Test set** (1,057 customers, held out, touched exactly once at the end of Phase 1)

| Metric | Value |
|---|---|
| Accuracy | 0.7474 |
| Precision | 0.5165 |
| Recall | 0.7794 |
| F1 | 0.6213 |
| ROC-AUC | 0.8448 |

**Why ROC-AUC and recall, not accuracy.** A model that predicted "no churn" for every single customer would already score about 73.5% accuracy, since that's the majority class, without catching a single real churner. ROC-AUC measures how well the model ranks churners above non-churners across every possible threshold, independent of that imbalance, and recall on the churn class directly measures how many actual churners the model successfully flags. Accuracy, precision, and F1 are reported above for completeness, but they were not the deciding factors during model selection.

## Ethical Considerations and Limitations

**No demographic fairness audit has been performed.** `gender` is used as a raw input feature, and the model's error rates have not been separately measured across gender, `SeniorCitizen`, or any other demographic subgroup. Before any use beyond this course project, a fairness audit across these groups should be run and reviewed.

**The dataset is synthetic.** It's IBM's public teaching dataset, not drawn from a real telecom's customer records. The performance numbers above describe how the model performs on this dataset specifically, they may not transfer directly to a real customer base without revalidation on real data first.

**An encoding limitation, documented so it isn't mistaken for a bug.** Seven of the twelve columns that look binary in the raw data (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) actually carry a third value, a "no service" state, in addition to Yes and No. These were label-encoded as 0, 1, 2 rather than one-hot encoded. That means the model technically treats "no internet service" as numerically sitting between "No" and "Yes" for these columns, rather than as its own independent category. This was a design choice made in `prepare.py` during Phase 1, not something introduced later, and it's called out here because a linear model like Logistic Regression is more sensitive to this kind of ordinal assumption than a tree-based model would be.

**Precision is moderate, by design.** At roughly 0.51 to 0.52, about half of customers the model flags as "will churn" will not actually churn. This isn't an error, it's the direct consequence of prioritizing recall, catching more true churners, over precision, minimizing false alarms. Retention teams using this model's output should plan outreach capacity with that ratio in mind rather than treating every flagged customer as a confirmed loss.

**No production drift testing yet.** This model has only been evaluated on a static, historical snapshot of data. Real churn drivers shift over time, pricing changes, competitor offers, seasonal patterns. Ongoing monitoring (see `src/monitor.py`, EvidentlyAI drift detection, Phase 2 Part 3) is required to catch when live customer patterns start to diverge from what this model was trained on.

## Caveats and Recommendations

- The assumption that a missed churner costs more than a wasted retention offer was a reasoned business judgment made by the team, not something empirically measured against real retention-offer costs. Validate that assumption with real numbers before relying on this model outside a course setting.
- Re-run the fairness audit described above before any use involving real customers.
- Retrain on a recurring schedule rather than treating this as a one-time model, see `src/retrain.py`.
- The API layer (`src/app.py`) validates every categorical input strictly and returns a 422 error for anything unexpected. Any system calling this API should handle that error case gracefully rather than assuming every request succeeds.

## Model Card Authors

Devreet Kaur, ML Lead, MAI201 Group 2, Seneca Polytechnic, Summer 2026
