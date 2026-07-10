# Experiment Comparison

| Experiment | Model | Key Params | Accuracy | Precision | Recall (Churn) | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | C=1.0, max_iter=200, class_weight=balanced | 0.74 | 0.51 | 0.81 | 0.63 | 0.84 |
| 2 | Random Forest (baseline) | n_estimators=200, max_depth=10, class_weight=balanced | 0.77 | 0.55 | 0.74 | 0.63 | **0.84** |
| 3 | Random Forest (tuned) | n_estimators=300, max_depth=15, class_weight=balanced | **0.79** | **0.60** | 0.58 | 0.59 | 0.83 |

## Best model: Random Forest (baseline, Experiment 2)

Tuning the Random Forest to a higher `max_depth` (10 → 15) and more estimators (200 → 300) increased accuracy and precision, but **reduced churn recall from 0.74 to 0.58** and slightly lowered ROC-AUC. Deeper trees fit the majority "no churn" class more precisely, at the cost of missing more actual churners, the opposite of what this project needs, since a missed churner (false negative) is more costly than a wasted retention offer (false positive).

Against the Phase 0 proposal's targets (ROC-AUC ≥ 0.84, churn recall ≥ 0.70), **only Experiments 1 and 2 clear the recall bar**; Experiment 3 falls short at 0.58. Between those two, the baseline Random Forest (Experiment 2) is the strongest overall it matches Logistic Regression's ROC-AUC (0.84) while improving accuracy, precision, and F1, at the cost of some recall (0.74 vs 0.81).

ROC-AUC and recall were prioritized over raw accuracy throughout, since the ~26.5% churn class imbalance makes accuracy misleading a model predicting "no churn" for every customer would still score roughly 73.5% accuracy while catching zero actual churners.
