# Experiment Comparison

| Experiment | Model | Key Params | Accuracy | Precision | Recall (Churn) | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | C=1.0, max_iter=200, class_weight=balanced | 0.74 | 0.51 | 0.81 | 0.63 | 0.84 |
| 2 | Random Forest (baseline) | n_estimators=200, max_depth=10, class_weight=balanced | 0.77 | 0.55 | 0.74 | 0.63 | **0.84** |
| 3 | Random Forest (tuned) | n_estimators=300, max_depth=15, class_weight=balanced | **0.79** | **0.60** | 0.58 | 0.59 | 0.83 |

## Best model: Random Forest (baseline, Experiment 2)

Tuning the Random Forest to a higher `max_depth` (10 → 15) and more estimators (200 → 300) increased accuracy and precision, but **reduced churn recall from 0.74 to 0.58** and slightly lowered ROC-AUC. Deeper trees fit the majority "no churn" class more precisely, at the cost of missing more actual churners, the opposite of what this project needs, since a missed churner (false negative) is more costly than a wasted retention offer (false positive).

Against the Phase 0 proposal's targets (ROC-AUC ≥ 0.84, churn recall ≥ 0.70), **only Experiments 1 and 2 clear the recall bar**; Experiment 3 falls short at 0.58. Between those two, the baseline Random Forest (Experiment 2) is the strongest overall it matches Logistic Regression's ROC-AUC (0.84) while improving accuracy, precision, and F1, at the cost of some recall (0.74 vs 0.81).

## Metric Used for Model Selection

**ROC-AUC and churn recall** were used as the primary metrics for comparing and selecting between experiments, not accuracy.

This dataset is imbalanced (~73.5% no-churn / 26.5% churned), which makes accuracy a misleading metric: a model that predicts "no churn" for every customer would still score ~73.5% accuracy while catching zero actual churners. ROC-AUC measures how well the model ranks churners above non-churners across all thresholds, independent of class balance, making it a more reliable signal of true model quality here.

Recall on the churn class was weighted just as heavily because of the asymmetric business cost in this problem: a **false negative** (a churner the model misses) means a lost customer with no retention attempt made, while a **false positive** (a loyal customer flagged as at-risk) only costs an unnecessary retention offer. Missing a churner is more expensive than a wasted discount, so a model with high recall even at some cost to precision is more aligned with the business goal. This matches the targets set in the Phase 0 proposal: ROC-AUC ≥ 0.84 and churn recall ≥ 0.70.

Accuracy, precision, and F1 are still reported for completeness, but were treated as secondary useful context, not the deciding factor.