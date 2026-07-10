# Experiment Comparison

| Experiment | Model | Key Params | Accuracy | Precision | Recall (Churn) | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | C=1.0, max_iter=200, class_weight=balanced | 0.74 | 0.51 | 0.81 | 0.63 | 0.84 |
| 2 | Random Forest (baseline) | n_estimators=200, max_depth=10, class_weight=balanced | 0.77 | 0.55 | 0.74 | 0.63 | **0.84** |
| 3 | Random Forest (tuned) | n_estimators=300, max_depth=15, class_weight=balanced | **0.79** | **0.60** | 0.58 | 0.59 | 0.83 |

## Best model: Logistic Regression (Experiment 1)

Tuning the Random Forest to a higher max_depth (10 to 15) and more estimators (200 to 300) increased accuracy and precision, but reduced churn recall from 0.74 to 0.58 and slightly lowered ROC-AUC. Deeper trees fit the majority no-churn class more precisely, at the cost of missing more actual churners, the opposite of what this project needs.

Against the Phase 0 proposal targets (ROC-AUC >= 0.84, churn recall >= 0.70), only Experiments 1 and 2 clear the recall bar. Experiment 3 falls short at 0.58. Between Experiments 1 and 2, Logistic Regression is the stronger choice: it achieves the highest ROC-AUC (0.8449 vs 0.8398) and the highest churn recall (0.81 vs 0.74). Since a missed churner (false negative) costs more than a wasted retention offer (false positive), recall is the tiebreaker and Logistic Regression wins on both primary metrics.

## Metric Used for Model Selection

**ROC-AUC and churn recall** were used as the primary metrics, not accuracy.

This dataset is imbalanced (73.5% no-churn / 26.5% churned), which makes accuracy misleading: a model that predicts no churn for every customer would score 73.5% accuracy while catching zero actual churners. ROC-AUC measures how well the model ranks churners above non-churners across all thresholds, independent of class balance, making it a more reliable signal of true model quality.

Recall on the churn class was weighted equally because of the asymmetric business cost: a false negative (a churner the model misses) means a lost customer with no retention attempt, while a false positive (a loyal customer flagged as at-risk) only costs an unnecessary retention offer. Missing a churner is more expensive than a wasted discount, so a model with high recall even at some cost to precision is more aligned with the business goal. This matches the targets set in the Phase 0 proposal: ROC-AUC >= 0.84 and churn recall >= 0.70.

Accuracy, precision, and F1 are reported for completeness but were treated as secondary context, not the deciding factor.
