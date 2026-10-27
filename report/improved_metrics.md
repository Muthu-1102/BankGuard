### Improved — Evaluation Summary

- **Primary Fraud Metric (PR-AUC / Average Precision):** **0.5598** (Baseline random guess: 0.0344)
- **ROC-AUC Score:** **0.9136**
- **Brier Calibration Score:** **0.0212** (Lower is better; 0.0 = perfect)
- **Log Loss:** 0.0867

#### Classification Metrics (Threshold = 0.5)

| Metric | Score | Operational Context |
|---|---:|---|
| **Precision** | **81.07%** | Percentage of flagged transactions that were actually fraudulent |
| **Recall (Sensitivity)** | **35.73%** | Percentage of true fraudulent transactions successfully detected |
| **F1 Score** | **0.4960** | Harmonic mean of precision and recall |
| **Specificity** | **99.70%** | Percentage of legitimate transactions correctly approved |
| **False Positive Rate** | **0.30%** | Rate of benign transactions incorrectly challenged |

#### Confusion Matrix (Validation Set: 118,108 transactions)

| | Predicted Legitimate | Predicted Fraud | Total Actual |
|---|---:|---:|---:|
| **Actual Legitimate** | 113,705 (TN) | 339 (FP) | 114,044 |
| **Actual Fraud** | 2,612 (FN) | 1,452 (TP) | 4,064 |
| **Total Predicted** | 116,317 | 1,791 | 118,108 |
