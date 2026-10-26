### Baseline — Evaluation Summary

- **Primary Fraud Metric (PR-AUC / Average Precision):** **0.1420** (Baseline random guess: 0.0344)
- **ROC-AUC Score:** **0.7844**
- **Brier Calibration Score:** **0.1716** (Lower is better; 0.0 = perfect)
- **Log Loss:** 0.6191

#### Classification Metrics (Threshold = 0.5)

| Metric | Score | Operational Context |
|---|---:|---|
| **Precision** | **12.03%** | Percentage of flagged transactions that were actually fraudulent |
| **Recall (Sensitivity)** | **62.28%** | Percentage of true fraudulent transactions successfully detected |
| **F1 Score** | **0.2017** | Harmonic mean of precision and recall |
| **Specificity** | **83.77%** | Percentage of legitimate transactions correctly approved |
| **False Positive Rate** | **16.23%** | Rate of benign transactions incorrectly challenged |

#### Confusion Matrix (Validation Set: 118,108 transactions)

| | Predicted Legitimate | Predicted Fraud | Total Actual |
|---|---:|---:|---:|
| **Actual Legitimate** | 95,537 (TN) | 18,507 (FP) | 114,044 |
| **Actual Fraud** | 1,533 (FN) | 2,531 (TP) | 4,064 |
| **Total Predicted** | 97,070 | 21,038 | 118,108 |
