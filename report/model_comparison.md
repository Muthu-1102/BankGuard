### Model Comparison: Baseline (LogisticRegression) vs Improved (LightGBM)

| Evaluation Metric | Baseline (LogisticRegression) | Improved (LightGBM) | Absolute Delta | Relative Change | Outcome |
|---|---:|---:|---:|---:|:---:|
| **PR-AUC (Average Precision)** | 0.1420 | 0.5598 | +0.4178 | +294.2% | [+] Improved |
| **ROC-AUC Score** | 0.7844 | 0.9136 | +0.1292 | +16.5% | [+] Improved |
| **Brier Score** | 0.1716 | 0.0212 | -0.1504 | -87.6% | [+] Improved |
| **Log Loss** | 0.6191 | 0.0867 | -0.5324 | -86.0% | [+] Improved |
| **Precision @ Threshold** | 0.1203 | 0.8107 | +0.6904 | +573.9% | [+] Improved |
| **Recall @ Threshold** | 0.6228 | 0.3573 | -0.2655 | -42.6% | [-] Regressed |
| **F1 Score @ Threshold** | 0.2017 | 0.4960 | +0.2943 | +145.9% | [+] Improved |
| **Specificity @ Threshold** | 0.8377 | 0.9970 | +0.1593 | +19.0% | [+] Improved |

#### Top 10 Predictive Features by Total Information Gain

| Rank | Feature | Total Information Gain | Relative Gain (%) |
|---:|---|---:|---:|
| 1 | `V258` | 64,408.9 | 8.07% |
| 2 | `C1` | 42,551.3 | 5.33% |
| 3 | `V257` | 42,526.5 | 5.33% |
| 4 | `C14` | 40,590.1 | 5.09% |
| 5 | `C13` | 28,956.7 | 3.63% |
| 6 | `V294` | 22,338.5 | 2.80% |
| 7 | `C4` | 22,135.4 | 2.77% |
| 8 | `card1` | 20,164.2 | 2.53% |
| 9 | `card2` | 18,332.0 | 2.30% |
| 10 | `TransactionAmt` | 17,794.8 | 2.23% |
