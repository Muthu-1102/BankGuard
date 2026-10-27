# BankGuard — Decision Threshold Evaluation & Selection Report (Task 5.3)

## 1. Executive Summary & Recommendation

In fraud risk estimation with extreme class imbalance (3.44% fraud prevalence), the conventional default decision threshold (tau = 0.50) is operationally sub-optimal because it requires an extreme degree of certainty before flagging an alert.

- **Recommended Primary Threshold:** **`tau = 0.22`** (Capacity-Constrained Review Strategy)
- **Selection Rule:** Maximize fraud detection recall while constraining manual investigation queue to <= 3.0% of total transaction volume.
- **Operational Performance at `tau = 0.22`:**
  - **Fraud Recall:** **49.75%** (2,022 out of 4,064 fraud cases successfully intercepted).
  - **Precision:** **57.82%** (More than 1 in 2 flagged transactions is true confirmed fraud).
  - **Manual Review Volume:** **3,497 transactions (2.96% of all volume)**.
  - **Benign Customer Disruption:** Specificity is **98.71%** (only 1,475 false positives out of 114,044 benign transactions; false positive rate = 1.29%).

---

## 2. Threshold Sweep Analysis (Validation Set: 118,108 Transactions)

| Decision Threshold (tau) | Flagged Volume (% Volume) | Precision | Recall (Sensitivity) | F1 Score | TP | FP | FN | Expected Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.02` | 25,914 (21.94%) | **13.38%** | **85.31%** | 0.2313 | 3,467 | 22,447 | 597 | $348,690 |
| `0.05` | 11,984 (10.15%) | **25.05%** | **73.87%** | 0.3741 | 3,002 | 8,982 | 1,062 | $279,140 |
| `0.08` | 7,905 (6.69%) | **34.23%** | **66.59%** | 0.4522 | 2,706 | 5,199 | 1,358 | $282,750 |
| `0.10` | 6,591 (5.58%) | **38.78%** | **62.89%** | 0.4798 | 2,556 | 4,035 | 1,508 | $292,110 |
| `0.20` | 3,785 (3.21%) | **55.59%** | **51.77%** | 0.5361 | 2,104 | 1,681 | 1,960 | $331,850 |
| `0.25` | 3,108 (2.63%) | **62.26%** | **47.61%** | 0.5396 | 1,935 | 1,173 | 2,129 | $350,430 |
| `0.30` | 2,696 (2.28%) | **67.43%** | **44.73%** | 0.5379 | 1,818 | 878 | 2,246 | $363,860 |
| `0.40` | 2,164 (1.83%) | **74.63%** | **39.74%** | 0.5186 | 1,615 | 549 | 2,449 | $388,990 |
| `0.50` | 1,791 (1.52%) | **81.07%** | **35.73%** | 0.4960 | 1,452 | 339 | 2,612 | $409,710 |
| `0.60` | 1,542 (1.31%) | **84.95%** | **32.23%** | 0.4674 | 1,310 | 232 | 2,754 | $428,520 |

---

## 3. Operational Objectives & Decision Rules

| Objective | Selection Rule | Optimal tau | Precision | Recall | F1 Score | Review Queue Volume |
|---|---|:---:|---:|---:|---:|---:|
| **Max F1-Score (Balanced Precision & Recall)** | `argmax(F1)` | `tau = 0.25` | 62.26% | 47.61% | 0.5396 | 3,108 (2.63%) |
| **Capacity-Constrained Review (Review Load <= 3.0%)** | `argmax(Recall) subject to flagged_rate <= 3.0%` | `tau = 0.22` | 57.82% | 49.75% | 0.5349 | 3,497 (2.96%) |
| **Target Recall Guarantee (Recall >= 70%)** | `argmax(Precision) subject to Recall >= 0.7` | `tau = 0.06` | 28.41% | 71.01% | 0.4058 | 10,159 (8.60%) |
| **Minimum Operational Cost (Loss + Review Cost)** | `argmin(Total Cost)` | `tau = 0.06` | 28.41% | 71.01% | 0.4058 | 10,159 (8.60%) |

---

## 4. Trade-off Analysis & Operational Context

### High Recall vs Review Queue Overload
- Lowering threshold to 0.05 elevates fraud recall to **73.87%**, but inflates the review queue to **10.15% of all transactions**, diluting precision to 25.05%.
- Setting threshold to 0.20 strikes a strong balance between precision (55.59%), recall (51.77%), and a lean review queue (3.20%).
- Setting threshold to 0.25 strictly enforces a review queue under **2.63%**, elevating precision to **62.26%** with **47.61%** recall.

### Baseline vs Improved Model Comparison across Thresholds
- To achieve **62.3% recall**, the Baseline Logistic Regression flagged **21,038 transactions (17.81% of total bank volume)** with an abysmal **12.03% precision** (18,507 false alarms).
- In contrast, the Improved LightGBM model achieves **62.89% recall** at tau = 0.10 by flagging only **6,591 transactions (5.58% volume)** with **38.78% precision** — **cutting false alarms by 78%**.

---

## 5. Multi-Tier Recommendation for Phase 6 Decision Policy

Based on this empirical threshold analysis, the fraud policy engine in Phase 6 will map continuous risk scores into three actionable tiers:

1. **Auto-Approve (`allow`):** `Score < 0.10`  
   - Accounts for ~94.4% of transactions.
   - Ultra-low risk; automated friction-free processing.
2. **Step-Up Verification / Manual Review (`review`):** `0.10 <= Score < 0.40`  
   - Accounts for ~3.7% of transactions.
   - Triggers SMS OTP challenge, biometric step-up, or analyst review queue.
3. **Automated Decline (`high_risk` / `block`):** `Score >= 0.40`  
   - Accounts for ~1.8% of transactions.
   - High certainty fraud (Precision >= 74.63%); transaction declined immediately to prevent loss.
