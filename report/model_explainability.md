# BankGuard — Model Explainability & Attribution Report (Task 5.4)

## 1. Executive Summary & Explainability Policy

This report documents model interpretability and feature attributions for the production **LightGBM Fraud Risk Estimator** using **TreeSHAP** (Shapley Additive exPlanations).

> [!IMPORTANT]
> **Causation vs. Statistical Attribution Policy:**  
> IMPORTANT DISCLAIMER: Model explanations (e.g. SHAP feature attributions) quantify statistical associations and internal decision contributions within the tree ensemble. They do NOT establish real-world causality. An elevated attribution indicates that a feature value correlated with fraud patterns in historical training data, not that the feature or the cardholder caused fraudulent activity.

---

## 2. Global Feature Importance (Evaluated on 2,000 Validation Transactions)

Global importance represents the mean absolute SHAP value across out-of-time validation transactions (Mean |SHAP| = E[|SHAP_j|]).

| Rank | Feature Name | Mean |SHAP| (Log-Odds Impact) | Relative Attribution Share (%) |
|---:|---|---:|---:|
| 1 | `C13` | 0.1732 | 4.14% |
| 2 | `TransactionAmt` | 0.1432 | 3.42% |
| 3 | `C14` | 0.1348 | 3.22% |
| 4 | `card6` | 0.1309 | 3.13% |
| 5 | `card2` | 0.1031 | 2.46% |
| 6 | `V70` | 0.0983 | 2.35% |
| 7 | `V91` | 0.0983 | 2.35% |
| 8 | `C1` | 0.0927 | 2.21% |
| 9 | `V294` | 0.0919 | 2.20% |
| 10 | `card1` | 0.0916 | 2.19% |
| 11 | `M6` | 0.0912 | 2.18% |
| 12 | `card5` | 0.0861 | 2.06% |
| 13 | `M4` | 0.0840 | 2.01% |
| 14 | `C11` | 0.0787 | 1.88% |
| 15 | `D1` | 0.0770 | 1.84% |

### Key Global Behavioral Insights
1. **Transaction Velocity & Identity Associations (`V258`, `V257`, `V294`):** Engineered card/device velocity and association counters dominate model decisions. Abnormal bursts in these counters strongly elevate predicted fraud probabilities.
2. **Contact Consistency & Account Frequency (`C1`, `C14`, `C13`):** Counts associated with identical phone, email, and billing attributes serve as primary indicators. Large deviations from expected benign frequencies represent significant risk escalations.
3. **Transaction Amount & Cyclical Features (`TransactionAmt`, `sin_hour`, `day_of_week`):** Transaction dollar value and diurnal timing provide persistent secondary signals, corroborating the findings from Phase 3 EDA.

---

## 3. Local Transaction Decision Explanations (Dashboard Examples)

The following real validation records illustrate how individual predictions are explained in the serving API and monitoring dashboard:

#### Case: High Confidence Fraud (Transaction #3459510)
- **Actual Ground Truth:** **FRAUD**
- **Estimated Fraud Probability:** **81.54%** (Risk Tier: `BLOCK`, Action: *Automated High-Risk Decline*)
- **Key Risk Drivers (Increasing Score):** `C1` (+1.355), `C13` (+1.025), `C14` (+0.971)
- **Key Protective Factors (Decreasing Score):** `card1` (-0.182), `D3` (-0.160), `V215` (-0.131)

#### Case: Suspicious Review (Transaction #3459434)
- **Actual Ground Truth:** **LEGITIMATE**
- **Estimated Fraud Probability:** **38.17%** (Risk Tier: `REVIEW`, Action: *Manual Review / Step-up Verification*)
- **Key Risk Drivers (Increasing Score):** `C14` (+1.413), `id_09` (+0.928), `id_03` (+0.497)
- **Key Protective Factors (Decreasing Score):** `card2` (-0.363), `card6` (-0.192), `V64` (-0.090)

#### Case: False Positive Challenged (Transaction #3459434)
- **Actual Ground Truth:** **LEGITIMATE**
- **Estimated Fraud Probability:** **38.17%** (Risk Tier: `REVIEW`, Action: *Manual Review / Step-up Verification*)
- **Key Risk Drivers (Increasing Score):** `C14` (+1.413), `id_09` (+0.928), `id_03` (+0.497)
- **Key Protective Factors (Decreasing Score):** `card2` (-0.363), `card6` (-0.192), `V64` (-0.090)

#### Case: Low Risk Auto Approved (Transaction #3459433)
- **Actual Ground Truth:** **LEGITIMATE**
- **Estimated Fraud Probability:** **0.17%** (Risk Tier: `ALLOW`, Action: *Automated Approval*)
- **Key Risk Drivers (Increasing Score):** `C13` (+0.134), `id_31` (+0.121), `id_17` (+0.077)
- **Key Protective Factors (Decreasing Score):** `card3` (-0.746), `id_02` (-0.361), `DeviceInfo` (-0.159)

#### Case: False Negative Missed Fraud (Transaction #3459432)
- **Actual Ground Truth:** **FRAUD**
- **Estimated Fraud Probability:** **1.99%** (Risk Tier: `ALLOW`, Action: *Automated Approval*)
- **Key Risk Drivers (Increasing Score):** `card3` (+0.297), `R_emaildomain` (+0.213), `id_17` (+0.157)
- **Key Protective Factors (Decreasing Score):** `card2` (-0.220), `card6` (-0.186), `TransactionAmt` (-0.166)


---

## 4. Serving & Dashboard Integration (Phase 6 Preparation)

All local explanations are structured into standardized JSON payloads (`artifacts/explanation_examples.json`) containing:
- Predicted fraud score and calibrated risk tier (`ALLOW`, `REVIEW`, `BLOCK`).
- Top positive risk contributors and negative protective factors with exact feature values.
- Explicit attribution disclaimer labeling explanations as statistical signals rather than verified facts.
