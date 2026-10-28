# BankGuard — Operational Decision Policy Report (Task 6.1)

## 1. Policy Overview & Architecture

The BankGuard Decision Policy Engine translates continuous fraud probability predictions $P(\text{fraud} \mid X) \in [0.0, 1.0]$ into actionable operational recommendations.

- **Policy Name:** `bankguard_three_tier_policy` (v1.0.0)
- **Review Threshold ($\tau_{\text{review}}$):** **`0.10`**
- **Block Threshold ($\tau_{\text{block}}$):** **`0.40`**

### Decision Boundary Definitions

| Tier | Mathematical Range | Action | Operational Target |
|---|---|---|---|
| **`allow`** | $\text{score} < 0.10$ | **Automated Approval** | Fast-path authorization for ultra-low risk transactions |
| **`review`** | $0.10 \le \text{score} < 0.40$ | **Step-Up Verification** | Queue for human review or trigger dynamic 3D-Secure challenge |
| **`high_risk`** | $\text{score} \ge 0.40$ | **Automated Decline** | Immediate hard decline on high-certainty fraud |

---

## 2. Validation Set Distribution (118,108 Out-of-Time Transactions)

| Decision Recommendation | Transaction Count | Volume Share (%) | Standard Operational Action |
|---|---:|---:|---|
| **`allow`** | **111,517** | **94.42%** | Automated Approval |
| **`review`** | **4,427** | **3.75%** | Manual Review / Step-up Verification |
| **`high_risk`** | **2,164** | **1.83%** | Automated Decline |

### Validation Set Ground Truth Breakdown

| Decision Tier | Total Flagged | Actual Legitimate | Actual Fraud | Tier Fraud Density (%) | Operational Fate |
|---|---:|---:|---:|---:|---|
| **`allow`** | 111,517 (94.42%) | 110,009 | 1,508 | 1.35% | Zero-friction authorization |
| **`review`** | 4,427 (3.75%) | 3,486 | 941 | 21.26% | Challenge via 3DS / Analyst Queue |
| **`high_risk`** | 2,164 (1.83%) | 549 | 1,615 | **74.63%** | Automated Immediate Decline |

- **Total Fraud Captured (Review + Block):** **62.89%** (2,556 out of 4,064 fraud transactions).
- **Auto-Decline Precision:** **74.63%** of automatically blocked transactions are verified fraud.
- **Escape Rate in Allow Tier:** Only **1.352%** of allowed transactions are fraudulent.

---

## 3. Operational Guarantees & Implementation Notes

1. **Strict Determinism:** The decision function is stateless and deterministic; identical input scores produce identical recommendations and actions.
2. **Boundary Precision:** Exactly equal-to-threshold scores trigger the higher risk tier:
   - $\text{score} == 0.10 \implies$ `review`
   - $\text{score} == 0.40 \implies$ `high_risk`
3. **Decoupled Configuration:** Thresholds are loaded from `configs/policy_config.yaml` at runtime, allowing operational teams to tune capacity constraints without code redeployment.
