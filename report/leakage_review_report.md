# BankGuard — Leakage Risk Audit & Feature Availability Review (Task 4.3)

This report provides the formal leakage review, split integrity audit, and feature availability policy for the BankGuard fraud risk scoring pipeline.

---

## 1. Audit Summary

| Check | Requirement | Result | Status |
|---|---|---|:---:|
| **Target Exclusion** | `isFraud` strictly excluded from model feature matrix | Target absent: `True` | **PASSED** |
| **Identifier Exclusion** | `TransactionID` excluded from model inputs | ID absent: `True` | **PASSED** |
| **Raw Timestamp Exclusion** | Raw monotonic `TransactionDT` excluded | Raw time absent: `True` | **PASSED** |
| **Temporal Split Ordering** | Train max timestamp <= Val min timestamp | Train max: 12,192,842 <= Val min: 12,192,900 | **PASSED** |
| **Cross-Split Contamination** | 0 overlapping `TransactionID`s between train and val | Overlap count: `0` | **PASSED** |
| **Overall Audit Status** | Zero leakage violations detected across all gates | Violations: `0` | **PASSED** |

---

## 2. Excluded Fields Registry & Policy Rationale

Every excluded field is documented below with its specific risk profile and production scoring constraint:

### `isFraud`
- **Risk Classification:** Direct Target Leakage
- **Available at Scoring Moment:** `No (Arrives post-event via chargeback)`
- **Detailed Rationale:** The ground truth label 'isFraud' represents the classification outcome. In production, chargebacks and fraud confirmation typically arrive 30–90 days post-transaction. Including isFraud as an input feature would result in trivial, catastrophically failing models that cannot operate in real-time inference.
- **Mandated Handling:** Strictly isolate as label vector y; automated assertion blocks feature inclusion.

### `TransactionID`
- **Risk Classification:** Identifier Memorization & Non-Stationary Index Leakage
- **Available at Scoring Moment:** `Yes`
- **Detailed Rationale:** TransactionID is an arbitrary monotonic primary key assigned by the processing gateway. Including high-cardinality sequence numbers allows models to memorize specific transaction batches or proxy temporal ordering rather than learning generalizable fraud patterns.
- **Mandated Handling:** Preserve for audit logs, API responses, and database joins; strictly exclude from model inputs.

### `TransactionDT`
- **Risk Classification:** Temporal Lookahead & Out-of-Distribution Linear Drift
- **Available at Scoring Moment:** `Yes`
- **Detailed Rationale:** TransactionDT is a monotonically increasing relative timestamp in seconds. Because production transactions will always have TransactionDT > max(train TransactionDT), tree models cannot extrapolate linear values and will place all future data in the rightmost leaf. Furthermore, random train-test splitting across timestamps leaks future fraud patterns into past training.
- **Mandated Handling:** Extract stationary cyclical diurnal features (hour_of_day, sin_hour, cos_hour, day_of_week) and exclude raw linear TransactionDT from the model feature vector.

---

## 3. Feature Availability at the Scoring Moment

In an online banking or payment processing fraud engine, transactions must be evaluated at milliseconds latency before authorization. Feature availability falls into distinct tiers:

1. **Pre-Authorization / Transaction-Moment Features (SAFE FOR SCORING):**
   - `TransactionAmt`, payment card metadata (`card1`–`card6`), product category (`ProductCD`), billing/shipping regions (`addr1`, `addr2`).
   - Purchaser/recipient email domain telemetry (`P_emaildomain`, `R_emaildomain`).
   - Device and browser fingerprint attributes from client sessions (`DeviceType`, `DeviceInfo`, `id_01`–`id_38`).
   - Relative hour-of-day proxy (`hour_of_day`, `sin_hour`, `cos_hour`, `day_of_week`).

2. **Timedelta Features (`D1`–`D15`):**
   - In IEEE-CIS, `D*` features represent days since previous events (e.g., previous transaction, card registration).
   - In production, these must be maintained by an online feature store or streaming aggregation service.
   - For our modeling pipeline, these are treated as pre-computed input features with median imputation, without future backward-filling.

3. **Identity Coverage Handling:**
   - Identity records exist for **24.42%** of transactions.
   - The preprocessor maps missing identity records to `'missing'` or median defaults. Missingness itself is treated as a valid feature signal rather than an operational failure.

---

## 4. Split Design: Temporal Splitting vs Random Cross-Validation

- **Why Random K-Fold Fails for Fraud Detection:**
  Random K-Fold randomly assigns transactions across the entire 6-month timeline into training and test folds. This causes catastrophic temporal lookahead leakage: a fraud cluster or compromised card seen in month 4 would be present in training folds when evaluating month 2. The resulting cross-validation score is artificially inflated and fails abruptly when deployed.
- **BankGuard Policy:**
  BankGuard enforces **chronological out-of-time validation** (first 80% chronological train, subsequent 20% validation). This ensures that validation metrics accurately reflect true prospective performance on future unseen transactions.
