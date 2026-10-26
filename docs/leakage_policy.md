# BankGuard — Data Leakage & Feature Integrity Policy

## 1. Executive Summary & Purpose

Data leakage occurs when information from outside the training dataset or information unavailable at the real-time scoring moment is inadvertently used to train a machine learning model. In financial fraud detection, leakage causes models to appear deceptively accurate during offline evaluation while failing catastrophically in production.

This document establishes the official **BankGuard Data Leakage Policy** for the IEEE-CIS Fraud Detection pipeline, covering target isolation, identifier handling, chronological split design, and feature availability at authorization time.

---

## 2. Policy Principles

### Principle 1: Ground Truth Target (`isFraud`) Isolation
- **Rule:** `isFraud` is the supervised target vector ($y$) and must never be included as an input feature ($X$).
- **Production Context:** Fraud labels are not available at transaction time. Real-world fraud labels originate from customer dispute notifications, chargebacks, and investigative reviews that arrive **30 to 90 days after transaction authorization**.
- **Automated Enforcement:**
  - `split_features_and_target()` enforces strict programmatic isolation.
  - `audit_feature_leakage()` runs case-insensitive checks on feature names.
  - Any inclusion of `isFraud` immediately raises a fatal `TargetLeakageError`.

### Principle 2: Monotonic Identifier (`TransactionID`) Exclusion
- **Rule:** `TransactionID` is strictly an audit key and must never enter the model feature space.
- **Production Context:** Gateway identifiers are arbitrary, non-stationary sequence numbers. Allowing models to train on raw transaction IDs results in index memorization and spurious correlation with chronological fraud bursts.
- **Automated Enforcement:** Filtered by default in `get_feature_names()` and excluded by the preprocessor ColumnTransformer.

### Principle 3: Chronological Temporal Splitting (Zero Lookahead)
- **Rule:** Random cross-validation (e.g. standard K-Fold) is strictly prohibited. All validation and test splits must be partitioned chronologically using `TransactionDT`.
- **Production Context:** A fraud detection system deployed on Day $T$ must score future transactions on Day $T + \Delta t$ using only patterns learned up to Day $T$. In random K-Fold, future compromised card clusters or merchant breaches leak into past training folds, inflating offline AUCPR by 15–30% compared to true prospective production performance.
- **Implementation:** `temporal_train_val_split()` strictly guarantees:
  $$\max(TransactionDT_{\text{train}}) \le \min(TransactionDT_{\text{val}})$$

### Principle 4: Raw Monotonic Timestamp (`TransactionDT`) Replacement
- **Rule:** Raw linear `TransactionDT` must be excluded from feature vectors; stationary cyclical proxies must be used instead.
- **Production Context:** `TransactionDT` represents relative seconds from a reference epoch ($86,400 \le TransactionDT \le 15,811,131$). In production, incoming transactions will have $TransactionDT > 15,811,131$. Decision trees cannot extrapolate linear splits beyond training ranges and will lump all future traffic into the maximum leaf node.
- **Implementation:**
  - `BankGuardFeatureEngineer` extracts cyclical diurnal features: `hour_of_day`, `sin_hour`, `cos_hour`, and `day_of_week`.
  - Raw `TransactionDT` is automatically dropped from the preprocessed matrix.

---

## 3. Excluded Fields Registry

The following table documents every excluded field along with its formal risk category and handling justification:

| Field Name | Source Table | Leakage Risk Category | Available at Scoring? | Mandated Treatment |
|---|---|---|:---:|---|
| `isFraud` | `train_transaction.csv` | **Direct Target Leakage** | No (Post-event chargeback) | Isolated strictly as label vector $y$; automated code assertion blocks inclusion in $X$. |
| `TransactionID` | Both CSVs | **Identifier Memorization / Non-Stationary Leakage** | Yes (Traceability key) | Excluded from $X$; preserved only for join operations, prediction audit logging, and API response IDs. |
| `TransactionDT` (raw) | Transaction CSVs | **Linear Drift & Temporal Lookahead** | Yes (Timestamp) | Replaced with stationary cyclical features (`hour_of_day`, `sin_hour`, `cos_hour`, `day_of_week`); raw timestamp dropped. |

---

## 4. Feature Availability at the Scoring Moment

In an online payment gateway, transaction authorization occurs within **50–200 milliseconds**. Feature availability falls into distinct operational tiers:

```
[Transaction Initiated]
          │
          ├──> 1. Transaction Telemetry (Real-time / In-stream)
          │    - Amount, Card metadata, ProductCD, Email domains, Addresses
          │
          ├──> 2. Session / Device Telemetry (Client-side / Webhook)
          │    - DeviceType, DeviceInfo, id_01 to id_38 (present in 24.42% of traffic)
          │
          ├──> 3. Historical Counters / Feature Store (Streaming aggregates)
          │    - C1–C14 (frequency counters), D1–D15 (timedeltas since prior events)
          │
     [BankGuard Scored] ──> Real-time Risk Decision (ALLOW / REVIEW / BLOCK)
          │
          └──> 4. Post-Event Labels (Arrival latency: 30-90 days)
               - Customer dispute, chargeback, or fraud confirmation -> isFraud
```

### Identity Coverage & Missingness Philosophy
Only **24.42%** of transactions have matching records in `train_identity.csv`. In BankGuard:
1. Missing identity records are **never** treated as an ingestion error or synthetic zero.
2. Missing categorical values are mapped to `'missing'`.
3. As proven in Task 3.2, missingness in identity fields is itself a powerful predictive signal (fraudulent transactions exhibit 31.5% higher identity reporting rates than benign transactions). Tree models natively branch on `NaN`, preserving this signal without leakage.

---

## 5. Verification & Audit Contract

Before training baseline or improved models (Phase 5), the following automated assertions must pass:
1. `audit_dataset_leakage()` confirms zero overlap between train and validation `TransactionID`s.
2. `audit_dataset_leakage()` confirms $\max(\text{train } TransactionDT) \le \min(\text{val } TransactionDT)$.
3. `audit_feature_matrix_leakage()` confirms `isFraud`, `TransactionID`, and raw `TransactionDT` are absent from preprocessor output features.
4. `BankGuardPreprocessor` passes cryptographic zero-refit verification between train and validation transforms.
