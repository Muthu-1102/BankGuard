# BankGuard — Target & Feature Definition Audit (Task 4.1)

This report documents the explicit definition of the model target and candidate features, along with automated leakage verification on the training dataset (590,540 rows).

---

## 1. Target & Identifier Specification

- **Target Column (`y`):** `isFraud` (binary classification: 1 = Fraud, 0 = Legitimate)
- **Primary Key Identifier:** `TransactionID` (traceability only; excluded from model input)
- **Target Presence in X:** **False** (PASSED: Target is completely absent from X)
- **ID Presence in X:** **False** (PASSED: Identifier is completely absent from X)
- **Leakage Security Audit:** **PASSED (0 violations)**

---

## 2. Feature Matrix Breakdown

- **Total Input Table Columns:** 434
- **Total Candidate Features (`X`):** 432
- **Numeric Features:** 401 (92.8%)
- **Categorical Features:** 31 (7.2%)

### High-Level Category Families in Feature Matrix

| Family Group | Column Prefix / Pattern | Count | Primary Type | Modeling Treatment |
|---|---|---:|---|---|
| Transaction Attributes | `TransactionAmt`, `dist1`, `dist2` | 3 | Numeric | Log transform, scaling, outlier clipping |
| Categorical Transaction | `ProductCD`, `card4`, `card6`, `P/R_emaildomain` | 5 | Categorical | Target encoding / Ordinal encoding / Frequency |
| Card & Address Identifiers | `card1`–`card3`, `card5`, `addr1`, `addr2` | 6 | Numeric/Categorical | High-cardinality binning / grouping |
| Count Features | `C1`–`C14` | 14 | Numeric | Native numeric scaling / tree splits |
| Timedelta Features | `D1`–`D15` | 15 | Numeric | Temporal consistency checks |
| Match Flags | `M1`–`M9` | 9 | Categorical | Boolean/Missing flag encoding |
| Vesta Rich Features | `V1`–`V339` | 339 | Numeric | Missingness-aware tree routing |
| Identity Device Metadata | `DeviceType`, `DeviceInfo` | 2 | Categorical | Frequency encoding / Missing flag |
| Identity Anonymized | `id_01`–`id_38` | 38 | Mixed Numeric/Categorical | Missing indicator + numerical imputation |

---

## 3. Leakage Prevention Contract

1. **Strict X/y Separation:** The `split_features_and_target()` function provides guaranteed programmatic isolation of `isFraud`.
2. **Automated Assertion:** Any call attempting to include `isFraud` in `feature_cols` or failing to exclude it triggers an explicit `TargetLeakageError`.
3. **Reproducibility:** Pipeline feature signatures are locked in `configs/feature_config.yaml`.
