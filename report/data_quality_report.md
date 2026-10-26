# BankGuard — Data Quality Report

- **Overall Status:** `PASSED`
- **Total Checks:** 8
- **Passed Checks:** 8
- **Critical Failures:** 0
- **Non-Critical Warnings:** 0

## Check Results

| Check Name | Critical? | Status | Details |
|---|:---:|:---:|---|
| `required_columns_transaction` | Yes | **PASS** | All 5 required columns present in transaction. |
| `key_uniqueness_transaction` | Yes | **PASS** | Key 'TransactionID' is strictly unique with 0 nulls across 590,540 rows in transaction. |
| `target_validity` | Yes | **PASS** | Target 'isFraud' is valid with allowed values {0, 1} and 0 nulls. |
| `numeric_ranges_TransactionAmt` | Yes | **PASS** | Amount column 'TransactionAmt' has 0 negative and 0 null values. |
| `null_patterns_transaction` | No | **PASS** | Missingness analyzed in transaction: 0 column(s) >= 95.0% null, 0 completely empty column(s). |
| `required_columns_identity` | Yes | **PASS** | All 1 required columns present in identity. |
| `key_uniqueness_identity` | Yes | **PASS** | Key 'TransactionID' is strictly unique with 0 nulls across 144,233 rows in identity. |
| `null_patterns_identity` | No | **PASS** | Missingness analyzed in identity: 9 column(s) >= 95.0% null, 0 completely empty column(s). |
