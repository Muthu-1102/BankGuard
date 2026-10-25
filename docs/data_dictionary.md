# BankGuard — Data Dictionary

## 1. Purpose

This document explains the expected fields in the IEEE-CIS Fraud Detection files and how BankGuard should treat them. It is a **working dictionary**: verify it against the exact CSV headers you downloaded.

Do not infer the real-world meaning of anonymized columns. If the official source does not define a field, document it as unknown or anonymized.

## 2. Core fields

| Field | Expected table | Meaning supported by schema/source | Type / handling | BankGuard use |
|---|---|---|---|---|
| `TransactionID` | Transaction and identity files | Identifier used to match transaction and identity rows | Identifier; preserve for joins, exclude from model features initially | Join key and traceability |
| `isFraud` | `train_transaction.csv` | Provided binary fraud label for training transactions | Target; validate allowed values | `y` target only; never an input feature |
| `TransactionDT` | Transaction files | Relative time delta; not a calendar timestamp by itself | Numeric; inspect ordering and distribution | Candidate temporal feature after leakage review |
| `TransactionAmt` | Transaction files | Transaction amount field | Numeric; inspect skew, missingness, and outliers | Candidate feature |
| `ProductCD` | Transaction files | Product code/category field | Categorical | Candidate feature |
| `card1`–`card6` | Transaction files | Card-related fields; exact meaning varies and may be obfuscated | Inspect dtype and category cardinality | Candidate features after privacy/leakage review |
| `addr1`, `addr2` | Transaction files | Address-related codes; exact geographic meaning should not be assumed | Often categorical-like; inspect values | Candidate features after review |
| `dist1`, `dist2` | Transaction files | Distance-related fields in the competition schema; precise business semantics are not fully established by the column names alone | Numeric-like; inspect missingness and distributions | Candidate features after review |
| `P_emaildomain` | Transaction files | Purchaser email-domain field | Categorical; handle missing values | Candidate feature |
| `R_emaildomain` | Transaction files | Recipient email-domain field | Categorical; handle missing values | Candidate feature |
| `DeviceType` | Identity files | Device type field | Categorical | Candidate feature if identity join is available |
| `DeviceInfo` | Identity files | Device information string/code | High-cardinality text/category; inspect and normalize cautiously | Candidate feature only after cardinality and leakage review |

## 3. Column families

These families are present in common copies of the competition dataset. Confirm exact ranges in your local files.

| Column family | Common names | What is safe to say | Handling guidance |
|---|---|---|---|
| `C` features | `C1`–`C14` | Anonymized/obfuscated transaction-related numeric features | Inspect dtypes, missingness, ranges, and distributions; do not invent business meanings |
| `D` features | `D1`–`D15` | Anonymized/obfuscated time/delta-related features in the competition schema | Inspect missingness and distributions; review availability at prediction time |
| `M` features | `M1`–`M9` | Match-related categorical/flag features in the competition schema | Inspect observed values and missingness; do not assume missing means false |
| `V` features | `V1`–`V339` | Anonymized/obfuscated features | Use as model features only after checking types, missingness, leakage, and stability |
| `id_` features | `id_01`–`id_38` | Anonymized identity-related features | Available only through identity table; do not infer personal attributes or exact meanings without source documentation |
| Other transaction fields | Verify local header | Additional competition-provided fields may exist | Add each exact field to the inventory and classify it before modeling |

## 4. File-level schema expectations

### `train_transaction.csv`
Expected to contain `TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, and other transaction attributes.

### `train_identity.csv`
Expected to contain `TransactionID` and identity/device-related fields such as `id_*`, `DeviceType`, and `DeviceInfo`.

### `test_transaction.csv`
Expected to contain transaction attributes and `TransactionID`, but normally not the `isFraud` target.

### `test_identity.csv`
Expected to contain `TransactionID` and identity/device-related fields.

### `sample_submission.csv`
Used to inspect the expected competition submission format. It is not ground truth for fraud labels.

## 5. Required local schema inventory

Generate a table from the downloaded files with these columns:

- `file_name`
- `column_name`
- `dtype`
- `null_count`
- `null_percentage`
- `distinct_count` (where practical)
- `sample_values` (avoid publishing sensitive/raw examples)
- `dictionary_status` (`documented`, `anonymized`, `unknown`, `target`, `identifier`)
- `model_use` (`candidate`, `exclude`, `target`, `join_key`, `review_required`)
- `reason`

## 6. Rules for feature use

1. `isFraud` is the label, never a feature.
2. `TransactionID` is a join/traceability key; exclude it from model features initially unless a rigorous reason is documented.
3. Fit imputation, encoding, scaling, feature selection, and any resampling only on training rows.
4. Review `TransactionDT` and other time-related fields for split leakage and prediction-time availability.
5. Do not assume missing values mean a negative event.
6. Do not infer real-world meanings from anonymized feature names.
7. Save the exact feature list with each model version.
8. If a field is excluded, record the reason rather than silently dropping it.

## 7. How to verify this dictionary

1. Read only the headers of each local CSV.
2. Compare the exact headers with this document.
3. Generate the local schema inventory.
4. Add missing fields and remove fields that are absent in your downloaded version.
5. Confirm the target and join key are handled correctly in code.
6. Record unresolved field meanings as `unknown` or `anonymized`.

**Completion criterion:** the dictionary matches the actual files, and every field has a documented type/role or an explicit uncertainty note.
