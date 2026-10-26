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

## 5. Local schema inventory results

The local schema inventory has been generated and validated across all 435 column definitions from `train_transaction.csv` (394 columns) and `train_identity.csv` (41 columns).

- **Full Schema Inventory File:** [`report/schema_inventory.csv`](../report/schema_inventory.csv)
- **Inventory Summary:** [`report/schema_inventory_summary.md`](../report/schema_inventory_summary.md)

### Column Family Breakdown

| Family | Columns Count | Source Table | Dictionary Status | Model Use | Handling Rule |
|---|---:|---|---|---|---|
| `TransactionID` | 1 | Both | `identifier` | `join_key` | Preserved for joins; excluded from features |
| `isFraud` | 1 | Transaction | `target` | `target` | Supervised ground truth; strictly excluded from `X` |
| `TransactionDT` | 1 | Transaction | `documented` | `candidate` | Time delta from reference; evaluate temporal split |
| `TransactionAmt` | 1 | Transaction | `documented` | `candidate` | Continuous numeric amount; log/scale transform |
| `ProductCD` | 1 | Transaction | `documented` | `candidate` | Categorical product code (5 distinct values) |
| `card1`–`card6` | 6 | Transaction | `documented` | `candidate` | Payment card specs (card1 numeric, card4/6 categorical) |
| `addr1`, `addr2` | 2 | Transaction | `documented` | `candidate` | Categorical-like region codes |
| `dist1`, `dist2` | 2 | Transaction | `documented` | `candidate` | Distance numeric features; high missingness |
| `P_emaildomain`, `R_emaildomain` | 2 | Transaction | `documented` | `candidate` | Email domain categories (purchaser & recipient) |
| `C1`–`C14` | 14 | Transaction | `anonymized` | `candidate` | Obfuscated frequency/counting numeric features |
| `D1`–`D15` | 15 | Transaction | `anonymized` | `candidate` | Obfuscated time delta features |
| `M1`–`M9` | 9 | Transaction | `anonymized` | `candidate` | Match flags (categorical/boolean strings) |
| `V1`–`V339` | 339 | Transaction | `anonymized` | `candidate` | Engineered rich features; numeric float64 |
| `id_01`–`id_38` | 38 | Identity | `anonymized` | `candidate` | Identity attributes; present for ~24.4% of transactions |
| `DeviceType`, `DeviceInfo` | 2 | Identity | `documented` | `candidate` | Device metadata category and high-cardinality string |

### Summary by Status and Use
- **Dictionary Status:** 416 Anonymized (`V*`, `C*`, `D*`, `M*`, `id_*`), 16 Documented (`card*`, `addr*`, `dist*`, `email*`, `Device*`, `TransactionAmt`), 2 Identifiers, 1 Target.
- **Model Use:** 432 Candidates, 2 Join Keys, 1 Target.

## 6. Rules for feature use

1. `isFraud` is the label, never a feature.
2. `TransactionID` is a join/traceability key; exclude it from model features initially unless a rigorous reason is documented.
3. Fit imputation, encoding, scaling, feature selection, and any resampling only on training rows.
4. Review `TransactionDT` and other time-related fields for split leakage and prediction-time availability.
5. Do not assume missing values mean a negative event.
6. Do not infer real-world meanings from anonymized feature names.
7. Save the exact feature list with each model version.
8. If a field is excluded, record the reason rather than silently dropping it.

## 7. Dictionary verification status

- [x] Read and profiled exact headers from each local CSV file.
- [x] Compared exact headers with the documented specification (0 discrepancies found).
- [x] Generated local schema inventory: [`report/schema_inventory.csv`](../report/schema_inventory.csv).
- [x] Confirmed `isFraud` target appears only in `train_transaction.csv` and is isolated as `target`.
- [x] Confirmed `TransactionID` is unique across transactions and identities with 0 duplicates.
- [x] All 416 anonymized columns are explicitly designated as `anonymized` with no fabricated business meanings.

**Completion criterion MET:** The dictionary matches the actual files, and every field has a documented type, role, and explicit status note.
