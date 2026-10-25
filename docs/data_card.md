# BankGuard — Data Card

## 1. Dataset identity

- **Dataset name:** IEEE-CIS Fraud Detection
- **Publisher/host:** Kaggle competition
- **Source URL:** https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Task:** Binary classification of online transaction fraud.
- **Primary target:** `isFraud` in the training transaction file.
- **Data type:** Historical transaction-level tabular data with a separate identity table.
- **Use in BankGuard:** Model development, analysis, and a local demonstration.
- **Download date:** Record the actual date here: `YYYY-MM-DD`.
- **Dataset version/files:** Record the downloaded file names, sizes, and any checksums here.

## 2. What the data represents

The competition provides transaction records and a separate identity table. The transaction table contains transaction-level attributes and the `isFraud` label for training records. The identity table contains additional identity/device-related fields for a subset of transactions. Join the tables using `TransactionID`.

The competition dataset is anonymized/obfuscated in important places. Column names such as `C1`, `D1`, `V1`, and `id_01` do not by themselves tell us the real-world meaning of the values. Do not invent business definitions for them.

## 3. Files expected

| File | Role | Target label available? | Use |
|---|---|---|---|
| `train_transaction.csv` | Training transaction records | Yes, expected `isFraud` | Model training and validation |
| `train_identity.csv` | Additional identity/device attributes | No target expected | Join to training transactions |
| `test_transaction.csv` | Competition test transactions | Typically no `isFraud` label | Schema checks or optional prediction demo only |
| `test_identity.csv` | Identity/device attributes for test records | No target expected | Join to test transactions |
| `sample_submission.csv` | Expected submission format | Contains example IDs/format, not ground-truth labels | Understand competition output format |

Confirm every detail against the downloaded headers. File names or schema can change if a different copy is used.

## 4. Intended use

Appropriate uses:
- Study class imbalance and missing data.
- Practice fraud classification on tabular data.
- Compare baseline and stronger models.
- Evaluate precision/recall trade-offs.
- Demonstrate an API, dashboard, model explanation, and basic monitoring.

Not appropriate:
- Real-world decisions about a real person's payment.
- Claiming production-grade banking fraud detection.
- Claiming the dataset is JPMorgan data.
- Claiming account-network or money-laundering detection without relationship data and suitable labels.
- Claiming performance on the unlabeled competition test set without external ground truth.

## 5. Population and coverage

The data is a competition benchmark focused on online transaction fraud. It does not represent every bank, country, payment method, customer population, or type of financial crime. The exact collection period, selection process, and representativeness should be described only to the extent supported by the official competition documentation.

The training and competition test files have different roles. Do not assume the competition test labels are available. A local validation set carved from labeled training data is needed for performance evaluation.

## 6. Known limitations and risks

1. **Anonymized features:** Many field meanings are not publicly interpretable.
2. **Identity coverage:** Not every transaction necessarily has a matching identity row.
3. **Class imbalance:** Fraud is much less common than non-fraud; accuracy alone can be misleading.
4. **Historical data:** Performance on historical rows does not guarantee future performance.
5. **Competition distribution:** Train/test construction may differ from a real deployment environment.
6. **Limited context:** The schema does not, by itself, provide a complete bank account graph, full customer histories, investigator notes, or a complete AML case lifecycle.
7. **Label limitations:** `isFraud` represents the dataset's provided label, not a universal definition of financial crime.
8. **Privacy and terms:** Follow Kaggle competition terms and avoid exposing raw records unnecessarily.

## 7. Data handling

- Store raw files under `data/raw/` and keep them read-only.
- Do not commit the dataset to Git by default.
- Do not upload raw records into public logs, screenshots, or demos.
- Keep only the fields required for the analysis or API demonstration.
- Record preprocessing decisions and feature exclusions.
- If redistributing the project, provide download instructions rather than redistributing data unless the terms explicitly allow it.

## 8. Required data quality checks

| Check | Why | Expected evidence |
|---|---|---|
| Required files exist | Prevent accidental use of incomplete data | File inventory report |
| Expected columns exist | Catch wrong file or schema changes | Schema validation report |
| `TransactionID` uniqueness | Protect transaction-level row identity | Duplicate count and test |
| Target values are valid | Ensure a valid supervised task | Counts of `isFraud` values |
| Identity key uniqueness/cardinality | Avoid multiplying rows in join | Key report and join assertion |
| Join row count | Ensure transaction rows are preserved | Before/after row count |
| Missingness | Understand field coverage | Per-column null percentage |
| Class distribution | Understand imbalance | Fraud count and percentage |
| Train/test schema comparison | Detect incompatible input fields | Schema diff report |

## 9. What to fill in after downloading

- Download date:
- Downloaded file sizes:
- Training transaction rows:
- Training identity rows:
- Training fraud count:
- Training fraud percentage:
- Duplicate transaction IDs:
- Identity match rate:
- Columns present that differ from this document:
- Data quality issues discovered:
- License/terms notes from the official competition page:

Do not fill these values from guesses. Calculate them from the actual downloaded files.
