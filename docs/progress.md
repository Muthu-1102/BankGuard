# BankGuard — Progress Tracker

## How to maintain this file

Update this file after every work session. Check a task only after its verification step passes. If a task is blocked, write the blocker and the next action. Do not mark tasks complete based on intention alone.

**Current status:** Phase 3 in progress (Task 3.1 completed)  
**Dataset decision:** IEEE-CIS Fraud Detection Kaggle competition dataset  
**Data policy:** Public competition data; no synthetic replacement  
**Last updated:** In progress

## Status legend

- `NOT STARTED` — work has not begun.
- `IN PROGRESS` — work is underway.
- `BLOCKED` — a dependency or issue prevents progress.
- `DONE` — implementation and verification both passed.

## Phase tracker

| Phase | Status | Evidence required before marking DONE | Notes / blocker |
|---|---|---|---|
| 0. Repository and environment | DONE | Folder tree, working environment, `.gitignore` checked | Phase 0 complete (Tasks 0.1, 0.2, 0.3) |
| 1. Dataset download and documentation | DONE | Expected files present; schema inventory; data card and dictionary updated from actual headers | Phase 1 complete (Tasks 1.1, 1.2, 1.3) |
| 2. Loading, validation, and join | DONE | Quality report; join checks; tests pass | Phase 2 complete (Tasks 2.1, 2.2, 2.3) |
| 3. Exploratory data analysis | IN PROGRESS | EDA report, labeled plots, written findings | Task 3.1 complete |
| 4. Leakage-safe features | NOT STARTED | Feature pipeline and target-leakage test | |
| 5. Baseline and improved model | NOT STARTED | Saved predictions, metrics, model settings | |
| 6. Evaluation and threshold choice | NOT STARTED | Metrics recomputed from validation predictions; threshold rationale documented | |
| 7. Explainability | NOT STARTED | Explanation examples and limitations documented | |
| 8. Decision policy and API | NOT STARTED | Unit/API tests and working health/prediction endpoints | |
| 9. Dashboard | NOT STARTED | UI launches and metrics match saved evaluation | |
| 10. Monitoring, tests, CI, Docker | NOT STARTED | Test suite, CI run, container health check | |
| 11. Final review | NOT STARTED | Clean setup verified by following README from scratch | |

## Work log

Add a new entry at the top after each work session.

### Task 3.1: Understand the Target
- **Tasks worked on:** Task 3.1 (Understand the target)
- **What I changed:** Switched to new branch `feature/phase-3-eda`; created `src/bankguard/data/target_analysis.py` to calculate exact fraud and non-fraud counts, percentages, imbalance ratio, and assert invariant summation (`0 + 1 == total`); generated publication-quality dual-panel visualization `report/figures/target_distribution.png`; added unit tests in `tests/test_target_analysis.py`; generated `report/target_analysis.md` and `report/target_analysis.json`.
- **Commands run:** `git checkout -b feature/phase-3-eda`, `python src/bankguard/data/target_analysis.py`, `pytest`
- **Outcome:** Verified 569,877 legitimate transactions (96.5010%) and 20,663 fraud transactions (3.4990%) adding up to exactly 590,540 rows (27.58 : 1 imbalance ratio); documented metric and stratified split requirements. All 33 unit tests passed.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/report/figures/target_distribution.png`, `d:/Projects/BankGurd/report/target_analysis.md`, `d:/Projects/BankGurd/report/target_analysis.json`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 3.2 (Investigate missingness and fields).

### Task 2.3: Join Identity Fields
- **Tasks worked on:** Task 2.3 (Join identity fields)
- **What I changed:** Developed `src/bankguard/data/join.py` implementing `safe_join_transaction_identity` with pre-join cardinality checks, post-join assertions, and metric tracking; added unit tests in `tests/test_join.py`; executed join on full training data (590,540 tx, 144,233 id) in 0.99s.
- **Commands run:** `python src/bankguard/data/join.py` and `pytest`
- **Outcome:** Primary key uniqueness guaranteed; joined row count exactly preserved (590,540 rows x 434 columns); identity match rate measured at 24.42%; generated `report/join_report.md` and `report/join_report.json`. All 30 unit tests passed.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/report/join_report.md`, `d:/Projects/BankGurd/report/join_report.json`, `d:/Projects/BankGurd/src/bankguard/data/join.py`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 2.2: Implement Quality Checks
- **Tasks worked on:** Task 2.2 (Implement quality checks)
- **What I changed:** Developed modular validation functions in `src/bankguard/data/quality_checks.py` (`check_required_columns`, `check_key_uniqueness`, `check_target_validity`, `check_numeric_ranges`, `check_null_patterns`, `run_data_quality_suite`); wrote runner `src/bankguard/data/run_quality_report.py`; added comprehensive unit tests in `tests/test_quality_checks.py` asserting pass on valid fixtures and explicit failures on deliberately invalid fixtures.
- **Commands run:** `python src/bankguard/data/run_quality_report.py` and `pytest`
- **Outcome:** Full validation suite executed on raw training data (590,540 tx, 144,233 id) passing 8 of 8 checks with 0 critical failures; generated `report/data_quality_report.md` and `report/data_quality_report.json`. All 24 unit tests passed.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/report/data_quality_report.md`, `d:/Projects/BankGurd/report/data_quality_report.json`, `d:/Projects/BankGurd/src/bankguard/data/quality_checks.py`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 2.1: Implement Loading
- **Tasks worked on:** Task 2.1 (Implement loading)
- **What I changed:** Switched to new branch `feature/data-pipeline`; created externalized configuration in `configs/data_config.yaml`; implemented `src/bankguard/data/loader.py` with memory optimization (`reduce_memory_usage`), custom `DatasetNotFoundError`, and configurable table loaders; installed `bankguard` in editable mode; added unit tests in `tests/test_loader.py`.
- **Commands run:** `git checkout -b feature/data-pipeline`, `python src/bankguard/data/loader.py --nrows 1000`, `pytest`
- **Outcome:** Repeatable loader successfully loads tables, downcasting memory by ~41% on transaction data and ~8% on identity data; raises descriptive `DatasetNotFoundError` and exits nonzero on missing files. All 12 unit tests passed.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/configs/data_config.yaml`, `d:/Projects/BankGurd/src/bankguard/data/loader.py`, `d:/Projects/BankGurd/tests/test_loader.py`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 1.3: Complete Data Card and Dictionary
- **Tasks worked on:** Task 1.3 (Complete the data card and dictionary)
- **What I changed:** Developed `generate_schema_inventory.py` to extract exact target distributions, duplicate checks, and identity match rates; updated `data_card.md` with measured dataset facts (590,540 train tx, 144,233 train id, 20,663 fraud cases = 3.4990% prevalence, 0 duplicates, 24.42% identity match rate); generated `report/schema_inventory.csv` profiling all 435 columns; updated `data_dictionary.md` with column family mappings and verified statuses.
- **Commands run:** `python src/bankguard/data/generate_schema_inventory.py` and `pytest`
- **Outcome:** Complete dataset provenance, empirical statistics, and schema documentation recorded; zero discrepancies between downloaded CSV headers and dictionary documentation.
- **Verification performed:** Generated `report/schema_inventory.csv` and confirmed all columns have explicit roles/types and non-fabrication guarantees.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/docs/data_card.md`, `d:/Projects/BankGurd/docs/data_dictionary.md`, `d:/Projects/BankGurd/report/schema_inventory.csv`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 1.2: Inspect Schemas and Sizes
- **Tasks worked on:** Task 1.2 (Inspect schemas and sizes)
- **What I changed:** Created designated `report/` folder; built memory-efficient inspection utility `inspect_dataset.py` with fast buffered line counting and chunked schema profiling; added unit tests in `tests/test_inspect_dataset.py`; generated `report/data_inventory_report.md` and `report/data_inventory_report.json`.
- **Commands run:** `python src/bankguard/data/inspect_dataset.py` and `pytest`
- **Outcome:** Measured exact dimensions, disk sizes, memory profiles, and dtypes for all raw files: `train_transaction.csv` (590,540 rows x 394 cols), `train_identity.csv` (144,233 rows x 41 cols), `test_transaction.csv` (506,691 rows x 393 cols), `test_identity.csv` (141,907 rows x 41 cols). All 5 unit tests passed.
- **Verification performed:** Report generated with zero load errors and verified on disk.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/report/data_inventory_report.md`, `d:/Projects/BankGurd/report/data_inventory_report.json`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 1.1: Dataset Download and Verification
- **Tasks worked on:** Task 1.1 (Download the dataset)
- **What I changed:** Obtained and placed the 4 core IEEE-CIS Fraud Detection competition files in `data/raw/ieee_cis/`; implemented `verify_raw_data.py` and unit tests in `tests/test_verify_raw_data.py` to assert required files and report sizes.
- **Commands run:** `python src/bankguard/data/verify_raw_data.py data/raw/ieee_cis` and `pytest`
- **Outcome:** All 4 core dataset files present locally (`train_transaction.csv`: 651.69 MB, `train_identity.csv`: 25.30 MB, `test_transaction.csv`: 584.79 MB, `test_identity.csv`: 24.60 MB); raw files are verified ignored by Git.
- **Verification performed:** `verify_raw_data.py` exited with status 0; unit tests passed.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/data/raw/ieee_cis/`, `d:/Projects/BankGurd/src/bankguard/data/verify_raw_data.py`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 0.3: Protect Data and Secrets
- **Tasks worked on:** Task 0.3 (Protect data and secrets)
- **What I changed:** Initialized Git repository tracking; created `.gitignore` covering `.venv/`, `data/raw/*`, `data/processed/*`, `artifacts/*`, `.env*`, Python bytecode caches, pytest/coverage caches, Jupyter checkpoints, IDE and OS files; preserved directory structure with `.gitkeep` files.
- **Commands run:** `git init`, `git status --ignored`
- **Outcome:** Large data, models, secrets, and caches are safely excluded from Git commits while preserving directory architecture.
- **Verification performed:** Verified with `git status --ignored` that test fixtures in `data/raw/`, `data/processed/`, `artifacts/`, and `.env` are strictly ignored.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/.gitignore`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 0.2: Python Environment Setup
- **Tasks worked on:** Task 0.2 (Create a Python environment)
- **What I changed:** Inspected system Python versions (3.10, 3.11, 3.12, 3.13); configured `.venv` using the mature Python 3.10 runtime pre-equipped with binary wheels; verified and linked dependencies specified in `requirements.txt`.
- **Commands run:** `py -3.10 -m venv --clear --system-site-packages .venv` and verification via `.venv\Scripts\python.exe -c "import pandas, sklearn, numpy, matplotlib, seaborn, pytest; print(...)"`
- **Outcome:** Virtual environment `.venv` fully operational with `pandas` 2.2.2, `scikit-learn` 1.7.2, `numpy` 1.26.4, `matplotlib` 3.8.4, `seaborn` 0.13.2, `pytest` 9.0.3, and `jupyter`.
- **Verification performed:** Executed Python import verification script successfully without errors.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/.venv`, `d:/Projects/BankGurd/requirements.txt`
- **Problems found:** None.
- **Next exact action:** Completed.

### Task 0.1: Repository Structure Setup
- **Tasks worked on:** Task 0.1 (Create the repository structure)
- **What I changed:** Created project directories as specified in `Architecture_and_Data_Flow.md` (`data/raw/`, `data/processed/`, `data/reports/`, `notebooks/`, `src/bankguard/` [with `data`, `features`, `models`, `decisioning`, `api`, `monitoring`], `app/`, `tests/`, `configs/`, `artifacts/`) and initialized Python packages with `__init__.py`.
- **Commands run:** `New-Item -ItemType Directory ...` and `Get-ChildItem -Recurse -Depth 3`
- **Outcome:** Clean directory structure established for BankGuard components.
- **Verification performed:** Verified folder hierarchy matches `Architecture_and_Data_Flow.md` specifications.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/`
- **Problems found:** None.
- **Next exact action:** Completed.

---

## Current blockers

| Blocker | Impact | Next action | Status |
|---|---|---|---|
| None recorded yet | — | Begin Phase 0 | — |

## Final acceptance checklist

- [ ] Dataset provenance and terms documented.
- [ ] No raw dataset or secrets committed to Git.
- [ ] Schema checks and join tests pass.
- [ ] Data dictionary matches actual downloaded headers.
- [ ] EDA conclusions are based on calculated values.
- [ ] Target leakage checks pass.
- [ ] Baseline and improved model use the same validation split.
- [ ] PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, and threshold analysis are saved.
- [ ] Threshold rationale is documented.
- [ ] API and dashboard work from saved artifacts.
- [ ] Tests and CI pass.
- [ ] Limitations are clearly stated.
- [ ] A clean setup/reproduction run is complete.
