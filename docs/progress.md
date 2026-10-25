# BankGuard — Progress Tracker

## How to maintain this file

Update this file after every work session. Check a task only after its verification step passes. If a task is blocked, write the blocker and the next action. Do not mark tasks complete based on intention alone.

**Current status:** Phase 0 complete  
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
| 1. Dataset download and documentation | NOT STARTED | Expected files present; schema inventory; data card and dictionary updated from actual headers | |
| 2. Loading, validation, and join | NOT STARTED | Quality report; join checks; tests pass | |
| 3. Exploratory data analysis | NOT STARTED | EDA report, labeled plots, written findings | |
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

### Task 0.3: Protect Data and Secrets
- **Tasks worked on:** Task 0.3 (Protect data and secrets)
- **What I changed:** Initialized Git repository tracking; created `.gitignore` covering `.venv/`, `data/raw/*`, `data/processed/*`, `artifacts/*`, `.env*`, Python bytecode caches, pytest/coverage caches, Jupyter checkpoints, IDE and OS files; preserved directory structure with `.gitkeep` files.
- **Commands run:** `git init`, `git status --ignored`
- **Outcome:** Large data, models, secrets, and caches are safely excluded from Git commits while preserving directory architecture.
- **Verification performed:** Verified with `git status --ignored` that test fixtures in `data/raw/`, `data/processed/`, `artifacts/`, and `.env` are strictly ignored.
- **Evidence/artifact paths:** `d:/Projects/BankGurd/.gitignore`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Phase 1 (Task 1.1: Download the dataset).

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
