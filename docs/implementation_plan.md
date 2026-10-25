# BankGuard — Implementation Plan

## How to use this plan

Complete the phases in order. Do not skip ahead to the dashboard or advanced model before the data checks and baseline work. After each phase, record evidence in `progress.md`.

**Dataset:** IEEE-CIS Fraud Detection, Kaggle  
**Source:** https://www.kaggle.com/competitions/ieee-fraud-detection/data  
**Rule:** Use public, non-synthetic competition data for this project. Do not substitute simulated data without explicitly changing the project decision.

## Phase 0 — Set up the project

### Task 0.1: Create the repository structure
**Do:** Create the folders listed in `Architecture_and_Data_Flow.md`.

**How:** Use a terminal to create `docs/`, `src/bankguard/`, `tests/`, `configs/`, `data/raw/`, `data/processed/`, and `data/reports/`.

**Outcome:** A predictable place for every project component.

**Verify:** `tree` (or your file explorer) shows the expected folders.

### Task 0.2: Create a Python environment
**Do:** Create and activate a virtual environment, then install the initial packages: pandas, numpy, scikit-learn, matplotlib, seaborn, pytest, and jupyter.

**How:** Use `python -m venv .venv`; activate it for your operating system; install dependencies from `requirements.txt`.

**Outcome:** Project dependencies are isolated from other Python projects.

**Verify:** `python -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"` succeeds.

### Task 0.3: Protect data and secrets
**Do:** Add `.venv/`, `data/raw/`, `data/processed/`, `artifacts/`, `.env`, and caches to `.gitignore`.

**Outcome:** Large/private local artifacts are not accidentally committed.

**Verify:** `git status --ignored` shows local data paths ignored; inspect staged files before every commit.

## Phase 1 — Obtain and understand the data

### Task 1.1: Download the dataset
**Do:** Download the files from the official Kaggle competition data page after accepting its terms.

**How:** Use the Kaggle website or the official Kaggle CLI. Store files under `data/raw/ieee_cis/`.

**Outcome:** Transaction and identity CSVs are available locally.

**Verify:** A script checks that `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`, and `sample_submission.csv` exist. Do not upload them to Git.

### Task 1.2: Inspect schemas and sizes
**Do:** Record row counts, column counts, dtypes, memory use, and a small sample of column names.

**How:** Use `pandas.read_csv(..., nrows=...)` for initial inspection and chunked reading if memory is limited. Do not load every file repeatedly without need.

**Outcome:** A data inventory report.

**Verify:** Report includes each file's path, size, rows (where measured), columns, and load errors.

### Task 1.3: Complete the data card and dictionary
**Do:** Update `data_card.md` and `data_dictionary.md` with observed facts, source caveats, and the actual columns in your downloaded copy.

**How:** Compare the files' real headers with the documented expected schema. Do not guess the meaning of anonymized `C*`, `D*`, `V*`, or `id_*` columns.

**Outcome:** Reviewers know where the data came from and what each field can safely mean.

**Verify:** Every column is covered by an exact name or a documented column group, and uncertain meanings are explicitly marked unknown.

## Phase 2 — Build a reliable data pipeline

### Task 2.1: Implement loading
**Do:** Write a loader for training transaction and identity data.

**How:** Put file paths in configuration; raise a readable error for missing files; avoid hard-coded machine-specific absolute paths.

**Outcome:** A repeatable command loads the raw tables.

**Verify:** Loader reports dimensions and exits with a nonzero status when a required file is missing.

### Task 2.2: Implement quality checks
**Do:** Check required columns, duplicate `TransactionID`, target values, null patterns, and unexpected schema changes.

**How:** Create small functions with clear messages; test them using tiny in-memory fixtures.

**Outcome:** A data-quality report plus explicit failures for critical problems.

**Verify:** Tests pass for valid fixtures and fail for intentionally invalid fixtures.

### Task 2.3: Join identity fields
**Do:** Left-join transaction and identity data on `TransactionID`.

**How:** Validate key uniqueness and join cardinality before joining. Measure identity match rate and missingness.

**Outcome:** Training rows retain their transaction record even when identity details are unavailable.

**Verify:** Joined row count equals transaction row count; transaction IDs remain unique; the match rate is recorded.

## Phase 3 — Explore the data

### Task 3.1: Understand the target
**Do:** Count fraud and non-fraud rows and calculate the fraud percentage.

**How:** Use `value_counts()` and `value_counts(normalize=True)` on `isFraud`.

**Outcome:** A documented understanding of class imbalance.

**Verify:** Counts add up to the training transaction row count.

### Task 3.2: Investigate missingness and fields
**Do:** Rank columns by missing percentage; compare missingness for fraud and non-fraud rows where appropriate.

**How:** Create a missingness table and plots; treat missingness as a potential signal, not automatically as a bug.

**Outcome:** A short list of useful data-quality findings and candidate features.

**Verify:** Table percentages match direct calculations; conclusions reference actual results.

### Task 3.3: Explore amount and time
**Do:** Analyze `TransactionAmt` distributions and `TransactionDT` ordering. Explore selected categorical fields.

**How:** Use robust plots and summaries; document outliers and long tails.

**Outcome:** An EDA report with actionable findings.

**Verify:** Each plot has labels and a short explanation. Do not infer calendar dates from `TransactionDT` without a valid reference.

## Phase 4 — Create leakage-safe features

### Task 4.1: Define the target and features
**Do:** Use `isFraud` as the target; exclude it from input features.

**How:** Maintain an explicit feature-selection function and test it.

**Outcome:** A clean X/y separation.

**Verify:** Automated test confirms the target is absent from X.

### Task 4.2: Create preprocessing
**Do:** Handle missing numeric/categorical values and encode categorical variables.

**How:** Use a scikit-learn `ColumnTransformer`/`Pipeline` where appropriate. Fit only on training rows.

**Outcome:** A transformation pipeline that can be reused at inference time.

**Verify:** Fit on training data; transform validation data; confirm no validation refit occurs.

### Task 4.3: Review leakage risks
**Do:** Check feature availability and split design before training.

**How:** Document that `TransactionDT` is a relative time delta, and that some fields may have competition-specific collection patterns. Avoid features that reveal the target directly or are unavailable at the intended scoring moment.

**Outcome:** Written leakage policy.

**Verify:** Review every excluded field and the reason for excluding it; test target exclusion.

## Phase 5 — Train and compare models

### Task 5.1: Create a baseline
**Do:** Train a simple baseline, such as logistic regression with preprocessing, or a small tree-based model.

**How:** Use a reproducible train/validation split and class-aware evaluation. Avoid judging success by accuracy alone.

**Outcome:** Baseline model, predictions, and metrics.

**Verify:** Training command completes; validation predictions and metrics are saved.

### Task 5.2: Train an improved model
**Do:** Train a stronger tabular model, such as LightGBM or XGBoost if installable and compatible.

**How:** Keep the same validation rows as the baseline. Record parameters, seeds, versions, and training duration.

**Outcome:** A comparable candidate model.

**Verify:** Evaluation uses the same validation set and metrics as the baseline.

### Task 5.3: Evaluate threshold choices
**Do:** Compare precision, recall, F1, PR-AUC, ROC-AUC, and confusion matrices. Examine multiple thresholds.

**How:** Choose a threshold based on a stated objective, such as high recall under a review-capacity constraint. Document false-positive and false-negative trade-offs.

**Outcome:** A model and threshold recommendation supported by measured results.

**Verify:** Recompute metrics from saved predictions; write down the selection rule and chosen threshold.

### Task 5.4: Add explanations
**Do:** Add SHAP or another compatible explanation method if feasible.

**How:** Explain global patterns and individual predictions; avoid treating feature contribution as causation.

**Outcome:** Explanation examples suitable for the dashboard.

**Verify:** Explanations run for several validation records and are clearly labeled as model explanations.

## Phase 6 — Turn scores into a usable demo

### Task 6.1: Implement decision policy
**Do:** Map scores to configurable recommendations such as `allow`, `review`, and `high_risk`.

**How:** Keep thresholds in configuration and document them as demonstration thresholds unless calibrated for a defined operational objective.

**Outcome:** The same score maps to the same recommendation.

**Verify:** Unit tests cover below-threshold, equal-to-threshold, and above-threshold values.

### Task 6.2: Build the API
**Do:** Implement a health endpoint and prediction endpoint.

**How:** Use FastAPI and Pydantic; load the saved preprocessing/model artifact once at startup; validate inputs.

**Outcome:** A local service returning a fraud score and decision recommendation.

**Verify:** API tests cover valid payload, invalid payload, unknown/missing fields policy, and model load failure.

### Task 6.3: Build the dashboard
**Do:** Show dataset summary, saved evaluation metrics, threshold, and demo predictions.

**How:** Use Streamlit; read saved artifacts rather than recalculating inconsistent metrics in the UI.

**Outcome:** A reviewer can understand the system without reading all source code.

**Verify:** Dashboard metric values match `model_evaluation.md` and saved metric files.

## Phase 7 — Reliability and delivery

### Task 7.1: Add monitoring summaries
**Do:** Track data quality, missing input fields, prediction failures, score distribution, and latency.

**How:** Start with structured logs and a batch summary. Do not log unnecessary raw personal or device details.

**Outcome:** Basic visibility into operational behavior.

**Verify:** Run a fixed test batch and confirm the expected counts and summaries appear.

### Task 7.2: Add tests and CI
**Do:** Test data checks, joins, preprocessing, metrics, policy, and API. Configure GitHub Actions to run tests.

**Outcome:** Regressions are caught automatically.

**Verify:** `pytest` passes locally and the CI workflow passes for a test commit.

### Task 7.3: Containerize
**Do:** Add a Dockerfile and documented run commands.

**How:** Use a small, reproducible base image; copy source and dependencies, not raw Kaggle data. Mount local data or configure paths as needed.

**Outcome:** The app can run consistently in a container.

**Verify:** Build the image and run a health check against the container.

### Task 7.4: Final review
**Do:** Review all docs, commands, limitations, and claims.

**Outcome:** A portfolio project that is explainable and reproducible.

**Verify:** Another person can follow the README, reproduce evaluation, launch the API/dashboard, and run tests without guessing missing steps.

## Phase exit rule

Do not mark a phase complete because code exists. Mark it complete only when the stated verification succeeds and the result/evidence is recorded in `progress.md`.
