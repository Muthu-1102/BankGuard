# BankGuard — Progress Tracker

## How to maintain this file

Update this file after every work session. Check a task only after its verification step passes. If a task is blocked, write the blocker and the next action. Do not mark tasks complete based on intention alone.

**Current status:** Phase 6 complete; Phase 7 in progress  
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
| 3. Exploratory data analysis | DONE | EDA report, labeled plots, written findings | Phase 3 complete (Tasks 3.1, 3.2, 3.3) |
| 4. Leakage-safe features | DONE | Feature pipeline and target-leakage test | Phase 4 complete (Tasks 4.1, 4.2, 4.3) |
| 5. Train and compare models | DONE | Saved predictions, metrics, threshold analysis, SHAP explanations | Phase 5 complete (Tasks 5.1, 5.2, 5.3, 5.4) |
| 6. Turn scores into a usable demo | DONE | Decision policy, FastAPI serving endpoint, Streamlit dashboard | Phase 6 complete (Tasks 6.1, 6.2, 6.3) |
| 7. Reliability and delivery | IN PROGRESS | Test suite, CI workflow, containerization, final review | Task 7.1 in progress |

## Work log

Add a new entry at the top after each work session.

### Task 6.3: Build the Interactive FinTech Dashboard
- **Tasks worked on:** Task 6.3 (Build the dashboard)
- **What I changed:** Implemented complete 7-page interactive dashboard in Streamlit (`src/bankguard/dashboard/` and `dashboard.py`) adhering to Berry FinTech architecture with WCAG 2.2 AA contrast compliance; created `src/bankguard/dashboard/theme.py` with design tokens, layout parameters, hide radio circle CSS, hero cards with ambient circles, standard Plotly layouts, and high-contrast badges; implemented `src/bankguard/dashboard/data_loader.py` reading strictly from saved artifacts without recalculating metrics in UI; created 7 pages: `page_overview.py` (Berry hero cards, growth bar chart, high-risk watchlist, 3-tier policy cards), `page_explorer.py` (real validation search, filter, and inspector), `page_risk_analysis.py` (threshold slider, confusion matrix, cost models), `page_model_performance.py` (PR-AUC, ROC curves, baseline vs LightGBM), `page_explainability.py` (TreeSHAP global and local waterfall), `page_data_quality.py` (8-check checklist, missingness breakdown), and `page_system_health.py` (API probe, artifact audit); configured `.streamlit/config.toml`; added test suites `tests/test_dashboard.py` and `tests/test_theme_contrast.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest tests/`
- **Outcome:** All 121 unit and integration tests passed in 49.72s. Contrast ratios verified >= 4.5:1 across all surfaces (primary text 15.6:1 - 17.6:1). All displayed metric values match `model_evaluation.md` and saved reports exactly.
- **Evidence/artifact paths:** `dashboard.py`, `src/bankguard/dashboard/`, `.streamlit/config.toml`, `tests/test_dashboard.py`, `tests/test_theme_contrast.py`
- **Problems found:** None.
- **Next exact action:** Proceeding to Task 7.1 (Add monitoring summaries).

### Task 6.2: Build the FastAPI Fraud Serving Service
- **Tasks worked on:** Task 6.2 (Build the API)
- **What I changed:** Integrated `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `pydantic>=2.0.0`, `httpx>=0.27.0` into `requirements.txt`; implemented `src/bankguard/api/schemas.py` defining Pydantic request/response models (`TransactionInput`, `BatchTransactionInput`, `PredictionResponse`, `BatchPredictionResponse`, `HealthResponse`, `PolicyResponse`); implemented `src/bankguard/api/service.py` providing `ModelService` which loads model, preprocessor, and decision policy once at startup and executes single/batch inference; implemented `src/bankguard/api/app.py` exposing REST endpoints (`GET /`, `GET /health`, `POST /predict`, `POST /predict/batch`, `GET /policy`); enforced unknown fields policy (extra unmodeled fields accepted and safely ignored) and missing fields policy (sparse payloads gracefully imputed via preprocessor); added error handling for model load failures (HTTP 503) and validation errors (HTTP 422); optimized `src/bankguard/features/preprocessor.py` using `pd.concat` for zero DataFrame fragmentation; added 13 comprehensive API integration tests in `tests/test_api.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest tests/test_api.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 110 unit tests in repository passed in 13.17s. Verified valid scoring, negative/missing amount validation errors (HTTP 422), unknown/missing fields policy, batch scoring, and model load failure behavior (HTTP 503).
- **Evidence/artifact paths:** `src/bankguard/api/app.py`, `src/bankguard/api/service.py`, `src/bankguard/api/schemas.py`, `src/bankguard/api/__init__.py`, `tests/test_api.py`, `requirements.txt`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 6.3 (Build the Streamlit dashboard).

### Task 6.1: Implement Configurable Fraud Decision Policy
- **Tasks worked on:** Task 6.1 (Implement decision policy)
- **What I changed:** Switched to new branch `feature/phase-6-serving`; implemented `configs/policy_config.yaml` defining operational risk boundaries (`review_threshold: 0.10`, `block_threshold: 0.40`) and action descriptions; implemented `src/bankguard/decisioning/policy.py` and `src/bankguard/decisioning/__init__.py` providing `FraudDecisionPolicy`, `DecisionRecommendation` enum (`allow`, `review`, `high_risk`), and `DecisionResult`; enforced strict determinism, equal-to-threshold escalation policy, and mathematical input validation; scored out-of-time validation transactions yielding 94.42% `allow` (1.35% escape rate), 3.75% `review` (21.26% fraud density), and 1.83% `high_risk` (74.63% auto-decline precision); generated `report/decision_policy_report.md` and `report/decision_policy_report.json`; added comprehensive unit tests in `tests/test_policy.py`.
- **Commands run:** `git checkout -b feature/phase-6-serving`, `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/decisioning/policy.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 97 unit tests in repository passed in 24.18s. Decision policy is verified to deterministically map scores to recommendations with full below-threshold, equal-to-threshold, and above-threshold coverage.
- **Evidence/artifact paths:** `configs/policy_config.yaml`, `src/bankguard/decisioning/policy.py`, `src/bankguard/decisioning/__init__.py`, `tests/test_policy.py`, `report/decision_policy_report.md`, `report/decision_policy_report.json`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 6.2 (Build the API with FastAPI).

### Task 5.4: Add Model Explanations (TreeSHAP)
- **Tasks worked on:** Task 5.4 (Add explanations)
- **What I changed:** Integrated `shap>=0.45.0` into `.venv` and `requirements.txt`; implemented `src/bankguard/models/explainability.py` providing global and local model explanations using `shap.TreeExplainer`; computed global mean absolute SHAP values across 2,000 out-of-time validation transactions (`C13`, `TransactionAmt`, `C14`, `card6`, `card2`, `V70`, `V91` identified as top global signals); generated publication visual artifacts `report/shap_summary.png` (global feature attributions) and `report/shap_local_waterfall.png` (individual transaction attribution waterfall); implemented `explain_single_prediction()` returning calibrated risk tiers, positive risk drivers, and negative protective factors; generated structured benchmark examples across 5 transaction archetypes (True Positive High-Confidence Fraud, Suspicious Review, False Positive Challenged, Clean Benign Auto-Approved, False Negative Missed Fraud) saved to `artifacts/explanation_examples.json`; added strict causation disclaimers to code and reports; generated `report/model_explainability.md` and `report/model_explainability.json`; added comprehensive unit tests in `tests/test_explainability.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/models/explainability.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 88 unit tests in repository passed in 36.69s. Explanations run for multiple validation records, are labeled as statistical associations rather than real-world causation, and are structured into dashboard-ready payloads. Phase 5 is fully complete.
- **Evidence/artifact paths:** `src/bankguard/models/explainability.py`, `tests/test_explainability.py`, `artifacts/explanation_examples.json`, `report/shap_summary.png`, `report/shap_local_waterfall.png`, `report/model_explainability.md`, `report/model_explainability.json`
- **Problems found:** None.
- **Next exact action:** Phase 5 is complete. Awaiting user approval to create a new branch (`feature/phase-6-serving`) and proceed to Phase 6 (Turn scores into a usable demo).

### Task 5.3: Evaluate Decision Threshold Choices and Operational Trade-offs
- **Tasks worked on:** Task 5.3 (Evaluate threshold choices)
- **What I changed:** Implemented `src/bankguard/models/threshold_tuning.py` providing `evaluate_threshold_grid()`, `find_optimal_thresholds()`, `plot_threshold_curves()`, and `run_threshold_evaluation_pipeline()`; recomputed classification metrics and business costs from saved validation predictions in `artifacts/improved_predictions.npz` and `artifacts/baseline_predictions.npz` across 30 fine-grained decision thresholds; modeled multiple operational objectives (Max F1 = 0.5396 at $\tau=0.25$, Capacity-Constrained $\le 3.0\%$ queue load at $\tau=0.22$, Target Recall $\ge 70\%$ at $\tau=0.06$, Minimum Total Cost at $\tau=0.06$); recommended primary threshold $\tau = 0.22$ achieving 49.75% fraud recall (2,022 fraud cases intercepted), 57.82% precision, and 98.71% specificity while keeping manual review queue strictly at 2.96% of total volume (3,497 reviews); defined 3-tier operational decision policy for Phase 6 (`allow` $< 0.10$, `review` $[0.10, 0.40)$, `block` $\ge 0.40$); generated multi-panel visual artifact `report/threshold_tradeoffs.png` and `artifacts/threshold_tradeoffs.png`; generated `report/threshold_evaluation.md` and `report/threshold_evaluation.json`; added comprehensive unit tests in `tests/test_threshold_tuning.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/models/threshold_tuning.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 81 unit tests in repository passed in 12.16s. Demonstrated that LightGBM achieves identical recall to baseline with 78% fewer false positive alarms and established an empirically grounded threshold selection rule.
- **Evidence/artifact paths:** `src/bankguard/models/threshold_tuning.py`, `tests/test_threshold_tuning.py`, `report/threshold_evaluation.md`, `report/threshold_evaluation.json`, `report/threshold_tradeoffs.png`, `artifacts/threshold_tradeoffs.png`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 5.4 (Add explanations with SHAP).

### Task 5.2: Train an Improved Model (LightGBM) and Compare with Baseline
- **Tasks worked on:** Task 5.2 (Train an improved model)
- **What I changed:** Integrated LightGBM 4.7.0 into environment and `requirements.txt`; implemented `src/bankguard/models/improved.py` providing reproducible gradient boosted decision tree training (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=63`, `max_depth=10`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=50`, `random_state=42`); trained on 472,432 chronologically partitioned training rows x 438 features (`scale_numeric=False`); evaluated on the exact same 118,108 out-of-time validation transactions as Task 5.1; updated `src/bankguard/models/metrics.py` with `compare_model_metrics()` and `format_comparison_markdown()`; extracted total gain and split feature importances (`improved_feature_importances.csv`); achieved PR-AUC of 0.5598 (+294.2% relative gain vs baseline 0.1420), ROC-AUC of 0.9136 (+16.5% vs baseline 0.7844), Brier calibration score of 0.0212 (-87.6% error reduction vs baseline 0.1716), and log loss of 0.0867 (-86.0% error reduction vs baseline 0.6191); false positive rate dropped from 16.23% to 0.30% (false positives reduced from 18,507 down to 339); saved model artifact to `artifacts/improved_model.joblib`; saved validation predictions to `artifacts/improved_predictions.npz`; generated `report/improved_metrics.md`, `report/improved_metrics.json`, `report/model_comparison.md`, and `report/model_comparison.json`; added comprehensive unit tests in `tests/test_improved.py`.
- **Commands run:** `pip install lightgbm`, `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/models/improved.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 76 unit tests in repository passed in 9.64s. LightGBM dramatically outperforms linear baseline across discrimination and probability calibration on identical validation transactions.
- **Evidence/artifact paths:** `src/bankguard/models/improved.py`, `src/bankguard/models/metrics.py`, `artifacts/improved_model.joblib`, `artifacts/improved_predictions.npz`, `artifacts/improved_feature_importances.csv`, `report/improved_metrics.md`, `report/improved_metrics.json`, `report/model_comparison.md`, `report/model_comparison.json`, `tests/test_improved.py`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 5.3 (Evaluate threshold choices).

### Task 5.1: Create a Baseline Model and Evaluate
- **Tasks worked on:** Task 5.1 (Create a baseline)
- **What I changed:** Switched to new branch `feature/phase-5-models`; implemented `src/bankguard/models/metrics.py` providing class-aware evaluation (PR-AUC, ROC-AUC, Brier score, log loss, precision, recall, F1, confusion matrix, specificity, false positive rate); implemented `src/bankguard/models/baseline.py` training a reproducible Logistic Regression baseline model with balanced class weighting on 472,432 chronologically partitioned training rows x 438 features; scored 118,108 out-of-time validation transactions; achieved PR-AUC of 0.1420 (4.1x above 0.0344 random guess baseline), ROC-AUC of 0.7844, and Brier score of 0.1716; saved model artifact to `artifacts/baseline_model.joblib`; saved validation predictions to `artifacts/baseline_predictions.npz`; generated `report/baseline_metrics.md` and `report/baseline_metrics.json`; added comprehensive unit tests in `tests/test_metrics.py` and `tests/test_baseline.py`.
- **Commands run:** `git checkout -b feature/phase-5-models`, `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/models/baseline.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 70 unit tests in repository passed in 9.47s. Baseline model weights and validation predictions are persisted and ready for comparative benchmarking against the candidate gradient boosted tree in Task 5.2.
- **Evidence/artifact paths:** `src/bankguard/models/baseline.py`, `src/bankguard/models/metrics.py`, `artifacts/baseline_model.joblib`, `artifacts/baseline_predictions.npz`, `report/baseline_metrics.md`, `report/baseline_metrics.json`, `tests/test_metrics.py`, `tests/test_baseline.py`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 5.2 (Train an improved model).

### Task 4.3: Review Leakage Risks and Establish Policy
- **Tasks worked on:** Task 4.3 (Review leakage risks)
- **What I changed:** Authored official written leakage policy in `docs/leakage_policy.md`; implemented `src/bankguard/features/leakage_audit.py` establishing `EXCLUDED_FEATURES_REGISTRY` with documented risk classifications, scoring-moment availability constraints, and mandated treatments for `isFraud`, `TransactionID`, and raw `TransactionDT`; implemented `audit_dataset_leakage()` verifying 0 key overlaps and strict chronological boundary ($\max(TransactionDT_{\text{train}}) = 12,192,842 \le \min(TransactionDT_{\text{val}}) = 12,192,900$); implemented `audit_feature_matrix_leakage()` verifying preprocessor output features strictly eliminate target, ID, and raw linear timestamp; generated `report/leakage_review_report.md` and `report/leakage_review_report.json`; added comprehensive unit tests in `tests/test_leakage_audit.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/features/leakage_audit.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 62 unit tests passed in 5.16s. Confirmed zero leakage violations across all gates; formalized policy prohibiting random K-Fold in favor of chronological out-of-time validation. Phase 4 is complete.
- **Evidence/artifact paths:** `docs/leakage_policy.md`, `src/bankguard/features/leakage_audit.py`, `tests/test_leakage_audit.py`, `report/leakage_review_report.md`, `report/leakage_review_report.json`
- **Problems found:** None.
- **Next exact action:** Phase 4 is complete. Awaiting user approval to create a new branch and proceed to Phase 5 (Train and compare models).

### Task 4.2: Create Preprocessing and Verify Zero-Refit
- **Tasks worked on:** Task 4.2 (Create preprocessing)
- **What I changed:** Implemented `src/bankguard/features/preprocessor.py` with `temporal_train_val_split` (chronological 80/20 train-val split on `TransactionDT`), `BankGuardFeatureEngineer` (incorporating `log1p(TransactionAmt)`, `has_cents`, `cents_fraction`, `hour_of_day`, `sin_hour`, `cos_hour`, `day_of_week`, dropping raw `TransactionDT`), and `BankGuardPreprocessor` (scikit-learn `ColumnTransformer` with median imputation for 407 numeric features and `missing` + `OrdinalEncoder` for 31 categorical features); verified on the full 590,540 rows (472,432 train, 118,108 val); confirmed zero validation refit via cryptographic state signature assertion (`zero_refit_verified: True`); confirmed zero NaNs in transformed output arrays; serialized pipeline artifact to `artifacts/preprocessor.joblib`; generated `report/preprocessing_pipeline_report.md` and `report/preprocessing_pipeline_report.json`; added unit tests in `tests/test_preprocessor.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/features/preprocessor.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 56 unit tests in repo passed in 5.18s. Transforming validation data does not alter any learned training parameters. Transformed output is ready for linear and tree baselines with zero unhandled NaNs.
- **Evidence/artifact paths:** `src/bankguard/features/preprocessor.py`, `tests/test_preprocessor.py`, `artifacts/preprocessor.joblib`, `report/preprocessing_pipeline_report.md`, `report/preprocessing_pipeline_report.json`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 4.3 (Review leakage risks).

### Task 4.1: Define Target and Features (Leakage Prevention)
- **Tasks worked on:** Task 4.1 (Define the target and features)
- **What I changed:** Switched to new branch `feature/phase-4-features`; created feature configuration in `configs/feature_config.yaml` explicitly defining target (`isFraud`), primary identifier (`TransactionID`), exclusions, and categorical families; implemented `src/bankguard/features/feature_definition.py` with `split_features_and_target()` enforcing clean programmatic X/y separation, `get_feature_names()`, `identify_feature_types()`, and automated leakage auditing with `audit_feature_leakage()` and `TargetLeakageError`; verified on the full 590,540 joined dataset (432 features in X, 401 numeric, 31 categorical); generated `report/feature_definition_report.md` and `report/feature_definition_report.json`; added comprehensive unit tests in `tests/test_feature_definition.py` verifying automated absence of target from X.
- **Commands run:** `git checkout -b feature/phase-4-features`, `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/features/feature_definition.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 49 unit tests passed in 3.11s. Automated tests confirm `isFraud` is completely absent from the feature matrix X; any attempted target leakage immediately raises `TargetLeakageError`.
- **Evidence/artifact paths:** `configs/feature_config.yaml`, `src/bankguard/features/feature_definition.py`, `tests/test_feature_definition.py`, `report/feature_definition_report.md`, `report/feature_definition_report.json`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 4.2 (Create preprocessing).

### Task 3.3: Explore Amount, Time, and Key Categorical Drivers
- **Tasks worked on:** Task 3.3 (Explore amount and time)
- **What I changed:** Implemented `src/bankguard/data/amount_time_analysis.py` to evaluate `TransactionAmt` distributions, extreme positive skewness (14.37), quantile spreads, and decimal cent precision signals; evaluated `TransactionDT` ordering, span (15.72M seconds / ~182 days), monotonicity, and cyclical diurnal patterns (off-peak hours 4-9 exhibiting over 7.5% fraud rate vs 3.5% baseline); analyzed high-risk categorical drivers (`ProductCD = 'C'` carrying 11.69% fraud rate; `card6 = 'credit'` carrying 6.68% fraud rate vs 2.43% for debit); generated publication-grade 4-panel visual dashboard `report/figures/amount_time_distribution.png`; generated structured markdown report `report/amount_time_analysis.md` and machine-readable data `report/amount_time_analysis.json`; added comprehensive unit tests in `tests/test_amount_time_analysis.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/data/amount_time_analysis.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 41 unit tests in repo passed in 3.38s. Established critical architectural takeaways for Phase 4: `log1p` transformation for amounts, decimal cent presence indicators, cyclical diurnal trigonometric features, and mandatory time-based validation splits to avoid future lookahead leakage.
- **Evidence/artifact paths:** `report/figures/amount_time_distribution.png`, `report/amount_time_analysis.md`, `report/amount_time_analysis.json`, `tests/test_amount_time_analysis.py`
- **Problems found:** None.
- **Next exact action:** Phase 3 is complete. Awaiting user approval to create a new branch and proceed to Phase 4 (Create leakage-safe features).

### Task 3.2: Investigate Missingness and Fields
- **Tasks worked on:** Task 3.2 (Investigate missingness and fields)
- **What I changed:** Implemented `src/bankguard/data/missingness_analysis.py` to compute overall and class-conditional missingness rates across all 432 feature columns; classified features into 5 standard tiers (`Complete`, `Low`, `Moderate`, `High`, `Extreme`); uncovered predictive missingness discrepancy signals (e.g., `R_emaildomain` null in 77.88% legit vs 45.67% fraud, a -32.21% gap; identity/device fields null in 77.2% legit vs 45.7% fraud); generated publication-grade dual-panel visualization `report/figures/missingness_distribution.png`; exported full rankings to `report/missingness_ranking.csv`, `report/missingness_analysis.md`, and `report/missingness_analysis.json`; added comprehensive unit tests in `tests/test_missingness_analysis.py`.
- **Commands run:** `d:\Projects\BankGurd\.venv\Scripts\python.exe src/bankguard/data/missingness_analysis.py`, `d:\Projects\BankGurd\.venv\Scripts\python.exe -m pytest`
- **Outcome:** All 36 unit tests passed in 2.08s. Confirmed missingness is non-random and carries direct predictive signal for fraud detection. Missing values should not be blindly discarded; native tree handling or missing indicator flags (`_isna`) are recommended.
- **Evidence/artifact paths:** `report/figures/missingness_distribution.png`, `report/missingness_analysis.md`, `report/missingness_analysis.json`, `report/missingness_ranking.csv`, `tests/test_missingness_analysis.py`
- **Problems found:** None.
- **Next exact action:** Awaiting user approval to proceed to Task 3.3 (Explore amount and time).

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
