# BankGuard — Functionality Specification

## 1. Purpose

This document defines what BankGuard must do. Each feature includes the task, implementation approach, expected result, and verification method.

## 2. Priority definitions

- **P0 — Required:** project cannot be considered complete without it.
- **P1 — Important:** required for a strong portfolio demonstration after P0 works.
- **P2 — Optional:** add only if earlier work is stable and there is time.

## 3. Functional requirements

| ID | Priority | Requirement | What to do / how to do it | Expected outcome | How to verify |
|---|---|---|---|---|---|
| BG-01 | P0 | Load source data | Read the four IEEE-CIS CSV files from configured paths; fail with a useful message if absent | Reproducible loading step | Run loader with valid and missing paths |
| BG-02 | P0 | Validate schema | Check required files/columns, data types, target values, and key uniqueness | Clear pass/fail report | Tests for missing column, duplicate ID, invalid target |
| BG-03 | P0 | Join identity data | Left-join by `TransactionID`; validate one-to-one or many-to-one assumptions based on actual key checks | Joined data preserves transaction rows | Assert joined row count equals transaction row count |
| BG-04 | P0 | Data quality report | Measure missingness, duplicates, target distribution, and identity match rate | Human-readable report | Compare report values with direct pandas calculations |
| BG-05 | P0 | Exploratory analysis | Analyze class imbalance, transaction amount, time ordering, missingness, and categorical distributions | EDA report with plots and conclusions | Every chart has title, axis labels, and written interpretation |
| BG-06 | P0 | Leakage-safe features | Keep `isFraud` out of X; fit transformations only on training data; document feature exclusions | Repeatable feature matrix | Automated test checks target absent from features |
| BG-07 | P0 | Baseline model | Train a simple model, such as logistic regression or a tree-based baseline, with suitable preprocessing | Baseline metrics and saved predictions | Training command finishes and metrics can be recomputed |
| BG-08 | P0 | Improved model | Train a stronger tabular model, such as LightGBM/XGBoost or a well-supported alternative | Candidate model for comparison | Same split and metrics as baseline |
| BG-09 | P0 | Model evaluation | Report PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, and threshold analysis | Evaluation artifact with honest conclusions | Recompute metrics from saved validation predictions |
| BG-10 | P0 | Decision policy | Map model scores to configurable recommendations, with thresholds documented as illustrative until calibrated | Consistent recommendation for a score | Unit tests below, at, and above thresholds |
| BG-11 | P0 | Prediction API | Provide a health endpoint and a prediction endpoint using a validated request schema | JSON response with score and decision | API tests for valid/invalid requests and model unavailable |
| BG-12 | P0 | Reproducible setup | Document environment, data setup, commands, and expected outputs | Another person can run the project | Follow README from a clean environment |
| BG-13 | P0 | Automated tests | Test validation, joins, features, policy, API, and metric computation | Repeatable test suite | `pytest` exits successfully |
| BG-14 | P1 | Explainability | Add SHAP or another compatible method; report global and per-example feature contributions where supported | Explanation view for a prediction | Explanation runs on test records and values align with model output method |
| BG-15 | P1 | Dashboard | Show dataset summary, evaluation metrics, threshold, and demo predictions | Simple reviewer-friendly UI | Dashboard launches and displays values from saved artifacts |
| BG-16 | P1 | Monitoring | Track prediction errors, missing input rates, score distributions, and data quality; avoid storing unnecessary sensitive data | Monitoring summary | Feed a test batch and verify counts/statistics update |
| BG-17 | P1 | Containerization | Package API and dashboard with documented configuration | Repeatable local launch | Build and run Docker image using documented commands |
| BG-18 | P1 | CI | Run linting/tests on pull requests or pushes | Automated quality checks | Create a test commit and inspect workflow result |
| BG-19 | P2 | Time-based robustness check | If a defensible chronological validation can be created from `TransactionDT`, compare it with the random split | Evidence about temporal generalization | Document split rules and compare metrics |
| BG-20 | P2 | SQL analytics | Store derived summary tables or query source data through a documented SQL layer | Demonstrates data engineering and SQL | Run saved queries and compare totals with pandas |

## 4. Prediction contract

The API should return a response conceptually like:

```json
{
  "transaction_id": "example-id",
  "fraud_score": 0.73,
  "decision": "review",
  "model_version": "bankguard-model-v1",
  "explanation": {
    "status": "available",
    "summary": "Example explanation generated by the configured explainer"
  }
}
```

This is an example response shape, not a claim that these exact fields are already implemented. The score and decision must be generated by the actual loaded model and configured policy. The explanation must not be fabricated.

### Decision policy

Use configurable thresholds, for example:
- score below the lower threshold → `allow` recommendation;
- score between thresholds → `review`;
- score above the upper threshold → `high_risk` recommendation.

The initial thresholds are **demo policy values**, not calibrated banking rules. Choose final thresholds using validation results and an explicitly stated cost/recall/precision objective. Do not describe a recommendation as a real transaction approval or decline.

## 5. Non-functional requirements

| Area | Requirement | How to implement | Verification |
|---|---|---|---|
| Reproducibility | Fixed random seeds and recorded versions | Config file, requirements lock/pin where practical | Re-run split and compare results |
| Correctness | Invalid data fails clearly | Explicit assertions and exceptions | Negative tests |
| Security | No credentials or raw dataset in Git | `.gitignore`, environment variables, repository scan | Inspect tracked files and secrets scan |
| Privacy | Avoid unnecessary raw-field logging | Log IDs only when appropriate, status, latency, score summaries | Inspect logs from a test run |
| Performance | Record inference latency on a defined sample | Benchmark script | Save sample size, machine context, and latency statistics |
| Maintainability | Separate data, features, model, policy, API, UI | Small modules and clear interfaces | New developer can locate each responsibility |
| Explainability | State limitations and uncertainty | Explain model contributions; do not claim causality | Review explanation documentation |
| Accessibility | Clear labels and error messages | UI/API validation | Test empty, invalid, and boundary inputs |

## 6. Out of scope

Do not implement or claim the following from IEEE-CIS alone:
- Account-to-account money movement graphs.
- Money-laundering network discovery.
- Identity verification of actual customers.
- Live payment authorization.
- Production-grade fraud operations or regulatory compliance.
- JPMorgan-specific behavior or integration.

## 7. Release acceptance checklist

- [ ] All P0 requirements pass.
- [ ] Tests cover both expected and failure behavior.
- [ ] Metrics are based on a held-out labeled validation set.
- [ ] Threshold rationale and limitations are written.
- [ ] The API and dashboard use the same saved model artifact.
- [ ] Documentation does not claim unsupported capabilities.
