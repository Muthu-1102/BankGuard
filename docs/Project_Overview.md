# BankGuard — Project Overview

## 1. What is BankGuard?

**BankGuard** is a portfolio project that demonstrates how a financial-technology team can analyze online transactions, estimate fraud risk, explain model predictions, and expose the result through an API and dashboard.

The project uses the public **IEEE-CIS Fraud Detection** Kaggle competition dataset. It is a historical, anonymized benchmark dataset—not JPMorgan data and not a live banking feed.

### The problem it addresses

Online payment fraud is difficult to detect because:
- Fraudulent transactions are a small minority of all transactions.
- A transaction can look normal in one field and suspicious when several fields are considered together.
- A model's score is not, by itself, an operational decision.
- Teams need measurable performance, traceable decisions, and monitoring for changes in data.

### The solution

BankGuard will:
1. Load the Kaggle transaction and identity files.
2. validate and join records safely.
3. Explore the data and document its limitations.
4. Build a simple baseline model and compare it with a stronger model.
5. produce fraud-risk scores and human-readable explanations where supported.
6. Apply a configurable risk policy to turn scores into review recommendations.
7. Serve predictions through FastAPI and display them in a simple dashboard.
8. Evaluate performance and monitor data/model behavior.

## 2. What BankGuard is—and is not

**In scope**
- Historical e-commerce transaction fraud classification.
- SQL-based data checks and analysis.
- Reproducible feature engineering and model training.
- Fraud-risk scores, a configurable decision policy, and explanations.
- API, dashboard, tests, and basic monitoring.

**Out of scope unless new suitable data is found**
- Claims of detecting money laundering across account networks.
- Graph-based account-ring detection.
- Real-time production banking integration.
- Claims that the system is used by or represents JPMorgan.
- Measuring performance on Kaggle's unlabeled test set using labels that are not available.

## 3. Dataset

**Source:** IEEE-CIS Fraud Detection, Kaggle  
**Competition page:** https://www.kaggle.com/competitions/ieee-fraud-detection/data

Expected files include:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The training transaction file contains the target column `isFraud`. Identity information is joined using `TransactionID`. Some transactions have no matching identity row, so use a **left join** and measure the missingness.

Do not commit downloaded data to Git unless the dataset's terms and repository policy permit it. Keep raw data outside source control.

## 4. High-level system flow

```text
Kaggle CSV files
      |
      v
Load + validate + record data quality
      |
      v
Join transaction and identity data on TransactionID
      |
      v
Exploration + leakage-safe feature pipeline
      |
      v
Train / validation split
      |
      v
Baseline model -> improved model -> evaluation
      |
      v
Risk score + explanation + configurable policy
      |
      v
FastAPI prediction service + dashboard
      |
      v
Tests + monitoring + documented results
```

## 5. Suggested technology stack

| Area | Technology | Purpose |
|---|---|---|
| Language | Python | Data and model code |
| Analysis | pandas, NumPy, Matplotlib/Seaborn | Inspect and understand data |
| SQL | PostgreSQL (optional for first milestone) | Store metadata/results and demonstrate SQL |
| ML | scikit-learn; LightGBM or XGBoost if available | Baseline and stronger model |
| Explainability | SHAP, if compatible | Explain model output |
| API | FastAPI, Pydantic | Serve predictions |
| UI | Streamlit | Demonstrate prediction and metrics |
| Testing | pytest | Verify expected behavior |
| Packaging | Docker | Reproducible execution |
| CI | GitHub Actions | Run automated checks |

Start with the simplest working stack. Do not add a tool merely to make the architecture look impressive.

## 6. What success looks like

A completed project has:
- A repeatable data preparation pipeline.
- A written data card and data dictionary.
- A baseline and a stronger model evaluated on a held-out validation set.
- PR-AUC and recall reported alongside precision, confusion matrix, and ROC-AUC.
- A documented threshold-selection method rather than an arbitrary claim of “best accuracy.”
- An API that validates input and returns a score and policy outcome.
- A dashboard showing model performance and example predictions.
- Tests, setup instructions, known limitations, and reproducible results.

## 7. Definition of done

Mark the project complete only when each item below can be demonstrated.

- [x] Data files are present locally and their source is documented.
- [x] A data validation report is generated (`report/data_quality_report.md`, `report/data_inventory_report.md`).
- [x] The joined dataset has one row per `TransactionID`; duplicate handling is tested (`report/join_report.md`).
- [x] Train/validation separation is reproducible and prevents target leakage (`docs/leakage_policy.md`, `report/leakage_review_report.md`).
- [x] Baseline and improved models are compared on the same validation split (`report/model_comparison.md`).
- [x] Metrics include PR-AUC, precision, recall, F1, ROC-AUC, and confusion matrix (`report/improved_metrics.md`).
- [x] The chosen operating threshold is documented (`report/threshold_evaluation.md`, `report/decision_policy_report.md`).
- [x] Explanations are tested on sample predictions and described as model explanations, not proof of fraud (`report/model_explainability.md`).
- [x] API tests cover valid input, invalid input, and model loading failure (`tests/test_api.py`).
- [x] Dashboard instructions work on a clean environment (`dashboard.py`, `src/bankguard/dashboard/app.py`).
- [x] Monitoring checks and limitations are documented (`report/monitoring_summary.md`, `docs/containerization.md`).
- [x] README commands reproduce the main results (`README.md`).

## 8. How to verify the project as a reviewer

1. Follow the setup steps from a clean clone.
2. Run the data validation command and inspect the generated report.
3. Run the training command and compare its metrics with the saved evaluation report.
4. Start the API and request a prediction using an example input generated by the project.
5. Open the dashboard and verify that displayed metrics match the evaluation artifact.
6. Run `pytest` and confirm all required tests pass.
7. Read `data_card.md` and `model_evaluation.md` to understand what the project can and cannot claim.
