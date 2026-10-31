# BankGuard — Model Evaluation Plan

## 1. Purpose

This document defines how to measure BankGuard fairly. The goal is not to produce a single impressive score; it is to understand how well the model identifies fraud, how many legitimate transactions it flags, and how the model behaves at different decision thresholds.

## 2. Evaluation data

Use labeled rows from `train_transaction.csv`, split into:
- **Training set:** used to fit the model and preprocessing.
- **Validation set:** used to compare models, choose a threshold, and report the project's main metrics.
- **Optional final holdout:** if sufficient data is available, reserve it until model and threshold decisions are complete.

Use the same split for all candidate models. A fixed random seed makes results repeatable. Because the data is historical and time-related, consider a chronological split as a robustness check; document the exact rule and do not claim a random split simulates future deployment.

Do not use the unlabeled competition test set to calculate local fraud metrics. Do not use `isFraud` as an input feature.

## 3. Metrics to report

| Metric | What it tells you | Why it matters here | Main caution |
|---|---|---|---|
| Confusion matrix | Counts of true positives, false positives, true negatives, and false negatives at a threshold | Shows the kinds of mistakes | Counts depend on the chosen threshold and validation population |
| Precision | Of transactions flagged as fraud, how many are labeled fraud | Helps estimate review burden | Can fall when recall is increased |
| Recall | Of labeled fraud transactions, how many are detected | Measures missed-fraud reduction | High recall can create many false positives |
| F1 | Harmonic mean of precision and recall | Useful single-threshold summary | Treats precision and recall symmetrically |
| PR-AUC / Average Precision | Precision-recall performance across thresholds | Especially useful when fraud is rare | Values depend on the class prevalence and exact implementation |
| ROC-AUC | Ranking ability across thresholds | Common comparison metric | Can look strong with severe class imbalance even when precision is poor |
| Accuracy | Fraction of all predictions that are correct | Include for context only | Can be misleading when most rows are non-fraud |
| Inference latency | Time required to score a record/batch | Useful for a demo service | Record hardware, batch size, and measurement method |

Report class counts and fraud prevalence beside metrics.

## 4. Required model comparison

Compare at least:
1. **Baseline:** a simple, well-understood model with preprocessing.
2. **Improved candidate:** a stronger tabular model.

Use the same validation rows and report:

| Item | Baseline | Improved model |
|---|---:|---:|
| Model name/version | `LogisticRegression (balanced)` | `LightGBM 4.7.0 (GBDT)` |
| Feature set/version | `v1.0 (438 features)` | `v1.0 (438 features)` |
| PR-AUC / Average Precision | `0.1420` | `0.5598` (+294.2%) |
| ROC-AUC | `0.7844` | `0.9136` (+16.5%) |
| Brier Score | `0.1716` | `0.0212` (-87.6%) |
| Log Loss | `0.6191` | `0.0867` (-86.0%) |
| Precision at default threshold (0.50) | `0.1203` | `0.8107` (+573.9%) |
| Recall at default threshold (0.50) | `0.6228` | `0.3573` (-42.6%) |
| F1 at default threshold (0.50) | `0.2017` | `0.4960` (+145.9%) |
| Specificity | `0.8377` | `0.9970` (+19.0%) |
| False positives | `18,507` | `339` (-98.2%) |
| False negatives | `1,533` | `2,612` |
| Inference latency (vectorized batch) | `~0.15 ms/txn` | `~0.07 ms/txn` |
| Training duration | `96.69 s` | `12.01 s` |

Do not fill this table with example or invented scores. Use values calculated by the project.

## 5. Threshold selection

A model produces a score; a policy converts it into an action recommendation. The threshold determines the trade-off between missed fraud and false alarms.

### Procedure
1. Generate validation probabilities.
2. Evaluate a range of thresholds.
3. For each threshold, calculate precision, recall, F1, false positives, and false negatives.
4. Define an explicit operating objective, for example:
   - maximize recall while keeping the false-positive/review rate below a stated capacity; or
   - meet a minimum precision while maximizing recall.
5. Choose the threshold using validation data only.
6. Record the objective, selected threshold, and resulting metrics.
7. Do not tune the threshold on the final holdout.

For a portfolio demonstration, thresholds can be illustrative. State clearly that they are not production banking rules or calibrated financial policy.

## 6. Leakage and validity checklist

Before accepting a result, verify:

- [ ] `isFraud` is excluded from all feature matrices.
- [ ] Preprocessing is fitted only on training rows.
- [ ] Validation rows were not used for model fitting or feature selection.
- [ ] The same split is used for candidate model comparison.
- [ ] Duplicate `TransactionID` values are checked.
- [ ] Any resampling is applied only inside the training process.
- [ ] The chosen threshold is selected on validation data, not the final holdout.
- [ ] The unlabeled Kaggle test set is not treated as labeled evaluation data.
- [ ] Feature availability at scoring time is considered.
- [ ] Results include class prevalence and confusion matrix, not accuracy alone.
- [ ] Any time-based evaluation is documented separately from random-split results.

## 7. Error analysis

Review examples of:
- False negatives: labeled fraud predicted below the chosen threshold.
- False positives: labeled non-fraud predicted above the chosen threshold.
- High-confidence correct predictions.
- Records with substantial missingness.
- Records with and without matching identity data.

For each group, inspect aggregate patterns and a small number of records responsibly. Do not publish raw identifiers or unnecessary device/email details in the report.

**Outcome:** a written account of where the model succeeds and fails, not just a score.

**Verify:** every conclusion is supported by validation predictions and calculated counts.

## 8. Model explanation

If SHAP or another explanation method is used:
- Explain global feature importance and selected individual predictions.
- State the explainer and model version.
- Make clear that contributions describe model behavior, not causality.
- Do not interpret anonymized fields as real business concepts unless official documentation supports that meaning.
- Verify explanations on multiple records and check that the displayed model score matches the score produced by the saved model.

## 9. Reproducibility record

For every accepted model run, save:
- Run date/time.
- Dataset file names and hashes if practical.
- Feature list/version.
- Train/validation split method and seed.
- Model name and hyperparameters.
- Python and package versions.
- Threshold-selection objective and selected threshold.
- Validation predictions and metrics.
- Training duration and inference measurement method.

## 10. Acceptance criteria

A model is not accepted merely because it has a high accuracy score. It is accepted when:
- Its evaluation is leakage-aware and reproducible.
- PR-AUC and threshold-specific metrics are reported.
- False-positive and false-negative trade-offs are explained.
- It is compared fairly with the baseline.
- The threshold rationale is documented.
- Limitations and data representativeness are stated.
- The results can be recalculated from saved validation predictions.
