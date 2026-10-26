# BankGuard — Preprocessing Pipeline & Zero-Refit Verification (Task 4.2)

This report documents the implementation and verification of the scikit-learn compatible preprocessing pipeline, fitted strictly on training data and tested on chronological validation data.

---

## 1. Pipeline Verification Summary

- **Chronological Split:** 80% Train (472,432 rows) / 20% Validation (118,108 rows) ordered by `TransactionDT`
- **Zero-Refit Verification:** **PASSED (True)** — State signature hash is identical before and after validation transform
- **Transformed Train Output Shape:** `472,432 rows x 438 features`
- **Transformed Validation Output Shape:** `118,108 rows x 438 features`
- **NaNs in Transformed Train:** 0 (0% missing)
- **NaNs in Transformed Validation:** 0 (0% missing)
- **Saved Pipeline Artifact:** [`artifacts\preprocessor.joblib`](artifacts\preprocessor.joblib)

---

## 2. Engineered Features

The preprocessing step automatically incorporates domain features derived from EDA:
1. `log_TransactionAmt`: Log-transformed payment amount (`log1p(TransactionAmt)`), dampening the extreme 14.37 skewness.
2. `has_cents`: Binary flag for fractional decimal cents (capturing foreign exchange / micro-testing transactions).
3. `cents_fraction`: Sub-dollar decimal precision.
4. `hour_of_day`: Relative hour within diurnal 24-hour cycle (0..23).
5. `sin_hour` & `cos_hour`: Continuous trigonometric cyclical hour features.
6. `day_of_week`: Day-of-week relative proxy (0..6).
7. `TransactionDT` Exclusion: Raw linear timestamp dropped to prevent non-stationary drift leakage while preserving cyclical temporal patterns.

---

## 3. Imputation and Encoding Specification

- **Numeric Pipeline (407 columns):**
  - Imputer: `SimpleImputer(strategy='median')` learned strictly from `X_train`.
  - Out-of-bounds/NaN in validation: Imputed using training medians.
- **Categorical Pipeline (31 columns):**
  - Missingness: Imputed as string `'missing'`.
  - Encoder: `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
  - Unseen categories in validation/inference: Encoded safely as `-1` without raising exceptions.

---

## 4. Production Reusability

The fitted preprocessor is saved as [`artifacts/preprocessor.joblib`](../artifacts/preprocessor.joblib) and can be loaded dynamically during batch evaluation (Phase 5–6) or real-time single-transaction API scoring (Phase 8).
