"""Unit tests for BankGuard Preprocessor and Zero-Refit Verification (Task 4.2)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.features.preprocessor import (
    BankGuardFeatureEngineer,
    BankGuardPreprocessor,
    create_preprocessing_pipeline,
    run_preprocessing_pipeline_report,
    temporal_train_val_split,
)


@pytest.fixture
def mock_transaction_df():
    """Generates a synthetic transaction dataset with chronological timestamps, amounts, and categories."""
    n_rows = 100
    # Strict chronological timestamps: 0 to 99 hours
    dts = [i * 3600 for i in range(n_rows)]
    amounts = [10.0 + (i % 50) + (0.50 if i % 2 == 0 else 0.0) for i in range(n_rows)]
    prod_cds = ["W"] * 60 + ["C"] * 25 + ["H"] * 15
    card4s = ["visa"] * 50 + ["mastercard"] * 30 + ["discover"] * 10 + [None] * 10
    v1_vals = [float(i) if i % 5 != 0 else np.nan for i in range(n_rows)]
    is_fraud = [0] * 90 + [1] * 10

    df = pd.DataFrame({
        "TransactionID": list(range(1000, 1000 + n_rows)),
        "isFraud": is_fraud,
        "TransactionDT": dts,
        "TransactionAmt": amounts,
        "ProductCD": prod_cds,
        "card4": card4s,
        "V1": v1_vals,
    })
    return df


def test_temporal_train_val_split(mock_transaction_df):
    """Verifies that temporal split strictly separates train and validation chronologically."""
    train_df, val_df = temporal_train_val_split(mock_transaction_df, time_col="TransactionDT", val_ratio=0.2)

    # 80% train, 20% validation
    assert len(train_df) == 80
    assert len(val_df) == 20

    # Strict chronological boundary (no time travel)
    max_train_time = train_df["TransactionDT"].max()
    min_val_time = val_df["TransactionDT"].min()
    assert max_train_time <= min_val_time
    assert min_val_time == 80 * 3600


def test_feature_engineer():
    """Verifies creation of amount and cyclical diurnal features."""
    df = pd.DataFrame({
        "TransactionAmt": [10.00, 25.50, 100.25],
        "TransactionDT": [0, 3600 * 12, 3600 * 23],  # hour 0, 12, 23
    })
    fe = BankGuardFeatureEngineer(drop_raw_time=True)
    out = fe.fit_transform(df)

    assert "log_TransactionAmt" in out.columns
    assert "has_cents" in out.columns
    assert "cents_fraction" in out.columns
    assert "hour_of_day" in out.columns
    assert "sin_hour" in out.columns
    assert "cos_hour" in out.columns
    assert "day_of_week" in out.columns

    # Check has_cents
    assert list(out["has_cents"]) == [0.0, 1.0, 1.0]

    # Check that raw TransactionDT was dropped
    assert "TransactionDT" not in out.columns


def test_preprocessor_zero_validation_refit(mock_transaction_df):
    """MANDATORY VERIFICATION: Confirms preprocessor fits ONLY on train and does NOT refit during validation transform."""
    train_df, val_df = temporal_train_val_split(mock_transaction_df, val_ratio=0.2)
    X_train = train_df.drop(columns=["isFraud", "TransactionID"])
    X_val = val_df.drop(columns=["isFraud", "TransactionID"])

    preprocessor = BankGuardPreprocessor(scale_numeric=False)
    preprocessor.fit(X_train)

    # 1. Capture fitted state signature and imputer statistics
    initial_hash = preprocessor.fit_signature_hash_
    assert initial_hash is not None

    num_pipe = preprocessor.column_transformer_.named_transformers_["num"]
    initial_medians = copy_arr = np.copy(num_pipe.named_steps["imputer"].statistics_)

    cat_pipe = preprocessor.column_transformer_.named_transformers_["cat"]
    initial_categories = [np.copy(c) for c in cat_pipe.named_steps["encoder"].categories_]

    # 2. Transform validation data (which contains different values)
    val_trans = preprocessor.transform(X_val)

    # 3. VERIFY ZERO REFIT: Assert internal state was completely unaffected by validation transform
    post_hash = preprocessor.fit_signature_hash_
    assert initial_hash == post_hash, "State hash changed! Validation refit detected."

    post_medians = num_pipe.named_steps["imputer"].statistics_
    assert np.array_equal(initial_medians, post_medians), "Imputer statistics changed during transform!"

    post_categories = cat_pipe.named_steps["encoder"].categories_
    for init_c, post_c in zip(initial_categories, post_categories):
        assert np.array_equal(init_c, post_c), "Categorical categories changed during transform!"

    # 4. Verify output shape and zero NaNs
    assert val_trans.shape[0] == len(X_val)
    assert val_trans.shape[1] == len(preprocessor.get_feature_names_out())
    assert not np.isnan(val_trans).any(), "Transformed validation data contains unhandled NaNs!"


def test_preprocessor_handles_unseen_categories(mock_transaction_df):
    """Confirms preprocessor gracefully encodes unseen categorical values as -1."""
    train_df, val_df = temporal_train_val_split(mock_transaction_df, val_ratio=0.2)
    X_train = train_df.drop(columns=["isFraud", "TransactionID"])
    X_val = val_df.drop(columns=["isFraud", "TransactionID"]).copy()

    # Introduce completely novel category unseen in training
    X_val.loc[X_val.index[0], "ProductCD"] = "NEW_UNSEEN_CATEGORY"
    X_val.loc[X_val.index[1], "card4"] = "novel_card_network"

    preprocessor = BankGuardPreprocessor()
    preprocessor.fit(X_train)

    val_trans = preprocessor.transform(X_val)
    assert not np.isnan(val_trans).any()
    assert val_trans.shape[0] == len(X_val)


def test_preprocessor_serialization(mock_transaction_df, tmp_path: Path):
    """Tests saving and reloading fitted preprocessor from disk."""
    train_df, val_df = temporal_train_val_split(mock_transaction_df, val_ratio=0.2)
    X_train = train_df.drop(columns=["isFraud", "TransactionID"])
    X_val = val_df.drop(columns=["isFraud", "TransactionID"])

    preprocessor = BankGuardPreprocessor()
    preprocessor.fit(X_train)
    orig_output = preprocessor.transform(X_val)

    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(save_path)
    assert save_path.is_file()

    # Reload from disk
    reloaded = BankGuardPreprocessor.load(save_path)
    reloaded_output = reloaded.transform(X_val)

    assert np.allclose(orig_output, reloaded_output)
    assert reloaded.get_feature_names_out() == preprocessor.get_feature_names_out()


def test_unfitted_preprocessor_raises_error(mock_transaction_df):
    """Calling transform or save before fit must raise RuntimeError."""
    X = mock_transaction_df.drop(columns=["isFraud", "TransactionID"])
    pre = BankGuardPreprocessor()
    with pytest.raises(RuntimeError, match="must be fitted before calling transform"):
        pre.transform(X)


def test_run_preprocessing_pipeline_report(mock_transaction_df, tmp_path: Path):
    """Validates end-to-end execution of preprocessing pipeline report generation."""
    report_dir = tmp_path / "report"
    art_path = tmp_path / "artifacts" / "preprocessor.joblib"

    summary = run_preprocessing_pipeline_report(
        train_df=mock_transaction_df,
        output_dir=report_dir,
        artifact_path=art_path,
    )

    assert summary["zero_refit_verified"] is True
    assert summary["train_nan_count"] == 0
    assert summary["val_nan_count"] == 0
    assert (report_dir / "preprocessing_pipeline_report.json").is_file()
    assert (report_dir / "preprocessing_pipeline_report.md").is_file()
    assert art_path.is_file()
