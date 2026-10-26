"""Unit tests for BankGuard data quality checks.

Tests pass on valid fixtures and intentionally fail on invalid fixtures.
"""

from pathlib import Path
import pandas as pd
import pytest

from bankguard.data.quality_checks import (
    DataQualityError,
    check_key_uniqueness,
    check_null_patterns,
    check_numeric_ranges,
    check_required_columns,
    check_target_validity,
    run_data_quality_suite,
)


@pytest.fixture
def valid_tx_df():
    """Valid transaction fixture."""
    return pd.DataFrame({
        "TransactionID": [1001, 1002, 1003],
        "isFraud": [0, 1, 0],
        "TransactionDT": [86400, 86450, 86500],
        "TransactionAmt": [35.5, 120.0, 50.0],
        "ProductCD": ["W", "H", "C"],
    })


@pytest.fixture
def valid_id_df():
    """Valid identity fixture."""
    return pd.DataFrame({
        "TransactionID": [1001, 1002],
        "DeviceType": ["desktop", "mobile"],
    })


# --- 1. Required Columns Checks ---

def test_check_required_columns_valid(valid_tx_df):
    """Passes when all required columns are present."""
    res = check_required_columns(valid_tx_df, ["TransactionID", "isFraud"])
    assert res.passed is True


def test_check_required_columns_missing_column(valid_tx_df):
    """Fails when a required column is absent."""
    res = check_required_columns(valid_tx_df, ["TransactionID", "non_existent_column"])
    assert res.passed is False
    assert "non_existent_column" in res.details["missing_columns"]


# --- 2. Key Uniqueness Checks ---

def test_check_key_uniqueness_valid(valid_tx_df):
    """Passes when keys are unique and non-null."""
    res = check_key_uniqueness(valid_tx_df, "TransactionID")
    assert res.passed is True


def test_check_key_uniqueness_duplicate():
    """Fails when duplicate TransactionIDs exist."""
    df_dup = pd.DataFrame({"TransactionID": [1001, 1002, 1001]})
    res = check_key_uniqueness(df_dup, "TransactionID")
    assert res.passed is False
    assert res.details["duplicate_count"] == 1


def test_check_key_uniqueness_null():
    """Fails when null TransactionIDs exist."""
    df_null = pd.DataFrame({"TransactionID": [1001, None, 1002]})
    res = check_key_uniqueness(df_null, "TransactionID")
    assert res.passed is False
    assert res.details["null_count"] == 1


# --- 3. Target Validity Checks ---

def test_check_target_validity_valid(valid_tx_df):
    """Passes when target is {0, 1}."""
    res = check_target_validity(valid_tx_df, "isFraud")
    assert res.passed is True


def test_check_target_validity_invalid_values():
    """Fails when target contains invalid values (e.g., 2 or -1)."""
    df_invalid = pd.DataFrame({"isFraud": [0, 1, 2, -1]})
    res = check_target_validity(df_invalid, "isFraud")
    assert res.passed is False
    assert set(res.details["invalid_values"]) == {2, -1}


def test_check_target_validity_null_target():
    """Fails when target contains nulls."""
    df_null_target = pd.DataFrame({"isFraud": [0, 1, None]})
    res = check_target_validity(df_null_target, "isFraud")
    assert res.passed is False
    assert res.details["null_count"] == 1


# --- 4. Numeric Range Checks ---

def test_check_numeric_ranges_valid(valid_tx_df):
    """Passes when amounts are non-negative and non-null."""
    res = check_numeric_ranges(valid_tx_df, "TransactionAmt")
    assert res.passed is True


def test_check_numeric_ranges_negative_amount():
    """Fails when amounts are negative."""
    df_neg = pd.DataFrame({"TransactionAmt": [50.0, -10.0, 25.0]})
    res = check_numeric_ranges(df_neg, "TransactionAmt")
    assert res.passed is False
    assert res.details["negative_count"] == 1


# --- 5. Full Quality Suite & Report Generation ---

def test_run_data_quality_suite_valid(valid_tx_df, valid_id_df, tmp_path: Path):
    """Suite passes and generates JSON and Markdown reports on valid fixtures."""
    summary = run_data_quality_suite(
        valid_tx_df, valid_id_df, is_training=True, output_dir=tmp_path
    )
    assert summary["overall_status"] == "PASSED"
    assert summary["critical_failures_count"] == 0
    assert (tmp_path / "data_quality_report.json").is_file()
    assert (tmp_path / "data_quality_report.md").is_file()


def test_run_data_quality_suite_raises_on_critical_failure():
    """Suite raises DataQualityError when critical checks fail."""
    bad_df = pd.DataFrame({"TransactionID": [1, 1], "isFraud": [5, 5]})
    with pytest.raises(DataQualityError) as exc_info:
        run_data_quality_suite(bad_df, is_training=True, fail_on_critical=True)

    assert "Critical data quality checks failed" in str(exc_info.value)
