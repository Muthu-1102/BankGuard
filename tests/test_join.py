"""Unit tests for BankGuard safe identity join module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.data.join import (
    JoinCardinalityError,
    JoinMetrics,
    safe_join_transaction_identity,
    save_join_report,
)


@pytest.fixture
def mock_transaction_df():
    """Sample transaction DataFrame with 3 unique transactions."""
    return pd.DataFrame({
        "TransactionID": [101, 102, 103],
        "isFraud": [0, 1, 0],
        "TransactionAmt": [50.0, 100.0, 20.0],
    })


@pytest.fixture
def mock_identity_df():
    """Sample identity DataFrame matching 2 of the 3 transactions."""
    return pd.DataFrame({
        "TransactionID": [101, 102],
        "DeviceType": ["mobile", "desktop"],
        "id_01": [-5.0, 0.0],
    })


def test_safe_join_success(mock_transaction_df, mock_identity_df):
    """Left join preserves transaction rows and attaches identity features."""
    joined_df, metrics = safe_join_transaction_identity(
        mock_transaction_df, mock_identity_df, verbose=False
    )

    # Invariant assertions
    assert len(joined_df) == len(mock_transaction_df) == 3
    assert joined_df["TransactionID"].tolist() == [101, 102, 103]
    assert joined_df["TransactionID"].duplicated().sum() == 0

    # Unmatched transaction receives NaN for identity columns
    assert joined_df.loc[joined_df["TransactionID"] == 103, "DeviceType"].isna().iloc[0]

    # Metrics
    assert metrics.identity_matches == 2
    assert metrics.identity_match_rate_pct == pytest.approx(66.67, rel=1e-2)
    assert metrics.unmatched_transactions == 1
    assert metrics.joined_cols == 5  # 3 tx cols + 2 id cols


def test_safe_join_missing_key_in_transaction(mock_identity_df):
    """Raises KeyError when key column is absent from transaction table."""
    bad_tx = pd.DataFrame({"WrongID": [101, 102], "isFraud": [0, 1]})
    with pytest.raises(KeyError) as exc_info:
        safe_join_transaction_identity(bad_tx, mock_identity_df, verbose=False)
    assert "Primary join key 'TransactionID' not found" in str(exc_info.value)


def test_safe_join_missing_key_in_identity(mock_transaction_df):
    """Raises KeyError when key column is absent from identity table."""
    bad_id = pd.DataFrame({"WrongID": [101, 102], "DeviceType": ["mobile", "desktop"]})
    with pytest.raises(KeyError) as exc_info:
        safe_join_transaction_identity(mock_transaction_df, bad_id, verbose=False)
    assert "Join key 'TransactionID' not found in identity table" in str(exc_info.value)


def test_safe_join_duplicate_in_identity_raises_cardinality_error(mock_transaction_df):
    """Raises JoinCardinalityError when identity table has duplicates to prevent row multiplication."""
    dup_id = pd.DataFrame({
        "TransactionID": [101, 101],  # Duplicate key
        "DeviceType": ["mobile", "desktop"],
    })
    with pytest.raises(JoinCardinalityError) as exc_info:
        safe_join_transaction_identity(mock_transaction_df, dup_id, verbose=False)

    assert "Identity table contains 1 duplicate 'TransactionID'" in str(exc_info.value)


def test_safe_join_duplicate_in_transactions_raises_cardinality_error(mock_identity_df):
    """Raises JoinCardinalityError when transaction table has duplicate keys."""
    dup_tx = pd.DataFrame({
        "TransactionID": [101, 101, 102],
        "isFraud": [0, 0, 1],
    })
    with pytest.raises(JoinCardinalityError) as exc_info:
        safe_join_transaction_identity(dup_tx, mock_identity_df, verbose=False)

    assert "Transaction table contains 1 duplicate 'TransactionID'" in str(exc_info.value)


def test_save_join_report(tmp_path: Path):
    """Saves Markdown and JSON join reports."""
    metrics = JoinMetrics(
        transaction_rows=100,
        identity_rows=30,
        joined_rows=100,
        identity_matches=30,
        identity_match_rate_pct=30.0,
        unmatched_transactions=70,
        transaction_cols=10,
        identity_cols=5,
        joined_cols=14,
        elapsed_seconds=0.05,
    )
    save_join_report(metrics, output_dir=tmp_path)

    assert (tmp_path / "join_report.json").is_file()
    assert (tmp_path / "join_report.md").is_file()
