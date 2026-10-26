"""Unit tests for BankGuard Leakage Audit and Excluded Features Registry (Task 4.3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.features.leakage_audit import (
    EXCLUDED_FEATURES_REGISTRY,
    audit_dataset_leakage,
    audit_feature_matrix_leakage,
    generate_leakage_audit_report,
)
from bankguard.features.preprocessor import (
    BankGuardPreprocessor,
    temporal_train_val_split,
)


@pytest.fixture
def mock_split_data():
    """Generates clean chronological train and validation splits with known timestamps and IDs."""
    train_df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 0, 1, 0, 0],
        "TransactionDT": [100, 200, 300, 400, 500],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ProductCD": ["W", "W", "C", "H", "W"],
    })
    val_df = pd.DataFrame({
        "TransactionID": [6, 7, 8],
        "isFraud": [0, 1, 0],
        "TransactionDT": [600, 700, 800],
        "TransactionAmt": [60.0, 70.0, 80.0],
        "ProductCD": ["W", "C", "R"],
    })
    return train_df, val_df


def test_excluded_features_registry_completeness():
    """Verifies every excluded field has documented risk category and handling rationale."""
    assert len(EXCLUDED_FEATURES_REGISTRY) >= 3

    excluded_names = [p.column_name for p in EXCLUDED_FEATURES_REGISTRY]
    assert "isFraud" in excluded_names
    assert "TransactionID" in excluded_names
    assert "TransactionDT" in excluded_names

    for policy in EXCLUDED_FEATURES_REGISTRY:
        assert policy.leakage_risk_type != ""
        assert policy.rationale != ""
        assert policy.recommended_treatment != ""


def test_audit_dataset_leakage_clean(mock_split_data):
    """Clean temporal split with zero key overlap passes all audit checks."""
    train_df, val_df = mock_split_data
    res = audit_dataset_leakage(train_df, val_df)

    assert res["passed"] is True
    assert len(res["violations"]) == 0
    assert res["key_overlap_count"] == 0
    assert res["temporal_order_valid"] is True
    assert res["max_train_timestamp"] == 500
    assert res["min_val_timestamp"] == 600


def test_audit_dataset_leakage_detects_key_overlap(mock_split_data):
    """Detects when same TransactionID exists in both train and validation."""
    train_df, val_df = mock_split_data
    # Contaminate validation set with a train key
    val_df_corrupt = val_df.copy()
    val_df_corrupt.loc[0, "TransactionID"] = 1

    res = audit_dataset_leakage(train_df, val_df_corrupt)
    assert res["passed"] is False
    assert res["key_overlap_count"] == 1
    assert any("contamination detected" in v for v in res["violations"])


def test_audit_dataset_leakage_detects_temporal_lookahead(mock_split_data):
    """Detects when validation records precede training records."""
    train_df, val_df = mock_split_data
    # Corrupt temporal ordering: validation transaction occurs at t=50 (before train max=500)
    val_df_corrupt = val_df.copy()
    val_df_corrupt.loc[0, "TransactionDT"] = 50

    res = audit_dataset_leakage(train_df, val_df_corrupt)
    assert res["passed"] is False
    assert res["temporal_order_valid"] is False
    assert any("Temporal lookahead violation" in v for v in res["violations"])


def test_audit_feature_matrix_leakage(mock_split_data):
    """Verifies that BankGuardPreprocessor output features exclude target, id, and raw timestamp."""
    train_df, val_df = mock_split_data
    X_train = train_df.drop(columns=["isFraud", "TransactionID"])

    preprocessor = BankGuardPreprocessor(drop_raw_time=True)
    preprocessor.fit(X_train)

    audit = audit_feature_matrix_leakage(preprocessor, X_train)
    assert audit["passed"] is True
    assert audit["target_excluded"] is True
    assert audit["id_excluded"] is True
    assert audit["raw_time_excluded"] is True
    assert len(audit["violations"]) == 0


def test_generate_leakage_audit_report(mock_split_data, tmp_path: Path):
    """Generates leakage audit reports and checks file creation."""
    train_df, val_df = mock_split_data
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    summary = generate_leakage_audit_report(full_df, output_dir=tmp_path)
    assert summary["overall_leakage_audit_passed"] is True
    assert (tmp_path / "leakage_review_report.json").is_file()
    assert (tmp_path / "leakage_review_report.md").is_file()
