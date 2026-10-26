"""Automated unit tests for Task 4.1: Feature and Target Definition with Leakage Prevention."""

from pathlib import Path
import pandas as pd
import pytest

from bankguard.features.feature_definition import (
    DEFAULT_EXCLUDED_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TargetLeakageError,
    audit_feature_leakage,
    generate_feature_inventory_report,
    get_feature_names,
    identify_feature_types,
    load_feature_config,
    split_features_and_target,
)


@pytest.fixture
def mock_dataset():
    """Generates a small synthetic DataFrame with IDs, target, numeric, and categorical columns."""
    return pd.DataFrame({
        "TransactionID": [1001, 1002, 1003, 1004, 1005],
        "isFraud": [0, 1, 0, 0, 1],
        "TransactionAmt": [50.0, 120.5, 99.0, 1000.0, 15.2],
        "ProductCD": ["W", "C", "W", "H", "R"],
        "card4": ["visa", "mastercard", "visa", "discover", "visa"],
        "C1": [1.0, 2.0, 1.0, 5.0, 1.0],
        "V1": [1.0, 0.0, 1.0, 1.0, 0.0],
    })


def test_get_feature_names_excludes_target_and_id(mock_dataset):
    """Verifies get_feature_names strictly excludes isFraud and TransactionID."""
    features = get_feature_names(mock_dataset)
    assert TARGET_COLUMN not in features
    assert ID_COLUMN not in features
    assert "TransactionAmt" in features
    assert "ProductCD" in features
    assert "card4" in features
    assert len(features) == 5  # 7 columns minus 2 excluded


def test_target_absent_from_X_automated_verification(mock_dataset):
    """MANDATORY VERIFICATION: Automated test confirms the target is completely absent from X."""
    X, y = split_features_and_target(mock_dataset)

    # 1. Target is strictly absent from X
    assert TARGET_COLUMN not in X.columns
    assert "isfraud" not in [c.lower() for c in X.columns]

    # 2. Identifier is strictly absent from X
    assert ID_COLUMN not in X.columns

    # 3. Target y is extracted correctly
    assert y is not None
    assert len(y) == len(mock_dataset)
    assert y.name == TARGET_COLUMN
    assert list(y) == [0, 1, 0, 0, 1]

    # 4. Shapes and indices align
    assert len(X) == len(mock_dataset)
    assert list(X.index) == list(y.index)


def test_explicit_target_leakage_raises_error(mock_dataset):
    """Raises TargetLeakageError if someone deliberately or accidentally specifies isFraud in feature_cols."""
    leaky_cols = ["TransactionAmt", "ProductCD", TARGET_COLUMN]
    with pytest.raises(TargetLeakageError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        split_features_and_target(mock_dataset, feature_cols=leaky_cols)


def test_split_with_return_ids(mock_dataset):
    """Returns a 3-tuple (X, y, ids) when return_ids=True."""
    X, y, ids = split_features_and_target(mock_dataset, return_ids=True)
    assert TARGET_COLUMN not in X.columns
    assert ID_COLUMN not in X.columns
    assert list(ids) == [1001, 1002, 1003, 1004, 1005]
    assert len(ids) == len(X)


def test_audit_feature_leakage_clean_matrix(mock_dataset):
    """Audits clean feature matrix and asserts pass with zero violations."""
    X, _ = split_features_and_target(mock_dataset)
    audit = audit_feature_leakage(X, target_col=TARGET_COLUMN, forbidden_columns=[ID_COLUMN])
    assert audit["passed"] is True
    assert audit["target_absent"] is True
    assert len(audit["violations"]) == 0
    assert audit["feature_count"] == 5


def test_audit_feature_leakage_detects_case_insensitive_leakage():
    """Detects sneaky case-insensitive variants like isfraud or ISFRAUD."""
    leaky_df = pd.DataFrame({"TransactionAmt": [10, 20], "isfraud": [0, 1]})
    with pytest.raises(TargetLeakageError, match="Target Leakage Audit Failed"):
        audit_feature_leakage(leaky_df, target_col="isFraud", raise_on_violation=True)


def test_identify_feature_types(mock_dataset):
    """Accurately partitions features into numeric and categorical lists."""
    X, _ = split_features_and_target(mock_dataset)
    types_dict = identify_feature_types(X)

    assert "numeric" in types_dict
    assert "categorical" in types_dict

    # Check numeric
    assert "TransactionAmt" in types_dict["numeric"]
    assert "C1" in types_dict["numeric"]
    assert "V1" in types_dict["numeric"]

    # Check categorical
    assert "ProductCD" in types_dict["categorical"]
    assert "card4" in types_dict["categorical"]

    # Ensure no overlap
    assert set(types_dict["numeric"]).isdisjoint(set(types_dict["categorical"]))


def test_generate_feature_inventory_report(mock_dataset, tmp_path: Path):
    """Generates feature definition report and audits output files."""
    summary = generate_feature_inventory_report(mock_dataset, output_dir=tmp_path)
    assert summary["leakage_audit_passed"] is True
    assert summary["target_in_features"] is False
    assert summary["id_in_features"] is False
    assert (tmp_path / "feature_definition_report.json").is_file()
    assert (tmp_path / "feature_definition_report.md").is_file()
