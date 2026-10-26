"""Unit tests for BankGuard target analysis module (Task 3.1)."""

from pathlib import Path
import pandas as pd
import pytest

from bankguard.data.target_analysis import TargetAnalysisResult, analyze_target


@pytest.fixture
def mock_df():
    """Mock DataFrame with 90 legitimate and 10 fraud transactions (10% fraud rate)."""
    return pd.DataFrame({
        "TransactionID": list(range(100)),
        "isFraud": [1] * 10 + [0] * 90,
        "TransactionAmt": [50.0] * 100,
    })


def test_analyze_target_metrics(mock_df, tmp_path: Path):
    """Accurately calculates class counts, percentages, and verifies total."""
    result = analyze_target(
        mock_df,
        target_col="isFraud",
        figures_dir=tmp_path / "figures",
        output_dir=tmp_path,
        verbose=False,
    )

    assert result.total_rows == 100
    assert result.fraud_count == 10
    assert result.legit_count == 90
    assert result.fraud_percentage == pytest.approx(10.0)
    assert result.legit_percentage == pytest.approx(90.0)
    assert result.imbalance_ratio == pytest.approx(9.0)  # 90 / 10
    assert result.counts_match_total is True

    # Output files created
    assert (tmp_path / "figures" / "target_distribution.png").is_file()
    assert (tmp_path / "target_analysis.md").is_file()
    assert (tmp_path / "target_analysis.json").is_file()


def test_analyze_target_missing_column():
    """Raises KeyError when target column is missing."""
    df_no_target = pd.DataFrame({"TransactionID": [1, 2], "TransactionAmt": [10.0, 20.0]})
    with pytest.raises(KeyError) as exc_info:
        analyze_target(df_no_target, target_col="isFraud", verbose=False)
    assert "Target column 'isFraud' not found" in str(exc_info.value)


def test_analyze_target_zero_fraud():
    """Handles edge case of zero fraud gracefully."""
    df_no_fraud = pd.DataFrame({"isFraud": [0, 0, 0, 0]})
    result = analyze_target(df_no_fraud, target_col="isFraud", figures_dir=None, output_dir=None, verbose=False)
    assert result.fraud_count == 0
    assert result.fraud_percentage == 0.0
    assert result.imbalance_ratio == float("inf")
