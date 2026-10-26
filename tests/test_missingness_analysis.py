"""Unit tests for BankGuard missingness analysis module (Task 3.2)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.data.missingness_analysis import (
    assign_missingness_tier,
    compute_missingness_table,
    generate_missingness_report,
    plot_missingness_analysis,
)


@pytest.fixture
def mock_missing_df():
    """Fixture with controlled missingness patterns across 10 fraud and 90 legit rows."""
    # 10 fraud (isFraud=1), 90 legit (isFraud=0)
    data = {
        "TransactionID": list(range(100)),
        "isFraud": [1] * 10 + [0] * 90,
        "col_complete": list(range(100)),  # 0% null
        "col_low": [None] * 5 + list(range(95)),  # 5% null
        # In col_signal: 8 nulls in fraud (80% fraud null), 10 nulls in legit (11.1% legit null)
        "col_signal": [None] * 8 + [1.0] * 2 + [None] * 10 + [2.0] * 80,
    }
    return pd.DataFrame(data)


def test_assign_missingness_tier():
    """Validates missingness tier threshold boundaries."""
    assert assign_missingness_tier(0.0) == "Complete (0%)"
    assert assign_missingness_tier(15.5) == "Low (<20%)"
    assert assign_missingness_tier(35.0) == "Moderate (20-50%)"
    assert assign_missingness_tier(75.0) == "High (50-80%)"
    assert assign_missingness_tier(95.0) == "Extreme (>=80%)"


def test_compute_missingness_table(mock_missing_df):
    """Accurately calculates null percentages and differential signals."""
    res = compute_missingness_table(mock_missing_df, target_col="isFraud", key_col="TransactionID")

    # Excludes keys and target
    assert "TransactionID" not in res["column_name"].values
    assert "isFraud" not in res["column_name"].values
    assert len(res) == 3  # col_complete, col_low, col_signal

    # Check col_complete
    row_complete = res[res["column_name"] == "col_complete"].iloc[0]
    assert row_complete["total_nulls"] == 0
    assert row_complete["overall_null_pct"] == 0.0
    assert row_complete["missingness_tier"] == "Complete (0%)"

    # Check col_low (5 nulls / 100 rows = 5%)
    row_low = res[res["column_name"] == "col_low"].iloc[0]
    assert row_low["total_nulls"] == 5
    assert row_low["overall_null_pct"] == 5.0
    assert row_low["missingness_tier"] == "Low (<20%)"

    # Check col_signal (18 nulls: 8 in fraud, 10 in legit)
    row_signal = res[res["column_name"] == "col_signal"].iloc[0]
    assert row_signal["total_nulls"] == 18
    assert row_signal["overall_null_pct"] == 18.0
    assert row_signal["fraud_null_pct"] == pytest.approx(80.0)  # 8 / 10
    assert row_signal["legit_null_pct"] == pytest.approx(11.11, rel=1e-2)  # 10 / 90
    assert row_signal["diff_pct"] == pytest.approx(68.89, rel=1e-2)  # 80 - 11.11


def test_missingness_report_and_plot(mock_missing_df, tmp_path: Path):
    """Tests visualization and markdown/json/csv report generation."""
    missing_df = compute_missingness_table(mock_missing_df)

    fig_path = tmp_path / "figures" / "missingness.png"
    plot_missingness_analysis(missing_df, output_path=fig_path)
    assert fig_path.is_file()

    summary = generate_missingness_report(missing_df, total_rows=100, output_dir=tmp_path)
    assert (tmp_path / "missingness_analysis.md").is_file()
    assert (tmp_path / "missingness_analysis.json").is_file()
    assert (tmp_path / "missingness_ranking.csv").is_file()
    assert summary["complete_columns_count"] == 1
