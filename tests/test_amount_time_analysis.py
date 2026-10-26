"""Unit tests for BankGuard amount and time analysis module (Task 3.3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.data.amount_time_analysis import (
    compute_amount_statistics,
    compute_categorical_patterns,
    compute_time_statistics,
    generate_amount_time_report,
    plot_amount_time_analysis,
)


@pytest.fixture
def mock_tx_df():
    """Generates a small controlled DataFrame mimicking transaction features."""
    # 100 rows: 10 fraud (isFraud=1), 90 legit (isFraud=0)
    # TransactionDT runs from 0 to 99 * 3600 (356,400 seconds = ~4.125 days)
    # Amount varies with known integer and decimal values
    amounts = [50.0] * 50 + [100.50] * 40 + [500.0] * 10
    # Let fraud have some of the decimal amounts and high amounts
    is_fraud = [0] * 45 + [1] * 5 + [0] * 35 + [1] * 5 + [0] * 10
    product_cd = ["W"] * 60 + ["C"] * 30 + ["H"] * 10
    card4 = ["visa"] * 50 + ["mastercard"] * 40 + ["discover"] * 10
    card6 = ["debit"] * 70 + ["credit"] * 30

    data = {
        "TransactionID": list(range(100)),
        "isFraud": is_fraud,
        "TransactionAmt": amounts,
        "TransactionDT": [i * 3600 for i in range(100)],
        "ProductCD": product_cd,
        "card4": card4,
        "card6": card6,
    }
    return pd.DataFrame(data)


def test_compute_amount_statistics(mock_tx_df):
    """Validates calculation of amount statistics and percentiles."""
    stats = compute_amount_statistics(mock_tx_df, amt_col="TransactionAmt", target_col="isFraud")

    ov = stats["overall"]
    assert ov["count"] == 100
    assert ov["min"] == 50.0
    assert ov["max"] == 500.0
    assert ov["median"] == 75.25
    assert ov["iqr"] > 0

    assert "legitimate" in stats
    assert "fraud" in stats
    assert stats["legitimate"]["count"] == 90
    assert stats["fraud"]["count"] == 10

    dec = stats["decimal_cents_analysis"]
    assert dec["fractional_amount_count"] == 40
    assert dec["fractional_amount_pct"] == 40.0
    assert dec["round_amount_count"] == 60


def test_compute_amount_statistics_missing_col():
    """Raises KeyError when amount column is missing."""
    df = pd.DataFrame({"colA": [1, 2, 3]})
    with pytest.raises(KeyError):
        compute_amount_statistics(df, amt_col="TransactionAmt")


def test_compute_time_statistics(mock_tx_df):
    """Validates temporal metrics, monotonicity, and cyclical proxies."""
    time_stats = compute_time_statistics(mock_tx_df, dt_col="TransactionDT", target_col="isFraud")

    assert time_stats["min_timestamp_sec"] == 0
    assert time_stats["max_timestamp_sec"] == 99 * 3600
    assert time_stats["time_span_seconds"] == 99 * 3600
    assert time_stats["is_monotonic_increasing"] is True
    assert time_stats["time_span_days"] == pytest.approx(4.125, rel=1e-2)

    # 24 hourly buckets
    assert len(time_stats["hourly_breakdown"]) == 24

    # 7 daily buckets
    assert len(time_stats["daily_breakdown"]) == 7


def test_compute_categorical_patterns(mock_tx_df):
    """Validates calculation of category volumes and fraud rates."""
    cat_stats = compute_categorical_patterns(mock_tx_df, columns=["ProductCD", "card4", "card6"], target_col="isFraud")

    assert "ProductCD" in cat_stats
    prod_c = [p for p in cat_stats["ProductCD"] if p["category"] == "C"][0]
    assert prod_c["count"] == 30
    assert prod_c["pct_volume"] == 30.0

    assert "card4" in cat_stats
    assert "card6" in cat_stats


def test_amount_time_reports_and_plots(mock_tx_df, tmp_path: Path):
    """Tests visualization rendering and report export to disk."""
    amt_stats = compute_amount_statistics(mock_tx_df)
    time_stats = compute_time_statistics(mock_tx_df)
    cat_stats = compute_categorical_patterns(mock_tx_df)

    fig_path = tmp_path / "figures" / "amount_time.png"
    plot_amount_time_analysis(mock_tx_df, amt_stats, time_stats, cat_stats, output_path=fig_path)
    assert fig_path.is_file()

    summary = generate_amount_time_report(amt_stats, time_stats, cat_stats, output_dir=tmp_path)
    assert (tmp_path / "amount_time_analysis.json").is_file()
    assert (tmp_path / "amount_time_analysis.md").is_file()
    assert "amount_statistics" in summary
