"""Unit tests for BankGuard Threshold Evaluation and Tuning (Task 5.3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.models.threshold_tuning import (
    evaluate_threshold_grid,
    find_optimal_thresholds,
    plot_threshold_curves,
    run_threshold_evaluation_pipeline,
)


@pytest.fixture
def mock_predictions_data():
    """Generates synthetic ground truth and predicted probabilities."""
    np.random.seed(42)
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[np.random.choice(n, size=20, replace=False)] = 1  # 10% fraud

    # Higher probabilities for true positives
    y_prob = np.random.uniform(0.01, 0.3, size=n)
    y_prob[y_true == 1] = np.random.uniform(0.4, 0.95, size=20)

    return y_true, y_prob


def test_evaluate_threshold_grid_validation():
    """Verifies input validation on empty and mismatched arrays."""
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_threshold_grid(np.array([0, 1]), np.array([0.5]))

    with pytest.raises(ValueError, match="Cannot evaluate thresholds on empty arrays"):
        evaluate_threshold_grid(np.array([]), np.array([]))


def test_evaluate_threshold_grid_metrics(mock_predictions_data):
    """Verifies calculated threshold metrics are mathematically sound."""
    y_true, y_prob = mock_predictions_data
    thresholds = [0.1, 0.3, 0.5, 0.8]

    df = evaluate_threshold_grid(y_true, y_prob, thresholds=thresholds)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    expected_cols = [
        "threshold", "flagged_count", "flagged_rate_pct", "tp", "fp", "fn", "tn",
        "precision", "recall", "f1_score", "specificity", "false_positive_rate", "total_cost_usd"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check bounds
    assert (df["precision"] >= 0.0).all() and (df["precision"] <= 1.0).all()
    assert (df["recall"] >= 0.0).all() and (df["recall"] <= 1.0).all()
    assert (df["f1_score"] >= 0.0).all() and (df["f1_score"] <= 1.0).all()
    assert (df["specificity"] >= 0.0).all() and (df["specificity"] <= 1.0).all()

    # Higher threshold generally results in lower or equal recall
    assert df["recall"].is_monotonic_decreasing or df["recall"].iloc[-1] <= df["recall"].iloc[0]


def test_find_optimal_thresholds(mock_predictions_data):
    """Verifies threshold optimization across multiple operational criteria."""
    y_true, y_prob = mock_predictions_data
    df = evaluate_threshold_grid(y_true, y_prob)

    opts = find_optimal_thresholds(df, max_capacity_pct=15.0, target_recall=0.70)

    assert "max_f1" in opts
    assert "capacity_constrained" in opts
    assert "target_recall" in opts
    assert "min_cost" in opts

    # Validate max F1
    max_f1_score = df["f1_score"].max()
    assert opts["max_f1"]["metrics"]["f1_score"] == pytest.approx(max_f1_score)

    # Validate capacity constraint
    cap_pct = opts["capacity_constrained"]["metrics"]["flagged_rate_pct"]
    assert cap_pct <= 15.0 or cap_pct == df["flagged_rate_pct"].min()


def test_plot_threshold_curves(mock_predictions_data, tmp_path: Path):
    """Verifies threshold curve visualization generates without error."""
    y_true, y_prob = mock_predictions_data
    df = evaluate_threshold_grid(y_true, y_prob)
    opts = find_optimal_thresholds(df, max_capacity_pct=15.0, target_recall=0.70)

    out_file = tmp_path / "test_curves.png"
    result = plot_threshold_curves(
        y_true=y_true,
        y_prob_improved=y_prob,
        y_prob_baseline=y_prob * 0.8,
        threshold_df_improved=df,
        optimal_thresholds=opts,
        output_path=out_file,
    )

    assert result.is_file()
    assert result.stat().st_size > 1000


def test_run_threshold_evaluation_pipeline(tmp_path: Path, mock_predictions_data):
    """Verifies end-to-end execution of the threshold evaluation pipeline."""
    y_true, y_prob = mock_predictions_data
    pred_path = tmp_path / "improved_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=(y_prob >= 0.5).astype(int),
        transaction_ids=np.arange(len(y_true)),
    )

    report_dir = tmp_path / "report"
    art_dir = tmp_path / "artifacts"

    results = run_threshold_evaluation_pipeline(
        improved_predictions_path=pred_path,
        baseline_predictions_path=None,
        output_dir=report_dir,
        artifact_dir=art_dir,
        max_capacity_pct=20.0,
    )

    assert "recommended_threshold" in results
    assert "optimal_thresholds" in results
    assert "reports" in results

    assert (report_dir / "threshold_evaluation.md").is_file()
    assert (report_dir / "threshold_evaluation.json").is_file()
    assert (report_dir / "threshold_tradeoffs.png").is_file()
    assert (art_dir / "threshold_tradeoffs.png").is_file()
