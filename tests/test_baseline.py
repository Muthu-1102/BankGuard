"""Unit tests for BankGuard Baseline Model (Task 5.1)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bankguard.models.baseline import (
    load_baseline_predictions,
    predict_baseline,
    run_baseline_pipeline,
    save_baseline_artifacts,
    train_baseline_model,
)


@pytest.fixture
def mock_pipeline_df():
    """Generates synthetic dataset for testing baseline pipeline."""
    n_rows = 120
    dts = [i * 3600 for i in range(n_rows)]
    amounts = [20.0 + (i % 30) for i in range(n_rows)]
    # Fraud distributed across timestamps (10% fraud in train and val)
    is_fraud = [1 if i % 10 == 0 else 0 for i in range(n_rows)]

    df = pd.DataFrame({
        "TransactionID": list(range(5000, 5000 + n_rows)),
        "isFraud": is_fraud,
        "TransactionDT": dts,
        "TransactionAmt": amounts,
        "ProductCD": ["W"] * 80 + ["C"] * 40,
        "card4": ["visa"] * 70 + ["mastercard"] * 50,
        "C1": [float(i % 10) for i in range(n_rows)],
    })
    return df


def test_train_and_predict_baseline():
    """Validates baseline model fitting and prediction generation."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    # Fraud correlates with feature 0
    y = (X[:, 0] > 1.0).astype(int)

    model = train_baseline_model(X, y, max_iter=200, random_state=42)
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 2

    y_pred, y_prob = predict_baseline(model, X, threshold=0.5)

    assert len(y_pred) == len(X)
    assert len(y_prob) == len(X)
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
    assert np.all(np.isin(y_pred, [0, 1]))


def test_save_and_load_baseline_predictions(tmp_path: Path):
    """Verifies artifact saving and loading roundtrip."""
    from sklearn.linear_model import LogisticRegression

    dummy_model = LogisticRegression().fit([[0], [1]], [0, 1])
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    y_pred = np.array([0, 1, 0, 1])
    val_ids = np.array([101, 102, 103, 104])

    saved = save_baseline_artifacts(
        model=dummy_model,
        y_val_true=y_true,
        y_val_prob=y_prob,
        y_val_pred=y_pred,
        val_ids=val_ids,
        artifact_dir=tmp_path,
    )

    assert Path(saved["model_path"]).is_file()
    assert Path(saved["predictions_path"]).is_file()

    # Load predictions back
    loaded = load_baseline_predictions(saved["predictions_path"])
    assert np.array_equal(loaded["y_true"], y_true)
    assert np.allclose(loaded["y_prob"], y_prob)
    assert np.array_equal(loaded["y_pred"], y_pred)
    assert np.array_equal(loaded["transaction_ids"], val_ids)


def test_run_baseline_pipeline_end_to_end(mock_pipeline_df, tmp_path: Path):
    """Tests end-to-end baseline training pipeline on mock data."""
    report_dir = tmp_path / "report"
    art_dir = tmp_path / "artifacts"

    results = run_baseline_pipeline(
        train_df=mock_pipeline_df,
        output_dir=report_dir,
        artifact_dir=art_dir,
        scale_numeric=True,
    )

    assert "metrics" in results
    assert "artifacts" in results
    assert "reports" in results

    # Verify metrics
    metrics = results["metrics"]
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics

    # Verify saved files
    assert (art_dir / "baseline_model.joblib").is_file()
    assert (art_dir / "baseline_predictions.npz").is_file()
    assert (report_dir / "baseline_metrics.json").is_file()
    assert (report_dir / "baseline_metrics.md").is_file()
