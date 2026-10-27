"""Unit tests for BankGuard Improved Model (Task 5.2 - LightGBM)."""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from bankguard.models.improved import (
    compute_feature_importances,
    get_default_lgbm_params,
    load_improved_predictions,
    predict_improved,
    run_improved_pipeline,
    save_improved_artifacts,
    train_improved_model,
)
from bankguard.models.metrics import compare_model_metrics, format_comparison_markdown


@pytest.fixture
def mock_pipeline_df():
    """Generates synthetic dataset for testing improved pipeline."""
    n_rows = 120
    dts = [i * 3600 for i in range(n_rows)]
    amounts = [20.0 + (i % 30) for i in range(n_rows)]
    is_fraud = [1 if i % 10 == 0 else 0 for i in range(n_rows)]

    df = pd.DataFrame({
        "TransactionID": list(range(6000, 6000 + n_rows)),
        "isFraud": is_fraud,
        "TransactionDT": dts,
        "TransactionAmt": amounts,
        "ProductCD": ["W"] * 80 + ["C"] * 40,
        "card4": ["visa"] * 70 + ["mastercard"] * 50,
        "C1": [float(i % 10) for i in range(n_rows)],
    })
    return df


def test_get_default_lgbm_params():
    """Validates default LightGBM hyperparameters."""
    params = get_default_lgbm_params(random_state=123)
    assert params["objective"] == "binary"
    assert params["random_state"] == 123
    assert params["n_estimators"] == 300
    assert params["learning_rate"] == 0.05
    assert params["importance_type"] == "gain"


def test_train_and_predict_improved():
    """Validates LightGBM model fitting and probability generation."""
    np.random.seed(42)
    X = np.random.randn(150, 8).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)

    model = train_improved_model(
        X, y,
        params={"n_estimators": 20, "num_leaves": 15, "min_child_samples": 5},
        random_state=42,
    )
    assert isinstance(model, lgb.LGBMClassifier)
    assert hasattr(model, "classes_")

    y_pred, y_prob = predict_improved(model, X, threshold=0.5)

    assert len(y_pred) == len(X)
    assert len(y_prob) == len(X)
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
    assert np.all(np.isin(y_pred, [0, 1]))


def test_compute_feature_importances():
    """Verifies feature importance calculation and ordering."""
    np.random.seed(42)
    X = np.random.randn(150, 5).astype(np.float32)
    y = (X[:, 0] > 0.2).astype(int)
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]

    model = train_improved_model(
        X, y,
        params={"n_estimators": 20, "num_leaves": 10, "min_child_samples": 5},
        random_state=42,
    )

    df_imp = compute_feature_importances(model, feature_names=feature_names, top_n=5)

    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5
    assert "gain_importance" in df_imp.columns
    assert "split_importance" in df_imp.columns
    assert "relative_importance_pct" in df_imp.columns
    # Gain should be sorted descending
    assert df_imp["gain_importance"].is_monotonic_decreasing


def test_save_and_load_improved_predictions(tmp_path: Path):
    """Verifies artifact serialization and loading roundtrip."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.array([0, 1] * 25)
    model = lgb.LGBMClassifier(n_estimators=5, random_state=42, verbose=-1).fit(X, y)

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.12, 0.88], dtype=np.float32)
    y_pred = np.array([0, 1, 0, 1])
    val_ids = np.array([201, 202, 203, 204])

    df_imp = pd.DataFrame({
        "feature": ["f1", "f2"],
        "gain_importance": [10.5, 3.2],
        "split_importance": [5, 2],
        "relative_importance_pct": [76.6, 23.4],
    })

    saved = save_improved_artifacts(
        model=model,
        y_val_true=y_true,
        y_val_prob=y_prob,
        y_val_pred=y_pred,
        val_ids=val_ids,
        artifact_dir=tmp_path,
        feature_importances=df_imp,
    )

    assert Path(saved["model_path"]).is_file()
    assert Path(saved["predictions_path"]).is_file()
    assert Path(saved["feature_importances_path"]).is_file()

    loaded = load_improved_predictions(saved["predictions_path"])
    assert np.array_equal(loaded["y_true"], y_true)
    assert np.allclose(loaded["y_prob"], y_prob)
    assert np.array_equal(loaded["y_pred"], y_pred)
    assert np.array_equal(loaded["transaction_ids"], val_ids)


def test_compare_model_metrics():
    """Verifies baseline vs candidate comparison calculations."""
    base_metrics = {
        "pr_auc": 0.1420,
        "roc_auc": 0.7844,
        "brier_score": 0.1716,
        "log_loss": 0.6191,
        "precision": 0.1203,
        "recall": 0.6228,
        "f1_score": 0.2017,
        "specificity": 0.8377,
    }
    cand_metrics = {
        "pr_auc": 0.6550,
        "roc_auc": 0.9250,
        "brier_score": 0.0245,
        "log_loss": 0.0820,
        "precision": 0.8100,
        "recall": 0.5500,
        "f1_score": 0.6550,
        "specificity": 0.9950,
    }

    comp = compare_model_metrics(base_metrics, cand_metrics)
    assert "metrics" in comp
    assert comp["metrics"]["pr_auc"]["is_improvement"] is True
    assert comp["metrics"]["pr_auc"]["diff"] > 0
    assert comp["metrics"]["brier_score"]["is_improvement"] is True  # lower is better!
    assert comp["metrics"]["brier_score"]["diff"] < 0

    md = format_comparison_markdown(comp)
    assert "PR-AUC (Average Precision)" in md
    assert "[+] Improved" in md


def test_run_improved_pipeline_end_to_end(mock_pipeline_df, tmp_path: Path):
    """Tests end-to-end improved training pipeline on mock data."""
    report_dir = tmp_path / "report"
    art_dir = tmp_path / "artifacts"

    # Pre-populate a fake baseline_metrics.json to test comparison generation
    report_dir.mkdir(parents=True, exist_ok=True)
    fake_baseline = {
        "pr_auc": 0.10,
        "roc_auc": 0.70,
        "brier_score": 0.20,
        "log_loss": 0.65,
        "decision_threshold": 0.5,
        "precision": 0.10,
        "recall": 0.60,
        "f1_score": 0.17,
        "specificity": 0.80,
    }
    import json
    with open(report_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fake_baseline, f)

    results = run_improved_pipeline(
        train_df=mock_pipeline_df,
        output_dir=report_dir,
        artifact_dir=art_dir,
        model_params={"n_estimators": 10, "num_leaves": 8, "min_child_samples": 2},
        scale_numeric=False,
    )

    assert "metrics" in results
    assert "comparison" in results
    assert "artifacts" in results
    assert "reports" in results

    assert (art_dir / "improved_model.joblib").is_file()
    assert (art_dir / "improved_predictions.npz").is_file()
    assert (art_dir / "improved_feature_importances.csv").is_file()
    assert (report_dir / "improved_metrics.json").is_file()
    assert (report_dir / "improved_metrics.md").is_file()
    assert (report_dir / "model_comparison.md").is_file()
    assert (report_dir / "model_comparison.json").is_file()
