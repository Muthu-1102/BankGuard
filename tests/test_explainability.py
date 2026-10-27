"""Unit tests for BankGuard Model Explainability and Attribution (Task 5.4)."""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap

from bankguard.models.explainability import (
    EXPLANATION_DISCLAIMER,
    compute_global_feature_importance,
    compute_shap_values,
    explain_single_prediction,
    generate_dashboard_explanation_examples,
    get_tree_explainer,
    plot_global_shap_summary,
    plot_local_waterfall,
)


@pytest.fixture
def mock_tree_model_and_data():
    """Generates a small fitted LightGBM model and synthetic feature matrix."""
    np.random.seed(42)
    n = 100
    n_features = 6
    X = np.random.randn(n, n_features).astype(np.float32)
    # Fraud correlates with feature 0 and feature 1
    y = (X[:, 0] + 0.8 * X[:, 1] > 0.5).astype(int)

    feature_names = [f"feat_{i}" for i in range(n_features)]

    model = lgb.LGBMClassifier(n_estimators=10, num_leaves=8, random_state=42, verbose=-1)
    model.fit(X, y)

    return model, X, y, feature_names


def test_get_tree_explainer_and_compute_shap(mock_tree_model_and_data):
    """Validates TreeExplainer initialization and SHAP array dimensions."""
    model, X, _, _ = mock_tree_model_and_data

    explainer = get_tree_explainer(model)
    assert isinstance(explainer, shap.TreeExplainer)

    shap_vals = compute_shap_values(explainer, X)
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == X.shape
    assert not np.isnan(shap_vals).any()


def test_compute_global_feature_importance():
    """Verifies global feature importance calculation and descending sorting."""
    shap_vals = np.array([
        [0.5, -0.2, 0.1],
        [-0.3, 0.4, 0.0],
        [0.2, -0.1, 0.05],
    ])
    feature_names = ["feat_A", "feat_B", "feat_C"]

    df = compute_global_feature_importance(shap_vals, feature_names, top_n=3)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["feature", "mean_abs_shap", "relative_importance_pct"]
    assert df["mean_abs_shap"].is_monotonic_decreasing
    # Total percentage should sum to approximately 100%
    assert df["relative_importance_pct"].sum() == pytest.approx(100.0, rel=1e-2)


def test_compute_global_feature_importance_empty():
    """Verifies error handling on empty array."""
    with pytest.raises(ValueError, match="Cannot compute importance on empty SHAP array"):
        compute_global_feature_importance(np.empty((0, 5)), ["f1", "f2", "f3", "f4", "f5"])


def test_explain_single_prediction(mock_tree_model_and_data):
    """Verifies single transaction local explanation structure and disclaimer."""
    model, X, y, feature_names = mock_tree_model_and_data
    explainer = get_tree_explainer(model)

    res = explain_single_prediction(
        explainer=explainer,
        model=model,
        x_row=X[0],
        feature_names=feature_names,
        top_k=3,
        transaction_id=9999,
        true_label=int(y[0]),
        threshold=0.22,
    )

    assert res["transaction_id"] == 9999
    assert res["true_label"] == int(y[0])
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["risk_tier"] in ("ALLOW", "REVIEW", "BLOCK")
    assert isinstance(res["top_risk_drivers"], list)
    assert isinstance(res["top_protective_factors"], list)
    assert "disclaimer" in res
    assert "statistical associations" in res["disclaimer"]
    assert "causality" in res["disclaimer"]


def test_explain_single_prediction_dimension_mismatch(mock_tree_model_and_data):
    """Verifies ValueError when feature counts mismatch."""
    model, X, _, _ = mock_tree_model_and_data
    explainer = get_tree_explainer(model)

    with pytest.raises(ValueError, match="Feature count mismatch"):
        explain_single_prediction(
            explainer=explainer,
            model=model,
            x_row=X[0, :3],  # 3 features instead of 6
            feature_names=["f0", "f1", "f2", "f3", "f4", "f5"],
        )


def test_generate_dashboard_explanation_examples(mock_tree_model_and_data):
    """Verifies curation of representative transaction archetypes."""
    model, X, y, feature_names = mock_tree_model_and_data
    explainer = get_tree_explainer(model)
    val_ids = np.arange(1000, 1000 + len(y))

    examples = generate_dashboard_explanation_examples(
        explainer=explainer,
        model=model,
        X_val=X,
        y_val=y,
        val_ids=val_ids,
        feature_names=feature_names,
    )

    assert isinstance(examples, list)
    assert len(examples) > 0
    for ex in examples:
        assert "archetype" in ex
        assert "fraud_probability" in ex
        assert "risk_tier" in ex
        assert "top_risk_drivers" in ex
        assert "disclaimer" in ex


def test_plot_global_shap_and_waterfall(mock_tree_model_and_data, tmp_path: Path):
    """Verifies image artifact generation for summary and waterfall plots."""
    model, X, y, feature_names = mock_tree_model_and_data
    explainer = get_tree_explainer(model)
    shap_vals = compute_shap_values(explainer, X)

    # 1. Global summary plot
    global_file = tmp_path / "test_shap_summary.png"
    out_global = plot_global_shap_summary(shap_vals, X, feature_names, global_file)
    assert out_global.is_file()
    assert out_global.stat().st_size > 1000

    # 2. Local waterfall plot
    explanation = explain_single_prediction(
        explainer=explainer,
        model=model,
        x_row=X[0],
        feature_names=feature_names,
        transaction_id=1234,
        true_label=int(y[0]),
    )
    local_file = tmp_path / "test_local_waterfall.png"
    out_local = plot_local_waterfall(explanation, local_file)
    assert out_local.is_file()
    assert out_local.stat().st_size > 1000
