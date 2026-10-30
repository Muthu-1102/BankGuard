"""Unit tests for BankGuard macOS Glass Dashboard modules."""

import pytest
import pandas as pd
from src.bankguard.dashboard import data_loader, theme
from src.bankguard.dashboard.pages import (
    page_data_quality,
    page_explainability,
    page_explorer,
    page_model_performance,
    page_overview,
    page_risk_analysis,
    page_system_health,
)


def test_dashboard_modules_importable():
    """Verifies that all page modules and theme helpers import cleanly."""
    assert page_overview.render is not None
    assert page_explorer.render is not None
    assert page_risk_analysis.render is not None
    assert page_model_performance.render is not None
    assert page_explainability.render is not None
    assert page_data_quality.render is not None
    assert page_system_health.render is not None


def test_data_loader_artifacts():
    """Verifies that saved evaluation artifacts load correctly from disk."""
    comp = data_loader.get_model_comparison()
    assert comp is not None
    assert "metrics" in comp
    assert comp["metrics"]["pr_auc"]["candidate"] == 0.5598

    thresh = data_loader.get_threshold_evaluation()
    assert thresh is not None
    assert thresh["recommended_threshold"] == 0.22

    policy = data_loader.get_decision_policy_report()
    assert policy is not None
    assert policy["policy_name"] == "bankguard_three_tier_policy"

    dq = data_loader.get_data_quality_report()
    assert dq is not None
    assert dq["total_checks"] == 8
    assert dq["passed_count"] == 8


def test_load_validation_predictions():
    """Verifies that actual validation predictions load correctly without fabrication."""
    df = data_loader.load_validation_predictions(max_rows=100)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "TransactionID" in df.columns
    assert "Fraud Score" in df.columns
    assert "Risk Tier" in df.columns
    assert "Actual Label" in df.columns
    assert set(df["Risk Tier"].unique()).issubset({"ALLOW", "REVIEW", "HIGH RISK"})


def test_plotly_theme_layout():
    """Verifies that Plotly layout uses the centralized theme palette."""
    layout = theme.get_plotly_layout(height=350, title="Test Performance")
    assert layout["height"] == 350
    assert layout["font"]["color"] == theme.TEXT_SECONDARY
    assert layout["paper_bgcolor"] == theme.BG_PANEL_SOLID
    assert "Inter" in layout["font"]["family"]

