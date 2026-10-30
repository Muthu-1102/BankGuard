"""Automated WCAG 2.2 AA Contrast Compliance Tests for macOS Glass Theme."""

import pytest
from src.bankguard.dashboard import theme


def test_primary_text_on_app_background_contrast():
    """Rule C: Normal text must have at least 4.5:1 contrast against app background."""
    ratio = theme.calculate_contrast_ratio(theme.TEXT_PRIMARY, theme.BG_APP)
    assert ratio >= 4.5, f"Contrast ratio {ratio:.2f} is below WCAG AA minimum 4.5:1"
    assert ratio >= 10.0, "Expected very high contrast for primary text"


def test_primary_text_on_panel_contrast():
    """Rule C: Primary text must have at least 4.5:1 contrast against glass panel."""
    ratio = theme.calculate_contrast_ratio(theme.TEXT_PRIMARY, theme.BG_PANEL_SOLID)
    assert ratio >= 4.5, f"Contrast ratio {ratio:.2f} is below WCAG AA minimum 4.5:1"


def test_secondary_text_on_panel_contrast():
    """Rule C: Secondary text must have at least 4.5:1 contrast against glass panel."""
    ratio = theme.calculate_contrast_ratio(theme.TEXT_SECONDARY, theme.BG_PANEL_SOLID)
    assert ratio >= 4.5, f"Contrast ratio {ratio:.2f} is below WCAG AA minimum 4.5:1"


def test_muted_text_on_panel_contrast():
    """Rule C: Muted text must remain readable (at least 4.5:1 contrast against panel)."""
    ratio = theme.calculate_contrast_ratio(theme.TEXT_MUTED, theme.BG_PANEL_SOLID)
    assert ratio >= 4.5, f"Muted text contrast ratio {ratio:.2f} is below WCAG AA minimum 4.5:1"


def test_input_text_on_input_background_contrast():
    """Rule C: Input text on input background must have at least 4.5:1 contrast."""
    ratio = theme.calculate_contrast_ratio(theme.TEXT_PRIMARY, theme.BG_INPUT)
    assert ratio >= 4.5, f"Input contrast ratio {ratio:.2f} is below WCAG AA minimum 4.5:1"


def test_semantic_risk_status_contrast():
    """Rule C: Semantic risk colors must be clearly legible against dark surfaces."""
    # Test each risk color against panel surface
    ratio_low = theme.calculate_contrast_ratio(theme.RISK_LOW, theme.BG_PANEL_SOLID)
    ratio_review = theme.calculate_contrast_ratio(theme.RISK_REVIEW, theme.BG_PANEL_SOLID)
    ratio_high = theme.calculate_contrast_ratio(theme.RISK_HIGH, theme.BG_PANEL_SOLID)

    assert ratio_low >= 4.5, f"Risk low contrast {ratio_low:.2f} < 4.5"
    assert ratio_review >= 4.5, f"Risk review contrast {ratio_review:.2f} < 4.5"
    assert ratio_high >= 4.5, f"Risk high contrast {ratio_high:.2f} < 4.5"


def test_decision_chip_rendering():
    """Rule A & C: Decision chips must render with deliberate icon, word, and color."""
    allow_html = theme.render_decision_chip("allow")
    assert "● ALLOW" in allow_html
    assert "chip-allow" in allow_html

    review_html = theme.render_decision_chip("review")
    assert "▲ REVIEW" in review_html
    assert "chip-review" in review_html

    block_html = theme.render_decision_chip("high_risk")
    assert "⬢ HIGH RISK" in block_html
    assert "chip-block" in block_html

