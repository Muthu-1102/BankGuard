"""Unit tests for BankGuard evaluation metrics module (Phase 5 & 6)."""

from pathlib import Path
import numpy as np
import pytest

from bankguard.models.metrics import (
    compute_classification_metrics,
    format_metrics_markdown,
    save_metrics_report,
)


def test_compute_classification_metrics_perfect():
    """Validates metrics on perfect predictions."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.02, 0.95, 0.98])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["brier_score"] < 0.05

    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] == 2
    assert cm["true_negatives"] == 3
    assert cm["false_positives"] == 0
    assert cm["false_negatives"] == 0


def test_compute_classification_metrics_worst():
    """Validates metrics when predictions are completely inverted."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["false_positive_rate"] == pytest.approx(1.0)


def test_metrics_threshold_effect():
    """Verifies that changing decision threshold alters precision and recall trade-off."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.8])

    # Lenient threshold (0.2) -> high recall, lower precision
    low_thresh = compute_classification_metrics(y_true, y_prob, threshold=0.2)
    assert low_thresh["recall"] == 1.0
    assert low_thresh["precision"] < 1.0

    # Strict threshold (0.75) -> lower recall, higher precision
    high_thresh = compute_classification_metrics(y_true, y_prob, threshold=0.75)
    assert high_thresh["precision"] == 1.0
    assert high_thresh["recall"] < 1.0


def test_metrics_invalid_inputs():
    """Validates error raising on invalid input shapes or classes."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics([0, 1], [0.5])

    with pytest.raises(ValueError, match="empty arrays"):
        compute_classification_metrics([], [])

    with pytest.raises(ValueError, match="invalid classes"):
        compute_classification_metrics([0, 2], [0.5, 0.8])


def test_save_metrics_report(tmp_path: Path):
    """Tests saving metrics to both JSON and Markdown report files."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob)
    json_path, md_path = save_metrics_report(metrics, model_name="test_model", output_dir=tmp_path)

    assert json_path.is_file()
    assert md_path.is_file()
    assert (tmp_path / "test_model_metrics.json").exists()
    assert (tmp_path / "test_model_metrics.md").exists()
