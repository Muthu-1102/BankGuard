"""Evaluation metrics module for BankGuard (Phase 5 & 6).

Implements class-imbalance-aware evaluation metrics:
- Precision-Recall AUC (PR-AUC / Average Precision)
- ROC-AUC
- Brier score (probability calibration measure)
- Precision, Recall, F1 score at configurable decision thresholds
- Full confusion matrix (TP, FP, TN, FN)
- Naive majority-class benchmark comparison
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive class-imbalance-appropriate evaluation metrics.

    Args:
        y_true: Binary ground-truth labels (0 = Legitimate, 1 = Fraud).
        y_prob: Predicted fraud probabilities in [0.0, 1.0].
        threshold: Decision threshold for binarizing probabilities (default 0.5).

    Returns:
        Dictionary of precision, recall, PR-AUC, ROC-AUC, Brier score, and confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} elements, "
            f"y_prob has {len(y_prob_arr)} elements."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Validate binary target
    unique_labels = np.unique(y_true_arr)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Target contains invalid classes: {unique_labels}. Expected [0, 1].")

    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob_arr, 1e-15, 1.0 - 1e-15)

    # Threshold binarization
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Class distribution statistics
    total_samples = len(y_true_arr)
    positives = int(np.sum(y_true_arr == 1))
    negatives = int(np.sum(y_true_arr == 0))
    fraud_prevalence = (positives / total_samples) * 100.0
    naive_accuracy = max(positives, negatives) / total_samples * 100.0

    # Discriminative metrics
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        pr_auc = float(positives / total_samples)

    # Calibration & loss metrics
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    logloss = float(log_loss(y_true_arr, y_prob_clipped))

    # Point metrics at specified threshold
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # False positive rate & specificity
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "brier_score": round(brier, 5),
        "log_loss": round(logloss, 5),
        "decision_threshold": round(threshold, 4),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1_score": round(f1, 5),
        "specificity": round(specificity, 5),
        "false_positive_rate": round(fpr, 5),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "sample_counts": {
            "total": total_samples,
            "legitimate": negatives,
            "fraud": positives,
            "fraud_prevalence_pct": round(fraud_prevalence, 3),
            "naive_majority_accuracy_pct": round(naive_accuracy, 3),
        },
    }


def format_metrics_markdown(metrics: Dict[str, Any], model_title: str) -> str:
    """Formats metrics dictionary into a publication-grade markdown table."""
    cm = metrics["confusion_matrix"]
    sc = metrics["sample_counts"]
    thresh = metrics["decision_threshold"]

    return f"""### {model_title} — Evaluation Summary

- **Primary Fraud Metric (PR-AUC / Average Precision):** **{metrics['pr_auc']:.4f}** (Baseline random guess: {sc['fraud_prevalence_pct']/100:.4f})
- **ROC-AUC Score:** **{metrics['roc_auc']:.4f}**
- **Brier Calibration Score:** **{metrics['brier_score']:.4f}** (Lower is better; 0.0 = perfect)
- **Log Loss:** {metrics['log_loss']:.4f}

#### Classification Metrics (Threshold = {thresh})

| Metric | Score | Operational Context |
|---|---:|---|
| **Precision** | **{metrics['precision']*100:.2f}%** | Percentage of flagged transactions that were actually fraudulent |
| **Recall (Sensitivity)** | **{metrics['recall']*100:.2f}%** | Percentage of true fraudulent transactions successfully detected |
| **F1 Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of precision and recall |
| **Specificity** | **{metrics['specificity']*100:.2f}%** | Percentage of legitimate transactions correctly approved |
| **False Positive Rate** | **{metrics['false_positive_rate']*100:.2f}%** | Rate of benign transactions incorrectly challenged |

#### Confusion Matrix (Validation Set: {sc['total']:,} transactions)

| | Predicted Legitimate | Predicted Fraud | Total Actual |
|---|---:|---:|---:|
| **Actual Legitimate** | {cm['true_negatives']:,} (TN) | {cm['false_positives']:,} (FP) | {sc['legitimate']:,} |
| **Actual Fraud** | {cm['false_negatives']:,} (FN) | {cm['true_positives']:,} (TP) | {sc['fraud']:,} |
| **Total Predicted** | {cm['true_negatives'] + cm['false_negatives']:,} | {cm['false_positives'] + cm['true_positives']:,} | {sc['total']:,} |
"""


def save_metrics_report(
    metrics: Dict[str, Any],
    model_name: str,
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Saves metrics dictionary to both JSON and Markdown report files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{model_name}_metrics.json"
    md_path = out_dir / f"{model_name}_metrics.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_metrics_markdown(metrics, model_title=model_name.replace("_", " ").title()))

    return json_path, md_path


def compare_model_metrics(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    baseline_name: str = "Baseline (LogisticRegression)",
    candidate_name: str = "Improved (LightGBM)",
) -> Dict[str, Any]:
    """Compares baseline and candidate model metrics, computing absolute and relative differences.

    Args:
        baseline_metrics: Metrics dictionary from baseline model.
        candidate_metrics: Metrics dictionary from candidate model.
        baseline_name: Display label for baseline model.
        candidate_name: Display label for candidate model.

    Returns:
        Structured comparison dictionary.
    """
    comparisons = {}
    keys_to_compare = [
        ("pr_auc", "PR-AUC (Average Precision)", True),
        ("roc_auc", "ROC-AUC Score", True),
        ("brier_score", "Brier Score", False),
        ("log_loss", "Log Loss", False),
        ("precision", "Precision @ Threshold", True),
        ("recall", "Recall @ Threshold", True),
        ("f1_score", "F1 Score @ Threshold", True),
        ("specificity", "Specificity @ Threshold", True),
    ]

    for key, label, higher_is_better in keys_to_compare:
        base_val = baseline_metrics.get(key)
        cand_val = candidate_metrics.get(key)
        if base_val is not None and cand_val is not None:
            diff = float(cand_val - base_val)
            pct_change = (diff / float(base_val) * 100.0) if base_val != 0 else 0.0
            is_improvement = (diff > 0) if higher_is_better else (diff < 0)
            comparisons[key] = {
                "label": label,
                "baseline": round(float(base_val), 5),
                "candidate": round(float(cand_val), 5),
                "diff": round(diff, 5),
                "pct_change": round(pct_change, 2),
                "higher_is_better": higher_is_better,
                "is_improvement": is_improvement,
            }

    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "metrics": comparisons,
    }


def format_comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Formats comparison dictionary into a markdown comparison table."""
    base_name = comparison["baseline_name"]
    cand_name = comparison["candidate_name"]
    metrics_data = comparison["metrics"]

    rows = []
    for key, data in metrics_data.items():
        base_str = f"{data['baseline']:.4f}"
        cand_str = f"{data['candidate']:.4f}"
        diff_str = f"{data['diff']:+.4f}"
        pct_sign = "+" if data['pct_change'] > 0 else ""
        pct_str = f"{pct_sign}{data['pct_change']:.1f}%"
        status = "[+] Improved" if data['is_improvement'] else "[-] Regressed"
        rows.append(f"| **{data['label']}** | {base_str} | {cand_str} | {diff_str} | {pct_str} | {status} |")

    table_rows = "\n".join(rows)

    return f"""### Model Comparison: {base_name} vs {cand_name}

| Evaluation Metric | {base_name} | {cand_name} | Absolute Delta | Relative Change | Outcome |
|---|---:|---:|---:|---:|:---:|
{table_rows}
"""
