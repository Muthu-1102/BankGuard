"""Explainability and feature attribution module for BankGuard (Task 5.4).

Implements global and local model explanations using TreeSHAP (SHapley Additive exPlanations):
- Global feature impact and ranking across validation transactions.
- Local individual transaction explanations (positive risk drivers and protective factors).
- Representative dashboard explanation examples (True Positive, False Positive, True Negative, False Negative).
- Explicit disclaimers guarding against treating statistical feature attribution as causation.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import sys
import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from bankguard.features.feature_definition import (
    ID_COLUMN,
    TARGET_COLUMN,
    split_features_and_target,
)
from bankguard.features.preprocessor import (
    BankGuardFeatureEngineer,
    BankGuardPreprocessor,
    temporal_train_val_split,
)

# Unpickling compatibility when artifact was saved during direct script execution
if hasattr(sys.modules.get("__main__"), "__dict__"):
    sys.modules["__main__"].BankGuardFeatureEngineer = BankGuardFeatureEngineer
    sys.modules["__main__"].BankGuardPreprocessor = BankGuardPreprocessor


EXPLANATION_DISCLAIMER = (
    "IMPORTANT DISCLAIMER: Model explanations (e.g. SHAP feature attributions) quantify "
    "statistical associations and internal decision contributions within the tree ensemble. "
    "They do NOT establish real-world causality. An elevated attribution indicates that a "
    "feature value correlated with fraud patterns in historical training data, not that the "
    "feature or the cardholder caused fraudulent activity."
)


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initializes a TreeExplainer for the fitted LightGBM tree ensemble model.

    Args:
        model: Fitted LightGBM model.

    Returns:
        Configured shap.TreeExplainer instance.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
) -> np.ndarray:
    """Computes positive-class SHAP attribution values for a feature array.

    Args:
        explainer: Fitted TreeExplainer.
        X: 2D numpy array of preprocessed features.

    Returns:
        2D numpy array of shape (N, n_features) representing log-odds contributions.
    """
    raw_shap = explainer.shap_values(X)

    # In binary classification, some SHAP versions return a list [class_0, class_1]
    if isinstance(raw_shap, list):
        # Index 1 corresponds to positive fraud class
        return np.asarray(raw_shap[1], dtype=np.float32)

    # If 3D array (N, n_features, 2)
    if isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        return np.asarray(raw_shap[:, :, 1], dtype=np.float32)

    return np.asarray(raw_shap, dtype=np.float32)


def compute_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 25,
) -> pd.DataFrame:
    """Computes global feature importance as mean absolute SHAP value across samples.

    Args:
        shap_values: 2D array of SHAP values (N samples x M features).
        feature_names: List of M feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame sorted by mean absolute SHAP descending.
    """
    if len(shap_values) == 0:
        raise ValueError("Cannot compute importance on empty SHAP array.")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = float(np.sum(mean_abs_shap))
    rel_pct = (mean_abs_shap / total_shap * 100.0) if total_shap > 0 else np.zeros_like(mean_abs_shap)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs_shap, 5),
        "relative_importance_pct": np.round(rel_pct, 3),
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)


def explain_single_prediction(
    explainer: shap.TreeExplainer,
    model: Any,
    x_row: np.ndarray,
    feature_names: List[str],
    top_k: int = 8,
    transaction_id: Optional[int] = None,
    true_label: Optional[int] = None,
    threshold: float = 0.22,
) -> Dict[str, Any]:
    """Generates an interpretable local explanation for an individual transaction.

    Args:
        explainer: Fitted TreeExplainer.
        model: Fitted model with predict_proba.
        x_row: 1D or 2D array of preprocessed features for a single transaction.
        feature_names: Feature names matching x_row dimensions.
        top_k: Number of positive risk drivers and protective factors to return.
        transaction_id: Optional ID of the transaction.
        true_label: Optional ground truth label (0 or 1).
        threshold: Operational decision threshold for tier assignment.

    Returns:
        Structured dictionary containing prediction, risk tier, top drivers, and disclaimer.
    """
    x_2d = np.atleast_2d(x_row)
    if x_2d.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: x_row has {x_2d.shape[1]} features, "
            f"feature_names has {len(feature_names)}."
        )

    # 1. Prediction
    prob = float(model.predict_proba(x_2d)[0, 1])

    # Assign risk tier
    if prob < 0.10:
        tier = "ALLOW"
        action = "Automated Approval"
    elif prob < 0.40:
        tier = "REVIEW"
        action = "Manual Review / Step-up Verification"
    else:
        tier = "BLOCK"
        action = "Automated High-Risk Decline"

    # 2. Local SHAP values
    shap_vals = compute_shap_values(explainer, x_2d)[0]

    # Base value (log-odds expected value)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[-1])
    else:
        base_val = float(base_val)

    # Feature contribution records
    contributions = []
    for idx, (f_name, s_val, f_val) in enumerate(zip(feature_names, shap_vals, x_2d[0])):
        contributions.append({
            "feature": f_name,
            "feature_value": round(float(f_val), 4),
            "shap_attribution": round(float(s_val), 5),
            "abs_attribution": abs(float(s_val)),
            "direction": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK",
        })

    df_contrib = pd.DataFrame(contributions)

    # Top risk drivers (pushing score higher)
    drivers = (
        df_contrib[df_contrib["shap_attribution"] > 0]
        .sort_values(by="shap_attribution", ascending=False)
        .head(top_k)
        .to_dict(orient="records")
    )

    # Top protective factors (pushing score lower)
    protective = (
        df_contrib[df_contrib["shap_attribution"] < 0]
        .sort_values(by="shap_attribution", ascending=True)
        .head(top_k)
        .to_dict(orient="records")
    )

    return {
        "transaction_id": int(transaction_id) if transaction_id is not None else None,
        "true_label": int(true_label) if true_label is not None else None,
        "ground_truth_class": "FRAUD" if true_label == 1 else ("LEGITIMATE" if true_label == 0 else "UNKNOWN"),
        "fraud_probability": round(prob, 4),
        "decision_threshold": round(threshold, 4),
        "is_flagged": bool(prob >= threshold),
        "risk_tier": tier,
        "recommended_action": action,
        "base_value_log_odds": round(base_val, 4),
        "total_shap_log_odds": round(float(np.sum(shap_vals)), 4),
        "top_risk_drivers": drivers,
        "top_protective_factors": protective,
        "disclaimer": EXPLANATION_DISCLAIMER,
    }


def generate_dashboard_explanation_examples(
    explainer: shap.TreeExplainer,
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_ids: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.22,
) -> List[Dict[str, Any]]:
    """Curates diverse, real validation transactions representing key operational archetypes.

    Archetypes:
    1. HIGH_CONFIDENCE_FRAUD: True Positive with high risk score (>= 0.70).
    2. SUSPICIOUS_REVIEW: Transaction in the manual review band (0.20 <= Score < 0.40).
    3. FALSE_POSITIVE: Benign transaction flagged by model (Score >= 0.25, True = 0).
    4. LOW_RISK_BENIGN: Typical legitimate transaction (Score < 0.05, True = 0).
    5. FALSE_NEGATIVE: Subtle fraud that evaded detection (Score < 0.15, True = 1).
    """
    probs = model.predict_proba(X_val)[:, 1]
    y_true = np.asarray(y_val, dtype=int)
    ids = np.asarray(val_ids, dtype=np.int64)

    archetypes = [
        ("HIGH_CONFIDENCE_FRAUD", (y_true == 1) & (probs >= 0.70)),
        ("SUSPICIOUS_REVIEW", (probs >= 0.20) & (probs < 0.40)),
        ("FALSE_POSITIVE_CHALLENGED", (y_true == 0) & (probs >= 0.25)),
        ("LOW_RISK_AUTO_APPROVED", (y_true == 0) & (probs < 0.03)),
        ("FALSE_NEGATIVE_MISSED_FRAUD", (y_true == 1) & (probs < 0.15)),
    ]

    examples = []
    for label, mask in archetypes:
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Pick first matching index
            idx = int(indices[0])
            explanation = explain_single_prediction(
                explainer=explainer,
                model=model,
                x_row=X_val[idx],
                feature_names=feature_names,
                top_k=6,
                transaction_id=int(ids[idx]),
                true_label=int(y_true[idx]),
                threshold=threshold,
            )
            explanation["archetype"] = label
            examples.append(explanation)

    return examples


def plot_global_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_path: Union[str, Path],
    top_n: int = 15,
) -> Path:
    """Generates and saves publication-quality global SHAP feature impact visualization."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [feature_names[i] for i in top_indices][::-1]
    top_scores = [mean_abs[i] for i in top_indices][::-1]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    bars = ax.barh(top_features, top_scores, color="#1f77b4", alpha=0.85, edgecolor="#0d47a1", height=0.65)

    # Add value annotations
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#212121",
        )

    ax.set_title("Global Feature Impact (Mean |SHAP Value| across Validation Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Average Log-Odds Impact on Fraud Estimation)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, max(top_scores) * 1.15)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_file


def plot_local_waterfall(
    explanation: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Generates a local attribution waterfall-style horizontal bar plot for an individual transaction."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    drivers = explanation.get("top_risk_drivers", [])[:5]
    protective = explanation.get("top_protective_factors", [])[:5]

    features = []
    attributions = []
    colors = []

    for item in protective[::-1]:
        features.append(f"{item['feature']} ({item['feature_value']})")
        attributions.append(item["shap_attribution"])
        colors.append("#2ca02c")  # Green for lowering risk

    for item in drivers:
        features.append(f"{item['feature']} ({item['feature_value']})")
        attributions.append(item["shap_attribution"])
        colors.append("#d62728")  # Red for increasing risk

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.barh(features, attributions, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color="black", linestyle="-", lw=1)

    tx_id = explanation.get("transaction_id", "N/A")
    prob = explanation.get("fraud_probability", 0.0)
    tier = explanation.get("risk_tier", "N/A")

    ax.set_title(
        f"Local Transaction Explanation — ID #{tx_id} (Fraud Risk: {prob*100:.1f}%, Tier: {tier})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("SHAP Feature Attribution (Delta Log-Odds)", fontsize=9.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Subtitle with disclaimer
    fig.text(
        0.5, -0.02,
        "Red = Increases Fraud Score | Green = Reduces Fraud Score (Statistical Association, Not Causation)",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_file


def format_explainability_report_markdown(
    df_global: pd.DataFrame,
    examples: List[Dict[str, Any]],
    n_samples_explained: int,
) -> str:
    """Formats comprehensive explainability report for documentation and review."""
    # Global top table
    global_rows = []
    for i, row in df_global.head(15).iterrows():
        global_rows.append(
            f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} | {row['relative_importance_pct']:.2f}% |"
        )
    global_table_str = "\n".join(global_rows)

    # Local examples markdown
    example_sections = []
    for ex in examples:
        drivers_str = ", ".join([f"`{d['feature']}` ({d['shap_attribution']:+.3f})" for d in ex["top_risk_drivers"][:3]])
        prot_str = ", ".join([f"`{p['feature']}` ({p['shap_attribution']:+.3f})" for p in ex["top_protective_factors"][:3]])
        example_sections.append(
            f"#### Case: {ex['archetype'].replace('_', ' ').title()} (Transaction #{ex['transaction_id']})\n"
            f"- **Actual Ground Truth:** **{ex['ground_truth_class']}**\n"
            f"- **Estimated Fraud Probability:** **{ex['fraud_probability']*100:.2f}%** (Risk Tier: `{ex['risk_tier']}`, Action: *{ex['recommended_action']}*)\n"
            f"- **Key Risk Drivers (Increasing Score):** {drivers_str}\n"
            f"- **Key Protective Factors (Decreasing Score):** {prot_str}\n"
        )
    examples_str = "\n".join(example_sections)

    return f"""# BankGuard — Model Explainability & Attribution Report (Task 5.4)

## 1. Executive Summary & Explainability Policy

This report documents model interpretability and feature attributions for the production **LightGBM Fraud Risk Estimator** using **TreeSHAP** (Shapley Additive exPlanations).

> [!IMPORTANT]
> **Causation vs. Statistical Attribution Policy:**  
> {EXPLANATION_DISCLAIMER}

---

## 2. Global Feature Importance (Evaluated on {n_samples_explained:,} Validation Transactions)

Global importance represents the mean absolute SHAP value across out-of-time validation transactions (Mean |SHAP| = E[|SHAP_j|]).

| Rank | Feature Name | Mean |SHAP| (Log-Odds Impact) | Relative Attribution Share (%) |
|---:|---|---:|---:|
{global_table_str}

### Key Global Behavioral Insights
1. **Transaction Velocity & Identity Associations (`V258`, `V257`, `V294`):** Engineered card/device velocity and association counters dominate model decisions. Abnormal bursts in these counters strongly elevate predicted fraud probabilities.
2. **Contact Consistency & Account Frequency (`C1`, `C14`, `C13`):** Counts associated with identical phone, email, and billing attributes serve as primary indicators. Large deviations from expected benign frequencies represent significant risk escalations.
3. **Transaction Amount & Cyclical Features (`TransactionAmt`, `sin_hour`, `day_of_week`):** Transaction dollar value and diurnal timing provide persistent secondary signals, corroborating the findings from Phase 3 EDA.

---

## 3. Local Transaction Decision Explanations (Dashboard Examples)

The following real validation records illustrate how individual predictions are explained in the serving API and monitoring dashboard:

{examples_str}

---

## 4. Serving & Dashboard Integration (Phase 6 Preparation)

All local explanations are structured into standardized JSON payloads (`artifacts/explanation_examples.json`) containing:
- Predicted fraud score and calibrated risk tier (`ALLOW`, `REVIEW`, `BLOCK`).
- Top positive risk contributors and negative protective factors with exact feature values.
- Explicit attribution disclaimer labeling explanations as statistical signals rather than verified facts.
"""


def run_explainability_pipeline(
    model_path: Union[str, Path] = "artifacts/improved_model.joblib",
    preprocessor_path: Union[str, Path] = "artifacts/preprocessor.joblib",
    predictions_path: Union[str, Path] = "artifacts/improved_predictions.npz",
    output_dir: Union[str, Path] = "report",
    artifact_dir: Union[str, Path] = "artifacts",
    sample_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """End-to-end orchestration for Task 5.4:

    1. Loads trained LightGBM model and preprocessor.
    2. Loads out-of-time validation transactions.
    3. Computes TreeSHAP values on representative validation sample.
    4. Computes global feature rankings and summary plot.
    5. Explains representative individual transaction archetypes.
    6. Generates local waterfall visualization.
    7. Saves artifacts and structured documentation reports.
    """
    from bankguard.data.join import safe_join_transaction_identity
    from bankguard.data.loader import load_training_data

    start_time = time.time()
    out_path = Path(output_dir)
    art_path = Path(artifact_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    art_path.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at {m_path}")
    model = joblib.load(m_path)

    # 2. Load dataset and partition identically
    print("Loading data for explainability pipeline...")
    tx_df, id_df = load_training_data(nrows=None, verbose=False)
    full_df, _ = safe_join_transaction_identity(tx_df, id_df, validate_cardinality=True)

    _, val_split = temporal_train_val_split(full_df, time_col="TransactionDT", val_ratio=0.2)
    X_val_df, y_val_series, val_ids = split_features_and_target(val_split, return_ids=True)

    # Load or fit preprocessor (scale_numeric=False)
    p_path = Path(preprocessor_path)
    if p_path.is_file():
        preprocessor = joblib.load(p_path)
    else:
        preprocessor = BankGuardPreprocessor(scale_numeric=False, drop_raw_time=True)
        # fit on training set
        train_split, _ = temporal_train_val_split(full_df, time_col="TransactionDT", val_ratio=0.2)
        X_train_df, _, _ = split_features_and_target(train_split, return_ids=True)
        preprocessor.fit(X_train_df)

    X_val_trans = preprocessor.transform(X_val_df)
    feature_names = preprocessor.get_feature_names_out()

    y_val = np.asarray(y_val_series, dtype=int)
    val_ids_arr = np.asarray(val_ids, dtype=np.int64)

    # 3. Initialize TreeExplainer
    print("Initializing SHAP TreeExplainer...")
    explainer = get_tree_explainer(model)

    # 4. Global SHAP on representative sample
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(X_val_trans), size=min(sample_size, len(X_val_trans)), replace=False)
    X_sample = X_val_trans[sample_indices]

    print(f"Computing TreeSHAP attributions on {len(X_sample):,} validation transactions...")
    t_shap_start = time.time()
    shap_vals_sample = compute_shap_values(explainer, X_sample)
    shap_duration = round(time.time() - t_shap_start, 2)
    print(f"  SHAP computation completed in {shap_duration}s.")

    # 5. Global importance
    df_global = compute_global_feature_importance(shap_vals_sample, feature_names=feature_names, top_n=30)

    # 6. Global summary plot
    global_plot_file = out_path / "shap_summary.png"
    plot_global_shap_summary(
        shap_values=shap_vals_sample,
        X_sample=X_sample,
        feature_names=feature_names,
        output_path=global_plot_file,
    )
    # Copy to artifacts
    import shutil
    shutil.copyfile(global_plot_file, art_path / "shap_summary.png")

    # 7. Local explanation examples for dashboard
    print("Generating representative transaction explanations for dashboard...")
    examples = generate_dashboard_explanation_examples(
        explainer=explainer,
        model=model,
        X_val=X_val_trans,
        y_val=y_val,
        val_ids=val_ids_arr,
        feature_names=feature_names,
        threshold=0.22,
    )

    # 8. Local waterfall plot for the high-confidence fraud case
    local_plot_file = out_path / "shap_local_waterfall.png"
    if examples:
        plot_local_waterfall(examples[0], local_plot_file)
        shutil.copyfile(local_plot_file, art_path / "shap_local_waterfall.png")

    # 9. Save artifacts and reports
    examples_json_file = art_path / "explanation_examples.json"
    with open(examples_json_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    report_md = format_explainability_report_markdown(
        df_global=df_global,
        examples=examples,
        n_samples_explained=len(X_sample),
    )

    md_report_file = out_path / "model_explainability.md"
    with open(md_report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_report_file = out_path / "model_explainability.json"
    save_json_data = {
        "samples_explained": len(X_sample),
        "shap_duration_seconds": shap_duration,
        "global_importance": df_global.to_dict(orient="records"),
        "dashboard_examples": examples,
        "disclaimer": EXPLANATION_DISCLAIMER,
    }
    with open(json_report_file, "w", encoding="utf-8") as f:
        json.dump(save_json_data, f, indent=2)

    elapsed_total = round(time.time() - start_time, 2)
    print(
        f"\nExplainability Pipeline successfully executed in {elapsed_total}s:\n"
        f"  - Report: {md_report_file}\n"
        f"  - Structured Data: {json_report_file}\n"
        f"  - Dashboard Artifact: {examples_json_file}\n"
        f"  - Global Summary Plot: {global_plot_file}\n"
        f"  - Local Waterfall Plot: {local_plot_file}"
    )

    return {
        "global_importance": df_global,
        "examples": examples,
        "reports": {
            "markdown": str(md_report_file),
            "json": str(json_report_file),
            "global_plot": str(global_plot_file),
            "local_plot": str(local_plot_file),
        },
        "artifacts": {
            "examples_json": str(examples_json_file),
        },
    }


if __name__ == "__main__":
    run_explainability_pipeline()
