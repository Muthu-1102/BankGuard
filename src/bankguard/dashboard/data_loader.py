"""Data Loader for BankGuard macOS Glass Dashboard.

Loads saved evaluation artifacts, predictions, metrics, and data quality reports.
Strictly displays saved results and real validation data (zero fabricated numbers).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

REPORT_DIR = Path("report")
ARTIFACTS_DIR = Path("artifacts")


@st.cache_data
def load_json_artifact(filename: str) -> Optional[Dict[str, Any]]:
    """Loads a JSON report from report/ directory."""
    path = REPORT_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_csv_artifact(filename: str) -> Optional[pd.DataFrame]:
    """Loads a CSV artifact from report/ or artifacts/ directory."""
    for folder in [REPORT_DIR, ARTIFACTS_DIR]:
        path = folder / filename
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return None


@st.cache_data
def load_validation_predictions(max_rows: int = 5000) -> Optional[pd.DataFrame]:
    """Loads actual validation predictions from artifacts/improved_predictions.npz."""
    npz_path = ARTIFACTS_DIR / "improved_predictions.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        tx_ids = data["transaction_ids"][:max_rows]
        y_prob = data["y_prob"][:max_rows]
        y_true = data["y_true"][:max_rows]

        # Determine policy tier
        tiers = []
        for p in y_prob:
            if p >= 0.40:
                tiers.append("HIGH RISK")
            elif p >= 0.10:
                tiers.append("REVIEW")
            else:
                tiers.append("ALLOW")

        labels = ["Fraud (1)" if y == 1 else "Legitimate (0)" for y in y_true]

        df = pd.DataFrame({
            "TransactionID": tx_ids,
            "Fraud Score": np.round(y_prob, 4),
            "Risk Tier": tiers,
            "Actual Label": labels,
            "Raw Probability": y_prob,
            "Is Actual Fraud": y_true,
        })
        return df
    except Exception:
        return None


@st.cache_data
def load_explanation_examples() -> Optional[List[Dict[str, Any]]]:
    """Loads saved SHAP explanation archetypes from artifacts/explanation_examples.json."""
    path = ARTIFACTS_DIR / "explanation_examples.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Specific Report Accessors
def get_model_comparison() -> Optional[Dict[str, Any]]:
    return load_json_artifact("model_comparison.json")


def get_baseline_metrics() -> Optional[Dict[str, Any]]:
    return load_json_artifact("baseline_metrics.json")


def get_improved_metrics() -> Optional[Dict[str, Any]]:
    return load_json_artifact("improved_metrics.json")


def get_threshold_evaluation() -> Optional[Dict[str, Any]]:
    return load_json_artifact("threshold_evaluation.json")


def get_decision_policy_report() -> Optional[Dict[str, Any]]:
    return load_json_artifact("decision_policy_report.json")


def get_model_explainability() -> Optional[Dict[str, Any]]:
    return load_json_artifact("model_explainability.json")


def get_data_quality_report() -> Optional[Dict[str, Any]]:
    return load_json_artifact("data_quality_report.json")


def get_join_report() -> Optional[Dict[str, Any]]:
    return load_json_artifact("join_report.json")


def get_missingness_analysis() -> Optional[Dict[str, Any]]:
    return load_json_artifact("missingness_analysis.json")


def get_data_inventory_report() -> Optional[Dict[str, Any]]:
    return load_json_artifact("data_inventory_report.json")


def get_target_analysis() -> Optional[Dict[str, Any]]:
    return load_json_artifact("target_analysis.json")


def get_amount_time_analysis() -> Optional[Dict[str, Any]]:
    return load_json_artifact("amount_time_analysis.json")


def get_feature_importances() -> Optional[pd.DataFrame]:
    return load_csv_artifact("improved_feature_importances.csv")

