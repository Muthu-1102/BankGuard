"""Improved model training and evaluation module for BankGuard (Task 5.2).

Implements a high-capacity gradient boosted decision tree (LightGBM) trained on chronologically
partitioned financial transactions and evaluated on the exact same out-of-time validation transactions
as the baseline model.

Captures non-linear risk interactions, handles high-cardinality ordinal features,
and evaluates primary class-imbalance metrics (PR-AUC / Average Precision, ROC-AUC, Brier score).
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from bankguard.features.feature_definition import (
    ID_COLUMN,
    TARGET_COLUMN,
    split_features_and_target,
)
from bankguard.features.preprocessor import (
    BankGuardPreprocessor,
    temporal_train_val_split,
)
from bankguard.models.metrics import (
    compare_model_metrics,
    compute_classification_metrics,
    format_comparison_markdown,
    format_metrics_markdown,
    save_metrics_report,
)


def get_default_lgbm_params(random_state: int = 42) -> Dict[str, Any]:
    """Returns standardized, reproducible LightGBM hyperparameters for fraud detection.

    Designed for tabular data with extreme class imbalance (3.5% fraud rate):
    - 300 estimators with learning rate 0.05 for stable gradient descent
    - num_leaves=63 and max_depth=10 to model multi-way feature interactions
    - min_child_samples=50 to prevent overfitting on low-density transaction leaves
    - subsample=0.8 and colsample_bytree=0.8 to decorrelate individual trees
    - importance_type='gain' to prioritize total information gain across splits
    """
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 10,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }


def train_improved_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Trains a LightGBM gradient boosting classifier on preprocessed features.

    Args:
        X_train: 2D numpy array of preprocessed training features.
        y_train: 1D array of binary fraud labels (0 or 1).
        params: Optional dictionary of hyperparameter overrides.
        random_state: Reproducibility seed.

    Returns:
        Fitted LGBMClassifier model.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot train on empty dataset.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Feature count ({len(X_train)}) does not match label count ({len(y_train)})."
        )

    effective_params = get_default_lgbm_params(random_state=random_state)
    if params is not None:
        effective_params.update(params)

    model = lgb.LGBMClassifier(**effective_params)
    model.fit(X_train, y_train)
    return model


def predict_improved(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates binary fraud predictions and calibrated fraud probabilities.

    Args:
        model: Fitted LGBMClassifier model.
        X: Preprocessed feature array.
        threshold: Decision threshold for positive fraud classification (default 0.5).

    Returns:
        Tuple of (y_pred, y_prob) where:
            y_pred: Binary predictions (0 or 1).
            y_prob: Continuous fraud probabilities in [0.0, 1.0].
    """
    probabilities = model.predict_proba(X)
    y_prob = probabilities[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def compute_feature_importances(
    model: lgb.LGBMClassifier,
    feature_names: Optional[List[str]] = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Computes total information gain and split frequency feature importances.

    Args:
        model: Fitted LGBMClassifier.
        feature_names: Optional list of feature names matching model inputs.
        top_n: Number of top features to return.

    Returns:
        DataFrame sorted by gain importance descending.
    """
    gain_importances = model.booster_.feature_importance(importance_type="gain")
    split_importances = model.booster_.feature_importance(importance_type="split")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(gain_importances))]

    total_gain = float(np.sum(gain_importances))
    rel_pct = (gain_importances / total_gain * 100.0) if total_gain > 0 else np.zeros_like(gain_importances)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "gain_importance": gain_importances,
        "split_importance": split_importances,
        "relative_importance_pct": np.round(rel_pct, 3),
    }).sort_values(by="gain_importance", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)


def save_improved_artifacts(
    model: lgb.LGBMClassifier,
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_val_pred: np.ndarray,
    val_ids: Optional[np.ndarray],
    artifact_dir: Union[str, Path],
    model_filename: str = "improved_model.joblib",
    predictions_filename: str = "improved_predictions.npz",
    feature_importances: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """Saves model weights, validation predictions, and feature rankings to disk."""
    art_path = Path(artifact_dir)
    art_path.mkdir(parents=True, exist_ok=True)

    model_file = art_path / model_filename
    pred_file = art_path / predictions_filename

    # Save model artifact
    joblib.dump(model, model_file)

    # Save validation predictions
    save_dict = {
        "y_true": np.asarray(y_val_true, dtype=int),
        "y_prob": np.asarray(y_val_prob, dtype=np.float32),
        "y_pred": np.asarray(y_val_pred, dtype=int),
    }
    if val_ids is not None:
        save_dict["transaction_ids"] = np.asarray(val_ids, dtype=np.int64)

    np.savez_compressed(pred_file, **save_dict)

    result_paths = {
        "model_path": model_file,
        "predictions_path": pred_file,
    }

    if feature_importances is not None:
        imp_file = art_path / "improved_feature_importances.csv"
        feature_importances.to_csv(imp_file, index=False)
        result_paths["feature_importances_path"] = imp_file

    return result_paths


def load_improved_predictions(predictions_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Loads saved validation predictions for downstream threshold and comparative analysis."""
    p_path = Path(predictions_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Predictions file not found at: {p_path}")

    data = np.load(p_path)
    return {key: data[key] for key in data.files}


def run_improved_pipeline(
    train_df: Optional[pd.DataFrame] = None,
    nrows: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    scale_numeric: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """End-to-end orchestration for Task 5.2:

    1. Loads dataset and chronologically splits 80% train / 20% val (identical to baseline).
    2. Preprocesses features strictly on training split (scale_numeric=False for trees, zero refit).
    3. Trains LightGBM model recording hyperparameters, seeds, and duration.
    4. Evaluates predictions on the exact same validation split.
    5. Computes gain-based feature importances.
    6. Compares performance directly against Task 5.1 baseline.
    7. Saves artifacts, predictions, and structured comparative reports.
    """
    from bankguard.data.join import safe_join_transaction_identity
    from bankguard.data.loader import load_training_data

    start_time = time.time()
    out_path = Path(output_dir) if output_dir else Path("report")
    art_path = Path(artifact_dir) if artifact_dir else Path("artifacts")

    out_path.mkdir(parents=True, exist_ok=True)
    art_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    if train_df is None:
        print(f"Loading training dataset for Improved LightGBM Pipeline (nrows={nrows or 'all'})...")
        tx_df, id_df = load_training_data(nrows=nrows, verbose=False)
        full_df, _ = safe_join_transaction_identity(tx_df, id_df, validate_cardinality=True)
    else:
        full_df = train_df

    # 2. Chronological split (80% train / 20% validation)
    print("Partitioning dataset chronologically by TransactionDT (80% train / 20% val)...")
    train_split, val_split = temporal_train_val_split(full_df, time_col="TransactionDT", val_ratio=0.2)

    X_train, y_train, train_ids = split_features_and_target(train_split, return_ids=True)
    X_val, y_val, val_ids = split_features_and_target(val_split, return_ids=True)

    # 3. Preprocess features strictly on training split
    print(f"Preprocessing features strictly on training rows ({len(X_train):,} samples)...")
    preprocessor = BankGuardPreprocessor(scale_numeric=scale_numeric, drop_raw_time=True)
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)

    # 4. Train LightGBM model
    effective_params = get_default_lgbm_params(random_state=random_state)
    if model_params is not None:
        effective_params.update(model_params)

    print(
        f"Fitting LightGBM on {X_train_trans.shape[0]:,} rows x {X_train_trans.shape[1]} features "
        f"(n_estimators={effective_params['n_estimators']}, learning_rate={effective_params['learning_rate']})..."
    )
    t_train_start = time.time()
    model = train_improved_model(
        X_train=X_train_trans,
        y_train=np.asarray(y_train, dtype=int),
        params=effective_params,
        random_state=random_state,
    )
    train_duration = round(time.time() - t_train_start, 2)
    print(f"  LightGBM model training completed in {train_duration}s.")

    # 5. Predict on out-of-time validation set
    print(f"Generating predictions on validation set ({len(X_val):,} transactions)...")
    y_val_pred, y_val_prob = predict_improved(model, X_val_trans, threshold=0.5)

    # 6. Evaluate metrics
    metrics = compute_classification_metrics(
        y_true=np.asarray(y_val, dtype=int),
        y_prob=y_val_prob,
        threshold=0.5,
    )
    metrics["model_type"] = "LightGBM"
    metrics["lightgbm_version"] = getattr(lgb, "__version__", "unknown")
    metrics["train_duration_seconds"] = train_duration
    metrics["train_samples"] = len(X_train)
    metrics["validation_samples"] = len(X_val)
    metrics["feature_count"] = X_train_trans.shape[1]
    metrics["hyperparameters"] = {
        k: v for k, v in effective_params.items() if k not in ("verbose", "n_jobs")
    }

    # 7. Extract feature importances
    feature_names = preprocessor.get_feature_names_out()
    df_importances = compute_feature_importances(model, feature_names=feature_names, top_n=30)

    # 8. Save model and predictions
    saved_artifacts = save_improved_artifacts(
        model=model,
        y_val_true=np.asarray(y_val, dtype=int),
        y_val_prob=y_val_prob,
        y_val_pred=y_val_pred,
        val_ids=np.asarray(val_ids, dtype=np.int64),
        artifact_dir=art_path,
        model_filename="improved_model.joblib",
        predictions_filename="improved_predictions.npz",
        feature_importances=df_importances,
    )

    # 9. Save metrics reports
    json_path, md_path = save_metrics_report(
        metrics=metrics,
        model_name="improved",
        output_dir=out_path,
    )

    # 10. Load baseline metrics for comparative analysis
    baseline_json_path = out_path / "baseline_metrics.json"
    comparison_data: Optional[Dict[str, Any]] = None
    comparison_md_path: Optional[Path] = None

    if baseline_json_path.is_file():
        try:
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
            comparison_data = compare_model_metrics(
                baseline_metrics=baseline_metrics,
                candidate_metrics=metrics,
                baseline_name="Baseline (LogisticRegression)",
                candidate_name="Improved (LightGBM)",
            )
            comp_md_str = format_comparison_markdown(comparison_data)

            # Add feature importance preview to comparison report
            comp_md_str += "\n#### Top 10 Predictive Features by Total Information Gain\n\n"
            comp_md_str += "| Rank | Feature | Total Information Gain | Relative Gain (%) |\n"
            comp_md_str += "|---:|---|---:|---:|\n"
            for i, row in df_importances.head(10).iterrows():
                comp_md_str += f"| {i+1} | `{row['feature']}` | {row['gain_importance']:,.1f} | {row['relative_importance_pct']:.2f}% |\n"

            comparison_md_path = out_path / "model_comparison.md"
            with open(comparison_md_path, "w", encoding="utf-8") as f:
                f.write(comp_md_str)

            comparison_json_path = out_path / "model_comparison.json"
            with open(comparison_json_path, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to compute baseline comparison: {e}")

    elapsed_total = round(time.time() - start_time, 2)
    print(
        f"\nImproved Pipeline successfully executed in {elapsed_total}s:\n"
        f"  - Model Saved: {saved_artifacts['model_path']}\n"
        f"  - Predictions Saved: {saved_artifacts['predictions_path']}\n"
        f"  - PR-AUC (Average Precision): {metrics['pr_auc']:.4f}\n"
        f"  - ROC-AUC: {metrics['roc_auc']:.4f}\n"
        f"  - Brier Score: {metrics['brier_score']:.4f}\n"
        f"  - Log Loss: {metrics['log_loss']:.4f}\n"
        f"  - Report: {md_path}"
    )
    if comparison_md_path:
        print(f"  - Comparative Analysis: {comparison_md_path}")

    return {
        "metrics": metrics,
        "comparison": comparison_data,
        "feature_importances": df_importances,
        "artifacts": {k: str(v) for k, v in saved_artifacts.items()},
        "reports": {
            "json": str(json_path),
            "markdown": str(md_path),
            "comparison_markdown": str(comparison_md_path) if comparison_md_path else None,
        },
    }


if __name__ == "__main__":
    run_improved_pipeline()
