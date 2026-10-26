"""Baseline model training and evaluation module for BankGuard (Task 5.1).

Implements a reproducible linear baseline (Logistic Regression with balanced class weighting)
fitted on chronologically partitioned training transactions and evaluated on out-of-time validation transactions.
Saves model artifacts, predictions, and class-aware metrics.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    compute_classification_metrics,
    format_metrics_markdown,
    save_metrics_report,
)


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 500,
    random_state: int = 42,
    class_weight: str = "balanced",
    solver: str = "lbfgs",
) -> LogisticRegression:
    """Trains a reproducible Logistic Regression baseline model with balanced class weighting.

    Args:
        X_train: Dense 2D numpy array of preprocessed training features.
        y_train: 1D array of binary ground-truth labels.
        max_iter: Maximum solver iterations.
        random_state: Reproducibility seed.
        class_weight: Weighting strategy ('balanced' handles 27.6:1 class imbalance).
        solver: Optimization algorithm.

    Returns:
        Fitted LogisticRegression model.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot train on empty dataset.")

    if len(X_train) != len(y_train):
        raise ValueError(f"Feature count ({len(X_train)}) does not match label count ({len(y_train)}).")

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_baseline(
    model: LogisticRegression,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates class predictions and calibrated fraud probabilities from baseline model.

    Args:
        model: Fitted LogisticRegression model.
        X: Preprocessed feature array.
        threshold: Decision threshold for positive fraud classification.

    Returns:
        Tuple of (y_pred, y_prob) where:
            y_pred: Binary predictions (0 or 1).
            y_prob: Predicted fraud probabilities for class 1.
    """
    probabilities = model.predict_proba(X)
    y_prob = probabilities[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def save_baseline_artifacts(
    model: LogisticRegression,
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_val_pred: np.ndarray,
    val_ids: Optional[np.ndarray],
    artifact_dir: Union[str, Path],
    model_filename: str = "baseline_model.joblib",
    predictions_filename: str = "baseline_predictions.npz",
) -> Dict[str, Path]:
    """Saves model weights and out-of-time validation predictions to disk."""
    art_path = Path(artifact_dir)
    art_path.mkdir(parents=True, exist_ok=True)

    model_file = art_path / model_filename
    pred_file = art_path / predictions_filename

    # Save model
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

    return {
        "model_path": model_file,
        "predictions_path": pred_file,
    }


def load_baseline_predictions(predictions_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Loads saved validation predictions for threshold analysis and evaluation."""
    p_path = Path(predictions_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Predictions file not found at: {p_path}")

    data = np.load(p_path)
    return {key: data[key] for key in data.files}


def run_baseline_pipeline(
    train_df: Optional[pd.DataFrame] = None,
    nrows: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    scale_numeric: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """End-to-end orchestration for Task 5.1:

    1. Loads dataset and splits chronologically (80% train / 20% val).
    2. Preprocesses features strictly on training split (zero validation refit).
    3. Trains Logistic Regression baseline model with balanced class weighting.
    4. Evaluates predictions on validation set (PR-AUC, ROC-AUC, Brier score, confusion matrix).
    5. Saves artifacts, predictions, and structured reports.
    """
    from bankguard.data.join import safe_join_transaction_identity
    from bankguard.data.loader import load_training_data

    start_time = time.time()
    if output_dir is None:
        out_path = Path("report")
    else:
        out_path = Path(output_dir)

    if artifact_dir is None:
        art_path = Path("artifacts")
    else:
        art_path = Path(artifact_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    art_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    if train_df is None:
        print(f"Loading training dataset for Baseline Pipeline (nrows={nrows or 'all'})...")
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

    # 4. Train baseline model
    print(f"Fitting Logistic Regression baseline on {X_train_trans.shape[0]:,} rows x {X_train_trans.shape[1]} features...")
    t_train_start = time.time()
    model = train_baseline_model(
        X_train=X_train_trans,
        y_train=np.asarray(y_train, dtype=int),
        max_iter=500,
        random_state=random_state,
        class_weight="balanced",
    )
    train_duration = round(time.time() - t_train_start, 2)
    print(f"  Model training completed in {train_duration}s.")

    # 5. Predict on out-of-time validation set
    print(f"Generating predictions on validation set ({len(X_val):,} transactions)...")
    y_val_pred, y_val_prob = predict_baseline(model, X_val_trans, threshold=0.5)

    # 6. Evaluate metrics
    metrics = compute_classification_metrics(
        y_true=np.asarray(y_val, dtype=int),
        y_prob=y_val_prob,
        threshold=0.5,
    )
    metrics["model_type"] = "LogisticRegression"
    metrics["class_weight"] = "balanced"
    metrics["train_duration_seconds"] = train_duration
    metrics["train_samples"] = len(X_train)
    metrics["validation_samples"] = len(X_val)
    metrics["feature_count"] = X_train_trans.shape[1]

    # 7. Save model and predictions
    saved_artifacts = save_baseline_artifacts(
        model=model,
        y_val_true=np.asarray(y_val, dtype=int),
        y_val_prob=y_val_prob,
        y_val_pred=y_val_pred,
        val_ids=np.asarray(val_ids, dtype=np.int64),
        artifact_dir=art_path,
        model_filename="baseline_model.joblib",
        predictions_filename="baseline_predictions.npz",
    )

    # 8. Save metrics reports
    json_path, md_path = save_metrics_report(
        metrics=metrics,
        model_name="baseline",
        output_dir=out_path,
    )

    elapsed_total = round(time.time() - start_time, 2)
    print(
        f"\nBaseline Pipeline successfully executed in {elapsed_total}s:\n"
        f"  - Model Saved: {saved_artifacts['model_path']}\n"
        f"  - Predictions Saved: {saved_artifacts['predictions_path']}\n"
        f"  - PR-AUC (Average Precision): {metrics['pr_auc']:.4f}\n"
        f"  - ROC-AUC: {metrics['roc_auc']:.4f}\n"
        f"  - Brier Score: {metrics['brier_score']:.4f}\n"
        f"  - Precision @ 0.5: {metrics['precision']*100:.2f}%\n"
        f"  - Recall @ 0.5: {metrics['recall']*100:.2f}%\n"
        f"  - Report: {md_path}"
    )

    return {
        "metrics": metrics,
        "artifacts": {k: str(v) for k, v in saved_artifacts.items()},
        "reports": {"json": str(json_path), "markdown": str(md_path)},
    }


if __name__ == "__main__":
    run_baseline_pipeline()
