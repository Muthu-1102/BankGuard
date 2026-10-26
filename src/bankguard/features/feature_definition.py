"""Feature and target definition module for BankGuard (Task 4.1).

Defines target (isFraud), identifiers, and feature sets with strict leakage prevention,
explicit X/y separation, automated leakage auditing, and typing breakdown.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import yaml

TARGET_COLUMN: str = "isFraud"
ID_COLUMN: str = "TransactionID"
DEFAULT_EXCLUDED_COLUMNS: List[str] = [TARGET_COLUMN, ID_COLUMN]


class TargetLeakageError(Exception):
    """Raised when the target column or a forbidden target-derived field enters the feature matrix."""
    pass


def get_project_root() -> Path:
    """Returns absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_feature_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads feature configuration from configs/feature_config.yaml."""
    if config_path is None:
        target_path = get_project_root() / "configs" / "feature_config.yaml"
    else:
        target_path = Path(config_path)

    if not target_path.is_file():
        # Fallback to defaults if file is not found
        return {
            "target": {"name": TARGET_COLUMN},
            "identifiers": {"primary_key": ID_COLUMN},
            "exclusions": {"default": DEFAULT_EXCLUDED_COLUMNS},
            "categorical_features": {"transaction": [], "identity": []},
        }

    with open(target_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_feature_names(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    additional_exclude: Optional[List[str]] = None,
) -> List[str]:
    """Returns a clean list of candidate feature column names from df, strictly excluding target and IDs.

    Args:
        df: Input DataFrame containing raw or joined columns.
        target_col: Target column name to exclude.
        id_col: Identifier column name to exclude.
        additional_exclude: Optional list of additional column names to exclude.

    Returns:
        List of feature column names present in df, excluding target and identifiers.
    """
    exclude_set = {target_col, id_col}
    if additional_exclude:
        exclude_set.update(additional_exclude)

    # Preserve natural column ordering from df while filtering exclusions
    features = [col for col in df.columns if col not in exclude_set]
    return features


@overload
def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    feature_cols: Optional[List[str]] = None,
    additional_exclude: Optional[List[str]] = None,
    return_ids: Literal[False] = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    ...


@overload
def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    feature_cols: Optional[List[str]] = None,
    additional_exclude: Optional[List[str]] = None,
    return_ids: Literal[True] = ...,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    ...


@overload
def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    feature_cols: Optional[List[str]] = None,
    additional_exclude: Optional[List[str]] = None,
    return_ids: bool = False,
) -> Union[Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]]:
    ...


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    feature_cols: Optional[List[str]] = None,
    additional_exclude: Optional[List[str]] = None,
    return_ids: bool = False,
) -> Union[Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]]:
    """Splits a dataset into features X and target y with rigorous target-leakage assertion.

    Args:
        df: Input DataFrame containing features, optional target, and optional ID.
        target_col: Target column name.
        id_col: Primary key/identifier column name.
        feature_cols: Explicit list of features to extract. If None, uses all columns minus exclusions.
        additional_exclude: Additional columns to exclude when feature_cols is None.
        return_ids: If True, returns a 3-tuple (X, y, ids).

    Returns:
        (X, y) or (X, y, ids) where:
            X: pd.DataFrame of features, strictly free of target_col.
            y: pd.Series of target values (or None if target_col not present in df).
            ids: pd.Series of ID values (if return_ids=True).

    Raises:
        TargetLeakageError: If target_col is detected inside X.
    """
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    # 1. Determine feature column list
    if feature_cols is not None:
        selected_features = list(feature_cols)
    else:
        selected_features = get_feature_names(
            df=df,
            target_col=target_col,
            id_col=id_col,
            additional_exclude=additional_exclude,
        )

    # 2. Extract X
    X = df[selected_features].copy()

    # 3. CRITICAL AUDIT: Enforce zero target leakage
    if target_col in X.columns:
        raise TargetLeakageError(
            f"CRITICAL TARGET LEAKAGE DETECTED!\n"
            f"Target column '{target_col}' was found in feature matrix X columns.\n"
            f"Target must never be an input feature."
        )

    # 4. Extract y if present
    y: Optional[pd.Series] = None
    if target_col in df.columns:
        y = df[target_col].copy()

    # 5. Extract IDs if requested
    if return_ids:
        if id_col in df.columns:
            ids = df[id_col].copy()
        else:
            ids = pd.Series(df.index, index=df.index, name=id_col)
        return X, y, ids

    return X, y


def identify_feature_types(
    X: pd.DataFrame,
    known_categorical: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Partitions feature columns into numeric and categorical types.

    Args:
        X: Feature DataFrame.
        known_categorical: Optional list of known categorical column names from config.

    Returns:
        Dictionary with 'numeric' and 'categorical' lists of column names.
    """
    if known_categorical is None:
        cfg = load_feature_config()
        cat_tx = cfg.get("categorical_features", {}).get("transaction", [])
        cat_id = cfg.get("categorical_features", {}).get("identity", [])
        known_cat_set = set(cat_tx + cat_id)
    else:
        known_cat_set = set(known_categorical)

    categorical_cols: List[str] = []
    numeric_cols: List[str] = []

    for col in X.columns:
        dtype = X[col].dtype
        is_object_or_cat = pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype)
        is_known_cat = col in known_cat_set

        if is_object_or_cat or is_known_cat:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        else:
            # Fallback for unrecognized types
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }


def audit_feature_leakage(
    X: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    forbidden_columns: Optional[List[str]] = None,
    raise_on_violation: bool = True,
) -> Dict[str, Any]:
    """Audits a feature matrix X to verify absolute absence of the target and any forbidden leaky fields.

    Args:
        X: Feature DataFrame to audit.
        target_col: Expected target column name.
        forbidden_columns: Optional additional forbidden columns.
        raise_on_violation: If True, raises TargetLeakageError on any violation.

    Returns:
        Dictionary with audit outcome details.

    Raises:
        TargetLeakageError: If any violation is found and raise_on_violation is True.
    """
    violations: List[str] = []
    lower_col_map = {col.lower(): col for col in X.columns}

    # 1. Exact target match
    if target_col in X.columns:
        violations.append(f"Exact target column '{target_col}' present in features.")

    # 2. Case-insensitive target match (e.g. 'isfraud', 'ISFRAUD')
    if target_col.lower() in lower_col_map and lower_col_map[target_col.lower()] != target_col:
        violations.append(f"Case-insensitive target variant '{lower_col_map[target_col.lower()]}' present in features.")

    # 3. Forbidden columns check
    if forbidden_columns:
        for f_col in forbidden_columns:
            if f_col in X.columns:
                violations.append(f"Forbidden column '{f_col}' present in features.")

    passed = len(violations) == 0

    if not passed and raise_on_violation:
        msg = "\n".join(violations)
        raise TargetLeakageError(f"Target Leakage Audit Failed!\n{msg}")

    return {
        "passed": passed,
        "target_col": target_col,
        "target_absent": target_col not in X.columns,
        "violations": violations,
        "feature_count": len(X.columns),
        "row_count": len(X),
    }


def generate_feature_inventory_report(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generates and exports feature inventory and X/y verification report for Task 4.1."""
    if output_dir is None:
        out_path = get_project_root() / "report"
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Perform X/y separation
    X, y, ids = split_features_and_target(df, return_ids=True)

    # 2. Perform leakage audit
    audit_results = audit_feature_leakage(X, target_col=TARGET_COLUMN, forbidden_columns=[ID_COLUMN])

    # 3. Categorize feature types
    types_breakdown = identify_feature_types(X)
    numeric_count = len(types_breakdown["numeric"])
    categorical_count = len(types_breakdown["categorical"])

    summary = {
        "total_columns_input": len(df.columns),
        "total_rows": len(df),
        "target_column": TARGET_COLUMN,
        "target_present": y is not None,
        "id_column": ID_COLUMN,
        "id_present": ids is not None,
        "feature_count": len(X.columns),
        "numeric_feature_count": numeric_count,
        "categorical_feature_count": categorical_count,
        "leakage_audit_passed": audit_results["passed"],
        "target_in_features": TARGET_COLUMN in X.columns,
        "id_in_features": ID_COLUMN in X.columns,
    }

    # Save JSON summary
    import json
    json_path = out_path / "feature_definition_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown summary
    md_path = out_path / "feature_definition_report.md"
    md_content = f"""# BankGuard — Target & Feature Definition Audit (Task 4.1)

This report documents the explicit definition of the model target and candidate features, along with automated leakage verification on the training dataset ({summary['total_rows']:,} rows).

---

## 1. Target & Identifier Specification

- **Target Column (`y`):** `{TARGET_COLUMN}` (binary classification: 1 = Fraud, 0 = Legitimate)
- **Primary Key Identifier:** `{ID_COLUMN}` (traceability only; excluded from model input)
- **Target Presence in X:** **{summary['target_in_features']}** (PASSED: Target is completely absent from X)
- **ID Presence in X:** **{summary['id_in_features']}** (PASSED: Identifier is completely absent from X)
- **Leakage Security Audit:** **{'PASSED (0 violations)' if summary['leakage_audit_passed'] else 'FAILED'}**

---

## 2. Feature Matrix Breakdown

- **Total Input Table Columns:** {summary['total_columns_input']}
- **Total Candidate Features (`X`):** {summary['feature_count']}
- **Numeric Features:** {numeric_count} ({round(numeric_count / summary['feature_count'] * 100, 1)}%)
- **Categorical Features:** {categorical_count} ({round(categorical_count / summary['feature_count'] * 100, 1)}%)

### High-Level Category Families in Feature Matrix

| Family Group | Column Prefix / Pattern | Count | Primary Type | Modeling Treatment |
|---|---|---:|---|---|
| Transaction Attributes | `TransactionAmt`, `dist1`, `dist2` | 3 | Numeric | Log transform, scaling, outlier clipping |
| Categorical Transaction | `ProductCD`, `card4`, `card6`, `P/R_emaildomain` | 5 | Categorical | Target encoding / Ordinal encoding / Frequency |
| Card & Address Identifiers | `card1`–`card3`, `card5`, `addr1`, `addr2` | 6 | Numeric/Categorical | High-cardinality binning / grouping |
| Count Features | `C1`–`C14` | 14 | Numeric | Native numeric scaling / tree splits |
| Timedelta Features | `D1`–`D15` | 15 | Numeric | Temporal consistency checks |
| Match Flags | `M1`–`M9` | 9 | Categorical | Boolean/Missing flag encoding |
| Vesta Rich Features | `V1`–`V339` | 339 | Numeric | Missingness-aware tree routing |
| Identity Device Metadata | `DeviceType`, `DeviceInfo` | 2 | Categorical | Frequency encoding / Missing flag |
| Identity Anonymized | `id_01`–`id_38` | 38 | Mixed Numeric/Categorical | Missing indicator + numerical imputation |

---

## 3. Leakage Prevention Contract

1. **Strict X/y Separation:** The `split_features_and_target()` function provides guaranteed programmatic isolation of `{TARGET_COLUMN}`.
2. **Automated Assertion:** Any call attempting to include `{TARGET_COLUMN}` in `feature_cols` or failing to exclude it triggers an explicit `TargetLeakageError`.
3. **Reproducibility:** Pipeline feature signatures are locked in `configs/feature_config.yaml`.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return summary


def run_feature_definition_pipeline(
    nrows: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Orchestrates loading training data, separating X/y, auditing leakage, and generating report."""
    from bankguard.data.join import safe_join_transaction_identity
    from bankguard.data.loader import load_training_data

    print(f"Loading training data (nrows={nrows or 'all'})...")
    tx_df, id_df = load_training_data(nrows=nrows, verbose=False)

    print("Joining transaction and identity tables...")
    joined_df, _ = safe_join_transaction_identity(tx_df, id_df, validate_cardinality=True)

    print("Executing feature separation and security leakage audit...")
    summary = generate_feature_inventory_report(joined_df, output_dir=output_dir)

    print(
        f"Feature definition pipeline completed successfully:\n"
        f"  - Total Features in X: {summary['feature_count']}\n"
        f"  - Target in X: {summary['target_in_features']} (Audited: Target is absent)\n"
        f"  - ID in X: {summary['id_in_features']} (Audited: ID is absent)\n"
        f"  - Leakage Audit Passed: {summary['leakage_audit_passed']}"
    )
    return summary


if __name__ == "__main__":
    run_feature_definition_pipeline()

