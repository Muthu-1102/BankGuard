"""Data quality and schema validation module for BankGuard.

Provides modular check functions to validate schema integrity, key uniqueness,
target values, and missingness patterns with actionable error reporting.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    """Raised when critical data quality checks fail."""
    pass


@dataclass
class CheckResult:
    """Represents the outcome of a single quality check."""
    check_name: str
    passed: bool
    is_critical: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def check_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    table_name: str = "table",
    is_critical: bool = True,
) -> CheckResult:
    """Checks that all required columns exist in the DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    passed = len(missing) == 0
    if passed:
        msg = f"All {len(required_cols)} required columns present in {table_name}."
    else:
        msg = f"Missing {len(missing)} required column(s) in {table_name}: {missing}"

    return CheckResult(
        check_name=f"required_columns_{table_name}",
        passed=passed,
        is_critical=is_critical,
        message=msg,
        details={"missing_columns": missing, "required_columns": required_cols},
    )


def check_key_uniqueness(
    df: pd.DataFrame,
    key_col: str = "TransactionID",
    table_name: str = "table",
    is_critical: bool = True,
) -> CheckResult:
    """Validates that the primary key column contains no duplicate or null values."""
    if key_col not in df.columns:
        return CheckResult(
            check_name=f"key_uniqueness_{table_name}",
            passed=False,
            is_critical=is_critical,
            message=f"Key column '{key_col}' not found in {table_name}.",
            details={"error": "Column missing"},
        )

    null_count = int(df[key_col].isnull().sum())
    dup_count = int(df[key_col].duplicated().sum())
    passed = (null_count == 0) and (dup_count == 0)

    if passed:
        msg = f"Key '{key_col}' is strictly unique with 0 nulls across {len(df):,} rows in {table_name}."
    else:
        msg = (
            f"Key '{key_col}' integrity violation in {table_name}: "
            f"{dup_count:,} duplicate(s), {null_count:,} null(s)."
        )

    return CheckResult(
        check_name=f"key_uniqueness_{table_name}",
        passed=passed,
        is_critical=is_critical,
        message=msg,
        details={"duplicate_count": dup_count, "null_count": null_count, "total_rows": len(df)},
    )


def check_target_validity(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    allowed_values: Optional[Set[Any]] = None,
    is_critical: bool = True,
) -> CheckResult:
    """Validates that the supervised target column contains only allowed binary values without nulls."""
    if allowed_values is None:
        allowed_values = {0, 1}

    if target_col not in df.columns:
        return CheckResult(
            check_name="target_validity",
            passed=False,
            is_critical=is_critical,
            message=f"Target column '{target_col}' not found in DataFrame.",
            details={"error": "Column missing"},
        )

    null_count = int(df[target_col].isnull().sum())
    unique_vals = set(df[target_col].dropna().unique())
    invalid_vals = list(unique_vals - allowed_values)
    passed = (null_count == 0) and (len(invalid_vals) == 0)

    distribution = df[target_col].value_counts(dropna=False).to_dict()
    distribution = {str(k): int(v) for k, v in distribution.items()}

    if passed:
        msg = f"Target '{target_col}' is valid with allowed values {allowed_values} and 0 nulls."
    else:
        msg = f"Target '{target_col}' invalid: {null_count} nulls, unexpected values: {invalid_vals}."

    return CheckResult(
        check_name="target_validity",
        passed=passed,
        is_critical=is_critical,
        message=msg,
        details={
            "null_count": null_count,
            "invalid_values": invalid_vals,
            "distribution": distribution,
        },
    )


def check_null_patterns(
    df: pd.DataFrame,
    table_name: str = "table",
    flag_threshold_pct: float = 95.0,
    is_critical: bool = False,
) -> CheckResult:
    """Analyzes missingness and flags columns with extreme null ratios."""
    total_rows = len(df)
    if total_rows == 0:
        return CheckResult(
            check_name=f"null_patterns_{table_name}",
            passed=True,
            is_critical=is_critical,
            message="DataFrame is empty.",
            details={},
        )

    null_counts = df.isnull().sum()
    null_pcts = (null_counts / total_rows) * 100
    high_null_cols = null_pcts[null_pcts >= flag_threshold_pct].to_dict()
    completely_empty_cols = null_pcts[null_pcts == 100.0].index.tolist()

    passed = len(completely_empty_cols) == 0
    msg = (
        f"Missingness analyzed in {table_name}: {len(high_null_cols)} column(s) >= {flag_threshold_pct}% null, "
        f"{len(completely_empty_cols)} completely empty column(s)."
    )

    return CheckResult(
        check_name=f"null_patterns_{table_name}",
        passed=passed,
        is_critical=is_critical,
        message=msg,
        details={
            "high_null_columns_count": len(high_null_cols),
            "high_null_columns": {k: round(v, 2) for k, v in high_null_cols.items()},
            "completely_empty_columns": completely_empty_cols,
        },
    )


def check_numeric_ranges(
    df: pd.DataFrame,
    amount_col: str = "TransactionAmt",
    is_critical: bool = True,
) -> CheckResult:
    """Validates that transaction amounts are non-negative numeric values."""
    if amount_col not in df.columns:
        return CheckResult(
            check_name="numeric_ranges_TransactionAmt",
            passed=False,
            is_critical=is_critical,
            message=f"Amount column '{amount_col}' missing.",
            details={},
        )

    negatives = int((df[amount_col] < 0).sum())
    nulls = int(df[amount_col].isnull().sum())
    passed = (negatives == 0) and (nulls == 0)

    if passed:
        msg = f"Amount column '{amount_col}' has 0 negative and 0 null values."
    else:
        msg = f"Amount column '{amount_col}' invalid: {negatives} negative values, {nulls} null values."

    return CheckResult(
        check_name="numeric_ranges_TransactionAmt",
        passed=passed,
        is_critical=is_critical,
        message=msg,
        details={"negative_count": negatives, "null_count": nulls},
    )


def run_data_quality_suite(
    tx_df: pd.DataFrame,
    id_df: Optional[pd.DataFrame] = None,
    is_training: bool = True,
    fail_on_critical: bool = True,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs all data quality checks on transaction and optional identity DataFrames.

    Raises:
        DataQualityError: If any critical check fails and fail_on_critical is True.
    """
    results: List[CheckResult] = []

    # 1. Required transaction columns
    core_tx_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]
    if is_training:
        core_tx_cols.append("isFraud")
    results.append(check_required_columns(tx_df, core_tx_cols, table_name="transaction", is_critical=True))

    # 2. Transaction key uniqueness
    results.append(check_key_uniqueness(tx_df, key_col="TransactionID", table_name="transaction", is_critical=True))

    # 3. Target validity
    if is_training and "isFraud" in tx_df.columns:
        results.append(check_target_validity(tx_df, target_col="isFraud", is_critical=True))

    # 4. Numeric range check
    if "TransactionAmt" in tx_df.columns:
        results.append(check_numeric_ranges(tx_df, amount_col="TransactionAmt", is_critical=True))

    # 5. Transaction null patterns
    results.append(check_null_patterns(tx_df, table_name="transaction", flag_threshold_pct=95.0, is_critical=False))

    # 6. Identity checks if provided
    if id_df is not None:
        results.append(check_required_columns(id_df, ["TransactionID"], table_name="identity", is_critical=True))
        results.append(check_key_uniqueness(id_df, key_col="TransactionID", table_name="identity", is_critical=True))
        results.append(check_null_patterns(id_df, table_name="identity", flag_threshold_pct=95.0, is_critical=False))

    # Aggregate
    critical_failures = [r for r in results if r.is_critical and not r.passed]
    non_critical_warnings = [r for r in results if not r.is_critical and not r.passed]
    overall_passed = len(critical_failures) == 0

    summary = {
        "overall_status": "PASSED" if overall_passed else "FAILED",
        "total_checks": len(results),
        "passed_count": len([r for r in results if r.passed]),
        "critical_failures_count": len(critical_failures),
        "warnings_count": len(non_critical_warnings),
        "checks": [
            {
                "name": r.check_name,
                "passed": r.passed,
                "is_critical": r.is_critical,
                "message": r.message,
                "details": r.details,
            }
            for r in results
        ],
    }

    # Save reports if output_dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "data_quality_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_file = out_path / "data_quality_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# BankGuard — Data Quality Report\n\n")
            f.write(f"- **Overall Status:** `{summary['overall_status']}`\n")
            f.write(f"- **Total Checks:** {summary['total_checks']}\n")
            f.write(f"- **Passed Checks:** {summary['passed_count']}\n")
            f.write(f"- **Critical Failures:** {summary['critical_failures_count']}\n")
            f.write(f"- **Non-Critical Warnings:** {summary['warnings_count']}\n\n")

            f.write("## Check Results\n\n")
            f.write("| Check Name | Critical? | Status | Details |\n")
            f.write("|---|:---:|:---:|---|\n")
            for c in summary["checks"]:
                status_icon = "PASS" if c["passed"] else ("FAIL" if c["is_critical"] else "WARN")
                crit_icon = "Yes" if c["is_critical"] else "No"
                f.write(f"| `{c['name']}` | {crit_icon} | **{status_icon}** | {c['message']} |\n")

    if fail_on_critical and not overall_passed:
        failure_messages = [f"- {r.check_name}: {r.message}" for r in critical_failures]
        raise DataQualityError("Critical data quality checks failed:\n" + "\n".join(failure_messages))

    return summary
