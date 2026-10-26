"""Identity join module for BankGuard.

Provides safe left-joining of transaction and identity tables on TransactionID
with pre-join cardinality checks, post-join integrity assertions, and join metric reporting.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from bankguard.data.loader import load_training_data


class JoinCardinalityError(ValueError):
    """Raised when join cardinality expectations are violated (e.g. duplicate keys in identity table)."""
    pass


@dataclass
class JoinMetrics:
    """Detailed metrics from the table join."""
    transaction_rows: int
    identity_rows: int
    joined_rows: int
    identity_matches: int
    identity_match_rate_pct: float
    unmatched_transactions: int
    transaction_cols: int
    identity_cols: int
    joined_cols: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_rows": self.transaction_rows,
            "identity_rows": self.identity_rows,
            "joined_rows": self.joined_rows,
            "identity_matches": self.identity_matches,
            "identity_match_rate_pct": round(self.identity_match_rate_pct, 2),
            "unmatched_transactions": self.unmatched_transactions,
            "transaction_cols": self.transaction_cols,
            "identity_cols": self.identity_cols,
            "joined_cols": self.joined_cols,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def safe_join_transaction_identity(
    tx_df: pd.DataFrame,
    id_df: pd.DataFrame,
    key_col: str = "TransactionID",
    validate_cardinality: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, JoinMetrics]:
    """Safely left-joins identity features to transactions on TransactionID.

    Args:
        tx_df: Primary transaction DataFrame.
        id_df: Auxiliary identity DataFrame.
        key_col: Primary key to join on.
        validate_cardinality: If True, asserts identity keys are unique to prevent row multiplication.
        verbose: If True, prints progress and metric summaries.

    Returns:
        Tuple of (joined_df, JoinMetrics).

    Raises:
        KeyError: If key_col is absent from either DataFrame.
        JoinCardinalityError: If key_col has duplicate records in identity or transactions.
    """
    start_time = time.time()

    # 1. Pre-join validation
    if key_col not in tx_df.columns:
        raise KeyError(f"Primary join key '{key_col}' not found in transaction table.")
    if key_col not in id_df.columns:
        raise KeyError(f"Join key '{key_col}' not found in identity table.")

    if validate_cardinality:
        tx_dups = int(tx_df[key_col].duplicated().sum())
        if tx_dups > 0:
            raise JoinCardinalityError(
                f"Transaction table contains {tx_dups:,} duplicate '{key_col}' values. "
                "Keys must be unique before joining."
            )

        id_dups = int(id_df[key_col].duplicated().sum())
        if id_dups > 0:
            raise JoinCardinalityError(
                f"Identity table contains {id_dups:,} duplicate '{key_col}' values. "
                "Joining would cause an unintended 1-to-many row multiplication."
            )

    tx_rows = len(tx_df)
    id_rows = len(id_df)

    if verbose:
        print(f"Joining {tx_rows:,} transactions with {id_rows:,} identity rows on '{key_col}'...")

    # 2. Execute left join
    # Check for overlapping column names other than key_col
    overlap_cols = set(tx_df.columns).intersection(set(id_df.columns)) - {key_col}
    if overlap_cols and verbose:
        print(f"  Warning: Overlapping columns detected: {overlap_cols} (will receive suffixes).")

    joined_df = tx_df.merge(id_df, on=key_col, how="left", suffixes=("", "_id"))

    # 3. Post-join assertions (fail loudly if invariants are violated)
    assert len(joined_df) == tx_rows, (
        f"CRITICAL JOIN INVARIANT VIOLATED: Joined row count ({len(joined_df):,}) "
        f"does not equal transaction row count ({tx_rows:,}). Rows were added or dropped."
    )

    assert joined_df[key_col].duplicated().sum() == 0, (
        f"CRITICAL JOIN INVARIANT VIOLATED: Duplicate '{key_col}' values found in joined data."
    )

    assert joined_df[key_col].isnull().sum() == 0, (
        f"CRITICAL JOIN INVARIANT VIOLATED: Null '{key_col}' values introduced after join."
    )

    # 4. Compute metrics
    tx_keys = set(tx_df[key_col])
    id_keys = set(id_df[key_col])
    matches = len(tx_keys.intersection(id_keys))
    match_rate = (matches / tx_rows) * 100 if tx_rows > 0 else 0.0
    unmatched = tx_rows - matches
    elapsed = time.time() - start_time

    metrics = JoinMetrics(
        transaction_rows=tx_rows,
        identity_rows=id_rows,
        joined_rows=len(joined_df),
        identity_matches=matches,
        identity_match_rate_pct=match_rate,
        unmatched_transactions=unmatched,
        transaction_cols=tx_df.shape[1],
        identity_cols=id_df.shape[1],
        joined_cols=joined_df.shape[1],
        elapsed_seconds=elapsed,
    )

    if verbose:
        print(f"  Join completed successfully in {elapsed:.2f}s:")
        print(f"    - Preserved transaction rows : {metrics.joined_rows:,}")
        print(f"    - Matched with identity rows : {metrics.identity_matches:,} ({metrics.identity_match_rate_pct:.2f}%)")
        print(f"    - Unmatched transactions     : {metrics.unmatched_transactions:,} (identity fields set to NaN)")
        print(f"    - Final dimensions           : {joined_df.shape[0]:,} rows x {joined_df.shape[1]} columns")

    return joined_df, metrics


def save_join_report(metrics: JoinMetrics, output_dir: Path | str = "report"):
    """Saves join execution report in JSON and Markdown formats."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "join_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    md_path = out_path / "join_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BankGuard — Identity Join Report\n\n")
        f.write("- **Join Type:** `LEFT JOIN` on `TransactionID`\n")
        f.write(f"- **Execution Time:** {metrics.elapsed_seconds:.2f} seconds\n")
        f.write(f"- **Row Preservation Invariant:** Verified (Joined: {metrics.joined_rows:,} == Transactions: {metrics.transaction_rows:,})\n\n")

        f.write("## Join Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Primary Transaction Rows | {metrics.transaction_rows:,} |\n")
        f.write(f"| Available Identity Rows | {metrics.identity_rows:,} |\n")
        f.write(f"| Joined Row Count | **{metrics.joined_rows:,}** |\n")
        f.write(f"| Transactions with Identity Record | **{metrics.identity_matches:,}** |\n")
        f.write(f"| Identity Match Rate | **{metrics.identity_match_rate_pct:.2f}%** |\n")
        f.write(f"| Transactions without Identity | {metrics.unmatched_transactions:,} ({100 - metrics.identity_match_rate_pct:.2f}%) |\n")
        f.write(f"| Transaction Columns | {metrics.transaction_cols} |\n")
        f.write(f"| Identity Columns | {metrics.identity_cols} |\n")
        f.write(f"| Total Joined Columns | {metrics.joined_cols} |\n")

    print(f"\nJoin reports written to:")
    print(f"  Markdown: {md_path.resolve()}")
    print(f"  JSON    : {json_path.resolve()}")


def main():
    """CLI to execute the join and generate join report."""
    print("Loading training dataset...")
    tx_df, id_df = load_training_data()
    joined_df, metrics = safe_join_transaction_identity(tx_df, id_df)
    save_join_report(metrics)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
