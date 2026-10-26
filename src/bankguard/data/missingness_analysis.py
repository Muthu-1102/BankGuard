"""Missingness analysis module for BankGuard (Task 3.2).

Ranks columns by missingness percentage, compares missingness across fraud vs legitimate
transactions to uncover predictive missingness signals, and generates publication plots and reports.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bankguard.data.join import safe_join_transaction_identity
from bankguard.data.loader import load_training_data


def assign_missingness_tier(null_pct: float) -> str:
    """Classifies missingness into standard operational tiers."""
    if null_pct == 0.0:
        return "Complete (0%)"
    elif null_pct < 20.0:
        return "Low (<20%)"
    elif null_pct < 50.0:
        return "Moderate (20-50%)"
    elif null_pct < 80.0:
        return "High (50-80%)"
    else:
        return "Extreme (>=80%)"


def compute_missingness_table(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    key_col: str = "TransactionID",
) -> pd.DataFrame:
    """Computes comprehensive missingness metrics for every column in the dataset.

    Args:
        df: DataFrame containing features and target.
        target_col: Binary target column name.
        key_col: Primary key column name.

    Returns:
        Sorted DataFrame of column missingness statistics.
    """
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("DataFrame is empty.")

    has_target = target_col in df.columns
    if has_target:
        fraud_mask = df[target_col] == 1
        legit_mask = df[target_col] == 0
        fraud_total = int(fraud_mask.sum())
        legit_total = int(legit_mask.sum())
    else:
        fraud_total = legit_total = 0

    records = []
    # Evaluate all feature columns
    cols_to_eval = [c for c in df.columns if c not in (target_col, key_col)]

    for col in cols_to_eval:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 4)
        tier = assign_missingness_tier(null_pct)

        if has_target and fraud_total > 0 and legit_total > 0:
            fraud_nulls = int(df.loc[fraud_mask, col].isnull().sum())
            legit_nulls = int(df.loc[legit_mask, col].isnull().sum())
            fraud_null_pct = round((fraud_nulls / fraud_total) * 100, 4)
            legit_null_pct = round((legit_nulls / legit_total) * 100, 4)
            diff_pct = round(fraud_null_pct - legit_null_pct, 4)
            abs_diff_pct = round(abs(diff_pct), 4)
        else:
            fraud_nulls = legit_nulls = 0
            fraud_null_pct = legit_null_pct = diff_pct = abs_diff_pct = 0.0

        records.append({
            "column_name": col,
            "total_nulls": null_count,
            "overall_null_pct": null_pct,
            "fraud_null_pct": fraud_null_pct,
            "legit_null_pct": legit_null_pct,
            "diff_pct": diff_pct,
            "abs_diff_pct": abs_diff_pct,
            "missingness_tier": tier,
        })

    result_df = pd.DataFrame(records)
    # Sort by overall missingness descending
    return result_df.sort_values(by="overall_null_pct", ascending=False).reset_index(drop=True)


def plot_missingness_analysis(
    missing_df: pd.DataFrame,
    output_path: Path | str = "report/figures/missingness_distribution.png",
):
    """Generates a dual-panel figure illustrating missingness tiers and informative missingness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Panel 1: Missingness Tier Distribution
    tier_order = [
        "Complete (0%)",
        "Low (<20%)",
        "Moderate (20-50%)",
        "High (50-80%)",
        "Extreme (>=80%)",
    ]
    tier_counts = missing_df["missingness_tier"].value_counts().reindex(tier_order, fill_value=0)
    colors = ["#28a745", "#5bc0de", "#f0ad4e", "#d9534f", "#6f42c1"]

    bars = ax1.bar(tier_order, tier_counts.values, color=colors, edgecolor="#333", width=0.6)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (len(missing_df) * 0.01),
            f"{int(height)}\n({height/len(missing_df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_title("Column Distribution by Missingness Tier", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Columns", fontsize=11)
    ax1.set_ylim(0, max(tier_counts.values) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=25)

    # Panel 2: Top 15 Informative Missingness Features (|Fraud Null % - Legit Null %|)
    top_diff = missing_df.sort_values(by="abs_diff_pct", ascending=False).head(15).iloc[::-1]

    y_pos = np.arange(len(top_diff))
    bar_width = 0.35

    ax2.barh(y_pos + bar_width / 2, top_diff["fraud_null_pct"], height=bar_width, label="Fraud", color="#d9534f", alpha=0.85)
    ax2.barh(y_pos - bar_width / 2, top_diff["legit_null_pct"], height=bar_width, label="Legitimate", color="#2b5c8f", alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_diff["column_name"], fontsize=9)
    ax2.set_xlabel("Missing Percentage (%)", fontsize=11)
    ax2.set_title("Top 15 Columns with Informative Missingness", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("BankGuard — Missingness Profile & Differential Signal Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {out_file.resolve()}")


def generate_missingness_report(
    missing_df: pd.DataFrame,
    total_rows: int,
    output_dir: Path | str = "report",
) -> Dict[str, Any]:
    """Generates detailed Markdown and JSON reports from missingness table."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save CSV ranking
    csv_path = out_path / "missingness_ranking.csv"
    missing_df.to_csv(csv_path, index=False)

    # Compute summary stats
    tier_order = [
        "Complete (0%)",
        "Low (<20%)",
        "Moderate (20-50%)",
        "High (50-80%)",
        "Extreme (>=80%)",
    ]
    tier_counts = missing_df["missingness_tier"].value_counts().reindex(tier_order, fill_value=0).to_dict()
    top_informative = missing_df.sort_values(by="abs_diff_pct", ascending=False).head(15).to_dict(orient="records")

    summary = {
        "total_columns_analyzed": len(missing_df),
        "total_rows_analyzed": total_rows,
        "complete_columns_count": tier_counts.get("Complete (0%)", 0),
        "tier_distribution": tier_counts,
        "top_informative_columns": top_informative,
    }

    # Save JSON summary
    json_path = out_path / "missingness_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    md_path = out_path / "missingness_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BankGuard — Dataset Missingness & Signal Analysis\n\n")
        f.write(f"- **Total Features Analyzed:** {len(missing_df)}\n")
        f.write(f"- **Total Rows:** {total_rows:,}\n")
        f.write(f"- **Complete Features (0% null):** {summary['complete_columns_count']} ({summary['complete_columns_count']/len(missing_df)*100:.1f}%)\n")
        f.write(f"- **Full Missingness Ranking:** [`report/missingness_ranking.csv`](missingness_ranking.csv)\n\n")

        f.write("## 1. Missingness Tier Breakdown\n\n")
        f.write("| Missingness Tier | Column Count | Percentage of Feature Space | Preprocessing Recommendation |\n")
        f.write("|---|---:|---:|---|\n")
        f.write(f"| `Complete (0%)` | {tier_counts['Complete (0%)']} | {tier_counts['Complete (0%)']/len(missing_df)*100:.1f}% | Ready for baseline / tree models without imputation. |\n")
        f.write(f"| `Low (<20%)` | {tier_counts['Low (<20%)']} | {tier_counts['Low (<20%)']/len(missing_df)*100:.1f}% | Simple median/most-frequent imputation or native tree handling. |\n")
        f.write(f"| `Moderate (20-50%)` | {tier_counts['Moderate (20-50%)']} | {tier_counts['Moderate (20-50%)']/len(missing_df)*100:.1f}% | Tree native routing; add missingness indicator flag for linear baseline. |\n")
        f.write(f"| `High (50-80%)` | {tier_counts['High (50-80%)']} | {tier_counts['High (50-80%)']/len(missing_df)*100:.1f}% | Identity features & distance; retain if informative, test exclusion in baseline. |\n")
        f.write(f"| `Extreme (>=80%)` | {tier_counts['Extreme (>=80%)']} | {tier_counts['Extreme (>=80%)']/len(missing_df)*100:.1f}% | Rare identity codes & device fields; filter out uninformative near-empty features. |\n\n")

        f.write("## 2. Visual Representation\n\n")
        f.write("![Missingness Distribution](figures/missingness_distribution.png)\n\n")

        f.write("## 3. Top Informative Missingness Features\n\n")
        f.write("Columns where missingness rate strongly differs between fraudulent and legitimate transactions, serving as a direct predictive signal:\n\n")
        f.write("| Column Name | Overall Null % | Fraud Null % | Legit Null % | Discrepancy (Fraud - Legit) | Tier |\n")
        f.write("|---|---:|---:|---:|---:|:---:|\n")
        for col_info in top_informative:
            f.write(
                f"| `{col_info['column_name']}` | {col_info['overall_null_pct']:.2f}% | "
                f"{col_info['fraud_null_pct']:.2f}% | {col_info['legit_null_pct']:.2f}% | "
                f"**{col_info['diff_pct']:+.2f}%** | `{col_info['missingness_tier']}` |\n"
            )

        f.write("\n## 4. Key Findings and Modeling Decisions\n\n")
        f.write("1. **Missingness is Informative (Not Noise):** Several key attributes (e.g. `dist2`, identity device attributes, and `R_emaildomain`) display substantial differences in missingness between fraud and non-fraud transactions. Missing values must not be blindly imputed or discarded.\n")
        f.write("2. **Tree-Based Advantage:** Modern gradient boosted trees (LightGBM, XGBoost) natively learn optimal branching directions for `NaN` values, naturally leveraging informative missingness without data distortion.\n")
        f.write("3. **Linear Baseline Requirement:** For linear models (Logistic Regression), adding a `_isna` binary indicator feature alongside median/mode imputation preserves the missingness signal without data loss.\n")

    print(f"Reports saved to:")
    print(f"  Markdown: {md_path.resolve()}")
    print(f"  JSON    : {json_path.resolve()}")
    print(f"  CSV     : {csv_path.resolve()}")

    return summary


def run_full_missingness_pipeline(output_dir: Path | str = "report"):
    """Loads joined training data, computes missingness ranking, generates plots and reports."""
    print("Loading training dataset for missingness analysis...")
    tx_df, id_df = load_training_data()
    joined_df, _ = safe_join_transaction_identity(tx_df, id_df, verbose=False)

    print("Computing missingness statistics across all columns...")
    missing_df = compute_missingness_table(joined_df, target_col="isFraud")

    print("Generating missingness visualizations...")
    plot_missingness_analysis(missing_df, output_path=Path(output_dir) / "figures" / "missingness_distribution.png")

    print("Compiling missingness report...")
    summary = generate_missingness_report(missing_df, total_rows=len(joined_df), output_dir=output_dir)

    print(f"\nMissingness analysis complete for {len(missing_df)} features.")
    return missing_df, summary


if __name__ == "__main__":
    run_full_missingness_pipeline()
