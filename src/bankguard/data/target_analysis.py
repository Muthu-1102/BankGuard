"""Target analysis module for BankGuard (Task 3.1).

Computes class counts, imbalance ratios, verification assertions,
generates publication-quality distribution visualizations, and documents modeling implications.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from bankguard.data.loader import load_train_transaction


@dataclass
class TargetAnalysisResult:
    """Class distribution metrics."""
    total_rows: int
    fraud_count: int
    legit_count: int
    fraud_percentage: float
    legit_percentage: float
    imbalance_ratio: float  # legit to fraud
    counts_match_total: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "fraud_count": self.fraud_count,
            "legit_count": self.legit_count,
            "fraud_percentage": round(self.fraud_percentage, 4),
            "legit_percentage": round(self.legit_percentage, 4),
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "counts_match_total": self.counts_match_total,
        }


def analyze_target(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    figures_dir: Optional[Path | str] = "report/figures",
    output_dir: Optional[Path | str] = "report",
    verbose: bool = True,
) -> TargetAnalysisResult:
    """Analyzes the supervised target column distribution and generates plots/reports."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    total_rows = len(df)
    counts = df[target_col].value_counts().to_dict()
    proportions = df[target_col].value_counts(normalize=True).to_dict()

    fraud_count = int(counts.get(1, 0))
    legit_count = int(counts.get(0, 0))
    fraud_pct = proportions.get(1, 0.0) * 100
    legit_pct = proportions.get(0, 0.0) * 100

    # Invariant verification: counts must add up to total rows
    counts_match = (fraud_count + legit_count) == total_rows
    if not counts_match:
        raise ValueError(
            f"Target verification failed: sum of fraud ({fraud_count:,}) and "
            f"legit ({legit_count:,}) does not equal total rows ({total_rows:,})."
        )

    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")

    result = TargetAnalysisResult(
        total_rows=total_rows,
        fraud_count=fraud_count,
        legit_count=legit_count,
        fraud_percentage=fraud_pct,
        legit_percentage=legit_pct,
        imbalance_ratio=imbalance_ratio,
        counts_match_total=counts_match,
    )

    if verbose:
        print("\nTarget Distribution Analysis:")
        print("-" * 55)
        print(f"  Total transactions  : {result.total_rows:,}")
        print(f"  Legitimate (0)      : {result.legit_count:,} ({result.legit_percentage:.2f}%)")
        print(f"  Fraudulent (1)      : {result.fraud_count:,} ({result.fraud_percentage:.2f}%)")
        print(f"  Imbalance ratio     : {result.imbalance_ratio:.1f} : 1 (legitimate per fraud)")
        print(f"  Verification check  : {'PASSED (counts match total)' if counts_match else 'FAILED'}")
        print("-" * 55)

    # Generate visualization
    if figures_dir:
        fig_path = Path(figures_dir)
        fig_path.mkdir(parents=True, exist_ok=True)
        plot_target_distribution(result, fig_path / "target_distribution.png")

    # Generate Markdown & JSON report
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_target_report(result, out_path)

    return result


def plot_target_distribution(result: TargetAnalysisResult, output_file: Path | str):
    """Generates a high-quality dual-panel plot of the target distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    colors = ["#2b5c8f", "#d9534f"]

    # 1. Bar Chart with exact counts and percentages
    labels = ["Legitimate (0)", "Fraud (1)"]
    counts = [result.legit_count, result.fraud_count]
    bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="#333", linewidth=1)

    for bar, pct in zip(bars, [result.legit_percentage, result.fraud_percentage]):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (result.total_rows * 0.015),
            f"{yval:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_title("Transaction Count by Class", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Transactions", fontsize=11)
    ax1.set_ylim(0, result.total_rows * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Donut Chart showing proportion
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.2f%%",
        startangle=140,
        pctdistance=0.75,
        explode=(0, 0.1),
        wedgeprops=dict(width=0.4, edgecolor="#333"),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax2.set_title("Class Imbalance Ratio (~27.6 : 1)", fontsize=12, fontweight="bold", pad=12)

    plt.suptitle("BankGuard — IEEE-CIS Fraud Detection Target Distribution (`isFraud`)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {Path(output_file).resolve()}")


def save_target_report(result: TargetAnalysisResult, output_dir: Path):
    """Saves Markdown and JSON reports detailing target imbalance and modeling implications."""
    json_path = output_dir / "target_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    md_path = output_dir / "target_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BankGuard — Target Distribution Analysis (`isFraud`)\n\n")
        f.write("## 1. Class Distribution Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Total Training Transactions | **{result.total_rows:,}** |\n")
        f.write(f"| Legitimate Transactions (`isFraud = 0`) | **{result.legit_count:,}** |\n")
        f.write(f"| Fraudulent Transactions (`isFraud = 1`) | **{result.fraud_count:,}** |\n")
        f.write(f"| Legitimate Percentage | **{result.legit_percentage:.4f}%** |\n")
        f.write(f"| Fraud Percentage | **{result.fraud_percentage:.4f}%** |\n")
        f.write(f"| Imbalance Ratio | **{result.imbalance_ratio:.2f} : 1** (legitimate per fraud) |\n")
        f.write(f"| Invariant Sum Check (`0 + 1 == total`) | **{'PASSED' if result.counts_match_total else 'FAILED'}** |\n\n")

        f.write("## 2. Visual Representation\n\n")
        f.write("![Target Distribution](figures/target_distribution.png)\n\n")

        f.write("## 3. Modeling and Evaluation Implications\n\n")
        f.write("1. **Accuracy is Misleading:** A naive model predicting `0` (legitimate) for all rows achieves **96.50% accuracy** while detecting exactly **0% of fraud**. Accuracy must never be used as the primary optimization or selection metric.\n")
        f.write("2. **Primary Evaluation Metrics:** Models must be compared using **PR-AUC (Average Precision)**, **Recall@Precision**, **ROC-AUC**, and the full confusion matrix.\n")
        f.write("3. **Split Strategy:** All cross-validation and train/validation splits must use **stratified sampling** (`StratifiedKFold` / `stratify=y`) to maintain the exact ~3.5% fraud prevalence across partitions.\n")
        f.write("4. **Decision Thresholds:** Default threshold 0.5 is almost certainly suboptimal for operational deployment; calibrated thresholds balancing review queue capacity against missed fraud costs are required (Phase 6).\n")

    print(f"Reports saved to:")
    print(f"  Markdown: {md_path.resolve()}")
    print(f"  JSON    : {json_path.resolve()}")


def main():
    """CLI to analyze target distribution on train_transaction."""
    print("Loading train_transaction data for target analysis...")
    tx_df = load_train_transaction()
    analyze_target(tx_df)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
