"""Script to execute data quality checks on raw training data and generate reports."""

import sys
import time
from bankguard.data.loader import load_training_data
from bankguard.data.quality_checks import run_data_quality_suite


def main():
    print("Executing BankGuard Data Quality Checks on training dataset...")
    start_time = time.time()

    tx_df, id_df = load_training_data()
    print("Running quality check suite...")
    summary = run_data_quality_suite(
        tx_df=tx_df,
        id_df=id_df,
        is_training=True,
        fail_on_critical=False,
        output_dir="report",
    )

    elapsed = time.time() - start_time
    print(f"\nQuality checks completed in {elapsed:.2f}s.")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Passed Checks : {summary['passed_count']} / {summary['total_checks']}")
    print(f"Critical Fails: {summary['critical_failures_count']}")
    print(f"Warnings      : {summary['warnings_count']}")

    print("\nReports written to:")
    print("  Markdown: report/data_quality_report.md")
    print("  JSON    : report/data_quality_report.json")

    return 0 if summary["overall_status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
