"""Unit tests for inspect_dataset module."""

import json
from pathlib import Path
import pandas as pd
from bankguard.data.inspect_dataset import count_lines_fast, generate_inventory_report, inspect_file


def test_count_lines_fast(tmp_path: Path):
    """Test fast line counting excluding header."""
    file = tmp_path / "test.csv"
    file.write_text("col1,col2\n1,a\n2,b\n3,c\n")
    assert count_lines_fast(file) == 3


def test_inspect_file(tmp_path: Path):
    """Test inspecting a single CSV file."""
    df = pd.DataFrame({
        "TransactionID": [101, 102],
        "isFraud": [0, 1],
        "TransactionAmt": [50.0, 120.5],
        "ProductCD": ["W", "H"],
    })
    filepath = tmp_path / "train_transaction.csv"
    df.to_csv(filepath, index=False)

    info = inspect_file(filepath)
    assert info["filename"] == "train_transaction.csv"
    assert info["row_count"] == 2
    assert info["column_count"] == 4
    assert info["column_families"]["isFraud"] is True
    assert info["column_families"]["TransactionID"] is True


def test_generate_inventory_report(tmp_path: Path):
    """Test full report generation on mock directory."""
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "report"
    raw_dir.mkdir()

    for name in ["train_transaction.csv", "train_identity.csv"]:
        df = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]} if "train_transaction" in name else {"TransactionID": [1, 2]})
        df.to_csv(raw_dir / name, index=False)

    report_data = generate_inventory_report(raw_dir=raw_dir, output_dir=report_dir)

    assert (report_dir / "data_inventory_report.md").is_file()
    assert (report_dir / "data_inventory_report.json").is_file()
    assert report_data["summary"]["total_files"] == 2
