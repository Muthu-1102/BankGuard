"""Unit tests for verify_raw_data module."""

from pathlib import Path
from bankguard.data.verify_raw_data import REQUIRED_FILES, verify_raw_data


def test_verify_raw_data_fails_on_empty_dir(tmp_path: Path):
    """Should return False when required files are missing."""
    assert verify_raw_data(tmp_path) is False


def test_verify_raw_data_succeeds_when_all_required_files_present(tmp_path: Path):
    """Should return True when all required files exist."""
    for filename in REQUIRED_FILES:
        (tmp_path / filename).write_text("test")

    assert verify_raw_data(tmp_path) is True
