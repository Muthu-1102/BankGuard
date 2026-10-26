"""Raw dataset verification script for BankGuard.

Checks for the presence and accessibility of the required IEEE-CIS Fraud Detection
competition files in data/raw/ieee_cis/.
"""

import sys
from pathlib import Path

REQUIRED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]

OPTIONAL_FILES = [
    "sample_submission.csv",
]


def verify_raw_data(data_dir: Path | str = "data/raw/ieee_cis") -> bool:
    """Verifies that all required competition CSV files are present in the target directory."""
    raw_dir = Path(data_dir)
    print(f"Checking raw dataset in: {raw_dir.resolve()}")

    if not raw_dir.exists():
        print(f"ERROR: Directory does not exist: {raw_dir}")
        return False

    all_present = True
    print("\nRequired Dataset Files:")
    print("-" * 65)
    for filename in REQUIRED_FILES:
        file_path = raw_dir / filename
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  [OK]      {filename:<25} ({size_mb:8.2f} MB)")
        else:
            print(f"  [MISSING] {filename:<25}")
            all_present = False
    print("-" * 65)

    print("\nOptional Competition Files:")
    print("-" * 65)
    for filename in OPTIONAL_FILES:
        file_path = raw_dir / filename
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  [OK]      {filename:<25} ({size_mb:8.2f} MB)")
        else:
            print(f"  [NOT FOUND] {filename:<25} (optional - Kaggle leaderboard only)")
    print("-" * 65)

    if all_present:
        print("\nAll required IEEE-CIS dataset files are present and verified.")
    else:
        print("\nVerification FAILED: One or more required dataset files are missing.")
        print("Please ensure the 4 core competition files are placed in:")
        print(f"  {raw_dir.resolve()}")

    return all_present


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ieee_cis"
    success = verify_raw_data(target)
    sys.exit(0 if success else 1)
