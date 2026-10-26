"""Data loading module for BankGuard.

Provides configurable, memory-optimized loading of IEEE-CIS Fraud Detection
transaction and identity tables with error handling for missing files.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required dataset file is missing with clear actionable instructions."""
    pass


def get_project_root() -> Path:
    """Returns the repository root directory assuming standard structure."""
    # This file is at src/bankguard/data/loader.py -> root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads dataset configuration from YAML file."""
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "data_config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.is_file():
        raise DatasetNotFoundError(
            f"Configuration file not found at: {target_path}.\n"
            "Please ensure configs/data_config.yaml exists in the project root."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcasts numeric types to reduce RAM footprint without loss of information."""
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df.columns:
        col_type = df[col].dtype

        if np.issubdtype(col_type, np.integer):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

        elif np.issubdtype(col_type, np.floating):
            # Downcast to float32 for model training efficiency
            df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if verbose:
        reduction = (1 - end_mem / start_mem) * 100
        print(f"  Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


def _resolve_file_path(relative_or_abs: str, root: Path) -> Path:
    """Resolves file path relative to root if not absolute."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else root / p


def load_table(
    table_key: str,
    config: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
    optimize_memory: Optional[bool] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loads a specific dataset table defined in configuration.

    Args:
        table_key: Key in config['data']['files'] (e.g. 'train_transaction', 'train_identity')
        config: Loaded config dictionary, or None to load default.
        nrows: Number of rows to read (useful for rapid testing).
        optimize_memory: If True, downcast numeric columns.
        verbose: If True, print progress and timings.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DatasetNotFoundError: If the requested CSV file is missing.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    files_cfg = config.get("data", {}).get("files", {})
    if table_key not in files_cfg:
        raise KeyError(f"Table key '{table_key}' not found in data configuration files section.")

    file_rel = files_cfg[table_key]
    file_path = _resolve_file_path(file_rel, root)

    if not file_path.is_file():
        raise DatasetNotFoundError(
            f"\nRequired dataset file '{table_key}' was not found at:\n"
            f"  {file_path.resolve()}\n\n"
            "Please ensure the official competition files are placed in data/raw/ieee_cis/\n"
            "as described in docs/data_card.md."
        )

    if optimize_memory is None:
        optimize_memory = config.get("data", {}).get("loading", {}).get("optimize_memory", True)

    if verbose:
        print(f"Loading '{table_key}' from: {file_path.name} (nrows={nrows or 'all'})...")

    start_time = time.time()
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)

    if optimize_memory:
        df = reduce_memory_usage(df, verbose=verbose)

    elapsed = time.time() - start_time
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    if verbose:
        print(
            f"  Successfully loaded '{table_key}': {df.shape[0]:,} rows x {df.shape[1]} columns "
            f"({mem_mb:.2f} MB) in {elapsed:.2f}s."
        )

    return df


def load_train_transaction(
    config: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
    optimize_memory: Optional[bool] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loads training transaction data."""
    return load_table("train_transaction", config=config, nrows=nrows, optimize_memory=optimize_memory, verbose=verbose)


def load_train_identity(
    config: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
    optimize_memory: Optional[bool] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loads training identity data."""
    return load_table("train_identity", config=config, nrows=nrows, optimize_memory=optimize_memory, verbose=verbose)


def load_training_data(
    config: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
    optimize_memory: Optional[bool] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both training transaction and identity tables."""
    if verbose:
        print("Loading BankGuard training dataset...")
    tx_df = load_train_transaction(config=config, nrows=nrows, optimize_memory=optimize_memory, verbose=verbose)
    id_df = load_train_identity(config=config, nrows=nrows, optimize_memory=optimize_memory, verbose=verbose)
    return tx_df, id_df


def main() -> int:
    """CLI entrypoint for data loader."""
    parser = argparse.ArgumentParser(description="BankGuard Dataset Loader")
    parser.add_argument("--config", type=str, default=None, help="Path to data config YAML")
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows to load")
    parser.add_argument(
        "--table",
        type=str,
        default="train",
        choices=["train", "train_tx", "train_id", "test_tx", "test_id"],
        help="Table(s) to load",
    )
    parser.add_argument("--no-optimize", action="store_true", help="Disable numeric downcasting")

    args = parser.parse_args()
    optimize = not args.no_optimize

    try:
        cfg = load_config(args.config)
        print("Configuration successfully loaded.")

        if args.table == "train":
            tx_df, id_df = load_training_data(config=cfg, nrows=args.nrows, optimize_memory=optimize)
            print("\nDataset Loading Summary:")
            print(f"  Transaction table dimensions : {tx_df.shape[0]:,} rows x {tx_df.shape[1]} columns")
            print(f"  Identity table dimensions    : {id_df.shape[0]:,} rows x {id_df.shape[1]} columns")
        elif args.table == "train_tx":
            df = load_train_transaction(config=cfg, nrows=args.nrows, optimize_memory=optimize)
            print(f"\nDimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
        elif args.table == "train_id":
            df = load_train_identity(config=cfg, nrows=args.nrows, optimize_memory=optimize)
            print(f"\nDimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
        elif args.table == "test_tx":
            df = load_table("test_transaction", config=cfg, nrows=args.nrows, optimize_memory=optimize)
            print(f"\nDimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
        elif args.table == "test_id":
            df = load_table("test_identity", config=cfg, nrows=args.nrows, optimize_memory=optimize)
            print(f"\nDimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")

        print("\nLoader command completed successfully.")
        return 0

    except DatasetNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
