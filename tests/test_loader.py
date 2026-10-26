"""Unit tests for BankGuard data loader module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from bankguard.data.loader import (
    DatasetNotFoundError,
    load_config,
    load_table,
    load_train_transaction,
    load_train_identity,
    load_training_data,
    reduce_memory_usage,
)


@pytest.fixture
def sample_config(tmp_path: Path):
    """Creates a temporary config and mock CSV files."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    tx_df = pd.DataFrame({
        "TransactionID": [1001, 1002, 1003],
        "isFraud": [0, 1, 0],
        "TransactionAmt": [50.5, 120.0, 99.9],
        "card1": [1000, 2000, 3000],
    })
    id_df = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "DeviceType": ["mobile", "desktop"],
        "id_01": [-5.0, 0.0],
    })

    tx_file = data_dir / "train_transaction.csv"
    id_file = data_dir / "train_identity.csv"
    tx_df.to_csv(tx_file, index=False)
    id_df.to_csv(id_file, index=False)

    cfg = {
        "data": {
            "raw_dir": str(data_dir),
            "files": {
                "train_transaction": str(tx_file),
                "train_identity": str(id_file),
                "missing_table": str(data_dir / "non_existent.csv"),
            },
            "loading": {
                "optimize_memory": True,
            },
        }
    }

    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    return cfg, cfg_file


def test_load_config_default():
    """Default config should load successfully from project root."""
    cfg = load_config()
    assert "data" in cfg
    assert "files" in cfg["data"]
    assert "train_transaction" in cfg["data"]["files"]


def test_load_config_missing_file():
    """Should raise DatasetNotFoundError for non-existent config path."""
    with pytest.raises(DatasetNotFoundError):
        load_config("non_existent_config.yaml")


def test_reduce_memory_usage():
    """Tests numeric downcasting and precision preservation."""
    df = pd.DataFrame({
        "int_col": np.array([1, 100, 200], dtype=np.int64),
        "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
    })
    reduced_df = reduce_memory_usage(df)
    assert reduced_df["float_col"].dtype == np.float32
    assert reduced_df["int_col"].dtype == np.uint8
    assert reduced_df["float_col"].iloc[0] == pytest.approx(1.5)


def test_load_table_success(sample_config):
    """Loads valid mock data and checks dimensions."""
    cfg, _ = sample_config
    df = load_table("train_transaction", config=cfg, verbose=False)
    assert df.shape == (3, 4)
    assert "TransactionID" in df.columns
    assert "isFraud" in df.columns


def test_load_table_nrows(sample_config):
    """Respects nrows parameter."""
    cfg, _ = sample_config
    df = load_table("train_transaction", config=cfg, nrows=2, verbose=False)
    assert df.shape == (2, 4)


def test_load_table_missing_file_raises_error(sample_config):
    """Raises informative DatasetNotFoundError when file is absent."""
    cfg, _ = sample_config
    with pytest.raises(DatasetNotFoundError) as exc_info:
        load_table("missing_table", config=cfg, verbose=False)

    assert "Required dataset file 'missing_table' was not found" in str(exc_info.value)


def test_load_training_data_pair(sample_config):
    """load_training_data loads both transaction and identity tables."""
    cfg, _ = sample_config
    tx_df, id_df = load_training_data(config=cfg, verbose=False)
    assert tx_df.shape == (3, 4)
    assert id_df.shape == (2, 3)
