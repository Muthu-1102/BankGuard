"""Schema inventory and data card metrics generation module.

Computes exact target statistics, duplicate counts, identity match rate,
and generates the local schema inventory for data_card.md and data_dictionary.md.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def compute_dataset_stats(raw_dir: Path | str = "data/raw/ieee_cis") -> Dict[str, Any]:
    """Computes exact statistics for data_card.md from the raw CSVs."""
    raw_path = Path(raw_dir)
    train_tx_path = raw_path / "train_transaction.csv"
    train_id_path = raw_path / "train_identity.csv"

    print("Reading target and keys from train_transaction.csv...")
    tx_df = pd.read_csv(train_tx_path, usecols=["TransactionID", "isFraud"])
    total_tx_rows = len(tx_df)
    fraud_counts = tx_df["isFraud"].value_counts().to_dict()
    fraud_count = int(fraud_counts.get(1, 0))
    non_fraud_count = int(fraud_counts.get(0, 0))
    fraud_pct = (fraud_count / total_tx_rows) * 100
    tx_dups = int(tx_df["TransactionID"].duplicated().sum())

    print("Reading keys from train_identity.csv...")
    id_df = pd.read_csv(train_id_path, usecols=["TransactionID"])
    total_id_rows = len(id_df)
    id_dups = int(id_df["TransactionID"].duplicated().sum())

    tx_ids = set(tx_df["TransactionID"])
    id_ids = set(id_df["TransactionID"])
    matches = len(tx_ids.intersection(id_ids))
    match_rate = (matches / total_tx_rows) * 100

    stats = {
        "total_train_transactions": total_tx_rows,
        "total_train_identities": total_id_rows,
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "duplicate_tx_ids_transaction": tx_dups,
        "duplicate_tx_ids_identity": id_dups,
        "identity_match_count": matches,
        "identity_match_rate_pct": round(match_rate, 2),
    }
    print(f"Stats computed: {json.dumps(stats, indent=2)}")
    return stats


def generate_column_inventory(raw_dir: Path | str = "data/raw/ieee_cis", output_dir: Path | str = "report"):
    """Profiles all columns in train_transaction and train_identity and outputs a schema inventory."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []

    for filename in ["train_transaction.csv", "train_identity.csv"]:
        filepath = raw_path / filename
        print(f"Profiling columns in {filename}...")
        # Read in chunks to compute exact nulls and types without high memory pressure
        chunk_iter = pd.read_csv(filepath, chunksize=100000, low_memory=False)
        total_rows = 0
        null_counts = None
        dtypes = None

        for i, chunk in enumerate(chunk_iter):
            total_rows += len(chunk)
            if null_counts is None:
                null_counts = chunk.isnull().sum()
                dtypes = chunk.dtypes
            else:
                null_counts += chunk.isnull().sum()

        # Read first 100 rows for sample values
        sample_df = pd.read_csv(filepath, nrows=100)

        for col in sample_df.columns:
            dtype_str = str(dtypes[col])
            null_cnt = int(null_counts[col])
            null_pct = round((null_cnt / total_rows) * 100, 2)
            sample_val = sample_df[col].dropna().iloc[0] if not sample_df[col].dropna().empty else "None"

            # Classify status and use
            if col == "isFraud":
                status = "target"
                model_use = "target"
                reason = "Primary supervised binary classification label; never an input feature."
            elif col == "TransactionID":
                status = "identifier"
                model_use = "join_key"
                reason = "Unique join and traceability key; exclude from model features."
            elif col.startswith("V"):
                status = "anonymized"
                model_use = "candidate"
                reason = "Vesta engineered rich transaction feature; obfuscated numeric."
            elif col.startswith("C") and col[1:].isdigit():
                status = "anonymized"
                model_use = "candidate"
                reason = "Counting/frequency feature; obfuscated numeric."
            elif col.startswith("D") and col[1:].isdigit():
                status = "anonymized"
                model_use = "candidate"
                reason = "Time delta feature; review for prediction-time availability."
            elif col.startswith("M") and col[1:].isdigit():
                status = "anonymized"
                model_use = "candidate"
                reason = "Match feature; categorical flag (T/F or null)."
            elif col.startswith("id_"):
                status = "anonymized"
                model_use = "candidate"
                reason = "Anonymized identity attribute; present for subset of transactions."
            elif col in ("DeviceType", "DeviceInfo"):
                status = "documented"
                model_use = "candidate"
                reason = "Device metadata; high cardinality requires normalization."
            elif col.startswith("card"):
                status = "documented"
                model_use = "candidate"
                reason = "Payment card attributes (brand, type, bank code)."
            elif col.startswith("addr"):
                status = "documented"
                model_use = "candidate"
                reason = "Address geographic billing/shipping region code."
            elif col.startswith("dist"):
                status = "documented"
                model_use = "candidate"
                reason = "Distance measure; substantial missingness."
            elif "emaildomain" in col:
                status = "documented"
                model_use = "candidate"
                reason = "Purchaser/recipient email domain category."
            elif col in ("TransactionAmt", "ProductCD", "TransactionDT"):
                status = "documented"
                model_use = "candidate"
                reason = f"Core transaction attribute ({col})."
            else:
                status = "unknown"
                model_use = "review_required"
                reason = "Unclassified column; inspect distribution before inclusion."

            records.append({
                "file_name": filename,
                "column_name": col,
                "dtype": dtype_str,
                "null_count": null_cnt,
                "null_percentage": null_pct,
                "sample_value": str(sample_val)[:50],
                "dictionary_status": status,
                "model_use": model_use,
                "reason": reason,
            })

    inventory_df = pd.DataFrame(records)
    csv_out = out_path / "schema_inventory.csv"
    inventory_df.to_csv(csv_out, index=False)
    print(f"Saved {len(records)} columns inventory to {csv_out.resolve()}")

    # Save summary markdown
    md_out = out_path / "schema_inventory_summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# BankGuard — Local Schema Inventory Summary\n\n")
        f.write(f"- **Total Documented Columns:** {len(records)}\n")
        f.write(f"- **Train Transaction Columns:** {len([r for r in records if r['file_name'] == 'train_transaction.csv'])}\n")
        f.write(f"- **Train Identity Columns:** {len([r for r in records if r['file_name'] == 'train_identity.csv'])}\n")
        f.write(f"- **Full Inventory CSV:** [`report/schema_inventory.csv`](schema_inventory.csv)\n\n")

        f.write("## Status Breakdown\n\n")
        status_counts = inventory_df["dictionary_status"].value_counts().to_dict()
        for s, c in status_counts.items():
            f.write(f"- `{s}`: {c} columns\n")

        f.write("\n## Model Use Breakdown\n\n")
        use_counts = inventory_df["model_use"].value_counts().to_dict()
        for u, c in use_counts.items():
            f.write(f"- `{u}`: {c} columns\n")

    return inventory_df


if __name__ == "__main__":
    stats = compute_dataset_stats()
    generate_column_inventory()
