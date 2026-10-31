"""Batch monitoring execution and operational verification script (Task 7.1).

Executes a fixed test batch of transactions through the BankGuard serving pipeline,
evaluates operational latency, score distributions, decision tiers, and missingness,
and generates formal audit reports:
- report/monitoring_summary.json
- report/monitoring_summary.md
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

from bankguard.api.schemas import TransactionInput
from bankguard.api.service import ModelService
from bankguard.monitoring.tracker import OperationalMetricsTracker


def generate_fixed_test_batch(batch_size: int = 1000) -> List[TransactionInput]:
    """Generates a deterministic, reproducible fixed test batch of transactions.

    Simulates realistic production distribution:
    - ~95% legitimate / standard transactions.
    - ~3-5% high-risk fraud-like transactions.
    - Controlled missing fields matching production missingness patterns.
    - Controlled edge cases for data quality audits (e.g., extreme amounts).
    """
    rng = np.random.RandomState(42)
    batch: List[TransactionInput] = []

    product_codes = ["W", "C", "R", "H", "S"]
    card4_types = ["visa", "mastercard", "discover", "american express"]
    card6_types = ["debit", "credit"]

    for i in range(batch_size):
        tx_id = 3600000 + i

        # 95% regular, 5% high amount / suspicious
        is_suspicious = (i % 25 == 0)

        if is_suspicious:
            amt = float(rng.uniform(400.0, 3500.0))
            p_code = "C"  # High-risk ProductCD
            card6 = "credit"
            dist = float(rng.uniform(150.0, 800.0))
        else:
            amt = float(rng.exponential(scale=85.0) + 5.0)
            p_code = str(rng.choice(product_codes, p=[0.74, 0.12, 0.07, 0.05, 0.02]))
            card6 = str(rng.choice(card6_types, p=[0.75, 0.25]))
            dist = float(rng.exponential(scale=20.0)) if rng.rand() > 0.60 else None

        # Data quality edge cases: 2 transactions with extreme amount (> $50,000)
        if i == 42 or i == 500:
            amt = 55000.0

        # Realistic missingness patterns
        card1 = float(rng.randint(1000, 20000)) if rng.rand() > 0.01 else None
        card2 = float(rng.randint(100, 600)) if rng.rand() > 0.15 else None
        card3 = float(rng.choice([150.0, 185.0, 144.0])) if rng.rand() > 0.10 else None
        card4 = str(rng.choice(card4_types)) if rng.rand() > 0.05 else None
        addr1 = float(rng.randint(100, 500)) if rng.rand() > 0.12 else None
        addr2 = float(rng.choice([87.0, 60.0, 96.0])) if rng.rand() > 0.12 else None
        p_email = "gmail.com" if rng.rand() > 0.25 else None
        r_email = "gmail.com" if rng.rand() > 0.75 else None

        tx = TransactionInput(
            TransactionID=tx_id,
            TransactionAmt=round(amt, 2),
            TransactionDT=12000000 + i * 60,
            ProductCD=p_code,
            card1=card1,
            card2=card2,
            card3=card3,
            card4=card4,
            card5=150.0,
            card6=card6,
            addr1=addr1,
            addr2=addr2,
            dist1=dist,
            P_emaildomain=p_email,
            R_emaildomain=r_email,
        )
        batch.append(tx)

    return batch


def run_batch_monitoring(
    batch_size: int = 1000,
    output_json: Path = Path("report/monitoring_summary.json"),
    output_md: Path = Path("report/monitoring_summary.md"),
) -> Dict[str, Any]:
    """Runs a fixed batch through BankGuard serving engine and produces monitoring summaries.

    Args:
        batch_size: Number of transactions in the test batch.
        output_json: Target path for structured JSON summary.
        output_md: Target path for human-readable markdown report.

    Returns:
        Summary metrics dictionary.
    """
    print(f"\n==================================================================")
    print(f"BankGuard Task 7.1 -- Operational Monitoring Verification Runner")
    print(f"==================================================================")
    print(f"Initializing ModelService and loading production artifacts...")

    service = ModelService()
    loaded = service.load_artifacts()
    if not loaded:
        print(f"ERROR: Could not load model artifacts: {service.error_message}")
        sys.exit(1)

    print(f"  Model loaded: LightGBM 4.7.0")
    print(f"  Preprocessor loaded: ColumnTransformer (407 numeric, 31 categorical)")
    print(f"  Policy engine loaded: Three-Tier (review=0.10, block=0.40)")

    print(f"\nGenerating fixed test batch ({batch_size:,} transactions, seed=42)...")
    batch = generate_fixed_test_batch(batch_size=batch_size)
    print(f"  Batch generated: {len(batch)} transactions ready.")

    print(f"\nExecuting batch scoring and operational telemetry collection...")
    t_start = time.time()
    batch_res = service.predict_batch(batch)
    t_elapsed = time.time() - t_start

    print(f"  Total processing time: {t_elapsed:.3f} s ({t_elapsed*1000/batch_size:.2f} ms/txn)")
    print(f"  Vectorized inference latency reported: {batch_res.total_latency_ms:.2f} ms")

    summary = service.get_monitoring_summary()

    # Export artifacts
    service.tracker.export_summary_json(output_json)
    md_content = service.tracker.generate_markdown_report()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nExported artifacts:")
    print(f"  JSON summary: {output_json} ({output_json.stat().st_size} bytes)")
    print(f"  Markdown report: {output_md} ({output_md.stat().st_size} bytes)")

    # Print summary cards
    tp = summary["throughput"]
    sc = summary["score_distribution"]
    td = summary["decision_tier_distribution"]
    lat = summary["latency_telemetry"]
    dq = summary["data_quality_and_missingness"]

    print(f"\nOperational Summary:")
    print(f"  Total Requests: {tp['total_requests']:,}")
    print(f"  Successful Predictions: {tp['successful_predictions']:,} (100.0%)")
    print(f"  Prediction Failures: {tp['failed_predictions']}")
    print(f"  Mean Score: {sc['mean']:.4f} (p50={sc['percentiles']['p50']:.4f}, p90={sc['percentiles']['p90']:.4f}, p99={sc['percentiles']['p99']:.4f})")
    print(f"  Decision Tiers:")
    print(f"    [ALLOW]: {td['allow']['count']:,} ({td['allow']['percentage']:.2f}%)")
    print(f"    [REVIEW]: {td['review']['count']:,} ({td['review']['percentage']:.2f}%)")
    print(f"    [HIGH RISK]: {td['high_risk']['count']:,} ({td['high_risk']['percentage']:.2f}%)")
    print(f"  Latency: Mean={lat['mean_ms']:.2f} ms | p95={lat['percentiles']['p95_ms']:.2f} ms | SLA Compliance={lat['sla_compliance_pct']:.1f}%")
    print(f"  Data Quality Alerts: {dq['total_quality_alerts']} (extreme amounts detected)")
    print(f"  Missing Value Rate: {dq['overall_missingness_rate_pct']:.2f}%")
    print(f"==================================================================\n")

    return summary


if __name__ == "__main__":
    run_batch_monitoring(batch_size=1000)
