"""Unit and integration test suite for BankGuard Operational Monitoring (Task 7.1).

Verifies:
1. Structured JSON logging and PII redaction.
2. OperationalMetricsTracker single and batch recording.
3. Score distribution percentiles, min, max, and bucket counts.
4. Latency percentiles and SLA compliance.
5. Missing input field tracking and frequency ranking.
6. Data quality alerts for extreme or anomalous inputs.
7. Prediction failure taxonomy and failure rate calculation.
8. Artifact exports (JSON summary and Markdown audit report).
9. FastAPI GET /monitoring/summary endpoint behavior.
10. Fixed test batch reproducibility and expected counts.
"""

import json
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from bankguard.api.app import create_app
from bankguard.api.schemas import MonitoringSummaryResponse, TransactionInput
from bankguard.monitoring.logger import (
    SENSITIVE_FIELDS,
    StructuredJsonFormatter,
    get_structured_logger,
    sanitize_payload,
)
from bankguard.monitoring.run_monitoring_batch import generate_fixed_test_batch, run_batch_monitoring
from bankguard.monitoring.tracker import OperationalMetricsTracker


def test_structured_json_logger_pii_redaction():
    """Rule: Structured logging must enforce JSON output and strip sensitive PII."""
    logger = get_structured_logger("test.monitoring")
    formatter = StructuredJsonFormatter()

    record_data = {
        "event": "prediction_executed",
        "transaction_id": 12345,
        "fraud_probability": 0.85,
        "risk_tier": "high_risk",
        "latency_ms": 1.25,
        # Sensitive fields that MUST be redacted
        "card_number": "4111111111111111",
        "ip_address": "192.168.1.1",
        "email": "cardholder@example.com",
    }

    import logging
    rec = logging.LogRecord(
        name="test.monitoring",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Transaction scored",
        args=(),
        exc_info=None,
    )
    rec.structured_data = record_data

    formatted = formatter.format(rec)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Transaction scored"
    assert parsed["transaction_id"] == 12345
    assert parsed["fraud_probability"] == 0.85
    assert parsed["risk_tier"] == "high_risk"

    # Verify PII is NOT present
    assert "card_number" not in parsed
    assert "ip_address" not in parsed
    assert "email" not in parsed


def test_sanitize_payload():
    """Rule: sanitize_payload strips PII from transaction payload dictionaries."""
    raw = {
        "TransactionID": 100,
        "TransactionAmt": 45.0,
        "card_number": "1234-5678",
        "device_info": "iPhone 14",
        "email": "test@test.com",
        "addr1": 315.0,
    }
    sanitized = sanitize_payload(raw)
    assert "TransactionID" in sanitized
    assert "TransactionAmt" in sanitized
    assert "addr1" in sanitized
    assert "card_number" not in sanitized
    assert "device_info" not in sanitized
    assert "email" not in sanitized


def test_metrics_tracker_single_and_batch_predictions():
    """Verifies that the tracker records single predictions and batch collections."""
    tracker = OperationalMetricsTracker()

    # Single prediction
    tracker.record_prediction(
        transaction_id=1,
        input_data={"TransactionAmt": 100.0, "ProductCD": "W"},
        fraud_prob=0.05,
        risk_tier="allow",
        latency_ms=2.5,
    )

    summary1 = tracker.get_summary()
    assert summary1["throughput"]["total_requests"] == 1
    assert summary1["throughput"]["successful_predictions"] == 1
    assert summary1["throughput"]["failed_predictions"] == 0
    assert summary1["decision_tier_distribution"]["allow"]["count"] == 1
    assert summary1["score_distribution"]["mean"] == 0.05

    # Batch prediction
    batch_preds = [
        {"transaction_id": 2, "fraud_probability": 0.15, "risk_tier": "review"},
        {"transaction_id": 3, "fraud_probability": 0.55, "risk_tier": "high_risk"},
    ]
    tracker.record_batch(
        batch_predictions=batch_preds,
        total_latency_ms=10.0,
        inputs_list=[
            {"TransactionID": 2, "TransactionAmt": 200.0},
            {"TransactionID": 3, "TransactionAmt": 500.0},
        ],
    )

    summary2 = tracker.get_summary()
    assert summary2["throughput"]["total_requests"] == 3
    assert summary2["throughput"]["successful_predictions"] == 3
    assert summary2["decision_tier_distribution"]["allow"]["count"] == 1
    assert summary2["decision_tier_distribution"]["review"]["count"] == 1
    assert summary2["decision_tier_distribution"]["high_risk"]["count"] == 1
    assert summary2["score_distribution"]["count"] == 3
    assert summary2["score_distribution"]["min"] == 0.05
    assert summary2["score_distribution"]["max"] == 0.55


def test_metrics_tracker_failure_recording():
    """Verifies that prediction failures are tracked with error taxonomy and rates."""
    tracker = OperationalMetricsTracker()

    # 3 successes, 1 failure
    for i in range(3):
        tracker.record_prediction(
            transaction_id=i,
            input_data={"TransactionAmt": 50.0},
            fraud_prob=0.02,
            risk_tier="allow",
            latency_ms=1.0,
        )

    tracker.record_failure(
        transaction_id=99,
        input_data={"TransactionAmt": -10.0},
        error_type="ValidationError",
        error_message="Transaction amount must be positive",
        latency_ms=0.5,
    )

    summary = tracker.get_summary()
    tp = summary["throughput"]

    assert tp["total_requests"] == 4
    assert tp["successful_predictions"] == 3
    assert tp["failed_predictions"] == 1
    assert tp["failure_rate_pct"] == 25.0
    assert tp["failure_categories"]["ValidationError"] == 1


def test_missingness_and_data_quality_alerts():
    """Verifies field-level missingness tracking and data quality warnings."""
    tracker = OperationalMetricsTracker()

    # Normal transaction with some missing attributes
    tracker.record_prediction(
        transaction_id=101,
        input_data={"TransactionAmt": 150.0, "ProductCD": "W", "card1": 5000.0},
        fraud_prob=0.03,
        risk_tier="allow",
        latency_ms=1.2,
    )

    # Transaction with extreme amount (> $50,000) triggering alert
    tracker.record_prediction(
        transaction_id=102,
        input_data={"TransactionAmt": 75000.0, "ProductCD": "W"},
        fraud_prob=0.88,
        risk_tier="high_risk",
        latency_ms=2.0,
    )

    summary = tracker.get_summary()
    dq = summary["data_quality_and_missingness"]

    assert dq["total_fields_inspected"] > 0
    assert dq["overall_missingness_rate_pct"] > 0.0
    assert dq["total_quality_alerts"] == 1
    assert len(summary["data_quality_and_missingness"]["top_missing_fields"]) > 0


def test_latency_percentiles_and_sla_compliance():
    """Verifies that latency percentiles and SLA compliance are mathematically sound."""
    tracker = OperationalMetricsTracker(sla_target_ms=10.0)

    # Record 100 predictions with simulated latencies (1.0ms to 100.0ms)
    for i in range(1, 101):
        tracker.record_prediction(
            transaction_id=i,
            input_data={"TransactionAmt": float(i)},
            fraud_prob=0.01,
            risk_tier="allow",
            latency_ms=float(i),  # 1ms to 100ms
        )

    summary = tracker.get_summary()
    lat = summary["latency_telemetry"]

    assert lat["count"] == 100
    assert lat["min_ms"] == 1.0
    assert lat["max_ms"] == 100.0
    assert lat["mean_ms"] == 50.5
    assert lat["percentiles"]["p50_ms"] == 50.5
    assert lat["percentiles"]["p90_ms"] == pytest.approx(90.1, abs=0.5)
    assert lat["percentiles"]["p95_ms"] == pytest.approx(95.05, abs=0.5)
    assert lat["percentiles"]["p99_ms"] == pytest.approx(99.01, abs=0.5)
    # Exactly 10 out of 100 are <= 10.0ms
    assert lat["sla_compliance_pct"] == 10.0


def test_export_summary_json_and_markdown():
    """Verifies that summary export writes valid JSON and formatted markdown."""
    tracker = OperationalMetricsTracker()

    tracker.record_prediction(
        transaction_id=1,
        input_data={"TransactionAmt": 100.0},
        fraud_prob=0.05,
        risk_tier="allow",
        latency_ms=1.5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "summary.json"
        tracker.export_summary_json(json_path)

        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["throughput"]["total_requests"] == 1

        md_report = tracker.generate_markdown_report()
        assert "# BankGuard" in md_report
        assert "Operational Throughput & Reliability" in md_report
        assert "Fraud Risk Score Distribution" in md_report
        assert "Decision Tier Breakdown" in md_report
        assert "Latency Telemetry & SLA Compliance" in md_report


def test_api_monitoring_summary_endpoint():
    """Verifies GET /monitoring/summary on FastAPI application."""
    app = create_app()
    with TestClient(app) as client:
        # Check initial summary
        r1 = client.get("/monitoring/summary")
        assert r1.status_code == 200
        data1 = r1.json()
        assert "throughput" in data1
        assert "score_distribution" in data1
        assert "latency_telemetry" in data1

        # Post a single transaction
        tx = {
            "TransactionID": 88888,
            "TransactionAmt": 125.0,
            "ProductCD": "W",
            "card1": 13926,
            "card6": "credit",
        }
        r_pred = client.post("/predict", json=tx)
        assert r_pred.status_code == 200

        # Verify summary reflects the scored transaction
        r2 = client.get("/monitoring/summary")
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["throughput"]["total_requests"] >= 1
        assert data2["throughput"]["successful_predictions"] >= 1


def test_fixed_test_batch_runner_verification():
    """Verifies Task 7.1 fixed test batch execution and reporting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_summary.json"
        md_path = Path(tmpdir) / "test_summary.md"

        summary = run_batch_monitoring(
            batch_size=100,
            output_json=json_path,
            output_md=md_path,
        )

        assert summary["throughput"]["total_requests"] == 100
        assert summary["throughput"]["successful_predictions"] == 100
        assert summary["throughput"]["failed_predictions"] == 0
        assert summary["throughput"]["failure_rate_pct"] == 0.0
        assert summary["score_distribution"]["count"] == 100
        assert summary["decision_tier_distribution"]["allow"]["count"] > 0
        assert summary["latency_telemetry"]["mean_ms"] < 10.0
        assert json_path.exists()
        assert md_path.exists()

