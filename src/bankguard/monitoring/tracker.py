"""Operational monitoring metrics tracker for BankGuard (Task 7.1).

Tracks:
1. Data Quality: schema anomalies, out-of-range amounts, extreme values.
2. Missing Input Fields: field-level and overall missingness rates.
3. Prediction Failures: error taxonomy, failure rates, and failure breakdown.
4. Score Distribution: min, max, mean, std, percentiles (p10..p99), score buckets, and tier breakdown.
5. Latency Telemetry: request latency, percentiles (p50, p90, p95, p99), and SLA compliance.
"""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from bankguard.monitoring.logger import get_structured_logger, sanitize_payload


CORE_INPUT_FIELDS: List[str] = [
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
]


class OperationalMetricsTracker:
    """Thread-safe operational telemetry tracker for fraud risk scoring."""

    def __init__(self, max_buffer_size: int = 10000, sla_target_ms: float = 10.0):
        """Initializes the operational metrics tracker.

        Args:
            max_buffer_size: Maximum rolling history length for scores and latencies.
            sla_target_ms: Service level agreement target latency threshold in milliseconds.
        """
        self.max_buffer_size = max_buffer_size
        self.sla_target_ms = sla_target_ms
        self.logger = get_structured_logger("bankguard.monitoring.tracker")

        self._lock = threading.Lock()
        self.start_time = datetime.now(timezone.utc)

        # Counter metrics
        self.total_requests: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.failure_categories: Counter = Counter()

        # Rolling buffers for statistical calculations
        self._scores: List[float] = []
        self._latencies_ms: List[float] = []
        self._tiers: List[str] = []

        # Data quality & missingness tracking
        self.data_quality_alerts: List[Dict[str, Any]] = []
        self.missing_field_counts: Counter = Counter()
        self.total_fields_inspected: int = 0
        self.total_missing_fields: int = 0

    def record_prediction(
        self,
        transaction_id: Optional[int],
        input_data: Dict[str, Any],
        fraud_prob: float,
        risk_tier: str,
        latency_ms: float,
        preprocessing_ms: float = 0.0,
        inference_ms: float = 0.0,
    ) -> None:
        """Records telemetry for a successful single prediction.

        Args:
            transaction_id: Unique transaction ID.
            input_data: Raw input dictionary before scoring.
            fraud_prob: Predicted fraud probability in [0.0, 1.0].
            risk_tier: Action tier ('allow', 'review', 'high_risk').
            latency_ms: Total execution time in milliseconds.
            preprocessing_ms: Feature transformation time in milliseconds.
            inference_ms: Model scoring time in milliseconds.
        """
        sanitized = sanitize_payload(input_data)
        warnings = self._audit_data_quality(sanitized, transaction_id)
        missing_count = self._track_missingness(sanitized)

        with self._lock:
            self.total_requests += 1
            self.successful_predictions += 1

            self._scores.append(float(fraud_prob))
            self._latencies_ms.append(float(latency_ms))
            self._tiers.append(str(risk_tier).lower())

            # Maintain buffer capacity
            if len(self._scores) > self.max_buffer_size:
                self._scores.pop(0)
                self._latencies_ms.pop(0)
                self._tiers.pop(0)

        # Structured log record
        log_payload = {
            "event": "prediction_executed",
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "risk_tier": risk_tier,
            "latency_ms": round(latency_ms, 2),
            "preprocessing_ms": round(preprocessing_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "missing_fields_count": missing_count,
            "quality_warnings_count": len(warnings),
        }
        self.logger.info(
            f"Transaction scored: id={transaction_id} score={fraud_prob:.4f} tier={risk_tier} lat={latency_ms:.2f}ms",
            extra={"structured_data": log_payload},
        )

    def record_failure(
        self,
        transaction_id: Optional[int],
        input_data: Optional[Dict[str, Any]],
        error_type: str,
        error_message: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Records telemetry for a failed prediction attempt.

        Args:
            transaction_id: Optional transaction ID.
            input_data: Optional payload that caused the failure.
            error_type: Classification (e.g. ValidationError, ModelNotLoadedError).
            error_message: Sanitized error description.
            latency_ms: Time spent before failure occurred.
        """
        with self._lock:
            self.total_requests += 1
            self.failed_predictions += 1
            self.failure_categories[error_type] += 1

        log_payload = {
            "event": "prediction_failed",
            "transaction_id": transaction_id,
            "error_type": error_type,
            "error_message": error_message,
            "latency_ms": round(latency_ms, 2),
        }
        self.logger.error(
            f"Prediction failed: type={error_type} msg={error_message}",
            extra={"structured_data": log_payload},
        )

    def record_batch(
        self,
        batch_predictions: List[Dict[str, Any]],
        total_latency_ms: float,
        inputs_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Records telemetry for a batch prediction pass.

        Args:
            batch_predictions: List of prediction dictionaries (each with fraud_probability, risk_tier, transaction_id).
            total_latency_ms: Total latency across the entire batch in milliseconds.
            inputs_list: Optional list of raw inputs for data quality & missingness audits.
        """
        n_items = len(batch_predictions)
        if n_items == 0:
            return

        per_item_lat = round(total_latency_ms / n_items, 3)

        with self._lock:
            for pred in batch_predictions:
                self.total_requests += 1
                self.successful_predictions += 1
                prob = float(pred.get("fraud_probability", 0.0))
                tier = str(pred.get("risk_tier", "allow")).lower()

                self._scores.append(prob)
                self._latencies_ms.append(per_item_lat)
                self._tiers.append(tier)

                if len(self._scores) > self.max_buffer_size:
                    self._scores.pop(0)
                    self._latencies_ms.pop(0)
                    self._tiers.pop(0)

        # Audit input quality if provided
        if inputs_list:
            for i, inp in enumerate(inputs_list):
                tx_id = inp.get("TransactionID")
                sanitized = sanitize_payload(inp)
                self._audit_data_quality(sanitized, tx_id)
                self._track_missingness(sanitized)

        log_payload = {
            "event": "batch_prediction_executed",
            "batch_size": n_items,
            "total_latency_ms": round(total_latency_ms, 2),
            "average_latency_ms": per_item_lat,
        }
        self.logger.info(
            f"Batch scored: size={n_items} total_lat={total_latency_ms:.2f}ms avg_lat={per_item_lat:.2f}ms",
            extra={"structured_data": log_payload},
        )

    def _audit_data_quality(self, data: Dict[str, Any], tx_id: Optional[int]) -> List[str]:
        """Checks incoming data against operational quality rules."""
        warnings: List[str] = []

        # 1. Amount validations
        amt = data.get("TransactionAmt")
        if amt is not None:
            try:
                amt_val = float(amt)
                if amt_val <= 0:
                    warnings.append("non_positive_amount")
                elif amt_val > 50000:
                    warnings.append("extreme_transaction_amount")
            except (ValueError, TypeError):
                warnings.append("invalid_amount_format")

        # 2. Key identifier validations
        if "card1" in data and data["card1"] is not None:
            try:
                c1 = float(data["card1"])
                if c1 <= 0 or c1 > 100000:
                    warnings.append("out_of_range_card1")
            except (ValueError, TypeError):
                warnings.append("invalid_card1_format")

        # Record warnings
        if warnings:
            alert_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transaction_id": tx_id,
                "warnings": warnings,
            }
            with self._lock:
                self.data_quality_alerts.append(alert_entry)
                if len(self.data_quality_alerts) > 1000:
                    self.data_quality_alerts.pop(0)

        return warnings

    def _track_missingness(self, data: Dict[str, Any]) -> int:
        """Tracks field missingness count and rates against expected schema and extra features."""
        missing_count = 0
        fields_to_check = set(CORE_INPUT_FIELDS).union(data.keys())

        with self._lock:
            for field in fields_to_check:
                self.total_fields_inspected += 1
                val = data.get(field)
                if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
                    missing_count += 1
                    self.missing_field_counts[field] += 1
                    self.total_missing_fields += 1

        return missing_count

    def get_summary(self) -> Dict[str, Any]:
        """Calculates and returns complete operational monitoring summary statistics."""
        with self._lock:
            total_req = self.total_requests
            succ_req = self.successful_predictions
            fail_req = self.failed_predictions
            fail_rate = round((fail_req / total_req * 100.0), 2) if total_req > 0 else 0.0

            scores_arr = np.array(self._scores) if self._scores else np.array([])
            lats_arr = np.array(self._latencies_ms) if self._latencies_ms else np.array([])
            tiers_list = list(self._tiers)

            # Score Distribution Statistics
            if len(scores_arr) > 0:
                score_stats = {
                    "count": int(len(scores_arr)),
                    "mean": round(float(np.mean(scores_arr)), 4),
                    "std": round(float(np.std(scores_arr)), 4),
                    "min": round(float(np.min(scores_arr)), 4),
                    "max": round(float(np.max(scores_arr)), 4),
                    "percentiles": {
                        "p10": round(float(np.percentile(scores_arr, 10)), 4),
                        "p25": round(float(np.percentile(scores_arr, 25)), 4),
                        "p50": round(float(np.percentile(scores_arr, 50)), 4),
                        "p75": round(float(np.percentile(scores_arr, 75)), 4),
                        "p90": round(float(np.percentile(scores_arr, 90)), 4),
                        "p95": round(float(np.percentile(scores_arr, 95)), 4),
                        "p99": round(float(np.percentile(scores_arr, 99)), 4),
                    },
                    "buckets": {
                        "0.0_to_0.10_safe": int(np.sum(scores_arr < 0.10)),
                        "0.10_to_0.20_low": int(np.sum((scores_arr >= 0.10) & (scores_arr < 0.20))),
                        "0.20_to_0.40_review": int(np.sum((scores_arr >= 0.20) & (scores_arr < 0.40))),
                        "0.40_to_0.60_high": int(np.sum((scores_arr >= 0.40) & (scores_arr < 0.60))),
                        "0.60_to_0.80_very_high": int(np.sum((scores_arr >= 0.60) & (scores_arr < 0.80))),
                        "0.80_to_1.00_critical": int(np.sum(scores_arr >= 0.80)),
                    },
                }
            else:
                score_stats = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "percentiles": {k: 0.0 for k in ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]},
                    "buckets": {},
                }

            # Decision Tier Distribution
            tier_counter = Counter(tiers_list)
            total_tiers = len(tiers_list)
            tier_distribution = {
                "allow": {
                    "count": tier_counter.get("allow", 0),
                    "percentage": round(tier_counter.get("allow", 0) / total_tiers * 100.0, 2) if total_tiers > 0 else 0.0,
                },
                "review": {
                    "count": tier_counter.get("review", 0),
                    "percentage": round(tier_counter.get("review", 0) / total_tiers * 100.0, 2) if total_tiers > 0 else 0.0,
                },
                "high_risk": {
                    "count": tier_counter.get("high_risk", 0),
                    "percentage": round(tier_counter.get("high_risk", 0) / total_tiers * 100.0, 2) if total_tiers > 0 else 0.0,
                },
            }

            # Latency Telemetry
            if len(lats_arr) > 0:
                under_sla = int(np.sum(lats_arr <= self.sla_target_ms))
                sla_comp = round(under_sla / len(lats_arr) * 100.0, 2)
                latency_stats = {
                    "count": int(len(lats_arr)),
                    "mean_ms": round(float(np.mean(lats_arr)), 2),
                    "min_ms": round(float(np.min(lats_arr)), 2),
                    "max_ms": round(float(np.max(lats_arr)), 2),
                    "percentiles": {
                        "p50_ms": round(float(np.percentile(lats_arr, 50)), 2),
                        "p90_ms": round(float(np.percentile(lats_arr, 90)), 2),
                        "p95_ms": round(float(np.percentile(lats_arr, 95)), 2),
                        "p99_ms": round(float(np.percentile(lats_arr, 99)), 2),
                    },
                    "sla_target_ms": self.sla_target_ms,
                    "sla_compliance_pct": sla_comp,
                }
            else:
                latency_stats = {
                    "count": 0,
                    "mean_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "percentiles": {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
                    "sla_target_ms": self.sla_target_ms,
                    "sla_compliance_pct": 100.0,
                }

            # Missingness Summary
            tot_inspected = self.total_fields_inspected
            tot_missing = self.total_missing_fields
            missing_rate = round(tot_missing / tot_inspected * 100.0, 2) if tot_inspected > 0 else 0.0
            top_missing = [
                {"field": k, "missing_count": v}
                for k, v in self.missing_field_counts.most_common(10)
            ]

            summary = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "service": "BankGuard Fraud Intelligence Serving Service",
                "uptime_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2),
                "throughput": {
                    "total_requests": total_req,
                    "successful_predictions": succ_req,
                    "failed_predictions": fail_req,
                    "failure_rate_pct": fail_rate,
                    "failure_categories": dict(self.failure_categories),
                },
                "score_distribution": score_stats,
                "decision_tier_distribution": tier_distribution,
                "latency_telemetry": latency_stats,
                "data_quality_and_missingness": {
                    "total_fields_inspected": tot_inspected,
                    "total_missing_values": tot_missing,
                    "overall_missingness_rate_pct": missing_rate,
                    "total_quality_alerts": len(self.data_quality_alerts),
                    "top_missing_fields": top_missing,
                },
            }
            return summary

    def export_summary_json(self, output_path: Union[str, Path]) -> Path:
        """Exports the monitoring summary to a JSON file on disk.

        Args:
            output_path: File path to save summary.

        Returns:
            Resolved Path object.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return p

    def generate_markdown_report(self) -> str:
        """Generates a clean markdown monitoring audit report."""
        s = self.get_summary()
        tp = s["throughput"]
        sc = s["score_distribution"]
        td = s["decision_tier_distribution"]
        lat = s["latency_telemetry"]
        dq = s["data_quality_and_missingness"]

        md = f"""# BankGuard — Operational Monitoring & Batch Telemetry Report

**Generated At:** {s['generated_at']}  
**Service:** {s['service']}  
**Uptime:** {s['uptime_seconds']} seconds  

---

## 1. Operational Throughput & Reliability

| Metric | Measured Value | Operational SLA / Target |
| :--- | :--- | :--- |
| **Total Requests Received** | **{tp['total_requests']:,}** | High-availability stream |
| **Successful Predictions** | **{tp['successful_predictions']:,}** | &ge; 99.9% uptime |
| **Failed Predictions** | **{tp['failed_predictions']:,}** | &le; 0.1% failures |
| **Failure Rate** | **{tp['failure_rate_pct']:.2f}%** | Zero unhandled exceptions |

### Failure Breakdown by Category
"""
        if tp["failure_categories"]:
            for cat, count in tp["failure_categories"].items():
                md += f"- **`{cat}`**: {count:,} occurrences\n"
        else:
            md += "- *Zero operational prediction failures recorded.*\n"

        md += f"""
---

## 2. Fraud Risk Score Distribution

| Statistic | Score Value | Interpretation |
| :--- | :--- | :--- |
| **Evaluated Count** | {sc['count']:,} | Sample size evaluated |
| **Mean Score** | {sc['mean']:.4f} | Overall fleet risk expectation |
| **Standard Deviation** | {sc['std']:.4f} | Score dispersion |
| **Minimum Score** | {sc['min']:.4f} | Lowest risk transaction observed |
| **p25 (1st Quartile)** | {sc['percentiles']['p25']:.4f} | 25% of traffic below this risk |
| **p50 (Median)** | {sc['percentiles']['p50']:.4f} | Central median risk score |
| **p75 (3rd Quartile)** | {sc['percentiles']['p75']:.4f} | 75% of traffic below this risk |
| **p90 (90th Percentile)**| {sc['percentiles']['p90']:.4f} | Top 10% suspicious tail |
| **p95 (95th Percentile)**| {sc['percentiles']['p95']:.4f} | Top 5% high suspicion threshold |
| **p99 (99th Percentile)**| {sc['percentiles']['p99']:.4f} | Critical outlier threshold |
| **Maximum Score** | {sc['max']:.4f} | Highest risk transaction observed |

### Score Histograms & Buckets
- `[0.00 - 0.10) Safe / Legitimate`: **{sc['buckets'].get('0.0_to_0.10_safe', 0):,}** transactions
- `[0.10 - 0.20) Low Suspicion`: **{sc['buckets'].get('0.10_to_0.20_low', 0):,}** transactions
- `[0.20 - 0.40) Elevated (Review)`: **{sc['buckets'].get('0.20_to_0.40_review', 0):,}** transactions
- `[0.40 - 0.60) High Risk`: **{sc['buckets'].get('0.40_to_0.60_high', 0):,}** transactions
- `[0.60 - 0.80) Very High Risk`: **{sc['buckets'].get('0.60_to_0.80_very_high', 0):,}** transactions
- `[0.80 - 1.00] Critical High Risk`: **{sc['buckets'].get('0.80_to_1.00_critical', 0):,}** transactions

---

## 3. Decision Tier Breakdown

| Policy Decision Tier | Cutoff Criteria | Transaction Count | Volume Share |
| :--- | :--- | :--- | :--- |
| **● ALLOW** | `Score < 0.10` | **{td['allow']['count']:,}** | **{td['allow']['percentage']:.2f}%** |
| **▲ REVIEW** | `0.10 <= Score < 0.40` | **{td['review']['count']:,}** | **{td['review']['percentage']:.2f}%** |
| **⬢ HIGH RISK** | `Score >= 0.40` | **{td['high_risk']['count']:,}** | **{td['high_risk']['percentage']:.2f}%** |

---

## 4. Latency Telemetry & SLA Compliance

| Latency Metric | Measured | SLA Standard | Status |
| :--- | :--- | :--- | :--- |
| **Average Latency** | **{lat['mean_ms']:.2f} ms** | &le; 5.0 ms | {'✔ PASS' if lat['mean_ms'] <= 5.0 else '⚠ ELEVATED'} |
| **Median Latency (p50)** | **{lat['percentiles']['p50_ms']:.2f} ms** | &le; 3.0 ms | {'✔ PASS' if lat['percentiles']['p50_ms'] <= 3.0 else '⚠ ELEVATED'} |
| **p90 Latency** | **{lat['percentiles']['p90_ms']:.2f} ms** | &le; 8.0 ms | {'✔ PASS' if lat['percentiles']['p90_ms'] <= 8.0 else '⚠ ELEVATED'} |
| **p95 Latency** | **{lat['percentiles']['p95_ms']:.2f} ms** | &le; 10.0 ms | {'✔ PASS' if lat['percentiles']['p95_ms'] <= 10.0 else '⚠ ELEVATED'} |
| **p99 Latency** | **{lat['percentiles']['p99_ms']:.2f} ms** | &le; 15.0 ms | {'✔ PASS' if lat['percentiles']['p99_ms'] <= 15.0 else '⚠ ELEVATED'} |
| **SLA Target Compliance** | **{lat['sla_compliance_pct']:.2f}%** | &ge; 99.0% &le; {lat['sla_target_ms']} ms | {'✔ PASS' if lat['sla_compliance_pct'] >= 99.0 else '⚠ BREACH'} |

---

## 5. Data Quality, Schema Health & Missingness

- **Total Feature Attributes Inspected:** {dq['total_fields_inspected']:,}
- **Missing Value Rate:** **{dq['overall_missingness_rate_pct']:.2f}%** ({dq['total_missing_values']:,} missing values imputed)
- **Data Quality Alerts Triggered:** **{dq['total_quality_alerts']}** alerts

### Top Missing Input Fields
"""
        if dq["top_missing_fields"]:
            for item in dq["top_missing_fields"]:
                md += f"- **`{item['field']}`**: missing in {item['missing_count']:,} requests\n"
        else:
            md += "- *All required features were completely populated across requests.*\n"

        md += """
---
*Privacy Notice: This monitoring telemetry contains zero raw personal data, card numbers, device fingerprints, or IP addresses in compliance with BankGuard Privacy & Data Protection Policies.*
"""
        return md

    def reset(self) -> None:
        """Resets all metrics to clean initial state."""
        with self._lock:
            self.total_requests = 0
            self.successful_predictions = 0
            self.failed_predictions = 0
            self.failure_categories.clear()
            self._scores.clear()
            self._latencies_ms.clear()
            self._tiers.clear()
            self.data_quality_alerts.clear()
            self.missing_field_counts.clear()
            self.total_fields_inspected = 0
            self.total_missing_fields = 0
            self.start_time = datetime.now(timezone.utc)
