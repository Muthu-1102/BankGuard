# BankGuard — Operational Monitoring & Batch Telemetry Report

**Generated At:** 2026-10-31T09:49:18.991379+00:00  
**Service:** BankGuard Fraud Intelligence Serving Service  
**Uptime:** 0.39 seconds  

---

## 1. Operational Throughput & Reliability

| Metric | Measured Value | Operational SLA / Target |
| :--- | :--- | :--- |
| **Total Requests Received** | **1,000** | High-availability stream |
| **Successful Predictions** | **1,000** | &ge; 99.9% uptime |
| **Failed Predictions** | **0** | &le; 0.1% failures |
| **Failure Rate** | **0.00%** | Zero unhandled exceptions |

### Failure Breakdown by Category
- *Zero operational prediction failures recorded.*

---

## 2. Fraud Risk Score Distribution

| Statistic | Score Value | Interpretation |
| :--- | :--- | :--- |
| **Evaluated Count** | 1,000 | Sample size evaluated |
| **Mean Score** | 0.0188 | Overall fleet risk expectation |
| **Standard Deviation** | 0.0237 | Score dispersion |
| **Minimum Score** | 0.0022 | Lowest risk transaction observed |
| **p25 (1st Quartile)** | 0.0071 | 25% of traffic below this risk |
| **p50 (Median)** | 0.0119 | Central median risk score |
| **p75 (3rd Quartile)** | 0.0200 | 75% of traffic below this risk |
| **p90 (90th Percentile)**| 0.0345 | Top 10% suspicious tail |
| **p95 (95th Percentile)**| 0.0554 | Top 5% high suspicion threshold |
| **p99 (99th Percentile)**| 0.1370 | Critical outlier threshold |
| **Maximum Score** | 0.1937 | Highest risk transaction observed |

### Score Histograms & Buckets
- `[0.00 - 0.10) Safe / Legitimate`: **972** transactions
- `[0.10 - 0.20) Low Suspicion`: **28** transactions
- `[0.20 - 0.40) Elevated (Review)`: **0** transactions
- `[0.40 - 0.60) High Risk`: **0** transactions
- `[0.60 - 0.80) Very High Risk`: **0** transactions
- `[0.80 - 1.00] Critical High Risk`: **0** transactions

---

## 3. Decision Tier Breakdown

| Policy Decision Tier | Cutoff Criteria | Transaction Count | Volume Share |
| :--- | :--- | :--- | :--- |
| **● ALLOW** | `Score < 0.10` | **972** | **97.20%** |
| **▲ REVIEW** | `0.10 <= Score < 0.40` | **28** | **2.80%** |
| **⬢ HIGH RISK** | `Score >= 0.40` | **0** | **0.00%** |

---

## 4. Latency Telemetry & SLA Compliance

| Latency Metric | Measured | SLA Standard | Status |
| :--- | :--- | :--- | :--- |
| **Average Latency** | **0.07 ms** | &le; 5.0 ms | ✔ PASS |
| **Median Latency (p50)** | **0.07 ms** | &le; 3.0 ms | ✔ PASS |
| **p90 Latency** | **0.07 ms** | &le; 8.0 ms | ✔ PASS |
| **p95 Latency** | **0.07 ms** | &le; 10.0 ms | ✔ PASS |
| **p99 Latency** | **0.07 ms** | &le; 15.0 ms | ✔ PASS |
| **SLA Target Compliance** | **100.00%** | &ge; 99.0% &le; 10.0 ms | ✔ PASS |

---

## 5. Data Quality, Schema Health & Missingness

- **Total Feature Attributes Inspected:** 15,000
- **Missing Value Rate:** **14.07%** (2,111 missing values imputed)
- **Data Quality Alerts Triggered:** **2** alerts

### Top Missing Input Fields
- **`R_emaildomain`**: missing in 750 requests
- **`dist1`**: missing in 566 requests
- **`P_emaildomain`**: missing in 280 requests
- **`card2`**: missing in 138 requests
- **`addr1`**: missing in 119 requests
- **`addr2`**: missing in 106 requests
- **`card3`**: missing in 96 requests
- **`card4`**: missing in 50 requests
- **`card1`**: missing in 6 requests

---
*Privacy Notice: This monitoring telemetry contains zero raw personal data, card numbers, device fingerprints, or IP addresses in compliance with BankGuard Privacy & Data Protection Policies.*
