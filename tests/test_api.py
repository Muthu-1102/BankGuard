"""Comprehensive API integration and contract tests for BankGuard Serving API (Task 6.2).

Verifies:
1. Root and health check endpoints.
2. Valid payload prediction scoring.
3. Invalid payload handling (negative amount, missing amount, malformed types -> HTTP 422).
4. Unknown fields policy (extra unmodeled fields accepted and safely ignored).
5. Missing fields policy (sparse transactions imputed gracefully).
6. Batch prediction processing and tier distribution.
7. Model load failure handling (HTTP 503 with diagnostic error message).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from bankguard.api.app import app, create_app


@pytest.fixture(scope="module")
def client():
    """Module-scoped test client for the default configured FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verifies service index root route."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BankGuard Fraud Detection API"
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint_healthy(client: TestClient):
    """Verifies health check when model and preprocessor are successfully loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["policy_loaded"] is True
    assert data["model_type"] == "LightGBMClassifier"
    assert data["uptime_seconds"] >= 0.0


def test_policy_endpoint(client: TestClient):
    """Verifies operational policy endpoint returns active thresholds and actions."""
    response = client.get("/policy")
    assert response.status_code == 200
    data = response.json()
    assert data["review_threshold"] == 0.10
    assert data["block_threshold"] == 0.40
    assert "tiers" in data
    assert "allow" in data["tiers"]
    assert "review" in data["tiers"]
    assert "high_risk" in data["tiers"]


def test_predict_valid_payload(client: TestClient):
    """Verifies single transaction scoring on a realistic valid transaction."""
    payload = {
        "TransactionID": 3663550,
        "TransactionAmt": 125.50,
        "TransactionDT": 12192900,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 360,
        "card3": 150,
        "card4": "discover",
        "card5": 142,
        "card6": "credit",
        "addr1": 315,
        "addr2": 87,
        "C1": 1.0,
        "C2": 1.0,
        "V258": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 3663550
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ("allow", "review", "high_risk")
    assert data["recommendation"] in ("ALLOW", "REVIEW", "HIGH_RISK")
    assert isinstance(data["action"], str)
    assert data["latency_ms"] >= 0.0
    assert "thresholds" in data


def test_predict_invalid_payload_negative_amount(client: TestClient):
    """Verifies HTTP 422 when TransactionAmt is non-positive."""
    payload = {
        "TransactionAmt": -50.0,
        "card1": 1000,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Validation Error" in data["error"]
    assert any("greater than 0" in str(err) for err in data["details"])


def test_predict_invalid_payload_zero_amount(client: TestClient):
    """Verifies HTTP 422 when TransactionAmt is exactly 0.0."""
    payload = {"TransactionAmt": 0.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_payload_missing_amount(client: TestClient):
    """Verifies HTTP 422 when required TransactionAmt is omitted."""
    payload = {
        "TransactionID": 1001,
        "ProductCD": "W",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("Field required" in str(err) for err in data["details"])


def test_predict_invalid_payload_corrupted_type(client: TestClient):
    """Verifies HTTP 422 when TransactionAmt is a non-numerical string."""
    payload = {
        "TransactionAmt": "not_a_valid_float",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unknown_fields_policy(client: TestClient):
    """Verifies unknown fields policy: extra unmodeled fields are accepted and ignored safely."""
    payload = {
        "TransactionAmt": 88.0,
        "client_ip": "192.168.1.100",
        "custom_internal_tag": "vip_customer",
        "vendor_payload_data": {"device_fingerprint": "abc123xyz"},
        "extra_unseen_column_123": 42.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ("allow", "review", "high_risk")


def test_missing_fields_policy(client: TestClient):
    """Verifies missing fields policy: minimal payload with only TransactionAmt succeeds via imputation."""
    minimal_payload = {
        "TransactionAmt": 45.0,
    }
    response = client.post("/predict", json=minimal_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ("allow", "review", "high_risk")


def test_batch_prediction_endpoint(client: TestClient):
    """Verifies batch scoring of multiple transactions in a single request."""
    batch_payload = {
        "transactions": [
            {"TransactionID": 1, "TransactionAmt": 25.0, "card4": "visa", "C1": 1.0},
            {"TransactionID": 2, "TransactionAmt": 1500.0, "card4": "mastercard", "C1": 15.0, "V258": 3.0},
            {"TransactionID": 3, "TransactionAmt": 50.0, "card4": "discover"},
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 3
    assert len(data["predictions"]) == 3
    assert "tier_distribution" in data
    assert sum(data["tier_distribution"].values()) == 3
    assert data["total_latency_ms"] >= 0.0


def test_batch_prediction_empty_list(client: TestClient):
    """Verifies HTTP 422 when an empty transactions batch is submitted."""
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422


def test_model_load_failure(tmp_path: Path):
    """Verifies model load failure handling when artifacts are missing or corrupt."""
    fake_model_path = tmp_path / "non_existent_model.joblib"
    fake_app = create_app(model_path=fake_model_path)

    with TestClient(fake_app) as fake_client:
        # Health endpoint should report 503 and unhealthy
        health_resp = fake_client.get("/health")
        assert health_resp.status_code == 503
        health_data = health_resp.json()
        assert health_data["status"] == "unhealthy"
        assert health_data["model_loaded"] is False
        assert "not found" in health_data["error_message"].lower()

        # Prediction endpoint should fail with 503
        pred_resp = fake_client.post("/predict", json={"TransactionAmt": 100.0})
        assert pred_resp.status_code == 503
        assert "not ready" in pred_resp.json()["detail"].lower()
