"""Pydantic data schemas for BankGuard fraud scoring API (Task 6.2).

Defines rigorous request validation, response serialization, and policy models:
- Single and batch transaction input schemas.
- Strict positive amount validation.
- Permissive extra attribute handling (unknown fields policy: accept and ignore gracefully).
- Structured response formats for fraud risk scores, decision recommendations, and health status.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    """Input payload for a single financial transaction scoring request.

    Enforces:
    - Positive transaction amount (TransactionAmt > 0.0).
    - Permissive schema allowing any subset of IEEE-CIS fields.
    - Unknown fields are allowed and dropped during preprocessing (unknown fields policy).
    - Missing fields are filled by the preprocessor's imputation strategy (missing fields policy).
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    TransactionID: Optional[int] = Field(
        None,
        description="Unique identifier for the transaction (optional; generated if omitted)",
        examples=[3663550],
    )
    TransactionAmt: float = Field(
        ...,
        gt=0.0,
        description="Transaction amount in USD (must be strictly positive)",
        examples=[125.50],
    )
    TransactionDT: Optional[int] = Field(
        None,
        ge=0,
        description="Relative time delta in seconds from dataset reference point",
        examples=[12192900],
    )
    ProductCD: Optional[str] = Field(
        None,
        description="Product code, e.g. W, C, R, H, S",
        examples=["W"],
    )
    card1: Optional[float] = Field(None, description="Payment card issue bank identifier", examples=[13926])
    card2: Optional[float] = Field(None, description="Payment card product ID", examples=[360])
    card3: Optional[float] = Field(None, description="Payment card country code", examples=[150])
    card4: Optional[str] = Field(None, description="Payment card network / brand", examples=["discover"])
    card5: Optional[float] = Field(None, description="Payment card issue category", examples=[142])
    card6: Optional[str] = Field(None, description="Payment card type (debit, credit, etc.)", examples=["credit"])
    addr1: Optional[float] = Field(None, description="Billing zip/region code", examples=[315])
    addr2: Optional[float] = Field(None, description="Billing country code", examples=[87])
    dist1: Optional[float] = Field(None, description="Distance metric", examples=[19.0])
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain", examples=["gmail.com"])
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain", examples=["gmail.com"])


class BatchTransactionInput(BaseModel):
    """Input payload for scoring a batch of financial transactions."""
    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Batch of transactions to evaluate (1 to 500 records)",
    )


class PredictionResponse(BaseModel):
    """Standardized fraud prediction and decision recommendation response."""
    transaction_id: Optional[int] = Field(None, description="Unique transaction identifier")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated fraud probability in [0.0, 1.0]")
    risk_tier: str = Field(..., description="Operational decision tier ('allow', 'review', 'high_risk')")
    recommendation: str = Field(..., description="Decision recommendation label")
    action: str = Field(..., description="Operational action recommendation")
    description: str = Field(..., description="Detailed operational guidance")
    thresholds: Dict[str, float] = Field(..., description="Active operational decision thresholds")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end scoring latency in milliseconds")
    model_version: str = Field(..., description="Active model release version")


class BatchPredictionResponse(BaseModel):
    """Batch scoring summary and collection of individual transaction predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_evaluated: int = Field(..., ge=0, description="Total transactions processed")
    tier_distribution: Dict[str, int] = Field(..., description="Breakdown of recommendations by tier")
    total_latency_ms: float = Field(..., ge=0.0, description="Total batch processing latency in milliseconds")


class HealthResponse(BaseModel):
    """Service health and diagnostic status."""
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether LightGBM model weights are loaded in memory")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessing pipeline is loaded in memory")
    policy_loaded: bool = Field(..., description="Whether decision policy configuration is active")
    model_type: str = Field(..., description="Model architecture type")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")
    error_message: Optional[str] = Field(None, description="Diagnostic error details if unhealthy")


class PolicyResponse(BaseModel):
    """Operational decision policy metadata and active risk thresholds."""
    policy_name: str
    version: str
    review_threshold: float
    block_threshold: float
    tiers: Dict[str, Any]
