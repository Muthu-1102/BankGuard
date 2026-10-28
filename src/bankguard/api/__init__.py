"""API package for BankGuard (Phase 6).

Provides FastAPI REST service for real-time fraud scoring, batch inference,
health checks, and policy inspection.
"""

from bankguard.api.app import app, create_app
from bankguard.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionInput,
    HealthResponse,
    PolicyResponse,
    PredictionResponse,
    TransactionInput,
)
from bankguard.api.service import ModelNotLoadedError, ModelService

__all__ = [
    "app",
    "create_app",
    "BatchPredictionResponse",
    "BatchTransactionInput",
    "HealthResponse",
    "PolicyResponse",
    "PredictionResponse",
    "TransactionInput",
    "ModelNotLoadedError",
    "ModelService",
]
