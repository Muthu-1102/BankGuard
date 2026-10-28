"""FastAPI application for BankGuard financial transaction fraud scoring (Task 6.2).

Exposes production REST endpoints:
- GET / : API service metadata and documentation links.
- GET /health : System readiness check and artifact status.
- POST /predict : Real-time single transaction fraud scoring and decisioning.
- POST /predict/batch : High-throughput batch transaction scoring.
- GET /policy : Current decision policy thresholds and actions.

Features:
- Artifacts loaded once at startup via ASGI lifespan context.
- Input validation with detailed Pydantic error reporting.
- Robust error handling for model load failures (HTTP 503).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from bankguard.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionInput,
    HealthResponse,
    PolicyResponse,
    PredictionResponse,
    TransactionInput,
)
from bankguard.api.service import ModelNotLoadedError, ModelService
from bankguard.decisioning.policy import DEFAULT_POLICY_CONFIG_PATH


def create_app(
    model_path: Union[str, Path] = "artifacts/improved_model.joblib",
    preprocessor_path: Union[str, Path] = "artifacts/preprocessor.joblib",
    policy_path: Union[str, Path] = DEFAULT_POLICY_CONFIG_PATH,
) -> FastAPI:
    """Factory function creating configured BankGuard FastAPI application instance.

    Args:
        model_path: Path to serialized model artifact.
        preprocessor_path: Path to serialized preprocessor artifact.
        policy_path: Path to decision policy configuration.

    Returns:
        Configured FastAPI app.
    """
    model_service = ModelService(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        policy_path=policy_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Load model artifacts once
        print("Starting BankGuard Serving API: loading artifacts into memory...")
        success = model_service.load_artifacts()
        if success:
            print("  Artifacts successfully loaded and warm-up ready.")
        else:
            print(f"  WARNING: Artifact loading failed: {model_service.error_message}")
        yield
        # Shutdown
        print("Shutting down BankGuard Serving API.")

    app = FastAPI(
        title="BankGuard Fraud Detection API",
        description=(
            "Real-time financial transaction fraud risk estimation, explainability, "
            "and automated decision recommendation service."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store service reference in app state
    app.state.model_service = model_service

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Transaction payload contains invalid or missing required fields.",
                "details": exc.errors(),
            },
        )

    @app.get(
        "/",
        summary="Service Index",
        description="Returns API service information and active routes.",
        tags=["System"],
    )
    async def root():
        return {
            "service": "BankGuard Fraud Detection API",
            "version": "1.0.0",
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/health",
            "policy_url": "/policy",
            "endpoints": {
                "predict_single": "POST /predict",
                "predict_batch": "POST /predict/batch",
                "health": "GET /health",
                "policy": "GET /policy",
            },
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        description="Verifies whether model artifacts, preprocessor, and policy engine are loaded.",
        tags=["System"],
    )
    async def health_check():
        health = model_service.get_health()
        if not health.model_loaded or not health.preprocessor_loaded:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health.model_dump(),
            )
        return health

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Score Single Transaction",
        description="Evaluates fraud risk probability for a transaction and assigns an operational action.",
        tags=["Inference"],
    )
    async def predict_single_transaction(payload: TransactionInput):
        if not model_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model service is not ready: {model_service.error_message or 'Artifacts not loaded'}",
            )
        try:
            return model_service.predict_single(payload)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(e)}",
            )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        summary="Score Batch of Transactions",
        description="Evaluates a list of up to 500 transactions in a vectorized pass.",
        tags=["Inference"],
    )
    async def predict_batch_transactions(payload: BatchTransactionInput):
        if not model_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model service is not ready: {model_service.error_message or 'Artifacts not loaded'}",
            )
        try:
            return model_service.predict_batch(payload.transactions)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch inference error: {str(e)}",
            )

    @app.get(
        "/policy",
        response_model=PolicyResponse,
        summary="Get Active Decision Policy",
        description="Retrieves current operational fraud decision thresholds and tier definitions.",
        tags=["Policy"],
    )
    async def get_policy():
        if not model_service.is_ready or model_service.policy is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Decision policy is not loaded.",
            )
        return model_service.get_policy_metadata()

    return app


# Default ASGI application instance
app = create_app()
