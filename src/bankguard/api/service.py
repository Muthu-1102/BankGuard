"""Service layer for BankGuard model serving, inference, and operational telemetry (Task 6.2 & 7.1).

Manages artifact lifecycle, ensures once-at-startup loading, handles missing
and unknown input fields, coordinates preprocessor, model, and decision policy,
and tracks operational metrics (latency, score distribution, missingness, failures).
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from bankguard.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    PolicyResponse,
    PredictionResponse,
    TransactionInput,
)
from bankguard.decisioning.policy import (
    DEFAULT_POLICY_CONFIG_PATH,
    DecisionResult,
    FraudDecisionPolicy,
)
from bankguard.features.preprocessor import (
    BankGuardFeatureEngineer,
    BankGuardPreprocessor,
)
from bankguard.monitoring.tracker import OperationalMetricsTracker

# Unpickling compatibility fix for joblib models
if hasattr(sys.modules.get("__main__"), "__dict__"):
    sys.modules["__main__"].BankGuardFeatureEngineer = BankGuardFeatureEngineer
    sys.modules["__main__"].BankGuardPreprocessor = BankGuardPreprocessor


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested but artifacts are not successfully loaded."""
    pass


class ModelService:
    """Manages model artifacts in memory and executes low-latency inference with operational monitoring."""

    def __init__(
        self,
        model_path: Union[str, Path] = "artifacts/improved_model.joblib",
        preprocessor_path: Union[str, Path] = "artifacts/preprocessor.joblib",
        policy_path: Union[str, Path] = DEFAULT_POLICY_CONFIG_PATH,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.policy_path = Path(policy_path)

        self.model: Optional[Any] = None
        self.preprocessor: Optional[BankGuardPreprocessor] = None
        self.policy: Optional[FraudDecisionPolicy] = None

        self.is_ready: bool = False
        self.error_message: Optional[str] = None
        self.start_time: float = time.time()
        self.model_version: str = "1.0.0"

        # Operational telemetry and monitoring tracker (Task 7.1)
        self.tracker: OperationalMetricsTracker = OperationalMetricsTracker()

    def load_artifacts(self) -> bool:
        """Loads model weights, preprocessor pipeline, and decision policy into memory once."""
        try:
            # 1. Check preprocessor artifact
            if not self.preprocessor_path.is_file():
                raise FileNotFoundError(f"Preprocessor artifact not found at: {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)

            # 2. Check model artifact
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
            self.model = joblib.load(self.model_path)

            # 3. Check and load decision policy
            if not self.policy_path.is_file():
                raise FileNotFoundError(f"Policy config not found at: {self.policy_path}")
            self.policy = FraudDecisionPolicy.from_yaml(self.policy_path)

            self.is_ready = True
            self.error_message = None
            return True

        except Exception as e:
            self.model = None
            self.preprocessor = None
            self.policy = None
            self.is_ready = False
            self.error_message = f"Failed to load artifacts: {str(e)}"
            return False

    def predict_single(self, input_data: TransactionInput) -> PredictionResponse:
        """Processes a single transaction and returns fraud score with recommendation.

        Handles:
        - Unknown fields policy: Extra fields in input_data are passed through or ignored.
        - Missing fields policy: Imputed by preprocessor median/missing encoders.
        - Operational telemetry tracking: logs score, latency, missingness, and data quality.
        """
        if not self.is_ready or self.model is None or self.preprocessor is None or self.policy is None:
            raise ModelNotLoadedError(f"Model service is not ready: {self.error_message or 'Artifacts not loaded'}")

        t0 = time.time()
        tx_dict = input_data.model_dump(exclude_none=True)
        tx_id = tx_dict.get("TransactionID")

        try:
            # Convert to single-row DataFrame
            df_row = pd.DataFrame([tx_dict])

            # Preprocess features (guaranteed zero refit, handles imputation)
            t_prep0 = time.time()
            X_trans = self.preprocessor.transform(df_row)
            prep_lat = round((time.time() - t_prep0) * 1000.0, 3)

            # Generate fraud probability
            t_inf0 = time.time()
            prob = float(self.model.predict_proba(X_trans)[0, 1])
            inf_lat = round((time.time() - t_inf0) * 1000.0, 3)

            # Apply deterministic decision policy
            decision = self.policy.evaluate(prob)
            latency = round((time.time() - t0) * 1000.0, 2)

            # Record operational telemetry (Task 7.1)
            self.tracker.record_prediction(
                transaction_id=tx_id,
                input_data=tx_dict,
                fraud_prob=prob,
                risk_tier=decision.recommendation.value,
                latency_ms=latency,
                preprocessing_ms=prep_lat,
                inference_ms=inf_lat,
            )

            return PredictionResponse(
                transaction_id=tx_id,
                fraud_probability=round(prob, 4),
                risk_tier=decision.recommendation.value,
                recommendation=decision.recommendation.value.upper(),
                action=decision.action,
                description=decision.description,
                thresholds={
                    "review_threshold": self.policy.review_threshold,
                    "block_threshold": self.policy.block_threshold,
                },
                latency_ms=latency,
                model_version=self.model_version,
            )

        except Exception as e:
            total_lat = round((time.time() - t0) * 1000.0, 2)
            self.tracker.record_failure(
                transaction_id=tx_id,
                input_data=tx_dict,
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=total_lat,
            )
            raise

    def predict_batch(self, batch_data: List[TransactionInput]) -> BatchPredictionResponse:
        """Processes a collection of transactions in a single vectorized pass."""
        if not self.is_ready or self.model is None or self.preprocessor is None or self.policy is None:
            raise ModelNotLoadedError(f"Model service is not ready: {self.error_message or 'Artifacts not loaded'}")

        t0 = time.time()
        dict_list = [tx.model_dump(exclude_none=True) for tx in batch_data]
        df_batch = pd.DataFrame(dict_list)

        try:
            # Preprocess entire batch
            X_trans = self.preprocessor.transform(df_batch)

            # Vectorized probability prediction
            probs = self.model.predict_proba(X_trans)[:, 1]

            # Evaluate each record
            predictions: List[PredictionResponse] = []
            tier_counts = {"allow": 0, "review": 0, "high_risk": 0}

            for i, prob in enumerate(probs):
                f_prob = float(prob)
                decision = self.policy.evaluate(f_prob)
                tx_id = dict_list[i].get("TransactionID")
                tier_str = decision.recommendation.value
                tier_counts[tier_str] = tier_counts.get(tier_str, 0) + 1

                predictions.append(
                    PredictionResponse(
                        transaction_id=tx_id,
                        fraud_probability=round(f_prob, 4),
                        risk_tier=tier_str,
                        recommendation=tier_str.upper(),
                        action=decision.action,
                        description=decision.description,
                        thresholds={
                            "review_threshold": self.policy.review_threshold,
                            "block_threshold": self.policy.block_threshold,
                        },
                        latency_ms=0.0,
                        model_version=self.model_version,
                    )
                )

            total_latency = round((time.time() - t0) * 1000.0, 2)
            avg_latency = round(total_latency / len(predictions), 2) if predictions else 0.0
            for p in predictions:
                p.latency_ms = avg_latency

            # Record batch telemetry in tracker (Task 7.1)
            pred_dicts = [
                {
                    "transaction_id": p.transaction_id,
                    "fraud_probability": p.fraud_probability,
                    "risk_tier": p.risk_tier,
                }
                for p in predictions
            ]
            self.tracker.record_batch(
                batch_predictions=pred_dicts,
                total_latency_ms=total_latency,
                inputs_list=dict_list,
            )

            return BatchPredictionResponse(
                predictions=predictions,
                total_evaluated=len(predictions),
                tier_distribution=tier_counts,
                total_latency_ms=total_latency,
            )

        except Exception as e:
            total_lat = round((time.time() - t0) * 1000.0, 2)
            self.tracker.record_failure(
                transaction_id=None,
                input_data=None,
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=total_lat,
            )
            raise

    def get_health(self) -> HealthResponse:
        """Returns comprehensive diagnostic and readiness state."""
        uptime = round(time.time() - self.start_time, 2)
        return HealthResponse(
            status="healthy" if self.is_ready else "unhealthy",
            model_loaded=bool(self.model is not None),
            preprocessor_loaded=bool(self.preprocessor is not None),
            policy_loaded=bool(self.policy is not None),
            model_type="LightGBMClassifier" if self.is_ready else "Unknown",
            uptime_seconds=uptime,
            error_message=self.error_message,
        )

    def get_policy_metadata(self) -> PolicyResponse:
        """Returns active operational decision policy thresholds and tiers."""
        if self.policy is None:
            raise ModelNotLoadedError("Decision policy is not loaded.")

        return PolicyResponse(
            policy_name=self.policy.policy_name,
            version=self.policy.version,
            review_threshold=self.policy.review_threshold,
            block_threshold=self.policy.block_threshold,
            tiers=self.policy.tier_metadata,
        )

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Returns live operational monitoring telemetry summary (Task 7.1)."""
        return self.tracker.get_summary()
