"""Unit tests for BankGuard Fraud Decision Policy Engine (Task 6.1)."""

import math
from pathlib import Path
import numpy as np
import pytest

from bankguard.decisioning.policy import (
    DEFAULT_POLICY_CONFIG_PATH,
    DecisionRecommendation,
    DecisionResult,
    FraudDecisionPolicy,
    load_default_policy,
)


@pytest.fixture
def default_policy() -> FraudDecisionPolicy:
    """Fixture providing a standard decision policy (review=0.10, block=0.40)."""
    return FraudDecisionPolicy(review_threshold=0.10, block_threshold=0.40)


def test_threshold_validation():
    """Verifies validation rules on invalid thresholds."""
    # Review >= Block
    with pytest.raises(ValueError, match="must be strictly less than"):
        FraudDecisionPolicy(review_threshold=0.50, block_threshold=0.30)

    with pytest.raises(ValueError, match="must be strictly less than"):
        FraudDecisionPolicy(review_threshold=0.20, block_threshold=0.20)

    # Out of bounds [0.0, 1.0]
    with pytest.raises(ValueError, match="must be strictly between 0.0 and 1.0"):
        FraudDecisionPolicy(review_threshold=0.0, block_threshold=0.50)

    with pytest.raises(ValueError, match="must be strictly between 0.0 and 1.0"):
        FraudDecisionPolicy(review_threshold=0.20, block_threshold=1.0)

    # NaN / Inf / non-numeric
    with pytest.raises(ValueError, match="cannot be NaN"):
        FraudDecisionPolicy(review_threshold=float("nan"), block_threshold=0.50)

    with pytest.raises(TypeError, match="must be numerical"):
        FraudDecisionPolicy(review_threshold="0.10", block_threshold=0.40)  # type: ignore


def test_below_threshold_values(default_policy: FraudDecisionPolicy):
    """Verifies that values strictly below review threshold map to ALLOW."""
    test_scores = [0.0, 0.01, 0.05, 0.09, 0.09999]
    for s in test_scores:
        res = default_policy.evaluate(s)
        assert res.recommendation == DecisionRecommendation.ALLOW
        assert res.recommendation.value == "allow"
        assert res.action == "Automated Approval"
        assert res.score == pytest.approx(s, abs=1e-5)


def test_equal_to_threshold_values(default_policy: FraudDecisionPolicy):
    """Verifies that values exactly equal to threshold boundaries trigger the higher risk tier."""
    # Exactly equal to review threshold -> REVIEW
    res_review = default_policy.evaluate(0.10)
    assert res_review.recommendation == DecisionRecommendation.REVIEW
    assert res_review.recommendation.value == "review"
    assert res_review.action == "Manual Review / Step-up Verification"

    # Exactly equal to block threshold -> HIGH_RISK
    res_block = default_policy.evaluate(0.40)
    assert res_block.recommendation == DecisionRecommendation.HIGH_RISK
    assert res_block.recommendation.value == "high_risk"
    assert res_block.action == "Automated Decline"


def test_above_threshold_values(default_policy: FraudDecisionPolicy):
    """Verifies that values above thresholds map to the appropriate tier."""
    # In review tier: 0.10 <= score < 0.40
    review_scores = [0.10001, 0.15, 0.22, 0.30, 0.39999]
    for s in review_scores:
        res = default_policy.evaluate(s)
        assert res.recommendation == DecisionRecommendation.REVIEW
        assert res.recommendation.value == "review"

    # In high risk tier: score >= 0.40
    high_risk_scores = [0.40001, 0.50, 0.75, 0.90, 1.0]
    for s in high_risk_scores:
        res = default_policy.evaluate(s)
        assert res.recommendation == DecisionRecommendation.HIGH_RISK
        assert res.recommendation.value == "high_risk"
        assert res.action == "Automated Decline"


def test_strict_determinism(default_policy: FraudDecisionPolicy):
    """Verifies that the same score always maps to the same recommendation across 500 iterations."""
    sample_scores = [0.0, 0.09999, 0.10, 0.25, 0.39999, 0.40, 0.85, 1.0]
    for s in sample_scores:
        initial = default_policy.evaluate(s)
        for _ in range(50):
            repeated = default_policy.evaluate(s)
            assert repeated.recommendation == initial.recommendation
            assert repeated.action == initial.action
            assert repeated.score == initial.score


def test_invalid_score_inputs(default_policy: FraudDecisionPolicy):
    """Verifies robust exception handling for invalid score inputs."""
    # Out of [0.0, 1.0]
    with pytest.raises(ValueError, match="within valid probability range"):
        default_policy.evaluate(-0.01)

    with pytest.raises(ValueError, match="within valid probability range"):
        default_policy.evaluate(1.0001)

    # NaN / Inf
    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        default_policy.evaluate(float("nan"))

    with pytest.raises(ValueError, match="cannot be NaN or Infinite"):
        default_policy.evaluate(float("inf"))

    # Non-numeric types
    with pytest.raises(TypeError, match="must be a float or integer"):
        default_policy.evaluate(None)  # type: ignore

    with pytest.raises(TypeError, match="must be a float or integer"):
        default_policy.evaluate("0.5")  # type: ignore


def test_from_yaml_configuration():
    """Verifies loading decision policy from YAML configuration file."""
    assert DEFAULT_POLICY_CONFIG_PATH.is_file()
    policy = FraudDecisionPolicy.from_yaml(DEFAULT_POLICY_CONFIG_PATH)

    assert policy.review_threshold == 0.10
    assert policy.block_threshold == 0.40
    assert policy.policy_name == "bankguard_three_tier_policy"

    # Test decision via loaded policy
    assert policy.evaluate(0.05).recommendation == DecisionRecommendation.ALLOW
    assert policy.evaluate(0.20).recommendation == DecisionRecommendation.REVIEW
    assert policy.evaluate(0.80).recommendation == DecisionRecommendation.HIGH_RISK


def test_evaluate_batch_and_vectorized_consistency(default_policy: FraudDecisionPolicy):
    """Verifies 100% equivalence between sequential batch evaluation and vectorized evaluation."""
    scores = np.linspace(0.0, 1.0, 101).astype(np.float32)

    batch_results = default_policy.evaluate_batch(scores)
    batch_labels = [r.recommendation.value for r in batch_results]

    vector_labels = default_policy.evaluate_vectorized(scores)

    assert list(batch_labels) == list(vector_labels)


def test_summarize_decisions(default_policy: FraudDecisionPolicy):
    """Verifies statistical summary and ground truth conversion metrics."""
    scores = np.array([0.02, 0.08, 0.12, 0.25, 0.45, 0.80])
    y_true = np.array([0, 0, 1, 0, 1, 1])

    summary = default_policy.summarize_decisions(scores, y_true=y_true)

    assert summary["total_transactions"] == 6
    assert summary["distribution"]["allow"]["count"] == 2
    assert summary["distribution"]["review"]["count"] == 2
    assert summary["distribution"]["high_risk"]["count"] == 2

    gt = summary["ground_truth_breakdown"]
    assert gt["total_actual_fraud"] == 3
    assert gt["total_actual_legitimate"] == 3
    # In allow (scores 0.02, 0.08): 0 frauds, 2 legit -> 0% escape
    assert gt["allow_tier_escape_rate_pct"] == 0.0
    # In high_risk (scores 0.45, 0.80): 2 frauds, 0 legit -> 100% auto decline precision
    assert gt["auto_decline_precision_pct"] == 100.0
