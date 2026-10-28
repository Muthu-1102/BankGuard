"""Operational fraud risk decision policy engine for BankGuard (Task 6.1).

Maps continuous model fraud risk scores into deterministic, configurable recommendations:
- `allow`: Automated friction-free approval for low-risk transactions.
- `review`: Step-up verification or manual analyst review for borderline transactions.
- `high_risk`: Immediate automated decline for critical high-probability fraud.

Guarantees:
- Strict determinism: The exact same score always maps to the exact same recommendation.
- Comprehensive boundary checking (below-threshold, equal-to-threshold, above-threshold).
- Configurable threshold definitions decoupled from model weights.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml


DEFAULT_POLICY_CONFIG_PATH = Path("configs/policy_config.yaml")


class DecisionRecommendation(str, Enum):
    """Enumeration of standard fraud decision recommendations."""
    ALLOW = "allow"
    REVIEW = "review"
    HIGH_RISK = "high_risk"


@dataclass(frozen=True)
class DecisionResult:
    """Immutable result of a single transaction fraud decision evaluation."""
    score: float
    recommendation: DecisionRecommendation
    action: str
    description: str
    review_threshold: float
    block_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts result into a serializable dictionary."""
        data = asdict(self)
        data["recommendation"] = self.recommendation.value
        return data


class FraudDecisionPolicy:
    """Production decision engine applying configurable risk boundaries to model scores."""

    def __init__(
        self,
        review_threshold: float = 0.10,
        block_threshold: float = 0.40,
        policy_name: str = "bankguard_three_tier_policy",
        version: str = "1.0.0",
        tier_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """Initializes decision policy with validated operational thresholds.

        Args:
            review_threshold: Lower threshold. Scores strictly below are allowed;
                              scores greater than or equal are flagged for review.
            block_threshold: Upper threshold. Scores greater than or equal are declined.
            policy_name: Descriptive name of the policy.
            version: Semantic version of the configuration.
            tier_metadata: Optional custom action and description metadata per tier.
        """
        self._validate_thresholds(review_threshold, block_threshold)
        self.review_threshold = float(review_threshold)
        self.block_threshold = float(block_threshold)
        self.policy_name = policy_name
        self.version = version

        # Default actions and descriptions
        default_meta = {
            DecisionRecommendation.ALLOW.value: {
                "action": "Automated Approval",
                "description": "Low fraud risk. Transaction processed with standard zero-friction flow.",
            },
            DecisionRecommendation.REVIEW.value: {
                "action": "Manual Review / Step-up Verification",
                "description": "Elevated fraud probability. Trigger step-up authentication or analyst review queue.",
            },
            DecisionRecommendation.HIGH_RISK.value: {
                "action": "Automated Decline",
                "description": "Critical fraud risk with high certainty. Automatically decline transaction to prevent loss.",
            },
        }
        if tier_metadata:
            for k, v in tier_metadata.items():
                if k in default_meta:
                    default_meta[k].update(v)

        self.tier_metadata = default_meta

    @staticmethod
    def _validate_thresholds(review_thresh: float, block_thresh: float) -> None:
        """Validates that thresholds satisfy mathematical and operational ordering."""
        if not isinstance(review_thresh, (int, float)) or not isinstance(block_thresh, (int, float)):
            raise TypeError("Decision thresholds must be numerical float or integer values.")

        if math.isnan(review_thresh) or math.isnan(block_thresh):
            raise ValueError("Decision thresholds cannot be NaN.")

        if not (0.0 < review_thresh < 1.0):
            raise ValueError(f"review_threshold must be strictly between 0.0 and 1.0, got {review_thresh}")

        if not (0.0 < block_thresh < 1.0):
            raise ValueError(f"block_threshold must be strictly between 0.0 and 1.0, got {block_thresh}")

        if review_thresh >= block_thresh:
            raise ValueError(
                f"review_threshold ({review_thresh}) must be strictly less than "
                f"block_threshold ({block_thresh})."
            )

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path] = DEFAULT_POLICY_CONFIG_PATH) -> "FraudDecisionPolicy":
        """Loads and instantiates policy from a YAML configuration file.

        Args:
            config_path: Path to policy_config.yaml.

        Returns:
            Configured FraudDecisionPolicy instance.
        """
        cfg_file = Path(config_path)
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Policy configuration file not found at: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        thresh = cfg.get("thresholds", {})
        rev_t = thresh.get("review_threshold", 0.10)
        blk_t = thresh.get("block_threshold", 0.40)
        name = cfg.get("policy_name", "bankguard_three_tier_policy")
        ver = cfg.get("version", "1.0.0")

        tier_meta = {}
        for tier_key in ("allow", "review", "high_risk"):
            if "tiers" in cfg and tier_key in cfg["tiers"]:
                tier_meta[tier_key] = {
                    "action": cfg["tiers"][tier_key].get("action", ""),
                    "description": cfg["tiers"][tier_key].get("description", ""),
                }

        return cls(
            review_threshold=rev_t,
            block_threshold=blk_t,
            policy_name=name,
            version=ver,
            tier_metadata=tier_meta,
        )

    def evaluate(self, score: float) -> DecisionResult:
        """Deterministically evaluates a single fraud risk probability score.

        Decision Rules:
        - score < review_threshold: ALLOW (e.g. score < 0.10)
        - review_threshold <= score < block_threshold: REVIEW (e.g. 0.10 <= score < 0.40)
        - score >= block_threshold: HIGH_RISK (e.g. score >= 0.40)

        Equal-to-Threshold Boundary Policy:
        - When score == review_threshold, it falls into REVIEW (escalate).
        - When score == block_threshold, it falls into HIGH_RISK (decline).

        Args:
            score: Fraud probability score in range [0.0, 1.0].

        Returns:
            Immutable DecisionResult.
        """
        if not isinstance(score, (int, float, np.floating, np.integer)):
            raise TypeError(f"Score must be a float or integer, got {type(score).__name__}")

        f_score = float(score)

        if math.isnan(f_score) or math.isinf(f_score):
            raise ValueError(f"Score cannot be NaN or Infinite, got {f_score}")

        if not (0.0 <= f_score <= 1.0):
            raise ValueError(f"Score must be within valid probability range [0.0, 1.0], got {f_score}")

        # Deterministic boundary mapping
        if f_score < self.review_threshold:
            rec = DecisionRecommendation.ALLOW
        elif f_score < self.block_threshold:
            rec = DecisionRecommendation.REVIEW
        else:
            rec = DecisionRecommendation.HIGH_RISK

        meta = self.tier_metadata[rec.value]
        return DecisionResult(
            score=round(f_score, 5),
            recommendation=rec,
            action=meta["action"],
            description=meta["description"],
            review_threshold=self.review_threshold,
            block_threshold=self.block_threshold,
        )

    def evaluate_batch(self, scores: Sequence[float]) -> List[DecisionResult]:
        """Evaluates a collection of risk scores sequentially.

        Args:
            scores: Sequence of numerical probability scores.

        Returns:
            List of DecisionResult objects matching input order.
        """
        return [self.evaluate(s) for s in scores]

    def evaluate_vectorized(self, scores: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Vectorized classification returning string array of recommendation labels.

        Args:
            scores: 1D numpy array of probability scores.

        Returns:
            1D numpy array of string labels ('allow', 'review', 'high_risk').
        """
        arr = np.asarray(scores, dtype=np.float32)
        if np.isnan(arr).any():
            raise ValueError("Input array contains NaN values.")
        if (arr < 0.0).any() or (arr > 1.0).any():
            raise ValueError("Input scores must be within [0.0, 1.0].")

        results = np.full(arr.shape, DecisionRecommendation.ALLOW.value, dtype=object)
        review_mask = (arr >= self.review_threshold) & (arr < self.block_threshold)
        block_mask = (arr >= self.block_threshold)

        results[review_mask] = DecisionRecommendation.REVIEW.value
        results[block_mask] = DecisionRecommendation.HIGH_RISK.value

        return results

    def summarize_decisions(
        self,
        scores: Union[np.ndarray, Sequence[float]],
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Computes aggregate distribution metrics for an array of evaluated scores.

        Args:
            scores: Array of evaluated fraud probability scores.
            y_true: Optional ground truth labels (0 = Legitimate, 1 = Fraud) for conversion analysis.

        Returns:
            Structured summary dictionary with counts, percentages, and fraud breakdown.
        """
        arr = np.asarray(scores, dtype=np.float32)
        labels = self.evaluate_vectorized(arr)
        total = len(labels)

        if total == 0:
            return {"total_transactions": 0}

        allow_cnt = int(np.sum(labels == DecisionRecommendation.ALLOW.value))
        review_cnt = int(np.sum(labels == DecisionRecommendation.REVIEW.value))
        block_cnt = int(np.sum(labels == DecisionRecommendation.HIGH_RISK.value))

        summary = {
            "policy_name": self.policy_name,
            "version": self.version,
            "thresholds": {
                "review_threshold": self.review_threshold,
                "block_threshold": self.block_threshold,
            },
            "total_transactions": total,
            "distribution": {
                "allow": {
                    "count": allow_cnt,
                    "percentage": round(allow_cnt / total * 100.0, 2),
                    "action": self.tier_metadata["allow"]["action"],
                },
                "review": {
                    "count": review_cnt,
                    "percentage": round(review_cnt / total * 100.0, 2),
                    "action": self.tier_metadata["review"]["action"],
                },
                "high_risk": {
                    "count": block_cnt,
                    "percentage": round(block_cnt / total * 100.0, 2),
                    "action": self.tier_metadata["high_risk"]["action"],
                },
            },
        }

        if y_true is not None:
            y_arr = np.asarray(y_true, dtype=int)
            total_fraud = int(np.sum(y_arr == 1))
            total_legit = int(np.sum(y_arr == 0))

            fraud_in_allow = int(np.sum((labels == "allow") & (y_arr == 1)))
            fraud_in_review = int(np.sum((labels == "review") & (y_arr == 1)))
            fraud_in_block = int(np.sum((labels == "high_risk") & (y_arr == 1)))

            legit_in_allow = int(np.sum((labels == "allow") & (y_arr == 0)))
            legit_in_review = int(np.sum((labels == "review") & (y_arr == 0)))
            legit_in_block = int(np.sum((labels == "high_risk") & (y_arr == 0)))

            summary["ground_truth_breakdown"] = {
                "total_actual_fraud": total_fraud,
                "total_actual_legitimate": total_legit,
                "fraud_capture_rate_pct": round((fraud_in_review + fraud_in_block) / total_fraud * 100.0, 2) if total_fraud else 0.0,
                "auto_decline_precision_pct": round(fraud_in_block / block_cnt * 100.0, 2) if block_cnt else 0.0,
                "review_queue_fraud_density_pct": round(fraud_in_review / review_cnt * 100.0, 2) if review_cnt else 0.0,
                "allow_tier_escape_rate_pct": round(fraud_in_allow / allow_cnt * 100.0, 3) if allow_cnt else 0.0,
                "matrix": {
                    "allow": {"actual_legitimate": legit_in_allow, "actual_fraud": fraud_in_allow},
                    "review": {"actual_legitimate": legit_in_review, "actual_fraud": fraud_in_review},
                    "high_risk": {"actual_legitimate": legit_in_block, "actual_fraud": fraud_in_block},
                },
            }

        return summary


def load_default_policy() -> FraudDecisionPolicy:
    """Convenience helper to load the default configured decision policy."""
    if DEFAULT_POLICY_CONFIG_PATH.is_file():
        return FraudDecisionPolicy.from_yaml(DEFAULT_POLICY_CONFIG_PATH)
    return FraudDecisionPolicy(review_threshold=0.10, block_threshold=0.40)


def generate_policy_report(
    summary: Dict[str, Any],
    output_markdown_path: Union[str, Path] = "report/decision_policy_report.md",
    output_json_path: Union[str, Path] = "report/decision_policy_report.json",
) -> Tuple[Path, Path]:
    """Saves structured markdown and JSON reports documenting decision policy performance."""
    md_file = Path(output_markdown_path)
    json_file = Path(output_json_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    dist = summary["distribution"]
    thresh = summary["thresholds"]
    gt = summary.get("ground_truth_breakdown", {})

    gt_section = ""
    if gt:
        mat = gt["matrix"]
        gt_section = f"""
### Validation Set Ground Truth Breakdown

| Decision Tier | Total Flagged | Actual Legitimate | Actual Fraud | Tier Fraud Density (%) | Operational Fate |
|---|---:|---:|---:|---:|---|
| **`allow`** | {dist['allow']['count']:,} ({dist['allow']['percentage']}%) | {mat['allow']['actual_legitimate']:,} | {mat['allow']['actual_fraud']:,} | {gt['allow_tier_escape_rate_pct']:.2f}% | Zero-friction authorization |
| **`review`** | {dist['review']['count']:,} ({dist['review']['percentage']}%) | {mat['review']['actual_legitimate']:,} | {mat['review']['actual_fraud']:,} | {gt['review_queue_fraud_density_pct']:.2f}% | Challenge via 3DS / Analyst Queue |
| **`high_risk`** | {dist['high_risk']['count']:,} ({dist['high_risk']['percentage']}%) | {mat['high_risk']['actual_legitimate']:,} | {mat['high_risk']['actual_fraud']:,} | **{gt['auto_decline_precision_pct']:.2f}%** | Automated Immediate Decline |

- **Total Fraud Captured (Review + Block):** **{gt['fraud_capture_rate_pct']:.2f}%** ({mat['review']['actual_fraud'] + mat['high_risk']['actual_fraud']:,} out of {gt['total_actual_fraud']:,} fraud transactions).
- **Auto-Decline Precision:** **{gt['auto_decline_precision_pct']:.2f}%** of automatically blocked transactions are verified fraud.
- **Escape Rate in Allow Tier:** Only **{gt['allow_tier_escape_rate_pct']:.3f}%** of allowed transactions are fraudulent.
"""

    md_content = f"""# BankGuard — Operational Decision Policy Report (Task 6.1)

## 1. Policy Overview & Architecture

The BankGuard Decision Policy Engine translates continuous fraud probability predictions $P(\\text{{fraud}} \\mid X) \\in [0.0, 1.0]$ into actionable operational recommendations.

- **Policy Name:** `{summary['policy_name']}` (v{summary['version']})
- **Review Threshold ($\\tau_{{\\text{{review}}}}$):** **`{thresh['review_threshold']:.2f}`**
- **Block Threshold ($\\tau_{{\\text{{block}}}}$):** **`{thresh['block_threshold']:.2f}`**

### Decision Boundary Definitions

| Tier | Mathematical Range | Action | Operational Target |
|---|---|---|---|
| **`allow`** | $\\text{{score}} < {thresh['review_threshold']:.2f}$ | **Automated Approval** | Fast-path authorization for ultra-low risk transactions |
| **`review`** | ${thresh['review_threshold']:.2f} \\le \\text{{score}} < {thresh['block_threshold']:.2f}$ | **Step-Up Verification** | Queue for human review or trigger dynamic 3D-Secure challenge |
| **`high_risk`** | $\\text{{score}} \\ge {thresh['block_threshold']:.2f}$ | **Automated Decline** | Immediate hard decline on high-certainty fraud |

---

## 2. Validation Set Distribution ({summary['total_transactions']:,} Out-of-Time Transactions)

| Decision Recommendation | Transaction Count | Volume Share (%) | Standard Operational Action |
|---|---:|---:|---|
| **`allow`** | **{dist['allow']['count']:,}** | **{dist['allow']['percentage']:.2f}%** | {dist['allow']['action']} |
| **`review`** | **{dist['review']['count']:,}** | **{dist['review']['percentage']:.2f}%** | {dist['review']['action']} |
| **`high_risk`** | **{dist['high_risk']['count']:,}** | **{dist['high_risk']['percentage']:.2f}%** | {dist['high_risk']['action']} |
{gt_section}
---

## 3. Operational Guarantees & Implementation Notes

1. **Strict Determinism:** The decision function is stateless and deterministic; identical input scores produce identical recommendations and actions.
2. **Boundary Precision:** Exactly equal-to-threshold scores trigger the higher risk tier:
   - $\\text{{score}} == {thresh['review_threshold']:.2f} \\implies$ `review`
   - $\\text{{score}} == {thresh['block_threshold']:.2f} \\implies$ `high_risk`
3. **Decoupled Configuration:** Thresholds are loaded from `configs/policy_config.yaml` at runtime, allowing operational teams to tune capacity constraints without code redeployment.
"""

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_file, json_file


def run_decision_policy_pipeline(
    predictions_path: Union[str, Path] = "artifacts/improved_predictions.npz",
    config_path: Union[str, Path] = DEFAULT_POLICY_CONFIG_PATH,
    output_dir: Union[str, Path] = "report",
) -> Dict[str, Any]:
    """Runs end-to-end policy evaluation on the saved validation predictions."""
    out_path = Path(output_dir)
    pred_file = Path(predictions_path)

    policy = FraudDecisionPolicy.from_yaml(config_path)

    if not pred_file.is_file():
        raise FileNotFoundError(f"Predictions file not found at: {pred_file}")

    data = np.load(pred_file)
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    print(f"Evaluating decision policy '{policy.policy_name}' on {len(y_prob):,} validation predictions...")
    summary = policy.summarize_decisions(y_prob, y_true=y_true)

    md_path, json_path = generate_policy_report(
        summary=summary,
        output_markdown_path=out_path / "decision_policy_report.md",
        output_json_path=out_path / "decision_policy_report.json",
    )

    print(
        f"\nDecision Policy Pipeline successfully executed:\n"
        f"  - Policy: {policy.policy_name} (v{policy.version})\n"
        f"  - Review Threshold: {policy.review_threshold:.2f} | Block Threshold: {policy.block_threshold:.2f}\n"
        f"  - Allow Volume: {summary['distribution']['allow']['count']:,} ({summary['distribution']['allow']['percentage']} %)\n"
        f"  - Review Volume: {summary['distribution']['review']['count']:,} ({summary['distribution']['review']['percentage']} %)\n"
        f"  - Block Volume: {summary['distribution']['high_risk']['count']:,} ({summary['distribution']['high_risk']['percentage']} %)\n"
        f"  - Report: {md_path}\n"
        f"  - Structured Data: {json_path}"
    )

    return {
        "policy": policy,
        "summary": summary,
        "reports": {
            "markdown": str(md_path),
            "json": str(json_path),
        },
    }


if __name__ == "__main__":
    run_decision_policy_pipeline()
