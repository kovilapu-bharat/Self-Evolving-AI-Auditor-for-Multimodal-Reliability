"""
Continuous-Context Bandit Policies for SEAA Multimodal Reliability Recovery.

Implements advanced contextual bandit formulations that operate directly on
continuous feature representations (disagreement, uncertainty, feature norms)
without discrete binning:
1. LinUCBPolicy (Disjoint Ridge Regression with Upper Confidence Bounds, Li et al. 2010)
2. ContextualThompsonSamplingPolicy (Bayesian Linear Regression with Gaussian Posterior Sampling, Agrawal & Goyal 2013)

Both policies strictly adhere to:
- Physical feasibility masking (enforcing modality availability)
- Outcome tracking with multi-label F1 utility
- Deterministic frozen deployment during test evaluation (zero test label leakage)
- Fast Sherman-Morrison rank-1 updates for sub-millisecond decision latency
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import logging
from pathlib import Path
import numpy as np

from src.policy.adaptive_policy import (
    ALL_ACTIONS,
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_IMAGE_ONLY,
    ACTION_RELIABILITY_WEIGHTED,
    ACTION_CONSERVATIVE_FUSION,
    ACTION_PRIORITY,
    PolicyContext
)

logger = logging.getLogger("ContinuousBandits")


def extract_continuous_context_vector(ctx: PolicyContext) -> np.ndarray:
    """
    Extracts a 9-dimensional continuous feature representation from the observable PolicyContext.
    Ground-truth labels and corruption flags are strictly excluded.
    
    Features:
    [0] Bias term (1.0)
    [1] Text modality presence (1.0 or 0.0)
    [2] Image modality presence (1.0 or 0.0)
    [3] Text predictive entropy H(p^T)
    [4] Image predictive entropy H(p^V)
    [5] Entropy asymmetry (H(p^V) - H(p^T))
    [6] Bernoulli Jensen-Shannon Divergence D_JS
    [7] Normalized text feature norm ||z^T||_2
    [8] Normalized image feature norm ||z^V||_2
    """
    has_t = 1.0 if ctx.has_text else 0.0
    has_v = 1.0 if ctx.has_image else 0.0
    h_t = float(ctx.text_entropy)
    h_v = float(ctx.image_entropy)
    h_diff = h_v - h_t
    jsd = float(ctx.bernoulli_jsd)
    norm_t = min(float(ctx.text_norm) / 50.0, 2.0)  # normalized
    norm_v = min(float(ctx.image_norm) / 50.0, 2.0)  # normalized

    return np.array([
        1.0,
        has_t,
        has_v,
        h_t,
        h_v,
        h_diff,
        jsd,
        norm_t,
        norm_v
    ], dtype=np.float64)


class LinUCBPolicy:
    """
    LinUCB Contextual Bandit with disjoint linear models per recovery action.
    Maintains A_a and b_a for each action, with an inverse covariance matrix A_inv_a
    updated via Sherman-Morrison rank-1 updates for ultra-low latency.
    """
    def __init__(
        self,
        d: int = 9,
        alpha: float = 0.5,
        ridge_lambda: float = 1.0,
        default_fusion_alpha: float = 0.55,
        seed: int = 42,
        learning_enabled: bool = True
    ):
        self.d = d
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self.default_fusion_alpha = default_fusion_alpha
        self.seed = seed
        self.learning_enabled = learning_enabled

        self.rng = np.random.RandomState(seed)
        self.actions = list(ALL_ACTIONS)

        # Initialize disjoint ridge regression parameters for each action
        self.A = {a: np.eye(d, dtype=np.float64) * ridge_lambda for a in self.actions}
        self.A_inv = {a: np.eye(d, dtype=np.float64) / ridge_lambda for a in self.actions}
        self.b = {a: np.zeros(d, dtype=np.float64) for a in self.actions}
        self.counts = {a: 0 for a in self.actions}

    def select_action(
        self,
        ctx: PolicyContext,
        frozen: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Selects optimal recovery action via UCB scoring subject to physical modality feasibility.
        If frozen=True (test mode), uses purely greedy expected value theta^T x without exploration bonus.
        """
        valid_actions = ctx.get_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0], {"ucb_scores": {valid_actions[0]: 1.0}, "strategy": "deterministic_single"}

        x = extract_continuous_context_vector(ctx)
        best_action = None
        best_score = -float("inf")
        scores = {}

        for a in valid_actions:
            theta_a = self.A_inv[a] @ self.b[a]
            expected_val = float(np.dot(theta_a, x))

            if frozen or not self.learning_enabled or self.alpha == 0.0:
                score = expected_val
            else:
                var = float(x.T @ self.A_inv[a] @ x)
                uncertainty = self.alpha * np.sqrt(max(0.0, var))
                score = expected_val + uncertainty

            scores[a] = score

            # Tie breaking using predefined deterministic action priority
            if (score > best_score) or (
                np.isclose(score, best_score, atol=1e-7) and
                ACTION_PRIORITY.get(a, 99) < ACTION_PRIORITY.get(best_action, 99)
            ):
                best_score = score
                best_action = a

        meta = {
            "strategy": "greedy" if frozen else "linucb",
            "scores": scores,
            "chosen_score": best_score,
            "context_dim": self.d
        }
        return best_action, meta

    def update(self, ctx: PolicyContext, action: str, reward: float) -> None:
        """
        Updates LinUCB parameters using Sherman-Morrison rank-1 update.
        Strictly skipped if learning_enabled is False.
        """
        if not self.learning_enabled:
            return

        x = extract_continuous_context_vector(ctx)
        
        # 1. Update b_a
        self.b[action] += reward * x
        self.counts[action] += 1

        # 2. Update A_a
        self.A[action] += np.outer(x, x)

        # 3. Fast Sherman-Morrison rank-1 update for A_inv:
        # (A + x x^T)^(-1) = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        A_inv_prev = self.A_inv[action]
        inv_x = A_inv_prev @ x
        denom = 1.0 + float(np.dot(x, inv_x))
        if denom > 1e-12:
            self.A_inv[action] = A_inv_prev - np.outer(inv_x, inv_x) / denom
        else:
            self.A_inv[action] = np.linalg.pinv(self.A[action])

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes LinUCB model parameters to JSON."""
        data = {
            "policy_type": "LinUCBPolicy",
            "d": self.d,
            "alpha": self.alpha,
            "ridge_lambda": self.ridge_lambda,
            "counts": self.counts,
            "theta": {a: (self.A_inv[a] @ self.b[a]).tolist() for a in self.actions},
            "b": {a: self.b[a].tolist() for a in self.actions},
            "A_diag": {a: np.diag(self.A[a]).tolist() for a in self.actions}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads serialized model parameters."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.d = data.get("d", self.d)
        self.alpha = data.get("alpha", self.alpha)
        self.counts = data.get("counts", self.counts)
        for a in self.actions:
            if a in data["b"]:
                self.b[a] = np.array(data["b"][a], dtype=np.float64)


class ContextualThompsonSamplingPolicy:
    """
    Contextual Gaussian Thompson Sampling with continuous context features.
    Samples weight vectors theta_tilde ~ N(theta_hat, v^2 B^-1) for posterior exploration.
    Freezes to deterministic MAP prediction (theta_hat) during test deployment.
    """
    def __init__(
        self,
        d: int = 9,
        v: float = 0.25,
        ridge_lambda: float = 1.0,
        seed: int = 42,
        learning_enabled: bool = True
    ):
        self.d = d
        self.v = v
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.learning_enabled = learning_enabled

        self.rng = np.random.RandomState(seed)
        self.actions = list(ALL_ACTIONS)

        self.B = {a: np.eye(d, dtype=np.float64) * ridge_lambda for a in self.actions}
        self.B_inv = {a: np.eye(d, dtype=np.float64) / ridge_lambda for a in self.actions}
        self.f = {a: np.zeros(d, dtype=np.float64) for a in self.actions}
        self.counts = {a: 0 for a in self.actions}

    def select_action(
        self,
        ctx: PolicyContext,
        frozen: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Selects action using posterior Thompson sample or greedy MAP under frozen mode.
        """
        valid_actions = ctx.get_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0], {"strategy": "deterministic_single"}

        x = extract_continuous_context_vector(ctx)
        best_action = None
        best_score = -float("inf")
        scores = {}

        for a in valid_actions:
            theta_hat = self.B_inv[a] @ self.f[a]
            if frozen or not self.learning_enabled:
                score = float(np.dot(theta_hat, x))
            else:
                # Sample posterior parameter vector
                cov = (self.v ** 2) * self.B_inv[a]
                # Ensure positive semi-definite
                cov = (cov + cov.T) / 2.0
                theta_sample = self.rng.multivariate_normal(theta_hat, cov)
                score = float(np.dot(theta_sample, x))

            scores[a] = score
            if (score > best_score) or (
                np.isclose(score, best_score, atol=1e-7) and
                ACTION_PRIORITY.get(a, 99) < ACTION_PRIORITY.get(best_action, 99)
            ):
                best_score = score
                best_action = a

        return best_action, {"strategy": "map" if frozen else "thompson_sampling", "scores": scores}

    def update(self, ctx: PolicyContext, action: str, reward: float) -> None:
        """Updates Bayesian posterior linear regression parameters."""
        if not self.learning_enabled:
            return

        x = extract_continuous_context_vector(ctx)
        self.f[action] += reward * x
        self.counts[action] += 1
        self.B[action] += np.outer(x, x)

        # Sherman-Morrison update
        B_inv_prev = self.B_inv[action]
        inv_x = B_inv_prev @ x
        denom = 1.0 + float(np.dot(x, inv_x))
        if denom > 1e-12:
            self.B_inv[action] = B_inv_prev - np.outer(inv_x, inv_x) / denom
        else:
            self.B_inv[action] = np.linalg.pinv(self.B[action])

