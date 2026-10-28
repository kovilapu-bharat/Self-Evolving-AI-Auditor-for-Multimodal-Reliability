"""
Adaptive Recovery Policy for SEAA.
Implements an outcome-learning contextual decision policy for multimodal failure recovery.

Features:
- Interpretable Contextual Action-Value Table with UCB / epsilon-greedy exploration.
- Supports 5 canonical recovery actions with strict modality feasibility masking.
- Outcome tracking with multi-label F1 utility and relative recovery reward calculation.
- Deterministic action selection under frozen evaluation mode (zero test label leakage).
- Full audit telemetry logging of context, actions, rewards, and parameter updates.
- Save / load serialization for reproducible benchmarking.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger("AdaptiveRecoveryPolicy")

# Supported Canonical Recovery Actions
ACTION_STATIC_FUSION = "STATIC_FUSION"
ACTION_TEXT_ONLY = "TEXT_ONLY"
ACTION_IMAGE_ONLY = "IMAGE_ONLY"
ACTION_RELIABILITY_WEIGHTED = "RELIABILITY_WEIGHTED"
ACTION_CONSERVATIVE_FUSION = "CONSERVATIVE_FUSION"

ALL_ACTIONS = [
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_IMAGE_ONLY,
    ACTION_RELIABILITY_WEIGHTED,
    ACTION_CONSERVATIVE_FUSION
]

# Deterministic tie-breaking priority
ACTION_PRIORITY = {
    ACTION_STATIC_FUSION: 0,
    ACTION_RELIABILITY_WEIGHTED: 1,
    ACTION_CONSERVATIVE_FUSION: 2,
    ACTION_TEXT_ONLY: 3,
    ACTION_IMAGE_ONLY: 4
}


@dataclass
class PolicyContext:
    """
    Observable context signals available to the policy strictly at decision time.
    Ground-truth labels and corruption flags are never included.
    """
    has_text: bool
    has_image: bool
    text_entropy: float = 0.5
    image_entropy: float = 0.5
    bernoulli_jsd: float = 0.0
    hamming_distance: float = 0.0
    text_norm: float = 1.0
    image_norm: float = 1.0
    diagnosis: str = "CLEAN"

    def get_state_key(self, jsd_threshold: float = 0.1139) -> str:
        """
        Discretizes continuous inference signals into an interpretable state tuple.
        Returns string representation: '(presence, disagreement, uncertainty_bias, diagnosis)'
        """
        # 1. Modality Presence
        if not self.has_text and not self.has_image:
            presence = "MISSING_BOTH"
        elif not self.has_text:
            presence = "TEXT_MISSING"
        elif not self.has_image:
            presence = "IMAGE_MISSING"
        else:
            presence = "BOTH_PRESENT"

        # 2. Cross-Modal Disagreement
        disagreement = "HIGH_JSD" if self.bernoulli_jsd > jsd_threshold else "LOW_JSD"

        # 3. Uncertainty Asymmetry
        unc_diff = self.image_entropy - self.text_entropy  # Positive => Image more uncertain => Text safer
        if unc_diff > 0.10:
            unc_bias = "TEXT_MORE_RELIABLE"
        elif unc_diff < -0.10:
            unc_bias = "IMAGE_MORE_RELIABLE"
        else:
            unc_bias = "BALANCED_UNCERTAINTY"

        # 4. Clean Diagnostic Tag
        clean_diag = self.diagnosis if self.diagnosis else "UNSPECIFIED"

        return f"({presence}|{disagreement}|{unc_bias}|{clean_diag})"

    def get_valid_actions(self) -> List[str]:
        """
        Determines admissible actions based on physical modality availability.
        """
        valid = []
        if self.has_text and self.has_image:
            valid = list(ALL_ACTIONS)
        elif self.has_text and not self.has_image:
            # Image is unavailable; can only use pure text
            valid = [ACTION_TEXT_ONLY]
        elif not self.has_text and self.has_image:
            # Text is unavailable; can only use pure image
            valid = [ACTION_IMAGE_ONLY]
        else:
            # Neither modality is available
            valid = [ACTION_CONSERVATIVE_FUSION]
        return valid


class AdaptiveRecoveryPolicy:
    """
    Contextual Action-Value Policy for Adaptive Multimodal Reliability Recovery.
    Learns state-action values Q(s, a) from observed recovery outcomes.
    """
    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_or_decay: float = 0.5,
        epsilon: float = 0.10,
        ucb_c: float = 0.5,
        default_alpha: float = 0.55,
        jsd_threshold: float = 0.1139,
        seed: int = 42,
        learning_enabled: bool = True
    ):
        self.learning_rate = learning_rate
        self.discount_or_decay = discount_or_decay
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.default_alpha = default_alpha
        self.jsd_threshold = jsd_threshold
        self.seed = seed
        self.learning_enabled = learning_enabled

        self.rng = np.random.RandomState(seed)

        # Tabular Q-values and visit counts: state_key -> {action: float}
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.visit_counts: Dict[str, Dict[str, int]] = {}

        # History log for auditability and verification
        self.update_history: List[Dict[str, Any]] = []

    def _init_state_if_needed(self, state_key: str, valid_actions: List[str]) -> None:
        """Initializes unvisited states with neutral priors."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            self.visit_counts[state_key] = {}
            for a in ALL_ACTIONS:
                # Default optimistic / neutral prior: 0.50
                # On clean inputs, give slight initial preference to static fusion (0.55)
                prior = 0.55 if (a == ACTION_STATIC_FUSION and "CLEAN" in state_key) else 0.50
                self.q_table[state_key][a] = prior
                self.visit_counts[state_key][a] = 0

    def select_action(
        self,
        context: PolicyContext,
        explore: bool = False
    ) -> str:
        """
        Selects an admissible recovery action for the given context.

        Args:
            context: Observable signals for the current input
            explore: Whether to apply exploration (UCB / epsilon-greedy)

        Returns:
            Selected action name string
        """
        state_key = context.get_state_key(self.jsd_threshold)
        valid_actions = context.get_valid_actions()
        self._init_state_if_needed(state_key, valid_actions)

        # If only one action is feasible, select it directly
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Exploration mode (during training / policy calibration)
        if explore and self.learning_enabled:
            # Epsilon-greedy random action
            if self.rng.uniform(0.0, 1.0) < self.epsilon:
                return str(self.rng.choice(valid_actions))

            # UCB Action Selection
            total_visits = sum(self.visit_counts[state_key][a] for a in valid_actions)
            scores = {}
            for a in valid_actions:
                q = self.q_table[state_key][a]
                n = self.visit_counts[state_key][a]
                bonus = self.ucb_c * np.sqrt(np.log(total_visits + 1.0) / (n + 1e-4))
                scores[a] = q + bonus

            best_score = max(scores[a] for a in valid_actions)
            best_actions = [a for a in valid_actions if abs(scores[a] - best_score) < 1e-7]
            # Tie-break deterministically
            best_actions.sort(key=lambda a: ACTION_PRIORITY.get(a, 99))
            return best_actions[0]

        # Exploitation mode (deterministic evaluation)
        best_q = -float("inf")
        best_action = valid_actions[0]
        # Sort valid actions by deterministic priority
        sorted_valid = sorted(valid_actions, key=lambda a: ACTION_PRIORITY.get(a, 99))

        for a in sorted_valid:
            q = self.q_table[state_key][a]
            if q > best_q:
                best_q = q
                best_action = a

        return best_action

    def apply_action(
        self,
        action: str,
        text_probs: np.ndarray,
        image_probs: np.ndarray,
        context: PolicyContext
    ) -> Tuple[np.ndarray, float]:
        """
        Applies chosen action to generate recovered probabilities and effective text weight alpha.

        Args:
            action: Action string from ALL_ACTIONS
            text_probs: (C,) or (N, C) text branch predictions
            image_probs: (C,) or (N, C) image branch predictions
            context: Observable context

        Returns:
            Tuple of (recovered_probs, effective_alpha)
        """
        # Safety fallback if modality is missing
        if not context.has_text and not context.has_image:
            # Conservative prior fallback: uniform 0.50
            return 0.5 * text_probs + 0.5 * image_probs, 0.50

        if not context.has_text:
            return image_probs.copy(), 0.0

        if not context.has_image:
            return text_probs.copy(), 1.0

        if action == ACTION_TEXT_ONLY:
            return text_probs.copy(), 1.0

        elif action == ACTION_IMAGE_ONLY:
            return image_probs.copy(), 0.0

        elif action == ACTION_STATIC_FUSION:
            fused = self.default_alpha * text_probs + (1.0 - self.default_alpha) * image_probs
            return fused, self.default_alpha

        elif action == ACTION_CONSERVATIVE_FUSION:
            fused = 0.50 * text_probs + 0.50 * image_probs
            return fused, 0.50

        elif action == ACTION_RELIABILITY_WEIGHTED:
            # Dynamic continuous gating using uncertainty shift
            unc_diff = context.image_entropy - context.text_entropy  # Image unc minus text unc
            # Positive shift increases text weight
            shift = 0.40 * float(np.tanh(3.0 * unc_diff))
            dyn_alpha = float(np.clip(self.default_alpha + shift, 0.05, 0.95))
            fused = dyn_alpha * text_probs + (1.0 - dyn_alpha) * image_probs
            return fused, dyn_alpha

        else:
            # Unknown action safe fallback: default fusion
            fused = self.default_alpha * text_probs + (1.0 - self.default_alpha) * image_probs
            return fused, self.default_alpha

    def compute_reward(
        self,
        recovered_probs: np.ndarray,
        unrecovered_probs: np.ndarray,
        y_true: np.ndarray,
        thresholds: np.ndarray,
        relative_weight: float = 0.5
    ) -> float:
        """
        Calculates instance recovery reward from outcome evaluation against ground truth.

        R = F1(rec, y) + lambda * [F1(rec, y) - F1(unrec, y)]

        Args:
            recovered_probs: (C,) recovered prediction probabilities
            unrecovered_probs: (C,) static late fusion probabilities
            y_true: (C,) ground truth multi-label vector
            thresholds: (C,) decision thresholds
            relative_weight: lambda weight for relative gain

        Returns:
            Scalar reward R
        """
        pred_rec = (recovered_probs >= thresholds).astype(int)
        pred_unrec = (unrecovered_probs >= thresholds).astype(int)
        y_int = y_true.astype(int)

        def instance_f1(pred: np.ndarray, target: np.ndarray) -> float:
            tp = np.sum((pred == 1) & (target == 1))
            fp = np.sum((pred == 1) & (target == 0))
            fn = np.sum((pred == 0) & (target == 1))
            denom = 2 * tp + fp + fn
            if denom == 0:
                # If neither prediction nor target has active genres, agreement is 1.0
                return 1.0
            return float(2.0 * tp / denom)

        f1_rec = instance_f1(pred_rec, y_int)
        f1_unrec = instance_f1(pred_unrec, y_int)

        reward = f1_rec + relative_weight * (f1_rec - f1_unrec)
        return float(np.clip(reward, -0.5, 1.5))

    def update(
        self,
        context: PolicyContext,
        action: str,
        reward: float
    ) -> Dict[str, Any]:
        """
        Updates internal action value Q(s, a) from observed recovery outcome.

        Update equation:
        Q_{t+1}(s, a) = Q_t(s, a) + alpha_t * [R - Q_t(s, a)]
        alpha_t = max(0.05, 1.0 / (N(s, a) + 1)^decay)

        Args:
            context: Observable context
            action: Action executed
            reward: Observed scalar reward

        Returns:
            Dictionary summarizing update telemetry
        """
        if not self.learning_enabled:
            return {"updated": False, "reason": "learning_disabled"}

        state_key = context.get_state_key(self.jsd_threshold)
        valid_actions = context.get_valid_actions()
        self._init_state_if_needed(state_key, valid_actions)

        old_q = self.q_table[state_key][action]
        n = self.visit_counts[state_key][action] + 1
        self.visit_counts[state_key][action] = n

        # Adaptive learning rate decaying with visit count
        lr = max(0.05, self.learning_rate / (float(n) ** self.discount_or_decay))
        new_q = old_q + lr * (reward - old_q)
        self.q_table[state_key][action] = float(new_q)

        update_record = {
            "step": len(self.update_history) + 1,
            "state_key": state_key,
            "action": action,
            "reward": round(float(reward), 4),
            "old_q": round(float(old_q), 4),
            "new_q": round(float(new_q), 4),
            "delta_q": round(float(new_q - old_q), 4),
            "visit_count": n,
            "effective_lr": round(float(lr), 4)
        }
        self.update_history.append(update_record)
        return update_record

    def save_policy(self, filepath: Union[str, Path]) -> None:
        """Saves complete policy configuration and learned Q-values to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_name": "AdaptiveRecoveryPolicy",
            "learning_rate": self.learning_rate,
            "discount_or_decay": self.discount_or_decay,
            "epsilon": self.epsilon,
            "ucb_c": self.ucb_c,
            "default_alpha": self.default_alpha,
            "jsd_threshold": self.jsd_threshold,
            "seed": self.seed,
            "learning_enabled": self.learning_enabled,
            "total_updates": len(self.update_history),
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "update_history": self.update_history
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved policy state to {path} ({len(self.q_table)} states, {len(self.update_history)} updates)")

    @classmethod
    def load_policy(cls, filepath: Union[str, Path]) -> "AdaptiveRecoveryPolicy":
        """Loads saved policy state from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        policy = cls(
            learning_rate=payload["learning_rate"],
            discount_or_decay=payload["discount_or_decay"],
            epsilon=payload["epsilon"],
            ucb_c=payload["ucb_c"],
            default_alpha=payload["default_alpha"],
            jsd_threshold=payload["jsd_threshold"],
            seed=payload["seed"],
            learning_enabled=payload["learning_enabled"]
        )
        policy.q_table = payload["q_table"]
        policy.visit_counts = payload["visit_counts"]
        policy.update_history = payload.get("update_history", [])
        return policy
