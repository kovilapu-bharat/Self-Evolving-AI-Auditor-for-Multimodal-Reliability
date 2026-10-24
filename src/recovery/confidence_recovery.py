"""
Confidence-Aware Decision and Fallback Recovery for SEAA.
Improvement #9: Adjusts decision thresholds, performs selective classification / prediction set abstention,
and manages human-in-the-loop review queues for unreliable inputs.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class ConfidenceRecoveryRouter:
    """
    Confidence-aware prediction set adjuster and human-review triage router.
    """
    def __init__(
        self,
        confidence_abstain_threshold: float = 0.35,
        elevate_threshold_factor: float = 1.15
    ):
        self.confidence_abstain_threshold = confidence_abstain_threshold
        self.elevate_threshold_factor = elevate_threshold_factor

    def apply_selective_abstention(
        self,
        probs: np.ndarray,
        base_thresholds: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Elevates decision thresholds for samples with high anomaly scores to prevent
        hallucinated false positives, and flags borderline samples for human audit.

        Args:
            probs: Predicted probabilities (N, C)
            base_thresholds: Calibrated decision thresholds (C,)
            anomaly_scores: Continuous anomaly scores in [0, 1] from auditor (N,)

        Returns:
            Dictionary containing:
            - 'binary_predictions': (N, C) adjusted binary labels
            - 'human_review_required': (N,) boolean mask
            - 'adjusted_thresholds': (N, C) sample-specific decision thresholds
        """
        n_samples, n_classes = probs.shape
        adjusted_thresh = np.tile(base_thresholds, (n_samples, 1))

        # Elevate thresholds proportionally to anomaly score
        for i in range(n_samples):
            a_score = anomaly_scores[i]
            if a_score > 0.5:
                scale = 1.0 + (self.elevate_threshold_factor - 1.0) * a_score
                adjusted_thresh[i] = np.clip(adjusted_thresh[i] * scale, 0.05, 0.95)

        binary_preds = (probs >= adjusted_thresh).astype(int)

        # Flag for human review if max probability is below abstain threshold or anomaly is critical (>0.85)
        max_prob = np.max(probs, axis=1)
        human_review = (max_prob < self.confidence_abstain_threshold) | (anomaly_scores > 0.85)

        return {
            "binary_predictions": binary_preds,
            "human_review_required": human_review,
            "adjusted_thresholds": adjusted_thresh
        }
