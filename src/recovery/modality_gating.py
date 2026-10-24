"""
Dynamic Modality Gating and Fallback Recovery for SEAA (Phase 5 Recovery).
Improvement #9: Implements adaptive recovery strategies that mitigate multimodal degradation
based on diagnostic labels and reliability signals from MultimodalReliabilityAuditor:

Recovery Rules:
- 'CLEAN': Retain standard validation-tuned late fusion.
- 'TEXT_FAILURE' or 'MISSING_TEXT': Dynamically route 100% weight to image modality (fallback to ImageBaseline).
- 'IMAGE_FAILURE' or 'MISSING_IMAGE': Dynamically route 100% weight to text modality (fallback to TextBaseline).
- 'CROSS_MODAL_CONFLICT': Route to the branch with lower predictive entropy / higher margin.
- 'HIGH_UNCERTAINTY' / 'MISSING_BOTH': Retain conservative predictions or flag for human review.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from src.evaluation.metrics import evaluate_multilabel_predictions


class DynamicModalityGating:
    """
    Adaptive fallback and modality weight gating engine conditioned on auditor diagnoses.
    """
    def __init__(
        self,
        default_alpha: float = 0.5,
        conflict_strategy: str = "lower_entropy"  # 'lower_entropy', 'prefer_text', 'average'
    ):
        self.default_alpha = default_alpha
        self.conflict_strategy = conflict_strategy

    def recover_predictions(
        self,
        text_probs: np.ndarray,
        image_probs: np.ndarray,
        diagnoses: List[str],
        text_uncertainties: Optional[np.ndarray] = None,
        image_uncertainties: Optional[np.ndarray] = None,
        default_fused_probs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Computes recovered probabilities and reports routing decisions per sample.

        Args:
            text_probs: Unimodal text branch probabilities (N, C)
            image_probs: Unimodal image branch probabilities (N, C)
            diagnoses: Diagnostic label strings produced by auditor (N,)
            text_uncertainties: Optional text entropy/uncertainty (N,)
            image_uncertainties: Optional image entropy/uncertainty (N,)
            default_fused_probs: Baseline static fused probabilities (N, C)

        Returns:
            Dictionary containing:
            - 'recovered_probs': (N, C) probability matrix after dynamic recovery
            - 'routing_actions': list of string descriptions of actions taken per sample
            - 'effective_alphas': (N,) array of final text weights applied
        """
        n_samples = len(text_probs)
        recovered_probs = np.zeros_like(text_probs)
        routing_actions = []
        effective_alphas = np.full(n_samples, self.default_alpha, dtype=float)

        for i in range(n_samples):
            diag = diagnoses[i]

            if diag in ("MISSING_TEXT", "TEXT_FAILURE"):
                # Fallback to image branch completely
                recovered_probs[i] = image_probs[i]
                routing_actions.append("ROUTED_TO_IMAGE_ONLY")
                effective_alphas[i] = 0.0

            elif diag in ("MISSING_IMAGE", "IMAGE_FAILURE"):
                # Fallback to text branch completely
                recovered_probs[i] = text_probs[i]
                routing_actions.append("ROUTED_TO_TEXT_ONLY")
                effective_alphas[i] = 1.0

            elif diag == "CROSS_MODAL_CONFLICT":
                if self.conflict_strategy == "lower_entropy" and text_uncertainties is not None and image_uncertainties is not None:
                    if text_uncertainties[i] <= image_uncertainties[i]:
                        recovered_probs[i] = text_probs[i]
                        routing_actions.append("CONFLICT_RESOLVED_FAVORING_TEXT")
                        effective_alphas[i] = 1.0
                    else:
                        recovered_probs[i] = image_probs[i]
                        routing_actions.append("CONFLICT_RESOLVED_FAVORING_IMAGE")
                        effective_alphas[i] = 0.0
                else:
                    # Default MM-IMDb text has significantly higher unimodal performance (F1 ~0.63 vs ~0.40)
                    recovered_probs[i] = text_probs[i]
                    routing_actions.append("CONFLICT_RESOLVED_FAVORING_TEXT_PRIOR")
                    effective_alphas[i] = 1.0

            elif diag in ("MISSING_BOTH", "HIGH_UNCERTAINTY"):
                # Conservative fusion: average if both available, else default fused
                if default_fused_probs is not None:
                    recovered_probs[i] = default_fused_probs[i]
                else:
                    recovered_probs[i] = 0.5 * text_probs[i] + 0.5 * image_probs[i]
                routing_actions.append("CONSERVATIVE_FUSION_HIGH_UNCERTAINTY")
                effective_alphas[i] = 0.5

            else:  # CLEAN or unclassified
                if default_fused_probs is not None:
                    recovered_probs[i] = default_fused_probs[i]
                else:
                    recovered_probs[i] = self.default_alpha * text_probs[i] + (1.0 - self.default_alpha) * image_probs[i]
                routing_actions.append("STANDARD_FUSION_RELIABLE")
                effective_alphas[i] = self.default_alpha

        return {
            "recovered_probs": recovered_probs,
            "routing_actions": routing_actions,
            "effective_alphas": effective_alphas
        }

    def evaluate_recovery_gain(
        self,
        unrecovered_probs: np.ndarray,
        recovered_probs: np.ndarray,
        y_true: np.ndarray,
        thresholds: np.ndarray,
        genre_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Quantifies performance delta before and after adaptive recovery.

        Returns:
            Dictionary comparing unrecovered vs recovered metrics (Micro-F1, Macro-F1, Gain).
        """
        unrec_metrics = evaluate_multilabel_predictions(
            y_true=y_true,
            y_pred_probs=unrecovered_probs,
            thresholds=thresholds,
            genre_names=genre_names
        )
        rec_metrics = evaluate_multilabel_predictions(
            y_true=y_true,
            y_pred_probs=recovered_probs,
            thresholds=thresholds,
            genre_names=genre_names
        )

        micro_f1_gain = rec_metrics["micro_f1"] - unrec_metrics["micro_f1"]
        macro_f1_gain = rec_metrics["macro_f1"] - unrec_metrics["macro_f1"]

        return {
            "unrecovered": unrec_metrics,
            "recovered": rec_metrics,
            "micro_f1_gain": round(float(micro_f1_gain), 4),
            "macro_f1_gain": round(float(macro_f1_gain), 4),
            "recovered_is_better": bool(micro_f1_gain >= 0.0)
        }
