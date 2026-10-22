"""
Evaluation utilities and metrics calculation for SEAA
(A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems).
"""

from src.evaluation.metrics import (
    evaluate_multilabel_predictions,
    optimize_thresholds_on_val
)

__all__ = [
    "evaluate_multilabel_predictions",
    "optimize_thresholds_on_val"
]
