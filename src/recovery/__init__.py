"""
Recovery module for SEAA.
Provides fallback mechanisms, degraded modality handlers, and recovery mitigation routines.
"""

from src.recovery.modality_gating import DynamicModalityGating
from src.recovery.confidence_recovery import ConfidenceRecoveryRouter
from src.policy.adaptive_policy import AdaptiveRecoveryPolicy

__all__ = [
    "DynamicModalityGating",
    "ConfidenceRecoveryRouter",
    "AdaptiveRecoveryPolicy"
]


