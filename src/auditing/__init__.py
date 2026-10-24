"""
Auditing module for SEAA.
Provides reliability monitoring, telemetry, and audit log handlers.
"""

from src.auditing.disagreement import compute_cross_modal_disagreement
from src.auditing.uncertainty import compute_predictive_uncertainty
from src.auditing.diagnosis import (
    DIAGNOSTIC_LABELS,
    DiagnosticConfig,
    calibrate_diagnostic_thresholds_on_val,
    MultimodalReliabilityAuditor,
    evaluate_diagnosis_performance
)
from src.auditing.learned_diagnostics import (
    LearnedDiagnosticClassifier,
    extract_diagnostic_feature_matrix
)

__all__ = [
    "compute_cross_modal_disagreement",
    "compute_predictive_uncertainty",
    "DIAGNOSTIC_LABELS",
    "DiagnosticConfig",
    "calibrate_diagnostic_thresholds_on_val",
    "MultimodalReliabilityAuditor",
    "evaluate_diagnosis_performance",
    "LearnedDiagnosticClassifier",
    "extract_diagnostic_feature_matrix"
]
