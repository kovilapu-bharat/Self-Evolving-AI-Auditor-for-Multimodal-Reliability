"""
Multi-Label Calibration and Reliability Metrics for SEAA.

Provides mathematical formulations and evaluation functions for:
- Multi-Label Expected Calibration Error (Micro-ECE and Macro-ECE)
- Maximum Calibration Error (MCE)
- Multi-Label Brier Score
- Reliability Diagram bin statistics for plotting calibration curves
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def compute_binary_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    for a 1D vector of binary labels and predicted probabilities.

    Args:
        y_true: 1D binary ground truth array (shape: N,)
        y_prob: 1D predicted probabilities array (shape: N,), values in [0, 1]
        n_bins: Number of equal-width probability bins (default: 10)

    Returns:
        Tuple of (ece, mce, bin_stats)
        where bin_stats contains:
            'bin_edges': Array of bin edges of length n_bins + 1
            'bin_acc': Array of bin accuracies of length n_bins
            'bin_conf': Array of bin confidences of length n_bins
            'bin_count': Array of sample counts per bin of length n_bins
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, 0.0, 1.0)
    
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0, 0.0, {
            "bin_edges": np.linspace(0.0, 1.0, n_bins + 1),
            "bin_acc": np.zeros(n_bins),
            "bin_conf": np.zeros(n_bins),
            "bin_count": np.zeros(n_bins, dtype=int),
        }

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=int)

    # Bin assignment: note that bin index for prob == 1.0 maps to last bin
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # 0 to n_bins - 1

    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = int(np.sum(mask))
        bin_count[b] = count
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            bin_acc[b] = acc
            bin_conf[b] = conf
            err = abs(acc - conf)
            ece += (count / n_samples) * err
            if err > mce:
                mce = err
        else:
            # For empty bins, set center of bin as nominal confidence
            bin_conf[b] = (bin_edges[b] + bin_edges[b + 1]) / 2.0
            bin_acc[b] = 0.0

    bin_stats = {
        "bin_edges": bin_edges,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
    }
    return float(ece), float(mce), bin_stats


def compute_multilabel_brier_score(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray
) -> float:
    """
    Computes the multi-label Brier Score (Mean Squared Error over all binary label predictions).

    Args:
        y_true: Binary ground-truth matrix of shape (N, C)
        y_pred_probs: Predicted probability matrix of shape (N, C)

    Returns:
        Scalar Brier score in [0, 1] (lower is better calibrated).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_probs = np.clip(np.asarray(y_pred_probs, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((y_pred_probs - y_true) ** 2))


def evaluate_multilabel_calibration(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
    genre_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive multi-label probability calibration evaluation.

    Computes:
    - Micro-ECE: Globally pooled Expected Calibration Error across all N * C predictions.
    - Macro-ECE: Unweighted mean of class-specific ECE across all C genres.
    - Micro-MCE: Globally pooled Maximum Calibration Error.
    - Macro-MCE: Unweighted mean of class-specific MCE across all C genres.
    - Brier Score: Global mean squared error of predicted probabilities.
    - Per-class calibration dictionaries.
    - Global reliability curve histogram data.

    Args:
        y_true: Binary ground-truth matrix (N, C)
        y_pred_probs: Predicted probability matrix (N, C)
        n_bins: Number of calibration bins (default: 10)
        genre_names: Optional list of class names of length C

    Returns:
        Dictionary of calibration metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float64)
    
    n_samples, n_classes = y_true.shape
    if genre_names is None:
        genre_names = [f"genre_{i}" for i in range(n_classes)]

    # 1. Globally pooled (Micro) calibration
    flat_true = y_true.ravel()
    flat_probs = y_pred_probs.ravel()
    micro_ece, micro_mce, global_stats = compute_binary_ece(flat_true, flat_probs, n_bins=n_bins)

    # 2. Per-class and Macro calibration
    per_class_ece = {}
    per_class_mce = {}
    per_class_brier = {}
    ece_list = []
    mce_list = []

    for c in range(n_classes):
        name = genre_names[c]
        c_true = y_true[:, c]
        c_prob = y_pred_probs[:, c]
        c_ece, c_mce, _ = compute_binary_ece(c_true, c_prob, n_bins=n_bins)
        c_brier = float(np.mean((c_prob - c_true) ** 2))

        per_class_ece[name] = float(c_ece)
        per_class_mce[name] = float(c_mce)
        per_class_brier[name] = c_brier
        ece_list.append(c_ece)
        mce_list.append(c_mce)

    macro_ece = float(np.mean(ece_list)) if ece_list else 0.0
    macro_mce = float(np.mean(mce_list)) if mce_list else 0.0
    global_brier = compute_multilabel_brier_score(y_true, y_pred_probs)

    return {
        "micro_ece": float(micro_ece),
        "macro_ece": float(macro_ece),
        "micro_mce": float(micro_mce),
        "macro_mce": float(macro_mce),
        "brier_score": float(global_brier),
        "n_bins": n_bins,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "per_class_ece": per_class_ece,
        "per_class_mce": per_class_mce,
        "per_class_brier": per_class_brier,
        "reliability_curve": {
            "bin_edges": global_stats["bin_edges"].tolist(),
            "bin_acc": global_stats["bin_acc"].tolist(),
            "bin_conf": global_stats["bin_conf"].tolist(),
            "bin_count": global_stats["bin_count"].tolist(),
        }
    }

