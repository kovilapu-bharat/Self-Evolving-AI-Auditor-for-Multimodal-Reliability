"""
Predictive Uncertainty Estimation for Multimodal Multi-Label Classification.
Implements multi-label binary entropy, distance to decision thresholds,
and threshold proximity diagnostics for text, vision, and fused modalities.

Scientific Notes:
1. Multi-label independence: MM-IMDb features 23 non-mutually exclusive genres.
   Predictive uncertainty is therefore modeled via per-genre Bernoulli entropy
   rather than categorical softmax entropy:
       H(p) = - [p * log2(p) + (1-p) * log2(1-p)] in [0, 1]
   Maximum uncertainty occurs at p = 0.5 (H = 1.0); minimum at p in {0, 1} (H = 0.0).
2. Decision Threshold Proximity: Because decision thresholds are optimized on validation
   data (tau_c != 0.5 for many genres), uncertainty relative to the decision boundary
   is defined by the margin |p_c - tau_c|. Predictions close to tau_c (within margin delta)
   carry high classification risk.
3. Calibration limitation: These uncertainty metrics serve as heuristic confidence
   and ambiguity indicators. They are not Bayesian posterior variances or calibrated
   probabilities of error.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


def _validate_probs_and_thresholds(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None,
    name_probs: str = "probs"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates probability matrix and converts thresholds into a matching 1D float array.
    """
    p_arr = np.asarray(probs, dtype=np.float64)

    if not np.all(np.isfinite(p_arr)):
        raise ValueError(f"{name_probs} contains non-finite values (NaN or Inf).")

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(f"{name_probs} contains values outside [0, 1].")

    n_classes = p_arr.shape[-1]

    if thresholds is None:
        t_arr = np.full(n_classes, 0.5, dtype=np.float64)
    elif isinstance(thresholds, (float, int)):
        if thresholds < 0.0 or thresholds > 1.0:
            raise ValueError(f"Threshold scalar {thresholds} is outside [0, 1].")
        t_arr = np.full(n_classes, float(thresholds), dtype=np.float64)
    else:
        t_arr = np.asarray(thresholds, dtype=np.float64)
        if t_arr.shape != (n_classes,):
            raise ValueError(
                f"Thresholds shape {t_arr.shape} does not match number of classes {n_classes}."
            )
        if not np.all(np.isfinite(t_arr)):
            raise ValueError("Thresholds contain non-finite values (NaN or Inf).")
        if np.any(t_arr < 0.0) or np.any(t_arr > 1.0):
            raise ValueError("Thresholds contain values outside [0, 1].")

    return p_arr, t_arr


def binary_predictive_entropy(
    probs: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes elementwise binary predictive entropy using base-2 logarithm.

    H(p) = - [p * log2(p) + (1-p) * log2(1-p)]

    Args:
        probs: Probability array of shape (N, C) or (C,) with values in [0, 1]
        eps: Small numerical guard to prevent log(0)

    Returns:
        Array of same shape as probs with binary entropy values in [0.0, 1.0].
    """
    p_arr, _ = _validate_probs_and_thresholds(probs)
    p_safe = np.clip(p_arr, eps, 1.0 - eps)

    entropy = - (p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
    return np.clip(entropy, 0.0, 1.0)


def mean_binary_entropy(
    probs: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes mean binary entropy across classes for each sample.

    Args:
        probs: Array of shape (N, C) or (C,)
        eps: Numerical guard

    Returns:
        Array of shape (N,) (or float if 1D) with mean entropy in [0.0, 1.0].
    """
    entropy = binary_predictive_entropy(probs, eps=eps)
    if entropy.ndim == 1:
        return float(np.mean(entropy))
    return np.mean(entropy, axis=-1)


def threshold_margins(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None
) -> np.ndarray:
    """
    Computes absolute distance to decision thresholds for each genre: |p_c - tau_c|.

    Args:
        probs: Probability array of shape (N, C) or (C,)
        thresholds: Array of shape (C,) or scalar in [0, 1]

    Returns:
        Array of margins of same shape as probs.
    """
    p_arr, t_arr = _validate_probs_and_thresholds(probs, thresholds)
    return np.abs(p_arr - t_arr)


def mean_threshold_margin(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None
) -> np.ndarray:
    """
    Computes mean threshold margin across all classes per sample.
    Larger margin signifies higher decision confidence (further from boundary).
    """
    margins = threshold_margins(probs, thresholds)
    if margins.ndim == 1:
        return float(np.mean(margins))
    return np.mean(margins, axis=-1)


def min_threshold_margin(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None
) -> np.ndarray:
    """
    Computes the minimum margin to any threshold per sample: min_c |p_c - tau_c|.
    Identifies the most vulnerable/ambiguous genre prediction for each sample.
    """
    margins = threshold_margins(probs, thresholds)
    if margins.ndim == 1:
        return float(np.min(margins))
    return np.min(margins, axis=-1)


def threshold_proximity_stats(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None,
    delta: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the count and proportion of classes falling within delta of their thresholds.

    Args:
        probs: Probability array of shape (N, C)
        thresholds: Thresholds of shape (C,) or scalar
        delta: Margin boundary tolerance (default 0.05)

    Returns:
        Tuple of (near_counts, near_proportions) of shape (N,)
    """
    if delta < 0.0:
        raise ValueError(f"Delta must be non-negative, got {delta}.")

    margins = threshold_margins(probs, thresholds)
    is_1d = (margins.ndim == 1)
    if is_1d:
        margins = np.expand_dims(margins, axis=0)

    n_classes = margins.shape[-1]
    near_mask = (margins <= delta)
    near_counts = np.sum(near_mask, axis=-1)
    near_proportions = near_counts / float(n_classes)

    if is_1d:
        return int(near_counts[0]), float(near_proportions[0])
    return near_counts, near_proportions


def compute_predictive_uncertainty(
    probs: np.ndarray,
    thresholds: Optional[Union[np.ndarray, float]] = None,
    delta: float = 0.05,
    genre_names: Optional[List[str]] = None,
    modality_name: str = "modality",
    eps: float = 1e-7
) -> Dict[str, Any]:
    """
    Comprehensive predictive uncertainty evaluation for a given modality.

    Calculates:
    - Per-genre and mean binary entropy
    - Per-genre, mean, and min margin to decision thresholds
    - Count and proportion of predictions within delta of decision thresholds
    - Heuristic composite uncertainty score in [0, 1]

    Args:
        probs: Array of shape (N, C) or (C,)
        thresholds: Array of shape (C,) or scalar (defaults to 0.5)
        delta: Threshold proximity boundary margin
        genre_names: Optional list of C genre names
        modality_name: Name of the modality for reporting (e.g. 'text', 'image', 'multimodal')
        eps: Numerical stability constant

    Returns:
        Dictionary containing sample-level uncertainty arrays and dataset summary.
    """
    p_arr, t_arr = _validate_probs_and_thresholds(probs, thresholds, name_probs=f"{modality_name}_probs")
    is_1d = (p_arr.ndim == 1)
    if is_1d:
        p_arr = np.expand_dims(p_arr, axis=0)

    n_samples, n_classes = p_arr.shape

    # 1. Entropy
    per_genre_ent = binary_predictive_entropy(p_arr, eps=eps)
    mean_ent = np.mean(per_genre_ent, axis=-1)

    # 2. Threshold Margins
    margins = np.abs(p_arr - t_arr)
    mean_margin = np.mean(margins, axis=-1)
    min_margin = np.min(margins, axis=-1)

    # 3. Proximity
    near_counts, near_props = threshold_proximity_stats(p_arr, t_arr, delta=delta)

    # 4. Composite Uncertainty Score (heuristic combining entropy and boundary proximity)
    # Scaled to [0, 1]: 0.5 * mean_entropy + 0.5 * (1.0 - 2 * min(0.5, mean_margin))
    composite_score = 0.5 * mean_ent + 0.5 * (1.0 - 2.0 * np.minimum(0.5, mean_margin))
    composite_score = np.clip(composite_score, 0.0, 1.0)

    summary = {
        "modality": modality_name,
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "delta": float(delta),
        "mean_binary_entropy": round(float(np.mean(mean_ent)), 4),
        "std_binary_entropy": round(float(np.std(mean_ent)), 4),
        "mean_threshold_margin": round(float(np.mean(mean_margin)), 4),
        "mean_min_threshold_margin": round(float(np.mean(min_margin)), 4),
        "mean_near_threshold_proportion": round(float(np.mean(near_props)), 4),
        "mean_composite_uncertainty": round(float(np.mean(composite_score)), 4)
    }

    return {
        "modality": modality_name,
        "mean_binary_entropy": mean_ent,
        "per_genre_entropy": per_genre_ent,
        "margins": margins,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "near_threshold_count": near_counts,
        "near_threshold_proportion": near_props,
        "composite_uncertainty": composite_score,
        "thresholds_used": t_arr,
        "summary": summary
    }
