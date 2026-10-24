"""
Cross-Modal Disagreement Auditing for Multimodal AI Systems.
Measures probability-level divergence (multi-label Bernoulli Jensen-Shannon divergence)
and prediction-level divergence (Jaccard distance, Hamming distance) between modalities.

In multi-label classification (e.g., MM-IMDb with 23 independent binary genre outputs),
class predictions are governed by independent Bernoulli distributions rather than a single
mutually exclusive multinomial categorical distribution. Treating 23 independent sigmoids
as a categorical softmax distribution is statistically invalid.

This module implements mathematically rigorous per-genre Bernoulli Jensen-Shannon divergence:
    For each genre c with text prob p and image prob q, and midpoint m = 0.5 * (p + q):
    D_JS(B_p || B_q) = 0.5 * D_KL(B_p || B_m) + 0.5 * D_KL(B_q || B_m)
using base-2 logarithm, guaranteeing D_JS in [0, 1].
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


def _validate_probability_arrays(
    p: np.ndarray,
    q: np.ndarray,
    name_p: str = "text_probs",
    name_q: str = "image_probs"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates that p and q are finite numpy arrays with identical shapes and values in [0, 1].
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"Shape mismatch between {name_p} {p_arr.shape} and {name_q} {q_arr.shape}."
        )

    if not np.all(np.isfinite(p_arr)):
        raise ValueError(f"{name_p} contains non-finite values (NaN or Inf).")

    if not np.all(np.isfinite(q_arr)):
        raise ValueError(f"{name_q} contains non-finite values (NaN or Inf).")

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(f"{name_p} contains values outside the valid probability range [0, 1].")

    if np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
        raise ValueError(f"{name_q} contains values outside the valid probability range [0, 1].")

    return p_arr, q_arr


def _validate_binary_arrays(
    y1: np.ndarray,
    y2: np.ndarray,
    name_y1: str = "y_pred1",
    name_y2: str = "y_pred2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates that y1 and y2 are binary (0 or 1) arrays with matching shapes.
    """
    arr1 = np.asarray(y1)
    arr2 = np.asarray(y2)

    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch between {name_y1} {arr1.shape} and {name_y2} {arr2.shape}."
        )

    if not np.all(np.isin(arr1, [0, 1])):
        raise ValueError(f"{name_y1} contains non-binary values; expected only 0 and 1.")

    if not np.all(np.isin(arr2, [0, 1])):
        raise ValueError(f"{name_y2} contains non-binary values; expected only 0 and 1.")

    return arr1.astype(int), arr2.astype(int)


def bernoulli_jsd(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes elementwise Jensen-Shannon divergence between independent Bernoulli distributions.

    For each element with probabilities p and q in [0, 1], defines Bernoulli distributions
    B_p = (p, 1-p) and B_q = (q, 1-q).
    With midpoint distribution B_m = (m, 1-m) where m = 0.5 * (p + q):
        D_JS(B_p || B_q) = 0.5 * D_KL(B_p || B_m) + 0.5 * D_KL(B_q || B_m)
    Using base-2 logarithm, D_JS is strictly bounded in [0.0, 1.0].

    Args:
        p: Array of probabilities, shape (N, C) or (C,)
        q: Array of probabilities, matching shape of p
        eps: Small numerical safeguard to clip probabilities away from 0 and 1

    Returns:
        Array of elementwise Bernoulli JSD values, same shape as p.
    """
    p_arr, q_arr = _validate_probability_arrays(p, q, "p", "q")

    # Safeguard against 0 and 1 for log computation
    p_safe = np.clip(p_arr, eps, 1.0 - eps)
    q_safe = np.clip(q_arr, eps, 1.0 - eps)
    m_safe = 0.5 * (p_safe + q_safe)

    # Elementwise KL divergences to midpoint using base 2 logarithm
    kl_p_m = p_safe * np.log2(p_safe / m_safe) + (1.0 - p_safe) * np.log2((1.0 - p_safe) / (1.0 - m_safe))
    kl_q_m = q_safe * np.log2(q_safe / m_safe) + (1.0 - q_safe) * np.log2((1.0 - q_safe) / (1.0 - m_safe))

    jsd = 0.5 * (kl_p_m + kl_q_m)
    # Numerical clip to ensure exact theoretical bounds [0, 1]
    return np.clip(jsd, 0.0, 1.0)


def mean_bernoulli_jsd(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes sample-level mean Bernoulli JSD across genres.

    Args:
        p: Probability matrix of shape (N, C)
        q: Probability matrix of shape (N, C)
        eps: Numerical safeguard

    Returns:
        Array of shape (N,) containing mean Bernoulli JSD across all C classes.
    """
    per_genre = bernoulli_jsd(p, q, eps=eps)
    if per_genre.ndim == 1:
        return np.mean(per_genre)
    return np.mean(per_genre, axis=-1)


def jaccard_distance(
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> np.ndarray:
    """
    Computes sample-wise Jaccard distance between two sets of binary multi-label predictions.

    Jaccard Distance = 1.0 - (|S1 ∩ S2| / |S1 ∪ S2|)
    Edge Case:
        When both prediction sets are empty (|S1 ∪ S2| == 0), both models completely agree
        that no positive genre is present. In this case, Jaccard distance is defined as 0.0.

    Args:
        y_pred1: Binary array of shape (N, C) or (C,)
        y_pred2: Binary array of shape (N, C) or (C,)

    Returns:
        Array of shape (N,) (or float if 1D) with Jaccard distances in [0.0, 1.0].
    """
    arr1, arr2 = _validate_binary_arrays(y_pred1, y_pred2, "y_pred1", "y_pred2")
    is_1d = (arr1.ndim == 1)
    if is_1d:
        arr1 = np.expand_dims(arr1, axis=0)
        arr2 = np.expand_dims(arr2, axis=0)

    intersection = np.sum((arr1 == 1) & (arr2 == 1), axis=-1)
    union = np.sum((arr1 == 1) | (arr2 == 1), axis=-1)

    # Where union == 0, both sets are empty -> distance is 0.0 (full agreement)
    distances = np.zeros(len(arr1), dtype=np.float64)
    non_empty_mask = union > 0
    distances[non_empty_mask] = 1.0 - (intersection[non_empty_mask] / union[non_empty_mask])

    return float(distances[0]) if is_1d else distances


def hamming_distance(
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> np.ndarray:
    """
    Computes sample-wise normalized Hamming distance (fraction of disagreeing labels).

    Hamming Distance = (1 / C) * sum_{c=1}^C |y1_c - y2_c|

    Args:
        y_pred1: Binary array of shape (N, C) or (C,)
        y_pred2: Binary array of shape (N, C) or (C,)

    Returns:
        Array of shape (N,) (or float if 1D) with values in [0.0, 1.0].
    """
    arr1, arr2 = _validate_binary_arrays(y_pred1, y_pred2, "y_pred1", "y_pred2")
    is_1d = (arr1.ndim == 1)
    if is_1d:
        arr1 = np.expand_dims(arr1, axis=0)
        arr2 = np.expand_dims(arr2, axis=0)

    n_classes = arr1.shape[-1]
    if n_classes == 0:
        return 0.0 if is_1d else np.zeros(len(arr1), dtype=np.float64)

    disagreements = np.sum(arr1 != arr2, axis=-1)
    distances = disagreements / float(n_classes)

    return float(distances[0]) if is_1d else distances


def per_genre_disagreement(
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    genre_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyzes per-genre disagreement patterns between two binary prediction sets.

    Args:
        y_pred1: Binary array of shape (N, C)
        y_pred2: Binary array of shape (N, C)
        genre_names: Optional list of C genre names

    Returns:
        Dictionary with:
            - disagree_mask: boolean array (N, C)
            - disagree_counts: array (C,) of disagreement counts
            - disagree_rates: array (C,) of disagreement proportions
            - per_genre_stats: dict mapping genre name to count and rate
    """
    arr1, arr2 = _validate_binary_arrays(y_pred1, y_pred2)
    if arr1.ndim == 1:
        arr1 = np.expand_dims(arr1, axis=0)
        arr2 = np.expand_dims(arr2, axis=0)

    n_samples, n_classes = arr1.shape
    disagree_mask = (arr1 != arr2)
    disagree_counts = np.sum(disagree_mask, axis=0)
    disagree_rates = disagree_counts / max(1, n_samples)

    per_genre_stats = {}
    for c in range(n_classes):
        name = genre_names[c] if genre_names and c < len(genre_names) else f"genre_{c}"
        per_genre_stats[name] = {
            "disagreement_count": int(disagree_counts[c]),
            "disagreement_rate": round(float(disagree_rates[c]), 4)
        }

    return {
        "disagree_mask": disagree_mask,
        "disagree_counts": disagree_counts,
        "disagree_rates": disagree_rates,
        "per_genre_stats": per_genre_stats
    }


def compute_cross_modal_disagreement(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    text_thresholds: Optional[np.ndarray] = None,
    image_thresholds: Optional[np.ndarray] = None,
    genre_names: Optional[List[str]] = None,
    eps: float = 1e-7
) -> Dict[str, Any]:
    """
    Comprehensive cross-modal disagreement auditor.

    Computes:
    - Multi-label Bernoulli Jensen-Shannon divergence (mean and per-genre)
    - Mean absolute probability difference (L1 divergence)
    - Set prediction disagreement (Jaccard and Hamming distances)
    - Per-genre disagreement mask and counts

    Args:
        text_probs: Text model predicted probabilities (N, C)
        image_probs: Image model predicted probabilities (N, C)
        text_thresholds: Optional per-genre thresholds for text model (shape C or float, default 0.5)
        image_thresholds: Optional per-genre thresholds for image model (shape C or float, default 0.5)
        genre_names: Optional list of C genre names
        eps: Numerical stability constant

    Returns:
        Dictionary of sample-level arrays and summary statistics.
    """
    t_probs, i_probs = _validate_probability_arrays(text_probs, image_probs, "text_probs", "image_probs")
    if t_probs.ndim == 1:
        t_probs = np.expand_dims(t_probs, axis=0)
        i_probs = np.expand_dims(i_probs, axis=0)

    n_samples, n_classes = t_probs.shape

    # Set default thresholds
    if text_thresholds is None:
        t_thresh = np.full(n_classes, 0.5, dtype=np.float64)
    elif isinstance(text_thresholds, (float, int)):
        t_thresh = np.full(n_classes, float(text_thresholds), dtype=np.float64)
    else:
        t_thresh = np.asarray(text_thresholds, dtype=np.float64)

    if image_thresholds is None:
        i_thresh = np.full(n_classes, 0.5, dtype=np.float64)
    elif isinstance(image_thresholds, (float, int)):
        i_thresh = np.full(n_classes, float(image_thresholds), dtype=np.float64)
    else:
        i_thresh = np.asarray(image_thresholds, dtype=np.float64)

    # 1. Probability divergence
    per_genre_jsd = bernoulli_jsd(t_probs, i_probs, eps=eps)
    mean_jsd = np.mean(per_genre_jsd, axis=-1)

    prob_l1 = np.abs(t_probs - i_probs)
    mean_l1 = np.mean(prob_l1, axis=-1)

    # 2. Thresholded binary predictions
    text_pred_bin = (t_probs >= t_thresh).astype(int)
    image_pred_bin = (i_probs >= i_thresh).astype(int)

    # 3. Set prediction divergence
    jaccard_dist = jaccard_distance(text_pred_bin, image_pred_bin)
    hamming_dist = hamming_distance(text_pred_bin, image_pred_bin)

    # 4. Per-genre analysis
    genre_analysis = per_genre_disagreement(text_pred_bin, image_pred_bin, genre_names=genre_names)
    num_disagreeing_genres = np.sum(genre_analysis["disagree_mask"], axis=-1)

    summary = {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "mean_bernoulli_jsd": round(float(np.mean(mean_jsd)), 4),
        "median_bernoulli_jsd": round(float(np.median(mean_jsd)), 4),
        "std_bernoulli_jsd": round(float(np.std(mean_jsd)), 4),
        "mean_jaccard_distance": round(float(np.mean(jaccard_dist)), 4),
        "median_jaccard_distance": round(float(np.median(jaccard_dist)), 4),
        "mean_hamming_distance": round(float(np.mean(hamming_dist)), 4),
        "mean_prob_l1": round(float(np.mean(mean_l1)), 4),
        "mean_disagreeing_genres_per_sample": round(float(np.mean(num_disagreeing_genres)), 2)
    }

    return {
        "mean_bernoulli_jsd": mean_jsd,
        "per_genre_bernoulli_jsd": per_genre_jsd,
        "jaccard_distance": jaccard_dist,
        "hamming_distance": hamming_dist,
        "num_disagreeing_genres": num_disagreeing_genres,
        "disagreeing_genre_mask": genre_analysis["disagree_mask"],
        "prob_l1": prob_l1,
        "mean_prob_l1": mean_l1,
        "text_binary_predictions": text_pred_bin,
        "image_binary_predictions": image_pred_bin,
        "per_genre_stats": genre_analysis["per_genre_stats"],
        "summary": summary
    }
