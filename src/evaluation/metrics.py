"""
Evaluation Metrics for Multi-Label Classification on MM-IMDb.
Calculates Micro-F1, Macro-F1, Per-genre Precision/Recall/F1,
Mean Average Precision (mAP), Hamming Loss, and optimal threshold selection.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    hamming_loss
)

def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    genre_names: Optional[List[str]] = None,
    default_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates multi-label classification predictions.
    
    Args:
        y_true: Binary ground-truth matrix of shape (N, C)
        y_pred_probs: Predicted probability matrix of shape (N, C)
        thresholds: Array of shape (C,) or single float threshold.
        genre_names: List of genre names of length C.
        default_threshold: Fallback threshold if thresholds is None.
        
    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    n_samples, n_classes = y_true.shape
    if thresholds is None:
        thresholds = np.full(n_classes, default_threshold, dtype=np.float32)
    elif isinstance(thresholds, (float, int)):
        thresholds = np.full(n_classes, float(thresholds), dtype=np.float32)
        
    # Apply thresholds to get binary predictions
    y_pred_binary = (y_pred_probs >= thresholds).astype(int)
    
    # Calculate overall metrics
    micro_f1 = float(f1_score(y_true, y_pred_binary, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred_binary, average="micro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred_binary, average="macro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred_binary, average="micro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred_binary, average="macro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred_binary))
    
    # Per-genre average precision and mAP
    per_genre_ap = []
    for c in range(n_classes):
        if np.sum(y_true[:, c]) > 0:
            ap = average_precision_score(y_true[:, c], y_pred_probs[:, c])
        else:
            ap = 0.0
        per_genre_ap.append(float(ap))
    mean_ap = float(np.mean(per_genre_ap))
    
    # Per-genre precision, recall, F1
    per_genre_prec = precision_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_genre_rec = recall_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_genre_f1 = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    
    per_genre_metrics = {}
    for c in range(n_classes):
        name = genre_names[c] if genre_names and c < len(genre_names) else f"class_{c}"
        per_genre_metrics[name] = {
            "precision": round(float(per_genre_prec[c]), 4),
            "recall": round(float(per_genre_rec[c]), 4),
            "f1": round(float(per_genre_f1[c]), 4),
            "average_precision": round(float(per_genre_ap[c]), 4),
            "support": int(np.sum(y_true[:, c])),
            "threshold": round(float(thresholds[c]), 3)
        }
        
    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "macro_precision": round(macro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "macro_recall": round(macro_recall, 4),
        "mean_average_precision": round(mean_ap, 4),
        "hamming_loss": round(h_loss, 4),
        "per_genre": per_genre_metrics
    }


def optimize_thresholds_on_val(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    metric: str = "f1",
    step: float = 0.02
) -> np.ndarray:
    """
    Finds optimal per-genre decision thresholds using validation data only.
    Optimizes each class's threshold to maximize its F1 score on validation set.
    """
    n_samples, n_classes = y_val_true.shape
    optimal_thresholds = np.zeros(n_classes, dtype=np.float32)
    candidate_thresholds = np.arange(0.1, 0.9, step)
    
    for c in range(n_classes):
        best_t = 0.5
        best_score = -1.0
        y_true_c = y_val_true[:, c]
        probs_c = y_val_probs[:, c]
        
        # If class has zero positives in val (should not happen with stratification)
        if np.sum(y_true_c) == 0:
            optimal_thresholds[c] = 0.5
            continue
            
        for t in candidate_thresholds:
            pred_binary = (probs_c >= t).astype(int)
            score = f1_score(y_true_c, pred_binary, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t
                
        optimal_thresholds[c] = best_t
        
    return optimal_thresholds
