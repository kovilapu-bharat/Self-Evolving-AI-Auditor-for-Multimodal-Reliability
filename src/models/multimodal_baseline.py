"""
Multimodal Baseline Model for MM-IMDb Multi-Label Movie Genre Classification in SEAA
(A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems).
Combines Text-Only and Image-Only predictions via Validation-Tuned Weighted Late Fusion.
Fusion weight alpha and decision thresholds are selected strictly on validation data.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.metrics import evaluate_multilabel_predictions, optimize_thresholds_on_val
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel

logger = logging.getLogger("MultimodalBaseline")


class MultimodalBaselineModel:
    def __init__(
        self,
        text_model: Optional[TextBaselineModel] = None,
        image_model: Optional[ImageBaselineModel] = None,
        alpha: float = 0.5
    ):
        self.text_model = text_model
        self.image_model = image_model
        self.alpha = alpha  # Weight for text model: P_mm = alpha * P_text + (1 - alpha) * P_image
        self.thresholds: Optional[np.ndarray] = None
        self.genre_names: List[str] = []

    def fit_fusion_parameters(
        self,
        val_text_probs: np.ndarray,
        val_image_probs: np.ndarray,
        y_val: np.ndarray,
        genre_names: List[str],
        alpha_grid: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Tunes the fusion weight alpha and decision thresholds STRICTLY on validation data.
        Test labels are NEVER used for parameter selection.
        """
        self.genre_names = list(genre_names)
        if alpha_grid is None:
            alpha_grid = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
            
        logger.info(f"Tuning fusion weight alpha across {len(alpha_grid)} candidates on validation set...")
        best_alpha = 0.5
        best_micro_f1 = -1.0
        best_val_probs = None
        
        # Grid search alpha on validation set using default threshold 0.5 first
        for a in alpha_grid:
            fused_val = a * val_text_probs + (1.0 - a) * val_image_probs
            pred_bin = (fused_val >= 0.5).astype(int)
            score = f1_score(y_val, pred_bin, average="micro", zero_division=0)
            if score > best_micro_f1:
                best_micro_f1 = score
                best_alpha = a
                best_val_probs = fused_val
                
        self.alpha = best_alpha
        logger.info(f"Optimal validation fusion weight selected: alpha={self.alpha:.2f} (Val Micro-F1={best_micro_f1:.4f})")
        
        # Now optimize per-genre thresholds on the best fused validation probabilities
        fused_val_optimal = self.alpha * val_text_probs + (1.0 - self.alpha) * val_image_probs
        self.thresholds = optimize_thresholds_on_val(y_val, fused_val_optimal)
        
        val_metrics_default = evaluate_multilabel_predictions(
            y_val, fused_val_optimal, default_threshold=0.5, genre_names=self.genre_names
        )
        val_metrics_tuned = evaluate_multilabel_predictions(
            y_val, fused_val_optimal, thresholds=self.thresholds, genre_names=self.genre_names
        )
        
        logger.info(
            f"Validation Results: Default Micro-F1={val_metrics_default['micro_f1']:.4f} -> "
            f"Tuned Micro-F1={val_metrics_tuned['micro_f1']:.4f} | "
            f"Tuned Macro-F1={val_metrics_tuned['macro_f1']:.4f} | "
            f"mAP={val_metrics_tuned['mean_average_precision']:.4f}"
        )
        
        return {
            "selected_alpha": float(self.alpha),
            "val_metrics_default": val_metrics_default,
            "val_metrics_tuned": val_metrics_tuned,
            "thresholds": self.thresholds.tolist()
        }

    def predict_proba(
        self,
        test_text_probs: np.ndarray,
        test_image_probs: np.ndarray
    ) -> np.ndarray:
        return self.alpha * test_text_probs + (1.0 - self.alpha) * test_image_probs

    def predict(
        self,
        test_text_probs: np.ndarray,
        test_image_probs: np.ndarray,
        use_tuned_thresholds: bool = True
    ) -> np.ndarray:
        probs = self.predict_proba(test_text_probs, test_image_probs)
        thresh = self.thresholds if use_tuned_thresholds and self.thresholds is not None else 0.5
        return (probs >= thresh).astype(int)

    def evaluate(
        self,
        test_text_probs: np.ndarray,
        test_image_probs: np.ndarray,
        y_test: np.ndarray,
        inference_time: float = 0.0
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Evaluates multimodal model on test split using validation-selected alpha and thresholds.
        """
        test_probs = self.predict_proba(test_text_probs, test_image_probs)
        metrics = evaluate_multilabel_predictions(
            y_true=y_test,
            y_pred_probs=test_probs,
            thresholds=self.thresholds,
            genre_names=self.genre_names
        )
        metrics["selected_alpha"] = float(self.alpha)
        metrics["inference_time_sec"] = round(inference_time, 4)
        metrics["samples_per_second"] = round(len(y_test) / max(0.001, inference_time), 2)
        return metrics, test_probs

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "alpha": float(self.alpha),
                "thresholds": self.thresholds.tolist() if self.thresholds is not None else None,
                "genre_names": self.genre_names,
                "model_type": "LateFusion_ProbabilityWeighted"
            }, f, indent=2)
        logger.info(f"Multimodal baseline configuration saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "MultimodalBaselineModel":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(alpha=data["alpha"])
        model.thresholds = np.array(data["thresholds"], dtype=np.float32) if data["thresholds"] else None
        model.genre_names = data["genre_names"]
        return model
