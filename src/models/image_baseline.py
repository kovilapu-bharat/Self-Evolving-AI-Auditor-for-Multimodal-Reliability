"""
Image-Only Baseline Model for MM-IMDb Multi-Label Movie Genre Classification.
Trains One-vs-Rest Logistic Regression on frozen ResNet50 2048-dimensional visual embeddings.
Tunes decision thresholds strictly on validation data.
"""

import time
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_multilabel_predictions, optimize_thresholds_on_val

logger = logging.getLogger("ImageBaseline")


class ImageBaselineModel:
    def __init__(
        self,
        c_param: float = 0.5,
        scale_features: bool = True,
        random_state: int = 42
    ):
        self.c_param = c_param
        self.scale_features = scale_features
        self.random_state = random_state
        
        self.scaler = StandardScaler() if scale_features else None
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=self.c_param,
                max_iter=500,
                solver="lbfgs",
                random_state=self.random_state
            )
        )
        self.thresholds: Optional[np.ndarray] = None
        self.genre_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        genre_names: List[str]
    ) -> Dict[str, Any]:
        """
        Fits image classifier on training embeddings ONLY.
        Tunes decision thresholds on validation embeddings ONLY.
        """
        self.genre_names = list(genre_names)
        logger.info(f"Fitting image classifier on {X_train.shape[0]} training samples (dim={X_train.shape[1]})...")
        t0 = time.time()
        
        if self.scaler is not None:
            X_train_proc = self.scaler.fit_transform(X_train)
            X_val_proc = self.scaler.transform(X_val)
        else:
            X_train_proc = X_train
            X_val_proc = X_val
            
        self.classifier.fit(X_train_proc, y_train)
        fit_time = time.time() - t0
        logger.info(f"Image classifier trained in {fit_time:.2f}s.")
        self.is_fitted = True

        # Validation evaluation and threshold tuning
        logger.info(f"Evaluating and tuning thresholds on {X_val.shape[0]} validation samples...")
        val_probs = self.classifier.predict_proba(X_val_proc)
        
        self.thresholds = optimize_thresholds_on_val(y_val, val_probs)
        val_metrics_default = evaluate_multilabel_predictions(
            y_val, val_probs, default_threshold=0.5, genre_names=self.genre_names
        )
        val_metrics_tuned = evaluate_multilabel_predictions(
            y_val, val_probs, thresholds=self.thresholds, genre_names=self.genre_names
        )
        
        logger.info(
            f"Validation Results: Default Micro-F1={val_metrics_default['micro_f1']:.4f} -> "
            f"Tuned Micro-F1={val_metrics_tuned['micro_f1']:.4f} | "
            f"Tuned Macro-F1={val_metrics_tuned['macro_f1']:.4f} | "
            f"mAP={val_metrics_tuned['mean_average_precision']:.4f}"
        )
        
        return {
            "fit_time_sec": round(fit_time, 2),
            "val_metrics_default": val_metrics_default,
            "val_metrics_tuned": val_metrics_tuned,
            "thresholds": self.thresholds.tolist()
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if self.scaler is not None:
            X_proc = self.scaler.transform(X)
        else:
            X_proc = X
        return self.classifier.predict_proba(X_proc)

    def predict(self, X: np.ndarray, use_tuned_thresholds: bool = True) -> np.ndarray:
        probs = self.predict_proba(X)
        thresh = self.thresholds if use_tuned_thresholds and self.thresholds is not None else 0.5
        return (probs >= thresh).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, float]:
        """
        Evaluates on test split using validation-selected thresholds.
        Measures exact inference time.
        """
        t0 = time.time()
        test_probs = self.predict_proba(X_test)
        inference_time = time.time() - t0
        
        metrics = evaluate_multilabel_predictions(
            y_true=y_test,
            y_pred_probs=test_probs,
            thresholds=self.thresholds,
            genre_names=self.genre_names
        )
        metrics["inference_time_sec"] = round(inference_time, 4)
        metrics["samples_per_second"] = round(len(X_test) / max(0.001, inference_time), 2)
        return metrics, test_probs, inference_time

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "classifier": self.classifier,
            "thresholds": self.thresholds,
            "genre_names": self.genre_names,
            "config": {
                "c_param": self.c_param,
                "scale_features": self.scale_features,
                "random_state": self.random_state
            }
        }, filepath)
        logger.info(f"Image baseline checkpoint saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "ImageBaselineModel":
        data = joblib.load(filepath)
        model = cls(**data["config"])
        model.scaler = data["scaler"]
        model.classifier = data["classifier"]
        model.thresholds = data["thresholds"]
        model.genre_names = data["genre_names"]
        model.is_fitted = True
        return model
