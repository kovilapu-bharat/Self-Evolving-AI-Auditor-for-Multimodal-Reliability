"""
Text-Only Baseline Model for MM-IMDb Multi-Label Movie Genre Classification.
Uses TF-IDF n-gram vectorization with One-vs-Rest Logistic Regression.
Tunes decision thresholds strictly on validation data.
"""

import time
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_multilabel_predictions, optimize_thresholds_on_val

logger = logging.getLogger("TextBaseline")


class TextBaselineModel:
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        c_param: float = 1.0,
        random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.c_param = c_param
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,
            sublinear_tf=True
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=self.c_param,
                max_iter=500,
                solver="liblinear",
                random_state=self.random_state
            )
        )
        self.thresholds: Optional[np.ndarray] = None
        self.genre_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        train_texts: List[str],
        y_train: np.ndarray,
        val_texts: List[str],
        y_val: np.ndarray,
        genre_names: List[str]
    ) -> Dict[str, Any]:
        """
        Fits vectorizer and classifier on train data ONLY.
        Tunes decision thresholds on validation data ONLY.
        """
        self.genre_names = list(genre_names)
        logger.info(f"Fitting TF-IDF Vectorizer on {len(train_texts)} training texts...")
        t0 = time.time()
        X_train = self.vectorizer.fit_transform(train_texts)
        fit_vect_time = time.time() - t0
        logger.info(f"TF-IDF fitted in {fit_vect_time:.2f}s. Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        logger.info("Training One-vs-Rest Logistic Regression classifiers...")
        t1 = time.time()
        self.classifier.fit(X_train, y_train)
        fit_clf_time = time.time() - t1
        logger.info(f"Classifiers trained in {fit_clf_time:.2f}s.")
        self.is_fitted = True

        # Validation evaluation and threshold tuning
        logger.info(f"Evaluating and tuning thresholds on {len(val_texts)} validation samples...")
        X_val = self.vectorizer.transform(val_texts)
        val_probs = self.classifier.predict_proba(X_val)
        
        # Optimize thresholds on validation data
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
            "fit_time_sec": round(fit_vect_time + fit_clf_time, 2),
            "val_metrics_default": val_metrics_default,
            "val_metrics_tuned": val_metrics_tuned,
            "thresholds": self.thresholds.tolist()
        }

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str], use_tuned_thresholds: bool = True) -> np.ndarray:
        probs = self.predict_proba(texts)
        thresh = self.thresholds if use_tuned_thresholds and self.thresholds is not None else 0.5
        return (probs >= thresh).astype(int)

    def evaluate(self, test_texts: List[str], y_test: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, float]:
        """
        Evaluates on test split using validation-selected thresholds.
        Measures exact inference time.
        """
        t0 = time.time()
        test_probs = self.predict_proba(test_texts)
        inference_time = time.time() - t0
        
        metrics = evaluate_multilabel_predictions(
            y_true=y_test,
            y_pred_probs=test_probs,
            thresholds=self.thresholds,
            genre_names=self.genre_names
        )
        metrics["inference_time_sec"] = round(inference_time, 4)
        metrics["samples_per_second"] = round(len(test_texts) / max(0.001, inference_time), 2)
        return metrics, test_probs, inference_time

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "thresholds": self.thresholds,
            "genre_names": self.genre_names,
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "c_param": self.c_param,
                "random_state": self.random_state
            }
        }, filepath)
        logger.info(f"Text baseline checkpoint saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TextBaselineModel":
        data = joblib.load(filepath)
        model = cls(**data["config"])
        model.vectorizer = data["vectorizer"]
        model.classifier = data["classifier"]
        model.thresholds = data["thresholds"]
        model.genre_names = data["genre_names"]
        model.is_fitted = True
        return model
