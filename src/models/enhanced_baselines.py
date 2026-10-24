"""
Enhanced Baseline Models and Dynamic Fusion for SEAA (Phase 5 Enhancements).
Improvement #7: Implements enhanced unimodal and multimodal baselines:
1. EnhancedTextBaseline: Sublinear TF-IDF with bigrams, balanced regularization,
   and feature norm extraction.
2. EnhancedImageBaseline: Deep multi-layer classification head with dropout and normalization.
3. EnhancedGatedMultimodalFusion: Dynamic instance-level gating that weights modalities
   according to feature energy, uncertainty, and reliability signals.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.evaluation.metrics import evaluate_multilabel_predictions, optimize_thresholds_on_val

logger = logging.getLogger("EnhancedBaselines")


class EnhancedTextBaseline:
    """
    Enhanced TF-IDF + Logistic Regression text baseline.
    Features:
    - Sublinear TF scaling (1 + log(tf))
    - Unigram + Bigram feature extraction (ngram_range=(1, 2))
    - Balanced class weighting for long-tail genres
    - L2 norm extraction for reliability auditing
    """
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 1.0,
        min_df: int = 1
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
            min_df=min_df
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs")
        )
        self.thresholds: Optional[np.ndarray] = None
        self.genre_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
        y: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_y: Optional[np.ndarray] = None,
        genre_names: Optional[List[str]] = None
    ) -> "EnhancedTextBaseline":
        self.genre_names = list(genre_names) if genre_names else [f"genre_{i}" for i in range(y.shape[1])]
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        self.is_fitted = True

        if val_texts is not None and val_y is not None:
            val_probs = self.predict_proba(val_texts)
            self.thresholds = optimize_thresholds_on_val(val_probs, val_y)
        else:
            self.thresholds = np.full(len(self.genre_names), 0.5)

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str], thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        probs = self.predict_proba(texts)
        thresh = thresholds if thresholds is not None else (self.thresholds if self.thresholds is not None else 0.5)
        return (probs >= thresh).astype(int)

    def get_feature_norms(self, texts: List[str]) -> np.ndarray:
        """Computes L2 norm of TF-IDF feature vectors for each sample."""
        X = self.vectorizer.transform(texts)
        # Compute sparse row norms
        norms = np.sqrt(np.asarray(X.power(2).sum(axis=1)).ravel())
        return norms


class EnhancedImageHead(nn.Module):
    """Deep multi-layer classification head for ResNet-50 visual features."""
    def __init__(self, in_features: int = 2048, num_classes: int = 23, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedImageBaseline:
    """Enhanced visual classifier head operating on frozen ResNet-50 embeddings."""
    def __init__(self, num_classes: int = 23, in_features: int = 2048, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.head = EnhancedImageHead(in_features=in_features, num_classes=num_classes).to(self.device)
        self.thresholds: Optional[np.ndarray] = None
        self.genre_names: List[str] = []
        self.is_fitted = False

    def predict_proba(self, image_features: np.ndarray) -> np.ndarray:
        self.head.eval()
        with torch.no_grad():
            x = torch.from_numpy(image_features).float().to(self.device)
            logits = self.head(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, image_features: np.ndarray, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        probs = self.predict_proba(image_features)
        thresh = thresholds if thresholds is not None else (self.thresholds if self.thresholds is not None else 0.5)
        return (probs >= thresh).astype(int)

    def get_feature_norms(self, image_features: np.ndarray) -> np.ndarray:
        """Computes L2 norm of image feature vectors."""
        return np.linalg.norm(image_features, axis=1)


class EnhancedGatedMultimodalFusion:
    """
    Dynamic Gated Multimodal Fusion.
    Instead of a fixed static scalar alpha, computes per-instance gating weights
    conditioned on feature norms, modality uncertainties, and presence indicators.
    """
    def __init__(
        self,
        base_alpha: float = 0.5,
        norm_floor_ratio: float = 0.1,
        uncertainty_penalty_weight: float = 0.5
    ):
        self.base_alpha = base_alpha
        self.norm_floor_ratio = norm_floor_ratio
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.thresholds: Optional[np.ndarray] = None

    def fuse(
        self,
        text_probs: np.ndarray,
        image_probs: np.ndarray,
        text_norms: Optional[np.ndarray] = None,
        image_norms: Optional[np.ndarray] = None,
        text_uncertainty: Optional[np.ndarray] = None,
        image_uncertainty: Optional[np.ndarray] = None,
        has_text: Optional[np.ndarray] = None,
        has_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamically fuses text and image predictions using instance-level gates.

        Returns:
            Tuple of (fused_probs, text_weights_per_sample)
        """
        n_samples = len(text_probs)
        weights_text = np.full(n_samples, self.base_alpha, dtype=np.float32)

        # 1. Modality presence hard gating
        if has_text is not None:
            weights_text[~np.asarray(has_text, dtype=bool)] = 0.0
        if has_image is not None:
            weights_text[~np.asarray(has_image, dtype=bool)] = 1.0

        # 2. Feature norm dynamic scaling (if both present)
        if text_norms is not None and image_norms is not None:
            valid_both = np.ones(n_samples, dtype=bool)
            if has_text is not None:
                valid_both &= np.asarray(has_text, dtype=bool)
            if has_image is not None:
                valid_both &= np.asarray(has_image, dtype=bool)

            t_norm_safe = np.maximum(text_norms, 1e-6)
            i_norm_safe = np.maximum(image_norms, 1e-6)
            norm_factor = t_norm_safe / (t_norm_safe + i_norm_safe)
            weights_text[valid_both] = 0.5 * weights_text[valid_both] + 0.5 * norm_factor[valid_both]

        # 3. Uncertainty penalty (higher uncertainty reduces modality weight)
        if text_uncertainty is not None and image_uncertainty is not None:
            unc_diff = image_uncertainty - text_uncertainty  # Positive means image is MORE uncertain
            shift = 0.5 * np.tanh(self.uncertainty_penalty_weight * unc_diff)
            weights_text = np.clip(weights_text + shift, 0.0, 1.0)

        # Broadcast weights across classes: (N, 1) * (N, C)
        w_t = weights_text[:, np.newaxis]
        w_i = 1.0 - w_t
        fused_probs = w_t * text_probs + w_i * image_probs

        return fused_probs, weights_text
