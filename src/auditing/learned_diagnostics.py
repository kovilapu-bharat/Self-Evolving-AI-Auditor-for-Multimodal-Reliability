"""
Learned Diagnostic Classifier for Multimodal Reliability Auditing.
Improvement #4: Trains a machine-learned diagnostic classifier on multimodal signals
(disagreement metrics, predictive uncertainties, feature norms, and threshold margins)
to replace or augment rule-based heuristic routing.

Trained strictly on validation split data (clean and synthetically corrupted validation sets)
with zero access to test labels or test split data.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


FEATURE_COLUMNS = [
    "mean_bernoulli_jsd",
    "jaccard_distance",
    "hamming_distance",
    "text_composite_uncertainty",
    "image_composite_uncertainty",
    "uncertainty_diff",
    "uncertainty_ratio",
    "text_mean_margin",
    "image_mean_margin",
    "text_feature_norm",
    "image_feature_norm",
    "norm_ratio",
    "has_text",
    "has_image"
]


def extract_diagnostic_feature_matrix(
    audit_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts and standardizes the numeric signal features required by the learned classifier.

    Args:
        audit_df: DataFrame output from MultimodalReliabilityAuditor.audit_batch()['audit_df']
        feature_cols: Optional subset of feature column names

    Returns:
        DataFrame containing numeric feature matrix with no NaNs.
    """
    cols = feature_cols or FEATURE_COLUMNS
    df = pd.DataFrame(index=audit_df.index)

    # Pre-extract series with safe defaults
    text_u = audit_df["text_composite_uncertainty"] if "text_composite_uncertainty" in audit_df else (
        audit_df["text_uncertainty"] if "text_uncertainty" in audit_df else pd.Series(0.0, index=audit_df.index)
    )
    img_u = audit_df["image_composite_uncertainty"] if "image_composite_uncertainty" in audit_df else (
        audit_df["image_uncertainty"] if "image_uncertainty" in audit_df else pd.Series(0.0, index=audit_df.index)
    )
    text_n = audit_df["text_feature_norm"] if "text_feature_norm" in audit_df else (
        audit_df["text_norm"] if "text_norm" in audit_df else pd.Series(1.0, index=audit_df.index)
    )
    img_n = audit_df["image_feature_norm"] if "image_feature_norm" in audit_df else (
        audit_df["image_norm"] if "image_norm" in audit_df else pd.Series(1.0, index=audit_df.index)
    )

    for col in cols:
        if col in audit_df.columns:
            df[col] = audit_df[col].astype(float)
        elif col == "text_composite_uncertainty":
            df[col] = text_u.astype(float)
        elif col == "image_composite_uncertainty":
            df[col] = img_u.astype(float)
        elif col == "uncertainty_diff":
            df[col] = (text_u.astype(float) - img_u.astype(float))
        elif col == "uncertainty_ratio":
            df[col] = (text_u.astype(float) + 1e-5) / (img_u.astype(float) + 1e-5)
        elif col == "text_feature_norm":
            df[col] = text_n.astype(float)
        elif col == "image_feature_norm":
            df[col] = img_n.astype(float)
        elif col == "norm_ratio":
            df[col] = (text_n.astype(float) + 1e-5) / (img_n.astype(float) + 1e-5)
        elif col == "text_mean_margin":
            df[col] = audit_df["min_margin"].astype(float) if "min_margin" in audit_df else 0.5
        elif col == "image_mean_margin":
            df[col] = audit_df["min_margin"].astype(float) if "min_margin" in audit_df else 0.5
        else:
            df[col] = 0.0

    return df.fillna(0.0)


class LearnedDiagnosticClassifier:
    """
    Learned failure diagnosis model.
    Supports both binary anomaly detection and multi-class failure diagnosis.
    """
    def __init__(
        self,
        classifier_type: str = "gradient_boosting",
        random_state: int = 42
    ):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.feature_names = FEATURE_COLUMNS
        self.binary_model = None
        self.multiclass_model = None
        self.classes_ = None
        self.is_fitted = False

    def _init_estimator(self):
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                class_weight="balanced"
            )
        else:
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                random_state=self.random_state
            )

    def fit(
        self,
        val_audit_df: pd.DataFrame,
        binary_labels: np.ndarray,
        multiclass_labels: Optional[List[str]] = None
    ) -> "LearnedDiagnosticClassifier":
        """
        Fits binary and multi-class diagnostic models on validation signals.

        Args:
            val_audit_df: Feature DataFrame extracted from validation batch
            binary_labels: (N,) binary array (0 = normal, 1 = corrupted/anomalous)
            multiclass_labels: (N,) list of string labels ('CLEAN', 'MISSING_TEXT', etc.)

        Returns:
            Self (fitted instance)
        """
        X = extract_diagnostic_feature_matrix(val_audit_df, self.feature_names).values

        # Binary failure detection model
        self.binary_model = self._init_estimator()
        self.binary_model.fit(X, binary_labels)

        # Multi-class diagnosis model
        if multiclass_labels is not None:
            self.multiclass_model = self._init_estimator()
            self.multiclass_model.fit(X, multiclass_labels)
            self.classes_ = self.multiclass_model.classes_

        self.is_fitted = True
        return self

    def predict(
        self,
        audit_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predicts binary anomaly status and multi-class failure diagnosis.

        Args:
            audit_df: Feature DataFrame from audit batch

        Returns:
            Dictionary with:
            - 'is_anomalous': boolean array (N,)
            - 'anomaly_probabilities': float array (N,)
            - 'diagnoses': list of string predictions (N,)
            - 'diagnosis_probabilities': dict of class -> probabilities (if multiclass fitted)
        """
        if not self.is_fitted or self.binary_model is None:
            raise RuntimeError("LearnedDiagnosticClassifier must be fitted before predict.")

        X = extract_diagnostic_feature_matrix(audit_df, self.feature_names).values

        bin_preds = self.binary_model.predict(X)
        bin_probs = self.binary_model.predict_proba(X)[:, 1] if hasattr(self.binary_model, "predict_proba") else bin_preds.astype(float)

        results = {
            "is_anomalous": bin_preds.astype(bool),
            "anomaly_probabilities": bin_probs
        }

        if self.multiclass_model is not None:
            mc_preds = self.multiclass_model.predict(X)
            results["diagnoses"] = mc_preds.tolist()
            if hasattr(self.multiclass_model, "predict_proba"):
                results["diagnosis_probabilities"] = self.multiclass_model.predict_proba(X)

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves fitted classifier state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "binary_model": self.binary_model,
                "multiclass_model": self.multiclass_model,
                "classes_": self.classes_,
                "feature_names": self.feature_names,
                "classifier_type": self.classifier_type,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LearnedDiagnosticClassifier":
        """Loads fitted classifier state from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(classifier_type=data["classifier_type"], random_state=data["random_state"])
        obj.binary_model = data["binary_model"]
        obj.multiclass_model = data["multiclass_model"]
        obj.classes_ = data["classes_"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = data["is_fitted"]
        return obj
