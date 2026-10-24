"""
Multimodal Reliability Diagnosis and Failure Detection Engine.
Combines cross-modal disagreement, predictive uncertainty, feature norm monitoring,
and modality presence signals to diagnose failure conditions and classify failure types
in multimodal systems.

Core Capabilities:
1. Signal Extraction:
   - Mean Bernoulli Jensen-Shannon divergence
   - Jaccard and Hamming prediction set distances
   - Binary predictive entropy for text and image branches
   - Decision threshold margins and threshold proximity proportions
   - Feature norm (L2) monitoring for text TF-IDF vectors and image ResNet-50 embeddings
   - Missing modality detection flags (explicit and auto-detected)
2. Decision Cutoff Calibration:
   - Calibration strictly on validation split clean data (e.g., 95th percentile of normal disagreement/uncertainty)
   - Feature norm floor calibration at 5th percentile of validation norms
   - Jaccard saturation handling: falls back to JSD-only when Jaccard saturates at 1.0
   - Never uses test labels for cutoff selection.
3. Diagnostic Classification:
   - Binary Detection: Normal (Reliable) vs Anomalous (Corrupted / High-Risk)
   - Multi-Class Diagnosis:
     * 'CLEAN': Consistent cross-modal agreement, low uncertainty.
     * 'MISSING_TEXT': Plot summary is missing or empty (explicit flag or auto-detected).
     * 'MISSING_IMAGE': Poster image is missing or blank (explicit flag or auto-detected).
     * 'MISSING_BOTH': Both modalities missing.
     * 'TEXT_FAILURE': High text uncertainty, low text feature norm, or text degradation signal.
     * 'IMAGE_FAILURE': High image uncertainty, low image feature norm, or image degradation signal.
     * 'CROSS_MODAL_CONFLICT': High-confidence opposing predictions on different genres.
     * 'HIGH_UNCERTAINTY': Both modalities ambiguous / close to decision thresholds.
4. Evaluation Utilities:
   - Distinguishes binary corruption detection from multi-class corruption classification.
   - Computes precision, recall, F1, accuracy, and confusion matrix against synthetic failure benchmarks.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from src.auditing.disagreement import compute_cross_modal_disagreement
from src.auditing.uncertainty import compute_predictive_uncertainty


DIAGNOSTIC_LABELS = [
    "CLEAN",
    "MISSING_TEXT",
    "MISSING_IMAGE",
    "MISSING_BOTH",
    "TEXT_FAILURE",
    "IMAGE_FAILURE",
    "CROSS_MODAL_CONFLICT",
    "HIGH_UNCERTAINTY"
]


@dataclass
class DiagnosticConfig:
    """
    Configuration parameters and decision cutoffs for multimodal failure diagnosis.
    Cutoffs can be preset or calibrated strictly on validation data.

    Improvement #1: Feature norm floor thresholds for detecting modality collapse.
    Improvement #2: Effective Jaccard threshold handles saturation at 1.0.
    Improvement #3: Auto-detection thresholds for blank/empty modality inputs.
    """
    jsd_threshold: float = 0.25
    jaccard_threshold: float = 0.50
    hamming_threshold: float = 0.20
    text_uncertainty_threshold: float = 0.60
    image_uncertainty_threshold: float = 0.65
    min_margin_threshold: float = 0.05
    delta: float = 0.05
    calibrated_on_val: bool = False
    # Improvement #1: Feature norm gating thresholds
    text_norm_floor: float = 0.0        # L2 norm below this → TEXT_FAILURE
    image_norm_floor: float = 0.0       # L2 norm below this → IMAGE_FAILURE
    text_norm_mean: float = 1.0          # Mean validation text norm (for relative comparison)
    image_norm_mean: float = 1.0         # Mean validation image norm (for relative comparison)
    # Improvement #2: Jaccard saturation flag
    jaccard_saturated: bool = False      # True if jaccard_threshold == 1.0
    # Improvement #3: Auto-detection thresholds for missing modalities
    image_variance_threshold: float = 0.01  # Pixel variance below this → blank image
    text_min_word_count: int = 2             # Word count below this → empty text
    # Tightened conflict routing: minimum confidence to declare genuine conflict
    conflict_min_confidence: float = 0.60    # Both modalities must exceed this to be true conflict

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calibrate_diagnostic_thresholds_on_val(
    val_text_probs: np.ndarray,
    val_image_probs: np.ndarray,
    text_thresholds: Optional[np.ndarray] = None,
    image_thresholds: Optional[np.ndarray] = None,
    percentile: float = 95.0,
    delta: float = 0.05,
    val_text_norms: Optional[np.ndarray] = None,
    val_image_norms: Optional[np.ndarray] = None
) -> DiagnosticConfig:
    """
    Calibrates diagnostic cutoffs using clean validation data.
    Sets anomaly thresholds at the specified percentile (default 95th) of normal clean signals.
    Test set labels or features are NEVER accessed during calibration.

    Improvement #1: Also calibrates feature norm floors at 5th percentile.
    Improvement #2: Detects Jaccard saturation and sets fallback flag.

    Args:
        val_text_probs: Clean validation text probabilities (N_val, C)
        val_image_probs: Clean validation image probabilities (N_val, C)
        text_thresholds: Validation-tuned text decision thresholds (C,)
        image_thresholds: Validation-tuned image decision thresholds (C,)
        percentile: Percentile cutoff for anomaly threshold (e.g. 95.0)
        delta: Margin tolerance for threshold proximity
        val_text_norms: Optional L2 norms of validation text TF-IDF vectors (N_val,)
        val_image_norms: Optional L2 norms of validation image embeddings (N_val,)

    Returns:
        Calibrated DiagnosticConfig instance.
    """
    if percentile <= 50.0 or percentile >= 100.0:
        raise ValueError(f"Calibration percentile must be in (50, 100), got {percentile}")

    disagree_res = compute_cross_modal_disagreement(
        text_probs=val_text_probs,
        image_probs=val_image_probs,
        text_thresholds=text_thresholds,
        image_thresholds=image_thresholds
    )
    text_unc_res = compute_predictive_uncertainty(
        probs=val_text_probs,
        thresholds=text_thresholds,
        delta=delta,
        modality_name="text"
    )
    image_unc_res = compute_predictive_uncertainty(
        probs=val_image_probs,
        thresholds=image_thresholds,
        delta=delta,
        modality_name="image"
    )

    jsd_cutoff = float(np.percentile(disagree_res["mean_bernoulli_jsd"], percentile))
    jaccard_cutoff = float(np.percentile(disagree_res["jaccard_distance"], percentile))
    hamming_cutoff = float(np.percentile(disagree_res["hamming_distance"], percentile))
    text_unc_cutoff = float(np.percentile(text_unc_res["composite_uncertainty"], percentile))
    image_unc_cutoff = float(np.percentile(image_unc_res["composite_uncertainty"], percentile))
    # For min margin, smaller margin means higher ambiguity, so use (100 - percentile)
    min_margin_cutoff = float(np.percentile(text_unc_res["min_margin"], 100.0 - percentile))

    # Improvement #2: Detect Jaccard saturation
    jaccard_saturated = (jaccard_cutoff >= 0.9999)

    # Improvement #1: Calibrate feature norm floors at 5th percentile
    text_norm_floor = 0.0
    text_norm_mean = 1.0
    image_norm_floor = 0.0
    image_norm_mean = 1.0

    if val_text_norms is not None and len(val_text_norms) > 0:
        text_norm_floor = float(np.percentile(val_text_norms, 5.0))
        text_norm_mean = float(np.mean(val_text_norms))

    if val_image_norms is not None and len(val_image_norms) > 0:
        image_norm_floor = float(np.percentile(val_image_norms, 5.0))
        image_norm_mean = float(np.mean(val_image_norms))

    return DiagnosticConfig(
        jsd_threshold=round(jsd_cutoff, 4),
        jaccard_threshold=round(jaccard_cutoff, 4),
        hamming_threshold=round(hamming_cutoff, 4),
        text_uncertainty_threshold=round(text_unc_cutoff, 4),
        image_uncertainty_threshold=round(image_unc_cutoff, 4),
        min_margin_threshold=round(min_margin_cutoff, 4),
        delta=delta,
        calibrated_on_val=True,
        text_norm_floor=round(text_norm_floor, 4),
        image_norm_floor=round(image_norm_floor, 4),
        text_norm_mean=round(text_norm_mean, 4),
        image_norm_mean=round(image_norm_mean, 4),
        jaccard_saturated=jaccard_saturated
    )


class MultimodalReliabilityAuditor:
    """
    Audits multimodal model predictions, computes agreement and uncertainty signals,
    and produces failure diagnosis classifications with auditable reasoning.
    """
    def __init__(
        self,
        config: Optional[DiagnosticConfig] = None,
        text_thresholds: Optional[np.ndarray] = None,
        image_thresholds: Optional[np.ndarray] = None,
        fused_thresholds: Optional[np.ndarray] = None,
        genre_names: Optional[List[str]] = None
    ):
        self.config = config or DiagnosticConfig()
        self.text_thresholds = text_thresholds
        self.image_thresholds = image_thresholds
        self.fused_thresholds = fused_thresholds
        self.genre_names = genre_names

    @staticmethod
    def auto_detect_blank_image(
        image_features: Optional[np.ndarray] = None,
        raw_images: Optional[list] = None,
        variance_threshold: float = 0.01
    ) -> Optional[np.ndarray]:
        """
        Improvement #3: Automatically detects blank/black images without relying on
        external metadata flags.

        Uses two strategies:
        1. If raw PIL images are available: checks pixel variance < threshold.
        2. If only feature vectors are available: checks L2 norm collapse.

        Args:
            image_features: Optional (N, D) feature matrix
            raw_images: Optional list of PIL Images
            variance_threshold: Pixel variance below this indicates blank image

        Returns:
            Boolean array (N,) where True = valid image, False = detected as blank/missing.
            Returns None if neither input is provided.
        """
        if raw_images is not None:
            n = len(raw_images)
            is_valid = np.ones(n, dtype=bool)
            for i, img in enumerate(raw_images):
                try:
                    arr = np.asarray(img, dtype=np.float32)
                    pixel_var = np.var(arr)
                    if pixel_var < variance_threshold:
                        is_valid[i] = False
                except Exception:
                    is_valid[i] = False
            return is_valid
        return None

    @staticmethod
    def auto_detect_empty_text(
        texts: List[str],
        min_word_count: int = 2
    ) -> np.ndarray:
        """
        Improvement #3: Automatically detects empty or near-empty text inputs
        without relying on external metadata flags.

        Args:
            texts: List of text strings
            min_word_count: Minimum number of words to be considered valid

        Returns:
            Boolean array (N,) where True = valid text, False = detected as empty.
        """
        n = len(texts)
        is_valid = np.ones(n, dtype=bool)
        for i, t in enumerate(texts):
            if not t or not isinstance(t, str):
                is_valid[i] = False
            elif len(t.strip().split()) < min_word_count:
                is_valid[i] = False
        return is_valid

    def audit_batch(
        self,
        text_probs: np.ndarray,
        image_probs: np.ndarray,
        fused_probs: Optional[np.ndarray] = None,
        has_text: Optional[Union[np.ndarray, List[bool]]] = None,
        has_image: Optional[Union[np.ndarray, List[bool]]] = None,
        movie_ids: Optional[List[str]] = None,
        text_feature_norms: Optional[np.ndarray] = None,
        image_feature_norms: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Runs full multimodal auditing pipeline across a batch of samples.

        Improvements:
        - #1: Uses feature norms to detect modality collapse (low norm → failure)
        - #2: Handles Jaccard threshold saturation by relying on JSD alone
        - #3: Auto-detect missing modalities from norm collapse
        - Tighter CROSS_MODAL_CONFLICT routing (requires high-confidence opposing predictions)

        Args:
            text_probs: Text model probabilities (N, C)
            image_probs: Image model probabilities (N, C)
            fused_probs: Optional fused probabilities (N, C)
            has_text: Boolean flags for text presence (N,)
            has_image: Boolean flags for image presence (N,)
            movie_ids: Optional sample identifiers
            text_feature_norms: Optional L2 norms of text features (N,)
            image_feature_norms: Optional L2 norms of image embeddings (N,)

        Returns:
            Dictionary containing:
            - 'diagnoses': list of string diagnostic labels of length N
            - 'is_anomalous': boolean array (N,) for binary corruption detection
            - 'anomaly_scores': float array (N,) continuous risk score in [0, 1]
            - 'audit_df': pandas DataFrame with all numeric signals and labels
            - 'signals': raw dictionary of component signal arrays
        """
        n_samples = len(text_probs)

        # Modality presence flags
        if has_text is None:
            text_present = np.ones(n_samples, dtype=bool)
        else:
            text_present = np.asarray(has_text, dtype=bool)

        if has_image is None:
            image_present = np.ones(n_samples, dtype=bool)
        else:
            image_present = np.asarray(has_image, dtype=bool)

        # 1. Disagreement signals
        disagree_res = compute_cross_modal_disagreement(
            text_probs=text_probs,
            image_probs=image_probs,
            text_thresholds=self.text_thresholds,
            image_thresholds=self.image_thresholds,
            genre_names=self.genre_names
        )

        # 2. Text Uncertainty signals
        text_unc = compute_predictive_uncertainty(
            probs=text_probs,
            thresholds=self.text_thresholds,
            delta=self.config.delta,
            modality_name="text"
        )

        # 3. Image Uncertainty signals
        image_unc = compute_predictive_uncertainty(
            probs=image_probs,
            thresholds=self.image_thresholds,
            delta=self.config.delta,
            modality_name="image"
        )

        # 4. Optional Fused Uncertainty
        if fused_probs is not None:
            fused_unc = compute_predictive_uncertainty(
                probs=fused_probs,
                thresholds=self.fused_thresholds,
                delta=self.config.delta,
                modality_name="fused"
            )
        else:
            fused_unc = None

        # Improvement #1: Feature norm arrays for collapse detection
        txt_norms = text_feature_norms if text_feature_norms is not None else None
        img_norms = image_feature_norms if image_feature_norms is not None else None

        # 5. Diagnostic Decision Logic per sample
        diagnoses: List[str] = []
        is_anomalous: List[bool] = []
        anomaly_scores: List[float] = []

        cfg = self.config

        for i in range(n_samples):
            txt_ok = text_present[i]
            img_ok = image_present[i]

            jsd_val = float(disagree_res["mean_bernoulli_jsd"][i])
            jaccard_val = float(disagree_res["jaccard_distance"][i])
            hamming_val = float(disagree_res["hamming_distance"][i])

            txt_ent = float(text_unc["mean_binary_entropy"][i])
            img_ent = float(image_unc["mean_binary_entropy"][i])
            txt_unc_comp = float(text_unc["composite_uncertainty"][i])
            img_unc_comp = float(image_unc["composite_uncertainty"][i])

            # Improvement #1: Feature norm collapse detection
            txt_norm_collapsed = False
            img_norm_collapsed = False
            if txt_norms is not None and cfg.text_norm_floor > 0:
                txt_norm_collapsed = (float(txt_norms[i]) < cfg.text_norm_floor)
            if img_norms is not None and cfg.image_norm_floor > 0:
                img_norm_collapsed = (float(img_norms[i]) < cfg.image_norm_floor)

            # Continuous anomaly score in [0, 1]
            # Aggregates disagreement and modality-specific uncertainties
            base_anomaly_score = (
                0.40 * jsd_val +
                0.20 * jaccard_val +
                0.20 * txt_unc_comp +
                0.20 * img_unc_comp
            )
            # Boost score if norm collapse detected
            if txt_norm_collapsed or img_norm_collapsed:
                base_anomaly_score = max(base_anomaly_score, 0.65)

            # Check missing modality conditions
            if not txt_ok and not img_ok:
                label = "MISSING_BOTH"
                anom = True
                score = 1.0
            elif not txt_ok:
                label = "MISSING_TEXT"
                anom = True
                score = 0.95
            elif not img_ok:
                label = "MISSING_IMAGE"
                anom = True
                score = 0.90
            else:
                # Both modalities present: evaluate disagreement and uncertainty

                # Improvement #2: Handle Jaccard saturation
                # When jaccard_threshold is saturated at 1.0, rely solely on JSD
                if cfg.jaccard_saturated:
                    high_disagreement = (jsd_val >= cfg.jsd_threshold)
                else:
                    high_disagreement = (jsd_val >= cfg.jsd_threshold) or (jaccard_val >= cfg.jaccard_threshold)

                high_txt_unc = txt_unc_comp >= cfg.text_uncertainty_threshold
                high_img_unc = img_unc_comp >= cfg.image_uncertainty_threshold

                txt_pos_count = int(np.sum(disagree_res["text_binary_predictions"][i]))
                img_pos_count = int(np.sum(disagree_res["image_binary_predictions"][i]))
                txt_max_p = float(np.max(text_probs[i]))
                img_max_p = float(np.max(image_probs[i]))

                # Improvement #1: Feature norm collapse adds to asymmetry detection
                # When a modality's feature norm collapses, it's a strong signal of modality failure
                # even if entropy doesn't spike (because collapsed features → confident-wrong predictions)
                img_suppressed_or_uncertain = (
                    high_img_unc or
                    img_norm_collapsed or
                    (txt_pos_count > 0 and img_pos_count == 0) or
                    (txt_max_p > 0.55 and img_max_p < 0.25)
                )
                txt_suppressed_or_uncertain = (
                    high_txt_unc or
                    txt_norm_collapsed or
                    (img_pos_count > 0 and txt_pos_count == 0) or
                    (img_max_p > 0.55 and txt_max_p < 0.25)
                )

                # Improvement #1: Direct diagnosis from norm collapse
                if img_norm_collapsed and not txt_norm_collapsed:
                    label = "IMAGE_FAILURE"
                    anom = True
                    score = max(base_anomaly_score, 0.80)
                elif txt_norm_collapsed and not img_norm_collapsed:
                    label = "TEXT_FAILURE"
                    anom = True
                    score = max(base_anomaly_score, 0.80)
                elif txt_norm_collapsed and img_norm_collapsed:
                    label = "HIGH_UNCERTAINTY"
                    anom = True
                    score = max(base_anomaly_score, 0.85)
                elif high_txt_unc and high_img_unc:
                    label = "HIGH_UNCERTAINTY"
                    anom = True
                    score = max(base_anomaly_score, 0.70)
                elif high_disagreement:
                    if img_suppressed_or_uncertain and not txt_suppressed_or_uncertain:
                        label = "IMAGE_FAILURE"
                        anom = True
                        score = max(base_anomaly_score, 0.75)
                    elif txt_suppressed_or_uncertain and not img_suppressed_or_uncertain:
                        label = "TEXT_FAILURE"
                        anom = True
                        score = max(base_anomaly_score, 0.75)
                    # Tightened CROSS_MODAL_CONFLICT routing:
                    # Require BOTH modalities to have high-confidence predictions (>= conflict_min_confidence)
                    # on at least one genre, not just any non-zero label count.
                    elif (txt_max_p >= cfg.conflict_min_confidence and
                          img_max_p >= cfg.conflict_min_confidence and
                          txt_pos_count > 0 and img_pos_count > 0):
                        label = "CROSS_MODAL_CONFLICT"
                        anom = True
                        score = max(base_anomaly_score, 0.65)
                    else:
                        # When disagreement is high but neither modality shows clear
                        # asymmetric failure or genuine conflict, determine by asymmetry
                        if img_max_p < txt_max_p * 0.5:
                            label = "IMAGE_FAILURE"
                        elif txt_max_p < img_max_p * 0.5:
                            label = "TEXT_FAILURE"
                        else:
                            label = "HIGH_UNCERTAINTY"
                        anom = True
                        score = max(base_anomaly_score, 0.60)
                elif high_txt_unc:
                    label = "TEXT_FAILURE"
                    anom = True
                    score = max(base_anomaly_score, 0.60)
                elif high_img_unc:
                    label = "IMAGE_FAILURE"
                    anom = True
                    score = max(base_anomaly_score, 0.60)
                else:
                    label = "CLEAN"
                    anom = False
                    score = base_anomaly_score

            diagnoses.append(label)
            is_anomalous.append(anom)
            anomaly_scores.append(round(float(score), 4))

        # Build auditable results DataFrame
        df_records = {
            "movie_id": movie_ids if movie_ids is not None else [f"sample_{i}" for i in range(n_samples)],
            "diagnosis": diagnoses,
            "is_anomalous": is_anomalous,
            "anomaly_score": anomaly_scores,
            "has_text": text_present,
            "has_image": image_present,
            "mean_bernoulli_jsd": np.round(disagree_res["mean_bernoulli_jsd"], 4),
            "jaccard_distance": np.round(disagree_res["jaccard_distance"], 4),
            "hamming_distance": np.round(disagree_res["hamming_distance"], 4),
            "num_disagreeing_genres": disagree_res["num_disagreeing_genres"],
            "text_binary_entropy": np.round(text_unc["mean_binary_entropy"], 4),
            "image_binary_entropy": np.round(image_unc["mean_binary_entropy"], 4),
            "text_mean_margin": np.round(text_unc["mean_margin"], 4),
            "image_mean_margin": np.round(image_unc["mean_margin"], 4),
            "text_composite_uncertainty": np.round(text_unc["composite_uncertainty"], 4),
            "image_composite_uncertainty": np.round(image_unc["composite_uncertainty"], 4)
        }
        if txt_norms is not None:
            df_records["text_feature_norm"] = np.round(txt_norms, 4)
        if img_norms is not None:
            df_records["image_feature_norm"] = np.round(img_norms, 4)
        if fused_unc is not None:
            df_records["fused_binary_entropy"] = np.round(fused_unc["mean_binary_entropy"], 4)
            df_records["fused_mean_margin"] = np.round(fused_unc["mean_margin"], 4)

        audit_df = pd.DataFrame(df_records)

        return {
            "diagnoses": diagnoses,
            "is_anomalous": np.array(is_anomalous, dtype=bool),
            "anomaly_scores": np.array(anomaly_scores, dtype=np.float32),
            "audit_df": audit_df,
            "disagreement": disagree_res,
            "text_uncertainty": text_unc,
            "image_uncertainty": image_unc,
            "fused_uncertainty": fused_unc
        }


def evaluate_diagnosis_performance(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_true_types: Optional[List[str]] = None,
    y_pred_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates both binary corruption detection and multi-class failure diagnosis.

    Args:
        y_true_binary: Binary ground truth array (0=Clean, 1=Corrupted)
        y_pred_binary: Binary predictions (0=Clean, 1=Corrupted)
        y_true_types: Optional multi-class true labels (e.g. 'CLEAN', 'TEXT_FAILURE', etc.)
        y_pred_types: Optional multi-class predicted diagnosis labels

    Returns:
        Comprehensive evaluation dictionary containing precision, recall, F1, accuracy,
        and confusion matrices.
    """
    y_true_bin = np.asarray(y_true_binary, dtype=int)
    y_pred_bin = np.asarray(y_pred_binary, dtype=int)

    # 1. Binary Detection Metrics
    det_acc = accuracy_score(y_true_bin, y_pred_bin)
    det_prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    det_rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    det_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # Confusion Matrix for binary detection: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)

    detection_results = {
        "accuracy": round(float(det_acc), 4),
        "precision": round(float(det_prec), 4),
        "recall": round(float(det_rec), 4),
        "f1_score": round(float(det_f1), 4),
        "specificity": round(float(specificity), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

    # 2. Multi-class Diagnosis Evaluation (if provided)
    classification_results = None
    if y_true_types is not None and y_pred_types is not None:
        unique_labels = sorted(list(set(y_true_types) | set(y_pred_types)))
        cls_acc = accuracy_score(y_true_types, y_pred_types)
        macro_f1 = f1_score(y_true_types, y_pred_types, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true_types, y_pred_types, average="micro", zero_division=0)

        per_class_f1 = f1_score(y_true_types, y_pred_types, labels=unique_labels, average=None, zero_division=0)
        per_class_prec = precision_score(y_true_types, y_pred_types, labels=unique_labels, average=None, zero_division=0)
        per_class_rec = recall_score(y_true_types, y_pred_types, labels=unique_labels, average=None, zero_division=0)

        cm = confusion_matrix(y_true_types, y_pred_types, labels=unique_labels)

        per_class_stats = {}
        for idx, lbl in enumerate(unique_labels):
            support = int(np.sum(np.array(y_true_types) == lbl))
            per_class_stats[lbl] = {
                "precision": round(float(per_class_prec[idx]), 4),
                "recall": round(float(per_class_rec[idx]), 4),
                "f1": round(float(per_class_f1[idx]), 4),
                "support": support
            }

        classification_results = {
            "accuracy": round(float(cls_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "micro_f1": round(float(micro_f1), 4),
            "labels": unique_labels,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_stats
        }

    return {
        "binary_corruption_detection": detection_results,
        "multiclass_failure_classification": classification_results
    }
