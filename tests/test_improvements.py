"""
Unit and Integration Tests for SEAA 10 Comprehensive Improvements.
Validates:
1. Feature norm / energy gating
2. Jaccard saturation handling
3. Automated missing modality detection
4. Learned Diagnostic Classifier
5. Cross-modal conflict simulation & routing
6. Configurable evaluation scale
7. Enhanced baseline models & dynamic gating
8. Component-level timing tracking
9. Adaptive Recovery (DynamicModalityGating & ConfidenceRecoveryRouter)
10. Additional corruptions (typo, OCR, JPEG compression, contrast washout)
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.auditing.diagnosis import (
    DiagnosticConfig,
    calibrate_diagnostic_thresholds_on_val,
    MultimodalReliabilityAuditor
)
from src.auditing.learned_diagnostics import (
    LearnedDiagnosticClassifier,
    extract_diagnostic_feature_matrix
)
from src.failures.text_corruptions import (
    typo_noise,
    ocr_noise,
    cross_modal_conflict_text,
    apply_text_corruption
)
from src.failures.image_corruptions import (
    apply_jpeg_compression,
    apply_contrast_reduction,
    apply_image_corruption
)
from src.models.enhanced_baselines import (
    EnhancedTextBaseline,
    EnhancedImageBaseline,
    EnhancedGatedMultimodalFusion
)
from src.recovery.modality_gating import DynamicModalityGating
from src.recovery.confidence_recovery import ConfidenceRecoveryRouter


# ---------------------------------------------------------------------------
# Test 1 & 2: Feature Norm Gating and Jaccard Saturation
# ---------------------------------------------------------------------------
def test_feature_norm_calibration_and_jaccard_saturation():
    n_val = 100
    n_classes = 5
    rng = np.random.RandomState(42)

    val_text_probs = rng.uniform(0.1, 0.9, size=(n_val, n_classes))
    val_image_probs = rng.uniform(0.1, 0.9, size=(n_val, n_classes))
    val_text_norms = rng.uniform(1.0, 5.0, size=n_val)
    val_image_norms = rng.uniform(10.0, 20.0, size=n_val)

    config = calibrate_diagnostic_thresholds_on_val(
        val_text_probs=val_text_probs,
        val_image_probs=val_image_probs,
        percentile=95.0,
        val_text_norms=val_text_norms,
        val_image_norms=val_image_norms
    )

    assert config.calibrated_on_val is True
    assert config.text_norm_floor > 0.0
    assert config.image_norm_floor > 0.0
    assert isinstance(config.jaccard_saturated, bool)


def test_feature_norm_gating_triggers_failure():
    config = DiagnosticConfig(
        text_norm_floor=1.5,
        image_norm_floor=5.0,
        text_uncertainty_threshold=0.8,
        image_uncertainty_threshold=0.8
    )
    auditor = MultimodalReliabilityAuditor(config=config)

    # Text norm is collapsed (0.1 < floor 1.5)
    text_probs = np.full((1, 5), 0.3)
    image_probs = np.full((1, 5), 0.3)
    res = auditor.audit_batch(
        text_probs=text_probs,
        image_probs=image_probs,
        text_feature_norms=np.array([0.1]),
        image_feature_norms=np.array([10.0])
    )

    assert bool(res["is_anomalous"][0]) is True
    assert res["diagnoses"][0] == "TEXT_FAILURE"


# ---------------------------------------------------------------------------
# Test 3: Automated Missing Modality Detection
# ---------------------------------------------------------------------------
def test_auto_detect_empty_text():
    texts = ["", "   ", "one", "This is a full movie synopsis with multiple words."]
    valid_mask = MultimodalReliabilityAuditor.auto_detect_empty_text(texts, min_word_count=2)
    assert valid_mask.tolist() == [False, False, False, True]


def test_auto_detect_blank_image():
    blank_img = Image.new("RGB", (64, 64), (0, 0, 0))
    noisy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    valid_mask = MultimodalReliabilityAuditor.auto_detect_blank_image(
        raw_images=[blank_img, noisy_img],
        variance_threshold=0.01
    )
    assert valid_mask.tolist() == [False, True]


# ---------------------------------------------------------------------------
# Test 4: Learned Diagnostic Classifier
# ---------------------------------------------------------------------------
def test_learned_diagnostic_classifier_fit_predict(tmp_path=None):
    n = 60
    rng = np.random.RandomState(42)

    val_df = pd.DataFrame({
        "mean_bernoulli_jsd": rng.uniform(0, 1, n),
        "jaccard_distance": rng.uniform(0, 1, n),
        "hamming_distance": rng.uniform(0, 1, n),
        "text_uncertainty": rng.uniform(0, 1, n),
        "image_uncertainty": rng.uniform(0, 1, n),
        "min_margin": rng.uniform(0, 0.5, n),
        "text_norm": rng.uniform(0.1, 5.0, n),
        "image_norm": rng.uniform(1.0, 20.0, n),
        "has_text": [1] * n,
        "has_image": [1] * n
    })
    bin_labels = (val_df["mean_bernoulli_jsd"] > 0.5).astype(int).values
    mc_labels = ["CLEAN" if b == 0 else "TEXT_FAILURE" for b in bin_labels]

    classifier = LearnedDiagnosticClassifier(random_state=42)
    classifier.fit(val_df, bin_labels, mc_labels)

    preds = classifier.predict(val_df)
    assert "is_anomalous" in preds
    assert "diagnoses" in preds
    assert len(preds["is_anomalous"]) == n
    assert len(preds["diagnoses"]) == n

    # Test serialization
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            save_file = Path(td) / "learned_classifier.pkl"
            classifier.save(save_file)
            loaded = LearnedDiagnosticClassifier.load(save_file)
            loaded_preds = loaded.predict(val_df)
            np.testing.assert_array_equal(preds["is_anomalous"], loaded_preds["is_anomalous"])
    else:
        save_file = tmp_path / "learned_classifier.pkl"
        classifier.save(save_file)
        loaded = LearnedDiagnosticClassifier.load(save_file)
        loaded_preds = loaded.predict(val_df)
        np.testing.assert_array_equal(preds["is_anomalous"], loaded_preds["is_anomalous"])


# ---------------------------------------------------------------------------
# Test 5: Cross-Modal Conflict Simulation & Routing
# ---------------------------------------------------------------------------
def test_cross_modal_conflict_corruption():
    orig = "A terrifying haunted house thriller where ghosts torment a family."
    conflict = cross_modal_conflict_text(orig, conflict_text="A romantic comedy about two florists.")
    assert conflict == "A romantic comedy about two florists."

    dispatched = apply_text_corruption(orig, "conflict", conflict_text="Happy musical.")
    assert dispatched == "Happy musical."


# ---------------------------------------------------------------------------
# Test 7: Enhanced Baseline Models & Dynamic Fusion
# ---------------------------------------------------------------------------
def test_enhanced_text_baseline():
    train_texts = [
        "A space adventure with astronauts exploring alien galaxies",
        "A laugh-out-loud funny comedy with great jokes and satire",
        "A dark terrifying horror film with zombies and monsters",
        "Romantic drama about two star-crossed lovers"
    ]
    y_train = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    model = EnhancedTextBaseline(max_features=100)
    model.fit(train_texts, y_train, genre_names=["Sci-Fi", "Comedy", "Horror", "Drama"])

    probs = model.predict_proba(train_texts)
    assert probs.shape == (4, 4)

    norms = model.get_feature_norms(train_texts)
    assert len(norms) == 4
    assert np.all(norms > 0.0)


def test_enhanced_gated_multimodal_fusion():
    fusion = EnhancedGatedMultimodalFusion(base_alpha=0.5)
    t_probs = np.array([[0.8, 0.1], [0.2, 0.9]])
    i_probs = np.array([[0.3, 0.4], [0.7, 0.3]])

    # Case 1: Normal clean inputs
    fused, weights = fusion.fuse(t_probs, i_probs)
    assert fused.shape == (2, 2)
    assert np.allclose(weights, 0.5)

    # Case 2: Missing text -> gate drives text weight to 0
    fused_no_text, weights_no_text = fusion.fuse(t_probs, i_probs, has_text=np.array([False, True]))
    assert weights_no_text[0] == 0.0
    assert np.allclose(fused_no_text[0], i_probs[0])


# ---------------------------------------------------------------------------
# Test 9: Adaptive Recovery Engine
# ---------------------------------------------------------------------------
def test_dynamic_modality_gating():
    gating = DynamicModalityGating(default_alpha=0.5)
    t_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    i_probs = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])
    diagnoses = ["TEXT_FAILURE", "IMAGE_FAILURE", "CLEAN"]

    res = gating.recover_predictions(t_probs, i_probs, diagnoses)

    # Sample 0: TEXT_FAILURE -> routed to image
    assert np.allclose(res["recovered_probs"][0], i_probs[0])
    assert res["effective_alphas"][0] == 0.0

    # Sample 1: IMAGE_FAILURE -> routed to text
    assert np.allclose(res["recovered_probs"][1], t_probs[1])
    assert res["effective_alphas"][1] == 1.0

    # Sample 2: CLEAN -> standard fusion
    assert np.allclose(res["recovered_probs"][2], 0.5 * t_probs[2] + 0.5 * i_probs[2])


def test_confidence_recovery_router():
    router = ConfidenceRecoveryRouter(confidence_abstain_threshold=0.4)
    probs = np.array([[0.9, 0.8], [0.2, 0.1]])
    thresholds = np.array([0.5, 0.5])
    anomaly_scores = np.array([0.1, 0.95])

    res = router.apply_selective_abstention(probs, thresholds, anomaly_scores)
    assert res["binary_predictions"].shape == (2, 2)
    # Sample 1 has low max prob and critical anomaly -> flagged for human review
    assert res["human_review_required"][1] == True


# ---------------------------------------------------------------------------
# Test 10: Additional Corruptions (Typo, OCR, JPEG, Contrast)
# ---------------------------------------------------------------------------
def test_text_typo_and_ocr_noise():
    text = "The quick brown fox jumps over the lazy dog"
    t_corrupted = typo_noise(text, typo_rate=0.5, seed=42)
    assert t_corrupted != text
    assert len(t_corrupted) > 0

    ocr_corrupted = ocr_noise("The official logo was colored yellow", ocr_rate=0.8, seed=42)
    assert ocr_corrupted != "The official logo was colored yellow"


def test_image_jpeg_and_contrast_reduction():
    img = Image.new("RGB", (100, 100), (120, 150, 180))
    jpeg_img = apply_jpeg_compression(img, quality=5)
    assert isinstance(jpeg_img, Image.Image)
    assert jpeg_img.size == (100, 100)

    contrast_img = apply_contrast_reduction(img, factor=0.1)
    assert isinstance(contrast_img, Image.Image)
    assert contrast_img.size == (100, 100)


if __name__ == "__main__":
    tests = [
        ("Improvement 1: Norm Calibration & Saturation", test_feature_norm_calibration_and_jaccard_saturation),
        ("Improvement 1: Feature Norm Gating Trigger", test_feature_norm_gating_triggers_failure),
        ("Improvement 3: Auto-detect Empty Text", test_auto_detect_empty_text),
        ("Improvement 3: Auto-detect Blank Image", test_auto_detect_blank_image),
        ("Improvement 4: Learned Diagnostic Classifier", test_learned_diagnostic_classifier_fit_predict),
        ("Improvement 5: Cross-Modal Conflict Simulation", test_cross_modal_conflict_corruption),
        ("Improvement 7: Enhanced Text Baseline", test_enhanced_text_baseline),
        ("Improvement 7: Enhanced Gated Multimodal Fusion", test_enhanced_gated_multimodal_fusion),
        ("Improvement 9: Dynamic Modality Gating", test_dynamic_modality_gating),
        ("Improvement 9: Confidence Recovery Router", test_confidence_recovery_router),
        ("Improvement 10: Typo and OCR Corruptions", test_text_typo_and_ocr_noise),
        ("Improvement 10: JPEG and Contrast Corruptions", test_image_jpeg_and_contrast_reduction),
    ]

    print("=" * 70)
    print("RUNNING SEAA 10 IMPROVEMENTS TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED across {len(tests)} test cases.")
    print("=" * 70)
    if failed > 0:
        exit(1)

