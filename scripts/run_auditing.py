"""
SEAA Inline Reliability Auditing Runner.
Executes comprehensive cross-modal disagreement auditing, predictive uncertainty estimation,
controlled modality corruption benchmarks, and reliability failure diagnosis.

Adheres strictly to SEAA auditing requirements:
- Uses existing baseline checkpoints (outputs/checkpoints/) and validation thresholds.
- Evaluates controlled synthetic corruptions without mutating source data.
- Calibrates diagnostic thresholds strictly on validation data (zero test label leakage).
- Evaluates both binary corruption detection and multi-class failure diagnosis.
- Accurately reports performance degradation (Micro-F1, Macro-F1, Hamming Loss, mAP).
- Distinguishes cached classifier timing from raw image feature extraction latency.
- Produces machine-readable outputs (JSON, CSV, figures) in outputs/auditing/.
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import (
    GENRES,
    get_default_image_transform
)
from src.evaluation.metrics import evaluate_multilabel_predictions
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel
from src.failures.text_corruptions import (
    VALID_TEXT_CORRUPTIONS,
    corrupt_text_batch
)
from src.failures.image_corruptions import (
    VALID_IMAGE_CORRUPTIONS,
    extract_corrupted_image_features,
    apply_missing_image
)
from src.auditing.diagnosis import (
    calibrate_diagnostic_thresholds_on_val,
    MultimodalReliabilityAuditor,
    evaluate_diagnosis_performance
)
from src.auditing.learned_diagnostics import (
    LearnedDiagnosticClassifier,
    extract_diagnostic_feature_matrix
)
from src.recovery.modality_gating import DynamicModalityGating
from src.recovery.confidence_recovery import ConfidenceRecoveryRouter


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Phase4Audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="SEAA Phase 4: Cross-Modal Disagreement, Uncertainty & Reliability Diagnosis Runner."
    )
    parser.add_argument(
        "--sample_size",
        "--samples",
        dest="sample_size",
        type=int,
        default=1000,
        help="Number of test samples to audit for end-to-end image corruptions (default: 1000, 0 for full test set)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible corruptions (default: 42)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/auditing",
        help="Directory to save auditing evaluation artifacts (default: outputs/auditing)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for feature extraction (default: cpu)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_file = project_root / "outputs" / "logs" / "phase4_audit.log"

    logger = setup_logger(log_file)
    logger.info("=" * 80)
    logger.info("SEAA PHASE 4: AUDITING, UNCERTAINTY & FAILURE DIAGNOSIS BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Target Output Directory: {output_dir}")
    logger.info(f"Execution Device: {args.device} | Base Seed: {args.seed}")

    # 1. Load Split Indices and Metadata
    data_dir = project_root / "data"
    with open(data_dir / "split_indices.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    meta_df = pd.read_csv(data_dir / "metadata.csv").sort_values("dataset_idx").reset_index(drop=True)
    logger.info(
        f"Dataset loaded: {len(meta_df)} total records | "
        f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}"
    )

    # 2. Load Checkpoints
    ckpt_dir = project_root / "outputs" / "checkpoints"
    logger.info("Loading Phase 3 baseline checkpoints...")

    text_model = TextBaselineModel.load(ckpt_dir / "text_baseline.joblib")
    image_model = ImageBaselineModel.load(ckpt_dir / "image_baseline.joblib")
    mm_model = MultimodalBaselineModel.load(ckpt_dir / "multimodal_baseline.json")

    logger.info(f"Text Baseline loaded. Vocabulary: {len(text_model.vectorizer.vocabulary_)} features.")
    logger.info(f"Image Baseline loaded. Scaler mean shape: {image_model.scaler.mean_.shape}.")
    logger.info(f"Multimodal Baseline loaded. Optimal alpha: {mm_model.alpha:.2f}.")

    # Load pre-extracted visual embeddings
    feats_file = data_dir / "processed" / "resnet50_features.npy"
    if not feats_file.exists():
        logger.error(f"Missing cached features at {feats_file}")
        sys.exit(1)
    all_cached_feats = np.memmap(feats_file, dtype=np.float32, mode="r", shape=(len(meta_df), 2048))

    # 3. Validation Split Calibration (Zero Test Leakage)
    logger.info("Extracting clean validation predictions and feature norms for diagnostic calibration...")
    val_indices = splits["val"]
    val_texts = [str(meta_df.iloc[i]["text"]) if pd.notnull(meta_df.iloc[i]["text"]) else "" for i in val_indices]
    val_cached_feats = np.asarray(all_cached_feats[val_indices])

    val_text_probs = text_model.predict_proba(val_texts)
    val_image_probs = image_model.predict_proba(val_cached_feats)

    # Improvement #1: Extract validation feature norms
    val_text_sparse = text_model.vectorizer.transform(val_texts)
    val_text_norms = np.sqrt(np.asarray(val_text_sparse.power(2).sum(axis=1)).ravel())
    val_image_norms = np.linalg.norm(val_cached_feats, axis=1)

    logger.info("Calibrating diagnostic cutoffs on validation data at 95th percentile (with norm floors)...")
    diag_config = calibrate_diagnostic_thresholds_on_val(
        val_text_probs=val_text_probs,
        val_image_probs=val_image_probs,
        text_thresholds=text_model.thresholds,
        image_thresholds=image_model.thresholds,
        percentile=95.0,
        val_text_norms=val_text_norms,
        val_image_norms=val_image_norms
    )
    logger.info(
        f"Calibrated Cutoffs: JSD={diag_config.jsd_threshold:.4f} | "
        f"Jaccard={diag_config.jaccard_threshold:.4f} (saturated={diag_config.jaccard_saturated}) | "
        f"TextUnc={diag_config.text_uncertainty_threshold:.4f} | "
        f"ImageUnc={diag_config.image_uncertainty_threshold:.4f} | "
        f"TextNormFloor={diag_config.text_norm_floor:.4f} | "
        f"ImageNormFloor={diag_config.image_norm_floor:.4f}"
    )

    # Instantiate Auditor
    auditor = MultimodalReliabilityAuditor(
        config=diag_config,
        text_thresholds=text_model.thresholds,
        image_thresholds=image_model.thresholds,
        fused_thresholds=mm_model.thresholds,
        genre_names=GENRES
    )

    # Improvement #4: Train Learned Diagnostic Classifier strictly on Validation Split
    logger.info("Training Learned Diagnostic Classifier on synthetic validation corruptions (zero test leakage)...")
    val_sub_n = min(300, len(val_texts))
    sub_val_texts = val_texts[:val_sub_n]
    sub_val_feats = val_cached_feats[:val_sub_n]
    sub_val_t_probs = val_text_probs[:val_sub_n]
    sub_val_i_probs = val_image_probs[:val_sub_n]
    sub_val_f_probs = mm_model.predict_proba(sub_val_t_probs, sub_val_i_probs)
    sub_val_t_norms = val_text_norms[:val_sub_n]
    sub_val_i_norms = val_image_norms[:val_sub_n]

    # Generate synthetic validation conditions for classifier training
    val_train_dfs = []
    val_train_bin = []
    val_train_mc = []

    # Clean val samples
    clean_val_audit = auditor.audit_batch(
        text_probs=sub_val_t_probs,
        image_probs=sub_val_i_probs,
        fused_probs=sub_val_f_probs,
        text_feature_norms=sub_val_t_norms,
        image_feature_norms=sub_val_i_norms
    )
    val_train_dfs.append(clean_val_audit["audit_df"])
    val_train_bin.extend([0] * val_sub_n)
    val_train_mc.extend(["CLEAN"] * val_sub_n)

    # Synthetic text failure on val (empty / truncated)
    empty_t_probs = text_model.predict_proba([""] * val_sub_n)
    empty_t_audit = auditor.audit_batch(
        text_probs=empty_t_probs,
        image_probs=sub_val_i_probs,
        fused_probs=sub_val_i_probs,
        has_text=[False] * val_sub_n,
        text_feature_norms=np.zeros(val_sub_n),
        image_feature_norms=sub_val_i_norms
    )
    val_train_dfs.append(empty_t_audit["audit_df"])
    val_train_bin.extend([1] * val_sub_n)
    val_train_mc.extend(["MISSING_TEXT"] * val_sub_n)

    # Synthetic image failure on val (zero features / missing)
    zero_img_feats = np.zeros_like(sub_val_feats)
    zero_i_probs = image_model.predict_proba(zero_img_feats)
    missing_i_audit = auditor.audit_batch(
        text_probs=sub_val_t_probs,
        image_probs=zero_i_probs,
        fused_probs=sub_val_t_probs,
        has_image=[False] * val_sub_n,
        text_feature_norms=sub_val_t_norms,
        image_feature_norms=np.zeros(val_sub_n)
    )
    val_train_dfs.append(missing_i_audit["audit_df"])
    val_train_bin.extend([1] * val_sub_n)
    val_train_mc.extend(["MISSING_IMAGE"] * val_sub_n)

    # Synthetic conflict on val
    conflict_t_texts = ["A hilarious slapstick romantic comedy about two puppies fall in love in New York."] * val_sub_n
    conflict_t_probs = text_model.predict_proba(conflict_t_texts)
    conflict_val_audit = auditor.audit_batch(
        text_probs=conflict_t_probs,
        image_probs=sub_val_i_probs,
        fused_probs=0.5 * conflict_t_probs + 0.5 * sub_val_i_probs,
        text_feature_norms=sub_val_t_norms,
        image_feature_norms=sub_val_i_norms
    )
    val_train_dfs.append(conflict_val_audit["audit_df"])
    val_train_bin.extend([1] * val_sub_n)
    val_train_mc.extend(["CROSS_MODAL_CONFLICT"] * val_sub_n)

    val_train_full_df = pd.concat(val_train_dfs, ignore_index=True)
    learned_classifier = LearnedDiagnosticClassifier(classifier_type="gradient_boosting", random_state=args.seed)
    learned_classifier.fit(
        val_audit_df=val_train_full_df,
        binary_labels=np.array(val_train_bin),
        multiclass_labels=val_train_mc
    )
    learned_ckpt_path = ckpt_dir / "learned_diagnostic_classifier.pkl"
    learned_classifier.save(learned_ckpt_path)
    logger.info(f"Learned Diagnostic Classifier trained on {len(val_train_full_df)} val samples and saved to {learned_ckpt_path}")

    # 4. Prepare Test Audit Samples
    test_indices = splits["test"]
    total_test_samples = len(test_indices)
    if args.sample_size > 0 and args.sample_size < total_test_samples:
        audit_n = args.sample_size
        audit_test_indices = test_indices[:audit_n]
        logger.info(f"Auditing representative test subset of {audit_n} samples for end-to-end corruptions.")
    else:
        audit_n = total_test_samples
        audit_test_indices = test_indices
        logger.info(f"Auditing full test set of {audit_n} samples.")

    audit_meta = meta_df.iloc[audit_test_indices].reset_index(drop=True)
    test_clean_texts = [str(t) if pd.notnull(t) else "" for t in audit_meta["text"]]
    test_movie_ids = audit_meta["movie_id"].tolist()
    test_cached_feats = np.asarray(all_cached_feats[audit_test_indices])

    # Convert ground-truth genres to multi-hot matrix
    y_test_true = np.array([
        [int(v) for v in row.split(",")]
        for row in audit_meta["target_multihot"]
    ], dtype=np.float32)

    # Access raw images from Hugging Face dataset for true visual corruptions
    logger.info("Loading raw images from MM-IMDb dataset cache...")
    hf_ds = load_dataset("pranavmr/MM-IMDb", split="train")
    raw_test_images: List[Image.Image] = []
    for g_idx in audit_test_indices:
        item = hf_ds[g_idx]
        img = item.get("image", None)
        if img is None:
            img = apply_missing_image()
        elif img.mode != "RGB":
            img = img.convert("RGB")
        raw_test_images.append(img)
    logger.info(f"Loaded {len(raw_test_images)} raw test images.")

    # Load frozen ResNet-50 feature extractor for corrupted image passes
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = nn.Identity()
    resnet.eval()
    device = torch.device(args.device)
    resnet.to(device)
    image_transform = get_default_image_transform()

    # Pre-extract single black image feature vector for missing image representation
    black_img = apply_missing_image()
    with torch.inference_mode():
        black_tensor = image_transform(black_img).unsqueeze(0).to(device)
        missing_image_single_feat = resnet(black_tensor).cpu().numpy()[0]  # (2048,)
    missing_image_features = np.tile(missing_image_single_feat, (audit_n, 1))

    # 5. Define Comprehensive Benchmark Conditions (Improvements #5 & #10)
    conditions: List[Dict[str, Any]] = [
        {
            "id": 0,
            "name": "Clean_Baseline",
            "modality_corrupted": "None",
            "expected_failure_type": "CLEAN",
            "is_corrupted": False,
            "text_type": "clean",
            "image_type": "clean"
        },
        {
            "id": 1,
            "name": "Text_Truncation_25%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "is_corrupted": True,
            "text_type": "truncation",
            "text_kwargs": {"keep_ratio": 0.25},
            "image_type": "clean"
        },
        {
            "id": 2,
            "name": "Text_Token_Masking_35%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "is_corrupted": True,
            "text_type": "masking",
            "text_kwargs": {"mask_rate": 0.35, "mask_token": "[MASK]"},
            "image_type": "clean"
        },
        {
            "id": 3,
            "name": "Text_Empty",
            "modality_corrupted": "Text",
            "expected_failure_type": "MISSING_TEXT",
            "is_corrupted": True,
            "text_type": "empty",
            "image_type": "clean"
        },
        {
            "id": 4,
            "name": "Image_Gaussian_Blur_r5",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "is_corrupted": True,
            "text_type": "clean",
            "image_type": "gaussian_blur",
            "image_kwargs": {"radius": 5.0}
        },
        {
            "id": 5,
            "name": "Image_Gaussian_Noise_s25",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "is_corrupted": True,
            "text_type": "clean",
            "image_type": "gaussian_noise",
            "image_kwargs": {"std": 0.25}
        },
        {
            "id": 6,
            "name": "Image_Missing_Blackout",
            "modality_corrupted": "Vision",
            "expected_failure_type": "MISSING_IMAGE",
            "is_corrupted": True,
            "text_type": "clean",
            "image_type": "missing_image"
        },
        {
            "id": 7,
            "name": "Dual_Corruption_Trunc_Noise",
            "modality_corrupted": "Both",
            "expected_failure_type": "HIGH_UNCERTAINTY",
            "is_corrupted": True,
            "text_type": "truncation",
            "text_kwargs": {"keep_ratio": 0.25},
            "image_type": "gaussian_noise",
            "image_kwargs": {"std": 0.25}
        },
        {
            "id": 8,
            "name": "Cross_Modal_Conflict_Mismatch",
            "modality_corrupted": "Cross_Modal",
            "expected_failure_type": "CROSS_MODAL_CONFLICT",
            "is_corrupted": True,
            "text_type": "conflict",
            "text_kwargs": {
                "conflict_text": "A hilarious romantic comedy featuring wacky high school antics and cheerful musical numbers."
            },
            "image_type": "clean"
        },
        {
            "id": 9,
            "name": "Text_Typo_Noise_15%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "is_corrupted": True,
            "text_type": "typo",
            "text_kwargs": {"typo_rate": 0.15},
            "image_type": "clean"
        },
        {
            "id": 10,
            "name": "Text_OCR_Noise_20%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "is_corrupted": True,
            "text_type": "ocr",
            "text_kwargs": {"ocr_rate": 0.20},
            "image_type": "clean"
        },
        {
            "id": 11,
            "name": "Image_JPEG_Compression_q10",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "is_corrupted": True,
            "text_type": "clean",
            "image_type": "jpeg_compression",
            "image_kwargs": {"quality": 10}
        },
        {
            "id": 12,
            "name": "Image_Contrast_Washout",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "is_corrupted": True,
            "text_type": "clean",
            "image_type": "contrast_reduction",
            "image_kwargs": {"factor": 0.20}
        }
    ]

    # Instantiate Adaptive Recovery Engine (Improvement #9)
    recovery_engine = DynamicModalityGating(default_alpha=mm_model.alpha, conflict_strategy="lower_entropy")

    # 6. Execute Benchmark across Conditions
    condition_results: List[Dict[str, Any]] = []
    all_sample_audit_records: List[pd.DataFrame] = []

    y_true_detection_all: List[int] = []
    y_pred_detection_all: List[int] = []
    y_true_types_all: List[str] = []
    y_pred_types_all: List[str] = []

    # Learned classifier evaluation trackers (Improvement #4)
    y_pred_learned_detection_all: List[int] = []
    y_pred_learned_types_all: List[str] = []

    logger.info(f"\nRunning Phase 4 Benchmark across {len(conditions)} Experimental Conditions...")

    # Caches for image features to avoid redundant ResNet passes across shared corruptions
    cached_corrupted_feats: Dict[str, np.ndarray] = {}

    for cond in conditions:
        c_name = cond["name"]
        logger.info("-" * 60)
        logger.info(f"Condition [{cond['id']}]: {c_name} (Modality: {cond['modality_corrupted']})")

        # --- A. Prepare Text Inputs ---
        t0_prep = time.perf_counter()
        if cond["text_type"] == "clean":
            cond_texts = list(test_clean_texts)
            has_text_flags = [bool(t.strip()) for t in cond_texts]
        else:
            text_kwargs = cast(Dict[str, Any], cond.get("text_kwargs", {}))
            cond_texts = corrupt_text_batch(test_clean_texts, corruption_type=cond["text_type"], base_seed=args.seed, **text_kwargs)
            has_text_flags = [False if cond["text_type"] == "empty" else bool(t.strip()) for t in cond_texts]

        # Improvement #3: Run automated missing modality detection on text
        auto_text_valid = MultimodalReliabilityAuditor.auto_detect_empty_text(cond_texts)
        if cond["text_type"] == "empty":
            has_text_flags = [bool(v) for v in auto_text_valid]

        # Improvement #1 & #8: Extract text feature norms and measure vectorization latency
        t0_text_extract = time.perf_counter()
        cond_text_sparse = text_model.vectorizer.transform(cond_texts)
        cond_text_norms = np.sqrt(np.asarray(cond_text_sparse.power(2).sum(axis=1)).ravel())
        time_text_extract_sec = time.perf_counter() - t0_text_extract

        # --- B. Prepare Visual Features ---
        raw_extraction_time = 0.0
        image_kwargs = cast(Dict[str, Any], cond.get("image_kwargs", {}))
        img_type = cond["image_type"]

        if img_type == "clean":
            cond_image_feats = test_cached_feats
            has_image_flags = [True] * audit_n
        elif img_type == "missing_image":
            cond_image_feats = missing_image_features
            has_image_flags = [False] * audit_n
        elif img_type in cached_corrupted_feats:
            logger.info(f"Reusing cached ResNet-50 features for {img_type}.")
            cond_image_feats = cached_corrupted_feats[img_type]
            has_image_flags = [True] * audit_n
        else:
            t0_extract = time.perf_counter()
            logger.info(f"Extracting ResNet-50 features for {img_type} images...")
            cond_image_feats = extract_corrupted_image_features(
                images=raw_test_images,
                corruption_type=img_type,
                feature_extractor_model=resnet,
                image_transform=image_transform,
                device=device,
                batch_size=32,
                base_seed=args.seed,
                **image_kwargs
            )
            raw_extraction_time = time.perf_counter() - t0_extract
            cached_corrupted_feats[img_type] = cond_image_feats
            has_image_flags = [True] * audit_n
            logger.info(f"Raw image feature extraction finished in {raw_extraction_time:.2f}s ({audit_n/max(0.1, raw_extraction_time):.1f} imgs/s).")

        # Improvement #1: Compute image feature norms
        cond_image_norms = np.linalg.norm(cond_image_feats, axis=1)

        # --- C. Component-Level Latency Benchmarks (Improvement #8) ---
        t0_text_inf = time.perf_counter()
        t_probs = text_model.classifier.predict_proba(cond_text_sparse)
        time_text_inf_sec = time.perf_counter() - t0_text_inf

        t0_image_inf = time.perf_counter()
        i_probs = image_model.predict_proba(cond_image_feats)
        time_image_inf_sec = time.perf_counter() - t0_image_inf

        t0_fusion_inf = time.perf_counter()
        mm_probs = mm_model.predict_proba(t_probs, i_probs)
        time_fusion_inf_sec = time.perf_counter() - t0_fusion_inf

        inference_latency_sec = time_text_inf_sec + time_image_inf_sec + time_fusion_inf_sec
        throughput = audit_n / max(0.001, inference_latency_sec)

        # Multi-label evaluation (Before Recovery)
        eval_t = evaluate_multilabel_predictions(y_test_true, t_probs, thresholds=text_model.thresholds)
        eval_i = evaluate_multilabel_predictions(y_test_true, i_probs, thresholds=image_model.thresholds)
        eval_mm = evaluate_multilabel_predictions(y_test_true, mm_probs, thresholds=mm_model.thresholds)

        # --- D. Auditing & Failure Diagnosis (Improvements #1, #2, #3) ---
        t0_audit = time.perf_counter()
        audit_res = auditor.audit_batch(
            text_probs=t_probs,
            image_probs=i_probs,
            fused_probs=mm_probs,
            has_text=has_text_flags,
            has_image=has_image_flags,
            movie_ids=test_movie_ids,
            text_feature_norms=cond_text_norms,
            image_feature_norms=cond_image_norms
        )
        time_audit_sec = time.perf_counter() - t0_audit

        # Tag condition metadata on audit DataFrame
        df_cond = audit_res["audit_df"].copy()
        df_cond["condition_id"] = cond["id"]
        df_cond["condition_name"] = c_name
        df_cond["ground_truth_failure_type"] = cond["expected_failure_type"]
        df_cond["is_actually_corrupted"] = cond["is_corrupted"]
        all_sample_audit_records.append(df_cond)

        # --- E. Learned Diagnostic Classifier Inference (Improvement #4) ---
        learned_pred = learned_classifier.predict(df_cond)
        y_pred_learned_detection_all.extend([1 if x else 0 for x in learned_pred["is_anomalous"]])
        y_pred_learned_types_all.extend(learned_pred["diagnoses"])

        # --- F. Adaptive Recovery Engine Evaluation (Improvement #9) ---
        recovery_output = recovery_engine.recover_predictions(
            text_probs=t_probs,
            image_probs=i_probs,
            diagnoses=audit_res["diagnoses"],
            text_uncertainties=audit_res["text_uncertainty"]["composite_uncertainty"],
            image_uncertainties=audit_res["image_uncertainty"]["composite_uncertainty"],
            default_fused_probs=mm_probs
        )
        rec_gain_dict = recovery_engine.evaluate_recovery_gain(
            unrecovered_probs=mm_probs,
            recovered_probs=recovery_output["recovered_probs"],
            y_true=y_test_true,
            thresholds=mm_model.thresholds,
            genre_names=GENRES
        )

        # Accumulate detection targets
        y_true_bin = [1 if cond["is_corrupted"] else 0] * audit_n
        y_pred_bin = [1 if x else 0 for x in audit_res["is_anomalous"]]
        y_true_detection_all.extend(y_true_bin)
        y_pred_detection_all.extend(y_pred_bin)

        y_true_types_all.extend([cond["expected_failure_type"]] * audit_n)
        y_pred_types_all.extend(audit_res["diagnoses"])

        # Condition summary metrics
        cond_summary = {
            "condition_id": cond["id"],
            "condition_name": c_name,
            "modality_corrupted": cond["modality_corrupted"],
            "expected_failure_type": cond["expected_failure_type"],
            "is_corrupted": cond["is_corrupted"],
            "sample_count": audit_n,
            "multimodal_micro_f1": eval_mm["micro_f1"],
            "multimodal_macro_f1": eval_mm["macro_f1"],
            "multimodal_hamming_loss": eval_mm["hamming_loss"],
            "multimodal_map": eval_mm["mean_average_precision"],
            "text_micro_f1": eval_t["micro_f1"],
            "image_micro_f1": eval_i["micro_f1"],
            "mean_bernoulli_jsd": round(float(np.mean(audit_res["disagreement"]["mean_bernoulli_jsd"])), 4),
            "mean_jaccard_distance": round(float(np.mean(audit_res["disagreement"]["jaccard_distance"])), 4),
            "mean_text_entropy": round(float(np.mean(audit_res["text_uncertainty"]["mean_binary_entropy"])), 4),
            "mean_image_entropy": round(float(np.mean(audit_res["image_uncertainty"]["mean_binary_entropy"])), 4),
            "mean_text_norm": round(float(np.mean(cond_text_norms)), 4),
            "mean_image_norm": round(float(np.mean(cond_image_norms)), 4),
            "anomaly_flag_rate": round(float(np.mean(audit_res["is_anomalous"])), 4),
            "mean_anomaly_score": round(float(np.mean(audit_res["anomaly_scores"])), 4),
            # Recovery metrics (Improvement #9)
            "recovered_micro_f1": rec_gain_dict["recovered"]["micro_f1"],
            "recovery_micro_f1_gain": rec_gain_dict["micro_f1_gain"],
            # Component-Level Timings (Improvement #8)
            "time_text_extract_sec": round(time_text_extract_sec, 4),
            "time_text_inf_sec": round(time_text_inf_sec, 4),
            "time_image_inf_sec": round(time_image_inf_sec, 4),
            "time_fusion_inf_sec": round(time_fusion_inf_sec, 4),
            "inference_latency_sec": round(inference_latency_sec, 4),
            "cached_classifier_throughput": round(throughput, 1),
            "raw_feature_extraction_sec": round(raw_extraction_time, 2),
            "audit_processing_sec": round(time_audit_sec, 4)
        }
        condition_results.append(cond_summary)

        logger.info(
            f"Results: MM Micro-F1={eval_mm['micro_f1']:.4f} -> Recovered F1={rec_gain_dict['recovered']['micro_f1']:.4f} "
            f"(Delta: {rec_gain_dict['micro_f1_gain']:+.4f}) | "
            f"JSD={cond_summary['mean_bernoulli_jsd']:.4f} | "
            f"AnomalyRate={cond_summary['anomaly_flag_rate']*100:.1f}% | "
            f"AuditTime={time_audit_sec:.3f}s"
        )

    # 7. Evaluate Diagnostic Quality (Rule-Based vs Learned)
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC PERFORMANCE EVALUATION (RULE-BASED AUDITOR)")
    logger.info("=" * 80)

    diag_eval = evaluate_diagnosis_performance(
        y_true_binary=np.array(y_true_detection_all),
        y_pred_binary=np.array(y_pred_detection_all),
        y_true_types=y_true_types_all,
        y_pred_types=y_pred_types_all
    )

    det_metrics = diag_eval["binary_corruption_detection"]
    logger.info("Binary Corruption Detection Metrics:")
    logger.info(f"  Accuracy:    {det_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {det_metrics['precision']:.4f}")
    logger.info(f"  Recall:      {det_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:    {det_metrics['f1_score']:.4f}")
    logger.info(f"  Specificity: {det_metrics['specificity']:.4f}")

    cls_metrics = diag_eval["multiclass_failure_classification"]
    logger.info("\nMulti-Class Failure Diagnosis Metrics:")
    logger.info(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
    logger.info(f"  Macro-F1: {cls_metrics['macro_f1']:.4f}")
    logger.info(f"  Micro-F1: {cls_metrics['micro_f1']:.4f}")
    for lbl, stats in cls_metrics["per_class"].items():
        logger.info(f"  Class [{lbl}]: F1={stats['f1']:.4f} | Prec={stats['precision']:.4f} | Rec={stats['recall']:.4f} (Support={stats['support']})")

    # Evaluate Learned Diagnostic Classifier (Improvement #4)
    logger.info("\n" + "=" * 80)
    logger.info("LEARNED DIAGNOSTIC CLASSIFIER EVALUATION (GRADIENT BOOSTING)")
    logger.info("=" * 80)

    learned_diag_eval = evaluate_diagnosis_performance(
        y_true_binary=np.array(y_true_detection_all),
        y_pred_binary=np.array(y_pred_learned_detection_all),
        y_true_types=y_true_types_all,
        y_pred_types=y_pred_learned_types_all
    )
    learned_det = learned_diag_eval["binary_corruption_detection"]
    learned_cls = learned_diag_eval["multiclass_failure_classification"]
    logger.info(f"Learned Detection F1:  {learned_det['f1_score']:.4f} | Accuracy: {learned_det['accuracy']:.4f}")
    logger.info(f"Learned Diagnosis Macro-F1: {learned_cls['macro_f1']:.4f} | Accuracy: {learned_cls['accuracy']:.4f}")

    # 8. Save Machine-Readable CSV & JSON Artifacts
    logger.info("\nSaving evaluation artifacts to disk...")

    # CSV 1: Comparison table across conditions
    cond_df = pd.DataFrame(condition_results)
    cond_csv_path = output_dir / "clean_vs_corrupted_metrics.csv"
    cond_df.to_csv(cond_csv_path, index=False)
    logger.info(f"Saved condition metrics table: {cond_csv_path}")

    # CSV 2: Full sample-level predictions and audit signals
    full_audit_df = pd.concat(all_sample_audit_records, ignore_index=True)
    audit_csv_path = output_dir / "audit_predictions_diagnoses.csv"
    full_audit_df.to_csv(audit_csv_path, index=False)
    logger.info(f"Saved sample-level diagnoses ({len(full_audit_df)} records): {audit_csv_path}")

    # JSON 1: Diagnostic confusion matrices
    conf_json_path = output_dir / "diagnostic_confusion_matrix.json"
    with open(conf_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "rule_based_diagnosis": diag_eval,
            "learned_diagnosis": learned_diag_eval
        }, f, indent=2)
    logger.info(f"Saved diagnostic confusion matrices: {conf_json_path}")

    # JSON 2: Consolidated Phase 4 summary
    summary_data = {
        "benchmark_name": "SEAA_Phase4_Auditing_Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_sample_count": audit_n,
        "seed": args.seed,
        "device": args.device,
        "calibrated_config": diag_config.to_dict(),
        "binary_corruption_detection": det_metrics,
        "multiclass_failure_classification": cls_metrics,
        "learned_classifier_metrics": {
            "binary_detection": learned_det,
            "multiclass_classification": learned_cls
        },
        "condition_summaries": condition_results
    }
    summary_json_path = output_dir / "phase4_audit_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved consolidated audit summary: {summary_json_path}")

    # 9. Generate Figures
    logger.info("Generating publication-quality comparison charts...")

    # Figure 1: Performance Degradation & Recovery
    cond_names = [c["condition_name"] for c in condition_results]
    f1_unrec = [c["multimodal_micro_f1"] for c in condition_results]
    f1_rec = [c["recovered_micro_f1"] for c in condition_results]

    x = np.arange(len(cond_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    bars1 = ax.bar(x - width/2, f1_unrec, width, label="Multimodal (Unrecovered)", color="#c25953", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_rec, width, label="SEAA Adaptive Recovery (Phase 5)", color="#2b8f5c", alpha=0.90)
    ax.set_ylabel("Micro-F1 Score", fontsize=12)
    ax.set_ylim(0, 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=35, ha="right", fontsize=9)
    ax.set_title("SEAA Reliability Benchmark: Performance Degradation vs Adaptive Recovery Gain", fontsize=14, pad=15)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig1_path = figures_dir / "corruption_performance_degradation.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved performance degradation & recovery figure: {fig1_path}")

    # Figure 2: Disagreement Distribution
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=full_audit_df,
        x="condition_name",
        y="mean_bernoulli_jsd",
        hue="condition_name",
        palette="viridis",
        legend=False,
        ax=ax
    )
    ax.axhline(
        y=diag_config.jsd_threshold,
        color="red",
        linestyle="--",
        label=f"Calibrated JSD Threshold ({diag_config.jsd_threshold:.4f})"
    )
    ax.set_title("Cross-Modal Disagreement (Mean Bernoulli JSD) by Modality Corruption Condition", fontsize=13)
    ax.set_xlabel("Experimental Condition", fontsize=11)
    ax.set_ylabel("Mean Bernoulli JSD [0, 1]", fontsize=11)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig2_path = figures_dir / "disagreement_distribution_by_condition.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved disagreement distribution figure: {fig2_path}")

    # Figure 3: Diagnostic Confusion Matrix Heatmap
    cm_data = np.array(cls_metrics["confusion_matrix"])
    labels = cls_metrics["labels"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title("Multi-Class Failure Diagnosis Confusion Matrix (Rule-Based)", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11)
    ax.set_ylabel("Ground Truth Failure Type", fontsize=11)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig3_path = figures_dir / "diagnostic_confusion_matrix.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure: {fig3_path}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

