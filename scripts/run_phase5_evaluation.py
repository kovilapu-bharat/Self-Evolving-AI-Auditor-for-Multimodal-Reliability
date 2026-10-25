"""
SEAA Phase 5: Adaptive Recovery Policy Evaluation & Benchmarking Runner.

Executes controlled evaluation comparing:
1. No Recovery (Static Late Fusion, alpha=0.55)
2. Fixed Recovery Policy (DynamicModalityGating heuristic router)
3. Adaptive Recovery Policy (Trained Contextual Bandit, frozen for evaluation)
4. Oracle Upper Bound (Theoretical optimal selection per sample)

Adheres strictly to Phase 5 requirements:
- Evaluates on identical sample IDs across all policies and conditions.
- Zero test label leakage: policy trained strictly on validation split and frozen for test evaluation.
- Preserves Phase 1-3 artifacts and writes all outputs to a dedicated run directory.
- Accurately reports Micro-F1, Macro-F1, Hamming Loss, mAP, per-condition deltas, action distributions, and latency.
- Produces publication-quality figures and machine-readable JSON/CSV summaries.
"""

import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
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

from src.data.dataset import GENRES, get_default_image_transform
from src.evaluation.metrics import evaluate_multilabel_predictions
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel
from src.failures.text_corruptions import corrupt_text_batch
from src.failures.image_corruptions import (
    extract_corrupted_image_features,
    apply_missing_image
)
from src.auditing.diagnosis import (
    calibrate_diagnostic_thresholds_on_val,
    MultimodalReliabilityAuditor
)
from src.auditing.learned_diagnostics import (
    LearnedDiagnosticClassifier,
    extract_diagnostic_feature_matrix
)
from src.recovery.modality_gating import DynamicModalityGating
from src.policy.adaptive_policy import (
    AdaptiveRecoveryPolicy,
    PolicyContext,
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_IMAGE_ONLY,
    ACTION_RELIABILITY_WEIGHTED,
    ACTION_CONSERVATIVE_FUSION,
    ALL_ACTIONS
)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Phase5Evaluation")
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
        description="SEAA Phase 5: Adaptive Recovery Policy Evaluation Runner."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of test samples to evaluate across all policies (default: 500, matching Phase 4 benchmark)."
    )
    parser.add_argument(
        "--val_train_size",
        type=int,
        default=500,
        help="Number of validation samples used to train the contextual bandit policy (default: 500)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output run directory (defaults to outputs/phase5/run_<timestamp>)."
    )
    return parser.parse_args()


def train_policy_on_validation(
    val_indices: List[int],
    meta_df: pd.DataFrame,
    all_cached_feats: np.ndarray,
    text_model: TextBaselineModel,
    image_model: ImageBaselineModel,
    auditor: MultimodalReliabilityAuditor,
    learned_classifier: Optional[LearnedDiagnosticClassifier],
    thresholds: np.ndarray,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[AdaptiveRecoveryPolicy, List[Dict[str, Any]]]:
    """
    Trains the Adaptive Recovery Policy strictly on validation split data.
    Simulates failure regimes to allow the bandit to learn state-action values.
    """
    if logger:
        logger.info(f"Training Adaptive Recovery Policy on {len(val_indices)} validation samples...")

    policy = AdaptiveRecoveryPolicy(
        learning_rate=0.20,
        discount_or_decay=0.5,
        epsilon=0.15,
        ucb_c=0.5,
        default_alpha=0.55,
        seed=seed,
        learning_enabled=True
    )

    val_meta = meta_df.iloc[val_indices].reset_index(drop=True)
    val_texts = [str(t) if pd.notnull(t) else "" for t in val_meta["text"]]
    val_feats = np.asarray(all_cached_feats[val_indices])
    y_val = np.array([
        [int(v) for v in row.split(",")]
        for row in val_meta["target_multihot"]
    ], dtype=np.float32)

    val_n = len(val_indices)

    # Precompute clean validation predictions
    val_t_probs = text_model.predict_proba(val_texts)
    val_i_probs = image_model.predict_proba(val_feats)
    val_mm_probs = 0.55 * val_t_probs + 0.45 * val_i_probs

    t_sparse = text_model.vectorizer.transform(val_texts)
    val_t_norms = np.sqrt(np.asarray(t_sparse.power(2).sum(axis=1)).ravel())
    val_i_norms = np.linalg.norm(val_feats, axis=1)

    # 1. Clean validation episodes
    audit_clean = auditor.audit_batch(
        text_probs=val_t_probs,
        image_probs=val_i_probs,
        fused_probs=val_mm_probs,
        text_feature_norms=val_t_norms,
        image_feature_norms=val_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            text_entropy=float(audit_clean["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_clean["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_clean["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_clean["disagreement"]["hamming_distance"][i]),
            text_norm=float(val_t_norms[i]),
            image_norm=float(val_i_norms[i]),
            diagnosis="CLEAN"
        )
        act = policy.select_action(ctx, explore=True)
        rec_probs, _ = policy.apply_action(act, val_t_probs[i], val_i_probs[i], ctx)
        reward = policy.compute_reward(rec_probs, val_mm_probs[i], y_val[i], thresholds)
        policy.update(ctx, act, reward)

    # 2. Text failure episodes on validation (text truncation & token masking)
    corrupted_t_texts = corrupt_text_batch(val_texts, corruption_type="truncation", keep_ratio=0.25, base_seed=seed)
    corrupted_t_probs = text_model.predict_proba(corrupted_t_texts)
    corrupted_t_sparse = text_model.vectorizer.transform(corrupted_t_texts)
    corrupted_t_norms = np.sqrt(np.asarray(corrupted_t_sparse.power(2).sum(axis=1)).ravel())
    unrec_t_fail = 0.55 * corrupted_t_probs + 0.45 * val_i_probs

    audit_tfail = auditor.audit_batch(
        text_probs=corrupted_t_probs,
        image_probs=val_i_probs,
        fused_probs=unrec_t_fail,
        text_feature_norms=corrupted_t_norms,
        image_feature_norms=val_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            text_entropy=float(audit_tfail["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_tfail["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_tfail["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_tfail["disagreement"]["hamming_distance"][i]),
            text_norm=float(corrupted_t_norms[i]),
            image_norm=float(val_i_norms[i]),
            diagnosis="TEXT_FAILURE"
        )
        act = policy.select_action(ctx, explore=True)
        rec_probs, _ = policy.apply_action(act, corrupted_t_probs[i], val_i_probs[i], ctx)
        reward = policy.compute_reward(rec_probs, unrec_t_fail[i], y_val[i], thresholds)
        policy.update(ctx, act, reward)

    # 3. Missing text episodes on validation
    empty_t_probs = np.full_like(val_t_probs, 0.05)
    unrec_empty_t = 0.55 * empty_t_probs + 0.45 * val_i_probs
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=False,
            has_image=True,
            text_entropy=0.5,
            image_entropy=float(audit_clean["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=0.20,
            text_norm=0.0,
            image_norm=float(val_i_norms[i]),
            diagnosis="MISSING_TEXT"
        )
        act = policy.select_action(ctx, explore=True)
        rec_probs, _ = policy.apply_action(act, empty_t_probs[i], val_i_probs[i], ctx)
        reward = policy.compute_reward(rec_probs, unrec_empty_t[i], y_val[i], thresholds)
        policy.update(ctx, act, reward)

    # 4. Image failure episodes on validation (noise / blur feature perturbation)
    rng_val = np.random.RandomState(seed + 1)
    noisy_val_feats = val_feats + rng_val.normal(0, 0.5, size=val_feats.shape).astype(np.float32)
    noisy_i_probs = image_model.predict_proba(noisy_val_feats)
    unrec_ifail = 0.55 * val_t_probs + 0.45 * noisy_i_probs

    audit_ifail = auditor.audit_batch(
        text_probs=val_t_probs,
        image_probs=noisy_i_probs,
        fused_probs=unrec_ifail,
        text_feature_norms=val_t_norms,
        image_feature_norms=np.linalg.norm(noisy_val_feats, axis=1)
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            text_entropy=float(audit_ifail["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_ifail["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_ifail["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_ifail["disagreement"]["hamming_distance"][i]),
            text_norm=float(val_t_norms[i]),
            image_norm=float(np.linalg.norm(noisy_val_feats[i])),
            diagnosis="IMAGE_FAILURE"
        )
        act = policy.select_action(ctx, explore=True)
        rec_probs, _ = policy.apply_action(act, val_t_probs[i], noisy_i_probs[i], ctx)
        reward = policy.compute_reward(rec_probs, unrec_ifail[i], y_val[i], thresholds)
        policy.update(ctx, act, reward)

    # 5. Missing image episodes on validation
    black_i_probs = np.full_like(val_i_probs, 0.10)
    unrec_empty_i = 0.55 * val_t_probs + 0.45 * black_i_probs
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True,
            has_image=False,
            text_entropy=float(audit_clean["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=0.5,
            bernoulli_jsd=0.20,
            text_norm=float(val_t_norms[i]),
            image_norm=0.0,
            diagnosis="MISSING_IMAGE"
        )
        act = policy.select_action(ctx, explore=True)
        rec_probs, _ = policy.apply_action(act, val_t_probs[i], black_i_probs[i], ctx)
        reward = policy.compute_reward(rec_probs, unrec_empty_i[i], y_val[i], thresholds)
        policy.update(ctx, act, reward)

    # Freeze policy for evaluation
    policy.learning_enabled = False
    policy.epsilon = 0.0

    if logger:
        logger.info(
            f"Policy training complete: {len(policy.update_history)} updates across "
            f"{len(policy.q_table)} distinct context regimes."
        )

    return policy, policy.update_history


def compute_oracle_action(
    t_prob: np.ndarray,
    i_prob: np.ndarray,
    y_true_row: np.ndarray,
    thresholds: np.ndarray,
    ctx: PolicyContext,
    policy: AdaptiveRecoveryPolicy
) -> Tuple[str, np.ndarray]:
    """
    Evaluates all admissible actions against ground truth and chooses the best.
    Theoretical upper bound for benchmarking only.
    """
    valid_actions = ctx.get_valid_actions()
    best_act = valid_actions[0]
    best_f1 = -1.0
    best_probs = t_prob

    for a in valid_actions:
        rec, _ = policy.apply_action(a, t_prob, i_prob, ctx)
        pred = (rec >= thresholds).astype(int)
        tp = np.sum((pred == 1) & (y_true_row == 1))
        fp = np.sum((pred == 1) & (y_true_row == 0))
        fn = np.sum((pred == 0) & (y_true_row == 1))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 1.0
        if f1 > best_f1:
            best_f1 = f1
            best_act = a
            best_probs = rec

    return best_act, best_probs


def main():
    args = parse_args()

    # 1. Initialize Run Directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = project_root / "outputs" / "phase5" / f"run_{timestamp_str}_adaptive_recovery"

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "phase5_evaluation.log")
    logger.info("=" * 70)
    logger.info("SEAA PHASE 5: ADAPTIVE RECOVERY POLICY BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Run Output Directory: {run_dir}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Test Sample Size: {args.sample_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # 2. Load Checkpoints & Calibration Data
    ckpt_dir = project_root / "outputs" / "checkpoints"
    text_model_path = ckpt_dir / "text_baseline.joblib"
    image_model_path = ckpt_dir / "image_baseline.joblib"
    mm_config_path = ckpt_dir / "multimodal_baseline.json"

    logger.info(f"Loading Text Baseline from {text_model_path}")
    text_model = TextBaselineModel.load(text_model_path)
    logger.info(f"Loading Image Baseline from {image_model_path}")
    image_model = ImageBaselineModel.load(image_model_path)
    logger.info(f"Loading Multimodal Config from {mm_config_path}")
    mm_model = MultimodalBaselineModel.load(mm_config_path)

    thresholds = mm_model.thresholds
    label_mapping_path = project_root / "data" / "label_mapping.json"
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    idx_to_genre = {v: k for k, v in label_data["genre_to_idx"].items()}

    split_path = project_root / "data" / "split_indices.json"
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    meta_csv_path = project_root / "data" / "metadata.csv"
    meta_df = pd.read_csv(meta_csv_path)

    features_path = project_root / "data" / "processed" / "resnet50_features.npy"
    all_cached_feats = np.memmap(features_path, dtype=np.float32, mode="r", shape=(len(meta_df), 2048))

    # 3. Setup Auditor & Calibrate on Validation Split
    val_indices = splits["val"]
    val_meta = meta_df.iloc[val_indices].reset_index(drop=True)
    val_texts = [str(t) if pd.notnull(t) else "" for t in val_meta["text"]]
    val_cached_feats = np.asarray(all_cached_feats[val_indices])

    val_t_probs = text_model.predict_proba(val_texts)
    val_i_probs = image_model.predict_proba(val_cached_feats)

    val_t_sparse = text_model.vectorizer.transform(val_texts)
    val_t_norms = np.sqrt(np.asarray(val_t_sparse.power(2).sum(axis=1)).ravel())
    val_i_norms = np.linalg.norm(val_cached_feats, axis=1)

    logger.info("Calibrating diagnostic thresholds on validation split...")
    diag_config = calibrate_diagnostic_thresholds_on_val(
        val_text_probs=val_t_probs,
        val_image_probs=val_i_probs,
        text_thresholds=text_model.thresholds,
        image_thresholds=image_model.thresholds,
        percentile=95.0,
        val_text_norms=val_t_norms,
        val_image_norms=val_i_norms
    )
    auditor = MultimodalReliabilityAuditor(
        config=diag_config,
        text_thresholds=text_model.thresholds,
        image_thresholds=image_model.thresholds,
        fused_thresholds=mm_model.thresholds,
        genre_names=GENRES
    )

    # Load learned diagnostic classifier if available
    learned_ckpt = ckpt_dir / "learned_diagnostic_classifier.pkl"
    learned_classifier = None
    if learned_ckpt.exists():
        try:
            learned_classifier = LearnedDiagnosticClassifier.load(learned_ckpt)
            logger.info("Loaded Learned Diagnostic Classifier checkpoint.")
        except Exception as e:
            logger.warning(f"Could not load learned diagnostic classifier: {e}")

    # 4. Train Adaptive Recovery Policy on Validation Failures
    val_train_indices = val_indices[:args.val_train_size]
    trained_policy, train_history = train_policy_on_validation(
        val_indices=val_train_indices,
        meta_df=meta_df,
        all_cached_feats=all_cached_feats,
        text_model=text_model,
        image_model=image_model,
        auditor=auditor,
        learned_classifier=learned_classifier,
        thresholds=thresholds,
        seed=args.seed,
        logger=logger
    )
    policy_ckpt_path = run_dir / "trained_adaptive_policy.json"
    trained_policy.save_policy(policy_ckpt_path)

    history_path = run_dir / "policy_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)

    # 5. Instantiate Fixed Policy Baseline (DynamicModalityGating)
    fixed_engine = DynamicModalityGating(default_alpha=mm_model.alpha, conflict_strategy="lower_entropy")

    # 6. Prepare Identical Test Subset
    test_indices = splits["test"]
    audit_n = min(args.sample_size, len(test_indices))
    audit_test_indices = test_indices[:audit_n]
    audit_meta = meta_df.iloc[audit_test_indices].reset_index(drop=True)

    test_movie_ids = audit_meta["movie_id"].tolist()
    # Save exact sample IDs for reproducibility
    sample_ids_file = run_dir / "evaluated_sample_ids.txt"
    with open(sample_ids_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_movie_ids))

    test_clean_texts = [str(t) if pd.notnull(t) else "" for t in audit_meta["text"]]
    test_clean_feats = np.asarray(all_cached_feats[audit_test_indices])
    y_test_true = np.array([
        [int(v) for v in row.split(",")]
        for row in audit_meta["target_multihot"]
    ], dtype=np.float32)

    logger.info(f"Loaded {audit_n} identical test samples with movie IDs saved to {sample_ids_file}")

    # Load raw images for visual corruptions
    logger.info("Loading raw images from Hugging Face dataset cache...")
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

    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet.to(device)
    resnet.eval()
    image_transform = get_default_image_transform()

    black_img = apply_missing_image()
    with torch.inference_mode():
        black_tensor = image_transform(black_img).unsqueeze(0).to(device)
        missing_image_single_feat = resnet(black_tensor).cpu().numpy()[0]
    missing_image_features = np.tile(missing_image_single_feat, (audit_n, 1))

    # 7. Benchmark Conditions (Matching Phase 4 Canonical Setup)
    conditions = [
        {
            "id": 0,
            "name": "Clean_Baseline",
            "modality_corrupted": "None",
            "expected_failure_type": "CLEAN",
            "text_type": "clean",
            "image_type": "clean"
        },
        {
            "id": 1,
            "name": "Text_Truncation_25%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "text_type": "truncation",
            "text_kwargs": {"keep_ratio": 0.25},
            "image_type": "clean"
        },
        {
            "id": 2,
            "name": "Text_Token_Masking_35%",
            "modality_corrupted": "Text",
            "expected_failure_type": "TEXT_FAILURE",
            "text_type": "masking",
            "text_kwargs": {"mask_rate": 0.35, "mask_token": "[MASK]"},
            "image_type": "clean"
        },
        {
            "id": 3,
            "name": "Text_Empty",
            "modality_corrupted": "Text",
            "expected_failure_type": "MISSING_TEXT",
            "text_type": "empty",
            "image_type": "clean"
        },
        {
            "id": 4,
            "name": "Image_Gaussian_Blur_r5",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "text_type": "clean",
            "image_type": "gaussian_blur",
            "image_kwargs": {"radius": 5.0}
        },
        {
            "id": 5,
            "name": "Image_Gaussian_Noise_s25",
            "modality_corrupted": "Vision",
            "expected_failure_type": "IMAGE_FAILURE",
            "text_type": "clean",
            "image_type": "gaussian_noise",
            "image_kwargs": {"std": 0.25}
        },
        {
            "id": 6,
            "name": "Image_Missing_Blackout",
            "modality_corrupted": "Vision",
            "expected_failure_type": "MISSING_IMAGE",
            "text_type": "clean",
            "image_type": "missing_image"
        },
        {
            "id": 7,
            "name": "Dual_Corruption_Trunc_Noise",
            "modality_corrupted": "Both",
            "expected_failure_type": "HIGH_UNCERTAINTY",
            "text_type": "truncation",
            "text_kwargs": {"keep_ratio": 0.25},
            "image_type": "gaussian_noise",
            "image_kwargs": {"std": 0.25}
        }
    ]

    # Shared cache across conditions
    cached_corrupted_feats: Dict[str, np.ndarray] = {}

    # Tracking records
    sample_records: List[Dict[str, Any]] = []
    condition_summary_rows: List[Dict[str, Any]] = []
    action_distributions: Dict[str, Dict[str, int]] = {}

    logger.info(f"\nBenchmarking 4 Policies across {len(conditions)} Experimental Conditions...")

    for cond in conditions:
        cond_id = cond["id"]
        c_name = cond["name"]
        logger.info("-" * 65)
        logger.info(f"Evaluating Condition [{cond_id}]: {c_name} (Corrupted: {cond['modality_corrupted']})")

        # --- A. Text Inputs ---
        if cond["text_type"] == "clean":
            cond_texts = list(test_clean_texts)
            has_text_flags = [bool(t.strip()) for t in cond_texts]
        else:
            text_kwargs = cast(Dict[str, Any], cond.get("text_kwargs", {}))
            cond_texts = corrupt_text_batch(test_clean_texts, corruption_type=cond["text_type"], base_seed=args.seed, **text_kwargs)
            has_text_flags = [False if cond["text_type"] == "empty" else bool(t.strip()) for t in cond_texts]

        # Auto empty text detection
        auto_text_valid = MultimodalReliabilityAuditor.auto_detect_empty_text(cond_texts)
        if cond["text_type"] == "empty":
            has_text_flags = [bool(v) for v in auto_text_valid]

        cond_text_sparse = text_model.vectorizer.transform(cond_texts)
        cond_text_norms = np.sqrt(np.asarray(cond_text_sparse.power(2).sum(axis=1)).ravel())

        # --- B. Image Inputs ---
        img_type = cond["image_type"]
        image_kwargs = cast(Dict[str, Any], cond.get("image_kwargs", {}))

        if img_type == "clean":
            cond_image_feats = test_clean_feats
            has_image_flags = [True] * audit_n
        elif img_type == "missing_image":
            cond_image_feats = missing_image_features
            has_image_flags = [False] * audit_n
        elif img_type in cached_corrupted_feats:
            cond_image_feats = cached_corrupted_feats[img_type]
            has_image_flags = [True] * audit_n
        else:
            logger.info(f"Extracting ResNet-50 features for visual corruption: {img_type}...")
            t0_img = time.perf_counter()
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
            cached_corrupted_feats[img_type] = cond_image_feats
            has_image_flags = [True] * audit_n
            logger.info(f"Extraction complete in {time.perf_counter() - t0_img:.2f}s.")

        cond_image_norms = np.linalg.norm(cond_image_feats, axis=1)

        # --- C. Model Forward Passes ---
        t_probs = text_model.predict_proba(cond_texts)
        i_probs = image_model.predict_proba(cond_image_feats)
        mm_unrecovered = mm_model.alpha * t_probs + (1.0 - mm_model.alpha) * i_probs

        # --- D. Multimodal Reliability Audit ---
        t0_audit = time.perf_counter()
        audit_res = auditor.audit_batch(
            text_probs=t_probs,
            image_probs=i_probs,
            fused_probs=mm_unrecovered,
            has_text=has_text_flags,
            has_image=has_image_flags,
            movie_ids=test_movie_ids,
            text_feature_norms=cond_text_norms,
            image_feature_norms=cond_image_norms
        )
        audit_time_sec = time.perf_counter() - t0_audit

        diagnoses = audit_res["diagnoses"]
        t_entropies = audit_res["text_uncertainty"]["mean_binary_entropy"]
        i_entropies = audit_res["image_uncertainty"]["mean_binary_entropy"]
        jsds = audit_res["disagreement"]["mean_bernoulli_jsd"]
        hammings = audit_res["disagreement"]["hamming_distance"]

        # --- E. Evaluate Policy 1: No Recovery ---
        p1_probs = mm_unrecovered

        # --- F. Evaluate Policy 2: Fixed Recovery Policy ---
        p2_output = fixed_engine.recover_predictions(
            text_probs=t_probs,
            image_probs=i_probs,
            diagnoses=diagnoses,
            text_uncertainties=audit_res["text_uncertainty"]["composite_uncertainty"],
            image_uncertainties=audit_res["image_uncertainty"]["composite_uncertainty"],
            default_fused_probs=mm_unrecovered
        )
        p2_probs = p2_output["recovered_probs"]
        p2_actions = p2_output["routing_actions"]

        # --- G. Evaluate Policy 3: Adaptive Recovery Policy ---
        p3_probs = np.zeros_like(t_probs)
        p3_actions = []
        p3_alphas = []
        t0_p3 = time.perf_counter()

        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]),
                has_image=bool(has_image_flags[i]),
                text_entropy=float(t_entropies[i]),
                image_entropy=float(i_entropies[i]),
                bernoulli_jsd=float(jsds[i]),
                hamming_distance=float(hammings[i]),
                text_norm=float(cond_text_norms[i]),
                image_norm=float(cond_image_norms[i]),
                diagnosis=str(diagnoses[i])
            )
            act = trained_policy.select_action(ctx, explore=False)
            rec, alpha = trained_policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            p3_probs[i] = rec
            p3_actions.append(act)
            p3_alphas.append(alpha)

        p3_latency_ms_per_sample = ((time.perf_counter() - t0_p3) / audit_n) * 1000.0

        # --- H. Evaluate Policy 4: Oracle Upper Bound ---
        p4_probs = np.zeros_like(t_probs)
        p4_actions = []
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]),
                has_image=bool(has_image_flags[i]),
                text_entropy=float(t_entropies[i]),
                image_entropy=float(i_entropies[i]),
                bernoulli_jsd=float(jsds[i]),
                hamming_distance=float(hammings[i]),
                text_norm=float(cond_text_norms[i]),
                image_norm=float(cond_image_norms[i]),
                diagnosis=str(diagnoses[i])
            )
            best_act, best_rec = compute_oracle_action(
                t_probs[i], i_probs[i], y_test_true[i], thresholds, ctx, trained_policy
            )
            p4_probs[i] = best_rec
            p4_actions.append(best_act)

        # --- I. Compute Performance Metrics ---
        eval_p1 = evaluate_multilabel_predictions(y_test_true, p1_probs, thresholds, GENRES)
        eval_p2 = evaluate_multilabel_predictions(y_test_true, p2_probs, thresholds, GENRES)
        eval_p3 = evaluate_multilabel_predictions(y_test_true, p3_probs, thresholds, GENRES)
        eval_p4 = evaluate_multilabel_predictions(y_test_true, p4_probs, thresholds, GENRES)

        # Delta relative to same-ID unrecovered baseline (P1)
        delta_p2_micro = eval_p2["micro_f1"] - eval_p1["micro_f1"]
        delta_p2_macro = eval_p2["macro_f1"] - eval_p1["macro_f1"]

        delta_p3_micro = eval_p3["micro_f1"] - eval_p1["micro_f1"]
        delta_p3_macro = eval_p3["macro_f1"] - eval_p1["macro_f1"]

        delta_p4_micro = eval_p4["micro_f1"] - eval_p1["micro_f1"]

        logger.info(f"P1 (No Recovery):        Micro-F1 = {eval_p1['micro_f1']:.4f} | Macro-F1 = {eval_p1['macro_f1']:.4f}")
        logger.info(f"P2 (Fixed Policy):       Micro-F1 = {eval_p2['micro_f1']:.4f} | Delta = {delta_p2_micro:+.4f}")
        logger.info(f"P3 (Adaptive Policy):    Micro-F1 = {eval_p3['micro_f1']:.4f} | Delta = {delta_p3_micro:+.4f}")
        logger.info(f"P4 (Oracle Upper Bound): Micro-F1 = {eval_p4['micro_f1']:.4f} | Delta = {delta_p4_micro:+.4f}")

        # Action distribution count for P3
        counts = {a: p3_actions.count(a) for a in ALL_ACTIONS}
        action_distributions[c_name] = counts

        summary_row = {
            "condition_id": cond_id,
            "condition_name": c_name,
            "modality_corrupted": cond["modality_corrupted"],
            "sample_count": audit_n,
            "no_recovery_micro_f1": eval_p1["micro_f1"],
            "no_recovery_macro_f1": eval_p1["macro_f1"],
            "no_recovery_map": eval_p1["mean_average_precision"],
            "no_recovery_hamming": eval_p1["hamming_loss"],
            "fixed_policy_micro_f1": eval_p2["micro_f1"],
            "fixed_policy_macro_f1": eval_p2["macro_f1"],
            "fixed_delta_micro_f1": round(delta_p2_micro, 4),
            "adaptive_policy_micro_f1": eval_p3["micro_f1"],
            "adaptive_policy_macro_f1": eval_p3["macro_f1"],
            "adaptive_delta_micro_f1": round(delta_p3_micro, 4),
            "adaptive_delta_macro_f1": round(delta_p3_macro, 4),
            "oracle_micro_f1": eval_p4["micro_f1"],
            "oracle_delta_micro_f1": round(delta_p4_micro, 4),
            "adaptive_latency_ms_per_sample": round(p3_latency_ms_per_sample, 4),
            "actions_static_fusion_pct": round(counts[ACTION_STATIC_FUSION] / audit_n * 100, 1),
            "actions_text_only_pct": round(counts[ACTION_TEXT_ONLY] / audit_n * 100, 1),
            "actions_image_only_pct": round(counts[ACTION_IMAGE_ONLY] / audit_n * 100, 1),
            "actions_rel_weighted_pct": round(counts[ACTION_RELIABILITY_WEIGHTED] / audit_n * 100, 1),
            "actions_conservative_pct": round(counts[ACTION_CONSERVATIVE_FUSION] / audit_n * 100, 1)
        }
        condition_summary_rows.append(summary_row)

        # Record sample-level decisions
        for i in range(audit_n):
            sample_records.append({
                "movie_id": test_movie_ids[i],
                "condition_id": cond_id,
                "condition_name": c_name,
                "diagnosis": diagnoses[i],
                "p1_no_recovery_pred": (p1_probs[i] >= thresholds).astype(int).tolist(),
                "p2_fixed_action": p2_actions[i],
                "p2_fixed_pred": (p2_probs[i] >= thresholds).astype(int).tolist(),
                "p3_adaptive_action": p3_actions[i],
                "p3_effective_alpha": round(p3_alphas[i], 3),
                "p3_adaptive_pred": (p3_probs[i] >= thresholds).astype(int).tolist(),
                "p4_oracle_action": p4_actions[i],
                "y_true": y_test_true[i].astype(int).tolist()
            })

    # 8. Save CSV and JSON Deliverables
    metrics_df = pd.DataFrame(condition_summary_rows)
    metrics_csv_path = run_dir / "policy_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"\nSaved policy comparison metrics to {metrics_csv_path}")

    sample_df = pd.DataFrame(sample_records)
    sample_csv_path = run_dir / "sample_level_decisions.csv"
    sample_df.to_csv(sample_csv_path, index=False)
    logger.info(f"Saved 4,000 sample-level decisions to {sample_csv_path}")

    dist_json_path = run_dir / "policy_action_distributions.json"
    with open(dist_json_path, "w", encoding="utf-8") as f:
        json.dump(action_distributions, f, indent=2)

    # 9. Overall Summary Statistics
    mean_p1_micro = float(metrics_df["no_recovery_micro_f1"].mean())
    mean_p2_micro = float(metrics_df["fixed_policy_micro_f1"].mean())
    mean_p3_micro = float(metrics_df["adaptive_policy_micro_f1"].mean())
    mean_p4_micro = float(metrics_df["oracle_micro_f1"].mean())

    clean_row = metrics_df[metrics_df["condition_name"] == "Clean_Baseline"].iloc[0]

    benchmark_summary = {
        "benchmark_name": "SEAA_Phase5_Adaptive_Recovery_Evaluation",
        "timestamp": timestamp_str,
        "seed": args.seed,
        "sample_size_per_condition": audit_n,
        "total_conditions": len(conditions),
        "total_evaluations_per_policy": audit_n * len(conditions),
        "overall_mean_micro_f1": {
            "no_recovery": round(mean_p1_micro, 4),
            "fixed_policy": round(mean_p2_micro, 4),
            "adaptive_policy": round(mean_p3_micro, 4),
            "oracle_upper_bound": round(mean_p4_micro, 4),
            "adaptive_gain_over_unrecovered": round(mean_p3_micro - mean_p1_micro, 4),
            "adaptive_gain_over_fixed": round(mean_p3_micro - mean_p2_micro, 4)
        },
        "clean_retention": {
            "clean_no_recovery_micro_f1": float(clean_row["no_recovery_micro_f1"]),
            "clean_adaptive_micro_f1": float(clean_row["adaptive_policy_micro_f1"]),
            "clean_delta": round(float(clean_row["adaptive_delta_micro_f1"]), 4),
            "clean_action_selection": {
                "static_fusion_pct": float(clean_row["actions_static_fusion_pct"]),
                "text_only_pct": float(clean_row["actions_text_only_pct"]),
                "image_only_pct": float(clean_row["actions_image_only_pct"]),
                "rel_weighted_pct": float(clean_row["actions_rel_weighted_pct"]),
                "conservative_pct": float(clean_row["actions_conservative_pct"])
            }
        },
        "runtime_performance": {
            "timing_method": "time.perf_counter()",
            "mean_adaptive_policy_latency_ms": round(float(metrics_df["adaptive_latency_ms_per_sample"].mean()), 4),
            "throughput_samples_per_sec": round(1000.0 / float(metrics_df["adaptive_latency_ms_per_sample"].mean()), 1)
        },
        "negative_deltas_visible": [
            {
                "condition": r["condition_name"],
                "policy": "Adaptive_Policy",
                "delta": r["adaptive_delta_micro_f1"]
            }
            for _, r in metrics_df.iterrows()
            if r["adaptive_delta_micro_f1"] < 0
        ]
    }

    summary_json_path = run_dir / "phase5_benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info(f"Saved benchmark summary to {summary_json_path}")

    # 10. Generate Visualizations
    logger.info("Generating publication-quality figures...")

    # Figure 1: Grouped Bar Chart of Micro-F1 across Policies
    plt.figure(figsize=(14, 7), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    cond_names = metrics_df["condition_name"].tolist()
    x = np.arange(len(cond_names))
    width = 0.20

    plt.bar(x - 1.5 * width, metrics_df["no_recovery_micro_f1"], width, label="No Recovery (Static 0.55)", color="#7f8c8d", alpha=0.9)
    plt.bar(x - 0.5 * width, metrics_df["fixed_policy_micro_f1"], width, label="Fixed Policy (Heuristic)", color="#e67e22", alpha=0.9)
    plt.bar(x + 0.5 * width, metrics_df["adaptive_policy_micro_f1"], width, label="Adaptive Policy (Ours)", color="#27ae60", alpha=0.9)
    plt.bar(x + 1.5 * width, metrics_df["oracle_micro_f1"], width, label="Oracle (Upper Bound)", color="#2980b9", alpha=0.7, linestyle="--", edgecolor="#1a5276")

    plt.xticks(x, [c.replace("_", "\n") for c in cond_names], rotation=0, fontsize=9.5)
    plt.ylabel("Micro-F1 Score", fontweight="bold")
    plt.title("SEAA Phase 5: Policy Micro-F1 Comparison Across Failure Conditions", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0.0, 0.75)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = figures_dir / "policy_micro_f1_comparison.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info(f"Saved Figure 1 to {fig1_path}")

    # Figure 2: Action Distribution Heatmap
    plt.figure(figsize=(11, 7), dpi=300)
    action_matrix = []
    for c in cond_names:
        action_matrix.append([
            action_distributions[c][ACTION_STATIC_FUSION] / audit_n * 100,
            action_distributions[c][ACTION_RELIABILITY_WEIGHTED] / audit_n * 100,
            action_distributions[c][ACTION_CONSERVATIVE_FUSION] / audit_n * 100,
            action_distributions[c][ACTION_TEXT_ONLY] / audit_n * 100,
            action_distributions[c][ACTION_IMAGE_ONLY] / audit_n * 100
        ])
    action_matrix_np = np.array(action_matrix)
    action_labels = ["Static Fusion", "Reliability Weighted", "Conservative", "Text Only", "Image Only"]

    sns.heatmap(
        action_matrix_np,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=action_labels,
        yticklabels=cond_names,
        cbar_kws={"label": "Action Selection Frequency (%)"}
    )
    plt.title("Adaptive Policy Action Selection Distribution (%)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Recovery Action", fontweight="bold")
    plt.ylabel("Failure Condition", fontweight="bold")
    plt.tight_layout()
    fig2_path = figures_dir / "action_distribution_heatmap.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info(f"Saved Figure 2 to {fig2_path}")

    # Figure 3: Per-Condition Recovery Deltas (ΔMicro-F1)
    plt.figure(figsize=(12, 6), dpi=300)
    deltas = metrics_df["adaptive_delta_micro_f1"].values
    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in deltas]

    bars = plt.bar(x, deltas, color=colors, width=0.55, edgecolor="#2c3e50")
    plt.axhline(0, color="black", linewidth=1.0, linestyle="--")
    plt.xticks(x, [c.replace("_", "\n") for c in cond_names], rotation=0, fontsize=9.5)
    plt.ylabel("Δ Micro-F1 (Adaptive - No Recovery)", fontweight="bold")
    plt.title("Per-Condition Recovery Gain Relative to Same-ID Unrecovered Baseline", fontsize=13, fontweight="bold", pad=15)

    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + (0.01 if val >= 0 else -0.02)
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontweight="bold", fontsize=9)

    plt.ylim(min(deltas) - 0.05, max(deltas) + 0.08)
    plt.tight_layout()
    fig3_path = figures_dir / "per_condition_recovery_deltas.png"
    plt.savefig(fig3_path)
    plt.close()
    logger.info(f"Saved Figure 3 to {fig3_path}")

    logger.info("=" * 70)
    logger.info("PHASE 5 BENCHMARK EVALUATION COMPLETE")
    logger.info(f"Overall Unrecovered Micro-F1: {mean_p1_micro:.4f}")
    logger.info(f"Overall Fixed Policy Micro-F1: {mean_p2_micro:.4f}")
    logger.info(f"Overall Adaptive Policy Micro-F1: {mean_p3_micro:.4f}")
    logger.info(f"Overall Oracle Upper Bound Micro-F1: {mean_p4_micro:.4f}")
    logger.info(f"Adaptive Absolute Gain over Unrecovered: {mean_p3_micro - mean_p1_micro:+.4f}")
    logger.info(f"Adaptive Absolute Gain over Fixed Policy: {mean_p3_micro - mean_p2_micro:+.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
