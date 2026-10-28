"""
SEAA Phase 5 & IEEE Benchmark: Ablation Studies Suite.

Rigorously benchmarks and compares:
1. Full SEAA (Trained Adaptive Recovery Policy)
2. Ablation 1: SEAA with Learning Disabled (Untrained Policy with prior Q-table)
3. Ablation 2: SEAA with No Diagnosis Features (Context state stripped of G_t)
4. Ablation 3: Random Action Selection (Uniform choice over admissible actions)
5. Baseline 1: No Recovery (Static Late Fusion, alpha=0.55)
6. Baseline 2: Fixed Policy (DynamicModalityGating heuristic router)
7. Theoretical Ceiling: Oracle Upper Bound (Best valid action per sample)

Evaluated across the identical 500 test movie samples over all 8 failure regimes.
Calculates condition-averaged and pooled Micro-F1, Macro-F1, and instance-level Mean F1.
Generates publication-quality ablation charts and machine-readable JSON/CSV summaries.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, cast

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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import GENRES, get_default_image_transform
from src.evaluation.metrics import evaluate_multilabel_predictions
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel
from src.failures.text_corruptions import corrupt_text_batch
from src.failures.image_corruptions import extract_corrupted_image_features, apply_missing_image
from src.auditing.diagnosis import calibrate_diagnostic_thresholds_on_val, MultimodalReliabilityAuditor
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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("AblationSuite")


def train_policy_variant(
    val_indices: List[int],
    meta_df: pd.DataFrame,
    all_cached_feats: np.ndarray,
    text_model: TextBaselineModel,
    image_model: ImageBaselineModel,
    auditor: MultimodalReliabilityAuditor,
    thresholds: np.ndarray,
    strip_diagnosis: bool = False,
    seed: int = 42
) -> AdaptiveRecoveryPolicy:
    """Trains an AdaptiveRecoveryPolicy on validation episodes, optionally stripping diagnosis."""
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

    val_t_probs = text_model.predict_proba(val_texts)
    val_i_probs = image_model.predict_proba(val_feats)
    val_mm_probs = 0.55 * val_t_probs + 0.45 * val_i_probs
    t_sparse = text_model.vectorizer.transform(val_texts)
    val_t_norms = np.sqrt(np.asarray(t_sparse.power(2).sum(axis=1)).ravel())
    val_i_norms = np.linalg.norm(val_feats, axis=1)

    # 1. Clean episodes
    audit_clean = auditor.audit_batch(
        text_probs=val_t_probs, image_probs=val_i_probs, fused_probs=val_mm_probs,
        text_feature_norms=val_t_norms, image_feature_norms=val_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(audit_clean["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_clean["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_clean["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_clean["disagreement"]["hamming_distance"][i]),
            text_norm=float(val_t_norms[i]), image_norm=float(val_i_norms[i]),
            diagnosis="UNSPECIFIED" if strip_diagnosis else "CLEAN"
        )
        act = policy.select_action(ctx, explore=True)
        rec, _ = policy.apply_action(act, val_t_probs[i], val_i_probs[i], ctx)
        r = policy.compute_reward(rec, val_mm_probs[i], y_val[i], thresholds)
        policy.update(ctx, act, r)

    # 2. Text failure episodes
    corrupted_t_texts = corrupt_text_batch(val_texts, corruption_type="truncation", keep_ratio=0.25, base_seed=seed)
    corrupted_t_probs = text_model.predict_proba(corrupted_t_texts)
    c_sparse = text_model.vectorizer.transform(corrupted_t_texts)
    c_t_norms = np.sqrt(np.asarray(c_sparse.power(2).sum(axis=1)).ravel())
    unrec_t_fail = 0.55 * corrupted_t_probs + 0.45 * val_i_probs
    audit_tfail = auditor.audit_batch(
        text_probs=corrupted_t_probs, image_probs=val_i_probs, fused_probs=unrec_t_fail,
        text_feature_norms=c_t_norms, image_feature_norms=val_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(audit_tfail["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_tfail["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_tfail["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_tfail["disagreement"]["hamming_distance"][i]),
            text_norm=float(c_t_norms[i]), image_norm=float(val_i_norms[i]),
            diagnosis="UNSPECIFIED" if strip_diagnosis else "TEXT_FAILURE"
        )
        act = policy.select_action(ctx, explore=True)
        rec, _ = policy.apply_action(act, corrupted_t_probs[i], val_i_probs[i], ctx)
        r = policy.compute_reward(rec, unrec_t_fail[i], y_val[i], thresholds)
        policy.update(ctx, act, r)

    # 3. Missing text episodes
    empty_t_probs = np.full_like(val_t_probs, 0.05)
    unrec_empty_t = 0.55 * empty_t_probs + 0.45 * val_i_probs
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=False, has_image=True, text_entropy=0.5,
            image_entropy=float(audit_clean["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=0.20, text_norm=0.0, image_norm=float(val_i_norms[i]),
            diagnosis="UNSPECIFIED" if strip_diagnosis else "MISSING_TEXT"
        )
        act = policy.select_action(ctx, explore=True)
        rec, _ = policy.apply_action(act, empty_t_probs[i], val_i_probs[i], ctx)
        r = policy.compute_reward(rec, unrec_empty_t[i], y_val[i], thresholds)
        policy.update(ctx, act, r)

    # 4. Image failure episodes
    noise_feats = val_feats + np.random.RandomState(seed).normal(0.0, 0.25, val_feats.shape)
    noise_i_probs = image_model.predict_proba(noise_feats)
    noise_i_norms = np.linalg.norm(noise_feats, axis=1)
    unrec_i_fail = 0.55 * val_t_probs + 0.45 * noise_i_probs
    audit_ifail = auditor.audit_batch(
        text_probs=val_t_probs, image_probs=noise_i_probs, fused_probs=unrec_i_fail,
        text_feature_norms=val_t_norms, image_feature_norms=noise_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(audit_ifail["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_ifail["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_ifail["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_ifail["disagreement"]["hamming_distance"][i]),
            text_norm=float(val_t_norms[i]), image_norm=float(noise_i_norms[i]),
            diagnosis="UNSPECIFIED" if strip_diagnosis else "IMAGE_FAILURE"
        )
        act = policy.select_action(ctx, explore=True)
        rec, _ = policy.apply_action(act, val_t_probs[i], noise_i_probs[i], ctx)
        r = policy.compute_reward(rec, unrec_i_fail[i], y_val[i], thresholds)
        policy.update(ctx, act, r)

    # 5. Dual failure episodes
    unrec_dual = 0.55 * corrupted_t_probs + 0.45 * noise_i_probs
    audit_dual = auditor.audit_batch(
        text_probs=corrupted_t_probs, image_probs=noise_i_probs, fused_probs=unrec_dual,
        text_feature_norms=c_t_norms, image_feature_norms=noise_i_norms
    )
    for i in range(val_n):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(audit_dual["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(audit_dual["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(audit_dual["disagreement"]["mean_bernoulli_jsd"][i]),
            hamming_distance=float(audit_dual["disagreement"]["hamming_distance"][i]),
            text_norm=float(c_t_norms[i]), image_norm=float(noise_i_norms[i]),
            diagnosis="UNSPECIFIED" if strip_diagnosis else "HIGH_UNCERTAINTY"
        )
        act = policy.select_action(ctx, explore=True)
        rec, _ = policy.apply_action(act, corrupted_t_probs[i], noise_i_probs[i], ctx)
        r = policy.compute_reward(rec, unrec_dual[i], y_val[i], thresholds)
        policy.update(ctx, act, r)

    policy.learning_enabled = False
    return policy


def run_ablation_benchmarks():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    ckpt_dir = base_dir / "outputs" / "checkpoints"
    recovery_dir = base_dir / "outputs" / "recovery"
    if not recovery_dir.exists():
        recovery_dir = base_dir / "outputs" / "phase5"
    figures_dir = recovery_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SEAA ABLATION STUDIES: ISOLATING SELF-EVOLUTION AND ROUTING MECHANISMS")
    logger.info("=" * 70)

    # Load metadata and splits
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    with open(data_dir / "split_indices.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    feats_path = data_dir / "processed" / "resnet50_features.npy"
    all_cached_feats = np.memmap(feats_path, dtype="float32", mode="r", shape=(25959, 2048))

    # Load models
    text_model = TextBaselineModel.load(ckpt_dir / "text_baseline.joblib")
    image_model = ImageBaselineModel.load(ckpt_dir / "image_baseline.joblib")
    mm_model = MultimodalBaselineModel.load(ckpt_dir / "multimodal_baseline.json")
    thresholds = mm_model.thresholds

    # Initialize auditor
    val_indices = splits["val"]
    val_sample_indices = val_indices[:500]
    val_cached_feats = np.asarray(all_cached_feats[val_sample_indices])
    val_texts = [str(t) if pd.notnull(t) else "" for t in meta_df.iloc[val_sample_indices]["text"]]
    val_t_probs = text_model.predict_proba(val_texts)
    val_i_probs = image_model.predict_proba(val_cached_feats)
    t_sparse = text_model.vectorizer.transform(val_texts)
    val_t_norms = np.sqrt(np.asarray(t_sparse.power(2).sum(axis=1)).ravel())
    val_i_norms = np.linalg.norm(val_cached_feats, axis=1)

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

    # 1. Load or train policies
    logger.info("Configuring policies for ablation analysis...")

    # Policy 1: Full SEAA
    trained_policy_path = recovery_dir / "run_20261025_phase5_adaptive_recovery" / "trained_adaptive_policy.json"
    full_seaa_policy = AdaptiveRecoveryPolicy.load_policy(trained_policy_path)
    full_seaa_policy.learning_enabled = False

    # Policy 2: Learning-Disabled (Untrained Q-Table)
    untrained_policy = AdaptiveRecoveryPolicy(
        learning_rate=0.20, default_alpha=0.55, seed=42, learning_enabled=False
    )

    # Policy 3: No-Diagnosis Features
    logger.info("Training Ablation Policy without Diagnosis features...")
    no_diag_policy = train_policy_variant(
        val_indices=val_indices[:500],
        meta_df=meta_df,
        all_cached_feats=all_cached_feats,
        text_model=text_model,
        image_model=image_model,
        auditor=auditor,
        thresholds=thresholds,
        strip_diagnosis=True,
        seed=42
    )

    # Baseline: Fixed Policy
    fixed_engine = DynamicModalityGating(default_alpha=mm_model.alpha, conflict_strategy="lower_entropy")

    # 2. Test Subset Setup
    test_indices = splits["test"][:500]
    test_meta = meta_df.iloc[test_indices].reset_index(drop=True)
    test_movie_ids = test_meta["movie_id"].tolist()
    test_clean_texts = [str(t) if pd.notnull(t) else "" for t in test_meta["text"]]
    test_clean_feats = np.asarray(all_cached_feats[test_indices])
    y_test_true = np.array([
        [int(v) for v in row.split(",")]
        for row in test_meta["target_multihot"]
    ], dtype=np.float32)
    audit_n = len(test_indices)

    # Load or extract corrupted image features
    cache_path = recovery_dir / "cached_test500_corrupted_feats.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cache_path.exists():
        logger.info(f"Loading cached corrupted visual features from {cache_path}...")
        corrupted_cache = np.load(cache_path)
        blur_feats = corrupted_cache["blur_feats"]
        noise_feats = corrupted_cache["noise_feats"]
        black_feats = corrupted_cache["black_feats"]
    else:
        logger.info("Extracting and caching corrupted visual features for test subset...")
        raw_dataset = load_dataset("pranavmr/MM-IMDb", split="train")
        raw_test_images = [raw_dataset[idx]["image"] for idx in test_indices]
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()
        resnet.to(device).eval()
        image_transform = get_default_image_transform()

        black_img = apply_missing_image()
        with torch.inference_mode():
            black_tensor = image_transform(black_img).unsqueeze(0).to(device)
            single_black = resnet(black_tensor).cpu().numpy()[0]
        black_feats = np.tile(single_black, (audit_n, 1))

        blur_feats = extract_corrupted_image_features(
            images=raw_test_images, corruption_type="gaussian_blur",
            feature_extractor_model=resnet, image_transform=image_transform,
            device=device, batch_size=32, base_seed=42, radius=5.0
        )
        noise_feats = extract_corrupted_image_features(
            images=raw_test_images, corruption_type="gaussian_noise",
            feature_extractor_model=resnet, image_transform=image_transform,
            device=device, batch_size=32, base_seed=42, std=0.25
        )
        np.savez_compressed(cache_path, blur_feats=blur_feats, noise_feats=noise_feats, black_feats=black_feats)
        logger.info(f"Saved corrupted features cache to {cache_path}.")

    # 3. Define 8 Benchmark Conditions
    conditions = [
        {"id": 0, "name": "Clean_Baseline", "modality": "None", "text": "clean", "image": "clean"},
        {"id": 1, "name": "Text_Truncation_25%", "modality": "Text", "text": "truncation", "image": "clean", "t_kw": {"keep_ratio": 0.25}},
        {"id": 2, "name": "Text_Token_Masking_35%", "modality": "Text", "text": "masking", "image": "clean", "t_kw": {"mask_rate": 0.35, "mask_token": "[MASK]"}},
        {"id": 3, "name": "Text_Empty", "modality": "Text", "text": "empty", "image": "clean"},
        {"id": 4, "name": "Image_Gaussian_Blur_r5", "modality": "Vision", "text": "clean", "image": "blur"},
        {"id": 5, "name": "Image_Gaussian_Noise_s25", "modality": "Vision", "text": "clean", "image": "noise"},
        {"id": 6, "name": "Image_Missing_Blackout", "modality": "Vision", "text": "clean", "image": "black"},
        {"id": 7, "name": "Dual_Corruption_Trunc_Noise", "modality": "Both", "text": "truncation", "image": "noise", "t_kw": {"keep_ratio": 0.25}}
    ]

    # Model evaluation containers
    evaluated_policies = [
        "Full_SEAA",
        "Ablation_Learning_Disabled",
        "Ablation_No_Diagnosis",
        "Ablation_Random_Action",
        "Baseline_No_Recovery",
        "Baseline_Fixed_Policy",
        "Oracle_Upper_Bound"
    ]

    # Structure to hold predictions per condition
    # policy -> condition_name -> y_pred_binary
    condition_f1_results: Dict[str, Dict[str, Dict[str, float]]] = {p: {} for p in evaluated_policies}
    all_y_true_pooled = []
    all_preds_pooled: Dict[str, List[np.ndarray]] = {p: [] for p in evaluated_policies}

    rng_random = np.random.RandomState(42)

    for cond in conditions:
        c_name = cond["name"]
        logger.info(f"Evaluating {c_name}...")

        # Prepare Text
        if cond["text"] == "clean":
            cond_texts = list(test_clean_texts)
            has_text_flags = [bool(t.strip()) for t in cond_texts]
        elif cond["text"] == "empty":
            cond_texts = [""] * audit_n
            has_text_flags = [False] * audit_n
        else:
            t_kw = cond.get("t_kw", {})
            cond_texts = corrupt_text_batch(test_clean_texts, corruption_type=cond["text"], base_seed=42, **t_kw)
            has_text_flags = [bool(t.strip()) for t in cond_texts]

        t_sparse = text_model.vectorizer.transform(cond_texts)
        t_norms = np.sqrt(np.asarray(t_sparse.power(2).sum(axis=1)).ravel())

        # Prepare Image
        if cond["image"] == "clean":
            cond_img_feats = test_clean_feats
            has_image_flags = [True] * audit_n
        elif cond["image"] == "blur":
            cond_img_feats = blur_feats
            has_image_flags = [True] * audit_n
        elif cond["image"] == "noise":
            cond_img_feats = noise_feats
            has_image_flags = [True] * audit_n
        elif cond["image"] == "black":
            cond_img_feats = black_feats
            has_image_flags = [False] * audit_n

        img_norms = np.linalg.norm(cond_img_feats, axis=1)

        # Baseline predictions
        t_probs = text_model.predict_proba(cond_texts)
        i_probs = image_model.predict_proba(cond_img_feats)
        mm_unrec = mm_model.alpha * t_probs + (1.0 - mm_model.alpha) * i_probs

        # Audit
        audit = auditor.audit_batch(
            text_probs=t_probs, image_probs=i_probs, fused_probs=mm_unrec,
            has_text=has_text_flags, has_image=has_image_flags,
            text_feature_norms=t_norms, image_feature_norms=img_norms
        )
        diagnoses = audit["diagnoses"]
        t_ent = audit["text_uncertainty"]["mean_binary_entropy"]
        i_ent = audit["image_uncertainty"]["mean_binary_entropy"]
        jsds = audit["disagreement"]["mean_bernoulli_jsd"]
        hamm = audit["disagreement"]["hamming_distance"]

        # 1. Full SEAA
        full_probs = np.zeros_like(t_probs)
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]), has_image=bool(has_image_flags[i]),
                text_entropy=float(t_ent[i]), image_entropy=float(i_ent[i]),
                bernoulli_jsd=float(jsds[i]), hamming_distance=float(hamm[i]),
                text_norm=float(t_norms[i]), image_norm=float(img_norms[i]),
                diagnosis=str(diagnoses[i])
            )
            act = full_seaa_policy.select_action(ctx, explore=False)
            rec, _ = full_seaa_policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            full_probs[i] = rec

        # 2. Learning-Disabled (Untrained Q-table)
        untrained_probs = np.zeros_like(t_probs)
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]), has_image=bool(has_image_flags[i]),
                text_entropy=float(t_ent[i]), image_entropy=float(i_ent[i]),
                bernoulli_jsd=float(jsds[i]), hamming_distance=float(hamm[i]),
                text_norm=float(t_norms[i]), image_norm=float(img_norms[i]),
                diagnosis=str(diagnoses[i])
            )
            act = untrained_policy.select_action(ctx, explore=False)
            rec, _ = untrained_policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            untrained_probs[i] = rec

        # 3. No-Diagnosis Feature
        no_diag_probs = np.zeros_like(t_probs)
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]), has_image=bool(has_image_flags[i]),
                text_entropy=float(t_ent[i]), image_entropy=float(i_ent[i]),
                bernoulli_jsd=float(jsds[i]), hamming_distance=float(hamm[i]),
                text_norm=float(t_norms[i]), image_norm=float(img_norms[i]),
                diagnosis="UNSPECIFIED"
            )
            act = no_diag_policy.select_action(ctx, explore=False)
            rec, _ = no_diag_policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            no_diag_probs[i] = rec

        # 4. Random Action
        rand_probs = np.zeros_like(t_probs)
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]), has_image=bool(has_image_flags[i])
            )
            valid = ctx.get_valid_actions()
            act = rng_random.choice(valid)
            rec, _ = full_seaa_policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            rand_probs[i] = rec

        # 5. Baseline: No Recovery
        no_rec_probs = mm_unrec

        # 6. Baseline: Fixed Policy
        p2_out = fixed_engine.recover_predictions(
            text_probs=t_probs, image_probs=i_probs, diagnoses=diagnoses,
            text_uncertainties=audit["text_uncertainty"]["composite_uncertainty"],
            image_uncertainties=audit["image_uncertainty"]["composite_uncertainty"],
            default_fused_probs=mm_unrec
        )
        fixed_probs = p2_out["recovered_probs"]

        # 7. Oracle Upper Bound
        oracle_probs = np.zeros_like(t_probs)
        for i in range(audit_n):
            ctx = PolicyContext(
                has_text=bool(has_text_flags[i]), has_image=bool(has_image_flags[i])
            )
            valid = ctx.get_valid_actions()
            best_f1 = -1.0
            best_rec = mm_unrec[i]
            y_i = y_test_true[i].astype(int)
            for a in valid:
                r_prob, _ = full_seaa_policy.apply_action(a, t_probs[i], i_probs[i], ctx)
                pred_bin = (r_prob >= thresholds).astype(int)
                tp = np.sum((pred_bin == 1) & (y_i == 1))
                denom = 2 * tp + np.sum((pred_bin == 1) & (y_i == 0)) + np.sum((pred_bin == 0) & (y_i == 1))
                f1_val = (2.0 * tp / denom) if denom > 0 else 1.0
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_rec = r_prob
            oracle_probs[i] = best_rec

        # Collect Predictions
        policy_preds_map = {
            "Full_SEAA": (full_probs >= thresholds).astype(int),
            "Ablation_Learning_Disabled": (untrained_probs >= thresholds).astype(int),
            "Ablation_No_Diagnosis": (no_diag_probs >= thresholds).astype(int),
            "Ablation_Random_Action": (rand_probs >= thresholds).astype(int),
            "Baseline_No_Recovery": (no_rec_probs >= thresholds).astype(int),
            "Baseline_Fixed_Policy": (fixed_probs >= thresholds).astype(int),
            "Oracle_Upper_Bound": (oracle_probs >= thresholds).astype(int)
        }

        all_y_true_pooled.append(y_test_true)
        for p_name, pred_mat in policy_preds_map.items():
            all_preds_pooled[p_name].append(pred_mat)
            ev = evaluate_multilabel_predictions(y_test_true, pred_mat, thresholds, GENRES)
            condition_f1_results[p_name][c_name] = {
                "micro_f1": ev["micro_f1"],
                "macro_f1": ev["macro_f1"]
            }

    y_pooled_mat = np.vstack(all_y_true_pooled)

    # 4. Compute Aggregate Metrics Across All 8 Conditions
    ablation_summary = []
    logger.info("\n" + "=" * 90)
    logger.info("FINAL ABLATION BENCHMARK RESULTS (500 Samples x 8 Failure Conditions = 4,000 Evaluations)")
    logger.info("=" * 90)

    for p_name in evaluated_policies:
        pred_pooled = np.vstack(all_preds_pooled[p_name])
        ev_pooled = evaluate_multilabel_predictions(y_pooled_mat, pred_pooled, thresholds, GENRES)

        # Condition-averaged Micro and Macro F1
        cond_micros = [condition_f1_results[p_name][c["name"]]["micro_f1"] for c in conditions]
        cond_macros = [condition_f1_results[p_name][c["name"]]["macro_f1"] for c in conditions]

        mean_cond_micro = float(np.mean(cond_micros))
        mean_cond_macro = float(np.mean(cond_macros))
        pooled_micro = float(ev_pooled["micro_f1"])
        pooled_macro = float(ev_pooled["macro_f1"])

        # Instance-level mean F1
        tp_inst = np.sum((pred_pooled == 1) & (y_pooled_mat == 1), axis=1)
        denom_inst = 2 * tp_inst + np.sum((pred_pooled == 1) & (y_pooled_mat == 0), axis=1) + np.sum((pred_pooled == 0) & (y_pooled_mat == 1), axis=1)
        inst_f1s = np.where(denom_inst > 0, 2.0 * tp_inst / denom_inst, 1.0)
        mean_instance_f1 = float(np.mean(inst_f1s))

        # Delta relative to Baseline_No_Recovery
        base_micro = float(np.mean([condition_f1_results["Baseline_No_Recovery"][c["name"]]["micro_f1"] for c in conditions]))
        delta_micro = mean_cond_micro - base_micro

        row_dict = {
            "Policy": p_name,
            "Condition_Mean_Micro_F1": mean_cond_micro,
            "Condition_Mean_Macro_F1": mean_cond_macro,
            "Globally_Pooled_Micro_F1": pooled_micro,
            "Globally_Pooled_Macro_F1": pooled_macro,
            "Instance_Mean_F1": mean_instance_f1,
            "Delta_vs_NoRecovery": delta_micro
        }
        ablation_summary.append(row_dict)
        logger.info(f"{p_name:28s} | Cond-Micro: {mean_cond_micro:.4f} | Cond-Macro: {mean_cond_macro:.4f} | Pooled-Micro: {pooled_micro:.4f} | Delta: {delta_micro:+.4f}")

    ablation_df = pd.DataFrame(ablation_summary)
    csv_out = recovery_dir / "ablation_study_results.csv"
    ablation_df.to_csv(csv_out, index=False)
    logger.info(f"Saved ablation CSV to {csv_out}")

    json_out = recovery_dir / "ablation_study_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "summary_table": ablation_summary,
            "per_condition_results": condition_f1_results
        }, f, indent=2)
    logger.info(f"Saved ablation JSON to {json_out}")

    # 5. Publication-Quality Plot: Ablation Comparison
    plt.figure(figsize=(11, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = [
        "#1b9e77",  # Full SEAA (Green)
        "#d95f02",  # Learning Disabled (Orange)
        "#7570b3",  # No Diagnosis (Purple)
        "#e7298a",  # Random Action (Pink)
        "#66a61e",  # Baseline No Recovery (Olive)
        "#e6ab02",  # Baseline Fixed Policy (Gold)
        "#666666"   # Oracle (Grey)
    ]

    friendly_names = [
        "Full SEAA\n(Adaptive Policy)",
        "Ablation 1:\nLearning Disabled",
        "Ablation 2:\nNo Diagnosis",
        "Ablation 3:\nRandom Action",
        "Baseline 1:\nNo Recovery",
        "Baseline 2:\nFixed Policy",
        "Theoretical:\nOracle Bound"
    ]

    bars = plt.bar(
        friendly_names,
        ablation_df["Condition_Mean_Micro_F1"],
        color=palette,
        edgecolor="#333333",
        linewidth=1.0,
        width=0.62
    )

    plt.axhline(
        ablation_df.loc[ablation_df["Policy"] == "Baseline_No_Recovery", "Condition_Mean_Micro_F1"].values[0],
        color="#888888", linestyle="--", linewidth=1.2, label="Unrecovered Baseline"
    )

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.008, f"{h:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.title("SEAA Component Ablation: Demonstrating Outcome Learning and Diagnostic Necessity", fontsize=12, fontweight="bold", pad=15)
    plt.ylabel("Condition-Averaged Micro-F1 Score", fontsize=11, fontweight="semibold")
    plt.ylim(0.30, 0.62)
    plt.xticks(fontsize=9.5)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    fig_png = figures_dir / "ablation_comparison_barchart.png"
    fig_pdf = figures_dir / "ablation_comparison_barchart.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info(f"Saved publication ablation charts to {fig_png} and {fig_pdf}")

if __name__ == "__main__":
    run_ablation_benchmarks()
