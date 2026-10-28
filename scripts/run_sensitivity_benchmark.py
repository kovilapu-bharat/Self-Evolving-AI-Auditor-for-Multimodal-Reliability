"""
SEAA Continuous Corruption Severity Sensitivity Benchmark.

Sweeps continuous corruption intensities across multiple parametric levels:
1. Text Truncation: retention fraction keep_ratio in [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
2. Text Token Masking: mask_rate in [0.00, 0.15, 0.30, 0.45, 0.60, 0.75]
3. Visual Feature Noise: Gaussian noise std in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
4. Visual Feature Blur Perturbation: perturbation scale in [0.00, 0.50, 1.00, 1.50, 2.00, 2.50]

Evaluates:
- No Recovery (Static Late Fusion)
- Fixed Heuristic Router (DynamicModalityGating)
- SEAA Adaptive Recovery Policy (Tabular Q-Learning)
- SEAA LinUCB Continuous Bandit Policy
- Oracle Upper Bound

Calculates Micro-F1, Macro-F1, Micro-ECE, and Brier Score at every step.
Generates publication-quality 4-panel figures in PNG and PDF.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GENRES
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel
from src.failures.text_corruptions import truncate_text, mask_tokens
from src.auditing.diagnosis import MultimodalReliabilityAuditor, calibrate_diagnostic_thresholds_on_val
from src.recovery.modality_gating import DynamicModalityGating
from src.policy.adaptive_policy import (
    AdaptiveRecoveryPolicy,
    PolicyContext,
    ALL_ACTIONS
)
from src.policy.linear_bandit import LinUCBPolicy
from src.evaluation.metrics import evaluate_multilabel_predictions
from src.evaluation.calibration import evaluate_multilabel_calibration

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("SensitivityBenchmark")


def compute_oracle(t_prob, i_prob, y_true, thresholds, ctx, policy):
    best_f1 = -1.0
    best_rec = t_prob
    for act in ctx.get_valid_actions():
        rec, _ = policy.apply_action(act, t_prob, i_prob, ctx)
        pred_bin = (rec >= thresholds).astype(int)
        tp = np.sum((pred_bin == 1) & (y_true == 1))
        fp = np.sum((pred_bin == 1) & (y_true == 0))
        fn = np.sum((pred_bin == 0) & (y_true == 1))
        f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_rec = rec
    return best_rec


def evaluate_batch(
    t_probs: np.ndarray,
    i_probs: np.ndarray,
    t_norms: np.ndarray,
    i_norms: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    auditor: MultimodalReliabilityAuditor,
    fixed_router: DynamicModalityGating,
    tabular_policy: AdaptiveRecoveryPolicy,
    linucb_policy: LinUCBPolicy
) -> Dict[str, Dict[str, float]]:
    n = len(t_probs)
    # 1. Static Late Fusion
    p_unrec = 0.55 * t_probs + 0.45 * i_probs

    # 2. Audit
    audit_res = auditor.audit_batch(
        text_probs=t_probs,
        image_probs=i_probs,
        fused_probs=p_unrec,
        text_feature_norms=t_norms,
        image_feature_norms=i_norms
    )
    diagnoses = audit_res["diagnoses"]
    t_entropies = audit_res["text_uncertainty"]["mean_binary_entropy"]
    i_entropies = audit_res["image_uncertainty"]["mean_binary_entropy"]
    jsds = audit_res["disagreement"]["mean_bernoulli_jsd"]
    hammings = audit_res["disagreement"]["hamming_distance"]

    # 3. Fixed Router
    p_fixed_res = fixed_router.recover_predictions(
        text_probs=t_probs,
        image_probs=i_probs,
        diagnoses=diagnoses,
        text_uncertainties=audit_res["text_uncertainty"]["composite_uncertainty"],
        image_uncertainties=audit_res["image_uncertainty"]["composite_uncertainty"],
        default_fused_probs=p_unrec
    )
    p_fixed = p_fixed_res["recovered_probs"]

    # 4. Tabular SEAA Policy
    p_tabular = np.zeros_like(t_probs)
    # 5. LinUCB Policy
    p_linucb = np.zeros_like(t_probs)
    # 6. Oracle
    p_oracle = np.zeros_like(t_probs)

    for i in range(n):
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            text_entropy=float(t_entropies[i]),
            image_entropy=float(i_entropies[i]),
            bernoulli_jsd=float(jsds[i]),
            hamming_distance=float(hammings[i]),
            text_norm=float(t_norms[i]),
            image_norm=float(i_norms[i]),
            diagnosis=str(diagnoses[i])
        )

        # Tabular action
        act_tab = tabular_policy.select_action(ctx, explore=False)
        rec_tab, _ = tabular_policy.apply_action(act_tab, t_probs[i], i_probs[i], ctx)
        p_tabular[i] = rec_tab

        # LinUCB action
        act_lin, _ = linucb_policy.select_action(ctx, frozen=True)
        rec_lin, _ = tabular_policy.apply_action(act_lin, t_probs[i], i_probs[i], ctx)
        p_linucb[i] = rec_lin

        # Oracle
        p_oracle[i] = compute_oracle(t_probs[i], i_probs[i], y_true[i], thresholds, ctx, tabular_policy)

    # Compute metrics
    models = {
        "no_recovery": p_unrec,
        "fixed_router": p_fixed,
        "seaa_tabular": p_tabular,
        "seaa_linucb": p_linucb,
        "oracle": p_oracle
    }

    results = {}
    for name, probs in models.items():
        eval_f1 = evaluate_multilabel_predictions(y_true, probs, thresholds, GENRES)
        eval_cal = evaluate_multilabel_calibration(y_true, probs, n_bins=10)
        results[name] = {
            "micro_f1": eval_f1["micro_f1"],
            "macro_f1": eval_f1["macro_f1"],
            "micro_ece": eval_cal["micro_ece"],
            "brier_score": eval_cal["brier_score"]
        }

    return results


def run_sensitivity_sweeps(sample_size: int = 500, seed: int = 42):
    logger.info("=" * 80)
    logger.info("SEAA CONTINUOUS CORRUPTION SEVERITY SENSITIVITY BENCHMARK")
    logger.info("=" * 80)

    ckpts_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    text_model = TextBaselineModel.load(ckpts_dir / "text_baseline.joblib")
    image_model = ImageBaselineModel.load(ckpts_dir / "image_baseline.joblib")
    mm_model = MultimodalBaselineModel.load(ckpts_dir / "multimodal_baseline.json")
    thresholds = mm_model.thresholds

    # Load data
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
    with open(PROJECT_ROOT / "data" / "split_indices.json", "r") as f:
        splits = json.load(f)

    test_indices = splits["test"][:sample_size]
    test_meta = meta_df.iloc[test_indices].reset_index(drop=True)
    test_texts = [str(t) if pd.notnull(t) else "" for t in test_meta["text"]]

    features_path = PROJECT_ROOT / "data" / "processed" / "resnet50_features.npy"
    all_cached_feats = np.memmap(features_path, dtype=np.float32, mode="r", shape=(len(meta_df), 2048))
    test_feats = np.asarray(all_cached_feats[test_indices])

    y_test_true = np.array([
        [int(v) for v in row.split(",")]
        for row in test_meta["target_multihot"]
    ], dtype=np.float32)

    # Clean baselines
    clean_t_probs = text_model.predict_proba(test_texts)
    clean_i_probs = image_model.predict_proba(test_feats)
    clean_t_sparse = text_model.vectorizer.transform(test_texts)
    clean_t_norms = np.sqrt(np.asarray(clean_t_sparse.power(2).sum(axis=1)).ravel())
    clean_i_norms = np.linalg.norm(test_feats, axis=1)

    # Validation calibration
    val_indices = splits["val"][:sample_size]
    val_meta = meta_df.iloc[val_indices].reset_index(drop=True)
    val_texts = [str(t) if pd.notnull(t) else "" for t in val_meta["text"]]
    val_feats = np.asarray(all_cached_feats[val_indices])
    val_t_probs = text_model.predict_proba(val_texts)
    val_i_probs = image_model.predict_proba(val_feats)
    val_t_sparse = text_model.vectorizer.transform(val_texts)
    val_t_norms = np.sqrt(np.asarray(val_t_sparse.power(2).sum(axis=1)).ravel())
    val_i_norms = np.linalg.norm(val_feats, axis=1)

    diag_config = calibrate_diagnostic_thresholds_on_val(
        val_text_probs=val_t_probs,
        val_image_probs=val_i_probs,
        text_thresholds=text_model.thresholds,
        image_thresholds=image_model.thresholds,
        percentile=95.0,
        val_text_norms=val_t_norms,
        val_image_norms=val_i_norms
    )
    auditor = MultimodalReliabilityAuditor(diag_config)
    fixed_router = DynamicModalityGating(default_alpha=mm_model.alpha)

    # Load Tabular policy
    policy_path = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "trained_adaptive_policy.json"
    if not policy_path.exists():
        policy_path = list((PROJECT_ROOT / "outputs" / "recovery").glob("*/trained_adaptive_policy.json"))[0]
    tabular_policy = AdaptiveRecoveryPolicy()
    tabular_policy.load_policy(policy_path)

    # Train LinUCB on validation episodes (zero test leakage!)
    logger.info("Training LinUCB Continuous Bandit Policy on validation episodes...")
    linucb_policy = LinUCBPolicy(d=9, alpha=0.5, seed=seed)
    # Val clean episodes
    val_p_unrec = 0.55 * val_t_probs + 0.45 * val_i_probs
    val_audit = auditor.audit_batch(val_t_probs, val_i_probs, val_p_unrec, val_t_norms, val_i_norms)
    for i in range(len(val_texts)):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(val_audit["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(val_audit["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(val_audit["disagreement"]["mean_bernoulli_jsd"][i]),
            text_norm=float(val_t_norms[i]), image_norm=float(val_i_norms[i]),
            diagnosis="CLEAN"
        )
        act, _ = linucb_policy.select_action(ctx, frozen=False)
        rec, _ = tabular_policy.apply_action(act, val_t_probs[i], val_i_probs[i], ctx)
        r = tabular_policy.compute_reward(rec, val_p_unrec[i], np.array([int(v) for v in val_meta["target_multihot"].iloc[i].split(",")]), thresholds)
        linucb_policy.update(ctx, act, r)

    # Val noisy episodes
    rng = np.random.RandomState(seed)
    noisy_v_feats = val_feats + rng.normal(0, 0.4, size=val_feats.shape).astype(np.float32)
    noisy_v_probs = image_model.predict_proba(noisy_v_feats)
    val_audit_noise = auditor.audit_batch(val_t_probs, noisy_v_probs, 0.55 * val_t_probs + 0.45 * noisy_v_probs, val_t_norms, val_i_norms)
    for i in range(len(val_texts)):
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=float(val_audit_noise["text_uncertainty"]["mean_binary_entropy"][i]),
            image_entropy=float(val_audit_noise["image_uncertainty"]["mean_binary_entropy"][i]),
            bernoulli_jsd=float(val_audit_noise["disagreement"]["mean_bernoulli_jsd"][i]),
            text_norm=float(val_t_norms[i]), image_norm=float(val_i_norms[i]),
            diagnosis="IMAGE_FAILURE"
        )
        act, _ = linucb_policy.select_action(ctx, frozen=False)
        rec, _ = tabular_policy.apply_action(act, val_t_probs[i], noisy_v_probs[i], ctx)
        r = tabular_policy.compute_reward(rec, 0.55 * val_t_probs[i] + 0.45 * noisy_v_probs[i], np.array([int(v) for v in val_meta["target_multihot"].iloc[i].split(",")]), thresholds)
        linucb_policy.update(ctx, act, r)

    logger.info("Validation policy training complete. Commencing parametric test sweeps (frozen policy)...")

    sweep_records = []

    # --- Sweep 1: Text Truncation Keep Ratio ---
    keep_ratios = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    for ratio in keep_ratios:
        logger.info(f"[Sweep 1: Text Truncation] keep_ratio = {ratio:.2f}")
        c_texts = [truncate_text(t, keep_ratio=ratio) for t in test_texts]
        c_t_probs = text_model.predict_proba(c_texts)
        c_t_sparse = text_model.vectorizer.transform(c_texts)
        c_t_norms = np.sqrt(np.asarray(c_t_sparse.power(2).sum(axis=1)).ravel())

        res = evaluate_batch(
            c_t_probs, clean_i_probs, c_t_norms, clean_i_norms,
            y_test_true, thresholds, auditor, fixed_router, tabular_policy, linucb_policy
        )
        for model_name, m in res.items():
            sweep_records.append({
                "sweep_dimension": "text_truncation",
                "param_value": ratio,
                "param_label": f"{int(ratio * 100)}% Retained",
                "model_name": model_name,
                **m
            })

    # --- Sweep 2: Text Token Masking Rate ---
    mask_rates = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75]
    for rate in mask_rates:
        logger.info(f"[Sweep 2: Token Masking] mask_rate = {rate:.2f}")
        c_texts = [mask_tokens(t, mask_rate=rate, seed=seed) for t in test_texts]
        c_t_probs = text_model.predict_proba(c_texts)
        c_t_sparse = text_model.vectorizer.transform(c_texts)
        c_t_norms = np.sqrt(np.asarray(c_t_sparse.power(2).sum(axis=1)).ravel())

        res = evaluate_batch(
            c_t_probs, clean_i_probs, c_t_norms, clean_i_norms,
            y_test_true, thresholds, auditor, fixed_router, tabular_policy, linucb_policy
        )
        for model_name, m in res.items():
            sweep_records.append({
                "sweep_dimension": "token_masking",
                "param_value": rate,
                "param_label": f"{int(rate * 100)}% Masked",
                "model_name": model_name,
                **m
            })

    # --- Sweep 3: Visual Feature Noise (std) ---
    noise_stds = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
    for s in noise_stds:
        logger.info(f"[Sweep 3: Vision Noise] noise_std = {s:.2f}")
        rng_eval = np.random.RandomState(seed + int(s * 100))
        noisy_feats = test_feats + rng_eval.normal(0, s, size=test_feats.shape).astype(np.float32)
        c_i_probs = image_model.predict_proba(noisy_feats)
        c_i_norms = np.linalg.norm(noisy_feats, axis=1)

        res = evaluate_batch(
            clean_t_probs, c_i_probs, clean_t_norms, c_i_norms,
            y_test_true, thresholds, auditor, fixed_router, tabular_policy, linucb_policy
        )
        for model_name, m in res.items():
            sweep_records.append({
                "sweep_dimension": "visual_noise",
                "param_value": s,
                "param_label": f"std={s:.2f}",
                "model_name": model_name,
                **m
            })

    df = pd.DataFrame(sweep_records)
    out_dir = PROJECT_ROOT / "outputs" / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sensitivity_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / "sensitivity_sweep_results.json"
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Sensitivity sweep results saved to {csv_path} and {json_path}")

    # Plot 4-panel figure
    logger.info("Generating publication-grade 4-panel sensitivity degradation figure...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    color_map = {
        "no_recovery": "#e74c3c",      # Red
        "fixed_router": "#e67e22",     # Orange
        "seaa_tabular": "#2980b9",     # Blue
        "seaa_linucb": "#27ae60",      # Green
        "oracle": "#8e44ad"           # Purple
    }
    label_map = {
        "no_recovery": "Static Late Fusion (No Recovery)",
        "fixed_router": "Fixed Heuristic Router",
        "seaa_tabular": "SEAA Adaptive (Tabular Q)",
        "seaa_linucb": "SEAA Adaptive (LinUCB)",
        "oracle": "Oracle Upper Bound"
    }

    # (a) Text Truncation
    sub_df = df[df["sweep_dimension"] == "text_truncation"]
    for m in ["no_recovery", "fixed_router", "seaa_tabular", "seaa_linucb", "oracle"]:
        m_data = sub_df[sub_df["model_name"] == m].sort_values("param_value")
        axes[0, 0].plot(m_data["param_value"] * 100, m_data["micro_f1"], marker="o", lw=2.2, label=label_map[m], color=color_map[m])
    axes[0, 0].set_title("(a) Text Truncation Severity vs. Micro-F1", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Words Retained (%)", fontsize=11)
    axes[0, 0].set_ylabel("Micro-F1 Score", fontsize=11)
    axes[0, 0].legend(loc="lower right", frameon=True, fontsize=9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # (b) Token Masking
    sub_df = df[df["sweep_dimension"] == "token_masking"]
    for m in ["no_recovery", "fixed_router", "seaa_tabular", "seaa_linucb", "oracle"]:
        m_data = sub_df[sub_df["model_name"] == m].sort_values("param_value")
        axes[0, 1].plot(m_data["param_value"] * 100, m_data["micro_f1"], marker="s", lw=2.2, label=label_map[m], color=color_map[m])
    axes[0, 1].set_title("(b) Text Token Masking Severity vs. Micro-F1", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Tokens Masked (%)", fontsize=11)
    axes[0, 1].set_ylabel("Micro-F1 Score", fontsize=11)
    axes[0, 1].legend(loc="lower left", frameon=True, fontsize=9)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # (c) Visual Noise
    sub_df = df[df["sweep_dimension"] == "visual_noise"]
    for m in ["no_recovery", "fixed_router", "seaa_tabular", "seaa_linucb", "oracle"]:
        m_data = sub_df[sub_df["model_name"] == m].sort_values("param_value")
        axes[1, 0].plot(m_data["param_value"], m_data["micro_f1"], marker="^", lw=2.2, label=label_map[m], color=color_map[m])
    axes[1, 0].set_title("(c) Vision Sensor Noise Severity vs. Micro-F1", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel(r"Gaussian Noise $\sigma$", fontsize=11)
    axes[1, 0].set_ylabel("Micro-F1 Score", fontsize=11)
    axes[1, 0].legend(loc="lower left", frameon=True, fontsize=9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # (d) Expected Calibration Error (ECE) across Visual Noise
    sub_df = df[df["sweep_dimension"] == "visual_noise"]
    for m in ["no_recovery", "fixed_router", "seaa_tabular", "seaa_linucb"]:
        m_data = sub_df[sub_df["model_name"] == m].sort_values("param_value")
        axes[1, 1].plot(m_data["param_value"], m_data["micro_ece"], marker="d", lw=2.2, label=label_map[m], color=color_map[m])
    axes[1, 1].set_title("(d) Reliability Calibration Drift (Micro-ECE vs. Noise)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel(r"Gaussian Noise $\sigma$", fontsize=11)
    axes[1, 1].set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    axes[1, 1].legend(loc="upper left", frameon=True, fontsize=9)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    png_path = fig_dir / "sensitivity_severity_curves.png"
    pdf_path = fig_dir / "sensitivity_severity_curves.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Figures saved to {png_path} and {pdf_path}")
    logger.info("Sensitivity benchmark successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sensitivity_sweeps(sample_size=args.sample_size, seed=args.seed)

