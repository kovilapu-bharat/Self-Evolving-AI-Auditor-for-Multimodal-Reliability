"""
SEAA Out-of-Distribution (OOD) & Unseen Corruption Generalization Benchmark.

Evaluates zero-shot robustness and inline failure recovery when confronted with
three realistic corruption types NEVER seen during validation policy training:
1. OOD 1: Typographical Keyboard Swap Noise (simulates OCR/data entry error)
2. OOD 2: Salt-and-Pepper Impulse Visual Noise (simulates sensor packet corruption)
3. OOD 3: Extreme Low-Contrast / Illumination Washout (simulates camera exposure failure)

Evaluates:
- No Recovery (Static Late Fusion)
- Fixed Heuristic Router (DynamicModalityGating)
- SEAA Adaptive Recovery Policy (Tabular Q-Learning)
- SEAA LinUCB Continuous Bandit Policy
- Oracle Upper Bound

Calculates Micro-F1, Macro-F1, Micro-ECE, and Brier Score.
Generates publication-grade comparison figures in PNG and PDF.
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

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
logger = logging.getLogger("OODBenchmark")


# --- OOD Corruption Operators ---

QWERTY_NEIGHBORS = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk'
}

def apply_typo_noise(text: str, char_swap_prob: float = 0.20, seed: int = 42) -> str:
    """Randomly mutates characters into neighboring keyboard keys (simulates OCR/typing error)."""
    if not text or not isinstance(text, str):
        return ""
    rng = np.random.RandomState(seed)
    chars = list(text)
    for i in range(len(chars)):
        c_lower = chars[i].lower()
        if c_lower in QWERTY_NEIGHBORS and rng.uniform(0.0, 1.0) < char_swap_prob:
            sub = rng.choice(list(QWERTY_NEIGHBORS[c_lower]))
            chars[i] = sub.upper() if chars[i].isupper() else sub
    return "".join(chars)


def apply_salt_pepper_features(feats: np.ndarray, impulse_prob: float = 0.20, seed: int = 42) -> np.ndarray:
    """Simulates salt-and-pepper packet drop on visual representations."""
    rng = np.random.RandomState(seed)
    corrupted = np.copy(feats)
    mask = rng.uniform(0.0, 1.0, size=feats.shape) < impulse_prob
    # Set corrupted dimensions to zero or extreme values
    corrupted[mask] = 0.0
    return corrupted.astype(np.float32)


def apply_contrast_washout(feats: np.ndarray, scale: float = 0.15, shift: float = 0.5) -> np.ndarray:
    """Simulates extreme contrast flattening / illumination washout."""
    return (feats * scale + shift).astype(np.float32)


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


def run_ood_evaluation(sample_size: int = 500, seed: int = 42):
    logger.info("=" * 80)
    logger.info("SEAA OUT-OF-DISTRIBUTION (OOD) UNSEEN CORRUPTION GENERALIZATION BENCHMARK")
    logger.info("=" * 80)

    ckpts_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    text_model = TextBaselineModel.load(ckpts_dir / "text_baseline.joblib")
    image_model = ImageBaselineModel.load(ckpts_dir / "image_baseline.joblib")
    mm_model = MultimodalBaselineModel.load(ckpts_dir / "multimodal_baseline.json")
    thresholds = mm_model.thresholds

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

    # Clean baselines for comparison
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

    # Load trained Tabular policy
    policy_path = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "trained_adaptive_policy.json"
    if not policy_path.exists():
        policy_path = list((PROJECT_ROOT / "outputs" / "recovery").glob("*/trained_adaptive_policy.json"))[0]
    tabular_policy = AdaptiveRecoveryPolicy()
    tabular_policy.load_policy(policy_path)

    # Train LinUCB on standard validation episodes (strictly without OOD exposure)
    linucb_policy = LinUCBPolicy(d=9, alpha=0.5, seed=seed)
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

    logger.info("Evaluating on 3 Unseen Out-of-Distribution Regimes...")

    ood_conditions = [
        {
            "name": "OOD_1_Typo_Keyboard_Noise",
            "desc": "Text Modality: Keyboard Typo Perturbation (25% char swap)",
            "text_fn": lambda t, i: apply_typo_noise(t, char_swap_prob=0.25, seed=seed + i),
            "image_fn": lambda f, i: f
        },
        {
            "name": "OOD_2_Salt_And_Pepper_Impulse",
            "desc": "Vision Modality: Salt & Pepper Feature Packet Drop (25% impulse mask)",
            "text_fn": lambda t, i: t,
            "image_fn": lambda f, i: apply_salt_pepper_features(f, impulse_prob=0.25, seed=seed + i)
        },
        {
            "name": "OOD_3_Low_Contrast_Washout",
            "desc": "Vision Modality: Low-Contrast Illumination Washout (15% scale + shift)",
            "text_fn": lambda t, i: t,
            "image_fn": lambda f, i: apply_contrast_washout(f, scale=0.15, shift=0.5)
        }
    ]

    records = []

    for cond in ood_conditions:
        c_name = cond["name"]
        logger.info(f"Evaluating {c_name} ({cond['desc']})...")

        # Corrupt text
        c_texts = [cond["text_fn"](t, i) for i, t in enumerate(test_texts)]
        c_t_probs = text_model.predict_proba(c_texts)
        c_t_sparse = text_model.vectorizer.transform(c_texts)
        c_t_norms = np.sqrt(np.asarray(c_t_sparse.power(2).sum(axis=1)).ravel())

        # Corrupt image
        c_feats = np.array([cond["image_fn"](test_feats[i], i) for i in range(len(test_feats))])
        c_i_probs = image_model.predict_proba(c_feats)
        c_i_norms = np.linalg.norm(c_feats, axis=1)

        # Baseline unrecovered late fusion
        p_unrec = 0.55 * c_t_probs + 0.45 * c_i_probs

        # Inline Auditing
        audit_res = auditor.audit_batch(
            text_probs=c_t_probs,
            image_probs=c_i_probs,
            fused_probs=p_unrec,
            text_feature_norms=c_t_norms,
            image_feature_norms=c_i_norms
        )
        diagnoses = audit_res["diagnoses"]
        t_entropies = audit_res["text_uncertainty"]["mean_binary_entropy"]
        i_entropies = audit_res["image_uncertainty"]["mean_binary_entropy"]
        jsds = audit_res["disagreement"]["mean_bernoulli_jsd"]
        hammings = audit_res["disagreement"]["hamming_distance"]

        # Fixed Heuristic Router
        p_fixed_res = fixed_router.recover_predictions(
            text_probs=c_t_probs,
            image_probs=c_i_probs,
            diagnoses=diagnoses,
            text_uncertainties=audit_res["text_uncertainty"]["composite_uncertainty"],
            image_uncertainties=audit_res["image_uncertainty"]["composite_uncertainty"],
            default_fused_probs=p_unrec
        )
        p_fixed = p_fixed_res["recovered_probs"]

        # SEAA Tabular & LinUCB
        p_tabular = np.zeros_like(c_t_probs)
        p_linucb = np.zeros_like(c_t_probs)
        p_oracle = np.zeros_like(c_t_probs)

        for i in range(len(test_texts)):
            ctx = PolicyContext(
                has_text=True, has_image=True,
                text_entropy=float(t_entropies[i]),
                image_entropy=float(i_entropies[i]),
                bernoulli_jsd=float(jsds[i]),
                hamming_distance=float(hammings[i]),
                text_norm=float(c_t_norms[i]),
                image_norm=float(c_i_norms[i]),
                diagnosis=str(diagnoses[i])
            )

            act_tab = tabular_policy.select_action(ctx, explore=False)
            rec_tab, _ = tabular_policy.apply_action(act_tab, c_t_probs[i], c_i_probs[i], ctx)
            p_tabular[i] = rec_tab

            act_lin, _ = linucb_policy.select_action(ctx, frozen=True)
            rec_lin, _ = tabular_policy.apply_action(act_lin, c_t_probs[i], c_i_probs[i], ctx)
            p_linucb[i] = rec_lin

            p_oracle[i] = compute_oracle(c_t_probs[i], c_i_probs[i], y_test_true[i], thresholds, ctx, tabular_policy)

        models = {
            "No Recovery (Static Late Fusion)": p_unrec,
            "Fixed Heuristic Router": p_fixed,
            "SEAA Adaptive (Tabular Q)": p_tabular,
            "SEAA Adaptive (LinUCB)": p_linucb,
            "Oracle Upper Bound": p_oracle
        }

        for m_name, probs in models.items():
            eval_f1 = evaluate_multilabel_predictions(y_test_true, probs, thresholds, GENRES)
            eval_cal = evaluate_multilabel_calibration(y_test_true, probs, n_bins=10)
            records.append({
                "condition_name": c_name,
                "description": cond["desc"],
                "model_name": m_name,
                "micro_f1": eval_f1["micro_f1"],
                "macro_f1": eval_f1["macro_f1"],
                "micro_ece": eval_cal["micro_ece"],
                "brier_score": eval_cal["brier_score"]
            })

    df = pd.DataFrame(records)
    out_dir = PROJECT_ROOT / "outputs" / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ood_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / "ood_benchmark_results.json"
    df.to_json(json_path, orient="records", indent=2)

    logger.info(f"OOD benchmark data saved to {csv_path} and {json_path}")

    # Plot figure
    logger.info("Generating OOD publication-grade comparison chart...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    conditions = [c["name"] for c in ood_conditions]
    x = np.arange(len(conditions))
    width = 0.16

    color_dict = {
        "No Recovery (Static Late Fusion)": "#e74c3c",
        "Fixed Heuristic Router": "#e67e22",
        "SEAA Adaptive (Tabular Q)": "#2980b9",
        "SEAA Adaptive (LinUCB)": "#27ae60",
        "Oracle Upper Bound": "#8e44ad"
    }

    model_names = list(color_dict.keys())
    for i, m_name in enumerate(model_names):
        sub = df[df["model_name"] == m_name]
        f1_vals = [sub[sub["condition_name"] == c]["micro_f1"].values[0] for c in conditions]
        offset = (i - 2) * width
        rects = ax.bar(x + offset, f1_vals, width, label=m_name, color=color_dict[m_name], alpha=0.9, edgecolor="black", linewidth=0.6)
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_ylabel("Micro-F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Zero-Shot Generalization on Unseen Out-of-Distribution (OOD) Regimes", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "OOD 1: Keyboard Typo Noise\n(Text Modality)",
        "OOD 2: Salt & Pepper Drop\n(Vision Feature Mask)",
        "OOD 3: Contrast Washout\n(Visual Exposure Wash)"
    ], fontsize=10)
    ax.set_ylim(0.2, 0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    png_path = fig_dir / "ood_generalization_comparison.png"
    pdf_path = fig_dir / "ood_generalization_comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"OOD Figures saved to {png_path} and {pdf_path}")
    logger.info("OOD benchmark successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ood_evaluation(sample_size=args.sample_size, seed=args.seed)

