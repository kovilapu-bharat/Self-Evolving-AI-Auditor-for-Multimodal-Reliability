"""
SEAA Latency & Computational Efficiency Benchmark.

Measures precise per-sample latency across all inference, auditing, and recovery stages
under a standardized, reproducible timing protocol:
- Hardware and software telemetry logging.
- JIT and CPU cache warm-up (50 iterations).
- High-resolution timing (time.perf_counter_ns) over N=1,000 independent single-sample runs.
- Full distribution metrics: mean, std, median (p50), p90, p95, p99, min, max, throughput.
- Stage-by-stage breakdown:
    1. Text Baseline Branch (TF-IDF vectorizer + Logistic Regression)
    2. Image Baseline Branch (Linear Classifier on 2048-dim embedding)
    3. Static Multimodal Late Fusion
    4. Auditor Disagreement & Uncertainty Signals (Bernoulli JSD + Entropies + Norms)
    5. Failure Diagnosis Module
    6. Contextual Decision Policy (Adaptive Action Selection)
    7. Recovery Action Execution (Dynamic Alpha Reweighting / Routing)
    8. Total SEAA Auditing + Recovery Overhead (Stages 4 + 5 + 6 + 7)
    9. Complete End-to-End Audited Multimodal Pipeline (Stages 1..7)

Generates publication-ready figures and machine-readable JSON/CSV reports.
"""

import sys
import os
import time
import json
import logging
import platform
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import GENRES
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel
from src.auditing.diagnosis import calibrate_diagnostic_thresholds_on_val, MultimodalReliabilityAuditor
from src.policy.adaptive_policy import (
    AdaptiveRecoveryPolicy,
    PolicyContext,
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_RELIABILITY_WEIGHTED
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("LatencyBenchmark")


def benchmark_efficiency(num_runs: int = 1000, warmup_runs: int = 50):
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    ckpt_dir = base_dir / "outputs" / "checkpoints"
    recovery_dir = base_dir / "outputs" / "recovery"
    if not recovery_dir.exists():
        recovery_dir = base_dir / "outputs" / "phase5"
    figures_dir = recovery_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"SEAA LATENCY & COMPUTATIONAL EFFICIENCY BENCHMARK (N={num_runs})")
    logger.info("=" * 70)

    # 1. Environment Details
    env_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "timer": "time.perf_counter_ns",
        "warmup_iterations": warmup_runs,
        "timed_iterations": num_runs,
        "evaluation_mode": "Single-sample inference (batch_size=1)"
    }
    logger.info(f"OS/Hardware: {env_info['platform']} | {env_info['processor']}")
    logger.info(f"Python: {env_info['python_version']} | Warmup: {warmup_runs} | Iterations: {num_runs}")

    # 2. Load Checkpoints
    text_model = TextBaselineModel.load(ckpt_dir / "text_baseline.joblib")
    image_model = ImageBaselineModel.load(ckpt_dir / "image_baseline.joblib")
    mm_model = MultimodalBaselineModel.load(ckpt_dir / "multimodal_baseline.json")

    meta_df = pd.read_csv(data_dir / "metadata.csv")
    with open(data_dir / "split_indices.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    feats_path = data_dir / "processed" / "resnet50_features.npy"
    all_cached_feats = np.memmap(feats_path, dtype="float32", mode="r", shape=(25959, 2048))

    # Initialize Auditor
    val_indices = splits["val"][:500]
    val_cached_feats = np.asarray(all_cached_feats[val_indices])
    val_texts = [str(t) if pd.notnull(t) else "" for t in meta_df.iloc[val_indices]["text"]]
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

    trained_policy_path = recovery_dir / "run_20261025_phase5_adaptive_recovery" / "trained_adaptive_policy.json"
    policy = AdaptiveRecoveryPolicy.load_policy(trained_policy_path)
    policy.learning_enabled = False

    # Sample inputs from test set
    test_idx = splits["test"][0]
    sample_text = str(meta_df.iloc[test_idx]["text"])
    sample_feat = np.asarray(all_cached_feats[test_idx]).reshape(1, -1)
    single_feat_1d = sample_feat[0]

    # Warmup
    logger.info("Executing warmup iterations...")
    for _ in range(warmup_runs):
        _ = text_model.predict_proba([sample_text])
        _ = image_model.predict_proba(sample_feat)
        _ = auditor.audit_batch(
            text_probs=val_t_probs[:1],
            image_probs=val_i_probs[:1],
            fused_probs=val_t_probs[:1],
            text_feature_norms=val_t_norms[:1],
            image_feature_norms=val_i_norms[:1]
        )
        ctx = PolicyContext(has_text=True, has_image=True, text_entropy=0.3, image_entropy=0.4, bernoulli_jsd=0.08)
        act = policy.select_action(ctx, explore=False)
        _ = policy.apply_action(act, val_t_probs[0], val_i_probs[0], ctx)

    logger.info("Warmup complete. Starting high-precision latency profiling...")

    # Timers (in microseconds)
    times_text = np.zeros(num_runs, dtype=np.float64)
    times_image = np.zeros(num_runs, dtype=np.float64)
    times_fusion = np.zeros(num_runs, dtype=np.float64)
    times_audit = np.zeros(num_runs, dtype=np.float64)
    times_policy = np.zeros(num_runs, dtype=np.float64)
    times_recovery = np.zeros(num_runs, dtype=np.float64)
    times_overhead = np.zeros(num_runs, dtype=np.float64)
    times_e2e = np.zeros(num_runs, dtype=np.float64)

    for i in range(num_runs):
        # 1. Total E2E Start
        t0_e2e = time.perf_counter_ns()

        # Stage 1: Text Baseline Forward Pass
        t0 = time.perf_counter_ns()
        t_prob = text_model.predict_proba([sample_text])
        t_sparse_single = text_model.vectorizer.transform([sample_text])
        t_norm = float(np.sqrt(t_sparse_single.power(2).sum()))
        times_text[i] = (time.perf_counter_ns() - t0) / 1000.0  # to microseconds

        # Stage 2: Image Classifier Forward Pass
        t0 = time.perf_counter_ns()
        i_prob = image_model.predict_proba(sample_feat)
        i_norm = float(np.linalg.norm(single_feat_1d))
        times_image[i] = (time.perf_counter_ns() - t0) / 1000.0

        # Stage 3: Static Late Fusion
        t0 = time.perf_counter_ns()
        fused_prob = 0.55 * t_prob + 0.45 * i_prob
        times_fusion[i] = (time.perf_counter_ns() - t0) / 1000.0

        # SEAA Overhead Start
        t0_ovhd = time.perf_counter_ns()

        # Stage 4: Auditor Signal Extraction & Diagnosis
        t0 = time.perf_counter_ns()
        audit_res = auditor.audit_batch(
            text_probs=t_prob,
            image_probs=i_prob,
            fused_probs=fused_prob,
            text_feature_norms=np.array([t_norm]),
            image_feature_norms=np.array([i_norm])
        )
        diag = str(audit_res["diagnoses"][0])
        t_ent = float(audit_res["text_uncertainty"]["mean_binary_entropy"][0])
        i_ent = float(audit_res["image_uncertainty"]["mean_binary_entropy"][0])
        jsd = float(audit_res["disagreement"]["mean_bernoulli_jsd"][0])
        times_audit[i] = (time.perf_counter_ns() - t0) / 1000.0

        # Stage 5: Contextual Policy Decision
        t0 = time.perf_counter_ns()
        ctx = PolicyContext(
            has_text=True, has_image=True,
            text_entropy=t_ent, image_entropy=i_ent,
            bernoulli_jsd=jsd, text_norm=t_norm,
            image_norm=i_norm, diagnosis=diag
        )
        action = policy.select_action(ctx, explore=False)
        times_policy[i] = (time.perf_counter_ns() - t0) / 1000.0

        # Stage 6: Recovery Execution
        t0 = time.perf_counter_ns()
        rec_prob, _ = policy.apply_action(action, t_prob[0], i_prob[0], ctx)
        times_recovery[i] = (time.perf_counter_ns() - t0) / 1000.0

        # Overhead Total
        times_overhead[i] = (time.perf_counter_ns() - t0_ovhd) / 1000.0

        # End-to-End Total
        times_e2e[i] = (time.perf_counter_ns() - t0_e2e) / 1000.0

    stages = [
        ("Text Branch Inference", times_text),
        ("Image Branch Inference (Cached)", times_image),
        ("Static Late Fusion", times_fusion),
        ("Auditor & Diagnosis Module", times_audit),
        ("Contextual Policy Decision", times_policy),
        ("Recovery Action Execution", times_recovery),
        ("SEAA Total Audit & Recovery Overhead", times_overhead),
        ("Complete End-to-End Audited System", times_e2e)
    ]

    def compute_stats(arr: np.ndarray) -> Dict[str, float]:
        # arr is in microseconds
        mean_us = float(np.mean(arr))
        mean_ms = mean_us / 1000.0
        return {
            "mean_ms": mean_ms,
            "mean_us": mean_us,
            "std_us": float(np.std(arr)),
            "median_us": float(np.median(arr)),
            "p90_us": float(np.percentile(arr, 90)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "min_us": float(np.min(arr)),
            "max_us": float(np.max(arr)),
            "throughput_samples_per_sec": float(1_000_000.0 / mean_us) if mean_us > 0 else 0.0
        }

    results = {}
    table_rows = []

    logger.info("\n" + "=" * 110)
    logger.info(f"{'Pipeline Stage':<38} | {'Mean (ms)':<10} | {'Median (μs)':<11} | {'p95 (μs)':<9} | {'p99 (μs)':<9} | {'Throughput (samp/s)':<19}")
    logger.info("-" * 110)

    for stage_name, time_arr in stages:
        stats = compute_stats(time_arr)
        results[stage_name] = stats
        table_rows.append({
            "Stage": stage_name,
            "Mean_ms": stats["mean_ms"],
            "Mean_us": stats["mean_us"],
            "Std_us": stats["std_us"],
            "Median_us": stats["median_us"],
            "p90_us": stats["p90_us"],
            "p95_us": stats["p95_us"],
            "p99_us": stats["p99_us"],
            "Min_us": stats["min_us"],
            "Max_us": stats["max_us"],
            "Throughput": stats["throughput_samples_per_sec"]
        })
        logger.info(f"{stage_name:<38} | {stats['mean_ms']:<10.4f} | {stats['median_us']:<11.1f} | {stats['p95_us']:<9.1f} | {stats['p99_us']:<9.1f} | {stats['throughput_samples_per_sec']:<19.0f}")

    logger.info("=" * 110)

    # Save to CSV and JSON
    df_lat = pd.DataFrame(table_rows)
    csv_path = recovery_dir / "latency_benchmark_results.csv"
    df_lat.to_csv(csv_path, index=False)

    json_payload = {
        "environment": env_info,
        "stages": results
    }
    json_path = recovery_dir / "latency_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    logger.info(f"Saved latency reports to {csv_path} and {json_path}")

    # Plot Latency Breakdown
    plt.figure(figsize=(10, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    sub_stages = [
        "Auditor & Diagnosis Module",
        "Contextual Policy Decision",
        "Recovery Action Execution"
    ]
    sub_means = [results[s]["mean_us"] for s in sub_stages]
    sub_labels = [
        "Audit & Diagnosis\n(Signal Extraction)",
        "Adaptive Policy\n(Context & Q-Lookup)",
        "Recovery Application\n(Dynamic Alpha)"
    ]

    colors = ["#2b83ba", "#abdda4", "#fdae61"]
    bars = plt.bar(sub_labels, sub_means, color=colors, edgecolor="#333333", width=0.55)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.2f} μs", ha="center", va="bottom", fontsize=10, fontweight="bold")

    total_ovhd = results["SEAA Total Audit & Recovery Overhead"]["mean_us"]
    plt.title(f"SEAA Runtime Overhead Breakdown (Total Mean: {total_ovhd:.2f} μs / {total_ovhd/1000.0:.4f} ms per sample)", fontsize=11.5, fontweight="bold", pad=15)
    plt.ylabel("Execution Time per Sample (Microseconds, μs)", fontsize=10.5, fontweight="semibold")
    plt.ylim(0, max(sub_means) * 1.35)
    plt.xticks(fontsize=10)
    plt.tight_layout()

    fig_png = figures_dir / "latency_overhead_breakdown.png"
    fig_pdf = figures_dir / "latency_overhead_breakdown.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info(f"Saved latency breakdown figures to {fig_png} and {fig_pdf}")

if __name__ == "__main__":
    benchmark_efficiency()
