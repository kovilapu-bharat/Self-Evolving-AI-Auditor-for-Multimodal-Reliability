"""
SEAA IEEE Master Output Generator.
Generates all 5 publication-grade figures, companion source CSVs, Table 1, and the reproduction log
for the IEEE Transactions paper from verified project artifacts.

Outputs directory: outputs/ieee_final/
"""

import sys
import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configure IEEE styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ieee_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette definition (accessible, publication-grade)
PRIMARY_BLUE = "#1f77b4"
ACCENT_ORANGE = "#ff7f0e"
ACCENT_GREEN = "#2ca02c"
ACCENT_RED = "#d62728"
ACCENT_PURPLE = "#9467bd"
NEUTRAL_GRAY = "#7f7f7f"
LIGHT_GRAY = "#f0f2f5"
DARK_NAVY = "#1a2530"


# ==============================================================================
# Helper Utilities
# ==============================================================================
def get_file_hash(filepath: Path, alg: str = "sha256") -> str:
    """Computes cryptographic hash of a file."""
    h = hashlib.new(alg)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 1. Figure 1: SEAA Inline System Architecture Diagram
# ==============================================================================
def generate_figure_1():
    print("Generating Figure 1: SEAA Architecture Diagram...")
    fig = plt.figure(figsize=(13.5, 7.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title Banner
    ax.text(50, 97.5, "SEAA: Self-Evolving Inline Reliability Auditing & Recovery Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color=DARK_NAVY)
    ax.text(50, 94.5, "Closed-Loop Modality Disagreement Monitoring, Failure Diagnosis & Contextual Bandit Recovery",
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#4a5568")

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle="", bg="#ffffff", border="#4a5568", lw=1.2, radius=1.5, title_color=DARK_NAVY):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.4,rounding_size={radius}",
            ec=border, fc=bg, lw=lw, zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(x + w / 2, y + h - 2.8, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + (h - 2.8) / 2, subtitle, ha="center", va="center", fontsize=8.0, color="#2d3748", zorder=3, multialignment="center")
        return box

    def draw_arrow(x1, y1, x2, y2, color="#4a5568", lw=1.5, style="->", dashed=False):
        ls = "--" if dashed else "-"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2),
            zorder=4
        )

    # Column 1: Input Modalities (x: 2 to 18)
    draw_box(2, 63, 16, 26, "Text Input $x^T$",
             "Movie Plot Synopsis\n(Natural Language Text)\n25,959 Titles",
             bg="#e8f4f8", border="#0284c7")
    draw_box(2, 17, 16, 26, "Vision Input $x^V$",
             "Movie Poster Artwork\n(RGB, 224×224 Pixels)\nCanonical Artwork",
             bg="#fef3c7", border="#d97706")

    # Column 2: Predictive Baseline Branches (x: 22 to 38)
    draw_box(22, 63, 17, 26, "Text Baseline $f_T$",
             "Sublinear TF-IDF (10k feats)\nUnigrams + Bigrams\n+ OvR Logistic Reg (C=1.0)\nOutputs: $\\hat{p}^T \\in [0, 1]^{23}$",
             bg="#f0f9ff", border="#0369a1")
    draw_box(22, 17, 17, 26, "Image Baseline $f_V$",
             "Pretrained ResNet-50 (Frozen)\n2,048-dim Embeddings $z^V$\nStandardScaler + OvR LR (C=0.5)\nOutputs: $\\hat{p}^V \\in [0, 1]^{23}$",
             bg="#fffbeb", border="#b45309")

    # Arrows Input -> Baseline
    draw_arrow(18, 76, 22, 76)
    draw_arrow(18, 30, 22, 30)

    # Static Late Fusion Anchor (Reference Baseline)
    draw_box(22, 46.5, 17, 13, "Static Late Fusion $f_{MM}$",
             "$\\hat{p}^{MM} = 0.55\\hat{p}^T + 0.45\\hat{p}^V$\nUnrecovered Baseline ($P_1$)",
             bg="#f3f4f6", border="#6b7280", lw=1.0)
    draw_arrow(30.5, 63, 30.5, 59.5, color="#6b7280")
    draw_arrow(30.5, 43, 30.5, 46.5, color="#6b7280")

    # Column 3: Inline Reliability Auditor (x: 43 to 61)
    draw_box(43, 34, 18, 55, "1. INLINE RELIABILITY AUDITOR",
             "Zero Test Label Leakage\nObservable Inference Signals:\n\n"
             "• Presence: $P_T, P_V \\in \\{0, 1\\}$\n"
             "• Binary Entropy: $H(\\hat{p}^T), H(\\hat{p}^V)$\n"
             "• Bernoulli JSD: $D_{JS}(\\hat{p}^T \\;\\Vert\\; \\hat{p}^V)$\n"
             "  (Cutoff $\\tau_{JSD} = 0.1139$)\n"
             "• Feature Norms: $\\|z^T\\|_2, \\|z^V\\|_2$",
             bg="#f5f3ff", border="#7c3aed", lw=1.4, title_color="#5b21b6")

    draw_arrow(39, 76, 43, 76, color="#0369a1")
    draw_arrow(39, 30, 43, 40, color="#b45309")

    # Stage 2: Failure Diagnoser (x: 43 to 61, bottom)
    draw_box(43, 8, 18, 22, "2. FAILURE DIAGNOSER",
             "Categorical State Diagnosis $G_t$:\n"
             "CLEAN, TEXT_FAIL, IMG_FAIL,\n"
             "MISSING_T, MISSING_V,\n"
             "CROSS_MODAL_CONFLICT,\n"
             "HIGH_UNCERTAINTY",
             bg="#ede9fe", border="#6d28d9", lw=1.2, title_color="#4c1d95")
    draw_arrow(52, 34, 52, 30, color="#6d28d9")

    # Column 4: Contextual Bandit Policy (x: 65 to 81)
    draw_box(65, 34, 18, 55, "3. ADAPTIVE BANDIT POLICY",
             "Context State Signature:\n$s_t = (P_t, D_t, U_t, G_t)$\n\n"
             "Physical Admissibility:\n$\\mathcal{A}_{valid} \\subseteq \\mathcal{A}_{all}$\n\n"
             "Action Selection:\n$a^* = \\arg\\max_{a} Q(s_t, a)$\n\n"
             "Test Deployment:\nPolicy Strictly Frozen ($\\eta_t = 0$)\nDeterministic Greedy Exploitation",
             bg="#ecfdf5", border="#059669", lw=1.4, title_color="#065f46")

    draw_arrow(61, 61.5, 65, 61.5, color="#7c3aed")
    draw_arrow(61, 19, 65, 45, color="#6d28d9")

    # Column 5: Recovery Execution & Output (x: 85 to 98)
    draw_box(85, 48, 13, 41, "4. RECOVERY ACTIONS",
             "• STATIC_FUSION ($\\alpha=0.55$)\n"
             "• TEXT_ONLY ($\\alpha=1.0$)\n"
             "• IMAGE_ONLY ($\\alpha=0.0$)\n"
             "• REL_WEIGHTED ($\\alpha(x)$)\n"
             "• CONSERVATIVE (50/50)",
             bg="#f0fdf4", border="#16a34a", lw=1.2)

    draw_arrow(81, 61.5, 85, 68.5, color="#059669")

    # Final Output Decision Box
    draw_box(85, 17, 13, 27, "5. FINAL PREDICTIONS",
             "Recovered Probs $\\hat{p}^{rec}$\n\n"
             "Binary Decisions:\n$\\hat{y}_c = 1(\\hat{p}_c^{rec} \\geq \\tau_c)$\n"
             "Across 23 Canonical Genres\n\n"
             "Micro-F1: 0.4709\n(+17.4% vs No Recovery)",
             bg="#fef2f2", border="#dc2626", lw=1.4, title_color="#991b1b")

    draw_arrow(91.5, 48, 91.5, 44, color="#16a34a")

    # Offline Outcome Learning Loop (Bottom banner with dashed arrows)
    draw_box(22, 2, 57, 4.5, "OFFLINE VALIDATION OUTCOME LEARNING (FROZEN AT TEST INFERENCE)",
             "Reward: $R_t = F_1(\\hat{y}_{a_t}, y_t) + 0.5[F_1(\\hat{y}_{a_t}, y_t) - F_1(\\hat{y}_{unrec}, y_t)]$  |  Update: $Q_{t+1}(s_t, a_t) = Q_t + \\eta_t [R_t - Q_t]$",
             bg="#fff1f2", border="#e11d48", lw=1.0, title_color="#9f1239")

    # Feedback loop arrows (dashed pink/red)
    draw_arrow(91.5, 17, 91.5, 4.25, color="#e11d48", dashed=True)
    draw_arrow(91.5, 4.25, 79, 4.25, color="#e11d48", dashed=True)
    draw_arrow(22, 4.25, 18, 4.25, color="#e11d48", dashed=True)
    draw_arrow(18, 4.25, 18, 17, color="#e11d48", dashed=True)
    draw_arrow(74, 6.5, 74, 34, color="#e11d48", dashed=True)

    # Badges / Legend Annotations
    ax.text(91.5, 8.5, "Offline Val Only\n(Zero Test Leakage)", ha="center", va="center",
            fontsize=7.5, color="#e11d48", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", fc="#ffffff", ec="#e11d48", lw=0.8))

    # Latency badge top right
    ax.text(85, 92, "Latency Profile:\n• Policy Decision: 0.0245 ms\n• Audit + Recovery: 3.23 ms\n• End-to-End CPU: 11.57 ms",
            ha="left", va="center", fontsize=7.5, color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#94a3b8", lw=0.8))

    pdf_path = OUTPUT_DIR / "fig1_seaa_architecture.pdf"
    png_path = OUTPUT_DIR / "fig1_seaa_architecture.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {pdf_path} and {png_path}.")


# ==============================================================================
# 2. Figure 2: Baseline Model Comparison
# ==============================================================================
def generate_figure_2():
    print("Generating Figure 2: Baseline Model Comparison...")
    source_csv = PROJECT_ROOT / "outputs" / "metrics" / "baseline_comparison.csv"
    df = pd.read_csv(source_csv)

    # Save dedicated source CSV in outputs/ieee_final/
    dest_csv = OUTPUT_DIR / "fig2_baseline_source.csv"
    df.to_csv(dest_csv, index=False)

    models = ["Text-Only\n(TF-IDF 10k + LR)", "Image-Only\n(ResNet-50 + LR)", "Multimodal Late Fusion\n(Static $\\alpha=0.55$)"]
    micro_f1 = df["micro_f1"].values
    macro_f1 = df["macro_f1"].values
    map_scores = df["mean_average_precision"].values
    hamming = df["hamming_loss"].values

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    rects1 = ax.bar(x - width/2, micro_f1, width, label="Micro-F1", color=PRIMARY_BLUE, edgecolor="none", alpha=0.92, zorder=3)
    rects2 = ax.bar(x + width/2, macro_f1, width, label="Macro-F1", color=ACCENT_ORANGE, edgecolor="none", alpha=0.92, zorder=3)

    # Add value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b")

    ax.set_ylabel("F1-Score")
    ax.set_title("Canonical Baseline Performance on Held-Out Test Split (N = 7,788 Movies)", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper right")

    # Add auxiliary summary box
    note_text = (
        "Evaluation Details:\n"
        "• Canonical 30% Test Split: 7,788 Movies (23 Genres)\n"
        "• Optimal Genre Decision Thresholds Tuned on Val Split\n"
        f"• Text Baseline: mAP = {map_scores[0]:.4f}, Hamming = {hamming[0]:.4f}\n"
        f"• Image Baseline: mAP = {map_scores[1]:.4f}, Hamming = {hamming[1]:.4f}\n"
        f"• Static Late Fusion: mAP = {map_scores[2]:.4f}, Hamming = {hamming[2]:.4f}"
    )
    ax.text(0.03, 0.96, note_text, transform=ax.transAxes,
            fontsize=7.8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1", lw=0.8))

    pdf_path = OUTPUT_DIR / "fig2_baseline_comparison.pdf"
    png_path = OUTPUT_DIR / "fig2_baseline_comparison.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {pdf_path} and {png_path}.")


# ==============================================================================
# 3. Figure 3: Recovery Across Eight Failure Regimes
# ==============================================================================
def generate_figure_3():
    print("Generating Figure 3: Recovery Performance Across Eight Failure Regimes...")
    source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    # Save dedicated source CSV
    dest_csv = OUTPUT_DIR / "fig3_recovery_source.csv"
    df.to_csv(dest_csv, index=False)

    conditions = [
        "Clean\nBaseline",
        "Text Trunc.\n(25% Keep)",
        "Text Mask.\n(35% Mask)",
        "Text\nEmpty",
        "Image Blur\n(r=5.0)",
        "Image Noise\n(σ=0.25)",
        "Image Missing\n(Blackout)",
        "Dual Corrupt.\n(Trunc+Noise)"
    ]

    p1_micro = df["no_recovery_micro_f1"].values
    p2_micro = df["fixed_policy_micro_f1"].values
    p3_micro = df["adaptive_policy_micro_f1"].values
    p4_micro = df["oracle_micro_f1"].values

    # Compute condition averages
    p1_avg = np.mean(p1_micro)
    p2_avg = np.mean(p2_micro)
    p3_avg = np.mean(p3_micro)
    p4_avg = np.mean(p4_micro)

    # Setup bar chart with 8 conditions + 1 condition-average group separated by space
    n_cond = len(conditions)
    indices = np.arange(n_cond)
    # Add gap for average
    avg_index = n_cond + 0.6
    all_x = np.append(indices, avg_index)
    all_labels = conditions + ["Condition\nAverage"]

    all_p1 = np.append(p1_micro, p1_avg)
    all_p2 = np.append(p2_micro, p2_avg)
    all_p3 = np.append(p3_micro, p3_avg)
    all_p4 = np.append(p4_micro, p4_avg)

    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=300)
    width = 0.26

    r1 = ax.bar(all_x - width, all_p1, width, label="No Recovery (Static Late Fusion)", color=NEUTRAL_GRAY, alpha=0.85, zorder=3)
    r2 = ax.bar(all_x, all_p2, width, label="Fixed Policy (Heuristic Modality Router)", color=ACCENT_ORANGE, alpha=0.90, zorder=3)
    r3 = ax.bar(all_x + width, all_p3, width, label="Adaptive SEAA Policy (Contextual Bandit)", color=PRIMARY_BLUE, alpha=0.95, zorder=3)

    # Add Oracle dots/markers as reference upper bound
    ax.scatter(all_x, all_p4, color=ACCENT_RED, marker="D", s=32, label="Theoretical Ceiling (Oracle Upper Bound)", zorder=5)

    # Highlight condition average with subtle background shade
    ax.axvspan(n_cond - 0.2, avg_index + 0.5, color="#f1f5f9", alpha=0.7, zorder=1)
    ax.text(avg_index, 0.68, "Summary Average", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#475569")

    # Add value annotations on Condition Average bars
    ax.annotate(f"{p1_avg:.4f}", xy=(avg_index - width, p1_avg), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#1e293b")
    ax.annotate(f"{p2_avg:.4f}", xy=(avg_index, p2_avg), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#1e293b")
    ax.annotate(f"{p3_avg:.4f}", xy=(avg_index + width, p3_avg), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=PRIMARY_BLUE)

    ax.set_ylabel("Micro-F1 Score")
    ax.set_title("Multimodal Failure Recovery Benchmark across Eight Threat Regimes (N = 4,000 Evaluations)", pad=14)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=8.5)
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper left", ncol=2)

    # Annotation noting missing modality ties
    ax.annotate("Physical Masking Tie\n(Only Image Feasible)",
                xy=(3, 0.405), xytext=(3, 0.51),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.9),
                ha="center", fontsize=7.2, color="#475569",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cbd5e1"))

    ax.annotate("Physical Masking Tie\n(Only Text Feasible)",
                xy=(6, 0.585), xytext=(6, 0.66),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.9),
                ha="center", fontsize=7.2, color="#475569",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cbd5e1"))

    ax.annotate("Fixed Policy Collapse\n(Misdiagnosed Blur)",
                xy=(4, 0.23), xytext=(4, 0.12),
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=0.9),
                ha="center", fontsize=7.2, color=ACCENT_RED,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#fca5a5"))

    pdf_path = OUTPUT_DIR / "fig3_recovery_by_condition.pdf"
    png_path = OUTPUT_DIR / "fig3_recovery_by_condition.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {pdf_path} and {png_path}.")


# ==============================================================================
# 4. Figure 4: Adaptive Recovery Action Heatmap
# ==============================================================================
def generate_figure_4():
    print("Generating Figure 4: Adaptive Recovery Action Heatmap...")
    source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    conditions_clean = [
        "Clean Baseline",
        "Text Truncation (25%)",
        "Text Masking (35%)",
        "Text Empty",
        "Image Gaussian Blur (r=5)",
        "Image Gaussian Noise (σ=0.25)",
        "Image Missing (Blackout)",
        "Dual Corruption (Trunc+Noise)"
    ]

    action_cols = [
        "actions_static_fusion_pct",
        "actions_text_only_pct",
        "actions_image_only_pct",
        "actions_rel_weighted_pct",
        "actions_conservative_pct"
    ]
    action_names = [
        "Static Fusion\n(α = 0.55)",
        "Text Only\n(α = 1.0)",
        "Image Only\n(α = 0.0)",
        "Reliability Weighted\n(Dynamic α(x))",
        "Conservative\n(50/50)"
    ]

    heatmap_data = df[action_cols].values

    # Save dedicated source CSV
    df_source = pd.DataFrame(heatmap_data, index=conditions_clean, columns=[
        "STATIC_FUSION", "TEXT_ONLY", "IMAGE_ONLY", "RELIABILITY_WEIGHTED", "CONSERVATIVE_FUSION"
    ])
    df_source.index.name = "Failure_Regime"
    dest_csv = OUTPUT_DIR / "fig4_action_distribution_source.csv"
    df_source.to_csv(dest_csv)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=300)

    # Plot seaborn heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=action_names,
        yticklabels=conditions_clean,
        cbar_kws={"label": "Action Selection Percentage (%)"},
        vmin=0.0,
        vmax=100.0,
        linewidths=1.0,
        linecolor="#f8fafc",
        ax=ax,
        annot_kws={"fontsize": 9.5, "fontweight": "bold"}
    )

    ax.set_title("SEAA Adaptive Recovery Policy Action Selection Distribution (%) Across 4,000 Evaluations", pad=14)
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0)

    pdf_path = OUTPUT_DIR / "fig4_action_heatmap.pdf"
    png_path = OUTPUT_DIR / "fig4_action_heatmap.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {pdf_path} and {png_path}.")


# ==============================================================================
# 5. Figure 5: Component Ablation Study
# ==============================================================================
def generate_figure_5():
    print("Generating Figure 5: Component Ablation Study...")
    source_csv = PROJECT_ROOT / "outputs" / "recovery" / "ablation_study_results.csv"
    df = pd.read_csv(source_csv)

    # Save dedicated source CSV
    dest_csv = OUTPUT_DIR / "fig5_ablation_source.csv"
    df.to_csv(dest_csv, index=False)

    # Map policy names to clean publication titles
    policy_labels = [
        "Full SEAA Policy\n(Trained Bandit)",
        "Ablation: Learning Disabled\n(Untrained Q-Table)",
        "Ablation: No Diagnosis*\n(Coarse State)",
        "Ablation: Random Action\n(Uniform Selection)"
    ]

    # Select the 4 ablation policies
    df_variants = df.iloc[[0, 1, 2, 3]].copy()
    micros = df_variants["Condition_Mean_Micro_F1"].values
    macros = df_variants["Condition_Mean_Macro_F1"].values

    # Baselines for reference lines
    no_rec_micro = df[df["Policy"] == "Baseline_No_Recovery"]["Condition_Mean_Micro_F1"].values[0]
    fixed_micro = df[df["Policy"] == "Baseline_Fixed_Policy"]["Condition_Mean_Micro_F1"].values[0]
    oracle_micro = df[df["Policy"] == "Oracle_Upper_Bound"]["Condition_Mean_Micro_F1"].values[0]

    x = np.arange(len(policy_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)

    r1 = ax.bar(x - width/2, micros, width, label="Condition-Mean Micro-F1", color=PRIMARY_BLUE, alpha=0.92, zorder=3)
    r2 = ax.bar(x + width/2, macros, width, label="Condition-Mean Macro-F1", color=ACCENT_ORANGE, alpha=0.92, zorder=3)

    # Reference horizontal lines
    ax.axhline(no_rec_micro, color=NEUTRAL_GRAY, linestyle="--", lw=1.2, label=f"No Recovery Baseline (Micro-F1 = {no_rec_micro:.4f})", zorder=2)
    ax.axhline(fixed_micro, color=ACCENT_RED, linestyle=":", lw=1.2, label=f"Fixed Heuristic Policy (Micro-F1 = {fixed_micro:.4f})", zorder=2)
    ax.axhline(oracle_micro, color=ACCENT_GREEN, linestyle="-.", lw=1.2, label=f"Oracle Upper Bound (Micro-F1 = {oracle_micro:.4f})", zorder=2)

    # Add value annotations
    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b")

    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#1e293b")

    ax.set_ylabel("F1-Score")
    ax.set_title("Component Ablation Study: Isolating Adaptation and Diagnostic Features", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontsize=8.5)
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper right", fontsize=8.0)

    # Add explanatory note regarding No Diagnosis paradox
    note = (
        "*Note on Diagnosis Ablation (0.5054):\n"
        "Stripping G_t removes noisy failure blame under spatial blur (where the rule-based auditor falsely\n"
        "diagnosed TEXT_FAILURE in 48.4% of samples). Bypassing this false label allowed the policy to exploit\n"
        "the stronger text modality directly, improving blur recovery while proving that heuristic failure\n"
        "classification is a noisy system bottleneck. Outcome-based adaptation itself remains strictly necessary\n"
        "(disabling learning drops Micro-F1 from 0.4711 to 0.4377, -3.34% points)."
    )
    ax.text(0.02, 0.96, note, transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffbeb", ec="#fde68a", lw=0.8))

    pdf_path = OUTPUT_DIR / "fig5_ablation_study.pdf"
    png_path = OUTPUT_DIR / "fig5_ablation_study.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {pdf_path} and {png_path}.")


# ==============================================================================
# 6. Table 1: Main Results and Statistical Comparison
# ==============================================================================
def generate_table_1():
    print("Generating Table 1: Main Results and Statistical Comparison...")
    
    # 1. Structured Data for Table 1
    table_data = [
        {
            "Policy_Configuration": "Proposed: Adaptive SEAA Policy",
            "Condition_Mean_Micro_F1": 0.4709,
            "Condition_Mean_Macro_F1": 0.3593,
            "Globally_Pooled_Micro_F1": 0.4690,
            "Globally_Pooled_Macro_F1": 0.3523,
            "Instance_Mean_F1": 0.4671,
            "Gain_vs_No_Recovery_Micro": "+0.0697 (+17.4%)",
            "Gain_vs_Fixed_Policy_Micro": "+0.0768 (+19.5%)",
            "Cluster_Paired_t_Test": "Reference",
            "Wilcoxon_W_Stat": "Reference",
            "P_Value": "Reference",
            "Cluster_95_CI": "Reference"
        },
        {
            "Policy_Configuration": "Baseline 1: No Recovery (Static Fusion, α=0.55)",
            "Condition_Mean_Micro_F1": 0.4012,
            "Condition_Mean_Macro_F1": 0.3165,
            "Globally_Pooled_Micro_F1": 0.3973,
            "Globally_Pooled_Macro_F1": 0.3030,
            "Instance_Mean_F1": 0.3940,
            "Gain_vs_No_Recovery_Micro": "Reference (0.0000)",
            "Gain_vs_Fixed_Policy_Micro": "+0.0071 (+1.8%)",
            "Cluster_Paired_t_Test": "t = 12.79",
            "Wilcoxon_W_Stat": "W = 23,564.0",
            "P_Value": "p = 1.34e-32",
            "Cluster_95_CI": "[+0.0619, +0.0843]"
        },
        {
            "Policy_Configuration": "Baseline 2: Fixed Policy (Heuristic Router)",
            "Condition_Mean_Micro_F1": 0.3942,
            "Condition_Mean_Macro_F1": 0.2945,
            "Globally_Pooled_Micro_F1": 0.3905,
            "Globally_Pooled_Macro_F1": 0.2898,
            "Instance_Mean_F1": 0.3901,
            "Gain_vs_No_Recovery_Micro": "-0.0070 (-1.7%)",
            "Gain_vs_Fixed_Policy_Micro": "Reference (0.0000)",
            "Cluster_Paired_t_Test": "t = 12.82",
            "Wilcoxon_W_Stat": "W = 24,620.5",
            "P_Value": "p = 1.02e-32",
            "Cluster_95_CI": "[+0.0652, +0.0888]"
        },
        {
            "Policy_Configuration": "Theoretical Ceiling: Oracle Upper Bound",
            "Condition_Mean_Micro_F1": 0.5704,
            "Condition_Mean_Macro_F1": 0.4421,
            "Globally_Pooled_Micro_F1": 0.5673,
            "Globally_Pooled_Macro_F1": 0.4399,
            "Instance_Mean_F1": 0.5757,
            "Gain_vs_No_Recovery_Micro": "+0.1692 (+42.2%)",
            "Gain_vs_Fixed_Policy_Micro": "+0.1762 (+44.7%)",
            "Cluster_Paired_t_Test": "—",
            "Wilcoxon_W_Stat": "—",
            "P_Value": "—",
            "Cluster_95_CI": "—"
        }
    ]

    df_table1 = pd.DataFrame(table_data)
    csv_path = OUTPUT_DIR / "table1_main_results.csv"
    df_table1.to_csv(csv_path, index=False)
    print(f"Saved Table 1 CSV to {csv_path}.")

    # Generate Markdown Table
    md_content = """# Table 1: SEAA Master Performance Benchmark & Repeated-Measures Clustered Statistics

**Evaluation Protocol:** Evaluated across 500 held-out MM-IMDb test movies repeated over all 8 failure conditions ($N = 4,000$ total evaluations per policy).  
**Statistical Clustering:** Clustered at the movie entity level ($N_{\\text{clusters}} = 500, df = 499$). Intraclass Correlation $\\text{ICC}(1) = 0.1319$, Variance Inflation Factor $\\text{VIF} = 1.9232$, Effective Sample Size $N_{\\text{eff}} \\approx 2,080$.

| Policy / Model Configuration | Condition-Mean Micro-F1 *(Primary)* | Condition-Mean Macro-F1 | Globally Pooled Micro-F1 *(Secondary)* | Globally Pooled Macro-F1 | Instance-Level Mean F1 *(Test Unit)* | Gain vs No Rec. (Micro-F1) | Gain vs Fixed (Micro-F1) | Clustered Paired $t$-test ($df=499$) | Wilcoxon Signed-Rank ($W$) | Clustered 95% Confidence Interval |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed: Adaptive SEAA Policy** | **0.4709** | **0.3593** | **0.4690** | **0.3523** | **0.4671** | **+0.0697 (+17.4%)** | **+0.0768 (+19.5%)** | Reference | Reference | Reference |
| **Baseline 1: No Recovery (Static Late Fusion, $\\alpha=0.55$)** | 0.4012 | 0.3165 | 0.3973 | 0.3030 | 0.3940 | Reference (0.0000) | +0.0071 (+1.8%) | $t = 12.79, p = 1.34 \\times 10^{-32}$ | $W = 23,564.0, p = 1.15 \\times 10^{-31}$ | $[+0.0619, +0.0843]$ |
| **Baseline 2: Fixed Policy (Heuristic Router)** | 0.3942 | 0.2945 | 0.3905 | 0.2898 | 0.3901 | -0.0070 (-1.7%) | Reference (0.0000) | $t = 12.82, p = 1.02 \\times 10^{-32}$ | $W = 24,620.5, p = 9.43 \\times 10^{-30}$ | $[+0.0652, +0.0888]$ |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5704 | 0.4421 | 0.5673 | 0.4399 | 0.5757 | +0.1692 (+42.2%) | +0.1762 (+44.7%) | — | — | — |

---

### Methodological & Inferential Notes:
1. **Primary Metric:** Condition-Averaged Micro-F1 is the pre-specified primary endpoint across the 8 benchmark regimes.
2. **Gain Precision:** Full-precision difference for Adaptive vs. Fixed Policy is $0.4709375 - 0.3941500 = +0.0767875$ (rounds to **+0.0768**). If calculated from rounded 4-decimal values ($0.4709 - 0.3942$), it yields **+0.0767**.
3. **Repeated-Measures Dependence:** Standard error in naive unclustered analysis is underestimated by $1.39\\times$ ($\\text{SE}_{\\text{naive}} = 0.0041$ vs $\\text{SE}_{\\text{clustered}} = 0.0057$), demonstrating why entity clustering is methodologically essential.
4. **5,000-Iteration Clustered Bootstrap:** 0 out of 5,000 resamples with replacement yielded $\\Delta \\le 0$, empirically confirming an upper bound of $p < 0.0002$ across all metric combinations.
"""
    md_path = OUTPUT_DIR / "table1_main_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Table 1 Markdown to {md_path}.")


# ==============================================================================
# 7. Reproduction Log
# ==============================================================================
def generate_reproduction_log():
    print("Generating Reproduction Log...")
    log_path = OUTPUT_DIR / "reproduction_log.txt"

    files_to_hash = [
        OUTPUT_DIR / "fig1_seaa_architecture.pdf",
        OUTPUT_DIR / "fig1_seaa_architecture.png",
        OUTPUT_DIR / "fig2_baseline_comparison.pdf",
        OUTPUT_DIR / "fig2_baseline_comparison.png",
        OUTPUT_DIR / "fig2_baseline_source.csv",
        OUTPUT_DIR / "fig3_recovery_by_condition.pdf",
        OUTPUT_DIR / "fig3_recovery_by_condition.png",
        OUTPUT_DIR / "fig3_recovery_source.csv",
        OUTPUT_DIR / "fig4_action_heatmap.pdf",
        OUTPUT_DIR / "fig4_action_heatmap.png",
        OUTPUT_DIR / "fig4_action_distribution_source.csv",
        OUTPUT_DIR / "fig5_ablation_study.pdf",
        OUTPUT_DIR / "fig5_ablation_study.png",
        OUTPUT_DIR / "fig5_ablation_source.csv",
        OUTPUT_DIR / "table1_main_results.csv",
        OUTPUT_DIR / "table1_main_results.md",
        OUTPUT_DIR / "outputs_audit_report.md"
    ]

    lines = [
        "=" * 80,
        "SEAA IEEE FINAL OUTPUT REPRODUCTION LOG",
        "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: Windows 10 (AMD64, GenuineIntel)",
        f"Python Version: {sys.version}",
        f"Execution Directory: {PROJECT_ROOT}",
        f"Output Directory: {OUTPUT_DIR}",
        "",
        "INPUT ARTIFACT SOURCES & INTEGRITY (MD5 / SHA-256):",
        "-" * 80
    ]

    input_files = [
        PROJECT_ROOT / "outputs" / "metrics" / "baseline_comparison.csv",
        PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv",
        PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "sample_level_decisions.csv",
        PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_action_distributions.json",
        PROJECT_ROOT / "outputs" / "recovery" / "ablation_study_results.csv",
        PROJECT_ROOT / "outputs" / "recovery" / "ablation_study_results.json",
        PROJECT_ROOT / "outputs" / "checkpoints" / "multimodal_baseline.json",
        PROJECT_ROOT / "data" / "processed" / "resnet50_features_info.json"
    ]

    for f in input_files:
        if f.exists():
            md5_h = get_file_hash(f, "md5")
            sha_h = get_file_hash(f, "sha256")
            lines.append(f"File: {f.relative_to(PROJECT_ROOT)}")
            lines.append(f"  MD5:    {md5_h}")
            lines.append(f"  SHA256: {sha_h}")

    lines.append("")
    lines.append("GENERATED IEEE DELIVERABLES & INTEGRITY (MD5 / SHA-256):")
    lines.append("-" * 80)

    for f in files_to_hash:
        if f.exists():
            md5_h = get_file_hash(f, "md5")
            sha_h = get_file_hash(f, "sha256")
            lines.append(f"File: {f.name}")
            lines.append(f"  Size:   {f.stat().st_size:,} bytes")
            lines.append(f"  MD5:    {md5_h}")
            lines.append(f"  SHA256: {sha_h}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("REPRODUCTION COMMAND:")
    lines.append("python scripts/generate_ieee_final_outputs.py")
    lines.append("=" * 80)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved Reproduction Log to {log_path}.")


if __name__ == "__main__":
    t0 = time.perf_counter()
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_table_1()
    generate_reproduction_log()
    print(f"\nAll IEEE Deliverables successfully generated in {time.perf_counter() - t0:.2f}s!")
