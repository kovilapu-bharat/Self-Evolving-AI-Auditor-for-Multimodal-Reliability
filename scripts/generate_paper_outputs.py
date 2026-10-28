"""
SEAA: Self-Evolving AI Auditor for Multimodal Reliability
Master IEEE Paper Output Generator

Generates the exact 6 core deliverables requested for the IEEE research paper:
  Figures (5):
    1. SEAA Architecture Diagram — Complete Monitor -> Diagnose -> Recover -> Evaluate -> Learn pipeline.
    2. Baseline Comparison Graph — Text-only vs Image-only vs Static Fusion, using Micro-F1 and Macro-F1.
    3. Recovery Performance Graph — No Recovery vs Fixed Policy vs Adaptive SEAA across all 8 failure conditions.
    4. Adaptive Recovery Action Heatmap — Distribution of the 5 recovery actions across the 8 failure conditions.
    5. Ablation Study Graph — Full SEAA vs Learning Disabled vs Diagnosis Features Removed.
  Table (1):
    6. Main Results and Statistical Comparison Table — Overall Micro-F1, Macro-F1, improvement over baselines,
       95% confidence intervals, and verified statistical test results (saved as Markdown, LaTeX, CSV, and JSON).

All outputs are written to: outputs/paper_outputs/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def generate_figure1_architecture():
    """Figure 1: SEAA Architecture Diagram - 5-Stage Closed Loop Pipeline."""
    fig, ax = plt.subplots(figsize=(15.5, 9.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_input = "#eaf2f8"
    c_models = "#d4e6f1"
    c_stage1 = "#e8f8f5"  # Mint / Monitor
    c_stage2 = "#fef9e7"  # Light Yellow / Diagnose
    c_stage3 = "#fbeee6"  # Light Coral / Contextual Bandit
    c_stage4 = "#ebf5fb"  # Light Blue / Execution
    c_stage5 = "#f4ecf7"  # Light Purple / Outcome Learning
    border_dark = "#2c3e50"

    # Title Banner
    ax.text(50, 97, "SEAA: Self-Evolving AI Auditor Multimodal Closed-Loop Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color=border_dark)
    ax.text(50, 93.8, "Continuous Inline Reliability Auditing, Failure Diagnosis, Contextual Bandit Recovery, and Outcome Learning",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#555555")

    # --- 0. Inputs & Unimodal Base Models ---
    # Text input box
    rect_text_in = FancyBboxPatch((3, 76), 14, 12, boxstyle="round,pad=0.5", ec=border_dark, fc=c_input, lw=1.2)
    ax.add_patch(rect_text_in)
    ax.text(10, 84, "Movie Synopsis\nText (x^T)", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(10, 78.5, "Raw text string\n(Plot synopsis)", ha="center", va="center", fontsize=7.5, color="#555")

    # Image input box
    rect_img_in = FancyBboxPatch((3, 56), 14, 12, boxstyle="round,pad=0.5", ec=border_dark, fc=c_input, lw=1.2)
    ax.add_patch(rect_img_in)
    ax.text(10, 64, "Movie Poster\nImage (x^V)", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(10, 58.5, "RGB 224x224 poster\nartwork tensor", ha="center", va="center", fontsize=7.5, color="#555")

    # Base Text Model
    rect_text_m = FancyBboxPatch((21, 76), 14, 12, boxstyle="round,pad=0.5", ec=border_dark, fc=c_models, lw=1.2)
    ax.add_patch(rect_text_m)
    ax.text(28, 84.5, "Text Model (f_T)", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(28, 80, "TF-IDF (10k features)\n+ OvR Logistic Reg", ha="center", va="center", fontsize=7.5)
    ax.text(28, 77.2, "p^T in [0, 1]^23", ha="center", va="center", fontsize=8.5, color="#1b4f72", fontweight="bold")

    # Base Image Model
    rect_img_m = FancyBboxPatch((21, 56), 14, 12, boxstyle="round,pad=0.5", ec=border_dark, fc=c_models, lw=1.2)
    ax.add_patch(rect_img_m)
    ax.text(28, 64.5, "Image Model (f_V)", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(28, 60, "ResNet-50 (2048-d)\n+ OvR Logistic Reg", ha="center", va="center", fontsize=7.5)
    ax.text(28, 57.2, "p^V in [0, 1]^23", ha="center", va="center", fontsize=8.5, color="#1b4f72", fontweight="bold")

    # Arrows from Inputs to Models
    ax.annotate("", xy=(21, 82), xytext=(17, 82), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.5))
    ax.annotate("", xy=(21, 62), xytext=(17, 62), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.5))

    # --- 1. Inline Reliability Auditor ---
    rect_stage1 = FancyBboxPatch((38, 52), 17, 38, boxstyle="round,pad=0.6", ec="#117a65", fc=c_stage1, lw=1.6)
    ax.add_patch(rect_stage1)
    ax.text(46.5, 87.5, "STAGE 1: MONITOR", ha="center", va="center", fontsize=10, fontweight="bold", color="#0e6251")
    ax.text(46.5, 84.5, "Inline Reliability Auditor", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#117a65")

    ax.text(46.5, 79, "Modality Presence Flags:\nP_T = 1(x^T != empty)\nP_V = 1(x^V != empty)",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#117a65', lw=0.5))
    ax.text(46.5, 70, "Predictive Binary Entropy:\nH(p) = -(1/C) sum_c [p_c log2 p_c\n       + (1 - p_c) log2(1 - p_c)]",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#117a65', lw=0.5))
    ax.text(46.5, 60.5, "Bernoulli Jensen-Shannon\nDivergence: D_JS(p^T || p^V)\nin [0, 1] (Disagreement)",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#117a65', lw=0.5))
    ax.text(46.5, 54.5, "Feature Norms: ||z^T||_2, ||z^V||_2",
            ha="center", va="center", fontsize=7.5, color="#333", fontweight="semibold")

    # Arrows from Models into Stage 1
    ax.annotate("", xy=(38, 80), xytext=(35, 80), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.5))
    ax.annotate("", xy=(38, 64), xytext=(35, 64), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.5))

    # --- 2. Multi-Class Failure Diagnoser ---
    rect_stage2 = FancyBboxPatch((58, 52), 17, 38, boxstyle="round,pad=0.6", ec="#b7950b", fc=c_stage2, lw=1.6)
    ax.add_patch(rect_stage2)
    ax.text(66.5, 87.5, "STAGE 2: DIAGNOSE", ha="center", va="center", fontsize=10, fontweight="bold", color="#7d6608")
    ax.text(66.5, 84.5, "Failure Diagnoser", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#b7950b")

    ax.text(66.5, 78.5, "Diagnostic Failure Class (G_t):\n* CLEAN (Normal)\n* TEXT_FAILURE\n* IMAGE_FAILURE\n* MISSING_TEXT\n* MISSING_IMAGE\n* CROSS_MODAL_CONFLICT\n* HIGH_UNCERTAINTY",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#b7950b', lw=0.5))
    ax.text(66.5, 61, "Discretized Context State:\ns_t = (P_t, D_t, U_t, G_t)",
            ha="center", va="center", fontsize=8, fontweight="bold", bbox=dict(boxstyle='round,pad=0.4', fc='#f9e79f', ec='#b7950b', lw=0.8))
    ax.text(66.5, 55, "Non-invasive: Zero test\nlabel or ground truth access",
            ha="center", va="center", fontsize=7, color="#555", fontstyle="italic")

    # Arrow from Stage 1 to Stage 2
    ax.annotate("", xy=(58, 71), xytext=(55, 71), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.8))
    ax.text(56.5, 72.8, "Signals", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#2c3e50")

    # --- 3. Contextual Bandit Recovery Policy ---
    rect_stage3 = FancyBboxPatch((78, 52), 19, 38, boxstyle="round,pad=0.6", ec="#af601a", fc=c_stage3, lw=1.6)
    ax.add_patch(rect_stage3)
    ax.text(87.5, 87.5, "STAGE 3: RECOVER", ha="center", va="center", fontsize=10, fontweight="bold", color="#784212")
    ax.text(87.5, 84.5, "Contextual Bandit Policy", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#af601a")

    ax.text(87.5, 78, "Admissible Action Space A(s_t):\n"
            "a_1: Static Fusion (alpha=0.55)\n"
            "a_2: Text Isolation (f_T only)\n"
            "a_3: Image Isolation (f_V only)\n"
            "a_4: Dynamic Continuous\n"
            "    alpha(x) = clip(0.55+0.4 tanh(3 Delta H))\n"
            "a_5: Conservative Fusion (50/50)",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#af601a', lw=0.5))

    ax.text(87.5, 62, "Action Value Selection:\na* = argmax_{a in A} Q(s_t, a)",
            ha="center", va="center", fontsize=8, fontweight="bold", bbox=dict(boxstyle='round,pad=0.4', fc='#edbb99', ec='#af601a', lw=0.8))

    ax.text(87.5, 55.5, "Physical Admissibility Mask:\nPrevents selecting missing inputs",
            ha="center", va="center", fontsize=7, color="#555", fontstyle="italic")

    # Arrow from Stage 2 to Stage 3
    ax.annotate("", xy=(78, 71), xytext=(75, 71), arrowprops=dict(arrowstyle="->", color=border_dark, lw=1.8))
    ax.text(76.5, 72.8, "s_t", ha="center", va="center", fontsize=8, fontweight="bold", color="#2c3e50")

    # --- 4. Recovery Execution & Deployment Output ---
    rect_stage4 = FancyBboxPatch((78, 22), 19, 23, boxstyle="round,pad=0.6", ec="#2874a6", fc=c_stage4, lw=1.6)
    ax.add_patch(rect_stage4)
    ax.text(87.5, 42.5, "STAGE 4: EXECUTE", ha="center", va="center", fontsize=10, fontweight="bold", color="#1b4f72")
    ax.text(87.5, 39.5, "Decision Execution", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#2874a6")

    ax.text(87.5, 34.5, "p_final = alpha p^T + (1 - alpha) p^V\n"
            "Decision: y_hat_c = 1(p_final_c >= tau_c)",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='#2874a6', lw=0.5))

    ax.text(87.5, 26, "Multi-Label Output Predictions:\ny_hat in {0, 1}^23 canonical genres",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#1b4f72")

    # Arrow from Stage 3 down to Stage 4
    ax.annotate("", xy=(87.5, 45), xytext=(87.5, 52), arrowprops=dict(arrowstyle="->", color=border_dark, lw=2.0))
    ax.text(89.5, 48.5, "a* in A", ha="left", va="center", fontsize=8, fontweight="bold", color="#784212")

    # --- 5. Outcome Learning Closed Loop (Offline / Validation Training) ---
    rect_stage5 = FancyBboxPatch((20, 22), 52, 23, boxstyle="round,pad=0.6", ec="#6c3483", fc=c_stage5, lw=1.6, ls="--")
    ax.add_patch(rect_stage5)
    ax.text(46, 42.5, "STAGE 5: LEARN (OFFLINE TRAINING & VALIDATION FEEDBACK LOOP)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#512e5f")
    ax.text(46, 39.5, "Tabular Outcome-Trained Q-Learning  [STRICT TEST FREEZE: eta_t = 0 during test]",
            ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#6c3483")

    ax.text(34, 31, "Hybrid Reward Function (R_t in [-0.5, 1.5]):\n"
            "R_t = F1(rec) + lambda [F1(rec) - F1(unrec)]\n"
            "Balances absolute F1 utility and relative gain (lambda = 0.5)",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.4', fc='white', ec='#6c3483', lw=0.5))

    ax.text(58, 31, "Count-Decayed Policy Update:\n"
            "eta_t = max(0.05, 0.20 / sqrt(N_{t+1}(s_t, a_t)))\n"
            "Q_{t+1}(s_t, a_t) = Q_t + eta_t [R_t - Q_t(s_t, a_t)]\n"
            "Zero test-label leakage: Policy frozen at inference time",
            ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle='square,pad=0.4', fc='white', ec='#6c3483', lw=0.5))

    # Feedback Loop Arrows
    # From Stage 4 left into Stage 5
    ax.annotate("", xy=(72, 33), xytext=(78, 33), arrowprops=dict(arrowstyle="->", color="#6c3483", lw=1.8, ls="--"))
    ax.text(75, 35, "y_hat, y_true", ha="center", va="center", fontsize=7.5, color="#6c3483", fontweight="bold")

    # From Stage 5 up into Stage 3 (Q-table update)
    ax.annotate("", xy=(78, 58), xytext=(46, 45),
                arrowprops=dict(arrowstyle="->", color="#6c3483", lw=2.0, connectionstyle="angle,angleA=90,angleB=180,rad=8", ls="--"))
    ax.text(58, 48, "Update Q(s_t, a_t)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#6c3483",
            bbox=dict(boxstyle='round,pad=0.3', fc='#f4ecf7', ec='#6c3483', lw=0.8))

    # Legend / Constraints box
    rect_legend = FancyBboxPatch((3, 22), 14, 23, boxstyle="round,pad=0.5", ec="#7f8c8d", fc="#f8f9f9", lw=1.0)
    ax.add_patch(rect_legend)
    ax.text(10, 42.5, "Guarantees & Invariants", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax.text(10, 33.5, "1. Zero Test Leakage\n   eta_t=0 on test set\n2. Low Latency\n   24.5 us decision\n3. Clustered Testing\n   N=500 movie units\n4. Closed Feedback\n   Self-evolving loop",
            ha="center", va="center", fontsize=7, color="#34495e")

    png_path = OUTPUT_DIR / "figure1_seaa_architecture.png"
    pdf_path = OUTPUT_DIR / "figure1_seaa_architecture.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 1: {png_path}")


def generate_figure2_baseline_comparison():
    """Figure 2: Baseline Comparison Graph — Text-only vs Image-only vs Static Fusion."""
    models = ["Text-Only\n(TF-IDF + LR)", "Image-Only\n(ResNet-50 + LR)", "Static Late Fusion\n(Late Fusion, alpha=0.55)"]
    micro_f1 = [0.6056, 0.4151, 0.5161]
    macro_f1 = [0.4955, 0.2847, 0.4003]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    rects1 = ax.bar(x - width/2, micro_f1, width, label='Micro-F1', color='#1b7837', edgecolor='#114a22', linewidth=1.2, alpha=0.92)
    rects2 = ax.bar(x + width/2, macro_f1, width, label='Macro-F1', color='#4575b4', edgecolor='#213a5a', linewidth=1.2, alpha=0.92)

    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_title('Baseline Model Performance Comparison on MM-IMDb\nUnimodal Baselines vs. Static Late Fusion',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0.0, 0.72)
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.legend(fontsize=11, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='upper right')

    # Data value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Note annotation
    ax.text(0.5, -0.15, 'Evaluated on MM-IMDb canonical test split (7,788 movies). Text modality provides dominant signal (+0.1905 Micro-F1 over vision).\nStatic late fusion (alpha=0.55) achieves 0.5161 Micro-F1 and 0.4003 Macro-F1 with validation-optimized decision thresholds.',
            ha='center', va='center', transform=ax.transAxes, fontsize=8.5, fontstyle='italic', color='#555555')

    png_path = OUTPUT_DIR / "figure2_baseline_comparison.png"
    pdf_path = OUTPUT_DIR / "figure2_baseline_comparison.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 2: {png_path}")


def generate_figure3_recovery_performance():
    """Figure 3: Recovery Performance Graph — No Recovery vs Fixed Policy vs Adaptive SEAA."""
    cond_names = [
        "Clean\nBaseline",
        "Text Trunc\n25%",
        "Text Mask\n35%",
        "Text\nEmpty",
        "Image Blur\n(r=5.0)",
        "Image Noise\n(s=0.25)",
        "Image\nBlackout",
        "Dual Trunc\n+ Noise",
        "Condition\nAverage"
    ]

    no_recovery = [0.5187, 0.4635, 0.4619, 0.3836, 0.3466, 0.4121, 0.3060, 0.3173, 0.4012]
    fixed_policy = [0.4652, 0.4432, 0.4364, 0.3996, 0.2272, 0.3183, 0.5820, 0.2813, 0.3942]
    adaptive_seaa = [0.5528, 0.4700, 0.4690, 0.3996, 0.4016, 0.4988, 0.5820, 0.3937, 0.4709]

    x = np.arange(len(cond_names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(13.5, 6.5), dpi=300)

    r1 = ax.bar(x - width, no_recovery, width, label='No Recovery (Static Late Fusion)', color='#7f8c8d', edgecolor='#4d5656', linewidth=1.0, alpha=0.9)
    r2 = ax.bar(x, fixed_policy, width, label='Fixed Policy (Heuristic Router)', color='#e67e22', edgecolor='#935116', linewidth=1.0, alpha=0.9)
    r3 = ax.bar(x + width, adaptive_seaa, width, label='Adaptive SEAA (Ours)', color='#27ae60', edgecolor='#145a32', linewidth=1.3, alpha=0.95)

    ax.set_ylabel('Micro-F1 Score', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_title('Recovery Performance across 8 Canonical Failure Regimes (MM-IMDb Test Split)\nNo Recovery vs. Fixed Heuristic Routing vs. Adaptive SEAA Policy',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.0, 0.72)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    # Highlight condition average block
    ax.axvspan(7.5, 8.5, color='#f2f4f4', alpha=0.6, linestyle=':')

    # Annotate deltas for SEAA on top of bars
    for i, bar in enumerate(r3):
        h = bar.get_height()
        delta = adaptive_seaa[i] - no_recovery[i]
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{h:.3f}\n({sign}{delta:.3f})",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='#145a32' if delta > 0 else '#900c3f')

    ax.legend(fontsize=10.5, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='upper right')

    png_path = OUTPUT_DIR / "figure3_recovery_performance.png"
    pdf_path = OUTPUT_DIR / "figure3_recovery_performance.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 3: {png_path}")


def generate_figure4_action_heatmap():
    """Figure 4: Adaptive Recovery Action Heatmap — Distribution across 8 conditions."""
    cond_labels = [
        "Clean Baseline",
        "Text Truncation 25%",
        "Text Token Masking 35%",
        "Text Empty",
        "Image Gaussian Blur (r=5.0)",
        "Image Gaussian Noise (s=0.25)",
        "Image Missing (Blackout)",
        "Dual Corruption (Trunc+Noise)"
    ]

    action_labels = [
        "Static Fusion\n(alpha=0.55)",
        "Text-Only\nIsolation",
        "Image-Only\nIsolation",
        "Reliability-Weighted\nDynamic alpha(x)",
        "Conservative\nFusion (50/50)"
    ]

    # Exact percentages from policy_comparison_metrics.csv
    action_matrix = np.array([
        [12.2, 72.8, 0.0, 15.0, 0.0],
        [10.8, 73.6, 0.0, 15.6, 0.0],
        [12.2, 77.0, 0.0, 10.8, 0.0],
        [0.0,   0.0, 100.0,  0.0, 0.0],
        [59.2, 39.0, 0.0,  1.8, 0.0],
        [25.6, 67.0, 0.0,  7.4, 0.0],
        [0.0, 100.0, 0.0,  0.0, 0.0],
        [22.6, 69.0, 0.0,  8.4, 0.0]
    ])

    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=300)

    sns.heatmap(
        action_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=action_labels,
        yticklabels=cond_labels,
        cbar_kws={'label': 'Action Selection Frequency (%)'},
        linewidths=0.8,
        linecolor='#e5e7e9',
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        ax=ax
    )

    ax.set_title("SEAA Contextual Bandit Action Allocation Heatmap (%)\nDistribution of 5 Recovery Actions Across 8 Failure Conditions",
                 fontsize=12.5, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel("Admissible Recovery Action Selected by Policy", fontsize=11, fontweight='bold', color='#2c3e50', labelpad=10)
    ax.set_ylabel("Experimental Failure Condition", fontsize=11, fontweight='bold', color='#2c3e50', labelpad=10)
    ax.tick_params(axis='x', labelsize=9.5)
    ax.tick_params(axis='y', labelsize=9.5)

    png_path = OUTPUT_DIR / "figure4_action_heatmap.png"
    pdf_path = OUTPUT_DIR / "figure4_action_heatmap.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 4: {png_path}")


def generate_figure5_ablation_study():
    """Figure 5: Ablation Study Graph — Full SEAA vs Learning Disabled vs Diagnosis Features Removed."""
    configs = [
        "Full SEAA\n(Adaptive Policy)",
        "Ablation 1:\nLearning Disabled\n(Untrained Q-Table)",
        "Ablation 2:\nDiagnosis Removed\n(Coarse Features)"
    ]

    cond_micro_f1 = [0.4711, 0.4377, 0.5054]
    cond_macro_f1 = [0.3611, 0.3361, 0.3714]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=300)

    r1 = ax.bar(x - width/2, cond_micro_f1, width, label='Condition-Mean Micro-F1', color='#1b9e77', edgecolor='#0e533e', linewidth=1.2, alpha=0.92)
    r2 = ax.bar(x + width/2, cond_macro_f1, width, label='Condition-Mean Macro-F1', color='#d95f02', edgecolor='#703101', linewidth=1.2, alpha=0.92)

    # Reference lines for baselines
    ax.axhline(0.4012, color='#7f8c8d', linestyle='--', linewidth=1.3, label='Baseline No Recovery Micro-F1 (0.4012)')
    ax.axhline(0.3931, color='#e6ab02', linestyle=':', linewidth=1.3, label='Baseline Fixed Policy Micro-F1 (0.3931)')

    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_title('SEAA Component Ablation: Impact of Outcome Learning and Diagnostic Features\nIsolating Outcome Learning and State Discretization Granularity',
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, fontweight='bold')
    ax.set_ylim(0.25, 0.60)
    ax.grid(axis='y', linestyle='--', alpha=0.45)

    def autolabel_bars(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    autolabel_bars(r1)
    autolabel_bars(r2)

    # Annotate Outcome Learning Gain
    delta_learn = cond_micro_f1[0] - cond_micro_f1[1]
    ax.annotate(f'Outcome Learning Gain:\n+{delta_learn*100:.2f}% points F1',
                xy=(0.5, 0.44), xytext=(0.5, 0.52),
                arrowprops=dict(facecolor='#1b4f72', arrowstyle='<->', lw=1.5),
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='#ebf5fb', ec='#1b4f72', lw=0.8))

    ax.legend(fontsize=9.5, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', loc='upper right')

    png_path = OUTPUT_DIR / "figure5_ablation_study.png"
    pdf_path = OUTPUT_DIR / "figure5_ablation_study.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 5: {png_path}")


def generate_table1_statistical_results():
    """Table 1: Main Results and Statistical Comparison Table."""
    table_data = [
        {
            "Policy / Model Configuration": "SEAA Adaptive Recovery Policy (Ours)",
            "Condition-Mean Micro-F1": "0.4709",
            "Condition-Mean Macro-F1": "0.3611",
            "Globally Pooled Micro-F1": "0.4690",
            "Instance-Level Mean F1": "0.4671",
            "Delta vs Unrecovered": "+0.0697 (+17.4%)",
            "Delta vs Fixed Policy": "+0.0767 (+19.5%)",
            "Clustered 95% CI vs Unrec": "[+0.0619, +0.0843]",
            "Clustered t-test (df=499)": "t = 12.79, p = 1.34e-32",
            "Wilcoxon Signed-Rank": "W = 23,564.0, p = 1.15e-31",
            "Bootstrap Empirical p": "p < 0.0002"
        },
        {
            "Policy / Model Configuration": "No Recovery (Static Late Fusion)",
            "Condition-Mean Micro-F1": "0.4012",
            "Condition-Mean Macro-F1": "0.3165",
            "Globally Pooled Micro-F1": "0.3973",
            "Instance-Level Mean F1": "0.3940",
            "Delta vs Unrecovered": "Reference (0.0000)",
            "Delta vs Fixed Policy": "+0.0070 (+1.8%)",
            "Clustered 95% CI vs Unrec": "---",
            "Clustered t-test (df=499)": "---",
            "Wilcoxon Signed-Rank": "---",
            "Bootstrap Empirical p": "---"
        },
        {
            "Policy / Model Configuration": "Fixed Heuristic Routing Policy",
            "Condition-Mean Micro-F1": "0.3942",
            "Condition-Mean Macro-F1": "0.2927",
            "Globally Pooled Micro-F1": "0.3905",
            "Instance-Level Mean F1": "0.3901",
            "Delta vs Unrecovered": "-0.0070 (-1.7%)",
            "Delta vs Fixed Policy": "Reference (0.0000)",
            "Clustered 95% CI vs Unrec": "[-0.0185, +0.0045]",
            "Clustered t-test (df=499)": "t = -1.20, p = 0.231",
            "Wilcoxon Signed-Rank": "W = 39,210.0, p = 0.245",
            "Bootstrap Empirical p": "---"
        },
        {
            "Policy / Model Configuration": "Ablation: Learning Disabled (Untrained)",
            "Condition-Mean Micro-F1": "0.4377",
            "Condition-Mean Macro-F1": "0.3361",
            "Globally Pooled Micro-F1": "0.4348",
            "Instance-Level Mean F1": "0.4340",
            "Delta vs Unrecovered": "+0.0365 (+9.1%)",
            "Delta vs Fixed Policy": "+0.0435 (+11.0%)",
            "Clustered 95% CI vs Unrec": "[+0.0270, +0.0460]",
            "Clustered t-test (df=499)": "t = 7.55, p = 2.11e-13",
            "Wilcoxon Signed-Rank": "W = 31,450.0, p = 4.88e-13",
            "Bootstrap Empirical p": "p < 0.0002"
        },
        {
            "Policy / Model Configuration": "Ablation: Diagnosis Features Removed",
            "Condition-Mean Micro-F1": "0.5054",
            "Condition-Mean Macro-F1": "0.3714",
            "Globally Pooled Micro-F1": "0.5056",
            "Instance-Level Mean F1": "0.5042",
            "Delta vs Unrecovered": "+0.1042 (+26.0%)",
            "Delta vs Fixed Policy": "+0.1112 (+28.2%)",
            "Clustered 95% CI vs Unrec": "[+0.0934, +0.1150]",
            "Clustered t-test (df=499)": "t = 18.91, p = 4.12e-60",
            "Wilcoxon Signed-Rank": "W = 15,210.0, p = 3.20e-56",
            "Bootstrap Empirical p": "p < 0.0002"
        },
        {
            "Policy / Model Configuration": "Theoretical Ceiling: Oracle Upper Bound",
            "Condition-Mean Micro-F1": "0.5695",
            "Condition-Mean Macro-F1": "0.4421",
            "Globally Pooled Micro-F1": "0.5673",
            "Instance-Level Mean F1": "0.5658",
            "Delta vs Unrecovered": "+0.1682 (+41.9%)",
            "Delta vs Fixed Policy": "+0.1753 (+44.5%)",
            "Clustered 95% CI vs Unrec": "[+0.1554, +0.1810]",
            "Clustered t-test (df=499)": "t = 25.84, p = 1.10e-94",
            "Wilcoxon Signed-Rank": "W = 4,120.0, p = 8.54e-90",
            "Bootstrap Empirical p": "p < 0.0002"
        }
    ]

    df = pd.DataFrame(table_data)

    # 1. Save CSV
    csv_path = OUTPUT_DIR / "table1_main_statistical_results.csv"
    df.to_csv(csv_path, index=False)

    # 2. Save JSON
    json_path = OUTPUT_DIR / "table1_main_statistical_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2)

    # 3. Save Markdown
    md_content = """# Table 1: Main Results and Statistical Comparison Table

**Repository:** `SEAA (Self-Evolving AI Auditor for Multimodal Reliability)`  
**Benchmark:** MM-IMDb (4,000 Evaluations Across 8 Canonical Failure Regimes; 500 Unique Clustered Movie Entities)  
**Statistical Design:** Repeated-measures paired analysis clustered at the movie entity level ($N=500$, $df=499$).  
**Cluster Dependence Metrics:** Intraclass Correlation Coefficient $\\rho_{\\text{ICC}} = 0.1319$, Variance Inflation Factor $\\text{VIF} = 1.9232$, Effective Sample Size $N_{\\text{eff}} \\approx 2,080$.

| Policy / Model Configuration | Condition-Mean Micro-F1 | Condition-Mean Macro-F1 | Globally Pooled Micro-F1 | Instance-Level Mean F1 | Delta vs Unrec. | Clustered 95% CI vs Unrec | Clustered Paired $t$-test ($df=499$) | Wilcoxon Signed-Rank Test | Bootstrap $p$-value ($B=5000$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEAA Adaptive Recovery Policy** | **0.4709** | **0.3611** | **0.4690** | **0.4671** | **+0.0697 (+17.4%)** | **[+0.0619, +0.0843]** | **$t = 12.79, p = 1.34 \\times 10^{-32}$** | **$W = 23,564.0, p = 1.15 \\times 10^{-31}$** | **$p < 0.0002$** |
| **No Recovery (Static Late Fusion)** | 0.4012 | 0.3165 | 0.3973 | 0.3940 | Reference (0.0000) | --- | --- | --- | --- |
| **Fixed Heuristic Routing Policy** | 0.3942 | 0.2927 | 0.3905 | 0.3901 | -0.0070 (-1.7%) | [-0.0185, +0.0045] | $t = -1.20, p = 0.231$ | $W = 39,210.0, p = 0.245$ | --- |
| **Ablation: Learning Disabled (Untrained)** | 0.4377 | 0.3361 | 0.4348 | 0.4340 | +0.0365 (+9.1%) | [+0.0270, +0.0460] | $t = 7.55, p = 2.11 \\times 10^{-13}$ | $W = 31,450.0, p = 4.88 \\times 10^{-13}$ | $p < 0.0002$ |
| **Ablation: Diagnosis Features Removed** | 0.5054 | 0.3714 | 0.5056 | 0.5042 | +0.1042 (+26.0%) | [+0.0934, +0.1150] | $t = 18.91, p = 4.12 \\times 10^{-60}$ | $W = 15,210.0, p = 3.20 \\times 10^{-56}$ | $p < 0.0002$ |
| **Theoretical Ceiling: Oracle Upper Bound** | 0.5695 | 0.4421 | 0.5673 | 0.5658 | +0.1682 (+41.9%) | [+0.1554, +0.1810] | $t = 25.84, p = 1.10 \\times 10^{-94}$ | $W = 4,120.0, p = 8.54 \\times 10^{-90}$ | $p < 0.0002$ |

*Notes:*
1. **Zero Test-Label Leakage:** The SEAA policy action-values $Q(s, a)$ are frozen during testing; no test labels or corruption metadata are exposed.
2. **Cluster Significance:** The clustered standard error ($SE = 0.0057$) accounts for repeated measures across 8 conditions per movie. Naive unclustered estimation underestimates variance by $1.39\\times$.
3. **5,000-Iteration Cluster Bootstrap:** Out of 5,000 cluster-resampled replications, zero resamples produced non-positive differences (empirical $p < 0.0002$).
"""
    md_path = OUTPUT_DIR / "table1_main_statistical_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 4. Save LaTeX
    tex_content = r"""\begin{table*}[t]
\centering
\caption{Main Results and Statistical Comparison Across 4,000 Repeated Evaluations on MM-IMDb ($N=500$ Clustered Movie Entities, $df=499$).}
\label{tab:main_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\hline
\textbf{Policy / Model Configuration} & \textbf{Condition Micro-F1} & \textbf{Condition Macro-F1} & \textbf{Pooled Micro-F1} & \textbf{$\Delta$ vs. Unrec.} & \textbf{Clustered 95\% CI} & \textbf{Clustered $t$-test / Wilcoxon ($p$-value)} \\
\hline
\textbf{SEAA Adaptive Recovery Policy (Ours)} & \textbf{0.4709} & \textbf{0.3611} & \textbf{0.4690} & \textbf{+0.0697 (+17.4\%)} & \textbf{[+0.0619, +0.0843]} & \textbf{$t = 12.79$ ($p = 1.34 \times 10^{-32}$) / $W = 23,564.0$ ($p = 1.15 \times 10^{-31}$)} \\
No Recovery (Static Late Fusion) & 0.4012 & 0.3165 & 0.3973 & Reference & --- & --- \\
Fixed Heuristic Routing Policy & 0.3942 & 0.2927 & 0.3905 & -0.0070 (-1.7\%) & [-0.0185, +0.0045] & $t = -1.20$ ($p = 0.231$) / $W = 39,210.0$ ($p = 0.245$) \\
Ablation 1: Learning Disabled (Untrained) & 0.4377 & 0.3361 & 0.4348 & +0.0365 (+9.1\%) & [+0.0270, +0.0460] & $t = 7.55$ ($p = 2.11 \times 10^{-13}$) / $W = 31,450.0$ ($p = 4.88 \times 10^{-13}$) \\
Ablation 2: Diagnosis Features Removed & 0.5054 & 0.3714 & 0.5056 & +0.1042 (+26.0\%) & [+0.0934, +0.1150] & $t = 18.91$ ($p = 4.12 \times 10^{-60}$) / $W = 15,210.0$ ($p = 3.20 \times 10^{-56}$) \\
Theoretical Ceiling: Oracle Upper Bound & 0.5695 & 0.4421 & 0.5673 & +0.1682 (+41.9\%) & [+0.1554, +0.1810] & $t = 25.84$ ($p = 1.10 \times 10^{-94}$) / $W = 4,120.0$ ($p = 8.54 \times 10^{-90}$) \\
\hline
\end{tabular}%
}
\end{table*}
"""
    tex_path = OUTPUT_DIR / "table1_main_statistical_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"[SUCCESS] Saved Table 1 (MD, TEX, CSV, JSON) to {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print("SEAA IEEE PAPER OUTPUT GENERATION: 5 Figures + 1 Table")
    print("=" * 70)
    generate_figure1_architecture()
    generate_figure2_baseline_comparison()
    generate_figure3_recovery_performance()
    generate_figure4_action_heatmap()
    generate_figure5_ablation_study()
    generate_table1_statistical_results()
    print("=" * 70)
    print("ALL 6 CORE OUTPUTS GENERATED SUCCESSFULLY!")
    print(f"Output files stored in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

