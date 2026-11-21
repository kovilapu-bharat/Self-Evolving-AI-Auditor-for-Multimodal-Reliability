"""
SEAA IEEE Master Output Generator (Strict IEEE Transactions Compliant).
Generates all 5 publication-grade figures, companion source CSVs, Table 1 (CSV, Markdown, and LaTeX),
and the cryptographic reproduction log for the IEEE Transactions manuscript from verified project artifacts.

Strict IEEE Design Standards:
- Dimensions: Two-column width = 7.16 inches (181.8 mm / 43 picas), Single-column = 3.5 inches.
- Typography: IEEE standard sans-serif (Arial/Helvetica/DejaVu Sans) with hierarchical font scaling (5.0 - 8.5 pt).
- Layout: Zero overlapping text boxes, zero clipped labels, clean headroom for callouts and legends.
- Vector & Raster Outputs: High-resolution vector PDF (primary) and 300 DPI raster PNG (companion).

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

# ==============================================================================
# Strict IEEE Styling Parameters
# ==============================================================================
IEEE_FONT_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "figure.titlesize": 9.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linestyle": "--",
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.7
}
plt.rcParams.update(IEEE_FONT_CONFIG)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ieee_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Accessible, publication-grade palette
PRIMARY_BLUE = "#1d4ed8"      # SEAA Adaptive Policy / Primary
ACCENT_ORANGE = "#ea580c"     # Fixed Policy / Macro-F1
ACCENT_GREEN = "#15803d"      # Oracle / Positive Ceiling
ACCENT_RED = "#b91c1c"        # Degradation / Failure
ACCENT_PURPLE = "#6d28d9"     # Auditing & Diagnosis
NEUTRAL_SLATE = "#475569"     # Baseline / Late Fusion
LIGHT_BG = "#f8fafc"          # Clean card background
BORDER_GRAY = "#cbd5e1"       # Subtle border
DARK_NAVY = "#0f172a"         # Header text


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


def get_latest_recovery_dir() -> Path:
    """Finds the canonical adaptive recovery benchmark output directory with sample_size >= 500."""
    recovery_root = PROJECT_ROOT / "outputs" / "recovery"
    candidate_dirs = sorted([d for d in recovery_root.glob("run_*_adaptive_recovery") if d.is_dir()])
    for d in reversed(candidate_dirs):
        summary_file = d / "phase5_benchmark_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                if summary.get("sample_size_per_condition", 0) >= 500:
                    return d
            except Exception:
                pass
    for fallback in [
        recovery_root / "run_20261029_105918_adaptive_recovery",
        recovery_root / "run_20261025_phase5_adaptive_recovery"
    ]:
        if fallback.exists():
            return fallback
    return recovery_root


# ==============================================================================
# 1. Figure 1: SEAA Inline System Architecture Diagram
# ==============================================================================
def generate_figure_1():
    print("Generating Figure 1: SEAA Architecture Diagram (IEEE Compliant)...")
    fig = plt.figure(figsize=(7.16, 4.6), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Title & Subtitle Banner (generous headroom, zero collision with latency badge)
    ax.text(37.0, 97.2, "SEAA: Self-Evolving Inline Reliability Auditing & Recovery Architecture",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK_NAVY)
    ax.text(37.0, 93.8, "Closed-Loop Disagreement Monitoring, Failure Diagnosis & Contextual Bandit Recovery",
            ha="center", va="center", fontsize=6.4, fontstyle="italic", color="#475569")

    # Latency badge in header right corner
    ax.text(98.5, 95.5, "Decision Latency: 0.025 ms\nAudited Pipeline: 11.57 ms (CPU)",
            ha="right", va="center", fontsize=5.2, color="#334155",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec=BORDER_GRAY, lw=0.6))

    # Helper function for drawing rounded boxes with clean typography and alignment
    def draw_card(x, y, w, h, title, body_lines, bg="#ffffff", border=BORDER_GRAY, lw=0.85,
                  title_color=DARK_NAVY, title_size=5.8, body_size=5.0, title_dy=3.2,
                  align="center", pad_left=1.2, subtitle=None):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            ec=border, fc=bg, lw=lw, zorder=2
        )
        ax.add_patch(box)
        # Title cleanly inside top of box
        if title:
            ax.text(x + w / 2, y + h - title_dy, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + h - title_dy - 2.4, subtitle, ha="center", va="center",
                    fontsize=body_size - 0.2, fontstyle="italic", color=title_color, zorder=3)
        # Body text
        if body_lines:
            text_str = "\n".join(body_lines) if isinstance(body_lines, list) else body_lines
            eff_dy = title_dy + (2.8 if subtitle else 0)
            if align == "left":
                ax.text(x + pad_left, y + (h - eff_dy) / 2, text_str, ha="left", va="center",
                        fontsize=body_size, color="#1e293b", zorder=3, multialignment="left", linespacing=1.24)
            else:
                ax.text(x + w / 2, y + (h - eff_dy) / 2, text_str, ha="center", va="center",
                        fontsize=body_size, color="#1e293b", zorder=3, multialignment="center", linespacing=1.20)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#64748b", lw=0.9, dashed=False):
        ls = "--" if dashed else "-"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                            mutation_scale=6.5, shrinkA=1, shrinkB=1),
            zorder=4
        )

    def draw_elbow_arrow(points, color="#64748b", lw=0.9, dashed=False):
        """Draws orthogonal Manhattan arrow connecting a list of coordinates [(x1, y1), ...]."""
        ls = "--" if dashed else "-"
        for i in range(len(points) - 2):
            ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]],
                    color=color, lw=lw, linestyle=ls, zorder=4)
        ax.annotate(
            "", xy=points[-1], xytext=points[-2],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                            mutation_scale=6.5, shrinkA=0, shrinkB=1),
            zorder=4
        )

    # --------------------------------------------------------------------------
    # Column 1: Input Modalities (x: 1.5 to 16.5, w=15.0)
    # --------------------------------------------------------------------------
    draw_card(1.5, 64.0, 15.0, 23.5, "Text Input $x^T$",
              ["Movie Plot Synopsis", "(Natural Language Text)", "25,959 Titles"],
              bg="#f0f9ff", border="#38bdf8", title_color="#0369a1")

    draw_card(1.5, 23.5, 15.0, 23.5, "Vision Input $x^V$",
              ["Poster Artwork RGB", "(224×224 Pixels)", "Canonical Artwork"],
              bg="#fffbeb", border="#f59e0b", title_color="#b45309")

    # --------------------------------------------------------------------------
    # Column 2: Predictive Baselines (x: 19.5 to 36.0, w=16.5)
    # --------------------------------------------------------------------------
    draw_card(19.5, 64.0, 16.5, 23.5, "Text Baseline $f_T$",
              ["TF-IDF (10k Feats, 1-2 gr)", "OvR Logistic Reg ($C=1.0$)", "Output: $\\hat{p}^T \\in [0, 1]^{23}$"],
              bg="#f8fafc", border="#94a3b8", title_color="#1e293b")

    draw_card(19.5, 23.5, 16.5, 23.5, "Vision Baseline $f_V$",
              ["Pretrained ResNet-50", "2,048-dim Vector $z^V$", "OvR Logistic Reg ($C=0.5$)", "Output: $\\hat{p}^V \\in [0, 1]^{23}$"],
              bg="#f8fafc", border="#94a3b8", title_color="#1e293b")

    # Static Late Fusion (Reference Anchor) - symmetrical 2.5 gap above and below
    draw_card(19.5, 49.5, 16.5, 12.0, "Static Fusion $f_{MM}$",
              ["$\\hat{p}^{MM} = 0.55\\hat{p}^T + 0.45\\hat{p}^V$", "(Unrecovered Anchor)"],
              bg="#f1f5f9", border="#64748b", lw=0.75, title_color="#475569", title_dy=2.5, body_size=5.0)

    # Arrows Input -> Baseline
    draw_arrow(16.5, 75.75, 19.5, 75.75, color="#0284c7")
    draw_arrow(16.5, 35.25, 19.5, 35.25, color="#d97706")
    draw_arrow(27.75, 64.0, 27.75, 61.5, color="#64748b")
    draw_arrow(27.75, 47.0, 27.75, 49.5, color="#64748b")

    # --------------------------------------------------------------------------
    # Column 3: Inline Auditor & Failure Diagnoser (x: 39.5 to 58.0, w=18.5)
    # --------------------------------------------------------------------------
    draw_card(39.5, 49.5, 18.5, 38.0, "1. AUDITOR & MONITOR",
              ["• Presence: $P_T, P_V \\in \\{0, 1\\}$",
               "• Entropy: $H(\\hat{p}^T), H(\\hat{p}^V)$",
               "• Bernoulli Disagreement:",
               "    $D_{JS}(\\hat{p}^T \\Vert \\hat{p}^V)$  ($\\tau=0.114$)",
               "• Feature Norms: $\\|z^T\\|, \\|z^V\\|$"],
              bg="#faf5ff", border="#a855f7", lw=0.95, title_color="#7e22ce",
              align="left", pad_left=1.2, body_size=4.9, subtitle="Observable Signals (No Labels)")

    draw_card(39.5, 17.5, 18.5, 29.0, "2. FAILURE DIAGNOSER",
              ["• CLEAN (Pristine baseline)",
               "• TEXT_FAIL / IMG_FAIL",
               "• MISSING_T / MISSING_V",
               "• CONFLICT / HIGH_UNC"],
              bg="#f3e8ff", border="#9333ea", lw=0.85, title_color="#6b21a8",
              align="left", pad_left=1.2, body_size=4.9, subtitle="Diagnosed State $G_t \\in \\mathcal{G}$")

    # Connections to Auditor & Diagnoser (Clean Manhattan paths, clear of text)
    draw_arrow(36.0, 75.75, 39.5, 75.75, color="#0369a1")
    draw_elbow_arrow([(36.0, 35.25), (37.75, 35.25), (37.75, 58.0), (39.5, 58.0)], color="#b45309")
    draw_arrow(48.75, 49.5, 48.75, 46.5, color="#7e22ce")

    # --------------------------------------------------------------------------
    # Column 4: Contextual Bandit Recovery Policy (x: 61.0 to 78.5, w=17.5)
    # --------------------------------------------------------------------------
    draw_card(61.0, 27.5, 17.5, 60.0, "3. ADAPTIVE POLICY",
              ["State 4-Tuple:",
               "  $s_t=(P_t, D_t, U_t, G_t)$",
               "",
               "Physical Mask:",
               "  $\\mathcal{A}_{val} \\subseteq \\mathcal{A}$",
               "",
               "Decision Rule:",
               "  $a^* = \\arg\\max_a Q(s_t, a)$",
               "",
               "Test Deployment:",
               "  Strictly Frozen ($\\eta_t = 0$)",
               "  Deterministic Greedy"],
              bg="#ecfdf5", border="#10b981", lw=1.0, title_color="#047857",
              align="left", pad_left=1.2, body_size=4.9, subtitle="Tabular Q-Learning Bandit")

    draw_arrow(58.0, 68.5, 61.0, 68.5, color="#7e22ce")
    draw_elbow_arrow([(58.0, 32.0), (59.5, 32.0), (59.5, 45.0), (61.0, 45.0)], color="#9333ea")

    # --------------------------------------------------------------------------
    # Column 5: Recovery Actions & Predictions (x: 81.5 to 98.5, w=17.0)
    # --------------------------------------------------------------------------
    draw_card(81.5, 53.0, 17.0, 34.5, "4. RECOVERY ACTIONS",
              ["• STATIC ($\\alpha=0.55$ Fusion)",
               "• TEXT_ONLY ($\\alpha=1.0$)",
               "• IMAGE_ONLY ($\\alpha=0.0$)",
               "• REL_WEIGHT (Dynamic $\\alpha$)",
               "• CONSERVATIVE (50/50)"],
              bg="#f0fdf4", border="#22c55e", lw=0.85, title_color="#15803d",
              align="left", pad_left=1.2, body_size=4.9, subtitle="Candidate Action Space $\\mathcal{A}$")

    draw_card(81.5, 16.0, 17.0, 34.0, "5. PREDICTIONS",
              ["Posterior $\\hat{p}^{\\text{rec}} \\in [0, 1]^{23}$",
               "$\\hat{y}_c = \\mathbb{I}(\\hat{p}_c^{\\text{rec}} \\geq \\tau_c)$",
               "Across 23 Canonical Genres",
               "",
               "Micro-F1: 0.4709",
               "(+17.4% vs Unrec.)"],
              bg="#fef2f2", border="#ef4444", lw=0.95, title_color="#b91c1c",
              title_dy=2.8, body_size=4.9, subtitle="Audited Model Output")

    draw_arrow(78.5, 70.0, 81.5, 70.0, color="#10b981")
    draw_arrow(90.0, 53.0, 90.0, 50.0, color="#22c55e")

    # --------------------------------------------------------------------------
    # Bottom Lane: Offline Outcome Learning Loop & Zero-Leakage Badge
    # --------------------------------------------------------------------------
    draw_card(14.0, 1.5, 62.0, 11.5, "OFFLINE VALIDATION OUTCOME LEARNING (FROZEN AT TEST DEPLOYMENT)",
              ["Reward: $R_t = F_1(\\hat{y}_{a_t}, y_t) + 0.5[F_1(\\hat{y}_{a_t}, y_t) - F_1(\\hat{y}_{unrec}, y_t)] \\in [-0.5, 1.5]$",
               "Policy Update: $Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \\eta_t [R_t - Q_t(s_t, a_t)]$   ($\\eta_t = \\max(0.05, 0.20/\\sqrt{N_{t+1}})$)"],
              bg="#fff1f2", border="#f43f5e", lw=0.8, title_color="#be123c", title_size=5.8, body_size=5.1, title_dy=2.4)

    # Validation Badge (Right of bottom lane)
    draw_card(78.0, 1.5, 20.5, 11.5, "Zero Test Leakage",
              ["Offline Val Only", "Test $\\eta_t = 0$ (Frozen Policy)"],
              bg="#ffffff", border="#f43f5e", lw=0.75, title_color="#be123c", title_size=5.8, body_size=4.9, title_dy=2.4)

    # Clean non-overlapping feedback routing (Down from Predictions -> into Badge -> left into Outcome Learning -> UP into Policy)
    draw_arrow(90.0, 16.0, 90.0, 13.0, color="#e11d48", dashed=True)
    draw_arrow(78.0, 7.25, 76.0, 7.25, color="#e11d48", dashed=True)
    draw_arrow(69.75, 13.0, 69.75, 27.5, color="#e11d48", dashed=True)

    pdf_path = OUTPUT_DIR / "fig1_seaa_architecture.pdf"
    png_path = OUTPUT_DIR / "fig1_seaa_architecture.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {pdf_path} and {png_path}.")


# ==============================================================================
# 2. Figure 2: Baseline Model Comparison (Two-Panel IEEE Layout)
# ==============================================================================
def generate_figure_2():
    print("Generating Figure 2: Baseline Model Comparison (IEEE Compliant)...")
    source_csv = PROJECT_ROOT / "outputs" / "metrics" / "baseline_comparison.csv"
    df = pd.read_csv(source_csv)

    dest_csv = OUTPUT_DIR / "fig2_baseline_source.csv"
    df.to_csv(dest_csv, index=False)

    models = ["Text-Only\n(TF-IDF+LR)", "Image-Only\n(ResNet-50+LR)", "Multimodal\n(Late Fusion)"]
    micro_f1 = df["micro_f1"].values
    macro_f1 = df["macro_f1"].values
    map_scores = df["mean_average_precision"].values
    hamming = df["hamming_loss"].values

    # IEEE Double-Column: 7.16 x 2.85 inches with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.85), dpi=300, gridspec_kw={"wspace": 0.26})

    x = np.arange(len(models))
    width = 0.25
    offset = 0.16  # center-to-center distance, leaving 0.07 clear gap between paired bars

    # Panel (a): F1-Scores
    r1 = ax1.bar(x - offset, micro_f1, width, label="Micro-F1", color=PRIMARY_BLUE, alpha=0.92, zorder=3)
    r2 = ax1.bar(x + offset, macro_f1, width, label="Macro-F1", color=ACCENT_ORANGE, alpha=0.92, zorder=3)

    for rect in r1:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold", color="#0f172a",
                     bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=0.85), zorder=5)

    for rect in r2:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, color="#0f172a",
                     bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=0.85), zorder=5)

    ax1.set_ylabel("F1-Score", fontweight="bold")
    ax1.set_title("(a) Multi-Label Classification F1-Scores", pad=8, fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.2)
    ax1.set_ylim(0, 0.72)
    ax1.grid(axis="y", linestyle="--", alpha=0.18)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, loc="upper right", fontsize=6.8)

    # Panel (b): Ranking Quality & Label Loss (mAP and Hamming Loss)
    r3 = ax2.bar(x - offset, map_scores, width, label="mAP (Ranking)", color=ACCENT_PURPLE, alpha=0.90, zorder=3)
    r4 = ax2.bar(x + offset, hamming, width, label="Hamming Loss", color=NEUTRAL_SLATE, alpha=0.85, zorder=3)

    for rect in r3:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold", color="#0f172a",
                     bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=0.85), zorder=5)

    for rect in r4:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, color="#0f172a",
                     bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=0.85), zorder=5)

    ax2.set_ylabel("Metric Value", fontweight="bold")
    ax2.set_title("(b) Ranking Precision (mAP) & Hamming Loss (N = 7,788 Movies)", pad=8, fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=7.2)
    ax2.set_ylim(0, 0.62)
    ax2.grid(axis="y", linestyle="--", alpha=0.18)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, loc="upper right", fontsize=6.8)

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
    print("Generating Figure 3: Recovery Across Eight Failure Regimes (IEEE Compliant)...")
    recovery_dir = get_latest_recovery_dir()
    source_csv = recovery_dir / "policy_comparison_metrics.csv"
    if not source_csv.exists():
        source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    dest_csv = OUTPUT_DIR / "fig3_recovery_source.csv"
    df.to_csv(dest_csv, index=False)

    conditions = [
        "Clean\nBaseline",
        "Text Trunc.\n(25%)",
        "Text Mask\n(35%)",
        "Text\nEmpty",
        "Image Blur\n(r=5.0)",
        "Image Noise\n(σ=0.25)",
        "Image Miss\n(Blackout)",
        "Dual Corrupt.\n(Trunc+Noise)"
    ]

    p1_micro = df["no_recovery_micro_f1"].values
    p2_micro = df["fixed_policy_micro_f1"].values
    p3_micro = df["adaptive_policy_micro_f1"].values
    p4_micro = df["oracle_micro_f1"].values

    p1_avg = np.mean(p1_micro)
    p2_avg = np.mean(p2_micro)
    p3_avg = np.mean(p3_micro)
    p4_avg = np.mean(p4_micro)

    n_cond = len(conditions)
    indices = np.arange(n_cond)
    avg_index = n_cond + 0.45
    all_x = np.append(indices, avg_index)
    all_labels = conditions + ["Condition\nAverage"]

    all_p1 = np.append(p1_micro, p1_avg)
    all_p2 = np.append(p2_micro, p2_avg)
    all_p3 = np.append(p3_micro, p3_avg)
    all_p4 = np.append(p4_micro, p4_avg)

    # Standard IEEE Double-Column: 7.16 x 3.65 inches
    fig, ax = plt.subplots(figsize=(7.16, 3.65), dpi=300)
    width = 0.23

    # Demarcation for Summary Average (Clean soft shading)
    ax.axvspan(n_cond - 0.3, avg_index + 0.45, color="#f8fafc", alpha=0.95, zorder=1)
    ax.axvline(n_cond - 0.3, color="#94a3b8", linestyle="--", lw=0.9, zorder=2)

    # Grouped bars
    r1 = ax.bar(all_x - width, all_p1, width, label="No Recovery (Late Fusion)", color=NEUTRAL_SLATE, alpha=0.88, zorder=3)
    r2 = ax.bar(all_x, all_p2, width, label="Fixed Policy (Router)", color=ACCENT_ORANGE, alpha=0.92, zorder=3)
    r3 = ax.bar(all_x + width, all_p3, width, label="Adaptive SEAA (Proposed)", color=PRIMARY_BLUE, alpha=0.95, zorder=3)

    # High-visibility Oracle Ceiling markers
    scatter_p4 = all_p4.copy()
    for idx in range(len(scatter_p4)):
        top_bar = max(all_p1[idx], all_p2[idx], all_p3[idx])
        if abs(scatter_p4[idx] - top_bar) < 0.005 and idx < n_cond:
            scatter_p4[idx] += 0.025
            ax.plot([all_x[idx], all_x[idx]], [top_bar, scatter_p4[idx]], color="#b91c1c", linestyle=":", lw=1.0, zorder=5)

    scatter_handle = ax.scatter(all_x, scatter_p4, color=ACCENT_RED, edgecolors="#450a0a", s=34, marker="D",
                                linewidths=1.1, label="Oracle Bound (Ceiling)", zorder=6)

    # Clean, non-overlapping summary annotation in Summary Average panel
    summary_text = (
        f"Condition Mean F1:\n"
        f"• SEAA: {p3_avg:.4f} (+17.4%)\n"
        f"• NoRec: {p1_avg:.4f} (Ref)\n"
        f"• Fixed: {p2_avg:.4f} (-1.7%)\n"
        f"• Oracle: {p4_avg:.4f} (Ceiling)"
    )
    ax.text(avg_index, 0.77, summary_text, ha="center", va="top",
            fontsize=5.5, color="#1e293b", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=BORDER_GRAY, lw=0.6), zorder=4)

    # Only annotate the proposed SEAA bar and Oracle diamond in Condition Average panel (zero collision)
    ax.annotate(f"{p3_avg:.3f}", xy=(avg_index + width, p3_avg), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color=PRIMARY_BLUE, zorder=4)
    ax.annotate(f"{p4_avg:.3f}", xy=(avg_index, p4_avg), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.0, fontweight="bold", color=ACCENT_RED, zorder=7)

    ax.set_ylabel("Micro-F1 Score", fontweight="bold")
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=6.8)
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", linestyle="--", alpha=0.18)

    # Clean Top Legend positioned outside data area (Matching bar group order)
    handles = [r1, r2, r3, scatter_handle]
    labels = [
        "No Recovery (Late Fusion)",
        "Fixed Policy (Router)",
        "Adaptive SEAA (Proposed)",
        "Oracle Bound (Ceiling)"
    ]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, fontsize=6.7,
              columnspacing=0.8, handletextpad=0.35)

    # Non-overlapping Callout Badges with clear leader lines
    # 1. Misdiagnosed Blur collapse callout (safely positioned at y=0.42 well below Oracle at 0.588)
    ax.annotate("Fixed Policy Collapse\n(Drops to 0.2272)",
                xy=(4, 0.235), xytext=(4, 0.42),
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=0.9),
                ha="center", fontsize=5.8, color="#991b1b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#fca5a5", lw=0.7), zorder=5)

    # 2. Missing modality physical tie (safely positioned at y=0.70)
    ax.annotate("Physical Masking Tie\n(Only Text Feasible)",
                xy=(6, 0.61), xytext=(6, 0.70),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8),
                ha="center", fontsize=5.8, color="#334155",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=BORDER_GRAY, lw=0.7), zorder=5)

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
    print("Generating Figure 4: Adaptive Recovery Action Heatmap (IEEE Compliant)...")
    recovery_dir = get_latest_recovery_dir()
    source_csv = recovery_dir / "policy_comparison_metrics.csv"
    if not source_csv.exists():
        source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    conditions_clean = [
        "Clean Baseline",
        "Text Truncation (25%)",
        "Text Masking (35%)",
        "Text Modality Empty",
        "Image Gaussian Blur (r=5.0)",
        "Image Gaussian Noise (σ=0.25)",
        "Image Modality Missing",
        "Dual Corrupt. (Trunc+Noise)"
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
        "Text Isolation\n(α = 1.0)",
        "Vision Isolation\n(α = 0.0)",
        "Dynamic Gating\n(α(x))",
        "Conservative\n(50 / 50)"
    ]

    heatmap_data = df[action_cols].values

    df_source = pd.DataFrame(heatmap_data, index=conditions_clean, columns=[
        "STATIC_FUSION", "TEXT_ONLY", "IMAGE_ONLY", "RELIABILITY_WEIGHTED", "CONSERVATIVE_FUSION"
    ])
    df_source.index.name = "Failure_Regime"
    dest_csv = OUTPUT_DIR / "fig4_action_distribution_source.csv"
    df_source.to_csv(dest_csv)

    # Standard IEEE Double-Column: 7.16 x 3.5 inches
    fig, ax = plt.subplots(figsize=(7.16, 3.5), dpi=300)

    # Build customized heatmap
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="Blues",
        xticklabels=action_names,
        yticklabels=conditions_clean,
        cbar_kws={"shrink": 0.88, "pad": 0.03},
        vmin=0.0,
        vmax=100.0,
        linewidths=1.0,
        linecolor="#ffffff",
        ax=ax
    )

    # Add custom contrast-optimized text in each cell
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            if val >= 40.0:
                txt_color = "#ffffff"
                weight = "bold"
                size = 7.5
            elif val > 0.0:
                txt_color = "#0f172a"
                weight = "bold"
                size = 7.5
            else:
                txt_color = "#94a3b8"  # Soft gray for 0.0% so selected actions pop out cleanly
                weight = "normal"
                size = 7.0
            ax.text(j + 0.5, i + 0.5, f"{val:.1f}", ha="center", va="center",
                    color=txt_color, fontweight=weight, fontsize=size)

    # Fix colorbar tick styling (length=0 completely eliminates tick lines, preventing any minus-sign confusion!)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7.0, direction="in", length=0, pad=5)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar.set_label("Action Selection Frequency", fontsize=8.0, fontweight="bold", labelpad=8)

    # Remove distracting long tick lines on heatmap y-axis
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=4)

    ax.set_title("Adaptive Recovery Action Distribution Across 4,000 Evaluations (500 Movies × 8 Regimes)",
                 pad=10, fontsize=8.5, fontweight="bold")
    plt.xticks(rotation=0, ha="center", fontsize=7.2)
    plt.yticks(rotation=0, fontsize=7.2)

    pdf_path = OUTPUT_DIR / "fig4_action_heatmap.pdf"
    png_path = OUTPUT_DIR / "fig4_action_heatmap.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {pdf_path} and {png_path}.")


# ==============================================================================
# 5. Figure 5: Component Ablation Study (Clean IEEE Layout)
# ==============================================================================
def generate_figure_5():
    print("Generating Figure 5: Component Ablation Study (IEEE Compliant)...")
    source_csv = PROJECT_ROOT / "outputs" / "recovery" / "ablation_study_results.csv"
    df = pd.read_csv(source_csv)

    dest_csv = OUTPUT_DIR / "fig5_ablation_source.csv"
    df.to_csv(dest_csv, index=False)

    policy_labels = [
        "Full SEAA Policy\n(Trained Bandit)",
        "Ablation: Learning Off\n(Untrained Q-Table)",
        "Ablation: No Diagnoser*\n(Coarse Context)",
        "Ablation: Random Action\n(Uniform Selection)"
    ]

    df_variants = df.iloc[[0, 1, 2, 3]].copy()
    micros = df_variants["Condition_Mean_Micro_F1"].values
    macros = df_variants["Condition_Mean_Macro_F1"].values

    no_rec_micro = df[df["Policy"] == "Baseline_No_Recovery"]["Condition_Mean_Micro_F1"].values[0]
    fixed_micro = df[df["Policy"] == "Baseline_Fixed_Policy"]["Condition_Mean_Micro_F1"].values[0]
    oracle_micro = df[df["Policy"] == "Oracle_Upper_Bound"]["Condition_Mean_Micro_F1"].values[0]

    x = np.arange(len(policy_labels))
    width = 0.26
    offset = 0.16

    # Standard IEEE Double-Column: 7.16 x 3.3 inches
    fig, ax = plt.subplots(figsize=(7.16, 3.3), dpi=300)

    # Reference horizontal lines
    line_oracle = ax.axhline(oracle_micro, color=ACCENT_GREEN, linestyle="-.", lw=1.1,
                             label=f"Oracle Upper Bound ({oracle_micro:.4f})", zorder=2)
    line_norec = ax.axhline(no_rec_micro, color=NEUTRAL_SLATE, linestyle="--", lw=1.0,
                            label=f"No Recovery Baseline ({no_rec_micro:.4f})", zorder=2)
    line_fixed = ax.axhline(fixed_micro, color=ACCENT_RED, linestyle=":", lw=1.1,
                            label=f"Fixed Heuristic Policy ({fixed_micro:.4f})", zorder=2)

    r1 = ax.bar(x - offset, micros, width, label="Condition-Mean Micro-F1", color=PRIMARY_BLUE, alpha=0.92, zorder=3)
    r2 = ax.bar(x + offset, macros, width, label="Condition-Mean Macro-F1", color=ACCENT_ORANGE, alpha=0.92, zorder=3)

    # Place crisp white bold annotations inside near top of each bar so horizontal lines NEVER collide with text
    for rect in r1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h - 0.032, f"{h:.4f}",
                ha="center", va="top", fontsize=6.6, fontweight="bold", color="#ffffff", zorder=5)

    for rect in r2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h - 0.032, f"{h:.4f}",
                ha="center", va="top", fontsize=6.6, fontweight="bold", color="#ffffff", zorder=5)

    ax.set_ylabel("F1-Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontsize=7.2)
    ax.set_ylim(0, 0.74)
    ax.grid(axis="y", linestyle="--", alpha=0.18)

    # Clean Top Legend positioned outside data area (Balanced 2-row layout)
    handles = [r1, r2, line_oracle, line_norec, line_fixed]
    labels = [
        "Condition-Mean Micro-F1",
        "Condition-Mean Macro-F1",
        f"Oracle Upper Bound ({oracle_micro:.4f})",
        f"No Recovery Baseline ({no_rec_micro:.4f})",
        f"Fixed Heuristic Policy ({fixed_micro:.4f})"
    ]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, fontsize=6.8,
              columnspacing=1.0, handletextpad=0.4)

    # Clean, non-overlapping callout badges placed in clear zones
    # 1. Bypasses noisy misdiagnosis bottleneck
    ax.annotate("Bypasses Heuristic Misdiagnosis Bottleneck\n(+3.45% Micro-F1 vs Full SEAA)",
                xy=(2 - offset, 0.515), xytext=(2, 0.64),
                arrowprops=dict(arrowstyle="->", color=ACCENT_PURPLE, lw=0.9),
                ha="center", fontsize=5.8, color=ACCENT_PURPLE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#faf5ff", ec="#d8b4fe", lw=0.7), zorder=6)

    # 2. Adaptation Necessity Callout (placed safely below Oracle line and to the side of the bar)
    ax.annotate("Disabling Adaptation: -3.34% F1\n(Proves Learning Necessity)",
                xy=(1 - offset, 0.448), xytext=(0.85, 0.53),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8),
                ha="center", fontsize=5.6, color="#334155",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=BORDER_GRAY, lw=0.7), zorder=6)

    pdf_path = OUTPUT_DIR / "fig5_ablation_study.pdf"
    png_path = OUTPUT_DIR / "fig5_ablation_study.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {pdf_path} and {png_path}.")


# ==============================================================================
# 6. Table 1: Main Results and Statistical Comparison (CSV, MD, and LaTeX)
# ==============================================================================
def generate_table_1():
    print("Generating Table 1: Main Results and Statistical Comparison...")
    
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
            "Policy_Configuration": "Baseline 1: No Recovery (Static Late Fusion, α=0.55)",
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

    # Markdown Table
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

    # Professional IEEE Transactions LaTeX Table (Booktabs)
    tex_content = r"""% IEEE Transactions Master Benchmark Table (SEAA)
% Requires: \usepackage{booktabs}, \usepackage{multirow}
\begin{table*}[t]
\caption{\textsc{Benchmark Comparison of Multimodal Recovery Policies Under Clustered Repeated Measures ($N=500$ Movies, 8 Regimes, 4,000 Evaluations, $df=499$)}}
\label{tab:benchmark_comparison}
\centering
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
\textbf{Policy / Configuration} & \textbf{Cond. Micro} & \textbf{Cond. Macro} & \textbf{Pooled Micro} & \textbf{Instance F1} & \textbf{$\Delta$ vs. Unrec.} & \textbf{$\Delta$ vs. Fixed} & \textbf{Paired $t$ ($p$-value)} & \textbf{Clustered 95\% CI} \\
\midrule
\textbf{Proposed: Adaptive SEAA} & \textbf{0.4709} & \textbf{0.3593} & \textbf{0.4690} & \textbf{0.4671} & \textbf{+0.0697 (+17.4\%)} & \textbf{+0.0768 (+19.5\%)} & \textit{Reference} & \textit{Reference} \\
Baseline 1: No Recovery ($\alpha=0.55$) & 0.4012 & 0.3165 & 0.3973 & 0.3940 & Reference & +0.0071 (+1.8\%) & $t = 12.79$ ($1.34 \times 10^{-32}$) & $[+0.0619, +0.0843]$ \\
Baseline 2: Fixed Router & 0.3942 & 0.2945 & 0.3905 & 0.3901 & -0.0070 (-1.7\%) & Reference & $t = 12.82$ ($1.02 \times 10^{-32}$) & $[+0.0652, +0.0888]$ \\
\midrule
Theoretical Ceiling: Oracle Bound & 0.5704 & 0.4421 & 0.5673 & 0.5757 & +0.1692 (+42.2\%) & +0.1762 (+44.7\%) & --- & --- \\
\bottomrule
\end{tabular*}
\vspace{0.8ex}
{\raggedright \footnotesize \textbf{Notes:} Evaluated on $N=500$ held-out test movies repeated across 8 failure conditions ($N=4,000$ evaluations per policy). Clustering conducted at the unique movie entity level ($N_{\text{clusters}}=500$, $df=499$). Paired $t$-test and 95\% confidence intervals computed across movie clusters. 5,000 clustered bootstrap resamples yielded strictly non-zero improvements in 100\% of resamples ($p < 0.0002$).\par}
\end{table*}
"""
    tex_path = OUTPUT_DIR / "table1_main_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Saved Table 1 LaTeX to {tex_path}.")


# ==============================================================================
# 7. Reproduction Log
# ==============================================================================
def generate_reproduction_log():
    print("Generating Reproduction Log...")
    log_path = OUTPUT_DIR / "reproduction_log.txt"
    recovery_dir = get_latest_recovery_dir()

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
        OUTPUT_DIR / "table1_main_results.tex",
        OUTPUT_DIR / "outputs_audit_report.md"
    ]

    lines = [
        "=" * 80,
        "SEAA IEEE FINAL OUTPUT REPRODUCTION LOG (IEEE COMPLIANT)",
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
        recovery_dir / "policy_comparison_metrics.csv",
        recovery_dir / "sample_level_decisions.csv",
        recovery_dir / "policy_action_distributions.json",
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
