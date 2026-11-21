"""
SEAA IEEE Single-Column Publication Figures Generator (Strict IEEE Compliant).

Formatting Standards:
- Width: Exactly 3.5 inches (88.9 mm / single column).
- Typography: IEEE standard sans-serif (Arial/DejaVu Sans/Helvetica), 6.5 - 8.0 pt.
- Grayscale Accessibility: Distinct bar hatch patterns (//, \\, xx) + high contrast.
- Archival Cleanliness: Zero embedded chart titles, zero speech bubbles, zero callout arrows.
- Outputs: Both lossless vector PDF and 600 DPI raster PNG.

Outputs directory: outputs/figures/ieee_single_column/
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "ieee_single_column"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strict IEEE Typography Configuration (Single Column: 3.5 inches)
IEEE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.7
}
plt.rcParams.update(IEEE_STYLE)

# IEEE Publication Color Palette
PRIMARY_BLUE = "#1d4ed8"    # SEAA Adaptive / Micro-F1
ACCENT_ORANGE = "#ea580c"   # Fixed Policy / Macro-F1
ACCENT_GREEN = "#15803d"    # Oracle Ceiling
ACCENT_RED = "#b91c1c"      # Degradation Reference
NEUTRAL_SLATE = "#64748b"   # No Recovery Baseline
BORDER_GRAY = "#94a3b8"


def get_latest_recovery_dir() -> Path:
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
        recovery_root / "run_20261025_phase5_adaptive_recovery",
        recovery_root / "run_20261029_105918_adaptive_recovery"
    ]:
        if fallback.exists():
            return fallback
    return recovery_root


# ==============================================================================
# Figure 1A: Regime Performance (Single-Column Horizontal - RECOMMENDED)
# ==============================================================================
def generate_figure_1_horizontal():
    print("Generating Figure 1 (Horizontal): Regime Performance (Single-Column)...")
    recovery_dir = get_latest_recovery_dir()
    source_csv = recovery_dir / "policy_comparison_metrics.csv"
    if not source_csv.exists():
        source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    regimes_clean = [
        "Clean Baseline",
        "Text Trunc. (25%)",
        "Text Mask (35%)",
        "Text Empty",
        "Image Blur (r=5.0)",
        "Image Noise (σ=0.25)",
        "Image Missing",
        "Dual Corrupt."
    ]

    p1 = df["no_recovery_micro_f1"].values
    p2 = df["fixed_policy_micro_f1"].values
    p3 = df["adaptive_policy_micro_f1"].values
    p4 = df["oracle_micro_f1"].values

    p1_avg, p2_avg, p3_avg, p4_avg = np.mean(p1), np.mean(p2), np.mean(p3), np.mean(p4)

    all_labels = ["Condition Average"] + regimes_clean
    all_p1 = np.array([p1_avg] + list(p1))
    all_p2 = np.array([p2_avg] + list(p2))
    all_p3 = np.array([p3_avg] + list(p3))
    all_p4 = np.array([p4_avg] + list(p4))

    # Reverse for top-to-bottom layout
    all_labels = all_labels[::-1]
    all_p1 = all_p1[::-1]
    all_p2 = all_p2[::-1]
    all_p3 = all_p3[::-1]
    all_p4 = all_p4[::-1]

    n_groups = len(all_labels)
    y = np.arange(n_groups)
    bar_height = 0.22

    fig, ax = plt.subplots(figsize=(3.5, 4.8), dpi=300)

    # Demarcate Condition Average (top group)
    ax.axhspan(n_groups - 1 - 0.46, n_groups - 1 + 0.46, color="#f8fafc", alpha=1.0, zorder=1)
    ax.axhline(n_groups - 2 + 0.50, color="#94a3b8", linestyle="--", lw=0.8, zorder=2)

    r1 = ax.barh(y - bar_height, all_p1, bar_height, label="No Recovery",
                 color=NEUTRAL_SLATE, edgecolor="#1e293b", linewidth=0.6, hatch="", zorder=3)
    r2 = ax.barh(y, all_p2, bar_height, label="Fixed Policy",
                 color=ACCENT_ORANGE, edgecolor="#7c2d12", linewidth=0.6, hatch="//", zorder=3)
    r3 = ax.barh(y + bar_height, all_p3, bar_height, label="Adaptive SEAA",
                 color=PRIMARY_BLUE, edgecolor="#0f172a", linewidth=0.6, hatch="xx", zorder=3)

    scatter_handle = ax.scatter(all_p4, y, color=ACCENT_RED, edgecolors="#450a0a",
                                s=28, marker="D", linewidths=1.0, label="Oracle Bound", zorder=5)

    ax.set_xlabel("Micro-F1 Score", fontweight="bold", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=6.8)
    ax.set_xlim(0.0, 0.74)
    ax.set_ylim(-0.6, n_groups - 0.35)
    ax.set_xticks(np.arange(0.0, 0.8, 0.1))
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)

    # Clean text annotations for Condition Average
    avg_y = n_groups - 1
    ax.text(all_p3[avg_y] + 0.012, avg_y + bar_height, f"{all_p3[avg_y]:.3f}",
            va="center", ha="left", fontsize=6.5, fontweight="bold", color=PRIMARY_BLUE, zorder=6)
    ax.text(all_p4[avg_y] + 0.012, avg_y, f"{all_p4[avg_y]:.3f}",
            va="center", ha="left", fontsize=6.2, fontweight="bold", color=ACCENT_RED, zorder=6)

    handles = [r1, r2, r3, scatter_handle]
    labels = ["No Recovery", "Fixed Policy", "Adaptive SEAA", "Oracle Bound"]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, fontsize=6.6,
              columnspacing=1.0, handletextpad=0.4, borderpad=0.35)

    pdf_path = OUTPUT_DIR / "fig1_recovery_by_regime_single_col.pdf"
    png_path = OUTPUT_DIR / "fig1_recovery_by_regime_single_col.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=600)
    plt.close()
    print(f"Saved Figure 1 (Horizontal) to {pdf_path} and {png_path}.")


# ==============================================================================
# Figure 1B: Regime Performance (Single-Column Vertical Alternative)
# ==============================================================================
def generate_figure_1_vertical():
    print("Generating Figure 1 (Vertical Alternative): Regime Performance (Single-Column)...")
    recovery_dir = get_latest_recovery_dir()
    source_csv = recovery_dir / "policy_comparison_metrics.csv"
    if not source_csv.exists():
        source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    regimes_clean = [
        "Clean",
        "Txt Trunc",
        "Txt Mask",
        "Txt Empty",
        "Img Blur",
        "Img Noise",
        "Img Miss",
        "Dual Corr"
    ]

    p1 = df["no_recovery_micro_f1"].values
    p2 = df["fixed_policy_micro_f1"].values
    p3 = df["adaptive_policy_micro_f1"].values
    p4 = df["oracle_micro_f1"].values

    p1_avg, p2_avg, p3_avg, p4_avg = np.mean(p1), np.mean(p2), np.mean(p3), np.mean(p4)

    all_labels = regimes_clean + ["Average"]
    all_p1 = np.append(p1, p1_avg)
    all_p2 = np.append(p2, p2_avg)
    all_p3 = np.append(p3, p3_avg)
    all_p4 = np.append(p4, p4_avg)

    n_groups = len(all_labels)
    x = np.arange(n_groups)
    width = 0.25

    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)

    # Shaded region for summary average
    ax.axvspan(n_groups - 1 - 0.48, n_groups - 1 + 0.48, color="#f8fafc", alpha=1.0, zorder=1)
    ax.axvline(n_groups - 2 + 0.50, color="#94a3b8", linestyle="--", lw=0.8, zorder=2)

    r1 = ax.bar(x - width, all_p1, width, label="No Recovery",
                color=NEUTRAL_SLATE, edgecolor="#1e293b", linewidth=0.5, hatch="", zorder=3)
    r2 = ax.bar(x, all_p2, width, label="Fixed Policy",
                color=ACCENT_ORANGE, edgecolor="#7c2d12", linewidth=0.5, hatch="//", zorder=3)
    r3 = ax.bar(x + width, all_p3, width, label="Adaptive SEAA",
                color=PRIMARY_BLUE, edgecolor="#0f172a", linewidth=0.5, hatch="xx", zorder=3)

    scatter_handle = ax.scatter(x, all_p4, color=ACCENT_RED, edgecolors="#450a0a",
                                s=22, marker="D", linewidths=0.9, label="Oracle Bound", zorder=5)

    ax.set_ylabel("Micro-F1 Score", fontweight="bold", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=6.2, rotation=38, ha="right")
    ax.set_ylim(0.0, 0.78)
    ax.grid(axis="y", linestyle="--", alpha=0.20, zorder=0)

    handles = [r1, r2, r3, scatter_handle]
    labels = ["No Rec.", "Fixed", "SEAA", "Oracle"]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, fontsize=6.0,
              columnspacing=0.6, handletextpad=0.3, borderpad=0.25)

    pdf_path = OUTPUT_DIR / "fig1_recovery_by_regime_single_col_vertical.pdf"
    png_path = OUTPUT_DIR / "fig1_recovery_by_regime_single_col_vertical.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=600)
    plt.close()
    print(f"Saved Figure 1 (Vertical Alternative) to {pdf_path} and {png_path}.")


# ==============================================================================
# Figure 2: Action Distribution Heatmap (Single-Column Compact)
# ==============================================================================
def generate_figure_2_action_heatmap():
    print("Generating Figure 2: Action Distribution Heatmap (Single-Column)...")
    recovery_dir = get_latest_recovery_dir()
    source_csv = recovery_dir / "policy_comparison_metrics.csv"
    if not source_csv.exists():
        source_csv = PROJECT_ROOT / "outputs" / "recovery" / "run_20261025_phase5_adaptive_recovery" / "policy_comparison_metrics.csv"
    df = pd.read_csv(source_csv)

    conditions_clean = [
        "Clean Baseline",
        "Text Trunc. (25%)",
        "Text Mask (35%)",
        "Text Empty",
        "Image Blur (r=5.0)",
        "Image Noise (σ=0.25)",
        "Image Missing",
        "Dual Corrupt."
    ]

    action_cols = [
        "actions_static_fusion_pct",
        "actions_text_only_pct",
        "actions_image_only_pct",
        "actions_rel_weighted_pct",
        "actions_conservative_pct"
    ]
    action_headers = [
        "Static\n(α=0.55)",
        "Text Iso.\n(α=1.0)",
        "Vision Iso.\n(α=0.0)",
        "Dyn. Gate\n(α(x))",
        "Conserv.\n(50/50)"
    ]

    heatmap_data = df[action_cols].values

    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="Blues",
        xticklabels=action_headers,
        yticklabels=conditions_clean,
        cbar_kws={
            "shrink": 0.85,
            "pad": 0.03,
            "aspect": 18
        },
        vmin=0.0,
        vmax=100.0,
        linewidths=0.7,
        linecolor="#ffffff",
        ax=ax
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            if val >= 45.0:
                txt_color = "#ffffff"
                weight = "bold"
                size = 6.6
            elif val > 0.0:
                txt_color = "#0f172a"
                weight = "bold"
                size = 6.6
            else:
                txt_color = "#94a3b8"
                weight = "normal"
                size = 6.0
            ax.text(j + 0.5, i + 0.5, f"{val:.1f}", ha="center", va="center",
                    color=txt_color, fontweight=weight, fontsize=size)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6.5, length=0, pad=3)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Action Frequency", fontsize=7.0, fontweight="bold", labelpad=6)

    ax.tick_params(axis="y", length=0, pad=3, labelsize=6.8)
    ax.tick_params(axis="x", length=0, pad=3, labelsize=6.8)

    pdf_path = OUTPUT_DIR / "fig2_action_heatmap_single_col.pdf"
    png_path = OUTPUT_DIR / "fig2_action_heatmap_single_col.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=600)
    plt.close()
    print(f"Saved Figure 2 to {pdf_path} and {png_path}.")


# ==============================================================================
# Figure 3: Component Ablation Study (Single-Column Clean Vertical)
# ==============================================================================
def generate_figure_3_ablation_study():
    print("Generating Figure 3: Component Ablation Study (Single-Column)...")
    source_csv = PROJECT_ROOT / "outputs" / "recovery" / "ablation_study_results.csv"
    df = pd.read_csv(source_csv)

    policy_labels = [
        "Full SEAA\n(Trained)",
        "Learning Off\n(Untrained)",
        "No Diagnoser\n(Coarse)",
        "Random\nAction"
    ]

    df_variants = df.iloc[[0, 1, 2, 3]].copy()
    micros = df_variants["Condition_Mean_Micro_F1"].values
    macros = df_variants["Condition_Mean_Macro_F1"].values

    no_rec_micro = df[df["Policy"] == "Baseline_No_Recovery"]["Condition_Mean_Micro_F1"].values[0]
    fixed_micro = df[df["Policy"] == "Baseline_Fixed_Policy"]["Condition_Mean_Micro_F1"].values[0]
    oracle_micro = df[df["Policy"] == "Oracle_Upper_Bound"]["Condition_Mean_Micro_F1"].values[0]

    x = np.arange(len(policy_labels))
    width = 0.30
    offset = 0.17

    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=300)

    # Reference horizontal lines (zorder=1, behind bars)
    line_oracle = ax.axhline(oracle_micro, color=ACCENT_GREEN, linestyle="-.", lw=1.0,
                             label=f"Oracle ({oracle_micro:.3f})", zorder=1)
    line_norec = ax.axhline(no_rec_micro, color=NEUTRAL_SLATE, linestyle="--", lw=0.9,
                            label=f"No Recovery ({no_rec_micro:.3f})", zorder=1)
    line_fixed = ax.axhline(fixed_micro, color=ACCENT_RED, linestyle=":", lw=1.0,
                            label=f"Fixed Policy ({fixed_micro:.3f})", zorder=1)

    # Completely opaque bars (zorder=3) so background lines never slice through
    r1 = ax.bar(x - offset, micros, width, label="Micro-F1",
                color=PRIMARY_BLUE, edgecolor="#0f172a", linewidth=0.6, hatch="//", alpha=1.0, zorder=3)
    r2 = ax.bar(x + offset, macros, width, label="Macro-F1",
                color=ACCENT_ORANGE, edgecolor="#7c2d12", linewidth=0.6, hatch="\\\\", alpha=1.0, zorder=3)

    # Clean text values placed on top of bars with opaque white background
    for rect in r1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=PRIMARY_BLUE,
                bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=1.0), zorder=6)

    for rect in r2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=ACCENT_ORANGE,
                bbox=dict(boxstyle="round,pad=0.10", fc="#ffffff", ec="none", alpha=1.0), zorder=6)

    ax.set_ylabel("F1-Score", fontweight="bold", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontsize=6.8)
    ax.set_ylim(0.0, 0.72)
    ax.grid(axis="y", linestyle="--", alpha=0.20, zorder=0)

    handles = [r1, r2, line_oracle, line_norec, line_fixed]
    labels = [
        "Micro-F1",
        "Macro-F1",
        f"Oracle ({oracle_micro:.3f})",
        f"No Recovery ({no_rec_micro:.3f})",
        f"Fixed Policy ({fixed_micro:.3f})"
    ]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=True, facecolor="#ffffff", edgecolor=BORDER_GRAY, fontsize=6.2,
              columnspacing=0.8, handletextpad=0.35, borderpad=0.3)

    pdf_path = OUTPUT_DIR / "fig3_ablation_study_single_col.pdf"
    png_path = OUTPUT_DIR / "fig3_ablation_study_single_col.png"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(png_path, format="png", dpi=600)
    plt.close()
    print(f"Saved Figure 3 to {pdf_path} and {png_path}.")


if __name__ == "__main__":
    print(f"Target Output Directory: {OUTPUT_DIR}")
    generate_figure_1_horizontal()
    generate_figure_1_vertical()
    generate_figure_2_action_heatmap()
    generate_figure_3_ablation_study()
    print("All IEEE single-column figures generated successfully!")
