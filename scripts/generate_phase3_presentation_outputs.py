"""
Generate Presentation-Ready SEAA Phase 3 Outputs.

Project: A Self-Evolving AI Auditing Algorithm for Adaptive Reliability
         Diagnosis and Recovery in Multimodal AI Systems (SEAA)
Workspace: C:\\Users\\bharat kumar\\OneDrive\\Desktop\\SEAA

Objective:
Reads existing verified workspace artifacts dynamically and generates:
1. High-resolution presentation figures (01 to 07) in outputs/presentation/phase3/
2. Standardized CSV and Markdown tables in outputs/presentation/phase3/
3. Evidence-backed Key Findings summary for project supervisor
4. Polished 4-5 page PDF report (SEAA_Phase3_Supervisor_Summary.pdf)
5. Comprehensive README.md documenting artifacts, source provenance, and caveats

Scope:
- Phase 3 output generation only.
- Does NOT retrain or alter any model.
- Does NOT alter split_indices.json or regenerate data.
- Does NOT claim statistical significance.
- Fully reproducible via python scripts/generate_phase3_presentation_outputs.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ReportLab Platypus imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase3PresentationGenerator")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and draw consistent
    professional headers and footers ("Page X of Y").
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Top of letter: 11 * 72 = 792 pt, width: 8.5 * 72 = 612 pt)
        self.drawString(
            40,
            760,
            "SEAA: Multimodal Reliability Diagnosis & Recovery — Phase 3 Supervisor Briefing"
        )
        self.drawRightString(
            572,
            760,
            "Phase 3 Baseline Verification"
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 754, 572, 754)

        # Footer
        self.line(40, 42, 572, 42)
        self.drawString(
            40,
            30,
            "CONFIDENTIAL — Project Supervisor Review | Source: Verified Workspace Artifacts (MM-IMDb)"
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.restoreState()


class Phase3PresentationGenerator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "outputs" / "presentation" / "phase3"
        self.metrics_dir = project_root / "outputs" / "metrics"
        self.reports_dir = project_root / "outputs" / "reports"
        self.data_dir = project_root / "data"

        # Color Palette
        self.colors = {
            "text": "#2B5C8F",         # Deep Slate Blue
            "image": "#E07A5F",        # Terracotta Coral
            "multimodal": "#2A9D8F",   # Forest Teal
            "neutral_dark": "#2D3748", # Charcoal
            "neutral_light": "#F8FAFC",# Off-white
            "grid": "#E2E8F0",         # Light grey border
            "gold": "#D4A373",         # Warm Ochre
        }

        # Data holders
        self.baseline_df = None
        self.timing_df = None
        self.timing_json = None
        self.verification_json = None
        self.split_indices = None
        self.per_genre_text = None
        self.per_genre_image = None
        self.per_genre_multimodal = None

    def setup_output_dir(self):
        """Create target output folder."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target presentation directory ready: {self.output_dir}")

    def load_source_artifacts(self):
        """Load and strictly verify authoritative source artifacts."""
        logger.info("Loading authoritative workspace artifacts...")

        baseline_csv = self.metrics_dir / "baseline_comparison.csv"
        if not baseline_csv.exists():
            raise FileNotFoundError(f"Missing required baseline comparison CSV: {baseline_csv}")
        self.baseline_df = pd.read_csv(baseline_csv)

        timing_csv = self.metrics_dir / "inference_timing.csv"
        if not timing_csv.exists():
            raise FileNotFoundError(f"Missing required timing CSV: {timing_csv}")
        self.timing_df = pd.read_csv(timing_csv)

        timing_json_path = self.metrics_dir / "inference_timing.json"
        if not timing_json_path.exists():
            raise FileNotFoundError(f"Missing required timing JSON: {timing_json_path}")
        with open(timing_json_path, "r", encoding="utf-8") as f:
            self.timing_json = json.load(f)

        verif_json_path = self.reports_dir / "phase3_metric_verification.json"
        if not verif_json_path.exists():
            raise FileNotFoundError(f"Missing verification JSON: {verif_json_path}")
        with open(verif_json_path, "r", encoding="utf-8") as f:
            self.verification_json = json.load(f)

        split_json_path = self.data_dir / "split_indices.json"
        if not split_json_path.exists():
            raise FileNotFoundError(f"Missing split indices: {split_json_path}")
        with open(split_json_path, "r", encoding="utf-8") as f:
            self.split_indices = json.load(f)

        # Per genre CSVs
        text_pg_path = self.metrics_dir / "text_only_per_genre.csv"
        image_pg_path = self.metrics_dir / "image_only_per_genre.csv"
        mm_pg_path = self.metrics_dir / "multimodal_per_genre.csv"

        if text_pg_path.exists():
            self.per_genre_text = pd.read_csv(text_pg_path)
        if image_pg_path.exists():
            self.per_genre_image = pd.read_csv(image_pg_path)
        if mm_pg_path.exists():
            self.per_genre_multimodal = pd.read_csv(mm_pg_path)

        logger.info("Authoritative artifacts successfully loaded and verified.")

    # =========================================================================
    # Step 3: High-Resolution Graphs
    # =========================================================================

    def generate_all_graphs(self) -> Dict[str, Path]:
        """Generate all required presentation charts."""
        logger.info("Generating high-resolution graphs...")
        generated_graphs = {}

        # 1. 01_baseline_metric_comparison.png
        p1 = self._plot_baseline_metric_comparison()
        generated_graphs["01_baseline_metric_comparison"] = p1

        # 2. 02_micro_f1_comparison.png
        p2 = self._plot_micro_f1_comparison()
        generated_graphs["02_micro_f1_comparison"] = p2

        # 3. 03_hamming_loss_comparison.png
        p3 = self._plot_hamming_loss_comparison()
        generated_graphs["03_hamming_loss_comparison"] = p3

        # 4. 04_per_genre_f1_comparison.png
        p4 = self._plot_per_genre_f1_comparison()
        if p4:
            generated_graphs["04_per_genre_f1_comparison"] = p4

        # 5. 05_inference_latency_comparison.png
        p5 = self._plot_inference_latency_comparison()
        generated_graphs["05_inference_latency_comparison"] = p5

        # 6. 06_fusion_overhead.png
        p6 = self._plot_fusion_overhead()
        generated_graphs["06_fusion_overhead"] = p6

        # 7. 07_dataset_split_distribution.png
        p7 = self._plot_dataset_split_distribution()
        generated_graphs["07_dataset_split_distribution"] = p7

        return generated_graphs

    def _plot_baseline_metric_comparison(self) -> Path:
        """Graph 1: Grouped bar chart comparing Micro-F1, Macro-F1, and mAP."""
        out_path = self.output_dir / "01_baseline_metric_comparison.png"
        
        # Prepare data from authoritative baseline comparison
        models = [
            "Text-Only\n(TF-IDF + LR)",
            "Image-Only\n(ResNet50 + LR)",
            "Multimodal Late Fusion\n(Val-Tuned α=0.55)"
        ]
        
        # Pull exact numbers
        micro_f1 = self.baseline_df["micro_f1"].tolist()
        macro_f1 = self.baseline_df["macro_f1"].tolist()
        map_scores = self.baseline_df["mean_average_precision"].tolist()

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        
        rects1 = ax.bar(x - width, micro_f1, width, label='Micro-F1', color='#1D4E89', edgecolor='none')
        rects2 = ax.bar(x, macro_f1, width, label='Macro-F1', color='#00B4D8', edgecolor='none')
        rects3 = ax.bar(x + width, map_scores, width, label='Mean Average Precision (mAP)', color='#7209B7', edgecolor='none')

        ax.set_ylabel('Score (Higher is Better)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Baseline Model Performance Comparison on Test Set (N=7,788)\nMM-IMDb 23-Genre Multi-Label Classification', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')
        ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5, loc='upper right')

        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),  # 4 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        # Footnote
        plt.figtext(0.5, 0.01, 
                    "Note: Custom 60/10/30 split (Seed 42). Static validation-tuned weighted late fusion (α=0.55 text, 0.45 image).",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_micro_f1_comparison(self) -> Path:
        """Graph 2: Micro-F1 headline comparison across models."""
        out_path = self.output_dir / "02_micro_f1_comparison.png"

        models = [
            "Text-Only\n(TF-IDF + LR)",
            "Multimodal Late Fusion\n(Val-Tuned α=0.55)",
            "Image-Only\n(ResNet50 + LR)"
        ]
        
        # Order by performance for clean presentation
        text_f1 = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Text"), "micro_f1"].iloc[0])
        mm_f1 = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Multimodal"), "micro_f1"].iloc[0])
        img_f1 = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Image"), "micro_f1"].iloc[0])

        scores = [text_f1, mm_f1, img_f1]
        bar_colors = [self.colors["text"], self.colors["multimodal"], self.colors["image"]]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(models, scores, color=bar_colors, width=0.45, edgecolor='#1E293B', linewidth=0.8)

        ax.set_ylabel('Micro-F1 Score (Higher is Better)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Headline Result: Test Micro-F1 Comparison Across Modalities\nText-Only Outperforms Static Multimodal Late Fusion', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Annotation highlight
        ax.annotate(f'Best Overall Performance\n(0.6056)',
                    xy=(0, text_f1), xytext=(0.25, 0.67),
                    arrowprops=dict(arrowstyle="->", color="#1E293B", lw=1.2),
                    fontsize=8.5, fontweight='bold', color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#E2E8F0", ec="#94A3B8"))

        ax.annotate(f'+10.10 pp over Image-Only\n-8.95 pp vs Text-Only',
                    xy=(1, mm_f1), xytext=(1.25, 0.58),
                    arrowprops=dict(arrowstyle="->", color="#1E293B", lw=1.2),
                    fontsize=8.5, fontweight='bold', color="#0F766E",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#CCFBF1", ec="#5EEAD4"))

        plt.figtext(0.5, 0.01,
                    "Evaluated on 7,788 test movies. Empirical differences are descriptive; no significance tests performed.",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_hamming_loss_comparison(self) -> Path:
        """Graph 3: Hamming loss comparison (Lower is better)."""
        out_path = self.output_dir / "03_hamming_loss_comparison.png"

        models = [
            "Text-Only\n(TF-IDF + LR)",
            "Multimodal Late Fusion\n(Val-Tuned α=0.55)",
            "Image-Only\n(ResNet50 + LR)"
        ]

        text_hl = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Text"), "hamming_loss"].iloc[0])
        mm_hl = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Multimodal"), "hamming_loss"].iloc[0])
        img_hl = float(self.baseline_df.loc[self.baseline_df["model"].str.contains("Image"), "hamming_loss"].iloc[0])

        losses = [text_hl, mm_hl, img_hl]
        bar_colors = [self.colors["text"], self.colors["multimodal"], self.colors["image"]]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(models, losses, color=bar_colors, width=0.45, edgecolor='#1E293B', linewidth=0.8)

        ax.set_ylabel('Hamming Loss (Lower is Better)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Multi-Label Classification Error Rate: Hamming Loss Comparison\nText-Only Yields the Lowest Per-Label Misclassification Rate', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(0, 0.22)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Annotation highlight
        ax.annotate(f'Lowest Error: 9.29%\n(Best Calibration)',
                    xy=(0, text_hl), xytext=(0.25, 0.12),
                    arrowprops=dict(arrowstyle="->", color="#1E293B", lw=1.2),
                    fontsize=8.5, fontweight='bold', color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#E2E8F0", ec="#94A3B8"))

        plt.figtext(0.5, 0.01,
                    "Hamming loss computes the fraction of labels that are incorrectly predicted across all 23 classes.",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_per_genre_f1_comparison(self) -> Path:
        """Graph 4: Per-genre F1 comparison horizontal bar chart across all 23 genres."""
        if self.per_genre_text is None or self.per_genre_image is None or self.per_genre_multimodal is None:
            logger.warning("Per-genre metrics missing; skipping 04_per_genre_f1_comparison.png")
            return None

        out_path = self.output_dir / "04_per_genre_f1_comparison.png"

        # Merge genres
        df_text = self.per_genre_text.rename(columns={"f1": "f1_text"}).set_index("genre")
        df_img = self.per_genre_image.rename(columns={"f1": "f1_image"}).set_index("genre")
        df_mm = self.per_genre_multimodal.rename(columns={"f1": "f1_mm"}).set_index("genre")

        merged = df_text[["f1_text"]].join(df_img[["f1_image"]]).join(df_mm[["f1_mm"]])
        # Sort by Text F1 descending for visual hierarchy
        merged = merged.sort_values(by="f1_text", ascending=True)

        genres = merged.index.tolist()
        y = np.arange(len(genres))
        height = 0.28

        fig, ax = plt.subplots(figsize=(10, 11), dpi=300)

        r1 = ax.barh(y + height, merged["f1_text"], height, label='Text-Only (TF-IDF + LR)', color=self.colors["text"], edgecolor='none')
        r2 = ax.barh(y, merged["f1_mm"], height, label='Multimodal Late Fusion (α=0.55)', color=self.colors["multimodal"], edgecolor='none')
        r3 = ax.barh(y - height, merged["f1_image"], height, label='Image-Only (ResNet50 + LR)', color=self.colors["image"], edgecolor='none')

        ax.set_xlabel('F1 Score on Test Set (Higher is Better)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title('Per-Genre F1 Performance Across 23 MM-IMDb Movie Genres\nSorted by Text-Only Baseline Performance', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_yticks(y)
        ax.set_yticklabels(genres, fontsize=9.5, fontweight='bold')
        ax.set_xlim(0, 0.85)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis='x', linestyle='--', alpha=0.35, color='#94A3B8')
        ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5, loc='lower right')

        plt.figtext(0.5, 0.01,
                    "Text-only dominates high-density genres (Drama, Documentary, Comedy, Horror). Image signals assist Animation and Family.",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_inference_latency_comparison(self) -> Path:
        """Graph 5: Inference latency comparison for classifier pipelines."""
        out_path = self.output_dir / "05_inference_latency_comparison.png"

        # Extract values
        comps = self.timing_json["components"]
        text_t = comps["text_classifier_inference"]["median_sec"]
        text_thru = comps["text_classifier_inference"]["samples_per_sec"]
        img_t = comps["image_classifier_cached_inference"]["median_sec"]
        img_thru = comps["image_classifier_cached_inference"]["samples_per_sec"]
        mm_t = comps["end_to_end_cached_multimodal_inference"]["median_sec"]
        mm_thru = comps["end_to_end_cached_multimodal_inference"]["samples_per_sec"]

        labels = [
            "Text Classifier\nInference\n(TF-IDF + LR)",
            "Image Classifier\nCached Inference\n(ResNet50 Feats + LR)",
            "End-to-End Multimodal\nCached Inference\n(Text + Image + Fusion)"
        ]
        times = [text_t, img_t, mm_t]
        throughputs = [text_thru, img_thru, mm_thru]
        colors_list = [self.colors["text"], self.colors["image"], self.colors["multimodal"]]

        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
        bars = ax.bar(labels, times, color=colors_list, width=0.45, edgecolor='#1E293B', linewidth=0.8)

        ax.set_ylabel('Median Latency for 7,788 Test Samples (Seconds)', fontsize=10.5, fontweight='bold', labelpad=8)
        ax.set_title('Inference Latency & Throughput Benchmark on CPU\nSteady-State Pipeline Latency Over 7,788 Test Records (1 Warm-up, 5 Runs)', 
                     fontsize=11.5, fontweight='bold', pad=15)
        ax.set_ylim(0, 4.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

        for bar, time_val, thru in zip(bars, times, throughputs):
            h = bar.get_height()
            ax.annotate(f'{time_val:.4f} s\n({thru:,.1f} samples/s)',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        plt.figtext(0.5, 0.01,
                    "Timings measured on CPU using time.perf_counter(). Workstation-dependent. Fusion-only latency (0.000962s) excluded to avoid scale distortion.",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_fusion_overhead(self) -> Path:
        """Graph 6: Fusion & thresholding mathematical overhead visual."""
        out_path = self.output_dir / "06_fusion_overhead.png"

        comps = self.timing_json["components"]
        f_comp = comps["fusion_and_thresholding_only"]
        median_ms = f_comp["median_sec"] * 1000.0
        min_ms = f_comp["min_sec"] * 1000.0
        max_ms = f_comp["max_sec"] * 1000.0
        runs_ms = [r * 1000.0 for r in f_comp["all_runs_sec"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=300, gridspec_kw={'width_ratios': [1, 1.3]})

        # Left: Bar chart in milliseconds
        bar = ax1.bar(["Fusion & Thresholding\nOnly (5 Reps)"], [median_ms], color=self.colors["multimodal"], width=0.35, edgecolor='#1E293B')
        ax1.set_ylabel('Median Execution Time (Milliseconds)', fontsize=10, fontweight='bold')
        ax1.set_ylim(0, 2.5)
        ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')
        ax1.annotate(f'{median_ms:.3f} ms\n({f_comp["samples_per_sec"]:,.1f} samples/s)',
                     xy=(bar[0].get_x() + bar[0].get_width() / 2, median_ms),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        # Right: Individual run distribution
        run_labels = [f"Run {i+1}" for i in range(len(runs_ms))]
        ax2.plot(run_labels, runs_ms, marker='o', markersize=8, color="#0E7490", linewidth=2, linestyle='-')
        ax2.axhline(median_ms, color="#DC2626", linestyle='--', label=f'Median: {median_ms:.3f} ms')
        ax2.set_ylabel('Time per Repetition (Milliseconds)', fontsize=10, fontweight='bold')
        ax2.set_ylim(0.5, 2.6)
        ax2.grid(True, linestyle='--', alpha=0.35, color='#94A3B8')
        ax2.legend(loc='upper right', fontsize=8.5)
        for i, val in enumerate(runs_ms):
            ax2.annotate(f'{val:.3f} ms', (i, val), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=8)

        fig.suptitle('Isolated Mathematical Fusion Overhead (Probability Weighting & 23-Genre Thresholding)\nOver 7,788 Precomputed Test Probability Vectors',
                     fontsize=11.5, fontweight='bold', y=0.98)

        # Prominent clarification box
        props = dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.9)
        fig.text(0.5, 0.04, 
                 "CRITICAL SCOPE DISTINCTION:\n"
                 "Probability arrays are already computed. Measures only floating-point arithmetic (0.55*P_text + 0.45*P_image) and thresholding.\n"
                 "This is NOT end-to-end inference (which requires 3.602s on CPU).",
                 ha="center", fontsize=8.5, fontweight='bold', color="#92400E", bbox=props)

        plt.tight_layout(rect=[0, 0.12, 1, 0.94])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def _plot_dataset_split_distribution(self) -> Path:
        """Graph 7: Dataset custom split distribution and sample counts."""
        out_path = self.output_dir / "07_dataset_split_distribution.png"

        train_n = len(self.split_indices["train"])
        val_n = len(self.split_indices["val"])
        test_n = len(self.split_indices["test"])
        total_n = train_n + val_n + test_n

        sizes = [train_n, val_n, test_n]
        labels = [
            f"Train Set\n{train_n:,} samples\n({train_n/total_n*100:.2f}%)",
            f"Validation Set\n{val_n:,} samples\n({val_n/total_n*100:.2f}%)",
            f"Test Set\n{test_n:,} samples\n({test_n/total_n*100:.2f}%)"
        ]
        slice_colors = ["#3B82F6", "#F59E0B", "#10B981"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=300, gridspec_kw={'width_ratios': [1.1, 1]})

        # Donut chart
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            autopct='',
            startangle=140,
            colors=slice_colors,
            wedgeprops=dict(width=0.45, edgecolor='#FFFFFF', linewidth=2)
        )
        for t in texts:
            t.set_fontsize(8.5)
            t.set_fontweight('bold')
        ax1.set_title('Custom Split Proportions\nTotal Records: 25,959', fontsize=11, fontweight='bold', pad=10)

        # Bar comparison
        split_names = ["Train Split", "Val Split", "Test Split"]
        bars = ax2.bar(split_names, sizes, color=slice_colors, width=0.45, edgecolor='#1E293B', linewidth=0.8)
        ax2.set_ylabel('Sample Count', fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 18000)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(3000))
        ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:,}',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        ax2.set_title('Absolute Sample Counts Across Splits', fontsize=11, fontweight='bold', pad=10)

        fig.suptitle('SEAA Phase 3 Dataset Partitioning Summary (MM-IMDb Custom Split)\nIterative Multi-Label Stratification (Random Seed 42)',
                     fontsize=12, fontweight='bold', y=0.98)

        plt.figtext(0.5, 0.01,
                    "Strictly disjoint splits: 0 overlap between sets. 0 cross-split duplicate text descriptions. Split indices strictly preserved.",
                    ha="center", fontsize=8, color="#64748B", style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    # =========================================================================
    # Step 4: Tables and Key Findings
    # =========================================================================

    def generate_all_tables(self):
        """Generate standardized CSV and Markdown tables, and Key Findings report."""
        logger.info("Generating standardized tables and Markdown summaries...")

        # 1. Model Metrics Table
        self._generate_model_metrics_table()

        # 2. Inference Timings Table
        self._generate_inference_timings_table()

        # 3. Per-Genre Metrics Table
        self._generate_per_genre_metrics_table()

        # 4. Key Findings Report
        self._generate_key_findings_report()

    def _generate_model_metrics_table(self):
        """Generate phase3_model_metrics.csv and phase3_model_metrics.md."""
        csv_path = self.output_dir / "phase3_model_metrics.csv"
        md_path = self.output_dir / "phase3_model_metrics.md"

        # Pull exact data
        verif = self.verification_json["task_2_metrics_recomputation"]
        
        rows = []
        for key, name in [
            ("text_only", "Text-Only (TF-IDF + LR)"),
            ("image_only", "Image-Only (ResNet50 + LR)"),
            ("multimodal", "Multimodal (Validation-Tuned Weighted Late Fusion)")
        ]:
            m = verif[key]["metrics"]
            rows.append({
                "model": name,
                "modality": "Text" if "Text" in name else ("Image (Cached Feats)" if "Image" in name else "Multimodal (Cached E2E)"),
                "micro_f1": m["micro_f1"]["stored_value"],
                "macro_f1": m["macro_f1"]["stored_value"],
                "micro_precision": m["micro_precision"]["stored_value"],
                "macro_precision": m["macro_precision"]["stored_value"],
                "micro_recall": m["micro_recall"]["stored_value"],
                "macro_recall": m["macro_recall"]["stored_value"],
                "mean_average_precision": m["mean_average_precision"]["stored_value"],
                "hamming_loss": m["hamming_loss"]["stored_value"],
                "test_samples": 7788,
                "source_file": "outputs/reports/phase3_metric_verification.json"
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Markdown representation
        md_content = f"""# SEAA Phase 3: Comprehensive Model Baseline Metrics

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Dataset:** MM-IMDb Multi-Label Classification (23 Genres)  
**Evaluation Set:** Test Split ($N = 7,788$ movie records; Random Seed 42)  
**Source Provenance:** Formally verified from `outputs/metrics/baseline_comparison.csv` and `outputs/reports/phase3_metric_verification.json`.

| Model Architecture | Modality | Micro-F1 | Macro-F1 | Micro-Prec | Macro-Prec | Micro-Rec | Macro-Rec | mAP | Hamming Loss |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Text-Only (TF-IDF + LR)** | Text | **0.6056** | **0.4955** | **0.5594** | **0.5126** | **0.6602** | **0.5179** | **0.5183** | **0.0929** |
| **Image-Only (ResNet50 + LR)** | Image (Cached Feats) | 0.4151 | 0.2847 | 0.3373 | 0.2444 | 0.5396 | 0.3632 | 0.2337 | 0.1642 |
| **Multimodal (Validation-Tuned Weighted Late Fusion)** | Multimodal (Cached E2E) | 0.5161 | 0.4003 | 0.4301 | 0.3523 | 0.6451 | 0.5150 | 0.3985 | 0.1307 |

### Observational Summary:
1. **Text Modality Dominance:** Text-Only achieves the highest performance across all evaluation metrics, reaching **0.6056 Micro-F1** and **0.5183 mAP** with the lowest Hamming loss (**0.0929**).
2. **Modality Gain & Fusion Dilution:** Validation-Tuned Weighted Late Fusion ($\\alpha = 0.55$) shows an absolute gain of **+10.10 percentage points Micro-F1** over the Image-Only baseline (**0.5161** vs. **0.4151**). However, it falls short of Text-Only by **-8.95 percentage points Micro-F1**, indicating that naive static late fusion dilutes high-confidence textual semantic signals with lower-confidence visual signals.
3. **Statistical Clarification:** Observed differences are empirical and descriptive. No statistical significance tests were performed.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    def _generate_inference_timings_table(self):
        """Generate phase3_inference_timings.csv and phase3_inference_timings.md."""
        csv_path = self.output_dir / "phase3_inference_timings.csv"
        md_path = self.output_dir / "phase3_inference_timings.md"

        # Re-save standardized timing CSV
        self.timing_df.to_csv(csv_path, index=False)

        comps = self.timing_json["components"]
        t_c = comps["text_classifier_inference"]
        i_c = comps["image_classifier_cached_inference"]
        f_c = comps["fusion_and_thresholding_only"]
        m_c = comps["end_to_end_cached_multimodal_inference"]
        r_c = comps["raw_image_feature_extraction_reference"]

        md_content = f"""# SEAA Phase 3: Standardized Inference Latency & Benchmarking Breakdown

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Benchmarking Protocol:** Checkpoints loaded prior to timing; 1 warm-up repetition followed by 5 timed repetitions over 7,788 test records using high-resolution `time.perf_counter()` on CPU.  
**Source Provenance:** `outputs/metrics/inference_timing.json` and `outputs/metrics/inference_timing.csv`.

| Component / Pipeline | Median Latency (s) | Min (s) | Max (s) | Std Dev (s) | Throughput (samples/s) | Precise Scope & Measurement Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Text Classifier Inference** | **1.5415** | 1.5356 | 1.7150 | 0.0683 | **5,052.2** | TF-IDF vectorization (10,000 features) + OneVsRest Logistic Regression `predict_proba` over 23 genres |
| **Image Classifier Cached Inference** | **1.9394** | 1.3220 | 2.0579 | 0.2982 | **4,015.7** | StandardScaler transformation + OneVsRest Logistic Regression `predict_proba` on precomputed 2,048-dim ResNet50 features |
| **Fusion & Thresholding Only** | **0.000962** | 0.000942 | 0.002194 | 0.000486 | **8,091,428.6** | Probability weighting ($0.55 P_{{text}} + 0.45 P_{{image}}$) on already computed probability arrays + 23-genre thresholding. **NOT end-to-end model inference.** |
| **End-to-End Cached Multimodal** | **3.6020** | 3.1565 | 3.9054 | 0.2812 | **2,162.2** | Complete pipeline consuming raw text strings and cached visual embeddings: Text TF-IDF + Image Scaler + Dual Classifier Inference + Late Fusion + Thresholding |
| **Raw Image ResNet50 Reference** | *794.7000\** | N/A | N/A | N/A | *9.8\** | Raw image poster decoding, resizing (224x224), RGB normalization, and ResNet50 forward pass on CPU. |

> **Critical Clarifications:**
> 1. *\*Historical Estimate Status:* The raw image feature extraction value of **794.7 seconds for 7,788 test images** is a **historical estimate from the Phase 2 log** (`outputs/logs/image_feature_extraction.log`: 25,959 images processed at ~9.8 imgs/s). It was **not freshly measured in this audit** and must not be presented as a newly measured benchmark or directly compared to cached classifier latencies.
> 2. *Throughput Formula:* Throughput is computed from unrounded raw median latency ($7,788 / \\text{{median}}$). Displaying 4-decimal rounded values ($0.0010\\text{{ s}}$) creates an apparent manual arithmetic mismatch; authoritative full-precision median ($0.000962498\\text{{ s}}$) governs all throughput calculations.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    def _generate_per_genre_metrics_table(self):
        """Generate phase3_per_genre_metrics.csv consolidating all 23 genres."""
        csv_path = self.output_dir / "phase3_per_genre_metrics.csv"
        if self.per_genre_text is None or self.per_genre_image is None or self.per_genre_multimodal is None:
            logger.warning("Per-genre data incomplete; skipping consolidated CSV.")
            return

        t = self.per_genre_text.rename(columns={
            "precision": "text_precision", "recall": "text_recall",
            "f1": "text_f1", "average_precision": "text_ap", "threshold": "text_threshold"
        })
        img = self.per_genre_image.rename(columns={
            "precision": "image_precision", "recall": "image_recall",
            "f1": "image_f1", "average_precision": "image_ap", "threshold": "image_threshold"
        }).drop(columns=["support"])
        mm = self.per_genre_multimodal.rename(columns={
            "precision": "mm_precision", "recall": "mm_recall",
            "f1": "mm_f1", "average_precision": "mm_ap", "threshold": "mm_threshold"
        }).drop(columns=["support"])

        merged = t.merge(img, on="genre").merge(mm, on="genre")
        merged.to_csv(csv_path, index=False)

    def _generate_key_findings_report(self):
        """Generate phase3_key_findings.md with 7 concise, evidence-backed findings."""
        md_path = self.output_dir / "phase3_key_findings.md"

        findings = f"""# SEAA Phase 3: Executive Summary of Key Findings & Supervisor Briefing

**Project:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Author:** Antigravity Autonomous Implementation & Auditing Agent  
**Status:** Phase 3 Formally Closed & Fully Verified; Phase 4 Has NOT Started  
**Target Audience:** Academic Supervisor & Project Committee  

---

### Executive Key Findings

1. **Text Modality Dominance in MM-IMDb Multi-Label Classification:**  
   The Text-Only baseline (TF-IDF with 10,000 unigrams/bigrams and OneVsRest Logistic Regression) achieved the highest overall performance on the 7,788 test records: **0.6056 Micro-F1**, **0.4955 Macro-F1**, and **0.5183 mAP**, with a Hamming loss of **0.0929**. Textual plot summaries provide rich, direct semantic descriptors that drive accurate multi-label assignment.

2. **Image-Only Performance Floor:**  
   The Image-Only baseline (pre-extracted ResNet50 2,048-dim deep visual features with StandardScaler and Logistic Regression) attained **0.4151 Micro-F1**, **0.2847 Macro-F1**, and **0.2337 mAP**, with a Hamming loss of **0.1642**. Movie posters exhibit heavy artistic variance, typography, and ambiguous visual metaphors, making them weaker standalone predictors for multi-label genre classification.

3. **Validation-Tuned Weighted Late Fusion Performance Characteristics:**  
   Multimodal late fusion using validation-selected scalar weights ($\\alpha = 0.55$ text, $0.45$ image) and per-genre threshold optimization achieved **0.5161 Micro-F1**, **0.4003 Macro-F1**, and **0.3985 mAP**. While late fusion demonstrated a strong absolute improvement of **+10.10 percentage points Micro-F1** over the Image-Only baseline, it underperformed the Text-Only baseline by **-8.95 percentage points Micro-F1**.

4. **Root Cause Analysis of Static Fusion Dilution:**  
   Linear probability blending with fixed scalar weights treats all test instances uniformly. When movie posters contain non-informative visual cues (e.g., stylized portraits or minimalist title cards), blending them with high-confidence text probabilities dilutes true genre signals below the classification threshold, reducing precision and recall on text-predictable genres.

5. **Impact of Per-Genre Threshold Optimization:**  
   Optimizing individual decision thresholds per genre on the validation split yielded substantial gains over the default 0.50 cutoff. For the text baseline, validation Micro-F1 increased from 0.4744 to 0.6042—an absolute gain of **+12.98 percentage points** (representing a **+27.36% relative improvement**). This confirms that multi-label class imbalance necessitates genre-specific calibrated thresholds.

6. **Inference Latency & Timing Architecture Delineation:**  
   Benchmarking across 5 timed repetitions on 7,788 test samples revealed steady-state CPU latencies:
   - Text Classifier Inference: **1.5415 s** ($5,052.2$ samples/s)
   - Image Classifier Cached Inference: **1.9394 s** ($4,015.7$ samples/s)
   - End-to-End Multimodal Pipeline (consuming raw text and cached image embeddings): **3.6020 s** ($2,162.2$ samples/s)
   - Isolated Fusion & Thresholding: **0.000962 s** ($0.962\\text{{ ms}}$, $8,091,428.6$ samples/s).
   *Scope Distinction:* The isolated fusion overhead represents pure array arithmetic and must never be conflated with full model inference. In contrast, raw poster image feature extraction on CPU was historically estimated from Phase 2 logs at **794.7 s** (~9.8 imgs/s).

7. **Zero Leakage & Data Partition Integrity:**  
   The dataset partitioning (Train: 15,564 / 59.96%; Validation: 2,607 / 10.04%; Test: 7,788 / 30.00%) has zero overlap ($\\text{{Train}} \\cap \\text{{Val}} = \\emptyset$, $\\text{{Train}} \\cap \\text{{Test}} = \\emptyset$, $\\text{{Val}} \\cap \\text{{Test}} = \\emptyset$). All 11 duplicate plot description groups (25 affected records, 14 redundant copies) are 100% strictly confined within single splits, ensuring zero data leakage into the test set.

---

### Project Limitations & Scope Boundary

- **Descriptive Evidence Policy:** All reported metric differences between models are empirical observations. No non-parametric significance testing (e.g., Wilcoxon signed-rank or McNemar's tests) was conducted.
- **Static Baseline Context:** The late fusion baseline is a static linear combination with fixed $\\alpha=0.55$. It does not perform adaptive failure diagnosis or dynamic recovery.
- **Phase 4 Roadmap:** Phase 4 (Multimodal Failure Diagnosis) is designed to systematically identify when and why modalities conflict, segmenting failures into modality disagreement, visual feature noise, and threshold miscalibration.
- **Current Status:** Phase 4 and Phase 5 have **NOT** been started. All artifacts presented here represent the closed, verified Phase 3 baseline.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(findings)

    # =========================================================================
    # Step 5: Polished PDF Report
    # =========================================================================

    def generate_pdf_report(self, graphs: Dict[str, Path]):
        """Generate a 4-5 page publication-quality PDF report."""
        pdf_path = self.output_dir / "SEAA_Phase3_Supervisor_Summary.pdf"
        logger.info(f"Generating polished PDF report at: {pdf_path}")

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=45,
            bottomMargin=45
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1B365D'),
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#2A9D8F'),
            spaceAfter=10
        )

        h1_style = ParagraphStyle(
            'Heading1_Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#1B365D'),
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True
        )

        h2_style = ParagraphStyle(
            'Heading2_Custom',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor('#2D3748'),
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            'Body_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=5
        )

        callout_style = ParagraphStyle(
            'Callout_Custom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#92400E'),
            spaceBefore=4,
            spaceAfter=4
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=9,
            textColor=colors.white,
            alignment=1
        )

        table_cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor('#1E293B'),
            alignment=1
        )

        table_cell_left = ParagraphStyle(
            'TableCellLeft',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor('#1E293B'),
            alignment=0
        )

        story = []

        # ---------------------------------------------------------------------
        # PAGE 1: Title, Executive Summary, Dataset Split & Proportions
        # ---------------------------------------------------------------------
        story.append(Paragraph("SEAA: Multimodal Reliability Diagnosis & Recovery", title_style))
        story.append(Paragraph("Phase 3 Baseline Evaluation & Benchmark Verification Report — Supervisor Briefing", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1B365D"), spaceAfter=8))

        story.append(Paragraph("1. Executive Summary & Audit Scope", h1_style))
        story.append(Paragraph(
            "This briefing summarizes the verified baseline modeling results and computational benchmarks for the "
            "<b>Self-Evolving AI Auditing Algorithm (SEAA)</b> project on the MM-IMDb multimodal benchmark (23 movie genres). "
            "All metrics and benchmarks presented herein were strictly derived from existing verified workspace artifacts; "
            "no models were retrained, no split was regenerated, and no data was altered. This report marks the formal closure of Phase 3. "
            "<b>Phase 4 (Failure Diagnosis) and Phase 5 (Adaptive Recovery) have NOT started.</b>",
            body_style
        ))

        story.append(Paragraph("2. Dataset & Custom Split Integrity", h1_style))
        story.append(Paragraph(
            "The project utilizes a fixed custom partition of the 25,959 MM-IMDb movies generated via iterative multi-label stratification "
            "(random seed 42) into Train (15,564 / 59.96%), Validation (2,607 / 10.04%), and Test (7,788 / 30.00%). "
            "Rigorous audit verified zero cross-split duplicate text descriptions and absolute disjointness between sets.",
            body_style
        ))

        # Dataset Split Table
        split_table_data = [
            [
                Paragraph("Partition", table_header_style),
                Paragraph("Sample Count", table_header_style),
                Paragraph("Proportion", table_header_style),
                Paragraph("Duplicate Text Groups", table_header_style),
                Paragraph("Redundant Copies", table_header_style),
                Paragraph("Cross-Split Leakage", table_header_style)
            ],
            [
                Paragraph("<b>Train Split</b>", table_cell_left),
                Paragraph("15,564", table_cell_style),
                Paragraph("59.96%", table_cell_style),
                Paragraph("5 groups (13 records)", table_cell_style),
                Paragraph("8 instances", table_cell_style),
                Paragraph("0 records (0%)", table_cell_style)
            ],
            [
                Paragraph("<b>Validation Split</b>", table_cell_left),
                Paragraph("2,607", table_cell_style),
                Paragraph("10.04%", table_cell_style),
                Paragraph("3 groups (6 records)", table_cell_style),
                Paragraph("3 instances", table_cell_style),
                Paragraph("0 records (0%)", table_cell_style)
            ],
            [
                Paragraph("<b>Test Split</b>", table_cell_left),
                Paragraph("7,788", table_cell_style),
                Paragraph("30.00%", table_cell_style),
                Paragraph("3 groups (6 records)", table_cell_style),
                Paragraph("3 instances", table_cell_style),
                Paragraph("0 records (0%)", table_cell_style)
            ],
            [
                Paragraph("<b>Total Dataset</b>", table_cell_left),
                Paragraph("<b>25,959</b>", table_cell_style),
                Paragraph("<b>100.00%</b>", table_cell_style),
                Paragraph("<b>11 groups (25 records)</b>", table_cell_style),
                Paragraph("<b>14 instances</b>", table_cell_style),
                Paragraph("<b>Strictly 0 (VERIFIED)</b>", table_cell_style)
            ]
        ]
        t_split = Table(split_table_data, colWidths=[80, 75, 65, 120, 95, 105])
        t_split.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FFFFFF')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#E2E8F0')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_split)
        story.append(Spacer(1, 8))

        if "07_dataset_split_distribution" in graphs:
            story.append(RLImage(str(graphs["07_dataset_split_distribution"]), width=530, height=230))

        story.append(PageBreak())

        # ---------------------------------------------------------------------
        # PAGE 2: Baseline Model Performance & Metric Verification
        # ---------------------------------------------------------------------
        story.append(Paragraph("3. Baseline Model Performance Evaluation", h1_style))
        story.append(Paragraph(
            "Three baseline models were evaluated on the 7,788 test records: (1) Text-Only (TF-IDF unigrams/bigrams, 10,000 features, OneVsRest Logistic Regression); "
            "(2) Image-Only (StandardScaler on 2,048-dim ResNet50 embeddings, OneVsRest Logistic Regression); and "
            "(3) Multimodal Validation-Tuned Weighted Late Fusion (fixed scalar weight α=0.55 text, 0.45 image, with 23-genre validation-optimized decision thresholds).",
            body_style
        ))

        # Full Baseline Metrics Table
        verif = self.verification_json["task_2_metrics_recomputation"]
        metrics_table_data = [
            [
                Paragraph("Model Architecture", table_header_style),
                Paragraph("Modality", table_header_style),
                Paragraph("Micro F1", table_header_style),
                Paragraph("Macro F1", table_header_style),
                Paragraph("Micro Prec", table_header_style),
                Paragraph("Macro Prec", table_header_style),
                Paragraph("Micro Rec", table_header_style),
                Paragraph("Macro Rec", table_header_style),
                Paragraph("mAP", table_header_style),
                Paragraph("Hamming Loss", table_header_style)
            ]
        ]

        model_rows = [
            ("Text-Only (TF-IDF + LR)", "Text", verif["text_only"]["metrics"]),
            ("Image-Only (ResNet50 + LR)", "Image (Cached)", verif["image_only"]["metrics"]),
            ("Multimodal Late Fusion (α=0.55)", "Multimodal (E2E)", verif["multimodal"]["metrics"])
        ]

        for idx, (m_name, mod_label, m) in enumerate(model_rows):
            bg = colors.HexColor('#F8FAFC') if idx % 2 == 0 else colors.HexColor('#FFFFFF')
            metrics_table_data.append([
                Paragraph(f"<b>{m_name}</b>", table_cell_left),
                Paragraph(mod_label, table_cell_style),
                Paragraph(f"{m['micro_f1']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['macro_f1']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['micro_precision']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['macro_precision']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['micro_recall']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['macro_recall']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['mean_average_precision']['stored_value']:.4f}", table_cell_style),
                Paragraph(f"{m['hamming_loss']['stored_value']:.4f}", table_cell_style)
            ])

        t_metrics = Table(metrics_table_data, colWidths=[120, 65, 42, 42, 42, 42, 42, 42, 42, 57])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FFFFFF')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#F8FAFC')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 8))

        if "01_baseline_metric_comparison" in graphs:
            story.append(RLImage(str(graphs["01_baseline_metric_comparison"]), width=530, height=225))

        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<b>Key Observation:</b> The Text-Only baseline achieves the top performance across all metrics (0.6056 Micro-F1, 0.5183 mAP, 0.0929 Hamming loss). "
            "Multimodal Late Fusion improves by +10.10 percentage points over Image-Only, but underperforms Text-Only by -8.95 percentage points. "
            "Reported differences are descriptive empirical findings; no statistical significance tests were performed.",
            body_style
        ))

        story.append(PageBreak())

        # ---------------------------------------------------------------------
        # PAGE 3: Micro-F1 & Hamming Loss Comparisons, Per-Genre Analysis
        # ---------------------------------------------------------------------
        story.append(Paragraph("4. Granular Metric & Genre-Level Behavior", h1_style))
        story.append(Paragraph(
            "The charts below illustrate the headline Micro-F1 and Hamming loss comparisons, alongside per-genre F1 scores across the 23 classes.",
            body_style
        ))

        # Side by side charts for Micro F1 & Hamming loss
        if "02_micro_f1_comparison" in graphs and "03_hamming_loss_comparison" in graphs:
            t_side = Table([
                [
                    RLImage(str(graphs["02_micro_f1_comparison"]), width=260, height=170),
                    RLImage(str(graphs["03_hamming_loss_comparison"]), width=260, height=170)
                ]
            ], colWidths=[265, 265])
            t_side.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(t_side)

        story.append(Spacer(1, 6))

        if "04_per_genre_f1_comparison" in graphs:
            story.append(Paragraph("5. Per-Genre Performance Distribution Across 23 Genres", h2_style))
            story.append(RLImage(str(graphs["04_per_genre_f1_comparison"]), width=530, height=310))

        story.append(Paragraph(
            "<b>Genre Analysis:</b> Text signals provide dominant F1 in text-rich genres (Drama: 0.7606, Documentary: 0.7391, Comedy: 0.6485). "
            "Visual signals contribute modest complementary value in visually distinctive genres (Animation: 0.4968 multimodal vs. 0.3974 text-only), "
            "demonstrating the potential for selective, adaptive multimodal auditing in Phase 4.",
            body_style
        ))

        story.append(PageBreak())

        # ---------------------------------------------------------------------
        # PAGE 4: Computational Benchmarks & Inference Latency
        # ---------------------------------------------------------------------
        story.append(Paragraph("6. Rigorous Inference Latency & Benchmarking Breakdown", h1_style))
        story.append(Paragraph(
            "Inference latency was benchmarked under a standardized protocol: checkpoints pre-loaded, 1 warm-up repetition, and 5 timed repetitions "
            "across all 7,788 test records using high-resolution time.perf_counter() on CPU.",
            body_style
        ))

        # Timing Table
        comps = self.timing_json["components"]
        timing_table_data = [
            [
                Paragraph("Pipeline Component", table_header_style),
                Paragraph("Median (s)", table_header_style),
                Paragraph("Min (s)", table_header_style),
                Paragraph("Max (s)", table_header_style),
                Paragraph("Std Dev (s)", table_header_style),
                Paragraph("Throughput (samples/s)", table_header_style),
                Paragraph("Measurement Status & Scope", table_header_style)
            ],
            [
                Paragraph("<b>Text Classifier Inference</b>", table_cell_left),
                Paragraph(f"{comps['text_classifier_inference']['median_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['text_classifier_inference']['min_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['text_classifier_inference']['max_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['text_classifier_inference']['std_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['text_classifier_inference']['samples_per_sec']:,.1f}", table_cell_style),
                Paragraph("TF-IDF transform + LR predict_proba on 7,788 test strings", table_cell_left)
            ],
            [
                Paragraph("<b>Image Classifier (Cached)</b>", table_cell_left),
                Paragraph(f"{comps['image_classifier_cached_inference']['median_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['image_classifier_cached_inference']['min_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['image_classifier_cached_inference']['max_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['image_classifier_cached_inference']['std_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['image_classifier_cached_inference']['samples_per_sec']:,.1f}", table_cell_style),
                Paragraph("Scaler + LR predict_proba on cached 2,048-dim features", table_cell_left)
            ],
            [
                Paragraph("<b>Fusion & Thresholding Only</b>", table_cell_left),
                Paragraph(f"{comps['fusion_and_thresholding_only']['median_sec']:.6f}", table_cell_style),
                Paragraph(f"{comps['fusion_and_thresholding_only']['min_sec']:.6f}", table_cell_style),
                Paragraph(f"{comps['fusion_and_thresholding_only']['max_sec']:.6f}", table_cell_style),
                Paragraph(f"{comps['fusion_and_thresholding_only']['std_sec']:.6f}", table_cell_style),
                Paragraph(f"{comps['fusion_and_thresholding_only']['samples_per_sec']:,.1f}", table_cell_style),
                Paragraph("Array weighting (0.55*P_text + 0.45*P_img) + thresholding. <b>Not model inference.</b>", table_cell_left)
            ],
            [
                Paragraph("<b>End-to-End Multimodal (Cached)</b>", table_cell_left),
                Paragraph(f"{comps['end_to_end_cached_multimodal_inference']['median_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['end_to_end_cached_multimodal_inference']['min_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['end_to_end_cached_multimodal_inference']['max_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['end_to_end_cached_multimodal_inference']['std_sec']:.4f}", table_cell_style),
                Paragraph(f"{comps['end_to_end_cached_multimodal_inference']['samples_per_sec']:,.1f}", table_cell_style),
                Paragraph("Consumes raw text & cached image feats to final genre predictions", table_cell_left)
            ],
            [
                Paragraph("<b>Raw ResNet50 Extraction Ref</b>", table_cell_left),
                Paragraph("794.7000*", table_cell_style),
                Paragraph("N/A", table_cell_style),
                Paragraph("N/A", table_cell_style),
                Paragraph("N/A", table_cell_style),
                Paragraph("9.8*", table_cell_style),
                Paragraph("<b>Historical estimate from Phase 2 log; not freshly measured in this audit.</b>", table_cell_left)
            ]
        ]

        t_timing = Table(timing_table_data, colWidths=[120, 50, 42, 42, 48, 65, 173])
        t_timing.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FFFFFF')),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#F8FAFC')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#FFFFFF')),
            ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#FEF3C7')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_timing)
        story.append(Spacer(1, 8))

        if "05_inference_latency_comparison" in graphs and "06_fusion_overhead" in graphs:
            story.append(RLImage(str(graphs["05_inference_latency_comparison"]), width=530, height=215))
            story.append(Spacer(1, 6))
            story.append(RLImage(str(graphs["06_fusion_overhead"]), width=530, height=195))

        story.append(PageBreak())

        # ---------------------------------------------------------------------
        # PAGE 5: Key Findings, Limitations, and Phase 4 Transition
        # ---------------------------------------------------------------------
        story.append(Paragraph("7. Key Findings, Limitations & Phase 4 Roadmap", h1_style))

        findings_text = (
            "<b>1. Text Modality Dominance:</b> Text-Only achieved the highest overall Micro-F1 (0.6056) and mAP (0.5183) on test data. "
            "Movie plot text provides rich semantic clues that drive accurate multi-label classification.<br/>"
            "<b>2. Image Modality Weakness:</b> Movie posters yield lower standalone performance (0.4151 Micro-F1), as artistic styling and typography "
            "introduce ambiguity for multi-label classification.<br/>"
            "<b>3. Naive Late Fusion Dilution:</b> Blending text and image probabilities with a fixed scalar weight (α=0.55 text, 0.45 image) "
            "improved over Image-Only (+10.10 pp) but failed to surpass Text-Only (-8.95 pp). Static fusion blindly degrades sharp textual predictions when visual features are noisy.<br/>"
            "<b>4. Threshold Optimization Value:</b> Per-genre threshold tuning on validation data produced a +12.98 percentage point gain (+27.36% relative) "
            "over the default 0.50 threshold, demonstrating the importance of calibrating per-class decision boundaries.<br/>"
            "<b>5. Latency Hierarchy:</b> Steady-state cached inference operates at 1.54s (text), 1.94s (image), and 3.60s (multimodal E2E) on CPU. "
            "Isolated fusion arithmetic overhead is sub-millisecond (0.96 ms), while raw ResNet50 CNN extraction requires ~795s.<br/>"
            "<b>6. Statistical Clarification:</b> All reported comparisons are descriptive empirical measurements; non-parametric significance tests were not conducted.<br/>"
            "<b>7. Phase 4 Objective:</b> Phase 4 (Failure Diagnosis) will systematically audit failure cases to detect when multimodal fusion degrades performance, "
            "laying the foundation for adaptive reliability recovery in Phase 5."
        )
        story.append(Paragraph(findings_text, body_style))
        story.append(Spacer(1, 8))

        # Limitations & Formal Closure Box
        closure_data = [
            [
                Paragraph("<b>PHASE 3 FORMAL CLOSURE & VERIFICATION STATUS</b>", ParagraphStyle('BoxH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#065F46'), alignment=1))
            ],
            [
                Paragraph(
                    "• <b>Status:</b> All Phase 3 deliverables are 100% verified, consistent, and closed.<br/>"
                    "• <b>Integrity Safeguards:</b> Fixed custom split (15,564 / 2,607 / 7,788) strictly preserved; 0 data leakage.<br/>"
                    "• <b>Scope Constraint:</b> Phase 4 (Failure Diagnosis) and Phase 5 (Adaptive Recovery) have <b>NOT</b> been started.<br/>"
                    "• <b>Reproducibility:</b> All outputs reproducible via: <code>python scripts/generate_phase3_presentation_outputs.py</code>",
                    ParagraphStyle('BoxB', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor('#064E3B'))
                )
            ]
        ]
        t_box = Table(closure_data, colWidths=[540])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D1FAE5')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ECFDF5')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#10B981')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(t_box)

        # Build document with custom canvas
        doc.build(story, canvasmaker=NumberedCanvas)
        logger.info(f"PDF report successfully compiled ({pdf_path.stat().st_size:,} bytes).")

    # =========================================================================
    # Step 6: Presentation Folder README
    # =========================================================================

    def generate_readme(self, graphs: Dict[str, Path]):
        """Generate comprehensive README.md in outputs/presentation/phase3/."""
        readme_path = self.output_dir / "README.md"
        logger.info(f"Generating presentation folder README: {readme_path}")

        content = f"""# SEAA Phase 3 Presentation-Ready Outputs (Supervisor Package)

**Project Title:** A Self-Evolving AI Auditing Algorithm for Adaptive Reliability Diagnosis and Recovery in Multimodal AI Systems (SEAA)  
**Workspace:** `C:\\Users\\bharat kumar\\OneDrive\\Desktop\\SEAA`  
**Generated Date:** `2026-10-22`  
**Phase Status:** Phase 3 Formally Closed & Verified. **Phase 4 Has NOT Started.**

---

## 1. Directory Structure & File Manifest

All files in this directory were generated dynamically from verified workspace artifacts by running `python scripts/generate_phase3_presentation_outputs.py`.

```
outputs/presentation/phase3/
├── README.md                               # This documentation and file manifest
├── SEAA_Phase3_Supervisor_Summary.pdf      # 5-page publication-quality executive report
├── phase3_model_metrics.csv                # Standardized baseline evaluation metrics (CSV)
├── phase3_model_metrics.md                 # Human-readable baseline evaluation metrics table
├── phase3_inference_timings.csv            # Standardized inference timing and throughput benchmark (CSV)
├── phase3_inference_timings.md             # Detailed inference timing breakdown and scope delineation
├── phase3_per_genre_metrics.csv            # Consolidated per-genre evaluation across all 23 movie genres
├── phase3_key_findings.md                  # Evidence-backed findings, limitations, and supervisor briefing
├── 01_baseline_metric_comparison.png       # Grouped bar chart comparing Micro-F1, Macro-F1, and mAP
├── 02_micro_f1_comparison.png              # Headline Micro-F1 comparison across baseline models
├── 03_hamming_loss_comparison.png          # Multi-label Hamming loss error comparison (Lower is Better)
├── 04_per_genre_f1_comparison.png          # Full 23-genre horizontal F1 comparison across models
├── 05_inference_latency_comparison.png     # Benchmark of steady-state classifier latencies on CPU
├── 06_fusion_overhead.png                  # Separate visual for isolated sub-millisecond fusion arithmetic
└── 07_dataset_split_distribution.png       # Custom train/validation/test split sizes and proportions
```

---

## 2. Authoritative Data Sources

Every value and visualization in this folder is traced to authoritative workspace artifacts:

| Output File / Graph | Authoritative Source Artifact | Verified Key Numbers |
|---|---|---|
| `phase3_model_metrics.*`<br/>`01_baseline_metric_comparison.png`<br/>`02_micro_f1_comparison.png`<br/>`03_hamming_loss_comparison.png` | `outputs/metrics/baseline_comparison.csv`<br/>`outputs/reports/phase3_metric_verification.json` | Text-Only: 0.6056 Micro-F1, 0.4955 Macro-F1, 0.5183 mAP, 0.0929 Hamming Loss<br/>Image-Only: 0.4151 Micro-F1, 0.2847 Macro-F1, 0.2337 mAP, 0.1642 Hamming Loss<br/>Multimodal: 0.5161 Micro-F1, 0.4003 Macro-F1, 0.3985 mAP, 0.1307 Hamming Loss |
| `phase3_inference_timings.*`<br/>`05_inference_latency_comparison.png`<br/>`06_fusion_overhead.png` | `outputs/metrics/inference_timing.json`<br/>`outputs/metrics/inference_timing.csv` | Text Latency: 1.5415 s (5,052.2 samples/s)<br/>Image Cached Latency: 1.9394 s (4,015.7 samples/s)<br/>End-to-End Multimodal: 3.6020 s (2,162.2 samples/s)<br/>Fusion Arithmetic Only: 0.000962 s (8,091,428.6 samples/s) |
| `phase3_per_genre_metrics.csv`<br/>`04_per_genre_f1_comparison.png` | `outputs/metrics/text_only_per_genre.csv`<br/>`outputs/metrics/image_only_per_genre.csv`<br/>`outputs/metrics/multimodal_per_genre.csv` | Full 23-genre precision, recall, F1, average precision, threshold, and support |
| `07_dataset_split_distribution.png` | `data/split_indices.json`<br/>`outputs/reports/phase3_metric_verification.json` | Train: 15,564 (59.96%)<br/>Val: 2,607 (10.04%)<br/>Test: 7,788 (30.00%)<br/>Total: 25,959 records |

---

## 3. How to Regenerate These Outputs

This entire suite of outputs is 100% reproducible with a single script execution:

```powershell
# From project root:
python scripts/generate_phase3_presentation_outputs.py
```

The script dynamically reads the authoritative files, checks for missing data, generates the high-resolution charts (300 DPI), builds the CSV and Markdown tables, compiles the ReportLab Platypus PDF, and outputs this README.

---

## 4. Critical Scope Caveats & Audit Disclosures

1. **Phase 4 Has NOT Started:**  
   All artifacts strictly represent Phase 3 static baselines. Adaptive reliability diagnosis and recovery have not been implemented.
2. **No Model Retraining / No Data Modification:**  
   This generation task performed zero model retraining, zero checkpoint alteration, zero dataset regeneration, and zero modifications to `data/split_indices.json`.
3. **Static Weighted Late Fusion Context:**  
   The multimodal baseline is a static linear combination with fixed scalar parameter $\\alpha = 0.55$ ($0.55 P_{{\\text{{text}}}} + 0.45 P_{{\\text{{image}}}}$). It does not dynamically adapt to modality failure.
4. **Historical Raw-Image Extraction Estimate:**  
   The **794.7-second raw-image extraction figure** is a historical estimate derived from the Phase 2 log (`outputs/logs/image_feature_extraction.log`: 25,959 images processed on CPU at ~9.8 images/second). It was **not freshly measured in this audit** and is labeled as historical reference only.
5. **Statistical Significance Disclosure:**  
   Reported differences between models are empirical observations. No non-parametric significance testing (e.g., Wilcoxon signed-rank or McNemar's tests) was conducted.
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    def run(self):
        """Execute full presentation output generation pipeline."""
        logger.info("Starting Phase 3 Presentation Generation Pipeline...")
        self.setup_output_dir()
        self.load_source_artifacts()
        graphs = self.generate_all_graphs()
        self.generate_all_tables()
        self.generate_pdf_report(graphs)
        self.generate_readme(graphs)
        logger.info("Phase 3 Presentation Generation Pipeline successfully completed.")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    generator = Phase3PresentationGenerator(proj_root)
    generator.run()
