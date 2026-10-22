"""
Runner script for MM-IMDb Multimodal Baseline Model.
Combines Text-Only and Image-Only predictions via Validation-Tuned Weighted Late Fusion.
Tunes fusion weight alpha and decision thresholds strictly on validation data.
Evaluates on untouched test split, exports metrics, predictions, checkpoint,
consolidates baseline_comparison.csv, and generates publication-grade figures.
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_baseline import MultimodalBaselineModel

def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_comparison_plots(
    comparison_df: pd.DataFrame,
    text_per_genre: pd.DataFrame,
    image_per_genre: pd.DataFrame,
    mm_per_genre: pd.DataFrame,
    figures_dir: Path
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot 1: Overall Baseline Model Comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics_to_plot = ["micro_f1", "macro_f1", "mean_average_precision"]
    metric_labels = ["Micro-F1", "Macro-F1", "Mean AP (mAP)"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    models = comparison_df["model"].tolist()
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]
    
    for idx, model_name in enumerate(models):
        row = comparison_df[comparison_df["model"] == model_name].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=model_name, color=colors[idx], alpha=0.9)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax.set_title("MM-IMDb Baseline Model Performance Comparison (Test Split)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.legend(title="Model", frameon=True)
    plt.tight_layout()
    
    fig1_path = figures_dir / "baseline_model_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Plot 2: Per-Genre F1 Comparison
    genres = text_per_genre.index.tolist()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(genres))
    width = 0.28
    
    t_f1 = text_per_genre["f1"].values
    i_f1 = image_per_genre["f1"].values
    m_f1 = mm_per_genre["f1"].values
    
    ax.bar(x - width, t_f1, width, label="Text-Only", color="#2b5c8f", alpha=0.85)
    ax.bar(x, i_f1, width, label="Image-Only", color="#d95f02", alpha=0.85)
    ax.bar(x + width, m_f1, width, label="Multimodal (Late Fusion)", color="#2ca02c", alpha=0.9)
    
    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_title("Per-Genre F1 Score Across Modalities (Test Split)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(title="Model Architecture", frameon=True)
    plt.tight_layout()
    
    fig2_path = figures_dir / "per_genre_f1_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

def main():
    log_path = project_root / "outputs" / "logs" / "multimodal_baseline.log"
    setup_logging(log_path)
    logger = logging.getLogger("TrainMultimodalBaseline")
    logger.info("=== Starting Multimodal Baseline Training & Evaluation ===")

    meta_path = project_root / "data" / "metadata.csv"
    splits_path = project_root / "data" / "split_indices.json"
    label_map_path = project_root / "data" / "label_mapping.json"
    pred_dir = project_root / "outputs" / "predictions"
    metrics_dir = project_root / "outputs" / "metrics"
    ckpt_dir = project_root / "outputs" / "checkpoints"
    figures_dir = project_root / "outputs" / "figures"
    
    text_preds_path = pred_dir / "text_only_val_predictions.npz"
    image_preds_path = pred_dir / "image_only_val_predictions.npz"
    
    if not text_preds_path.exists() or not image_preds_path.exists():
        logger.error("Predictions from text and image baselines must exist before running multimodal fusion.")
        sys.exit(1)

    df_meta = pd.read_csv(meta_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
        
    genres = label_map["genres"]
    df_sorted = df_meta.sort_values("dataset_idx").reset_index(drop=True)
    
    target_matrix = np.array([
        [int(v) for v in row["target_multihot"].split(",")]
        for _, row in df_sorted.iterrows()
    ], dtype=np.float32)

    val_idx = splits["val"]
    test_idx = splits["test"]
    y_val = target_matrix[val_idx]
    y_test = target_matrix[test_idx]

    # Load unimodal predicted probabilities
    text_npz = np.load(text_preds_path)
    image_npz = np.load(image_preds_path)
    
    val_text_probs = text_npz["val_probs"]
    test_text_probs = text_npz["test_probs"]
    val_image_probs = image_npz["val_probs"]
    test_image_probs = image_npz["test_probs"]

    logger.info(f"Loaded validation probabilities: Text={val_text_probs.shape}, Image={val_image_probs.shape}")
    logger.info(f"Loaded test probabilities: Text={test_text_probs.shape}, Image={test_image_probs.shape}")

    # 1. Initialize and tune multimodal late fusion model
    model = MultimodalBaselineModel()
    fit_summary = model.fit_fusion_parameters(
        val_text_probs=val_text_probs,
        val_image_probs=val_image_probs,
        y_val=y_val,
        genre_names=genres
    )

    # 2. Evaluate on untouched test split
    logger.info("Evaluating multimodal model on test split using validation-selected alpha and thresholds...")
    t0 = time.time()
    test_metrics, test_probs = model.evaluate(
        test_text_probs=test_text_probs,
        test_image_probs=test_image_probs,
        y_test=y_test,
        inference_time=time.time() - t0
    )
    test_binary = (test_probs >= model.thresholds).astype(int)

    logger.info("=== Multimodal Test Evaluation Results ===")
    logger.info(f"Selected Alpha (Text Weight): {model.alpha:.2f}")
    logger.info(f"Micro-F1: {test_metrics['micro_f1']:.4f}")
    logger.info(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Mean Average Precision (mAP): {test_metrics['mean_average_precision']:.4f}")
    logger.info(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    # 3. Save outputs
    # A. Multimodal metrics JSON
    metrics_file = metrics_dir / "multimodal_metrics.json"
    full_metrics = {
        "model_name": "Multimodal_ValidationTuned_WeightedLateFusion",
        "split": "test",
        "num_test_samples": len(test_idx),
        "selected_alpha": float(model.alpha),
        "validation_selection": fit_summary,
        "test_metrics": test_metrics
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved multimodal metrics JSON to: {metrics_file}")

    # B. Per-genre metrics CSV
    per_genre_df = pd.DataFrame.from_dict(test_metrics["per_genre"], orient="index")
    per_genre_df.index.name = "genre"
    per_genre_file = metrics_dir / "multimodal_per_genre.csv"
    per_genre_df.to_csv(per_genre_file)
    logger.info(f"Saved multimodal per-genre CSV to: {per_genre_file}")

    # C. Test predictions CSV
    test_records = []
    test_meta_subset = df_sorted.iloc[test_idx].reset_index(drop=True)
    for i in range(len(test_idx)):
        row_dict = {
            "movie_id": test_meta_subset.iloc[i]["movie_id"],
            "dataset_idx": int(test_meta_subset.iloc[i]["dataset_idx"]),
            "ground_truth_genres": test_meta_subset.iloc[i]["genres"],
            "predicted_genres": json.dumps([genres[c] for c in np.where(test_binary[i] == 1)[0]]),
            "num_true_genres": int(np.sum(y_test[i])),
            "num_pred_genres": int(np.sum(test_binary[i]))
        }
        for c, g in enumerate(genres):
            row_dict[f"prob_{g}"] = round(float(test_probs[i, c]), 4)
            row_dict[f"pred_{g}"] = int(test_binary[i, c])
        test_records.append(row_dict)

    pred_df = pd.DataFrame(test_records)
    pred_file = pred_dir / "multimodal_test_predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved multimodal test predictions CSV to: {pred_file}")

    # D. Save checkpoint
    ckpt_file = ckpt_dir / "multimodal_baseline.json"
    model.save(ckpt_file)

    # 4. Build Consolidated baseline_comparison.csv
    with open(metrics_dir / "text_only_metrics.json", "r", encoding="utf-8") as f:
        t_data = json.load(f)["test_metrics"]
    with open(metrics_dir / "image_only_metrics.json", "r", encoding="utf-8") as f:
        i_data = json.load(f)["test_metrics"]
    m_data = test_metrics

    comp_rows = [
        {
            "model": "Text-Only (TF-IDF + LR)",
            "modality": "Text",
            "micro_f1": t_data["micro_f1"],
            "macro_f1": t_data["macro_f1"],
            "micro_precision": t_data["micro_precision"],
            "micro_recall": t_data["micro_recall"],
            "mean_average_precision": t_data["mean_average_precision"],
            "hamming_loss": t_data["hamming_loss"],
            "inference_time_sec": t_data.get("inference_time_sec", 0.0),
            "samples_per_sec": t_data.get("samples_per_second", 0.0)
        },
        {
            "model": "Image-Only (ResNet50 + LR)",
            "modality": "Image (Cached Feats)",
            "micro_f1": i_data["micro_f1"],
            "macro_f1": i_data["macro_f1"],
            "micro_precision": i_data["micro_precision"],
            "micro_recall": i_data["micro_recall"],
            "mean_average_precision": i_data["mean_average_precision"],
            "hamming_loss": i_data["hamming_loss"],
            "inference_time_sec": i_data.get("inference_time_sec", 0.0),
            "samples_per_sec": i_data.get("samples_per_second", 0.0)
        },
        {
            "model": "Multimodal (Validation-Tuned Weighted Late Fusion)",
            "modality": "Multimodal (Cached E2E)",
            "micro_f1": m_data["micro_f1"],
            "macro_f1": m_data["macro_f1"],
            "micro_precision": m_data["micro_precision"],
            "micro_recall": m_data["micro_recall"],
            "mean_average_precision": m_data["mean_average_precision"],
            "hamming_loss": m_data["hamming_loss"],
            "inference_time_sec": m_data.get("inference_time_sec", 0.0),
            "samples_per_sec": m_data.get("samples_per_second", 0.0)
        }
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_file = metrics_dir / "baseline_comparison.csv"
    comp_df.to_csv(comp_file, index=False)
    logger.info(f"Saved consolidated baseline comparison to: {comp_file}")

    # 5. Generate Figures
    text_pg = pd.read_csv(metrics_dir / "text_only_per_genre.csv", index_col="genre")
    image_pg = pd.read_csv(metrics_dir / "image_only_per_genre.csv", index_col="genre")
    mm_pg = per_genre_df
    
    logger.info("Generating publication-grade figures...")
    generate_comparison_plots(comp_df, text_pg, image_pg, mm_pg, figures_dir)
    logger.info(f"Saved figures to: {figures_dir}")

    logger.info("=== Multimodal Baseline Finished Successfully ===")

if __name__ == "__main__":
    main()
