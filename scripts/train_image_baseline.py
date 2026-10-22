"""
Runner script for MM-IMDb Image-Only Baseline Model.
Trains One-vs-Rest Logistic Regression on frozen ResNet50 visual embeddings.
Tunes decision thresholds strictly on validation data.
Evaluates on untouched test split and exports metrics, predictions, and checkpoint.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.image_baseline import ImageBaselineModel

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

def main():
    log_path = project_root / "outputs" / "logs" / "image_baseline.log"
    setup_logging(log_path)
    logger = logging.getLogger("TrainImageBaseline")
    logger.info("=== Starting Image-Only Baseline Training & Evaluation ===")

    # 1. Load metadata and split indices
    meta_path = project_root / "data" / "metadata.csv"
    splits_path = project_root / "data" / "split_indices.json"
    label_map_path = project_root / "data" / "label_mapping.json"
    features_path = project_root / "data" / "processed" / "resnet50_features.npy"
    info_path = project_root / "data" / "processed" / "resnet50_features_info.json"
    
    if not features_path.exists() or not info_path.exists():
        logger.error(f"Features file {features_path} not found. Run extract_image_features.py first.")
        sys.exit(1)

    with open(info_path, "r", encoding="utf-8") as f:
        feat_info = json.load(f)
    if feat_info.get("status") != "COMPLETE":
        logger.warning(f"Feature extraction not marked COMPLETE (current: {feat_info.get('completed_samples')}/{feat_info.get('total_samples')}). Loading available features.")

    df_meta = pd.read_csv(meta_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
        
    genres = label_map["genres"]
    df_sorted = df_meta.sort_values("dataset_idx").reset_index(drop=True)
    
    # Parse multi-hot targets into matrix (N, 23)
    target_matrix = np.array([
        [int(v) for v in row["target_multihot"].split(",")]
        for _, row in df_sorted.iterrows()
    ], dtype=np.float32)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Load memory-mapped features
    logger.info(f"Loading ResNet50 features from: {features_path}...")
    try:
        all_features = np.load(features_path, mmap_mode="r")
    except Exception:
        total_samples = feat_info.get("total_samples", 25959)
        feature_dim = feat_info.get("feature_dim", 2048)
        all_features = np.memmap(features_path, dtype=np.float32, mode="r", shape=(total_samples, feature_dim))

    X_train = np.array(all_features[train_idx])
    X_val = np.array(all_features[val_idx])
    X_test = np.array(all_features[test_idx])

    y_train = target_matrix[train_idx]
    y_val = target_matrix[val_idx]
    y_test = target_matrix[test_idx]

    logger.info(f"Loaded features: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Initialize and fit image baseline model
    model = ImageBaselineModel(
        c_param=0.5,
        scale_features=True,
        random_state=42
    )
    
    fit_summary = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        genre_names=genres
    )

    # 3. Evaluate on untouched test split
    logger.info(f"Evaluating image baseline on {X_test.shape[0]} test samples...")
    test_metrics, test_probs, test_inf_time = model.evaluate(X_test, y_test)
    test_binary = (test_probs >= model.thresholds).astype(int)

    logger.info("=== Image-Only Test Evaluation Results ===")
    logger.info(f"Micro-F1: {test_metrics['micro_f1']:.4f}")
    logger.info(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Mean Average Precision (mAP): {test_metrics['mean_average_precision']:.4f}")
    logger.info(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    logger.info(f"Test Inference Time: {test_metrics['inference_time_sec']:.2f}s ({test_metrics['samples_per_second']:.1f} samples/s)")

    # 4. Save outputs
    metrics_dir = project_root / "outputs" / "metrics"
    pred_dir = project_root / "outputs" / "predictions"
    ckpt_dir = project_root / "outputs" / "checkpoints"
    for d in [metrics_dir, pred_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # A. Save test metrics JSON
    metrics_file = metrics_dir / "image_only_metrics.json"
    full_metrics = {
        "model_name": "ImageOnly_ResNet50_LogisticRegression",
        "split": "test",
        "num_test_samples": len(test_idx),
        "validation_selection": fit_summary,
        "test_metrics": test_metrics
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved test metrics JSON to: {metrics_file}")

    # B. Save per-genre metrics CSV
    per_genre_df = pd.DataFrame.from_dict(test_metrics["per_genre"], orient="index")
    per_genre_df.index.name = "genre"
    per_genre_file = metrics_dir / "image_only_per_genre.csv"
    per_genre_df.to_csv(per_genre_file)
    logger.info(f"Saved per-genre metrics CSV to: {per_genre_file}")

    # C. Save test predictions CSV
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
    pred_file = pred_dir / "image_only_test_predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved test predictions CSV to: {pred_file}")

    # D. Save model checkpoint
    ckpt_file = ckpt_dir / "image_baseline.joblib"
    model.save(ckpt_file)

    # Save validation & test predictions for multimodal fusion
    val_probs = model.predict_proba(X_val)
    val_pred_file = pred_dir / "image_only_val_predictions.npz"
    np.savez_compressed(val_pred_file, val_probs=val_probs, test_probs=test_probs)
    logger.info(f"Saved validation & test probability arrays for fusion to: {val_pred_file}")

    logger.info("=== Image-Only Baseline Finished Successfully ===")

if __name__ == "__main__":
    main()
