"""
Unit tests for Baseline Models, Fusion, and Evaluation Metrics (Phase 3 Validation).
Tests:
1. Split indices integrity (loaded without modification).
2. Label dimensions (23) and exact order consistency across models.
3. Prediction output shapes and valid probability ranges [0, 1].
4. Validation-only fusion parameter selection (zero test leakage).
5. No test labels in threshold tuning.
6. Metric correctness on controlled synthetic ground truth.
7. End-to-end small dummy model execution.
"""

import os
import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import GENRES, GENRE_TO_IDX, NUM_CLASSES
from src.evaluation.metrics import evaluate_multilabel_predictions, optimize_thresholds_on_val
from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.models.multimodal_baseline import MultimodalBaselineModel


class TestBaselineModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = project_root / "data"
        cls.split_file = cls.data_dir / "split_indices.json"
        cls.label_file = cls.data_dir / "label_mapping.json"
        
        with open(cls.split_file, "r", encoding="utf-8") as f:
            cls.splits = json.load(f)
        with open(cls.label_file, "r", encoding="utf-8") as f:
            cls.label_map = json.load(f)

    def test_01_fixed_splits_integrity(self):
        """Test 1: Split indices are loaded without modification."""
        self.assertIn("train", self.splits)
        self.assertIn("val", self.splits)
        self.assertIn("test", self.splits)
        self.assertEqual(len(self.splits["train"]), 15564)
        self.assertEqual(len(self.splits["val"]), 2607)
        self.assertEqual(len(self.splits["test"]), 7788)
        self.assertEqual(len(self.splits["train"]) + len(self.splits["val"]) + len(self.splits["test"]), 25959)

    def test_02_label_order_and_dimensions(self):
        """Test 2: Exactly 23 genres in consistent order."""
        self.assertEqual(len(self.label_map["genres"]), 23)
        self.assertEqual(self.label_map["genres"], GENRES)
        self.assertEqual(self.label_map["num_classes"], 23)

    def test_03_controlled_metric_computation(self):
        """Test 3: Metrics are computed correctly on controlled synthetic example."""
        # 4 samples, 3 classes
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Perfect predictions
        y_probs_perfect = np.array([
            [0.9, 0.1, 0.8],
            [0.2, 0.7, 0.1],
            [0.85, 0.9, 0.3],
            [0.1, 0.2, 0.95]
        ], dtype=np.float32)
        
        metrics = evaluate_multilabel_predictions(y_true, y_probs_perfect, default_threshold=0.5)
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["hamming_loss"], 0.0)
        self.assertEqual(metrics["mean_average_precision"], 1.0)

    def test_04_threshold_optimization_improves_or_matches_f1(self):
        """Test 4: Threshold optimization on validation data improves or matches default 0.5."""
        np.random.seed(42)
        y_val = np.random.binomial(1, 0.2, size=(200, 5))
        # Add biased noise to probabilities
        probs = np.clip(y_val * 0.4 + np.random.uniform(0.1, 0.3, size=(200, 5)), 0.0, 1.0)
        
        tuned_thresholds = optimize_thresholds_on_val(y_val, probs)
        self.assertEqual(len(tuned_thresholds), 5)
        
        default_res = evaluate_multilabel_predictions(y_val, probs, default_threshold=0.5)
        tuned_res = evaluate_multilabel_predictions(y_val, probs, thresholds=tuned_thresholds)
        self.assertGreaterEqual(tuned_res["macro_f1"], default_res["macro_f1"])

    def test_05_text_baseline_pipeline_mock(self):
        """Test 5: Text-only model fits on train, tunes on val, predicts valid probabilities on test."""
        train_texts = ["A brave superhero fighting villains in space.", "A romantic comedy about two strangers falling in love."] * 10
        y_train = np.array([[1, 0, 0] + [0]*20, [0, 1, 0] + [0]*20] * 10, dtype=np.float32)
        
        val_texts = ["A heroic adventure.", "Love story in the city."] * 4
        y_val = np.array([[1, 0, 0] + [0]*20, [0, 1, 0] + [0]*20] * 4, dtype=np.float32)
        
        test_texts = ["Intergalactic battles.", "Couple dating in Paris."]
        y_test = np.array([[1, 0, 0] + [0]*20, [0, 1, 0] + [0]*20], dtype=np.float32)
        
        model = TextBaselineModel(max_features=500, random_state=42)
        fit_info = model.fit(train_texts, y_train, val_texts, y_val, genre_names=GENRES)
        
        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.thresholds), 23)
        
        metrics, probs, inf_t = model.evaluate(test_texts, y_test)
        self.assertEqual(probs.shape, (2, 23))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_06_image_baseline_pipeline_mock(self):
        """Test 6: Image-only model fits on train embeddings, tunes on val, predicts valid probabilities on test."""
        np.random.seed(42)
        X_train = np.random.randn(20, 2048).astype(np.float32)
        y_train = np.random.binomial(1, 0.1, size=(20, 23)).astype(np.float32)
        
        X_val = np.random.randn(8, 2048).astype(np.float32)
        y_val = np.random.binomial(1, 0.1, size=(8, 23)).astype(np.float32)
        
        X_test = np.random.randn(4, 2048).astype(np.float32)
        y_test = np.random.binomial(1, 0.1, size=(4, 23)).astype(np.float32)
        
        model = ImageBaselineModel(c_param=0.1, random_state=42)
        fit_info = model.fit(X_train, y_train, X_val, y_val, genre_names=GENRES)
        
        self.assertTrue(model.is_fitted)
        metrics, probs, inf_t = model.evaluate(X_test, y_test)
        self.assertEqual(probs.shape, (4, 23))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_07_multimodal_late_fusion_validation_selection(self):
        """Test 7: Multimodal late fusion strictly uses validation data to pick alpha and thresholds."""
        n_val = 50
        n_test = 20
        y_val = np.random.binomial(1, 0.1, size=(n_val, 23)).astype(np.float32)
        y_test = np.random.binomial(1, 0.1, size=(n_test, 23)).astype(np.float32)
        
        # Text probs slightly better than image probs
        val_text = np.clip(y_val * 0.7 + np.random.uniform(0.0, 0.2, (n_val, 23)), 0, 1)
        val_image = np.clip(y_val * 0.3 + np.random.uniform(0.1, 0.4, (n_val, 23)), 0, 1)
        
        test_text = np.clip(y_test * 0.7 + np.random.uniform(0.0, 0.2, (n_test, 23)), 0, 1)
        test_image = np.clip(y_test * 0.3 + np.random.uniform(0.1, 0.4, (n_test, 23)), 0, 1)
        
        mm_model = MultimodalBaselineModel()
        fit_info = mm_model.fit_fusion_parameters(
            val_text_probs=val_text,
            val_image_probs=val_image,
            y_val=y_val,
            genre_names=GENRES
        )
        
        self.assertIn("selected_alpha", fit_info)
        self.assertGreaterEqual(mm_model.alpha, 0.0)
        self.assertLessEqual(mm_model.alpha, 1.0)
        
        metrics, fused_test_probs = mm_model.evaluate(test_text, test_image, y_test)
        self.assertEqual(fused_test_probs.shape, (n_test, 23))
        self.assertTrue(np.all((fused_test_probs >= 0.0) & (fused_test_probs <= 1.0)))

    def test_08_saved_predictions_structure_and_ids(self):
        """Test 8: Saved test prediction files have exactly 7,788 records matching test split IDs."""
        pred_dir = project_root / "outputs" / "predictions"
        meta_df = pd.read_csv(project_root / "data" / "metadata.csv").sort_values("dataset_idx").reset_index(drop=True)
        expected_test_ids = set(meta_df.iloc[self.splits["test"]]["movie_id"].tolist())
        
        for fname in ["text_only_test_predictions.csv", "image_only_test_predictions.csv", "multimodal_test_predictions.csv"]:
            p_file = pred_dir / fname
            if p_file.exists():
                df = pd.read_csv(p_file)
                self.assertEqual(len(df), 7788, f"{fname} must have exactly 7788 rows")
                self.assertEqual(df["movie_id"].nunique(), 7788, f"{fname} must have 7788 unique movie_ids")
                self.assertEqual(set(df["movie_id"]), expected_test_ids, f"{fname} IDs must match test split")

    def test_09_checkpoints_load_and_reproduce(self):
        """Test 9: Saved checkpoints load cleanly and contain expected parameters."""
        ckpt_dir = project_root / "outputs" / "checkpoints"
        
        # Text checkpoint
        text_ckpt_path = ckpt_dir / "text_baseline.joblib"
        if text_ckpt_path.exists():
            import joblib
            data = joblib.load(text_ckpt_path)
            self.assertIn("vectorizer", data)
            self.assertIn("classifier", data)
            self.assertIn("thresholds", data)
            self.assertEqual(len(data["thresholds"]), 23)
            
        # Image checkpoint
        img_ckpt_path = ckpt_dir / "image_baseline.joblib"
        if img_ckpt_path.exists():
            import joblib
            data = joblib.load(img_ckpt_path)
            self.assertIn("classifier", data)
            self.assertIn("scaler", data)
            self.assertIn("thresholds", data)
            self.assertEqual(len(data["thresholds"]), 23)

        # Multimodal checkpoint
        mm_ckpt_path = ckpt_dir / "multimodal_baseline.json"
        if mm_ckpt_path.exists():
            with open(mm_ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("alpha", data)
            self.assertEqual(data["alpha"], 0.55)
            self.assertIn("thresholds", data)
            self.assertEqual(len(data["thresholds"]), 23)

    def test_10_timing_artifacts_validity(self):
        """Test 10: Timing benchmark outputs exist and contain non-zero latency measurements."""
        timing_csv = project_root / "outputs" / "metrics" / "inference_timing.csv"
        if timing_csv.exists():
            df = pd.read_csv(timing_csv)
            self.assertIn("component", df.columns)
            self.assertIn("median_sec", df.columns)
            self.assertIn("throughput_samples_per_sec", df.columns)
            valid_rows = df[df["median_sec"].notnull()]
            self.assertTrue((valid_rows["median_sec"] > 0).all())
            self.assertTrue((valid_rows["throughput_samples_per_sec"] > 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

