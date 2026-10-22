"""
Unit tests for SEAA Phase 3 Presentation-Ready Outputs.
Verifies that all presentation figures, tables, PDF, and README exist,
have non-zero sizes, and adhere strictly to authoritative verified values.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestPhase3PresentationOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pres_dir = project_root / "outputs" / "presentation" / "phase3"
        cls.metrics_dir = project_root / "outputs" / "metrics"
        cls.reports_dir = project_root / "outputs" / "reports"
        cls.data_dir = project_root / "data"

    def test_01_all_files_exist(self):
        """Check 1: Verify all 15 expected presentation outputs exist."""
        expected_files = [
            "01_baseline_metric_comparison.png",
            "02_micro_f1_comparison.png",
            "03_hamming_loss_comparison.png",
            "04_per_genre_f1_comparison.png",
            "05_inference_latency_comparison.png",
            "06_fusion_overhead.png",
            "07_dataset_split_distribution.png",
            "phase3_model_metrics.csv",
            "phase3_model_metrics.md",
            "phase3_inference_timings.csv",
            "phase3_inference_timings.md",
            "phase3_per_genre_metrics.csv",
            "phase3_key_findings.md",
            "SEAA_Phase3_Supervisor_Summary.pdf",
            "README.md"
        ]
        for f in expected_files:
            file_path = self.pres_dir / f
            self.assertTrue(file_path.exists(), f"Missing expected output: {f}")
            self.assertGreater(file_path.stat().st_size, 0, f"File {f} is empty")

    def test_02_graphs_validity_and_size(self):
        """Check 2: Verify high-resolution graphs have valid non-trivial file sizes (> 50 KB)."""
        graph_files = [
            "01_baseline_metric_comparison.png",
            "02_micro_f1_comparison.png",
            "03_hamming_loss_comparison.png",
            "04_per_genre_f1_comparison.png",
            "05_inference_latency_comparison.png",
            "06_fusion_overhead.png",
            "07_dataset_split_distribution.png"
        ]
        for gf in graph_files:
            p = self.pres_dir / gf
            self.assertGreater(p.stat().st_size, 50 * 1024, f"Graph {gf} size unexpectedly small: {p.stat().st_size} bytes")

    def test_03_model_metrics_match_authoritative(self):
        """Check 3: CSV model metrics match authoritative baseline_comparison.csv exactly."""
        pres_df = pd.read_csv(self.pres_dir / "phase3_model_metrics.csv").set_index("model")
        auth_df = pd.read_csv(self.metrics_dir / "baseline_comparison.csv").set_index("model")

        # Text-only
        text_row_pres = pres_df.loc["Text-Only (TF-IDF + LR)"]
        text_row_auth = auth_df.loc["Text-Only (TF-IDF + LR)"]
        self.assertAlmostEqual(text_row_pres["micro_f1"], text_row_auth["micro_f1"], places=4)
        self.assertAlmostEqual(text_row_pres["macro_f1"], text_row_auth["macro_f1"], places=4)
        self.assertAlmostEqual(text_row_pres["mean_average_precision"], text_row_auth["mean_average_precision"], places=4)
        self.assertAlmostEqual(text_row_pres["hamming_loss"], text_row_auth["hamming_loss"], places=4)

        # Image-only
        img_row_pres = pres_df.loc["Image-Only (ResNet50 + LR)"]
        img_row_auth = auth_df.loc["Image-Only (ResNet50 + LR)"]
        self.assertAlmostEqual(img_row_pres["micro_f1"], img_row_auth["micro_f1"], places=4)
        self.assertAlmostEqual(img_row_pres["macro_f1"], img_row_auth["macro_f1"], places=4)
        self.assertAlmostEqual(img_row_pres["mean_average_precision"], img_row_auth["mean_average_precision"], places=4)
        self.assertAlmostEqual(img_row_pres["hamming_loss"], img_row_auth["hamming_loss"], places=4)

        # Multimodal
        mm_row_pres = pres_df.loc["Multimodal (Validation-Tuned Weighted Late Fusion)"]
        mm_row_auth = auth_df.loc["Multimodal (Validation-Tuned Weighted Late Fusion)"]
        self.assertAlmostEqual(mm_row_pres["micro_f1"], mm_row_auth["micro_f1"], places=4)
        self.assertAlmostEqual(mm_row_pres["macro_f1"], mm_row_auth["macro_f1"], places=4)
        self.assertAlmostEqual(mm_row_pres["mean_average_precision"], mm_row_auth["mean_average_precision"], places=4)
        self.assertAlmostEqual(mm_row_pres["hamming_loss"], mm_row_auth["hamming_loss"], places=4)

    def test_04_inference_timings_match_authoritative(self):
        """Check 4: Inference timings match authoritative inference_timing.json."""
        with open(self.metrics_dir / "inference_timing.json", "r", encoding="utf-8") as f:
            timing_json = json.load(f)
        df_timing = pd.read_csv(self.pres_dir / "phase3_inference_timings.csv").set_index("component")

        t_comps = timing_json["components"]
        self.assertAlmostEqual(
            df_timing.loc["text_classifier_inference", "median_sec"],
            t_comps["text_classifier_inference"]["median_sec"],
            places=4
        )
        self.assertAlmostEqual(
            df_timing.loc["image_classifier_cached_inference", "median_sec"],
            t_comps["image_classifier_cached_inference"]["median_sec"],
            places=4
        )
        self.assertAlmostEqual(
            df_timing.loc["fusion_and_thresholding_only", "median_sec"],
            t_comps["fusion_and_thresholding_only"]["median_sec"],
            places=6
        )
        self.assertAlmostEqual(
            df_timing.loc["end_to_end_cached_multimodal_inference", "median_sec"],
            t_comps["end_to_end_cached_multimodal_inference"]["median_sec"],
            places=4
        )

    def test_05_per_genre_metrics_coverage(self):
        """Check 5: Per-genre metrics CSV contains exactly 23 genres with positive support."""
        df_pg = pd.read_csv(self.pres_dir / "phase3_per_genre_metrics.csv")
        self.assertEqual(len(df_pg), 23, "Must have exactly 23 genres")
        self.assertTrue((df_pg["support"] > 0).all(), "Every genre must have positive support")
        for col in ["text_f1", "image_f1", "mm_f1"]:
            self.assertIn(col, df_pg.columns)
            self.assertTrue((df_pg[col] >= 0.0).all())
            self.assertTrue((df_pg[col] <= 1.0).all())

    def test_06_pdf_report_validity(self):
        """Check 6: PDF report exists, has substantial size (> 500 KB), and contains 5 pages."""
        pdf_path = self.pres_dir / "SEAA_Phase3_Supervisor_Summary.pdf"
        self.assertTrue(pdf_path.exists())
        self.assertGreater(pdf_path.stat().st_size, 500 * 1024)

        with open(pdf_path, "rb") as f:
            content = f.read()
        import re
        page_objects = re.findall(rb'/Type\s*/Page\b', content)
        self.assertEqual(len(page_objects), 5, f"PDF should have exactly 5 pages, found {len(page_objects)}")

    def test_07_split_integrity_unaltered(self):
        """Check 7: split_indices.json was strictly not modified and preserves counts."""
        with open(self.data_dir / "split_indices.json", "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.assertEqual(len(splits["train"]), 15564)
        self.assertEqual(len(splits["val"]), 2607)
        self.assertEqual(len(splits["test"]), 7788)


if __name__ == "__main__":
    unittest.main(verbosity=2)
