"""
Unit tests for SEAA Multi-Label Calibration and Reliability Metrics.
Verifies compute_binary_ece, compute_multilabel_brier_score, and evaluate_multilabel_calibration.
"""

import unittest
import numpy as np
from src.evaluation.calibration import (
    compute_binary_ece,
    compute_multilabel_brier_score,
    evaluate_multilabel_calibration,
)


class TestCalibrationMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_perfect_calibration(self):
        """When probabilities match empirical frequencies exactly, ECE should be zero."""
        # 100 samples in [0.2, 0.3) with exactly 25% ones
        # and 100 samples in [0.7, 0.8) with exactly 75% ones
        n_per_bin = 1000
        p1 = 0.25
        y1 = np.zeros(n_per_bin)
        y1[:int(n_per_bin * p1)] = 1.0
        prob1 = np.full(n_per_bin, p1)

        p2 = 0.75
        y2 = np.zeros(n_per_bin)
        y2[:int(n_per_bin * p2)] = 1.0
        prob2 = np.full(n_per_bin, p2)

        y_true = np.concatenate([y1, y2])
        y_prob = np.concatenate([prob1, prob2])

        ece, mce, stats = compute_binary_ece(y_true, y_prob, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)
        self.assertAlmostEqual(mce, 0.0, places=5)

    def test_complete_miscalibration(self):
        """When predictions are completely inverted, ECE should be 1.0."""
        y_true = np.ones(100)
        y_prob = np.zeros(100)
        ece, mce, stats = compute_binary_ece(y_true, y_prob, n_bins=10)
        self.assertAlmostEqual(ece, 1.0, places=5)
        self.assertAlmostEqual(mce, 1.0, places=5)

    def test_brier_score_known_values(self):
        """Verify Brier score against manual MSE computation."""
        y_true = np.array([[1, 0], [0, 1]], dtype=float)
        y_prob = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
        expected_brier = np.mean([
            (0.8 - 1)**2, (0.2 - 0)**2,
            (0.1 - 0)**2, (0.9 - 1)**2
        ])
        brier = compute_multilabel_brier_score(y_true, y_prob)
        self.assertAlmostEqual(brier, expected_brier, places=6)

    def test_evaluate_multilabel_calibration_structure(self):
        """Verify the full dictionary output of evaluate_multilabel_calibration."""
        n_samples = 200
        n_classes = 5
        genre_names = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]

        y_true = (np.random.rand(n_samples, n_classes) > 0.7).astype(float)
        y_probs = np.random.uniform(0.0, 1.0, size=(n_samples, n_classes))

        res = evaluate_multilabel_calibration(
            y_true=y_true,
            y_pred_probs=y_probs,
            n_bins=10,
            genre_names=genre_names
        )

        self.assertIn("micro_ece", res)
        self.assertIn("macro_ece", res)
        self.assertIn("micro_mce", res)
        self.assertIn("macro_mce", res)
        self.assertIn("brier_score", res)
        self.assertIn("per_class_ece", res)
        self.assertIn("reliability_curve", res)

        self.assertGreaterEqual(res["micro_ece"], 0.0)
        self.assertLessEqual(res["micro_ece"], 1.0)
        self.assertGreaterEqual(res["macro_ece"], 0.0)
        self.assertLessEqual(res["macro_ece"], 1.0)
        self.assertEqual(len(res["per_class_ece"]), n_classes)

        # Check reliability curve structure
        rc = res["reliability_curve"]
        self.assertEqual(len(rc["bin_edges"]), 11)
        self.assertEqual(len(rc["bin_acc"]), 10)
        self.assertEqual(len(rc["bin_conf"]), 10)
        self.assertEqual(len(rc["bin_count"]), 10)
        self.assertEqual(sum(rc["bin_count"]), n_samples * n_classes)

    def test_empty_and_boundary_handling(self):
        """Verify handling of edge values: 0.0, 1.0, clipping, and single sample."""
        y_true = np.array([0, 1], dtype=float)
        y_prob = np.array([-0.5, 1.5], dtype=float)  # Needs clipping
        ece, mce, _ = compute_binary_ece(y_true, y_prob, n_bins=10)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main()

