"""
Unit tests for Predictive Uncertainty Estimation (src/auditing/uncertainty.py).
Tests:
1. Binary entropy bounds and extremes: H(0)=0, H(1)=0, H(0.5)=1.0.
2. Entropy symmetry and monotonicity around 0.5.
3. Threshold margin accuracy with per-class and scalar thresholds.
4. Threshold boundary proximity (delta filtering and counts).
5. Composite uncertainty score bounds [0, 1].
6. Integration test for compute_predictive_uncertainty across modalities.
7. Validation on invalid inputs (NaNs, negatives, >1.0, shape mismatches, negative delta).
"""

import unittest
import numpy as np

from src.auditing.uncertainty import (
    binary_predictive_entropy,
    mean_binary_entropy,
    threshold_margins,
    mean_threshold_margin,
    min_threshold_margin,
    threshold_proximity_stats,
    compute_predictive_uncertainty
)


class TestUncertaintyAuditing(unittest.TestCase):

    def test_01_binary_entropy_extremes(self):
        """H(0) = 0, H(1) = 0, H(0.5) = 1.0 (with base-2 log)."""
        probs = np.array([[0.0, 0.5, 1.0]])
        h = binary_predictive_entropy(probs)
        self.assertAlmostEqual(h[0, 0], 0.0, places=4)
        self.assertAlmostEqual(h[0, 1], 1.0, places=4)
        self.assertAlmostEqual(h[0, 2], 0.0, places=4)

    def test_02_binary_entropy_symmetry_and_range(self):
        """Entropy must be symmetric: H(p) == H(1-p), bounded in [0, 1]."""
        p_vals = np.linspace(0.01, 0.99, 50)
        h_vals = binary_predictive_entropy(p_vals)
        h_inv = binary_predictive_entropy(1.0 - p_vals)
        np.testing.assert_allclose(h_vals, h_inv, atol=1e-5)
        self.assertTrue(np.all(h_vals >= 0.0) and np.all(h_vals <= 1.0))

    def test_03_mean_binary_entropy_shape(self):
        """Mean entropy over classes must reduce last dimension to (N,)."""
        probs = np.full((12, 23), 0.5)
        mean_h = mean_binary_entropy(probs)
        self.assertEqual(mean_h.shape, (12,))
        np.testing.assert_allclose(mean_h, np.ones(12), atol=1e-4)

    def test_04_threshold_margins_scalar_and_vector(self):
        """Threshold margin must accurately compute |p - tau|."""
        probs = np.array([[0.2, 0.7, 0.5]])
        
        # Scalar threshold 0.5
        m_scalar = threshold_margins(probs, thresholds=0.5)
        np.testing.assert_allclose(m_scalar, [[0.3, 0.2, 0.0]], atol=1e-6)

        # Vector threshold [0.3, 0.6, 0.4]
        thresh_vec = np.array([0.3, 0.6, 0.4])
        m_vec = threshold_margins(probs, thresholds=thresh_vec)
        np.testing.assert_allclose(m_vec, [[0.1, 0.1, 0.1]], atol=1e-6)

    def test_05_mean_and_min_threshold_margin(self):
        """Mean and min margin reductions."""
        probs = np.array([
            [0.45, 0.50, 0.90],
            [0.10, 0.20, 0.30]
        ])
        thresh = np.array([0.50, 0.50, 0.50])
        
        mean_m = mean_threshold_margin(probs, thresh)
        min_m = min_threshold_margin(probs, thresh)

        self.assertEqual(mean_m.shape, (2,))
        self.assertEqual(min_m.shape, (2,))
        # First sample margins: [0.05, 0.0, 0.40] -> min is 0.0, mean is 0.15
        self.assertAlmostEqual(min_m[0], 0.0, places=5)
        self.assertAlmostEqual(mean_m[0], 0.15, places=5)

    def test_06_threshold_proximity_stats(self):
        """Counts and proportions of predictions within delta margin."""
        probs = np.array([
            [0.49, 0.51, 0.80, 0.10]
        ])
        thresh = np.array([0.50, 0.50, 0.50, 0.50])
        # With delta=0.02: indices 0 and 1 have margin 0.01 <= 0.02; others have 0.30, 0.40 > 0.02
        counts, props = threshold_proximity_stats(probs, thresh, delta=0.02)
        self.assertEqual(counts[0], 2)
        self.assertEqual(props[0], 0.5)

    def test_07_compute_predictive_uncertainty_integration(self):
        """Full uncertainty auditor test on simulated 23-genre batch."""
        np.random.seed(42)
        n_samples = 30
        probs = np.random.uniform(0.05, 0.95, size=(n_samples, 23))
        thresh = np.random.uniform(0.3, 0.7, size=23)

        res = compute_predictive_uncertainty(
            probs=probs,
            thresholds=thresh,
            delta=0.05,
            modality_name="text"
        )

        self.assertEqual(res["modality"], "text")
        self.assertEqual(len(res["mean_binary_entropy"]), n_samples)
        self.assertEqual(len(res["mean_margin"]), n_samples)
        self.assertEqual(len(res["min_margin"]), n_samples)
        self.assertEqual(len(res["composite_uncertainty"]), n_samples)
        self.assertTrue(np.all(res["composite_uncertainty"] >= 0.0))
        self.assertTrue(np.all(res["composite_uncertainty"] <= 1.0))
        self.assertIn("summary", res)
        self.assertEqual(res["summary"]["n_samples"], n_samples)
        self.assertEqual(res["summary"]["n_classes"], 23)

    def test_08_invalid_inputs_raise_errors(self):
        """Invalid probabilities or thresholds must trigger informative ValueErrors."""
        # Non-finite probability
        with self.assertRaises(ValueError):
            binary_predictive_entropy(np.array([[0.5, np.nan]]))

        # Probability out of bounds
        with self.assertRaises(ValueError):
            binary_predictive_entropy(np.array([[-0.1, 0.5]]))

        with self.assertRaises(ValueError):
            binary_predictive_entropy(np.array([[1.1, 0.5]]))

        # Threshold shape mismatch (2 classes in probs, 3 in thresholds)
        with self.assertRaises(ValueError):
            threshold_margins(np.array([[0.5, 0.5]]), thresholds=np.array([0.5, 0.5, 0.5]))

        # Threshold out of bounds
        with self.assertRaises(ValueError):
            threshold_margins(np.array([[0.5, 0.5]]), thresholds=1.5)

        # Negative delta
        with self.assertRaises(ValueError):
            threshold_proximity_stats(np.array([[0.5, 0.5]]), delta=-0.01)


if __name__ == "__main__":
    unittest.main()
