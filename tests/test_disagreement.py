"""
Unit tests for Cross-Modal Disagreement Auditing (src/auditing/disagreement.py).
Tests:
1. Bernoulli JSD bounds: identity (0.0), maximal divergence (1.0), intermediate range [0, 1].
2. Safe clipping: handles exact 0.0 and 1.0 without NaN/Inf.
3. Jaccard distance: full overlap, disjoint sets, edge case of both empty sets -> 0.0.
4. Hamming distance: normalized symmetric difference.
5. Per-genre disagreement mask and counts.
6. End-to-end compute_cross_modal_disagreement integration.
7. Input validation: negative values, >1.0, NaNs, shape mismatches, non-binary matrices.
"""

import unittest
import numpy as np

from src.auditing.disagreement import (
    bernoulli_jsd,
    mean_bernoulli_jsd,
    jaccard_distance,
    hamming_distance,
    per_genre_disagreement,
    compute_cross_modal_disagreement
)


class TestDisagreementAuditing(unittest.TestCase):

    def test_01_bernoulli_jsd_identity(self):
        """When probabilities are identical, JSD must be exactly 0.0."""
        p = np.array([[0.1, 0.5, 0.9], [0.0, 0.7, 1.0]])
        q = np.copy(p)
        jsd = bernoulli_jsd(p, q)
        np.testing.assert_allclose(jsd, np.zeros_like(p), atol=1e-6)

    def test_02_bernoulli_jsd_extremes_and_bounds(self):
        """When probabilities are completely opposite (0 vs 1), JSD must be close to 1.0."""
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.0, 1.0]])
        jsd = bernoulli_jsd(p, q)
        self.assertTrue(np.all(jsd >= 0.0) and np.all(jsd <= 1.0))
        np.testing.assert_allclose(jsd, np.ones_like(jsd), atol=1e-4)

    def test_03_bernoulli_jsd_symmetry(self):
        """JSD must be symmetric: JSD(p, q) == JSD(q, p)."""
        np.random.seed(42)
        p = np.random.uniform(0.0, 1.0, size=(10, 23))
        q = np.random.uniform(0.0, 1.0, size=(10, 23))
        jsd_pq = bernoulli_jsd(p, q)
        jsd_qp = bernoulli_jsd(q, p)
        np.testing.assert_allclose(jsd_pq, jsd_qp, atol=1e-6)

    def test_04_mean_bernoulli_jsd_shape(self):
        """Mean Bernoulli JSD must return an array of shape (N,)."""
        p = np.full((15, 23), 0.3)
        q = np.full((15, 23), 0.7)
        mean_jsd = mean_bernoulli_jsd(p, q)
        self.assertEqual(mean_jsd.shape, (15,))
        self.assertTrue(np.all(mean_jsd > 0.0))
        self.assertTrue(np.all(mean_jsd <= 1.0))

    def test_05_jaccard_distance_cases(self):
        """Test Jaccard distance for identity, disjoint, and empty sets."""
        # 1. Identical non-empty sets -> distance 0.0
        y1 = np.array([[1, 0, 1, 0]])
        y2 = np.array([[1, 0, 1, 0]])
        self.assertAlmostEqual(jaccard_distance(y1, y2)[0], 0.0)

        # 2. Completely disjoint non-empty sets -> distance 1.0
        y1 = np.array([[1, 1, 0, 0]])
        y2 = np.array([[0, 0, 1, 1]])
        self.assertAlmostEqual(jaccard_distance(y1, y2)[0], 1.0)

        # 3. Partial overlap: intersection = 1, union = 3 -> distance 1 - 1/3 = 2/3
        y1 = np.array([[1, 1, 0, 0]])
        y2 = np.array([[1, 0, 1, 0]])
        self.assertAlmostEqual(jaccard_distance(y1, y2)[0], 2.0 / 3.0)

        # 4. Critical edge case: both sets empty -> distance 0.0 (full agreement on null set)
        y1 = np.array([[0, 0, 0, 0]])
        y2 = np.array([[0, 0, 0, 0]])
        self.assertAlmostEqual(jaccard_distance(y1, y2)[0], 0.0)

    def test_06_hamming_distance(self):
        """Hamming distance must return fraction of disagreeing labels."""
        y1 = np.array([[1, 0, 1, 0]])
        y2 = np.array([[1, 1, 1, 0]])  # 1 disagreement out of 4 -> 0.25
        self.assertAlmostEqual(hamming_distance(y1, y2)[0], 0.25)

    def test_07_per_genre_disagreement(self):
        """Per-genre disagreement provides mask, counts, and stats dictionary."""
        y1 = np.array([
            [1, 0, 1],
            [0, 1, 0]
        ])
        y2 = np.array([
            [1, 1, 1],  # genre 1 disagrees
            [0, 1, 1]   # genre 2 disagrees
        ])
        res = per_genre_disagreement(y1, y2, genre_names=["Action", "Comedy", "Drama"])
        self.assertEqual(res["disagree_counts"].tolist(), [0, 1, 1])
        self.assertEqual(res["per_genre_stats"]["Action"]["disagreement_count"], 0)
        self.assertEqual(res["per_genre_stats"]["Comedy"]["disagreement_count"], 1)
        self.assertEqual(res["per_genre_stats"]["Drama"]["disagreement_count"], 1)

    def test_08_compute_cross_modal_disagreement_full(self):
        """Verify full auditing pipeline returns all required fields and valid metrics."""
        np.random.seed(42)
        n_samples, n_classes = 50, 23
        t_probs = np.random.uniform(0.1, 0.9, size=(n_samples, n_classes))
        i_probs = np.random.uniform(0.1, 0.9, size=(n_samples, n_classes))
        t_thresh = np.full(n_classes, 0.4)
        i_thresh = np.full(n_classes, 0.5)

        res = compute_cross_modal_disagreement(
            text_probs=t_probs,
            image_probs=i_probs,
            text_thresholds=t_thresh,
            image_thresholds=i_thresh
        )

        self.assertIn("mean_bernoulli_jsd", res)
        self.assertIn("jaccard_distance", res)
        self.assertIn("hamming_distance", res)
        self.assertIn("num_disagreeing_genres", res)
        self.assertIn("summary", res)

        self.assertEqual(len(res["mean_bernoulli_jsd"]), n_samples)
        self.assertEqual(len(res["jaccard_distance"]), n_samples)
        self.assertEqual(len(res["hamming_distance"]), n_samples)
        self.assertEqual(res["summary"]["n_samples"], n_samples)
        self.assertEqual(res["summary"]["n_classes"], n_classes)

    def test_09_invalid_inputs_raise_errors(self):
        """Invalid inputs must raise ValueError with clear messages."""
        p = np.array([[0.5, 0.5]])
        q_mismatch = np.array([[0.5, 0.5, 0.5]])
        with self.assertRaises(ValueError):
            bernoulli_jsd(p, q_mismatch)

        p_nan = np.array([[0.5, np.nan]])
        with self.assertRaises(ValueError):
            bernoulli_jsd(p_nan, p)

        p_out_of_bounds = np.array([[1.5, 0.5]])
        with self.assertRaises(ValueError):
            bernoulli_jsd(p_out_of_bounds, p)

        y_non_binary = np.array([[0, 2]])
        with self.assertRaises(ValueError):
            jaccard_distance(y_non_binary, np.array([[0, 1]]))


if __name__ == "__main__":
    unittest.main()
