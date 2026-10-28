"""
Unit tests for SEAA Continuous-Context Bandit Policies (LinUCB and Contextual Thompson Sampling).
"""

import unittest
import tempfile
import os
import numpy as np
from src.policy.adaptive_policy import (
    PolicyContext,
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_IMAGE_ONLY,
    ACTION_RELIABILITY_WEIGHTED,
    ACTION_CONSERVATIVE_FUSION
)
from src.policy.linear_bandit import (
    extract_continuous_context_vector,
    LinUCBPolicy,
    ContextualThompsonSamplingPolicy
)


class TestLinearBandits(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_feature_vector_extraction(self):
        ctx = PolicyContext(
            has_text=True,
            has_image=False,
            text_entropy=0.35,
            image_entropy=0.85,
            bernoulli_jsd=0.22,
            text_norm=25.0,
            image_norm=10.0,
            diagnosis="IMAGE_FAILURE"
        )
        x = extract_continuous_context_vector(ctx)
        self.assertEqual(x.shape, (9,))
        self.assertEqual(x[0], 1.0)      # Bias
        self.assertEqual(x[1], 1.0)      # has_text
        self.assertEqual(x[2], 0.0)      # has_image
        self.assertAlmostEqual(x[3], 0.35)
        self.assertAlmostEqual(x[4], 0.85)
        self.assertAlmostEqual(x[5], 0.50)  # h_diff = 0.85 - 0.35
        self.assertAlmostEqual(x[6], 0.22)

    def test_linucb_physical_feasibility_masking(self):
        """When image is missing, LinUCB must strictly return ACTION_TEXT_ONLY."""
        policy = LinUCBPolicy(d=9, alpha=0.5)
        ctx = PolicyContext(has_text=True, has_image=False)
        action, meta = policy.select_action(ctx, frozen=True)
        self.assertEqual(action, ACTION_TEXT_ONLY)

        # When text is missing, must strictly return ACTION_IMAGE_ONLY
        ctx_no_text = PolicyContext(has_text=False, has_image=True)
        action_no_text, _ = policy.select_action(ctx_no_text, frozen=True)
        self.assertEqual(action_no_text, ACTION_IMAGE_ONLY)

    def test_linucb_update_and_learning(self):
        """Verify that positive rewards increase the score of an action."""
        policy = LinUCBPolicy(d=9, alpha=0.0, ridge_lambda=1.0)
        ctx = PolicyContext(has_text=True, has_image=True, text_entropy=0.2, image_entropy=0.8)

        # Baseline: before updates, all actions start with b=0, score=0
        action1, meta1 = policy.select_action(ctx, frozen=True)
        
        # Give high positive reward to ACTION_TEXT_ONLY
        for _ in range(5):
            policy.update(ctx, ACTION_TEXT_ONLY, reward=1.0)

        # Give negative reward to ACTION_IMAGE_ONLY
        for _ in range(5):
            policy.update(ctx, ACTION_IMAGE_ONLY, reward=-0.5)

        action2, meta2 = policy.select_action(ctx, frozen=True)
        self.assertEqual(action2, ACTION_TEXT_ONLY)
        self.assertGreater(meta2["scores"][ACTION_TEXT_ONLY], meta2["scores"][ACTION_IMAGE_ONLY])

    def test_thompson_sampling_frozen_determinism(self):
        """Under frozen mode, Thompson Sampling must be deterministic (greedy MAP)."""
        policy = ContextualThompsonSamplingPolicy(d=9, v=0.5, seed=123)
        ctx = PolicyContext(has_text=True, has_image=True, text_entropy=0.5, image_entropy=0.5)

        # Update an action
        policy.update(ctx, ACTION_RELIABILITY_WEIGHTED, reward=2.0)

        action_a, meta_a = policy.select_action(ctx, frozen=True)
        action_b, meta_b = policy.select_action(ctx, frozen=True)
        self.assertEqual(action_a, action_b)
        self.assertEqual(meta_a["scores"], meta_b["scores"])

    def test_linucb_serialization(self):
        """Verify model serialization to and from JSON."""
        policy = LinUCBPolicy(d=9, alpha=0.3)
        ctx = PolicyContext(has_text=True, has_image=True)
        policy.update(ctx, ACTION_CONSERVATIVE_FUSION, reward=0.85)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            policy.save(tmp_path)
            loaded_policy = LinUCBPolicy(d=9)
            loaded_policy.load(tmp_path)
            self.assertEqual(loaded_policy.counts[ACTION_CONSERVATIVE_FUSION], 1)
            np.testing.assert_allclose(
                loaded_policy.b[ACTION_CONSERVATIVE_FUSION],
                policy.b[ACTION_CONSERVATIVE_FUSION]
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

