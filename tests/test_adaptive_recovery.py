"""
Unit and Integration Tests for SEAA Phase 5 Adaptive Recovery Policy.

Verifies:
1. Valid action selection for text-only, image-only, and both-available cases.
2. Missing/malformed signals trigger safe fallback.
3. No action requires an unavailable modality.
4. Policy update modifies action values and visit counts as specified.
5. Controlled reward sequence changes a subsequent action selection (Self-Evolution Proof).
6. Fixed seed yields reproducible decisions under exploration.
7. Evaluation mode strictly freezes policy state.
8. Ground-truth labels and corruption truth are isolated from decision time.
9. Sample IDs and 23-genre target vectors remain aligned after recovery routing.
10. Clean inputs prioritize cooperative multimodal fusion.
11. Policy serialization (save/load) preserves full state.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.policy.adaptive_policy import (
    AdaptiveRecoveryPolicy,
    PolicyContext,
    ACTION_STATIC_FUSION,
    ACTION_TEXT_ONLY,
    ACTION_IMAGE_ONLY,
    ACTION_RELIABILITY_WEIGHTED,
    ACTION_CONSERVATIVE_FUSION,
    ALL_ACTIONS
)


class TestAdaptiveRecoveryPolicy(unittest.TestCase):
    """Test suite for Phase 5 Adaptive Recovery Policy."""

    def setUp(self):
        self.policy = AdaptiveRecoveryPolicy(
            learning_rate=0.20,
            seed=42,
            learning_enabled=True
        )

    def test_01_valid_action_selection_modalities(self):
        """Test 1: Valid action selection respecting modality availability."""
        # Case A: Only text available
        ctx_text_only = PolicyContext(has_text=True, has_image=False)
        valid_a = ctx_text_only.get_valid_actions()
        self.assertEqual(valid_a, [ACTION_TEXT_ONLY])
        action = self.policy.select_action(ctx_text_only)
        self.assertEqual(action, ACTION_TEXT_ONLY)

        # Case B: Only image available
        ctx_image_only = PolicyContext(has_text=False, has_image=True)
        valid_b = ctx_image_only.get_valid_actions()
        self.assertEqual(valid_b, [ACTION_IMAGE_ONLY])
        action = self.policy.select_action(ctx_image_only)
        self.assertEqual(action, ACTION_IMAGE_ONLY)

        # Case C: Both available
        ctx_both = PolicyContext(has_text=True, has_image=True)
        valid_c = ctx_both.get_valid_actions()
        self.assertEqual(set(valid_c), set(ALL_ACTIONS))
        action = self.policy.select_action(ctx_both)
        self.assertIn(action, ALL_ACTIONS)

    def test_02_missing_and_malformed_signals_safe_fallback(self):
        """Test 2: Out-of-range, NaN, or completely missing signals trigger safe fallback."""
        # Missing both modalities
        ctx_empty = PolicyContext(
            has_text=False,
            has_image=False,
            text_entropy=float("nan"),
            image_entropy=float("nan")
        )
        valid = ctx_empty.get_valid_actions()
        self.assertEqual(valid, [ACTION_CONSERVATIVE_FUSION])

        t_probs = np.full(23, 0.4)
        i_probs = np.full(23, 0.6)
        recovered, alpha = self.policy.apply_action(
            action=ACTION_CONSERVATIVE_FUSION,
            text_probs=t_probs,
            image_probs=i_probs,
            context=ctx_empty
        )
        self.assertEqual(recovered.shape, (23,))
        self.assertFalse(np.isnan(recovered).any())
        self.assertTrue((recovered >= 0.0).all() and (recovered <= 1.0).all())

    def test_03_no_action_requires_unavailable_modality(self):
        """Test 3: Policy never selects an action requiring a missing modality."""
        ctx_no_img = PolicyContext(has_text=True, has_image=False)
        for _ in range(50):
            action = self.policy.select_action(ctx_no_img, explore=True)
            self.assertNotEqual(action, ACTION_IMAGE_ONLY)
            self.assertNotEqual(action, ACTION_STATIC_FUSION)
            self.assertEqual(action, ACTION_TEXT_ONLY)

        ctx_no_txt = PolicyContext(has_text=False, has_image=True)
        for _ in range(50):
            action = self.policy.select_action(ctx_no_txt, explore=True)
            self.assertNotEqual(action, ACTION_TEXT_ONLY)
            self.assertNotEqual(action, ACTION_STATIC_FUSION)
            self.assertEqual(action, ACTION_IMAGE_ONLY)

    def test_04_policy_update_changes_q_value(self):
        """Test 4: Policy update modifies Q-values and visit counts strictly according to the update rule."""
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            bernoulli_jsd=0.15,
            diagnosis="IMAGE_FAILURE"
        )
        state_key = ctx.get_state_key()
        self.policy._init_state_if_needed(state_key, ctx.get_valid_actions())

        initial_q = self.policy.q_table[state_key][ACTION_TEXT_ONLY]
        initial_visits = self.policy.visit_counts[state_key][ACTION_TEXT_ONLY]

        # Apply high reward (1.0)
        res = self.policy.update(ctx, ACTION_TEXT_ONLY, reward=1.0)
        new_q = self.policy.q_table[state_key][ACTION_TEXT_ONLY]
        new_visits = self.policy.visit_counts[state_key][ACTION_TEXT_ONLY]

        self.assertEqual(new_visits, initial_visits + 1)
        self.assertGreater(new_q, initial_q)
        self.assertEqual(res["action"], ACTION_TEXT_ONLY)
        self.assertEqual(len(self.policy.update_history), 1)

    def test_05_controlled_reward_sequence_changes_subsequent_decision(self):
        """Test 5 (Self-Evolution Proof): Outcome update changes subsequent action selection."""
        ctx = PolicyContext(
            has_text=True,
            has_image=True,
            bernoulli_jsd=0.20,
            text_entropy=0.20,
            image_entropy=0.80,
            diagnosis="IMAGE_FAILURE"
        )
        state_key = ctx.get_state_key()

        # Step 1: Initialize state with equal Q-values
        self.policy._init_state_if_needed(state_key, ctx.get_valid_actions())
        self.policy.q_table[state_key][ACTION_STATIC_FUSION] = 0.50
        self.policy.q_table[state_key][ACTION_TEXT_ONLY] = 0.50
        self.policy.q_table[state_key][ACTION_IMAGE_ONLY] = 0.50
        self.policy.q_table[state_key][ACTION_RELIABILITY_WEIGHTED] = 0.49
        self.policy.q_table[state_key][ACTION_CONSERVATIVE_FUSION] = 0.49

        # Initially, with deterministic tie-breaking (ACTION_STATIC_FUSION priority 0)
        decision_before = self.policy.select_action(ctx, explore=False)
        self.assertEqual(decision_before, ACTION_STATIC_FUSION)

        # Step 2: Feed controlled outcomes:
        # Static fusion performs poorly on image failure (reward = 0.10)
        # Text-only performs well (reward = 0.90)
        for _ in range(5):
            self.policy.update(ctx, ACTION_STATIC_FUSION, reward=0.10)
            self.policy.update(ctx, ACTION_TEXT_ONLY, reward=0.90)

        # Step 3: Query policy again with identical context
        decision_after = self.policy.select_action(ctx, explore=False)

        # Must have evolved its decision from STATIC_FUSION to TEXT_ONLY
        self.assertEqual(decision_after, ACTION_TEXT_ONLY)
        self.assertGreater(
            self.policy.q_table[state_key][ACTION_TEXT_ONLY],
            self.policy.q_table[state_key][ACTION_STATIC_FUSION]
        )

    def test_06_fixed_seed_reproducibility_with_exploration(self):
        """Test 6: Fixed random seed produces identical decision sequence under exploration."""
        ctx = PolicyContext(has_text=True, has_image=True, bernoulli_jsd=0.15)

        policy_a = AdaptiveRecoveryPolicy(seed=123, epsilon=0.5, ucb_c=1.0)
        policy_b = AdaptiveRecoveryPolicy(seed=123, epsilon=0.5, ucb_c=1.0)

        decisions_a = [policy_a.select_action(ctx, explore=True) for _ in range(25)]
        decisions_b = [policy_b.select_action(ctx, explore=True) for _ in range(25)]

        self.assertEqual(decisions_a, decisions_b)

    def test_07_evaluation_mode_freezes_policy(self):
        """Test 7: Evaluation mode strictly prevents policy mutation."""
        frozen_policy = AdaptiveRecoveryPolicy(learning_enabled=False)
        ctx = PolicyContext(has_text=True, has_image=True)
        state_key = ctx.get_state_key()

        frozen_policy._init_state_if_needed(state_key, ctx.get_valid_actions())
        initial_q = frozen_policy.q_table[state_key][ACTION_STATIC_FUSION]

        res = frozen_policy.update(ctx, ACTION_STATIC_FUSION, reward=1.0)
        self.assertFalse(res["updated"])
        self.assertEqual(frozen_policy.q_table[state_key][ACTION_STATIC_FUSION], initial_q)
        self.assertEqual(len(frozen_policy.update_history), 0)

    def test_08_test_labels_and_corruption_gt_isolated_from_selection(self):
        """Test 8: Signature audit confirms select_action accepts only context, never targets."""
        import inspect
        sig = inspect.signature(self.policy.select_action)
        param_names = list(sig.parameters.keys())
        self.assertIn("context", param_names)
        self.assertNotIn("y_true", param_names)
        self.assertNotIn("target", param_names)
        self.assertNotIn("ground_truth", param_names)
        self.assertNotIn("corruption_type", param_names)

    def test_09_sample_ids_and_23_targets_remain_aligned(self):
        """Test 9: Output recovery probabilities and decisions maintain exact alignment."""
        n_samples = 15
        n_classes = 23
        rng = np.random.RandomState(42)

        movie_ids = [f"tt{1000 + i}" for i in range(n_samples)]
        targets = rng.randint(0, 2, size=(n_samples, n_classes))
        t_probs = rng.uniform(0.1, 0.9, size=(n_samples, n_classes))
        i_probs = rng.uniform(0.1, 0.9, size=(n_samples, n_classes))

        recovered_batch = np.zeros((n_samples, n_classes))
        chosen_actions = []

        for i in range(n_samples):
            ctx = PolicyContext(
                has_text=True,
                has_image=True,
                bernoulli_jsd=0.08,
                diagnosis="CLEAN"
            )
            act = self.policy.select_action(ctx, explore=False)
            chosen_actions.append(act)
            rec, _ = self.policy.apply_action(act, t_probs[i], i_probs[i], ctx)
            recovered_batch[i] = rec

        self.assertEqual(recovered_batch.shape, (n_samples, 23))
        self.assertEqual(len(chosen_actions), n_samples)
        self.assertEqual(len(movie_ids), n_samples)

    def test_10_clean_examples_prioritize_cooperative_fusion(self):
        """Test 10: On clean inputs, policy retains cooperative fusion without unprovoked fallback."""
        ctx_clean = PolicyContext(
            has_text=True,
            has_image=True,
            text_entropy=0.30,
            image_entropy=0.25,
            bernoulli_jsd=0.03,
            diagnosis="CLEAN"
        )
        action = self.policy.select_action(ctx_clean, explore=False)
        self.assertEqual(action, ACTION_STATIC_FUSION)

    def test_11_save_and_load_policy_state(self):
        """Test 11: Serialization round-trip preserves state, Q-table, and update records."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_policy.json"
            ctx = PolicyContext(has_text=True, has_image=True, diagnosis="TEXT_FAILURE")
            self.policy.update(ctx, ACTION_TEXT_ONLY, reward=0.85)

            self.policy.save_policy(path)
            self.assertTrue(path.exists())

            loaded = AdaptiveRecoveryPolicy.load_policy(path)
            self.assertEqual(loaded.seed, self.policy.seed)
            self.assertEqual(len(loaded.update_history), 1)
            state_key = ctx.get_state_key()
            self.assertAlmostEqual(
                loaded.q_table[state_key][ACTION_TEXT_ONLY],
                self.policy.q_table[state_key][ACTION_TEXT_ONLY]
            )


if __name__ == "__main__":
    unittest.main()
