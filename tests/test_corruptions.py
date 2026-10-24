"""
Unit tests for Modality Failure Simulation (src/failures/text_corruptions.py and image_corruptions.py).
Tests:
1. Text truncation: keep_ratio, max_words, empty strings, single words.
2. Token masking: determinism with seeds, mask token replacement, preservation of length.
3. Token deletion: drop_rate, empty return when drop_rate=1.0.
4. Missing text: returns empty string.
5. Image Gaussian blur: shape and mode preservation, pixel alterations.
6. Image Gaussian noise: determinism, value clipping to valid pixel ranges.
7. Missing image: black image representation matches (224, 224, 3) RGB standard.
8. Validation & edge cases: invalid parameters, negative rates, NaNs, unsupported types.
9. Preservation: original inputs remain completely unmodified.
"""

import unittest
import numpy as np
import torch
from PIL import Image

from src.failures.text_corruptions import (
    truncate_text,
    mask_tokens,
    delete_tokens,
    make_empty_text,
    apply_text_corruption,
    corrupt_text_batch
)
from src.failures.image_corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_missing_image,
    apply_image_corruption
)


class TestModalityCorruptions(unittest.TestCase):

    def setUp(self):
        self.sample_text = "The quick brown fox jumps over the lazy dog in a vibrant forest."
        self.sample_img = Image.new("RGB", (64, 64), color=(128, 128, 128))

    # --- Text Corruption Tests ---

    def test_01_text_truncation_ratio_and_max_words(self):
        """Truncation preserves the beginning and trims words to target count."""
        words = self.sample_text.split()
        n_words = len(words)

        # Truncate by keep_ratio = 0.5
        t_ratio = truncate_text(self.sample_text, keep_ratio=0.5)
        self.assertEqual(len(t_ratio.split()), int(np.ceil(n_words * 0.5)))
        self.assertTrue(self.sample_text.startswith(t_ratio))

        # Truncate by max_words = 4
        t_max = truncate_text(self.sample_text, max_words=4)
        self.assertEqual(len(t_max.split()), 4)
        self.assertEqual(t_max, "The quick brown fox")

        # Edge case: empty text
        self.assertEqual(truncate_text("", max_words=5), "")

    def test_02_token_masking_determinism(self):
        """Masking with fixed seed produces identical outputs; different seeds vary."""
        masked_1 = mask_tokens(self.sample_text, mask_rate=0.4, mask_token="[MASK]", seed=42)
        masked_2 = mask_tokens(self.sample_text, mask_rate=0.4, mask_token="[MASK]", seed=42)
        masked_3 = mask_tokens(self.sample_text, mask_rate=0.4, mask_token="[MASK]", seed=999)

        self.assertEqual(masked_1, masked_2)
        self.assertIn("[MASK]", masked_1)
        self.assertEqual(len(masked_1.split()), len(self.sample_text.split()))
        # Different seeds produce different mask selections
        self.assertNotEqual(masked_1, masked_3)

    def test_03_token_deletion(self):
        """Token deletion drops words; drop_rate=1.0 returns empty string."""
        deleted = delete_tokens(self.sample_text, drop_rate=0.4, seed=42)
        self.assertLess(len(deleted.split()), len(self.sample_text.split()))

        # Complete deletion
        empty_del = delete_tokens(self.sample_text, drop_rate=1.0, seed=42)
        self.assertEqual(empty_del, "")

    def test_04_missing_text_and_batch_runner(self):
        """make_empty_text always returns empty string; corrupt_text_batch works across samples."""
        self.assertEqual(make_empty_text(self.sample_text), "")

        batch = [self.sample_text, "A romantic drama set in New York.", "An intergalactic space battle."]
        corrupted_batch = corrupt_text_batch(batch, corruption_type="truncation", keep_ratio=0.33)
        self.assertEqual(len(corrupted_batch), 3)
        for original, corrupted in zip(batch, corrupted_batch):
            self.assertLess(len(corrupted.split()), len(original.split()))

    def test_05_text_corruption_validation(self):
        """Invalid rates or types raise ValueError."""
        with self.assertRaises(ValueError):
            truncate_text(self.sample_text, keep_ratio=1.5)
        with self.assertRaises(ValueError):
            mask_tokens(self.sample_text, mask_rate=-0.1)
        with self.assertRaises(ValueError):
            delete_tokens(self.sample_text, drop_rate=1.1)
        with self.assertRaises(ValueError):
            apply_text_corruption(self.sample_text, corruption_type="unknown_corruption")

    # --- Image Corruption Tests ---

    def test_06_gaussian_blur_pil(self):
        """Gaussian blur modifies pixel values while retaining image dimensions and RGB mode."""
        # Create patterned image with sharp edge
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        for x in range(16, 32):
            for y in range(32):
                img.putpixel((x, y), (255, 255, 255))

        blurred = apply_gaussian_blur(img, radius=3.0)
        self.assertEqual(blurred.size, (32, 32))
        self.assertEqual(blurred.mode, "RGB")

        # Pixels near edge (x=15, 16) should have intermediate blended values
        blurred_arr = np.asarray(blurred)
        self.assertTrue(np.any((blurred_arr > 0) & (blurred_arr < 255)))

    def test_07_gaussian_noise_pil(self):
        """Gaussian noise adds variance, preserves shape and range [0, 255]."""
        noisy_1 = apply_gaussian_noise(self.sample_img, std=0.2, seed=42)
        noisy_2 = apply_gaussian_noise(self.sample_img, std=0.2, seed=42)
        noisy_3 = apply_gaussian_noise(self.sample_img, std=0.2, seed=123)

        self.assertEqual(noisy_1.size, self.sample_img.size)
        self.assertEqual(noisy_1.mode, "RGB")
        
        arr_1 = np.asarray(noisy_1)
        arr_2 = np.asarray(noisy_2)
        arr_3 = np.asarray(noisy_3)

        # Determinism
        np.testing.assert_array_equal(arr_1, arr_2)
        # Differing seeds
        self.assertFalse(np.array_equal(arr_1, arr_3))
        # Valid uint8 bounds
        self.assertTrue(np.all(arr_1 >= 0) and np.all(arr_1 <= 255))

    def test_08_missing_image_representation(self):
        """Missing image creates black (0, 0, 0) image of default size (224, 224)."""
        black_img = apply_missing_image()
        self.assertEqual(black_img.size, (224, 224))
        self.assertEqual(black_img.mode, "RGB")
        arr = np.asarray(black_img)
        self.assertEqual(np.sum(arr), 0)

    def test_09_image_corruption_validation(self):
        """Invalid blur radius, noise std, or non-finite inputs raise ValueError."""
        with self.assertRaises(ValueError):
            apply_gaussian_blur(self.sample_img, radius=-1.0)
        with self.assertRaises(ValueError):
            apply_gaussian_noise(self.sample_img, std=-0.5)
        with self.assertRaises(ValueError):
            apply_image_corruption(self.sample_img, corruption_type="invalid_type")

        # Non-finite numpy array
        nan_arr = np.array([[[np.nan, 0.0, 0.0]]], dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_gaussian_noise(nan_arr)

    def test_10_original_source_preservation(self):
        """Original text and images remain untouched by corruptions."""
        orig_text = str(self.sample_text)
        _ = truncate_text(self.sample_text, keep_ratio=0.2)
        _ = mask_tokens(self.sample_text, mask_rate=0.8, seed=42)
        self.assertEqual(self.sample_text, orig_text)

        orig_pixels = np.copy(np.asarray(self.sample_img))
        _ = apply_gaussian_noise(self.sample_img, std=0.5, seed=42)
        _ = apply_gaussian_blur(self.sample_img, radius=10.0)
        np.testing.assert_array_equal(np.asarray(self.sample_img), orig_pixels)


if __name__ == "__main__":
    unittest.main()
