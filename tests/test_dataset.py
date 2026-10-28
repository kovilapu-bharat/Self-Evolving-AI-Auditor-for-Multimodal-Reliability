"""
Comprehensive automated test suite for MM-IMDb Data Pipeline.
Covers all 12 mandatory validation checks.
"""

import os
import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import (
    GENRES,
    GENRE_TO_IDX,
    IDX_TO_GENRE,
    NUM_CLASSES,
    normalize_text,
    labels_to_multihot,
    multihot_to_labels,
    MMIMDbDataset
)


class TestMMIMDbDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = project_root / "data"
        cls.reports_dir = project_root / "outputs" / "reports"
        
        # Load artifacts
        cls.metadata_path = cls.data_dir / "metadata.csv"
        cls.label_mapping_path = cls.data_dir / "label_mapping.json"
        cls.split_indices_path = cls.data_dir / "split_indices.json"
        cls.summary_path = cls.reports_dir / "dataset_summary.json"
        
        cls.df_meta = pd.read_csv(cls.metadata_path)
        with open(cls.label_mapping_path, "r", encoding="utf-8") as f:
            cls.label_mapping = json.load(f)
        with open(cls.split_indices_path, "r", encoding="utf-8") as f:
            cls.split_indices = json.load(f)
        with open(cls.summary_path, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)
            
        cls.hf_ds = load_dataset("pranavmr/MM-IMDb", split="train")

    def test_01_unique_movie_ids(self):
        """Check 1: Unique movie IDs across entire metadata."""
        total_rows = len(self.df_meta)
        unique_ids = self.df_meta["movie_id"].nunique()
        self.assertEqual(total_rows, 25959, "Total metadata rows should be exactly 25,959")
        self.assertEqual(unique_ids, 25959, "All movie_ids must be strictly unique")

    def test_02_correct_text_and_image_linkage(self):
        """Check 2: Correct text and image linkage between metadata and source dataset."""
        sample_indices = [0, 100, 1000, 5000, 15000, 25000]
        for s_idx in sample_indices:
            row = self.df_meta.iloc[s_idx]
            raw_item = self.hf_ds[int(row["dataset_idx"])]
            
            # Check text matches normalized source
            self.assertEqual(row["text"], normalize_text(raw_item["text"]))
            
            # Check genres match
            meta_genres = json.loads(row["genres"])
            raw_genres = [g for g in raw_item["labels"] if g in GENRE_TO_IDX]
            self.assertEqual(sorted(meta_genres), sorted(raw_genres))
            
            # Check image exists and is readable
            img = raw_item["image"]
            self.assertIsNotNone(img, f"Image at index {s_idx} must not be None")
            self.assertEqual(img.mode, "RGB")

    def test_03_valid_genre_labels(self):
        """Check 3: All genre labels belong to the predefined 23 genres."""
        self.assertEqual(len(GENRES), 23, "Must have exactly 23 predefined genres")
        for idx, row in self.df_meta.iterrows():
            genres = json.loads(row["genres"])
            self.assertTrue(len(genres) > 0, f"Movie at row {idx} has empty genre list")
            for g in genres:
                self.assertIn(g, GENRE_TO_IDX, f"Genre '{g}' is not a recognized MM-IMDb genre")

    def test_04_correct_multihot_target_dimensions(self):
        """Check 4: Correct multi-hot target dimensions (23-dimensional binary vector)."""
        for i in [0, 500, 10000, 25958]:
            row = self.df_meta.iloc[i]
            target_vals = [int(v) for v in row["target_multihot"].split(",")]
            self.assertEqual(len(target_vals), 23, "Target multihot vector length must be 23")
            self.assertTrue(all(v in [0, 1] for v in target_vals), "Vector elements must be binary 0 or 1")
            self.assertEqual(sum(target_vals), row["num_genres"], "Sum of 1s must equal num_genres")

    def test_05_readable_images_and_expected_channels(self):
        """Check 5: Readable images and expected image channels."""
        test_dataset = MMIMDbDataset(
            hf_dataset=self.hf_ds,
            indices=[0, 1, 2],
            split_name="test"
        )
        sample = test_dataset[0]
        self.assertTrue(sample["has_image"], "Sample has_image flag should be True")
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(sample["image"].shape, (3, 224, 224), "Image tensor shape must be [3, 224, 224]")
        self.assertEqual(sample["image"].dtype, torch.float32)

    def test_06_missing_text_and_missing_images(self):
        """Check 6: Zero missing text and zero missing images."""
        empty_texts = self.df_meta["text"].isna().sum() + (self.df_meta["text"].str.strip() == "").sum()
        self.assertEqual(empty_texts, 0, "No records should have missing or blank text")
        self.assertEqual(self.summary["missing_text_count"], 0)
        self.assertEqual(self.summary["missing_image_count"], 0)

    def test_07_duplicate_records_and_leakage_prevention(self):
        """Check 7: Duplicate text records are strictly confined to the same split with exact reconciliation."""
        import re
        from collections import defaultdict
        text_group_map = defaultdict(list)
        for idx, row in self.df_meta.iterrows():
            norm_text = str(row["text"])
            k = re.sub(r'\W+', '', norm_text.lower())
            if not k:
                k = f"empty_{idx}"
            text_group_map[k].append((idx, row["split"]))
            
        dup_groups = {k: v for k, v in text_group_map.items() if len(v) > 1}
        num_dup_groups = len(dup_groups)
        num_affected = sum(len(v) for v in dup_groups.values())
        num_redundant = num_affected - num_dup_groups
        
        self.assertEqual(num_dup_groups, 11, "Must have exactly 11 duplicate text groups")
        self.assertEqual(num_affected, 25, "Must have exactly 25 affected records across duplicate groups")
        self.assertEqual(num_redundant, 14, "Must have exactly 14 redundant duplicate records (25 - 11 = 14)")
        
        split_groups = {"train": 0, "val": 0, "test": 0}
        split_affected = {"train": 0, "val": 0, "test": 0}
        split_redundant = {"train": 0, "val": 0, "test": 0}
        leakage_count = 0
        
        for k, entries in dup_groups.items():
            splits_present = set(s for _, s in entries)
            if len(splits_present) > 1:
                leakage_count += 1
            else:
                s = list(splits_present)[0]
                split_groups[s] += 1
                split_affected[s] += len(entries)
                split_redundant[s] += len(entries) - 1
                
        self.assertEqual(leakage_count, 0, "Duplicate plot descriptions must never span different splits")
        self.assertEqual(split_groups, {"train": 5, "val": 3, "test": 3}, "Duplicate groups must be 5 Train, 3 Val, 3 Test")
        self.assertEqual(split_affected, {"train": 13, "val": 6, "test": 6}, "Affected records must be 13 Train, 6 Val, 6 Test")
        self.assertEqual(split_redundant, {"train": 8, "val": 3, "test": 3}, "Redundant records must be 8 Train, 3 Val, 3 Test")

    def test_08_disjoint_train_val_test_ids(self):
        """Check 8: Disjoint train, validation, and test index sets."""
        tr = set(self.split_indices["train"])
        va = set(self.split_indices["val"])
        te = set(self.split_indices["test"])
        
        # Test pairwise intersection
        self.assertEqual(len(tr.intersection(va)), 0, "Train and Val must be disjoint")
        self.assertEqual(len(tr.intersection(te)), 0, "Train and Test must be disjoint")
        self.assertEqual(len(va.intersection(te)), 0, "Val and Test must be disjoint")
        
        # Test completeness
        total_indexed = len(tr) + len(va) + len(te)
        self.assertEqual(total_indexed, 25959, "Union of splits must equal full dataset count (25,959)")

    def test_09_consistent_label_mappings_across_splits(self):
        """Check 9: All 23 genres have positive representations in train, val, and test."""
        for s_name in ["train", "val", "test"]:
            s_freqs = self.summary["split_statistics"][s_name]["genre_frequencies"]
            self.assertEqual(len(s_freqs), 23, f"{s_name} split must track all 23 genres")
            for genre, count in s_freqs.items():
                self.assertGreater(count, 0, f"Genre '{genre}' must have at least 1 positive sample in {s_name}")

    def test_10_reproducible_split_generation(self):
        """Check 10: Split generation is deterministic and reproducible."""
        from src.data.pipeline import iterative_multilabel_split
        # Re-run on small matrix with seed 42
        mock_targets = np.array([
            [1, 0], [0, 1], [1, 1], [1, 0], [0, 1],
            [1, 0], [0, 1], [1, 1], [0, 1], [1, 0]
        ], dtype=np.float32)
        mock_groups = [f"g_{i}" for i in range(10)]
        
        split1 = iterative_multilabel_split(mock_targets, mock_groups, (0.6, 0.2, 0.2), seed=42)
        split2 = iterative_multilabel_split(mock_targets, mock_groups, (0.6, 0.2, 0.2), seed=42)
        
        self.assertEqual(split1["train"], split2["train"])
        self.assertEqual(split1["val"], split2["val"])
        self.assertEqual(split1["test"], split2["test"])

    def test_11_successful_loading_small_batch(self):
        """Check 11: Successful batch loading through PyTorch DataLoader."""
        batch_size = 8
        ds = MMIMDbDataset(
            hf_dataset=self.hf_ds,
            indices=self.split_indices["train"][:16],
            split_name="train"
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        
        self.assertEqual(batch["image"].shape, (batch_size, 3, 224, 224))
        self.assertEqual(batch["target"].shape, (batch_size, 23))
        self.assertEqual(len(batch["text"]), batch_size)
        self.assertEqual(len(batch["movie_id"]), batch_size)

    def test_12_no_accidental_test_leakage_in_train(self):
        """Check 12: Absolute test isolation."""
        test_ids = set(self.split_indices["test"])
        train_ids = set(self.split_indices["train"])
        val_ids = set(self.split_indices["val"])
        
        self.assertTrue(test_ids.isdisjoint(train_ids))
        self.assertTrue(test_ids.isdisjoint(val_ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
