"""
Pretrained ResNet50 Frozen Feature Extractor for MM-IMDb Poster Images.
Extracts 2048-dimensional global average pooled visual embeddings in batches.
Uses memory-mapped numpy array for incremental checkpointing and zero RAM accumulation.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import MMIMDbDataset, get_default_image_transform

logger = logging.getLogger("ImageFeatureExtractor")


class ResNet50FeatureExtractor:
    def __init__(
        self,
        output_dir: Path,
        batch_size: int = 64,
        num_threads: int = 8,
        device: str = "cpu"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device = torch.device(device)
        
        torch.set_num_threads(self.num_threads)
        
        # Load frozen pretrained ResNet50
        logger.info("Loading pretrained ResNet50 weights...")
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        self.model = model.to(self.device)
        self.feature_dim = 2048
        self.transform = get_default_image_transform(image_size=(224, 224))
        
        self.features_file = self.output_dir / "resnet50_features.npy"
        self.info_file = self.output_dir / "resnet50_features_info.json"

    def extract_all(self, force_recompute: bool = False) -> np.ndarray:
        logger.info("Accessing MM-IMDb dataset...")
        hf_ds = load_dataset("pranavmr/MM-IMDb", split="train")
        total_samples = len(hf_ds)
        logger.info(f"Total samples to extract: {total_samples}")

        # Check if already fully extracted
        if not force_recompute and self.features_file.exists() and self.info_file.exists():
            with open(self.info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            if info.get("status") == "COMPLETE" and info.get("total_samples") == total_samples:
                logger.info(f"Features already fully extracted at {self.features_file}. Loading...")
                return np.memmap(self.features_file, dtype=np.float32, mode="r", shape=(total_samples, self.feature_dim))

        # Create or resume memmap array
        start_idx = 0
        if self.features_file.exists() and self.info_file.exists() and not force_recompute:
            try:
                with open(self.info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                start_idx = info.get("completed_samples", 0)
                logger.info(f"Resuming feature extraction from sample index {start_idx}/{total_samples}...")
            except Exception:
                start_idx = 0

        if start_idx == 0:
            mmap_array = np.memmap(
                self.features_file,
                dtype=np.float32,
                mode="w+",
                shape=(total_samples, self.feature_dim)
            )
        else:
            mmap_array = np.memmap(
                self.features_file,
                dtype=np.float32,
                mode="r+",
                shape=(total_samples, self.feature_dim)
            )

        if start_idx < total_samples:
            remaining_indices = list(range(start_idx, total_samples))
            subset_dataset = MMIMDbDataset(
                hf_dataset=hf_ds,
                indices=remaining_indices,
                split_name="all",
                image_transform=self.transform
            )
            loader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

            logger.info(f"Starting batch extraction for {len(remaining_indices)} images (batch_size={self.batch_size})...")
            start_time = time.time()
            curr_idx = start_idx

            with torch.inference_mode():
                for b_idx, batch in enumerate(loader):
                    imgs = batch["image"].to(self.device)
                    feats = self.model(imgs).cpu().numpy()
                    
                    b_size = feats.shape[0]
                    mmap_array[curr_idx : curr_idx + b_size] = feats
                    curr_idx += b_size
                    
                    # Flush to disk every 5 batches (~320 samples)
                    if (b_idx + 1) % 5 == 0 or curr_idx == total_samples:
                        mmap_array.flush()
                        elapsed = time.time() - start_time
                        processed = curr_idx - start_idx
                        fps = processed / max(0.1, elapsed)
                        eta_sec = (total_samples - curr_idx) / max(0.1, fps)
                        logger.info(
                            f"Extracted {curr_idx}/{total_samples} images "
                            f"({curr_idx/total_samples*100:.1f}%) | "
                            f"Speed: {fps:.1f} imgs/s | "
                            f"ETA: {eta_sec/60:.1f} min"
                        )
                        # Update checkpoint info
                        with open(self.info_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "status": "IN_PROGRESS" if curr_idx < total_samples else "COMPLETE",
                                "total_samples": total_samples,
                                "completed_samples": curr_idx,
                                "feature_dim": self.feature_dim,
                                "model": "resnet50_weights_default",
                                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                            }, f, indent=2)

        mmap_array.flush()
        with open(self.info_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "COMPLETE",
                "total_samples": total_samples,
                "completed_samples": total_samples,
                "feature_dim": self.feature_dim,
                "model": "resnet50_weights_default",
                "completion_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        logger.info("ResNet50 feature extraction completed successfully!")
        return mmap_array
