"""
Data module for MM-IMDb dataset handling, preprocessing, and pipeline execution.
"""

from src.data.dataset import (
    GENRES,
    GENRE_TO_IDX,
    IDX_TO_GENRE,
    NUM_CLASSES,
    normalize_text,
    labels_to_multihot,
    multihot_to_labels,
    get_default_image_transform,
    MMIMDbDataset
)

__all__ = [
    "GENRES",
    "GENRE_TO_IDX",
    "IDX_TO_GENRE",
    "NUM_CLASSES",
    "normalize_text",
    "labels_to_multihot",
    "multihot_to_labels",
    "get_default_image_transform",
    "MMIMDbDataset"
]
