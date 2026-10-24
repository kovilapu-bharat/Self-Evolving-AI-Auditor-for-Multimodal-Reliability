"""
Failures module for SEAA.
Provides failure detection, anomaly tracking, and modality disagreement diagnosis.
"""

from src.failures.text_corruptions import (
    VALID_TEXT_CORRUPTIONS,
    truncate_text,
    mask_tokens,
    delete_tokens,
    make_empty_text,
    typo_noise,
    ocr_noise,
    cross_modal_conflict_text,
    apply_text_corruption,
    corrupt_text_batch
)

from src.failures.image_corruptions import (
    VALID_IMAGE_CORRUPTIONS,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_missing_image,
    apply_jpeg_compression,
    apply_contrast_reduction,
    apply_image_corruption,
    extract_corrupted_image_features
)

__all__ = [
    "VALID_TEXT_CORRUPTIONS",
    "truncate_text",
    "mask_tokens",
    "delete_tokens",
    "make_empty_text",
    "typo_noise",
    "ocr_noise",
    "cross_modal_conflict_text",
    "apply_text_corruption",
    "corrupt_text_batch",
    "VALID_IMAGE_CORRUPTIONS",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_missing_image",
    "apply_jpeg_compression",
    "apply_contrast_reduction",
    "apply_image_corruption",
    "extract_corrupted_image_features"
]
