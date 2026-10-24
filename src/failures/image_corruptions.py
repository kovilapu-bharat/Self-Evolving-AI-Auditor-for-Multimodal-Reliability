"""
Controlled Image Modality Failure Simulation for Multimodal Auditing.
Implements reproducible, deterministic visual corruptions applied BEFORE feature extraction:
- Gaussian Blur (spatial frequency degradation)
- Gaussian Noise (pixel-level sensor/channel noise)
- Missing / Unavailable Image (black image fallback per MM-IMDb dataset convention)

Strictly operates on copies of images and tensors without mutating original datasets on disk.
Validates input shape, channel count (RGB), dtypes, and valid numerical ranges.
"""

import io
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance


VALID_IMAGE_CORRUPTIONS = {
    "clean", "gaussian_blur", "gaussian_noise", "missing_image",
    "jpeg_compression", "contrast_reduction"
}


def _validate_image_input(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> str:
    """
    Validates image type, dimensions, and finite values.
    Returns format string: 'pil', 'numpy', or 'torch'.
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            # Image is converted to RGB
            pass
        return "pil"

    elif isinstance(image, np.ndarray):
        if not np.all(np.isfinite(image)):
            raise ValueError("Numpy image array contains NaN or Inf.")
        if image.ndim not in (3, 4):
            raise ValueError(f"Expected 3D (H, W, C) or 4D (N, H, W, C) numpy image, got shape {image.shape}")
        # Validate channel count
        channels = image.shape[-1] if image.ndim == 3 else image.shape[-1]
        if channels not in (1, 3):
            # Check if channel-first (C, H, W)
            if image.shape[0] in (1, 3):
                channels = image.shape[0]
            else:
                raise ValueError(f"Expected 1 or 3 image channels, got {channels}")
        return "numpy"

    elif isinstance(image, torch.Tensor):
        if not torch.all(torch.isfinite(image)):
            raise ValueError("Torch image tensor contains NaN or Inf.")
        if image.ndim not in (3, 4):
            raise ValueError(f"Expected 3D (C, H, W) or 4D (N, C, H, W) image tensor, got shape {image.shape}")
        channels = image.shape[0] if image.ndim == 3 else image.shape[1]
        if channels not in (1, 3):
            raise ValueError(f"Expected 1 or 3 image channels, got {channels}")
        return "torch"

    else:
        raise TypeError(f"Unsupported image type {type(image)}. Expected PIL.Image, np.ndarray, or torch.Tensor.")


def apply_gaussian_blur(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    radius: float = 5.0
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Applies Gaussian blur to image.

    Args:
        image: PIL Image, numpy array (H, W, C), or torch tensor (C, H, W)
        radius: Blur standard deviation / kernel radius

    Returns:
        Blurred image in same format as input.
    """
    if radius < 0.0:
        raise ValueError(f"Blur radius must be non-negative, got {radius}")
    if radius == 0.0:
        return image

    img_format = _validate_image_input(image)

    if img_format == "pil":
        rgb_img = image.convert("RGB") if image.mode != "RGB" else image.copy()
        return rgb_img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif img_format == "numpy":
        # Convert to PIL, apply blur, convert back
        arr = np.copy(image)
        is_float = np.issubdtype(arr.dtype, np.floating)
        if is_float and np.max(arr) <= 1.0:
            pil_img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            return (np.asarray(blurred_pil, dtype=arr.dtype) / 255.0)
        else:
            pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            return np.asarray(blurred_pil, dtype=arr.dtype)

    elif img_format == "torch":
        # Convert torch tensor to numpy/PIL, blur, convert back
        t = image.clone().detach().cpu()
        if t.ndim == 3:
            # (C, H, W) -> (H, W, C)
            np_img = t.permute(1, 2, 0).numpy()
            blurred_np = apply_gaussian_blur(np_img, radius=radius)
            return torch.from_numpy(blurred_np).permute(2, 0, 1).to(image.device)
        else:
            # Batch of tensors
            blurred_tensors = [apply_gaussian_blur(t[i], radius=radius) for i in range(t.shape[0])]
            return torch.stack(blurred_tensors).to(image.device)


def apply_gaussian_noise(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    std: float = 0.25,
    seed: Optional[int] = None
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Adds zero-mean Gaussian noise with specified standard deviation to the image.

    Args:
        image: PIL Image, numpy array, or torch tensor
        std: Noise standard deviation (relative to [0, 1] range; for PIL, scaled by 255)
        seed: Fixed seed for deterministic noise

    Returns:
        Noisy image in same format as input, safely clipped to valid range.
    """
    if std < 0.0:
        raise ValueError(f"Noise std must be non-negative, got {std}")
    if std == 0.0:
        return image

    img_format = _validate_image_input(image)
    rng = np.random.RandomState(seed)

    if img_format == "pil":
        rgb_img = image.convert("RGB") if image.mode != "RGB" else image.copy()
        arr = np.asarray(rgb_img, dtype=np.float32)
        # Scaled by 255
        noise = rng.normal(0.0, std * 255.0, size=arr.shape)
        noisy_arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy_arr, mode="RGB")

    elif img_format == "numpy":
        arr = np.copy(image).astype(np.float32)
        is_normalized = (np.max(arr) <= 1.0 and np.min(arr) >= 0.0)
        scale = 1.0 if is_normalized else 255.0
        noise = rng.normal(0.0, std * scale, size=arr.shape)
        noisy_arr = np.clip(arr + noise, 0.0, scale).astype(image.dtype)
        return noisy_arr

    elif img_format == "torch":
        t = image.clone().detach()
        # Use torch generator for determinism if seed given
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(t.shape, generator=gen, dtype=t.dtype, device="cpu").to(t.device) * std
        else:
            noise = torch.randn_like(t) * std
        return t + noise


def apply_missing_image(
    image: Optional[Any] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Simulates missing or unavailable image by creating a black (0, 0, 0) RGB image.
    Follows MM-IMDb project fallback convention (src/data/dataset.py).
    """
    return Image.new("RGB", image_size, (0, 0, 0))


def apply_jpeg_compression(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    quality: int = 10
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Improvement #10: Simulates low-bandwidth / lossy compression artifacts.

    Args:
        image: Original clean image
        quality: JPEG compression quality level in [1, 100] (lower = more degraded)

    Returns:
        JPEG compressed image in original input format.
    """
    fmt = _validate_image_input(image)
    if fmt == "pil":
        pil_img = cast_pil = image.convert("RGB") if image.mode != "RGB" else image.copy()
    elif fmt == "numpy":
        arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        cast_pil = Image.fromarray(arr)
    else:
        arr = image.detach().cpu().permute(1, 2, 0).numpy()
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        cast_pil = Image.fromarray(arr)

    buffer = io.BytesIO()
    cast_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_pil = Image.open(buffer).convert("RGB")

    if fmt == "pil":
        return compressed_pil
    elif fmt == "numpy":
        res_arr = np.array(compressed_pil, dtype=np.float32)
        return res_arr / 255.0 if image.max() <= 1.0 else res_arr
    else:
        res_arr = np.array(compressed_pil, dtype=np.float32) / 255.0
        return torch.from_numpy(res_arr).permute(2, 0, 1).to(image.device, dtype=image.dtype)


def apply_contrast_reduction(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    factor: float = 0.2
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Improvement #10: Simulates sensor washout / severe contrast reduction.

    Args:
        image: Original clean image
        factor: Contrast multiplier in [0.0, 1.0] (0.0 = solid gray, 1.0 = original)

    Returns:
        Contrast-reduced image in original input format.
    """
    fmt = _validate_image_input(image)
    if fmt == "pil":
        cast_pil = image.convert("RGB") if image.mode != "RGB" else image.copy()
    elif fmt == "numpy":
        arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        cast_pil = Image.fromarray(arr)
    else:
        arr = image.detach().cpu().permute(1, 2, 0).numpy()
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        cast_pil = Image.fromarray(arr)

    enhancer = ImageEnhance.Contrast(cast_pil)
    degraded_pil = enhancer.enhance(factor)

    if fmt == "pil":
        return degraded_pil
    elif fmt == "numpy":
        res_arr = np.array(degraded_pil, dtype=np.float32)
        return res_arr / 255.0 if image.max() <= 1.0 else res_arr
    else:
        res_arr = np.array(degraded_pil, dtype=np.float32) / 255.0
        return torch.from_numpy(res_arr).permute(2, 0, 1).to(image.device, dtype=image.dtype)


def apply_image_corruption(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    corruption_type: str,
    seed: Optional[int] = 42,
    **kwargs
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Dispatches image corruption by name.

    Args:
        image: Original clean image
        corruption_type: One of {'clean', 'gaussian_blur', 'gaussian_noise', 'missing_image', 'jpeg_compression', 'contrast_reduction'}
        seed: Random seed for stochastic corruptions
        **kwargs: Additional parameters (e.g. radius for blur, std for noise)

    Returns:
        Corrupted image.
    """
    ctype = corruption_type.lower().strip()
    if ctype not in VALID_IMAGE_CORRUPTIONS:
        raise ValueError(f"Unsupported image corruption '{corruption_type}'. Valid: {VALID_IMAGE_CORRUPTIONS}")

    if ctype == "clean":
        return image
    elif ctype == "gaussian_blur":
        radius = kwargs.get("radius", 5.0)
        return apply_gaussian_blur(image, radius=radius)
    elif ctype == "gaussian_noise":
        std = kwargs.get("std", 0.25)
        return apply_gaussian_noise(image, std=std, seed=seed)
    elif ctype == "missing_image":
        size = kwargs.get("image_size", (224, 224))
        return apply_missing_image(image, image_size=size)
    elif ctype == "jpeg_compression":
        quality = kwargs.get("quality", 10)
        return apply_jpeg_compression(image, quality=quality)
    elif ctype == "contrast_reduction":
        factor = kwargs.get("factor", 0.2)
        return apply_contrast_reduction(image, factor=factor)


def extract_corrupted_image_features(
    images: List[Image.Image],
    corruption_type: str,
    feature_extractor_model: nn.Module,
    image_transform: Any,
    device: torch.device,
    batch_size: int = 32,
    base_seed: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Applies image corruptions on raw PIL images BEFORE feature extraction,
    then transforms and passes them through ResNet-50 in streaming batches.

    Guarantees feature extraction authentically reflects image-level degradation.

    Args:
        images: List of raw PIL Images
        corruption_type: Corruption to apply ('gaussian_blur', 'gaussian_noise', etc.)
        feature_extractor_model: Frozen ResNet-50 model (output dim 2048)
        image_transform: torchvision transform pipeline (Resize, ToTensor, Normalize)
        device: torch device (cpu or cuda)
        batch_size: Batch size for forward pass
        base_seed: Seed for reproducible noise across images
        **kwargs: Corruption parameters (e.g. radius, std)

    Returns:
        (N, 2048) numpy float32 array of corrupted features.
    """
    feature_extractor_model.eval()
    n_images = len(images)
    features = np.zeros((n_images, 2048), dtype=np.float32)

    with torch.inference_mode():
        for start_idx in range(0, n_images, batch_size):
            end_idx = min(start_idx + batch_size, n_images)
            batch_tensors = []

            for i in range(start_idx, end_idx):
                seed = base_seed + i if base_seed is not None else None
                corrupted_img = apply_image_corruption(
                    images[i], corruption_type=corruption_type, seed=seed, **kwargs
                )
                tensor_img = image_transform(corrupted_img)
                batch_tensors.append(tensor_img)

            batch_stack = torch.stack(batch_tensors).to(device)
            batch_feats = feature_extractor_model(batch_stack).cpu().numpy()
            features[start_idx:end_idx] = batch_feats

    return features
