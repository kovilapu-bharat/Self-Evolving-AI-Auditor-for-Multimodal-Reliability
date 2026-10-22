"""
MM-IMDb PyTorch Dataset and Preprocessing Utilities.
Handles text normalization, image validation, multi-hot target encoding,
and lazy-loading of multimodal samples.
"""

import re
import html
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 23 canonical MM-IMDb genres
GENRES: List[str] = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
    'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Short', 'Sport', 'Thriller', 'War', 'Western'
]

GENRE_TO_IDX: Dict[str, int] = {genre: i for i, genre in enumerate(GENRES)}
IDX_TO_GENRE: Dict[int, str] = {i: genre for i, genre in enumerate(GENRES)}
NUM_CLASSES: int = len(GENRES)


def normalize_text(text: str) -> str:
    """
    Cleans and normalizes text description without stripping informative content.
    - Decodes HTML entities (&amp;, &quot;, etc.)
    - Normalizes unicode characters (NFKC)
    - Collapses redundant whitespace
    - Preserves sentence structure, punctuation, and casing
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Replace unconventional whitespace characters with standard space
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def labels_to_multihot(labels: List[str], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Converts list of genre string labels to a binary multi-hot vector.
    """
    multihot = np.zeros(num_classes, dtype=np.float32)
    for lbl in labels:
        if lbl in GENRE_TO_IDX:
            multihot[GENRE_TO_IDX[lbl]] = 1.0
    return multihot


def multihot_to_labels(multihot: Union[np.ndarray, torch.Tensor]) -> List[str]:
    """
    Converts binary multi-hot vector back to genre string labels.
    """
    if isinstance(multihot, torch.Tensor):
        multihot = multihot.cpu().numpy()
    indices = np.where(multihot > 0.5)[0]
    return [IDX_TO_GENRE[idx] for idx in indices if idx in IDX_TO_GENRE]


def get_default_image_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Standard image transform pipeline:
    Resizes to (224, 224), converts to RGB tensor, normalizes with ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class MMIMDbDataset(Dataset):
    """
    PyTorch Dataset for MM-IMDb multimodal data.
    Lazily loads images from Hugging Face dataset cache to minimize memory overhead.
    """
    def __init__(
        self,
        hf_dataset,
        indices: List[int],
        split_name: str = "train",
        image_transform: Optional[transforms.Compose] = None,
        return_raw_image: bool = False,
        return_raw_text: bool = True
    ):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.split_name = split_name
        self.image_transform = image_transform or get_default_image_transform()
        self.return_raw_image = return_raw_image
        self.return_raw_text = return_raw_text

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        item = self.hf_dataset[global_idx]
        
        movie_id = f"mmimdb_{global_idx:05d}"
        raw_text = item.get("text", "")
        clean_text = normalize_text(raw_text)
        
        raw_labels = item.get("labels", [])
        target_multihot = labels_to_multihot(raw_labels)
        target_tensor = torch.tensor(target_multihot, dtype=torch.float32)
        
        # Load and transform image
        raw_img = item.get("image", None)
        if raw_img is None:
            # Fallback black image if corrupted/missing
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            has_image = False
        else:
            if raw_img.mode != "RGB":
                raw_img = raw_img.convert("RGB")
            img_tensor = self.image_transform(raw_img)
            has_image = True
            
        result = {
            "movie_id": movie_id,
            "global_idx": global_idx,
            "split": self.split_name,
            "target": target_tensor,
            "image": img_tensor,
            "has_image": has_image,
            "has_text": bool(clean_text)
        }
        
        if self.return_raw_text:
            result["text"] = clean_text
        if self.return_raw_image and raw_img is not None:
            result["raw_image"] = raw_img
            
        return result
