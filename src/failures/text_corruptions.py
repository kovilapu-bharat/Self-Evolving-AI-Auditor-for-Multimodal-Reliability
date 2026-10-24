"""
Controlled Text Modality Failure Simulation for Multimodal Auditing.
Implements reproducible, deterministic text corruptions:
- Text Truncation (by word limit or retention fraction)
- Token Masking (substituting words with a mask token)
- Token Deletion (stochastic word dropout)
- Missing / Empty Text (empty string fallback)

All functions strictly operate on string copies and do not mutate original inputs or files.
"""

from typing import List, Optional, Union
import numpy as np


VALID_TEXT_CORRUPTIONS = {"clean", "truncation", "masking", "deletion", "empty"}


def truncate_text(
    text: str,
    keep_ratio: Optional[float] = None,
    max_words: Optional[int] = None
) -> str:
    """
    Truncates text by retaining either a fraction of words or a maximum word count.

    Args:
        text: Input text string
        keep_ratio: Fraction of words to retain in (0.0, 1.0]
        max_words: Absolute maximum number of words to retain

    Returns:
        Truncated text string.
    """
    if not text or not isinstance(text, str):
        return ""

    words = text.split()
    if not words:
        return ""

    n_words = len(words)

    if keep_ratio is not None:
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"keep_ratio must be in (0.0, 1.0], got {keep_ratio}")
        retain_n = max(1, int(np.ceil(n_words * keep_ratio)))
    elif max_words is not None:
        if max_words < 0:
            raise ValueError(f"max_words must be non-negative, got {max_words}")
        retain_n = max_words
    else:
        raise ValueError("Either keep_ratio or max_words must be provided.")

    return " ".join(words[:retain_n])


def mask_tokens(
    text: str,
    mask_rate: float = 0.3,
    mask_token: str = "[MASK]",
    seed: Optional[int] = None
) -> str:
    """
    Deterministically masks words in text with a mask token.

    Args:
        text: Input string
        mask_rate: Probability of masking each word in [0.0, 1.0]
        mask_token: Replacement string token
        seed: Deterministic random seed

    Returns:
        Text with masked tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    if mask_rate < 0.0 or mask_rate > 1.0:
        raise ValueError(f"mask_rate must be in [0.0, 1.0], got {mask_rate}")

    words = text.split()
    if not words:
        return ""

    if mask_rate == 0.0:
        return text

    rng = np.random.RandomState(seed)
    mask_decisions = rng.uniform(0.0, 1.0, size=len(words)) < mask_rate

    # If mask_rate > 0 and no token got masked by chance, force at least one mask
    if not np.any(mask_decisions) and len(words) > 0 and mask_rate > 0.0:
        forced_idx = rng.randint(0, len(words))
        mask_decisions[forced_idx] = True

    corrupted_words = [mask_token if masked else w for w, masked in zip(words, mask_decisions)]
    return " ".join(corrupted_words)


def delete_tokens(
    text: str,
    drop_rate: float = 0.3,
    seed: Optional[int] = None
) -> str:
    """
    Deterministically drops words with a specified dropout rate.

    Args:
        text: Input string
        drop_rate: Probability of dropping each word in [0.0, 1.0]
        seed: Deterministic random seed

    Returns:
        Text with selected tokens deleted. Returns empty string if all words are dropped.
    """
    if not text or not isinstance(text, str):
        return ""

    if drop_rate < 0.0 or drop_rate > 1.0:
        raise ValueError(f"drop_rate must be in [0.0, 1.0], got {drop_rate}")

    words = text.split()
    if not words:
        return ""

    if drop_rate == 0.0:
        return text
    if drop_rate == 1.0:
        return ""

    rng = np.random.RandomState(seed)
    keep_decisions = rng.uniform(0.0, 1.0, size=len(words)) >= drop_rate

    surviving_words = [w for w, kept in zip(words, keep_decisions) if kept]
    return " ".join(surviving_words)


def make_empty_text(text: str = "") -> str:
    """
    Simulates completely missing text description. Always returns empty string.
    """
    return ""


def typo_noise(
    text: str,
    typo_rate: float = 0.15,
    seed: Optional[int] = None
) -> str:
    """
    Improvement #10: Simulates keyboard typos and character swaps/substitutions.

    Args:
        text: Input string
        typo_rate: Probability of corrupting each character in [0.0, 1.0]
        seed: Deterministic random seed

    Returns:
        Text with typographical noise.
    """
    if not text or not isinstance(text, str):
        return ""

    if typo_rate <= 0.0:
        return text

    rng = np.random.RandomState(seed)
    chars = list(text)
    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'hukmn',
        'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }

    for i in range(len(chars)):
        c = chars[i].lower()
        if c in keyboard_neighbors and rng.uniform(0.0, 1.0) < typo_rate:
            options = keyboard_neighbors[c]
            replacement = options[rng.randint(0, len(options))]
            chars[i] = replacement.upper() if chars[i].isupper() else replacement

    return "".join(chars)


def ocr_noise(
    text: str,
    ocr_rate: float = 0.20,
    seed: Optional[int] = None
) -> str:
    """
    Improvement #10: Simulates optical character recognition (OCR) degradation.
    Substitutes visually ambiguous characters (e.g. l -> 1, O -> 0, rn -> m).

    Args:
        text: Input string
        ocr_rate: Probability of perturbing ambiguous tokens in [0.0, 1.0]
        seed: Deterministic random seed

    Returns:
        Text with simulated OCR errors.
    """
    if not text or not isinstance(text, str):
        return ""

    if ocr_rate <= 0.0:
        return text

    rng = np.random.RandomState(seed)
    ocr_map = {
        'o': '0', 'O': '0', 'l': '1', 'I': '1', 'i': '!', 's': '5', 'S': '5',
        'b': '6', 'g': '9', 'z': '2', 'Z': '2', 'e': 'c', 'c': 'e'
    }

    chars = list(text)
    for i in range(len(chars)):
        c = chars[i]
        if c in ocr_map and rng.uniform(0.0, 1.0) < ocr_rate:
            chars[i] = ocr_map[c]

    return "".join(chars)


def cross_modal_conflict_text(
    text: str,
    conflict_text: str = "A heartwarming romantic comedy about two florists falling in love in Paris.",
    seed: Optional[int] = None
) -> str:
    """
    Improvement #5: Injects cross-modal conflict by replacing or prepending
    counter-genre text to induce genuine cross-modal disagreement.

    Args:
        text: Original text
        conflict_text: Conflicting synopsis text to replace/inject
        seed: Deterministic random seed

    Returns:
        Conflicting text string.
    """
    return conflict_text


VALID_TEXT_CORRUPTIONS = {
    "clean", "truncation", "masking", "deletion", "empty", "typo", "ocr", "conflict"
}


def apply_text_corruption(
    text: str,
    corruption_type: str,
    seed: Optional[int] = 42,
    **kwargs
) -> str:
    """
    Dispatches text corruption by type name with parameter defaults.

    Args:
        text: Original clean text
        corruption_type: One of {'clean', 'truncation', 'masking', 'deletion', 'empty', 'typo', 'ocr', 'conflict'}
        seed: Fixed seed for stochastic corruptions
        **kwargs: Additional parameters for specific corruptions (e.g. keep_ratio, mask_rate)

    Returns:
        Corrupted text string.
    """
    ctype = corruption_type.lower().strip()
    if ctype not in VALID_TEXT_CORRUPTIONS:
        raise ValueError(f"Unsupported text corruption '{corruption_type}'. Valid: {VALID_TEXT_CORRUPTIONS}")

    if ctype == "clean":
        return str(text) if text else ""
    elif ctype == "truncation":
        keep_ratio = kwargs.get("keep_ratio", 0.25)
        max_words = kwargs.get("max_words", None)
        return truncate_text(text, keep_ratio=keep_ratio, max_words=max_words)
    elif ctype == "masking":
        mask_rate = kwargs.get("mask_rate", 0.3)
        mask_token = kwargs.get("mask_token", "[MASK]")
        return mask_tokens(text, mask_rate=mask_rate, mask_token=mask_token, seed=seed)
    elif ctype == "deletion":
        drop_rate = kwargs.get("drop_rate", 0.3)
        return delete_tokens(text, drop_rate=drop_rate, seed=seed)
    elif ctype == "empty":
        return make_empty_text(text)
    elif ctype == "typo":
        typo_rate = kwargs.get("typo_rate", 0.15)
        return typo_noise(text, typo_rate=typo_rate, seed=seed)
    elif ctype == "ocr":
        ocr_rate = kwargs.get("ocr_rate", 0.20)
        return ocr_noise(text, ocr_rate=ocr_rate, seed=seed)
    elif ctype == "conflict":
        conflict_text = kwargs.get(
            "conflict_text",
            "A hilarious romantic comedy featuring quirky antics, lighthearted laughs, and heartfelt romance."
        )
        return cross_modal_conflict_text(text, conflict_text=conflict_text, seed=seed)


def corrupt_text_batch(
    texts: List[str],
    corruption_type: str,
    base_seed: int = 42,
    **kwargs
) -> List[str]:
    """
    Applies deterministic corruption across a list of texts.
    Uses sample index offset to ensure diverse yet fully reproducible corruptions across batch.
    """
    corrupted_batch = []
    for i, t in enumerate(texts):
        seed = base_seed + i if base_seed is not None else None
        corrupted_batch.append(apply_text_corruption(t, corruption_type=corruption_type, seed=seed, **kwargs))
    return corrupted_batch

