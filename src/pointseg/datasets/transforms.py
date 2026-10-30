"""Aligned image/mask transforms.

Every geometric transform is applied to the image and the mask with the same
parameters, so image/mask alignment is preserved. Mask values are never
interpolated: crops and flips/rotations are exact pixel operations.
"""

from __future__ import annotations

import numpy as np

from pointseg.constants import LOVEDA_ID_TO_TRAIN

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Map original LoveDA ids (0-7) to training ids (0-6, 255 for no-data)."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.dtype != np.uint8:
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"mask must be integer, got {mask.dtype}")
        if mask.max(initial=0) > 255 or mask.min(initial=0) < 0:
            raise ValueError("mask values must fit in uint8")
        mask = mask.astype(np.uint8)
    return LOVEDA_ID_TO_TRAIN[mask]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW, ImageNet-normalized."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HWC RGB, got shape {image.shape}")
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def denormalize_image(tensor: np.ndarray) -> np.ndarray:
    """float32 CHW normalized -> uint8 HWC RGB (for figures)."""
    x = tensor.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Take an aligned random crop, padding reflectively when the image is small."""
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("image and mask spatial shapes differ")
    h, w = mask.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
        h, w = mask.shape
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return (
        image[top : top + size, left : left + size],
        mask[top : top + size, left : left + size],
    )


def augment(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    rotations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal/vertical flips and optional 90-degree rotations."""
    if rng.random() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        image = image[::-1, :]
        mask = mask[::-1, :]
    if rotations:
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
