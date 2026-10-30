from __future__ import annotations

import numpy as np
import pytest

from pointseg.constants import IGNORE_INDEX
from pointseg.datasets.transforms import (
    augment,
    normalize_image,
    random_crop,
    remap_mask,
)


def test_remap_matches_loveda_table():
    original = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 255]], dtype=np.uint8)
    expected = np.array([[255, 0, 1, 2, 3, 4, 5, 6, 255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(remap_mask(original), expected)


def test_remap_rejects_non_integer():
    with pytest.raises(ValueError):
        remap_mask(np.zeros((4, 4), dtype=np.float32))


def test_normalize_shape_and_range():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    normalized = normalize_image(image)
    assert normalized.shape == (3, 8, 8)
    assert normalized.dtype == np.float32
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(normalized[:, 0, 0], expected, rtol=1e-6)


def test_random_crop_keeps_alignment():
    rows, cols = np.mgrid[0:16, 0:16]
    pixel = (rows * 16 + cols).astype(np.uint8)
    image = np.stack([pixel, pixel, pixel], axis=-1)
    mask = ((rows + cols) % 7).astype(np.uint8)
    rng = np.random.default_rng(0)
    crop_image, crop_mask = random_crop(image, mask, 8, rng)
    assert crop_image.shape == (8, 8, 3)
    assert crop_mask.shape == (8, 8)
    # Every mask pixel must still correspond to the same image pixel.
    for row in range(8):
        for col in range(8):
            value = int(crop_image[row, col, 0])
            original_row, original_col = divmod(value, 16)
            assert crop_mask[row, col] == mask[original_row, original_col]


def test_augment_keeps_alignment():
    # row index is encoded in the image and class in the mask, so alignment
    # can be checked after any flip/rotation.
    rows, cols = np.mgrid[0:8, 0:8]
    image = np.stack([rows * 10 + cols] * 3, axis=-1).astype(np.uint8)
    mask = (rows % 7).astype(np.uint8)
    for seed in range(10):
        aug_image, aug_mask = augment(image, mask, np.random.default_rng(seed))
        decoded_rows = aug_image[..., 0].astype(int) // 10
        np.testing.assert_array_equal(aug_mask, (decoded_rows % 7).astype(np.uint8))


def test_small_image_is_padded_not_crashing():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    crop_image, crop_mask = random_crop(image, mask, 8, np.random.default_rng(0))
    assert crop_image.shape == (8, 8, 3)
    assert crop_mask.shape == (8, 8)
    assert (crop_mask == IGNORE_INDEX).any()
