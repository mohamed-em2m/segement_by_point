from __future__ import annotations

import numpy as np

from pointseg.constants import CLASS_COLORS, IGNORE_INDEX
from pointseg.evaluation.visualize import colorize, points_overlay


def test_colorize_maps_classes_and_ignore():
    mask = np.array([[0, 1], [6, IGNORE_INDEX]], dtype=np.int64)
    rgb = colorize(mask)
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[0, 0], CLASS_COLORS[0])
    np.testing.assert_array_equal(rgb[0, 1], CLASS_COLORS[1])
    np.testing.assert_array_equal(rgb[1, 0], CLASS_COLORS[6])
    np.testing.assert_array_equal(rgb[1, 1], np.array([220, 220, 220]))


def test_points_overlay_dilates_without_touching_the_mask():
    mask = np.full((9, 9), IGNORE_INDEX, dtype=np.int64)
    mask[4, 4] = 2
    before = mask.copy()
    overlay = points_overlay(mask, radius=1)
    np.testing.assert_array_equal(mask, before)  # display only
    np.testing.assert_array_equal(overlay[4, 4], CLASS_COLORS[2])
    # The 3x3 neighbourhood around the point is painted too.
    np.testing.assert_array_equal(overlay[3, 3], CLASS_COLORS[2])
    np.testing.assert_array_equal(overlay[0, 0], np.array([235, 235, 235]))


def test_points_overlay_uses_the_image_as_background():
    mask = np.full((5, 5), IGNORE_INDEX, dtype=np.int64)
    mask[2, 2] = 1
    base = np.full((5, 5, 3), 77, dtype=np.uint8)
    overlay = points_overlay(mask, base=base, radius=0)
    np.testing.assert_array_equal(overlay[0, 0], np.array([77, 77, 77]))
    np.testing.assert_array_equal(overlay[2, 2], CLASS_COLORS[1])
