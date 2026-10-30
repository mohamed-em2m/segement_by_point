"""Shared constants for the LoveDA point-supervision project."""

from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES = 7

CLASS_NAMES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]

# RGB palette used by every figure in the project (one consistent colour map).
CLASS_COLORS = np.array(
    [
        [0, 0, 0],  # background  (black)
        [230, 0, 0],  # building    (red)
        [128, 128, 128],  # road        (grey)
        [0, 80, 255],  # water       (blue)
        [180, 150, 80],  # barren      (brown)
        [0, 160, 0],  # forest      (green)
        [255, 220, 0],  # agriculture (yellow)
    ],
    dtype=np.uint8,
)

# Original LoveDA id -> training id. 0 (no-data) and every unknown id map to 255.
LOVEDA_ID_TO_TRAIN = np.full(256, IGNORE_INDEX, dtype=np.uint8)
LOVEDA_ID_TO_TRAIN[0] = IGNORE_INDEX
for _src, _dst in enumerate(range(NUM_CLASSES), start=1):
    LOVEDA_ID_TO_TRAIN[_src] = _dst
