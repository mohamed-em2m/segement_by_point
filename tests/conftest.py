from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_pair(root: Path, split: str, domain: str, stem: str, seed: int) -> None:
    image_dir = root / split / domain / "images_png"
    mask_dir = root / split / domain / "masks_png"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = rng.integers(0, 8, size=(64, 64), dtype=np.uint8)
    cv2.imwrite(str(image_dir / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)


@pytest.fixture()
def synthetic_loveda(tmp_path: Path) -> Path:
    root = tmp_path / "LoveDA"
    for index, stem in enumerate(["0", "1", "10"]):
        _write_pair(root, "Train", "Rural", stem, seed=index)
    for index, stem in enumerate(["20", "21"]):
        _write_pair(root, "Train", "Urban", stem, seed=10 + index)
    for index, stem in enumerate(["100", "101"]):
        _write_pair(root, "Val", "Rural", stem, seed=20 + index)
    _write_pair(root, "Val", "Urban", "200", seed=30)
    return root
