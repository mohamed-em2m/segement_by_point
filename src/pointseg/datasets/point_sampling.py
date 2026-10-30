"""Point-annotation simulation.

A fully labelled crop is turned into a sparse mask that keeps the original class
id at exactly ``num_points`` pixel locations and sets every other pixel to the
ignore index (255). No-data pixels are never sampled.

Two strategies are provided:

* ``sample_random_points`` -- uniform over all valid pixels. This follows the
  natural class frequency of the crop.
* ``sample_class_balanced_points`` -- round-robin over the classes present in
  the crop, so rare classes receive the same number of points as common ones.
"""

from __future__ import annotations

import logging

import numpy as np

from pointseg.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def _validate(mask: np.ndarray, num_points: int, ignore_index: int) -> None:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"mask must be integer, got {mask.dtype}")
    if num_points < 0:
        raise ValueError("num_points must be >= 0")


def _empty_sparse(mask: np.ndarray, ignore_index: int) -> np.ndarray:
    return np.full(mask.shape, ignore_index, dtype=mask.dtype)


def _place(sparse: np.ndarray, mask: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    rows, cols = np.unravel_index(flat_indices, mask.shape)
    sparse[rows, cols] = mask[rows, cols]
    return sparse


def _warn_shortfall(requested: int, available: int) -> None:
    logger.warning(
        "requested %d points but the crop only has %d valid pixels; using all of them",
        requested,
        available,
    )


def sample_random_points(
    mask: np.ndarray,
    num_points: int,
    ignore_index: int = IGNORE_INDEX,
    seed: int | None = None,
) -> np.ndarray:
    """Uniformly sample ``num_points`` valid pixels from a fully labelled crop.

    Args:
        mask: 2-D integer array with class ids and ``ignore_index`` for no-data.
        num_points: annotation budget for this crop.
        ignore_index: value that must never be sampled or supervised.
        seed: optional seed for a private ``numpy.random.Generator``.

    Returns:
        A sparse mask of the same shape and dtype. Only sampled locations keep
        their class id; everything else is ``ignore_index``.
    """
    _validate(mask, num_points, ignore_index)
    sparse = _empty_sparse(mask, ignore_index)
    valid = np.flatnonzero(mask.reshape(-1) != ignore_index)
    if valid.size == 0 or num_points == 0:
        if num_points:
            _warn_shortfall(num_points, 0)
        return sparse

    rng = np.random.default_rng(seed)
    if num_points >= valid.size:
        _warn_shortfall(num_points, valid.size)
        chosen = valid
    else:
        chosen = rng.choice(valid, size=num_points, replace=False)
    return _place(sparse, mask, chosen)


def sample_class_balanced_points(
    mask: np.ndarray,
    num_points: int,
    ignore_index: int = IGNORE_INDEX,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points round-robin across the classes present in the crop.

    Classes with fewer pixels than their share contribute all their pixels and
    the leftover budget is redistributed to classes that still have pixels
    available. The total is therefore exactly ``min(num_points, valid_pixels)``.
    """
    _validate(mask, num_points, ignore_index)
    sparse = _empty_sparse(mask, ignore_index)
    flat = mask.reshape(-1)
    valid = np.flatnonzero(flat != ignore_index)
    if valid.size == 0 or num_points == 0:
        if num_points:
            _warn_shortfall(num_points, 0)
        return sparse
    if num_points >= valid.size:
        _warn_shortfall(num_points, valid.size)
        return _place(sparse, mask, valid)

    rng = np.random.default_rng(seed)
    classes = np.unique(flat[valid])
    # Random class order avoids systematically favouring low class ids when the
    # budget is smaller than the number of classes.
    rng.shuffle(classes)
    pools = {int(c): rng.permutation(valid[flat[valid] == c]) for c in classes}
    pointers = {int(c): 0 for c in classes}

    chosen: list[int] = []
    active = [int(c) for c in classes]
    while len(chosen) < num_points and active:
        for c in list(active):
            if len(chosen) >= num_points:
                break
            p = pointers[c]
            if p < len(pools[c]):
                chosen.append(int(pools[c][p]))
                pointers[c] = p + 1
            if pointers[c] >= len(pools[c]):
                active.remove(c)

    return _place(sparse, mask, np.asarray(chosen, dtype=np.int64))


SAMPLERS = {
    "random": sample_random_points,
    "class_balanced": sample_class_balanced_points,
}


def sample_points(
    mask: np.ndarray,
    num_points: int,
    strategy: str = "random",
    ignore_index: int = IGNORE_INDEX,
    seed: int | None = None,
) -> np.ndarray:
    """Dispatch to the requested sampling strategy."""
    try:
        sampler = SAMPLERS[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown sampling strategy {strategy!r}") from exc
    return sampler(mask, num_points, ignore_index=ignore_index, seed=seed)
