from __future__ import annotations

import numpy as np
import pytest

from pointseg.constants import IGNORE_INDEX
from pointseg.datasets.point_sampling import (
    sample_class_balanced_points,
    sample_points,
    sample_random_points,
)


def _mask_with_classes(per_class: int = 100, ignore: int = 100) -> np.ndarray:
    values = np.concatenate(
        [np.full(per_class, c, dtype=np.uint8) for c in range(7)]
        + [np.full(ignore, IGNORE_INDEX, dtype=np.uint8)]
    )
    rng = np.random.default_rng(0)
    rng.shuffle(values)
    return values.reshape(20, 40)


@pytest.mark.parametrize("strategy", ["random", "class_balanced"])
def test_exact_budget_and_valid_values(strategy):
    mask = _mask_with_classes()
    sparse = sample_points(mask, 100, strategy=strategy, seed=1)
    labelled = sparse != IGNORE_INDEX
    assert labelled.sum() == 100
    assert set(np.unique(sparse[labelled])) <= set(range(7))
    # Labels are copied, never invented.
    np.testing.assert_array_equal(sparse[labelled], mask[labelled])


def test_never_samples_no_data():
    mask = np.full((50, 50), IGNORE_INDEX, dtype=np.uint8)
    mask[:10, :10] = 3
    sparse = sample_random_points(mask, 10_000, seed=0)
    assert np.count_nonzero(sparse != IGNORE_INDEX) == 100
    assert set(np.unique(sparse[sparse != IGNORE_INDEX])) == {3}


def test_shortfall_returns_all_valid_pixels():
    mask = np.full((10, 10), IGNORE_INDEX, dtype=np.uint8)
    mask[0, :5] = 2
    sparse = sample_random_points(mask, 50, seed=0)
    assert np.count_nonzero(sparse != IGNORE_INDEX) == 5


def test_deterministic_with_seed():
    mask = _mask_with_classes()
    first = sample_random_points(mask, 50, seed=123)
    second = sample_random_points(mask, 50, seed=123)
    third = sample_random_points(mask, 50, seed=124)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_class_balanced_is_balanced():
    mask = _mask_with_classes(per_class=100)
    sparse = sample_class_balanced_points(mask, 35, seed=0)
    counts = [int(np.count_nonzero(sparse == c)) for c in range(7)]
    assert sum(counts) == 35
    assert max(counts) - min(counts) <= 1


def test_class_balanced_respects_small_classes():
    mask = np.full((20, 20), IGNORE_INDEX, dtype=np.uint8)
    mask[0, :2] = 0
    mask[1, :20] = 1
    sparse = sample_class_balanced_points(mask, 10, seed=0)
    assert np.count_nonzero(sparse == 0) == 2
    assert np.count_nonzero(sparse == 1) == 8


def test_all_ignored_returns_all_ignored():
    mask = np.full((5, 5), IGNORE_INDEX, dtype=np.uint8)
    sparse = sample_random_points(mask, 5, seed=0)
    assert np.all(sparse == IGNORE_INDEX)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        sample_points(np.zeros((4, 4), dtype=np.uint8), 1, strategy="nope")
