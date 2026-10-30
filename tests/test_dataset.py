from __future__ import annotations

import numpy as np
import pytest

from pointseg.constants import IGNORE_INDEX
from pointseg.datasets.loveda import (
    LoveDAEvalDataset,
    LoveDATrainDataset,
    check_data_main,
    discover_pairs,
)


def test_discover_pairs(synthetic_loveda):
    pairs = discover_pairs(synthetic_loveda, "Train")
    assert len(pairs) == 5
    assert {pair["domain"] for pair in pairs} == {"Rural", "Urban"}


def test_train_dataset_returns_sparse_mask_only(synthetic_loveda):
    dataset = LoveDATrainDataset(
        synthetic_loveda,
        crop_size=32,
        points_per_crop=10,
        seed=0,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (32, 32)
    labelled = item["mask"] != IGNORE_INDEX
    assert int(labelled.sum()) <= 10
    assert set(item["mask"][labelled].unique().tolist()) <= set(range(7))


def test_fixed_points_are_identical_across_epochs(synthetic_loveda):
    dataset = LoveDATrainDataset(
        synthetic_loveda,
        crop_size=32,
        points_per_crop=10,
        seed=0,
        fixed_points=True,
        augment_data=False,
    )
    first = dataset[0]["mask"]
    dataset.set_epoch(5)
    second = dataset[0]["mask"]
    assert first.shape == second.shape
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_resampled_points_change(synthetic_loveda):
    dataset = LoveDATrainDataset(
        synthetic_loveda,
        crop_size=32,
        points_per_crop=10,
        seed=0,
        fixed_points=False,
        augment_data=False,
    )
    first = dataset[0]["mask"].numpy()
    dataset.set_epoch(1)
    second = dataset[0]["mask"].numpy()
    assert not np.array_equal(first, second)


def test_eval_dataset_returns_full_mask(synthetic_loveda):
    dataset = LoveDAEvalDataset(synthetic_loveda)
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert set(item["mask"].unique().tolist()) <= set(range(7)) | {IGNORE_INDEX}


def test_check_data_cli(synthetic_loveda, capsys):
    assert check_data_main(["--root", str(synthetic_loveda)]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_data_missing_root(tmp_path, capsys):
    assert check_data_main(["--root", str(tmp_path / "missing")]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_mismatched_pairs_are_reported(synthetic_loveda):
    (synthetic_loveda / "Train" / "Rural" / "masks_png" / "10.png").unlink()
    with pytest.raises(ValueError, match="mismatch"):
        discover_pairs(synthetic_loveda, "Train")
