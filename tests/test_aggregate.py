from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointseg.experiments.aggregate import (
    _key,
    _mean_std,
    aggregate,
    collect,
    is_experiment_run,
)


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    points: int,
    strategy: str,
    miou: float,
    gamma: float = 2.0,
) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": name,
        "best_miou": miou,
        "best_epoch": 3,
        "final_miou": miou - 0.01,
        "best_metrics": {
            "mean_f1": miou - 0.05,
            "pixel_accuracy": miou + 0.2,
            "per_class_iou": [miou] * 7,
        },
        "config": {
            "seed": seed,
            "loss": {"name": "partial_focal", "gamma": gamma},
            "point_sampling": {"strategy": strategy, "points_per_crop": points},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_key_encodes_loss_strategy_and_budget():
    config = {
        "loss": {"name": "partial_focal", "gamma": 2.0},
        "point_sampling": {"strategy": "random", "points_per_crop": 100},
    }
    assert _key(config) == "partial_focal_gamma2.0_random_100pts"


def test_mean_std_handles_single_and_empty():
    assert _mean_std([0.5])["std"] == 0.0
    assert _mean_std([0.5])["n"] == 1
    empty = _mean_std([])
    assert empty["mean"] is None and empty["n"] == 0
    stats = _mean_std([0.2, 0.4, 0.6])
    assert stats["mean"] == pytest.approx(0.4)
    assert stats["std"] == pytest.approx(0.2)


def test_aggregate_groups_seeds(tmp_path):
    _write_run(
        tmp_path / "exp_density", "a_seed1", seed=1, points=100, strategy="random", miou=0.30
    )
    _write_run(
        tmp_path / "exp_density", "a_seed2", seed=2, points=100, strategy="random", miou=0.40
    )
    _write_run(tmp_path / "exp_density", "b_seed1", seed=1, points=10, strategy="random", miou=0.10)

    groups = collect(tmp_path)
    assert set(groups) == {
        "partial_focal_gamma2.0_random_100pts",
        "partial_focal_gamma2.0_random_10pts",
    }

    result = aggregate(tmp_path)
    stats = result["partial_focal_gamma2.0_random_100pts"]["best_miou"]
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.35)
    assert stats["std"] == pytest.approx(0.0707106781, rel=1e-6)
    np.testing.assert_allclose(
        result["partial_focal_gamma2.0_random_100pts"]["per_class_iou_mean"],
        [0.35] * 7,
    )


def test_aggregate_ignores_unrelated_json(tmp_path):
    (tmp_path / "random.json").write_text('{"unrelated": true}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert aggregate(tmp_path) == {}


def test_adhoc_runs_are_excluded_but_can_be_included(tmp_path):
    _write_run(tmp_path / "exp_density", "real", seed=1, points=100, strategy="random", miou=0.30)
    _write_run(tmp_path / "bench", "scratch", seed=1, points=100, strategy="random", miou=0.05)

    default = aggregate(tmp_path)
    assert list(default) == ["partial_focal_gamma2.0_random_100pts"]
    assert default["partial_focal_gamma2.0_random_100pts"]["best_miou"]["mean"] == pytest.approx(
        0.30
    )

    everything = aggregate(tmp_path, include_all=True)
    assert everything["partial_focal_gamma2.0_random_100pts"]["best_miou"]["n"] == 2


def test_is_experiment_run():
    root = Path("outputs")
    assert is_experiment_run(root / "exp_density" / "run" / "metrics.json", root)
    assert not is_experiment_run(root / "bench" / "run" / "metrics.json", root)
    assert not is_experiment_run(root / "metrics.json", root)
