from __future__ import annotations

import pytest

from pointseg.config import ConfigError, load_config


def test_load_default():
    config = load_config("configs/default.yaml")
    assert config.model.num_classes == 7
    assert config.point_sampling.points_per_crop == 100
    assert config.point_sampling.fixed_points is True
    assert config.evaluation.tile_overlap == 0.25


def test_overrides_are_applied():
    config = load_config(
        "configs/default.yaml",
        ["point_sampling.points_per_crop=500", "training.epochs=3"],
    )
    assert config.point_sampling.points_per_crop == 500
    assert config.training.epochs == 3


def test_unknown_override_is_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", ["point_sampling.nope=1"])


def test_invalid_strategy_is_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", ["point_sampling.strategy=magic"])


def test_invalid_overlap_is_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/default.yaml", ["evaluation.tile_overlap=1.0"])


def test_save_roundtrip(tmp_path):
    config = load_config("configs/default.yaml")
    path = tmp_path / "resolved.yaml"
    config.save(path)
    loaded = load_config(path)
    assert loaded.to_dict() == config.to_dict()
