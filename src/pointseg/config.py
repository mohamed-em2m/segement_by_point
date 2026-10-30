"""Typed, validated configuration loading.

The resolved configuration is written next to every run so that experiments are
reproducible. Command-line overrides use dotted keys, e.g.::

    pointseg-train --config configs/default.yaml \
        --set point_sampling.points_per_crop=500 --set training.epochs=10
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pointseg.constants import NUM_CLASSES


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass
class DatasetConfig:
    name: str = "loveda"
    root: str = "./data/LoveDA"
    crop_size: int = 512
    domains: list[str] = field(default_factory=lambda: ["Rural", "Urban"])
    max_train_images: int | None = None
    max_val_images: int | None = None
    subset_seed: int = 42
    augment: bool = True
    rotations: bool = True


@dataclass
class ModelConfig:
    architecture: str = "unet"
    encoder: str = "resnet34"
    pretrained: bool = True
    num_classes: int = NUM_CLASSES
    in_channels: int = 3


@dataclass
class LossConfig:
    name: str = "partial_focal"
    gamma: float = 2.0
    alpha: list[float] | None = None
    strict: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    amp: bool = True
    grad_clip: float | None = None
    log_interval: int = 20
    max_train_batches: int | None = None


@dataclass
class PointSamplingConfig:
    strategy: str = "random"
    points_per_crop: int = 100
    ignore_index: int = 255
    fixed_points: bool = True


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["miou", "f1", "pixel_accuracy"])
    tile_size: int = 512
    tile_overlap: float = 0.25
    batch_size: int = 2
    every_n_epochs: int = 1


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "./outputs"
    run_name: str = "baseline"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    point_sampling: PointSamplingConfig = field(default_factory=PointSamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def copy(self) -> Config:
        return copy.deepcopy(self)

    def validate(self) -> None:
        if self.dataset.name != "loveda":
            raise ConfigError(f"unsupported dataset: {self.dataset.name!r}")
        if self.dataset.crop_size <= 0:
            raise ConfigError("dataset.crop_size must be positive")
        if self.dataset.max_train_images is not None and self.dataset.max_train_images <= 0:
            raise ConfigError("dataset.max_train_images must be positive or null")
        if self.dataset.max_val_images is not None and self.dataset.max_val_images <= 0:
            raise ConfigError("dataset.max_val_images must be positive or null")
        allowed_domains = {"Rural", "Urban"}
        unknown = set(self.dataset.domains) - allowed_domains
        if unknown:
            raise ConfigError(f"unknown domains: {sorted(unknown)}")

        if self.model.num_classes != NUM_CLASSES:
            raise ConfigError(
                f"model.num_classes must be {NUM_CLASSES} for LoveDA, got {self.model.num_classes}"
            )
        if self.model.in_channels != 3:
            raise ConfigError("model.in_channels must be 3")

        if self.loss.name not in {"partial_focal", "partial_ce"}:
            raise ConfigError("loss.name must be 'partial_focal' or 'partial_ce'")
        if self.loss.gamma < 0:
            raise ConfigError("loss.gamma must be >= 0")
        if self.loss.alpha is not None and len(self.loss.alpha) != self.model.num_classes:
            raise ConfigError("loss.alpha must have one value per class")

        if self.training.epochs < 1:
            raise ConfigError("training.epochs must be >= 1")
        if self.training.batch_size < 1:
            raise ConfigError("training.batch_size must be >= 1")
        if self.training.optimizer not in {"adamw", "adam", "sgd"}:
            raise ConfigError("training.optimizer must be adamw, adam or sgd")
        if self.training.scheduler not in {"cosine", "none", "step"}:
            raise ConfigError("training.scheduler must be cosine, step or none")

        if self.point_sampling.strategy not in {"random", "class_balanced"}:
            raise ConfigError("point_sampling.strategy must be random or class_balanced")
        if self.point_sampling.points_per_crop < 1:
            raise ConfigError("point_sampling.points_per_crop must be >= 1")
        if self.point_sampling.ignore_index != 255:
            raise ConfigError("point_sampling.ignore_index must be 255")

        if not 0.0 <= self.evaluation.tile_overlap < 1.0:
            raise ConfigError("evaluation.tile_overlap must be in [0, 1)")
        if self.evaluation.tile_size <= 0:
            raise ConfigError("evaluation.tile_size must be positive")
        if self.evaluation.every_n_epochs < 1:
            raise ConfigError("evaluation.every_n_epochs must be >= 1")


def _set_nested(node: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse ``["a.b=1", "c=x"]`` into a nested dictionary of overrides.

    Unknown keys are rejected later by :func:`config_from_dict`.
    """
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"empty override key in {item!r}")
        _set_nested(parsed, key, yaml.safe_load(raw))
    return parsed


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_from_dict(data: dict[str, Any]) -> Config:
    known = set(Config.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")

    sections: dict[str, Any] = {}
    section_types = {
        "dataset": DatasetConfig,
        "model": ModelConfig,
        "loss": LossConfig,
        "training": TrainingConfig,
        "point_sampling": PointSamplingConfig,
        "evaluation": EvaluationConfig,
    }
    for name, cls in section_types.items():
        raw = data.get(name, {}) or {}
        bad = set(raw) - set(cls.__dataclass_fields__)
        if bad:
            raise ConfigError(f"unknown keys in {name}: {sorted(bad)}")
        sections[name] = cls(**raw)

    scalars = {k: v for k, v in data.items() if k not in section_types}
    config = Config(**scalars, **sections)
    config.validate()
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    merged = _merge(raw, parse_overrides(overrides))
    return config_from_dict(merged)


def load_config_dict(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Alias kept for readability in scripts."""
    return load_config(path, overrides)
