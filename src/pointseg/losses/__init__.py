from __future__ import annotations

from pointseg.config import LossConfig
from pointseg.losses.partial_ce import PartialCrossEntropyLoss
from pointseg.losses.partial_focal import PartialFocalLoss


def build_loss(config: LossConfig):
    if config.name == "partial_focal":
        return PartialFocalLoss(
            gamma=config.gamma,
            alpha=config.alpha,
            strict=config.strict,
        )
    if config.name == "partial_ce":
        return PartialCrossEntropyLoss(strict=config.strict)
    raise ValueError(f"unknown loss: {config.name!r}")


__all__ = ["PartialCrossEntropyLoss", "PartialFocalLoss", "build_loss"]
