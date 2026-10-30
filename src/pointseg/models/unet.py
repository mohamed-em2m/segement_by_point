"""U-Net with a ResNet-34 encoder, returning raw logits."""

from __future__ import annotations

import logging
from typing import Any

import segmentation_models_pytorch as smp
from torch import nn

from pointseg.config import ModelConfig
from pointseg.constants import NUM_CLASSES

logger = logging.getLogger(__name__)


def build_model(config: ModelConfig) -> nn.Module:
    """Build the segmentation network described by the configuration.

    ImageNet weights are downloaded by segmentation-models-pytorch on first use.
    If the download fails, we fall back to random initialisation with a loud
    warning rather than silently pretending the encoder was pretrained.
    """
    if config.architecture != "unet":
        raise ValueError(f"unsupported architecture: {config.architecture!r}")

    kwargs: dict[str, Any] = {
        "encoder_name": config.encoder,
        "encoder_weights": "imagenet" if config.pretrained else None,
        "in_channels": config.in_channels,
        "classes": config.num_classes,
        "activation": None,
    }
    try:
        model = smp.Unet(**kwargs)
    except Exception as exc:  # pragma: no cover - network dependent
        if not config.pretrained:
            raise
        logger.warning(
            "could not load ImageNet encoder weights (%s); continuing with random "
            "initialisation. Set model.pretrained=false to silence this.",
            exc,
        )
        kwargs["encoder_weights"] = None
        model = smp.Unet(**kwargs)
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    return sum(p.numel() for p in params if p.requires_grad or not trainable_only)


def describe_model(model: nn.Module) -> str:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return (
        f"{model.__class__.__name__}: {total / 1e6:.2f}M parameters "
        f"({trainable / 1e6:.2f}M trainable), {NUM_CLASSES} output classes"
    )
