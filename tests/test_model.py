from __future__ import annotations

import torch

from pointseg.config import ModelConfig
from pointseg.models.unet import build_model, count_parameters, describe_model


def test_unet_output_shape():
    config = ModelConfig(pretrained=False)
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7, 64, 64)
    assert count_parameters(model) > 1_000_000
    assert "parameters" in describe_model(model)


def test_non_pretrained_encoder_is_supported():
    model = build_model(ModelConfig(pretrained=False, encoder="resnet18"))
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7, 64, 64)
