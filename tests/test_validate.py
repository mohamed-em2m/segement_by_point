from __future__ import annotations

import numpy as np
import torch

from pointseg.evaluation.validate import predict_logits, tile_starts


def test_tile_starts_cover_image():
    starts = tile_starts(1024, 512, 0.25)
    assert starts[0] == 0
    assert starts[-1] == 512
    assert all(b - a <= 512 for a, b in zip(starts, starts[1:], strict=False))


def test_tile_starts_single_tile():
    assert tile_starts(256, 512, 0.25) == [0]


def test_tiled_prediction_matches_full_image():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x.repeat(1, 7, 1, 1)[:, :7]

    model = Identity()
    image = torch.randn(3, 64, 64)
    full = predict_logits(
        model, image, tile_size=128, overlap=0.0, device=torch.device("cpu"), amp=False
    )
    tiled = predict_logits(
        model, image, tile_size=32, overlap=0.5, device=torch.device("cpu"), amp=False
    )
    assert full.shape == (7, 64, 64)
    np.testing.assert_allclose(full.numpy(), tiled.numpy(), atol=1e-5)
