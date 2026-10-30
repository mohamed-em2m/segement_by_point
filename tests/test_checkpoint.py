from __future__ import annotations

import torch
from torch import nn

from pointseg.config import load_config
from pointseg.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    config = load_config("configs/default.yaml")
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        epoch=3,
        config=config,
        seed=7,
        best_miou=0.42,
        best_epoch=2,
        history=[{"epoch": 1, "train_loss": 1.0}],
    )

    restored = nn.Linear(4, 2)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, restored, optimizer=new_optimizer)
    assert payload["epoch"] == 3
    assert payload["best_miou"] == 0.42
    assert payload["seed"] == 7
    assert payload["config"]["model"]["encoder"] == "resnet34"
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)
