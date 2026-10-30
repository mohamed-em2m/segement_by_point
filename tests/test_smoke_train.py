from __future__ import annotations

import importlib
import json

import torch
from torch import nn

from pointseg.config import load_config
from pointseg.training.train import train


class TinyModel(nn.Module):
    """1x1 convolution stand-in so the smoke test does not need a full U-Net."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_training_smoke(tmp_path, monkeypatch, synthetic_loveda):
    train_module = importlib.import_module("pointseg.training.train")
    monkeypatch.setattr(train_module, "build_model", lambda config: TinyModel(config.num_classes))
    config = load_config("configs/smoke.yaml")
    config.dataset.root = str(synthetic_loveda)
    config.dataset.crop_size = 32
    config.dataset.max_train_images = 4
    config.dataset.max_val_images = 2
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.max_train_batches = 2
    config.training.amp = False
    config.evaluation.tile_size = 32
    config.evaluation.tile_overlap = 0.0
    config.evaluation.batch_size = 1
    config.output_dir = str(tmp_path / "outputs")
    config.run_name = "smoke"

    summary = train(config, device=torch.device("cpu"))
    run_dir = tmp_path / "outputs" / "smoke"

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert summary["train_images"] == 4
    assert 0.0 <= summary["best_miou"] <= 1.0

    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["run_name"] == "smoke"
    assert len(saved["history"]) == 1
