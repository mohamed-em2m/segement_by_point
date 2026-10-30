"""Deterministic validation with tiled inference and logit averaging."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pointseg.config import Config, load_config
from pointseg.constants import CLASS_NAMES, NUM_CLASSES
from pointseg.datasets.loveda import build_eval_dataset
from pointseg.evaluation.metrics import ConfusionMatrix
from pointseg.utils import get_logger, write_json

logger = get_logger(__name__)


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets covering ``length`` with at least the requested overlap."""
    if tile >= length:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _forward_tiles(
    model: nn.Module,
    tiles: torch.Tensor,
    device: torch.device,
    amp: bool,
    chunk_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, tiles.shape[0], chunk_size):
        chunk = tiles[start : start + chunk_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs.append(model(chunk).float())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: float = 0.25,
    device: torch.device | None = None,
    amp: bool = True,
    chunk_size: int = 4,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Predict full-resolution logits for one normalized ``[3, H, W]`` image."""
    if image.ndim != 3:
        raise ValueError(f"image must be [3, H, W], got {tuple(image.shape)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, height, width = image.shape
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if tile_size >= height and tile_size >= width:
        logits = _forward_tiles(model, image.unsqueeze(0), device, amp, 1)
        return logits[0].cpu()

    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    tiles = []
    positions = []
    for top in ys:
        for left in xs:
            tiles.append(image[:, top : top + tile_size, left : left + tile_size])
            positions.append((top, left))
    batch = torch.stack(tiles, dim=0)
    outputs = _forward_tiles(model, batch, device, amp, chunk_size)

    if outputs.shape[1] != num_classes:
        raise ValueError(f"model produced {outputs.shape[1]} classes but num_classes={num_classes}")
    logit_sum = torch.zeros((num_classes, height, width), dtype=torch.float32)
    counts = torch.zeros((1, height, width), dtype=torch.float32)
    for logits, (top, left) in zip(outputs, positions, strict=True):
        logit_sum[:, top : top + tile_size, left : left + tile_size] += logits.cpu()
        counts[:, top : top + tile_size, left : left + tile_size] += 1.0
    return logit_sum / counts.clamp_min(1.0)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    config: Config,
    device: torch.device | None = None,
) -> dict:
    """Run deterministic validation and return split-level metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cm = ConfusionMatrix(config.model.num_classes, config.point_sampling.ignore_index)
    num_images = 0
    for batch in loader:
        images = batch["image"]
        masks = batch["mask"]
        for index in range(images.shape[0]):
            logits = predict_logits(
                model,
                images[index],
                tile_size=config.evaluation.tile_size,
                overlap=config.evaluation.tile_overlap,
                device=device,
                amp=config.training.amp,
                num_classes=config.model.num_classes,
            )
            prediction = logits.argmax(dim=0)
            cm.update(prediction, masks[index])
            num_images += 1
    metrics = cm.compute()
    metrics["num_images"] = num_images
    return metrics


def format_metrics(metrics: dict) -> str:
    lines = [
        f"mIoU: {metrics['miou']:.4f} | mean F1: {metrics['mean_f1']:.4f} "
        f"| pixel acc: {metrics['pixel_accuracy']:.4f} | images: {metrics.get('num_images', '?')}",
        "  class          IoU      F1",
    ]
    for name, iou, f1 in zip(
        CLASS_NAMES, metrics["per_class_iou"], metrics["per_class_f1"], strict=False
    ):
        iou_text = "   n/a" if np.isnan(iou) else f"{iou:.4f}"
        f1_text = "   n/a" if np.isnan(f1) else f"{f1:.4f}"
        lines.append(f"  {name:<12} {iou_text:>7} {f1_text:>7}")
    return "\n".join(lines)


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on LoveDA validation")
    parser.add_argument("--config", required=True, help="resolved training config")
    parser.add_argument("--checkpoint", required=True, help="path to best.pt/last.pt")
    parser.add_argument("--output", default=None, help="where to write metrics.json")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="dotted config override, e.g. dataset.max_val_images=50",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pointseg.models.unet import build_model

    model = build_model(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    logger.info("loaded checkpoint %s", args.checkpoint)

    loader = torch.utils.data.DataLoader(
        build_eval_dataset(config),
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = validate(model, loader, config, device)
    print(format_metrics(metrics))

    output = (
        Path(args.output) if args.output else Path(args.checkpoint).parent / "eval_metrics.json"
    )
    write_json(output, metrics)
    logger.info("wrote %s", output)
    return 0


# Alias used by the ``pointseg-eval`` console script.
main = evaluate_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(evaluate_main())
