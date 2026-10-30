"""Figures for training curves, per-class metrics and qualitative predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pointseg.config import load_config  # noqa: E402
from pointseg.constants import CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX  # noqa: E402
from pointseg.datasets.loveda import LoveDAEvalDataset  # noqa: E402
from pointseg.datasets.point_sampling import sample_random_points  # noqa: E402
from pointseg.datasets.transforms import denormalize_image  # noqa: E402
from pointseg.evaluation.validate import predict_logits  # noqa: E402
from pointseg.models.unet import build_model  # noqa: E402


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map class ids to RGB; ignore-index pixels become light grey."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        out[mask == class_id] = color
    out[mask == IGNORE_INDEX] = (220, 220, 220)
    return out


def points_overlay(
    mask: np.ndarray,
    base: np.ndarray | None = None,
    radius: int = 3,
) -> np.ndarray:
    """Draw sparse points enlarged so they are visible in a figure.

    The mask itself is never modified; this is display-only dilation.
    """
    import cv2

    if base is None:
        overlay = np.full((*mask.shape, 3), 235, dtype=np.uint8)
    else:
        overlay = base.astype(np.uint8).copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        points = (mask == class_id).astype(np.uint8)
        if points.any():
            overlay[cv2.dilate(points, kernel).astype(bool)] = color
    return overlay


def _read_history(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"history.csv not found in {run_dir}")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _series(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value not in (None, ""):
            xs.append(int(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def plot_history(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rows = _read_history(run_dir)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    loss_x, loss_y = _series(rows, "train_loss")
    if loss_x:
        axes[0].plot(loss_x, loss_y, marker="o", color="#1f77b4")
        axes[0].set_title("Training loss (partial focal)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
    miou_x, miou_y = _series(rows, "val_miou")
    if miou_x:
        axes[1].plot(miou_x, miou_y, marker="o", color="#2ca02c", label="mIoU")
        f1_x, f1_y = _series(rows, "val_mean_f1")
        axes[1].plot(f1_x, f1_y, marker="s", color="#ff7f0e", label="mean F1")
        axes[1].set_title("Validation (full masks)")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = figures / "history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_class(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data.get("best_metrics") or data.get("final_metrics") or {}
    iou = metrics.get("per_class_iou") or []
    f1 = metrics.get("per_class_f1") or []
    names = CLASS_NAMES[: len(iou)]

    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2, iou, width, label="IoU", color="#1f77b4")
    ax.bar(x + width / 2, f1, width, label="F1", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class metrics - {data.get('run_name', run_dir.name)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = figures / "per_class.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@torch.no_grad()
def plot_predictions(
    config,
    checkpoint: str | Path,
    run_dir: str | Path,
    num_images: int = 3,
    points: int = 100,
) -> Path:
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload))
    model.eval()

    dataset = LoveDAEvalDataset(
        root=config.dataset.root,
        domains=config.dataset.domains,
        max_images=max(num_images, 1),
        subset_seed=config.dataset.subset_seed,
    )
    rng = np.random.default_rng(config.seed)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_images, 5, figsize=(18, 3.6 * num_images), squeeze=False)
    for row in range(num_images):
        item = dataset[row]
        image = item["image"]
        target = item["mask"].numpy()
        sparse = sample_random_points(target, points, seed=int(rng.integers(0, 2**31 - 1)))
        logits = predict_logits(
            model,
            image,
            tile_size=config.evaluation.tile_size,
            overlap=config.evaluation.tile_overlap,
            device=device,
            amp=config.training.amp,
            num_classes=config.model.num_classes,
        )
        prediction = logits.argmax(0).numpy()

        image_rgb = denormalize_image(image.numpy())
        panels = [
            (image_rgb, "image", None),
            (colorize(target), "full ground truth", None),
            (
                points_overlay(sparse, base=(image_rgb * 0.55).astype(np.uint8)),
                f"{points} point labels",
                None,
            ),
            (colorize(prediction), "prediction", None),
            (colorize(target), "prediction errors", prediction != target),
        ]
        for col, (panel, title, error) in enumerate(panels):
            axes[row, col].imshow(panel)
            if error is not None:
                axes[row, col].imshow(np.where(error, 255, 0), cmap="Reds", alpha=0.55)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")
    fig.tight_layout()
    out = figures / "predictions.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate figures for a run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None, help="needed for prediction figures")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-images", type=int, default=3)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    outputs = [plot_history(run_dir), plot_per_class(run_dir)]
    if args.config and args.checkpoint:
        config = load_config(args.config)
        outputs.append(
            plot_predictions(config, args.checkpoint, run_dir, num_images=args.num_images)
        )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
