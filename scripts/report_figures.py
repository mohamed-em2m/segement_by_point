#!/usr/bin/env python
"""Build the report figures from real experiment outputs.

Reads ``outputs/aggregate.json`` (written by ``pointseg-aggregate``) and writes
PNG figures into ``report/figures/``:

* ``density_curve.png``   -- mIoU / mean F1 versus annotation budget
* ``density_per_class.png`` -- per-class IoU for every budget
* ``sampling_comparison.png`` -- random vs class-balanced at 100 points
* ``sampling_per_class.png``  -- per-class IoU for the two strategies

Every value comes from the saved metrics; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pointseg.constants import CLASS_NAMES  # noqa: E402

DENSITY_COLOR = "#1f77b4"
F1_COLOR = "#ff7f0e"
STRATEGY_COLORS = {"random": "#1f77b4", "class_balanced": "#d62728"}


def _points(key: str) -> int:
    return int(key.rsplit("_", 1)[-1].removesuffix("pts"))


def _strategy(key: str) -> str:
    return "class_balanced" if "_class_balanced_" in key else "random"


def _density_entries(data: dict) -> list[tuple[int, dict]]:
    entries = [
        (_points(key), value)
        for key, value in data.items()
        if key.startswith("partial_focal") and "_random_" in key
    ]
    return sorted(entries, key=lambda item: item[0])


def plot_density_curve(data: dict, out: Path) -> Path | None:
    entries = _density_entries(data)
    if not entries:
        return None
    points = [p for p, _ in entries]
    miou = [v["best_miou"]["mean"] for _, v in entries]
    f1 = [v["mean_f1"]["mean"] for _, v in entries]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(points, miou, marker="o", color=DENSITY_COLOR, label="mIoU")
    ax.plot(points, f1, marker="s", color=F1_COLOR, label="mean F1")
    ax.set_xscale("log")
    ax.set_xticks(points)
    ax.set_xticklabels([str(p) for p in points])
    ax.set_xlabel("labelled points per training crop")
    ax.set_ylabel("validation score (full masks)")
    ax.set_ylim(0, max(max(miou), max(f1)) * 1.18)
    for x, y in zip(points, miou, strict=True):
        ax.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8
        )
    ax.set_title("Experiment A: annotation density (random sampling)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_density_per_class(data: dict, out: Path) -> Path | None:
    entries = _density_entries(data)
    if not entries:
        return None
    matrix = np.array([v["per_class_iou_mean"] for _, v in entries], dtype=float)
    labels = [f"{p} pts" for p, _ in entries]

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(0.4, np.nanmax(matrix)))
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.28 else "black",
                fontsize=8,
            )
    ax.set_title("Per-class IoU versus annotation budget")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="IoU")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_sampling_comparison(data: dict, out: Path) -> Path | None:
    entries = {
        _strategy(key): value
        for key, value in data.items()
        if key.startswith("partial_focal") and "_100pts" in key
    }
    if len(entries) < 2:
        return None
    strategies = ["random", "class_balanced"]
    labels = ["random", "class-balanced"]
    miou = [entries[s]["best_miou"]["mean"] for s in strategies]
    f1 = [entries[s]["mean_f1"]["mean"] for s in strategies]
    acc = [entries[s]["pixel_accuracy"]["mean"] for s in strategies]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(strategies))
    width = 0.26
    ax.bar(x - width, miou, width, label="mIoU", color="#1f77b4")
    ax.bar(x, f1, width, label="mean F1", color="#ff7f0e")
    ax.bar(x + width, acc, width, label="pixel accuracy", color="#2ca02c")
    for offset, values in ((-width, miou), (0, f1), (width, acc)):
        for xi, value in zip(x + offset, values, strict=True):
            ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("validation score (full masks)")
    ax.set_title("Experiment B: sampling strategy at 100 points per crop")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_sampling_per_class(data: dict, out: Path) -> Path | None:
    entries = {
        _strategy(key): value
        for key, value in data.items()
        if key.startswith("partial_focal") and "_100pts" in key
    }
    if len(entries) < 2:
        return None
    strategies = ["random", "class_balanced"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for index, strategy in enumerate(strategies):
        values = np.array(entries[strategy]["per_class_iou_mean"], dtype=float)
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=strategy.replace("_", "-"),
            color=STRATEGY_COLORS[strategy],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU: random vs class-balanced sampling (100 points)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_point_sampling(
    dataset_root: str,
    out: Path,
    points: int = 100,
    seed: int = 42,
) -> Path | None:
    """Illustrate random vs class-balanced point labels on a real crop."""
    import cv2

    from pointseg.constants import IGNORE_INDEX
    from pointseg.datasets.loveda import discover_pairs
    from pointseg.datasets.point_sampling import (
        sample_class_balanced_points,
        sample_random_points,
    )
    from pointseg.datasets.transforms import remap_mask
    from pointseg.evaluation.visualize import colorize, points_overlay

    try:
        pairs = discover_pairs(dataset_root, "Train")
    except (FileNotFoundError, ValueError):
        return None

    best: tuple[int, np.ndarray, np.ndarray, str] | None = None
    for pair in pairs[:25]:
        image = cv2.cvtColor(cv2.imread(str(pair["image"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = remap_mask(cv2.imread(str(pair["mask"]), cv2.IMREAD_UNCHANGED))
        crop = mask[256:768, 256:768]
        n_classes = len(np.unique(crop[crop != IGNORE_INDEX]))
        if best is None or n_classes > best[0]:
            best = (n_classes, image[256:768, 256:768], crop, str(pair["id"]))
    if best is None:
        return None
    n_classes, image_crop, mask_crop, sample_id = best

    random_points = sample_random_points(mask_crop, points, seed=seed)
    balanced_points = sample_class_balanced_points(mask_crop, points, seed=seed)
    dimmed = (image_crop.astype(np.float32) * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(image_crop)
    axes[0].set_title("image crop")
    axes[1].imshow(colorize(mask_crop))
    axes[1].set_title(f"full mask ({n_classes} classes)")
    axes[2].imshow(points_overlay(random_points, base=dimmed))
    axes[2].set_title(f"{points} random points")
    axes[3].imshow(points_overlay(balanced_points, base=dimmed))
    axes[3].set_title(f"{points} class-balanced points")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Point annotation simulation ({sample_id})", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--dataset-root", default="data/LoveDA")
    parser.add_argument("--output", default="report/figures")
    args = parser.parse_args()

    aggregate_path = Path(args.root) / "aggregate.json"
    if not aggregate_path.is_file():
        raise SystemExit(
            f"{aggregate_path} not found; run `pointseg-aggregate --root {args.root}` first"
        )
    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    out = Path(args.output)
    written = [
        plot_density_curve(data, out / "density_curve.png"),
        plot_density_per_class(data, out / "density_per_class.png"),
        plot_sampling_comparison(data, out / "sampling_comparison.png"),
        plot_sampling_per_class(data, out / "sampling_per_class.png"),
        plot_point_sampling(args.dataset_root, out / "point_sampling.png"),
    ]
    written = [path for path in written if path is not None]
    if not written:
        raise SystemExit("no experiment results found in aggregate.json")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
