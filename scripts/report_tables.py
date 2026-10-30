#!/usr/bin/env python
"""Build markdown result tables from real ``metrics.json`` outputs.

Usage::

    python scripts/report_tables.py --root outputs --output report/results_tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(stat: dict) -> str:
    if stat.get("mean") is None:
        return "n/a"
    if stat.get("n", 0) > 1 and stat.get("std") is not None:
        return f"{stat['mean']:.4f} +/- {stat['std']:.4f}"
    return f"{stat['mean']:.4f}"


def _run_details(root: Path) -> list[str]:
    """Wall-clock and budget details for every experiment run."""
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        if not {"exp_density", "exp_sampling"} & set(path.parts):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        config = data.get("config") or {}
        points = (config.get("point_sampling") or {}).get("points_per_crop")
        strategy = (config.get("point_sampling") or {}).get("strategy")
        history = data.get("history") or []
        seconds = sum(float(row.get("epoch_seconds") or 0.0) for row in history)
        val_seconds = sum(
            float(row.get("epoch_seconds") or 0.0) for row in history if row.get("val_miou")
        )
        rows.append(
            {
                "run": data.get("run_name", path.parent.name),
                "strategy": strategy,
                "points": points,
                "epochs": len(history),
                "best_epoch": data.get("best_epoch"),
                "miou": data.get("best_miou"),
                "minutes": seconds / 60.0,
                "val_minutes": val_seconds / 60.0,
                "points_seen": history[0].get("mean_points_per_crop") if history else None,
            }
        )
    if not rows:
        return []
    rows.sort(key=lambda row: (str(row["strategy"]), int(row["points"] or 0)))
    lines = [
        "### Run details",
        "",
        "| Run | Strategy | Points | Mean points seen | Epochs | Best epoch | Best mIoU | "
        "Wall clock (min) | Validation (min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['strategy']} | {row['points']} | "
            f"{row['points_seen']:.1f} | {row['epochs']} | {row['best_epoch']} | "
            f"{row['miou']:.4f} | {row['minutes']:.1f} | {row['val_minutes']:.1f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="report/results_tables.md")
    args = parser.parse_args()

    aggregate_path = Path(args.root) / "aggregate.json"
    if not aggregate_path.is_file():
        raise SystemExit(
            f"{aggregate_path} not found; run `pointseg-aggregate --root {args.root}` first"
        )
    data = json.loads(aggregate_path.read_text())
    if not data:
        raise SystemExit("aggregate.json is empty")

    lines: list[str] = []

    density = {
        key: value
        for key, value in data.items()
        if "_random_" in key and key.startswith("partial_focal")
    }
    if density:
        lines += [
            "### Experiment A: annotation density (random sampling)",
            "",
            "| Points per crop | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy | Best epoch |",
            "|---:|---:|---|---:|---:|---:|",
        ]
        rows = []
        for key, value in density.items():
            points = int(key.rsplit("_", 1)[-1].removesuffix("pts"))
            runs = value["runs"]
            rows.append((points, value, runs))
        for points, value, runs in sorted(rows):
            best_epochs = ", ".join(str(run.get("best_epoch")) for run in runs)
            lines.append(
                f"| {points} | {value['best_miou']['n']} | {_fmt(value['best_miou'])} "
                f"| {_fmt(value['mean_f1'])} | {_fmt(value['pixel_accuracy'])} | {best_epochs} |"
            )
        lines.append("")

    sampling = {
        key: value
        for key, value in data.items()
        if ("_random_" in key or "_class_balanced_" in key) and "_100pts" in key
    }
    if sampling:
        lines += [
            "### Experiment B: sampling strategy (100 points per crop)",
            "",
            "| Strategy | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy |",
            "|---|---:|---|---:|---:|",
        ]
        for key, value in sorted(sampling.items()):
            strategy = "class-balanced" if "class_balanced" in key else "random"
            lines.append(
                f"| {strategy} | {value['best_miou']['n']} | {_fmt(value['best_miou'])} "
                f"| {_fmt(value['mean_f1'])} | {_fmt(value['pixel_accuracy'])} |"
            )
        lines.append("")

    # Per-class IoU for the most informative settings.
    per_class_keys = [
        key
        for key in data
        if key.endswith("_10pts") or key.endswith("_100pts") or key.endswith("_1000pts")
    ]
    if per_class_keys:
        lines += [
            "### Per-class IoU",
            "",
            "| Setting | background | building | road | water | barren | forest | agriculture |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in sorted(per_class_keys):
            values = data[key].get("per_class_iou_mean")
            if not values:
                continue
            cells = " | ".join("n/a" if v is None else f"{v:.3f}" for v in values)
            label = key.replace("partial_focal_gamma2.0_", "").replace("_", " ")
            lines.append(f"| {label} | {cells} |")
        lines.append("")

    output = Path(args.output)
    lines += _run_details(Path(args.root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(lines)} lines)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
