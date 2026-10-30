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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(lines)} lines)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
