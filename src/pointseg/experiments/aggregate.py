"""Aggregate repeated runs into mean +/- std over seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pointseg.utils import write_json


def _key(config: dict[str, Any]) -> str:
    strategy = config.get("point_sampling", {}).get("strategy", "?")
    points = config.get("point_sampling", {}).get("points_per_crop", "?")
    gamma = config.get("loss", {}).get("gamma", "?")
    loss = config.get("loss", {}).get("name", "?")
    return f"{loss}_gamma{gamma}_{strategy}_{points}pts"


def collect(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(root).rglob("metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "best_metrics" not in data or "config" not in data:
            continue
        groups[_key(data["config"])].append(
            {
                "run": data.get("run_name", path.parent.name),
                "seed": data.get("config", {}).get("seed"),
                "best_miou": data.get("best_miou"),
                "best_epoch": data.get("best_epoch"),
                "final_miou": data.get("final_miou"),
                "mean_f1": (data.get("best_metrics") or {}).get("mean_f1"),
                "pixel_accuracy": (data.get("best_metrics") or {}).get("pixel_accuracy"),
                "per_class_iou": (data.get("best_metrics") or {}).get("per_class_iou"),
            }
        )
    return groups


def _mean_std(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if clean.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "n": int(clean.size),
    }


def aggregate(root: str | Path) -> dict[str, Any]:
    groups = collect(root)
    result: dict[str, Any] = {}
    for key, runs in groups.items():
        per_class = [r["per_class_iou"] for r in runs if r.get("per_class_iou")]
        class_means = None
        if per_class:
            arr = np.asarray(per_class, dtype=float)
            class_means = np.nanmean(arr, axis=0).tolist()
        result[key] = {
            "runs": runs,
            "best_miou": _mean_std([r["best_miou"] for r in runs]),
            "final_miou": _mean_std([r["final_miou"] for r in runs]),
            "mean_f1": _mean_std([r["mean_f1"] for r in runs]),
            "pixel_accuracy": _mean_std([r["pixel_accuracy"] for r in runs]),
            "per_class_iou_mean": class_means,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files over seeds")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = aggregate(args.root)
    if not result:
        print(f"no metrics.json files found under {args.root!r}")
        return 1

    print(f"{'setting':<48} {'mIoU (mean +/- std)':>22} {'n':>3}")
    for key, info in sorted(result.items()):
        stat = info["best_miou"]
        mean = "n/a" if stat["mean"] is None else f"{stat['mean']:.4f}"
        std = "n/a" if stat["std"] is None else f"{stat['std']:.4f}"
        print(f"{key:<48} {mean + ' +/- ' + std:>22} {stat['n']:>3}")

    output = Path(args.output) if args.output else Path(args.root) / "aggregate.json"
    write_json(output, result)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
