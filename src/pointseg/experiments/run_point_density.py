"""Experiment A: annotation density (random sampling)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pointseg.config import load_config
from pointseg.training.train import train
from pointseg.utils import get_logger

logger = get_logger(__name__)

DEFAULT_POINTS = [10, 50, 100, 500, 1000]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the annotation-density experiment")
    parser.add_argument("--config", default="configs/exp_density/A3.yaml")
    parser.add_argument("--points", nargs="+", type=int, default=DEFAULT_POINTS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="one or more seeds; defaults to the config seed",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    base = load_config(args.config, args.overrides)
    seeds = args.seeds or [base.seed]
    output_dir = Path(args.output_dir) if args.output_dir else Path(base.output_dir)
    summaries = []
    for points in args.points:
        for seed in seeds:
            config = base.copy()
            config.point_sampling.points_per_crop = int(points)
            config.point_sampling.strategy = "random"
            config.seed = int(seed)
            config.output_dir = str(output_dir)
            config.run_name = f"density_{points}pts_seed{seed}"
            if args.epochs is not None:
                config.training.epochs = int(args.epochs)
            logger.info("=== density run: %s points, seed %d ===", points, seed)
            summaries.append(train(config))
    for summary in summaries:
        logger.info(
            "%-28s best mIoU %.4f (epoch %d)",
            summary["run_name"],
            summary["best_miou"],
            summary["best_epoch"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
