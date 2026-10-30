"""Experiment B: random vs class-balanced point sampling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pointseg.config import load_config
from pointseg.training.train import train
from pointseg.utils import get_logger

logger = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the sampling-strategy experiment")
    parser.add_argument("--config", default="configs/exp_sampling/random.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "class_balanced"],
        choices=["random", "class_balanced"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    base = load_config(args.config, args.overrides)
    seeds = args.seeds or [base.seed]
    output_dir = Path(args.output_dir) if args.output_dir else Path(base.output_dir)
    summaries = []
    for strategy in args.strategies:
        for seed in seeds:
            config = base.copy()
            config.point_sampling.strategy = strategy
            config.seed = int(seed)
            config.output_dir = str(output_dir)
            config.run_name = (
                f"sampling_{strategy}_{config.point_sampling.points_per_crop}pts_seed{seed}"
            )
            if args.epochs is not None:
                config.training.epochs = int(args.epochs)
            logger.info("=== sampling run: %s, seed %d ===", strategy, seed)
            summaries.append(train(config))
    for summary in summaries:
        logger.info(
            "%-36s best mIoU %.4f (epoch %d)",
            summary["run_name"],
            summary["best_miou"],
            summary["best_epoch"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
