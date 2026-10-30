"""Training loop for point-supervised segmentation."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pointseg.config import Config, load_config
from pointseg.datasets.loveda import build_loaders
from pointseg.evaluation.validate import format_metrics, validate
from pointseg.losses import build_loss
from pointseg.models.unet import build_model, count_parameters, describe_model
from pointseg.training.checkpoint import load_checkpoint, save_checkpoint
from pointseg.utils import get_logger, run_metadata, set_seed, write_json

logger = get_logger(__name__)


def _make_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    lr = config.training.learning_rate
    wd = config.training.weight_decay
    if config.training.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if config.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if config.training.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {config.training.optimizer!r}")


def _make_scheduler(config: Config, optimizer: torch.optim.Optimizer):
    if config.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=config.training.learning_rate * 0.01
        )
    if config.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def _write_history_header(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _append_history(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writerow(row)


def train(
    config: Config,
    run_dir: str | Path | None = None,
    resume: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one run and write ``config.yaml``, ``history.csv``, ``metrics.json``,
    ``last.pt`` and ``best.pt`` into ``run_dir``."""
    config.validate()
    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir) if run_dir is not None else Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")
    metadata = run_metadata()
    write_json(run_dir / "metadata.json", metadata)

    logger.info("run directory: %s", run_dir)
    logger.info("device: %s", device)

    train_loader, val_loader = build_loaders(config)
    logger.info(
        "data: %d training images, %d validation images",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    model = build_model(config.model).to(device)
    logger.info(describe_model(model))
    criterion = build_loss(config.loss)
    optimizer = _make_optimizer(config, model)
    scheduler = _make_scheduler(config, optimizer)
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_miou = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    if resume:
        payload = load_checkpoint(
            resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0))
        best_miou = float(payload.get("best_miou", float("-inf")))
        best_epoch = int(payload.get("best_epoch", -1))
        best_metrics = payload.get("best_metrics")
        history = list(payload.get("history", []))
        logger.info("resumed from %s at epoch %d", resume, start_epoch)

    history_path = run_dir / "history.csv"
    if not history_path.exists() and history:
        _write_history_header(history_path, history[0])

    epochs = config.training.epochs
    if start_epoch >= epochs:
        logger.warning("checkpoint is already at epoch %d >= epochs=%d", start_epoch, epochs)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        num_batches = 0
        point_counts: list[float] = []

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if config.training.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach())
            num_batches += 1
            point_counts.append(float(batch["num_points"].float().mean()))
            if config.training.log_interval and (step + 1) % config.training.log_interval == 0:
                logger.info(
                    "epoch %d/%d step %d/%d loss %.4f",
                    epoch + 1,
                    epochs,
                    step + 1,
                    len(train_loader),
                    running_loss / num_batches,
                )
            if config.training.max_train_batches and step + 1 >= config.training.max_train_batches:
                break

        train_loss = running_loss / max(1, num_batches)
        if scheduler is not None:
            scheduler.step()

        should_validate = (epoch + 1) % config.evaluation.every_n_epochs == 0 or epoch + 1 == epochs
        metrics: dict[str, Any] | None = None
        if should_validate:
            metrics = validate(model, val_loader, config, device)
            logger.info(
                "epoch %d/%d validation\n%s",
                epoch + 1,
                epochs,
                format_metrics(metrics),
            )
            if not math.isnan(metrics["miou"]) and metrics["miou"] > best_miou:
                best_miou = float(metrics["miou"])
                best_epoch = epoch + 1
                best_metrics = metrics
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    config=config,
                    seed=config.seed,
                    best_miou=best_miou,
                    best_epoch=best_epoch,
                    history=history,
                    metadata=metadata,
                )
                logger.info("new best mIoU %.4f at epoch %d", best_miou, best_epoch)

        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_miou": metrics["miou"] if metrics else "",
            "val_mean_f1": metrics["mean_f1"] if metrics else "",
            "val_pixel_accuracy": metrics["pixel_accuracy"] if metrics else "",
            "mean_points_per_crop": sum(point_counts) / max(1, len(point_counts)),
            "epoch_seconds": epoch_seconds,
        }
        history.append({k: v for k, v in row.items() if v != ""})
        if history_path.exists():
            _append_history(history_path, row)
        else:
            _write_history_header(history_path, row)

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            config=config,
            seed=config.seed,
            best_miou=best_miou,
            best_epoch=best_epoch,
            history=history,
            metadata=metadata,
        )
        logger.info(
            "epoch %d/%d done in %.1fs | train loss %.4f | best mIoU %.4f",
            epoch + 1,
            epochs,
            epoch_seconds,
            train_loss,
            best_miou,
        )

    final_metrics = metrics if metrics is not None else None
    summary = {
        "run_name": config.run_name,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "final_miou": final_metrics["miou"] if final_metrics else None,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "epochs": epochs,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "num_parameters": count_parameters(model),
        "config": config.to_dict(),
        "metadata": metadata,
        "history": history,
    }
    write_json(run_dir / "metrics.json", summary)
    logger.info(
        "finished run %s; best mIoU %.4f (epoch %d)", config.run_name, best_miou, best_epoch
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train point-supervised LoveDA segmentation")
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="dotted override, e.g. point_sampling.points_per_crop=500",
    )
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.run_name:
        config.run_name = args.run_name
    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
