"""LoveDA dataset loading with the official split and no-data handling.

Expected layout (the official Zenodo release)::

    LoveDA/
    ├── Train/
    │   ├── Rural/{images_png,masks_png}
    │   └── Urban/{images_png,masks_png}
    └── Val/
        ├── Rural/{images_png,masks_png}
        └── Urban/{images_png,masks_png}

The training dataset returns ``(image, sparse_mask)`` only. The full mask is
used internally to draw the point annotations and is then discarded, so it can
never leak into the loss. The validation dataset returns the full mask because
evaluation is supervised with complete ground truth.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pointseg.config import Config
from pointseg.constants import IGNORE_INDEX, NUM_CLASSES
from pointseg.datasets.point_sampling import sample_points
from pointseg.datasets.transforms import (
    augment,
    normalize_image,
    random_crop,
    remap_mask,
)
from pointseg.utils import worker_init_fn

logger = logging.getLogger(__name__)

DOWNLOAD_HELP = """\
LoveDA was not found at {root!r}.

Download the official release (Train.zip and Val.zip) from Zenodo:
    https://doi.org/10.5281/zenodo.5706578

Then unpack it so the directory looks like:
    {root}/Train/Rural/images_png
    {root}/Train/Rural/masks_png
    {root}/Val/Rural/images_png
    ...

If the archive expands to ``LoveDA/Train/...``, either point dataset.root at
the inner ``LoveDA`` folder or keep it at the parent (both are supported).
The dataset is never downloaded automatically.
"""

DOMAINS = ("Rural", "Urban")


def resolve_root(root: str | Path) -> Path:
    """Accept either ``.../LoveDA`` or its parent directory."""
    root = Path(root).expanduser()
    candidates = [root, root / "LoveDA"]
    for candidate in candidates:
        if (candidate / "Train").is_dir() and (candidate / "Val").is_dir():
            return candidate
    raise FileNotFoundError(DOWNLOAD_HELP.format(root=root))


def _pairs_for_domain(root: Path, split: str, domain: str) -> list[dict[str, object]]:
    image_dir = root / split / domain / "images_png"
    mask_dir = root / split / domain / "masks_png"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"missing LoveDA directory for {split}/{domain}: {image_dir} or {mask_dir}"
        )
    images = {p.stem: p for p in sorted(image_dir.glob("*.png"))}
    masks = {p.stem: p for p in sorted(mask_dir.glob("*.png"))}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"image/mask mismatch in {split}/{domain}: "
            f"{len(missing_masks)} images without masks, "
            f"{len(missing_images)} masks without images "
            f"(examples: {(missing_masks + missing_images)[:5]})"
        )
    if not images:
        raise ValueError(f"no image/mask pairs found in {split}/{domain}")
    return [
        {
            "image": images[stem],
            "mask": masks[stem],
            "id": f"{split}/{domain}/{stem}",
            "domain": domain,
        }
        for stem in sorted(images, key=_natural_key)
    ]


def _natural_key(stem: str) -> tuple[int, str]:
    return (int(stem), stem) if stem.isdigit() else (10**12, stem)


def discover_pairs(
    root: str | Path,
    split: str,
    domains: Sequence[str] = DOMAINS,
) -> list[dict[str, object]]:
    """List validated image/mask pairs for a split."""
    resolved = resolve_root(root)
    pairs: list[dict[str, object]] = []
    for domain in domains:
        pairs.extend(_pairs_for_domain(resolved, split, domain))
    if not pairs:
        raise ValueError(f"no pairs found for split {split!r}")
    return pairs


def stratified_subset(
    pairs: Sequence[dict[str, object]],
    max_images: int | None,
    seed: int,
) -> list[dict[str, object]]:
    """Deterministically keep at most ``max_images``, balanced across domains."""
    if max_images is None or max_images >= len(pairs):
        return list(pairs)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair["domain"])].append(pair)
    shuffled = {
        domain: [items[i] for i in rng.permutation(len(items))]
        for domain, items in sorted(groups.items())
    }
    selected: list[dict[str, object]] = []
    cursors = {domain: 0 for domain in shuffled}
    while len(selected) < max_images:
        progressed = False
        for domain in sorted(shuffled):
            if len(selected) >= max_images:
                break
            cursor = cursors[domain]
            if cursor < len(shuffled[domain]):
                selected.append(shuffled[domain][cursor])
                cursors[domain] = cursor + 1
                progressed = True
        if not progressed:
            break
    return selected


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return remap_mask(mask)


class LoveDATrainDataset(Dataset):
    """Training dataset that only ever exposes sparse point labels.

    When ``fixed_points`` is true the crop window *and* the point set are
    deterministic for a given image (seeded by ``seed`` and the image index).
    The annotation budget is therefore honest: the same N pixels are used in
    every epoch. Geometric augmentation is still random and is applied to the
    image and the sparse mask together. When false, a new crop and a new point
    set are drawn every epoch (the network then sees more labels over training).
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 512,
        points_per_crop: int = 100,
        strategy: str = "random",
        fixed_points: bool = True,
        seed: int = 42,
        domains: Sequence[str] = DOMAINS,
        max_images: int | None = None,
        subset_seed: int = 42,
        augment_data: bool = True,
        rotations: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.pairs = stratified_subset(
            discover_pairs(root, "Train", domains), max_images, subset_seed
        )
        self.crop_size = int(crop_size)
        self.points_per_crop = int(points_per_crop)
        self.strategy = strategy
        self.fixed_points = bool(fixed_points)
        self.seed = int(seed)
        self.augment_data = bool(augment_data)
        self.rotations = bool(rotations)
        self.ignore_index = int(ignore_index)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        image = _read_image(pair["image"])  # type: ignore[arg-type]
        full_mask = _read_mask(pair["mask"])  # type: ignore[arg-type]

        if self.fixed_points:
            rng = np.random.default_rng([self.seed, index])
        else:
            rng = np.random.default_rng([self.seed, self.epoch, index])

        image, full_mask = random_crop(image, full_mask, self.crop_size, rng)
        point_seed = int(rng.integers(0, 2**31 - 1))
        sparse = sample_points(
            full_mask,
            self.points_per_crop,
            strategy=self.strategy,
            ignore_index=self.ignore_index,
            seed=point_seed,
        )
        del full_mask  # the full mask must never leave this method

        if self.augment_data:
            aug_rng = np.random.default_rng([self.seed, self.epoch, index, 7])
            image, sparse = augment(image, sparse, aug_rng, rotations=self.rotations)

        return {
            "image": torch.from_numpy(normalize_image(image)),
            "mask": torch.from_numpy(sparse.astype(np.int64)),
            "image_id": index,
            "num_points": int(np.count_nonzero(sparse != self.ignore_index)),
        }


class LoveDAEvalDataset(Dataset):
    """Validation dataset: complete image and complete ground-truth mask."""

    def __init__(
        self,
        root: str | Path,
        domains: Sequence[str] = DOMAINS,
        max_images: int | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.pairs = stratified_subset(
            discover_pairs(root, "Val", domains), max_images, subset_seed
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        image = _read_image(pair["image"])  # type: ignore[arg-type]
        mask = _read_mask(pair["mask"])  # type: ignore[arg-type]
        return {
            "image": torch.from_numpy(normalize_image(image)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "image_id": index,
            "path": str(pair["image"]),
        }


def build_train_dataset(config: Config) -> LoveDATrainDataset:
    ds = config.dataset
    ps = config.point_sampling
    return LoveDATrainDataset(
        root=ds.root,
        crop_size=ds.crop_size,
        points_per_crop=ps.points_per_crop,
        strategy=ps.strategy,
        fixed_points=ps.fixed_points,
        seed=config.seed,
        domains=ds.domains,
        max_images=ds.max_train_images,
        subset_seed=ds.subset_seed,
        augment_data=ds.augment,
        rotations=ds.rotations,
        ignore_index=ps.ignore_index,
    )


def build_eval_dataset(config: Config) -> LoveDAEvalDataset:
    ds = config.dataset
    return LoveDAEvalDataset(
        root=ds.root,
        domains=ds.domains,
        max_images=ds.max_val_images,
        subset_seed=ds.subset_seed,
    )


def build_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_ds = build_train_dataset(config)
    val_ds = build_eval_dataset(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.training.num_workers > 0,
    )
    return train_loader, val_loader


def dataset_summary(root: str | Path, domains: Iterable[str] = DOMAINS) -> dict[str, object]:
    resolved = resolve_root(root)
    summary: dict[str, object] = {"root": str(resolved), "splits": {}}
    for split in ("Train", "Val"):
        counts = {}
        total = 0
        for domain in domains:
            pairs = _pairs_for_domain(resolved, split, domain)
            counts[domain] = len(pairs)
            total += len(pairs)
        summary["splits"][split] = {"total": total, "domains": counts}
    return summary


def check_data_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a LoveDA directory layout")
    parser.add_argument("--root", default=None, help="path to LoveDA (or its parent)")
    parser.add_argument("--config", default=None, help="read dataset.root from this config")
    args = parser.parse_args(argv)

    root = args.root
    if root is None and args.config:
        from pointseg.config import load_config

        root = load_config(args.config).dataset.root
    if root is None:
        root = "./data/LoveDA"

    try:
        summary = dataset_summary(root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[check-data] FAILED: {exc}")
        return 1

    print(f"[check-data] OK: {summary['root']}")
    for split, info in summary["splits"].items():  # type: ignore[union-attr]
        print(f"  {split}: {info['total']} pairs {info['domains']}")
    print(f"  classes: {NUM_CLASSES} (no-data is mapped to {IGNORE_INDEX})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(check_data_main())
