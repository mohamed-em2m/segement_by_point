# Point-Supervised Semantic Segmentation on LoveDA

Training a semantic segmentation network with **only sparse point labels**
(a few labelled pixels per crop) and evaluating it against **full** validation
masks. The loss is the partial focal loss from the assignment:

```
L_pFCE = sum_i [ FocalLoss(pred_i, GT_i) * MASK_labeled,i ] / sum_i MASK_labeled,i
```

The implementation follows [`AGENTS.md`](AGENTS.md) (also copied to
[`PLAN.md`](PLAN.md)).

## What is implemented

- `PartialFocalLoss` and an explicit `PartialCrossEntropyLoss`; `gamma=0`
  reproduces partial CE exactly (unit-tested).
- LoveDA loader with the official split and label remapping
  (`0 -> 255`, `1..7 -> 0..6`), aligned crops/flips/rotations.
- Random and class-balanced point simulation with a configurable budget.
- U-Net + ResNet-34 (ImageNet weights) returning raw logits.
- Training with AdamW, cosine schedule, AMP, checkpointing and resume.
- Tiled, deterministic validation with split-level confusion-matrix metrics.
- Experiment runners for annotation density (A) and sampling strategy (B),
  plus seed aggregation and figures.
- A self-contained notebook and a technical report generated from real runs.

## Install

The project uses [uv](https://docs.astral.sh/uv/). On a machine that already
has a CUDA PyTorch build, reuse it to avoid downloading torch again:

```bash
uv venv --python 3.13 --system-site-packages .venv
uv pip install -e . --group dev
```

On a clean machine, the standard flow downloads PyTorch from PyPI:

```bash
uv sync --group dev
```

CPU-only fallback:

```bash
uv sync --group dev --no-install-package torch --no-install-package torchvision
# then install CPU wheels, e.g.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Dataset

Download the official LoveDA release (Train.zip, Val.zip) from
<https://doi.org/10.5281/zenodo.5706578> and unpack it as:

```
data/LoveDA/
├── Train/{Rural,Urban}/{images_png,masks_png}
└── Val/{Rural,Urban}/{images_png,masks_png}
```

Then validate the layout:

```bash
uv run pointseg-check-data --config configs/default.yaml
```

The dataset is never downloaded automatically and is git-ignored. The
experiments in this repository use a **deterministic subset** (600 training and
200 validation images, balanced across Rural/Urban) because the available GPU is
a single T4; the subset is selected by `dataset.max_train_images`,
`dataset.max_val_images` and `dataset.subset_seed`. Set both limits to `null`
to train on the full official split.

## Train

```bash
# full default configuration (30 epochs, 100 random points)
uv run pointseg-train --config configs/default.yaml

# quick smoke test
uv run pointseg-train --config configs/smoke.yaml

# override any field
uv run pointseg-train --config configs/default.yaml \
  --set point_sampling.points_per_crop=500 --set training.epochs=10
```

Run artefacts are written to `outputs/<run_name>/`:
`config.yaml`, `metadata.json`, `history.csv`, `metrics.json`, `best.pt`,
`last.pt`, `figures/`.

## Evaluate

```bash
uv run pointseg-eval --config outputs/baseline_100pts/config.yaml \
  --checkpoint outputs/baseline_100pts/best.pt
```

## Experiments

```bash
uv run pointseg-exp-density  --config configs/exp_density/A3.yaml
uv run pointseg-exp-sampling --config configs/exp_sampling/random.yaml
uv run pointseg-aggregate    --root outputs
uv run pointseg-visualize    --run-dir outputs/exp_density/density_100pts_seed42 \
  --config outputs/exp_density/density_100pts_seed42/config.yaml \
  --checkpoint outputs/exp_density/density_100pts_seed42/best.pt
```

## Tests and linting

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Design decisions

- **Fixed point sets (default).** For a given image the crop window and the
  point set are deterministic, seeded by `(seed, image_index)`. The annotation
  budget is therefore honest: the same N pixels are used every epoch.
  Geometric augmentation is still random and is applied to image and sparse
  mask together. Setting `point_sampling.fixed_points=false` draws a new crop
  and new points every epoch (the network then sees more labels over training);
  it is reported as a separate setting.
- **Full-mask isolation.** The training dataset never returns the full mask.
  Sampling happens inside `__getitem__`, and the full mask is deleted before
  the item is returned. Validation always uses complete ground-truth masks.
- **No-data.** Original id 0 and any unknown id map to 255 and are never
  sampled or supervised.
- **Absent classes.** A class with zero ground-truth pixels in the validation
  split is reported as `NaN` and excluded from mIoU/mean-F1.
- **Determinism.** Python, NumPy and torch are seeded; DataLoader workers get
  deterministic seeds. GPU determinism is only approximate.

## Compute note

All reported experiments were trained on a single Tesla T4 (15 GB) with the
subset described above. Results are produced by the scripts in this repository;
no numbers are hardcoded.

## Layout

```
configs/            YAML configurations (default, smoke, experiments)
src/pointseg/       package (data, losses, models, training, evaluation, experiments)
tests/              unit and smoke tests
scripts/            dataset helper
notebooks/          end-to-end walkthrough
report/             technical report
outputs/            git-ignored run artefacts
data/               git-ignored LoveDA dataset
```
