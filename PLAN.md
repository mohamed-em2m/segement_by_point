# Project Plan: Point-Supervised Remote Sensing Semantic Segmentation (LoveDA)

> **Status:** plan only. No implementation starts until this plan is approved.

## Task alignment (from the assignment brief)

The assignment asks for three things, and the plan maps onto them as follows.

| Assignment item | How this plan covers it |
|---|---|
| 1. Implement the partial loss (`pfCE = sum(Focal(pred, GT) x MASK_labeled) / sum(MASK_labeled)`) | `PartialFocalLoss` (the equation as written) plus an explicit `PartialCrossEntropyLoss`. Focal with `gamma=0` must equal partial CE, which is tested. |
| 2. Use any remote sensing segmentation data, simulate random point labels, add the loss to any segmentation network | LoveDA, random (and class-balanced) point simulation, U-Net with ResNet-34 (ImageNet pretrained, matching the brief's transfer-learning baseline). |
| 3. Design experiments on one or two factors and write a technical report (method + experiment: purpose/hypothesis, process, results) | Two factors: annotation density (A) and sampling strategy (B). The report is structured exactly as the brief asks. |

**Deliverable format.** The brief accepts a Python file or a Jupyter notebook plus a supporting document. This plan delivers:

- the `pointseg` package and CLI (reproducible, tested);
- one **self-contained Jupyter notebook** (`notebooks/point_supervised_segmentation.ipynb`) that walks through loss, point simulation, training, experiments, and results by importing the package, so the reviewer can run it top to bottom;
- the **technical report** (`report/technical_report.md`, also exported to PDF) as the supporting document.

**Out of scope (listed on the brief's slide as parts of the larger system, not as tasks).** Semi-supervised learning on unlabeled pixels, test-time augmentation, model ensembling, and histogram matching are not implemented. They are mentioned in the report's "limitations and future work" section only.

## Objective

Build a reproducible PyTorch pipeline that trains a semantic segmentation network on LoveDA using **only sparse point labels** (a small number of labeled pixels per training crop), and evaluates it against **full** validation masks.

## What changed from the original draft

| Area | Change |
|---|---|
| Loss | Partial Cross Entropy is replaced by **Partial Focal Loss**, matching the stated equation. |
| Packaging | `uv` + `pyproject.toml` with a `src/` layout. `requirements.txt` is dropped. |
| Layout | Everything lives inside one project folder, `remote-sensing-point-segmentation/`. |
| Git | Commit and push after each milestone, gated by passing tests. |
| Experiments | Optional Experiment C (focal gamma ablation) added. |
| Design | Fixed vs. resampled points decision, plus several reproducibility fixes. |

---

## 1. Technology stack

- Python 3.10+, managed with **uv**
- PyTorch, torchvision
- segmentation-models-pytorch (U-Net, ResNet-34, ImageNet weights, 3 in-channels, 7 classes, `activation=None`)
- Albumentations, NumPy, OpenCV (headless), Matplotlib, tqdm, PyYAML
- Optional: scikit-learn, pandas (analysis group)

```python
smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=7,
    activation=None,
)
```

The model returns raw logits. Do not add YOLO or instance segmentation components.

---

## 2. Dataset

**Dataset:** LoveDA ([repository](https://github.com/Junjue-Wang/LoveDA), [Zenodo](https://doi.org/10.5281/zenodo.5706578)). Images are 1024x1024 RGB with single-channel masks and an official train/val split.

**Label remap**

| Original ID | Class | Training ID |
|---|---|---|
| 0 | No-data | 255 (`IGNORE_INDEX`) |
| 1 | Background | 0 |
| 2 | Building | 1 |
| 3 | Road | 2 |
| 4 | Water | 3 |
| 5 | Barren | 4 |
| 6 | Forest | 5 |
| 7 | Agriculture | 6 |

**Rules**

- No-data pixels are never a semantic class and are never supervised.
- The full training mask is never used as a training target.
- Full validation masks are used for evaluation.
- The official split is preserved.
- Dataset files are never committed to Git.
- Dataset paths are set in `configs/default.yaml`.
- The loader supports the real LoveDA folder structure and validates image/mask pairs before training (`pointseg-check-data`).
- If the dataset is missing, print clear download and directory instructions. Never download silently.

**Preprocessing**

- RGB image loading; mask loading without altering class values.
- Random 512x512 crops for training.
- Horizontal and vertical flips, optional 90-degree rotations.
- ImageNet normalization.
- Nearest-neighbor interpolation for all mask resizing.
- Image and mask stay spatially aligned (tested).
- Validation is deterministic: full image or tiled 512x512 windows with configurable overlap and logit averaging.

---

## 3. Point annotation simulation

**File:** `src/pointseg/datasets/point_sampling.py`

```python
sample_random_points(mask, num_points, ignore_index=255, seed=None)
sample_class_balanced_points(mask, num_points, ignore_index=255, seed=None)
```

Requirements:

1. Take a fully labeled crop internally.
2. Identify valid pixels (exclude `IGNORE_INDEX`).
3. Select N pixel locations.
4. Keep original labels only at those locations; set all others to 255.
5. Never sample no-data pixels.
6. Handle crops with fewer than N valid pixels (return all valid pixels, log it).
7. Deterministic with a seed.
8. Support random and class-balanced strategies.
9. Point count is configurable per crop.

**Full-mask isolation (critical)**

- The training dataset returns only `image, sparse_mask`.
- The validation dataset returns `image, full_mask`.
- Sampling happens inside the training dataset's `__getitem__`; the full mask never leaves it.
- A test asserts the returned training mask has at most N non-255 pixels.

**Fixed vs. resampled points (key decision)**

If points are resampled every epoch, the network sees many more distinct labels over training than the nominal budget, which weakens the "N points per crop" claim.

- **Recommended default:** pre-sample a fixed point set per training image, seeded by `(seed, image_id)`. The annotation budget is then honest and reproducible.
- **Option:** `resample_each_epoch: true`, reported as a separate setting.

---

## 4. Loss: Partial Focal Loss

**File:** `src/pointseg/losses/partial_focal.py`, class `PartialFocalLoss`

Equation:

```
L_pFCE = ( sum_i [ FocalLoss(pred_i, GT_i) * MASK_labeled,i ] ) / ( sum_i MASK_labeled,i )
```

Per-pixel focal loss:

```
FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

**Design**

- Inputs: logits `[B, C, H, W]`, sparse targets `[B, H, W]`, ignore value 255.
- Compute `log_softmax` in float32, even under AMP.
- Obtain `log p_t` with `gather`; clamp ignored targets to a safe index before gathering.
- Compute `p_t = exp(log p_t)`, then the focal term.
- Multiply by the labeled mask and divide by the labeled pixel count.
- Parameters: `gamma` (default 2.0), optional per-class `alpha`, `ignore_index` (255).
- `gamma=0` and `alpha=None` reduces exactly to Partial CE (sanity check and ablation baseline).
- A separate `PartialCrossEntropyLoss` in `src/pointseg/losses/partial_ce.py` is implemented explicitly, because the assignment lists partial CE as item 1. It uses `F.cross_entropy(ignore_index=255, reduction="sum")` divided by the labeled pixel count, with the same all-ignored policy.
- **All-ignored batch:** by default return a differentiable zero (`logits.sum() * 0`) and log a warning; `strict=True` raises an error. Documented in the docstring.
- **Alpha:** default none. If used, estimate class weights only from sparse labels, never from full masks.

**Unit tests**

1. `gamma=0`, no alpha matches `F.cross_entropy(ignore_index=255)`.
2. Matches a hand-computed focal value on a tiny tensor.
3. Changing logits at ignored pixels does not change the loss.
4. Gradient is exactly zero at ignored pixels.
5. Finite with extreme logits (+/-1e4) and in fp16/bf16.
6. All-ignored behavior tested in default and strict modes.
7. Higher gamma lowers the loss for confident, correct predictions.

---

## 5. Model

**File:** `src/pointseg/models/unet.py`

- U-Net via segmentation-models-pytorch, ResNet-34 ImageNet encoder, 7 output channels, raw logits.
- CPU and CUDA support.
- Initialization helper and parameter-count report.

---

## 6. Training pipeline

**File:** `src/pointseg/training/train.py`

Suggested initial configuration:

```yaml
seed: 42

dataset:
  name: loveda
  root: ./data/LoveDA
  crop_size: 512

model:
  architecture: unet
  encoder: resnet34
  pretrained: true
  num_classes: 7

loss:
  name: partial_focal
  gamma: 2.0
  alpha: null
  strict: false

training:
  epochs: 30
  batch_size: 4
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine
  amp: true

point_sampling:
  strategy: random          # random | class_balanced
  points_per_crop: 100
  ignore_index: 255
  fixed_points: true        # pre-sampled per image; false = resample each epoch

evaluation:
  metrics: [miou, f1, pixel_accuracy]
  tile_size: 512
  tile_overlap: 0.25
```

Loop: load image and full mask internally, apply aligned preprocessing and crop, sample points, pass only image and sparse mask onward, forward, Partial Focal Loss, backward, update, log loss, validate on full masks each epoch.

Includes AdamW, LR scheduler, optional AMP, logging, checkpoint saving (weights, optimizer, scheduler, epoch, config, seed), best checkpoint by validation mIoU, and resume support.

Training loss never touches full masks.

---

## 7. Evaluation

**File:** `src/pointseg/evaluation/metrics.py`

- Confusion matrix accumulated **once over the whole split** (not per-image averages).
- Per-class IoU, mean IoU, per-class F1, mean F1, pixel accuracy.
- Ignore-index pixels excluded.
- Classes with zero ground-truth pixels are reported as NaN and excluded via `nanmean`; this policy is documented.
- Validation uses complete ground-truth masks only.

---

## 8. Experiments

All experiments share the same split, architecture, preprocessing, optimizer, schedule, and epoch budget. One seed first; three seeds (mean +/- std) if compute allows. Results are never hardcoded or fabricated.

### Experiment A: annotation density (random sampling)

| Run | Points per crop |
|---|---:|
| A1 | 10 |
| A2 | 50 |
| A3 | 100 |
| A4 | 500 |
| A5 | 1000 |

*Hypothesis:* more points generally improve performance, with eventual saturation.

### Experiment B: sampling strategy (100 points per crop)

- Random sampling
- Class-balanced sampling

*Hypothesis:* class-balanced sampling may improve underrepresented classes.

> The brief asks for one or two factors. **A and B are the required experiments.** C is an optional extra, run only if compute allows, and reported separately.

### Experiment C (optional): focal gamma ablation

`gamma` in {0, 1, 2, 3} at 100 random points. `gamma=0` is the Partial CE baseline, directly testing the benefit of focal loss.

---

## 9. Visualizations

- Original image, full ground truth, sparse point mask, prediction, prediction vs. ground truth.
- Training loss curves, validation mIoU curves.
- Per-class IoU charts, experiment comparison charts.
- One consistent class color map.

---

## 10. Project structure

Everything goes inside a single folder with a `src/` layout.

```text
remote-sensing-point-segmentation/
├── pyproject.toml              # deps, scripts, ruff, pytest config
├── uv.lock                     # committed for reproducibility
├── .python-version
├── .gitignore                  # data/, outputs/, checkpoints/, .venv/
├── README.md
├── PLAN.md
├── configs/
│   ├── default.yaml
│   ├── exp_density/            # one config per run (A1-A5)
│   └── exp_sampling/           # random vs class_balanced
├── src/pointseg/
│   ├── config.py               # typed config loading and validation
│   ├── datasets/               # loveda.py, point_sampling.py, transforms.py
│   ├── models/unet.py
│   ├── losses/                 # partial_focal.py, partial_ce.py
│   ├── training/               # train.py, validate.py, checkpoint.py
│   ├── evaluation/             # metrics.py, visualize.py
│   ├── experiments/            # run_point_density.py, run_sampling_strategy.py, aggregate.py
│   └── utils/                  # seed.py, logging.py
├── tests/
│   ├── test_dataset.py
│   ├── test_point_sampling.py
│   ├── test_partial_focal.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_checkpoint.py
│   └── test_smoke_train.py
├── scripts/                    # check_dataset.py
├── report/technical_report.md  # generated from real outputs
├── outputs/                    # git-ignored
├── data/                       # git-ignored, holds LoveDA
└── notebooks/
    └── point_supervised_segmentation.ipynb   # main deliverable walkthrough
```

Tests also include `tests/test_partial_ce.py` for the explicit partial CE loss.

`main.py` is replaced by CLI entry points defined in `pyproject.toml`:

`pointseg-train`, `pointseg-eval`, `pointseg-exp-density`, `pointseg-exp-sampling`, `pointseg-check-data`

---

## 11. Tooling: uv and pyproject.toml

- `requires-python = ">=3.10"`, pinned via `.python-version`.
- Runtime dependencies: torch, torchvision, segmentation-models-pytorch, albumentations, numpy, opencv-python-headless, matplotlib, tqdm, pyyaml.
- Dependency groups:
  - `dev`: pytest, pytest-cov, ruff, pre-commit
  - `analysis`: scikit-learn, pandas
- PyTorch/CUDA: `[tool.uv.sources]` plus an index entry for the right CUDA wheel, with a documented CPU fallback. This is the most likely dependency problem, so it is checked first.

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pointseg-train --config configs/default.yaml
```

---

## 12. Git workflow

- `git init` inside the project folder; `.gitignore` is added before the first commit; the remote is confirmed.
- If no remote is configured or push credentials are missing, report it as a blocker. Never skip pushes silently.
- Commit and push after each big change, **only when `ruff` and `pytest` pass**.
- Conventional commit messages.
- Never committed: dataset, checkpoints, outputs, `.venv`. Small result summaries (JSON/CSV) may be committed if wanted.

| # | Milestone | Commit message |
|---|---|---|
| 1 | Scaffold: pyproject, uv.lock, .gitignore, README skeleton, config | `chore: scaffold project with uv and pyproject` |
| 2 | Dataset, label remap, transforms, data validator + tests | `feat(data): LoveDA loader with label remapping` |
| 3 | Point sampling (random, class-balanced) + tests | `feat(data): point annotation simulation` |
| 4 | Model + Partial Focal Loss + tests | `feat(loss): partial focal loss` and `feat(model): U-Net ResNet-34` |
| 5 | Metrics + validation + tests | `feat(eval): confusion-matrix metrics` |
| 6 | Training loop, scheduler, AMP, checkpoints, resume + smoke test | `feat(train): training pipeline with checkpointing` |
| 7 | Experiment runners and aggregation | `feat(exp): density and sampling experiments` |
| 8 | Visualizations and report generation | `feat(viz): plots and technical report` |

---

## 13. Reproducibility

- Seed Python, NumPy, and torch; seed DataLoader workers.
- Log library versions and git commit hash in each run's metadata.
- Document that GPU determinism is only approximate.
- Typed, validated config with CLI overrides, e.g. `--set point_sampling.points_per_crop=500`. The resolved config is saved per run.
- Run layout: `outputs/<experiment>/<run_name>/{config.yaml, metrics.json, history.csv, best.pt, last.pt, figures/}`
- `aggregate.py` computes mean +/- std over seeds from saved metrics files.

---

## 14. Implementation phases

1. **Inspect** environment (GPU, CUDA, uv, git remote, dataset presence) and finalize the plan.
2. **Scaffold**, then dataset and point sampling.
3. **Model**, Partial Focal Loss, unit tests.
4. **Metrics**, validation, training, checkpointing.
5. **Smoke test** on a tiny subset. No long runs until it passes.
6. **Baseline:** 100 random points, gamma=2.
7. **Experiments** A, B, and optionally C.
8. **Visualizations** and technical report from real outputs.

Each phase ends with tests, a commit, and a push.

---

## 15. Pre-training checks

1. Dataset loading.
2. Original-to-training label mapping.
3. Image/mask alignment after transforms.
4. Sparse sampling and ignored pixels.
5. Partial Focal Loss calculations.
6. Model forward pass.
7. Backward pass.
8. Small training smoke test.
9. Validation and metric calculations.
10. Checkpoint save/load.

## 16. Acceptance criteria

- Model outputs `[B, 7, H, W]` logits.
- Sparse masks contain only class IDs 0-6 or 255.
- Exactly N points are sampled whenever enough valid pixels exist.
- No-data pixels are never supervised.
- Loss uses labeled pixels only.
- Validation uses full ground truth.
- Training and validation work on CPU and CUDA.
- All settings are reproducible and configurable.
- `PartialFocalLoss(gamma=0)` equals Partial CE in tests.
- `uv sync && uv run pytest` passes on a clean clone.
- Every phase has a pushed commit.

## 17. Technical report

`report/technical_report.md` follows the structure the brief requires: **method** and **experiment (purpose/hypothesis, experimental process, results)**.

1. Introduction and problem (why point supervision is hard).
2. **Method:** dataset, architecture, point generation, Partial Focal Loss and Partial CE formulation, training configuration.
3. **Experiments**, one subsection each for A and B (and C if run):
   - Purpose and hypothesis
   - Experimental process
   - Results (tables and figures)
4. Discussion, limitations, and future work (semi-supervised learning, TTA, ensembling, histogram matching).
5. Conclusion.

All numbers come from real experiment outputs. The report is also exported to PDF, and the notebook reproduces its tables and figures.

## 18. Final deliverables

Files created or modified, installation commands, dataset setup instructions, training command, evaluation command, experiment commands, test results, and remaining limitations. Blockers, missing data, dependency issues, and failed tests are reported clearly, and missing components are never replaced silently with dummy implementations.

---

## Open questions

1. Should the fixed per-image point set be the default?
2. Which git remote should I push to, and is it already configured?
3. Should Experiment C (focal gamma ablation) be included? (The brief only requires one or two factors.)
4. Is the package plus one notebook plus the report the right delivery format, or do you want a single `.py` file or notebook only?
5. What compute is available? If limited, I will cut epochs or use a LoveDA subset for the experiment runs and state that clearly in the report.
