#!/usr/bin/env bash
# Reproduce every experiment in the report.
#
#   bash scripts/run_all_experiments.sh            # full runs (10 epochs)
#   EPOCHS=2 bash scripts/run_all_experiments.sh   # quick check
#
# Requires a validated LoveDA directory (see README) and a CUDA GPU for the
# timings quoted in the report; everything also runs on CPU, just slower.
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-10}"
SEEDS="${SEEDS:-42}"

echo "== checking dataset =="
uv run pointseg-check-data --config configs/default.yaml

echo "== Experiment A: annotation density (random sampling) =="
uv run pointseg-exp-density \
  --config configs/exp_density/A3.yaml \
  --points 10 50 100 500 1000 \
  --seeds ${SEEDS} \
  --epochs "${EPOCHS}" \
  --output-dir outputs/exp_density

echo "== Experiment B: sampling strategy at 100 points =="
# The random arm is the 100-point density run, so only the class-balanced arm
# has to be trained here.
uv run pointseg-exp-sampling \
  --config configs/exp_sampling/class_balanced.yaml \
  --strategies class_balanced \
  --seeds ${SEEDS} \
  --epochs "${EPOCHS}" \
  --output-dir outputs/exp_sampling

echo "== aggregating =="
uv run pointseg-aggregate --root outputs

echo "== tables and figures =="
uv run python scripts/report_tables.py --root outputs
uv run python scripts/report_figures.py --root outputs

echo "== done; see outputs/ and report/figures/ =="
