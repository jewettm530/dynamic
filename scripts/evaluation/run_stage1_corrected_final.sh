#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Run only AFTER W1/W2/W3 are complete and LOCKED_WEIGHT.txt exists.
# Usage:
#   bash scripts/evaluation/run_stage1_corrected_final.sh /path/to/datasets

DATA_ROOT="${1:-/data/jewettm/dynamic/datasets}"
RUN_ROOT="${RUN_ROOT:-output/stage1/corrected}"
RESULTS_ROOT="${RESULTS_ROOT:-results/stage1/corrected}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python scripts/verification/verify_stage1_corrected_runs.py \
  --run-root "$RUN_ROOT" \
  --results-root "$RESULTS_ROOT"

# This script is the explicit video-only EF test path and does not load tracings.
python scripts/evaluation/evaluate_stage1_corrected_ef.py \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

# Segmentation is evaluated separately on labeled ED/ES frames.
python scripts/evaluation/evaluate_stage1_corrected_segmentation.py \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

python scripts/analysis/summarize_stage1_corrected_final.py \
  --results-root "$RESULTS_ROOT"
