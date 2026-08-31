#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Train B1 (3), B2 (3), and B3 W1/W2/W3 (9) using train/validation only,
# then lock one B3 weight using validation EF MAE only.
# Usage:
#   bash scripts/training/run_stage1_corrected_training.sh /path/to/datasets

DATA_ROOT="${1:-/data/jewettm/dynamic/datasets}"
RUN_ROOT="${RUN_ROOT:-output/stage1/corrected}"
RESULTS_ROOT="${RESULTS_ROOT:-results/stage1/corrected}"
EPOCHS="${EPOCHS:-45}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FRAMES="${FRAMES:-32}"
PERIOD="${PERIOD:-2}"
LR="${LR:-0.0001}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LR_STEP="${LR_STEP:-15}"
DECODER_WIDTH="${DECODER_WIDTH:-128}"
SEEDS=(42 2026 3407)

COMMON=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --frames "$FRAMES"
  --period "$PERIOD"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --learning-rate "$LR"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --lr-step-period "$LR_STEP"
  --segmentation-decoder-width "$DECODER_WIDTH"
)

mkdir -p "$RUN_ROOT" "$RESULTS_ROOT"

for seed in "${SEEDS[@]}"; do
  python scripts/training/train_stage1_b1_video_ef.py \
    "${COMMON[@]}" \
    --seed "$seed" \
    --output "$RUN_ROOT/B1_video_ef/seed_${seed}"
done

# B2 must be rerun: its architecture changed to match B3's segmentation path.
for seed in "${SEEDS[@]}"; do
  python scripts/training/train_stage1_b2_segmentation.py \
    "${COMMON[@]}" \
    --seed "$seed" \
    --output "$RUN_ROOT/B2_segmentation/seed_${seed}"
done

for weight in W1 W2 W3; do
  for seed in "${SEEDS[@]}"; do
    python scripts/training/train_stage1_b3_video_multitask.py \
      "${COMMON[@]}" \
      --weight "$weight" \
      --seed "$seed" \
      --output "$RUN_ROOT/B3_video_mtl/$weight/seed_${seed}"
  done
done

python scripts/analysis/summarize_stage1_corrected_validation.py \
  --run-root "$RUN_ROOT" \
  --results-root "$RESULTS_ROOT/validation_weight_selection"

echo "Validation-only training complete. Locked weight:"
cat "$RESULTS_ROOT/validation_weight_selection/LOCKED_WEIGHT.txt"
echo "Do not run test evaluation until this lock is accepted/frozen."
