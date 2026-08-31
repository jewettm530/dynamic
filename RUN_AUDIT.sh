#!/usr/bin/env bash
set -euo pipefail

# Corrected Stage 1 audit/evidence runner.
# Usage:
#   bash RUN_AUDIT.sh <DATASET_ROOT> <OUTPUT_DIR> [B1_OR_B3_CHECKPOINT]

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash RUN_AUDIT.sh <DATASET_ROOT> <OUTPUT_DIR> [B1_OR_B3_CHECKPOINT]"
  exit 1
fi

DATASET_ROOT="$(realpath "$1")"
OUTPUT_DIR="$2"
CHECKPOINT="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$DATASET_ROOT/FileList.csv" ] || [ ! -d "$DATASET_ROOT/Videos" ]; then
  echo "ERROR: corrected EF audit requires FileList.csv and Videos/."
  exit 1
fi

echo "1/2: architecture/gradient smoke test"
python scripts/verification/smoke_test_stage1_corrected.py \
  --output "$OUTPUT_DIR/model_smoke_test.json"

echo "2/2: video-only EF inference-path test (temporary root has NO VolumeTracings.csv)"
ARGS=(
  --data-root "$DATASET_ROOT"
  --split test
  --output "$OUTPUT_DIR/video_only_inference.json"
)
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
python scripts/verification/verify_video_only_inference.py "${ARGS[@]}"

echo "Corrected Stage 1 audit complete: $OUTPUT_DIR"
echo "No frames, masks, tracings, videos, or patient-derived visualizations were exported."
