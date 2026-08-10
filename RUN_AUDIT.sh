#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EchoNet-Dynamic Stage 1 Audit Runner
#
# Usage:
#   bash RUN_AUDIT.sh <DATASET_ROOT> <OUTPUT_DIR>
#
# Example:
#   bash RUN_AUDIT.sh \
#       /data/jewettm/dynamic/datasets \
#       outputs/stage1_audit
# ============================================================

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  bash RUN_AUDIT.sh <DATASET_ROOT> <OUTPUT_DIR>"
    exit 1
fi

DATASET_ROOT="$(realpath "$1")"
OUTPUT_DIR="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Convert output to an absolute path after moving to repository root.
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

echo "============================================================"
echo " EchoNet-Dynamic Stage 1 Audit"
echo "============================================================"
echo
echo "Project root : $PROJECT_ROOT"
echo "Dataset root : $DATASET_ROOT"
echo "Output dir   : $OUTPUT_DIR"
echo "Python       : $(which python)"
echo "PYTHONPATH   : $PYTHONPATH"
echo

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
if [ ! -f "$PROJECT_ROOT/echonet/__init__.py" ]; then
    echo "ERROR: Missing $PROJECT_ROOT/echonet/__init__.py"
    echo "Restore the package initializer before running the audit."
    exit 1
fi

if [ ! -f "$DATASET_ROOT/FileList.csv" ]; then
    echo "ERROR: Could not find $DATASET_ROOT/FileList.csv"
    exit 1
fi

if [ ! -f "$DATASET_ROOT/VolumeTracings.csv" ]; then
    echo "ERROR: Could not find $DATASET_ROOT/VolumeTracings.csv"
    exit 1
fi

if [ ! -d "$DATASET_ROOT/Videos" ]; then
    echo "ERROR: Could not find $DATASET_ROOT/Videos/"
    exit 1
fi

# ------------------------------------------------------------
# Confirm Python is loading the local package and utilities.
# ------------------------------------------------------------
echo "Checking EchoNet import..."
python - <<'PY'
import os
import sys
import echonet
from echonet.datasets.echo import Echo

print("Python executable:", sys.executable)
print("EchoNet package:", echonet.__file__)
print("Echo dataset class:", Echo)
print("Echo utilities module:", echonet.utils.__file__)
print("loadvideo:", echonet.utils.loadvideo)
print("Working directory:", os.getcwd())
PY

echo
echo "EchoNet import successful."
echo

# ------------------------------------------------------------
# 1. Main data/code audit
# ------------------------------------------------------------
echo "============================================================"
echo "1/3 Running Stage 1 data/code audit"
echo "============================================================"

python scripts/verification/audit_stage1.py \
    --data-root "$DATASET_ROOT" \
    --output "$OUTPUT_DIR" \
    --samples-per-split 5 \
    --seed 42

echo
echo "Main audit complete."
echo

# ------------------------------------------------------------
# 2. One-batch smoke test
# ------------------------------------------------------------
echo "============================================================"
echo "2/3 Running Stage 1 smoke test"
echo "============================================================"

python scripts/verification/smoke_test_stage1.py \
    --data-root "$DATASET_ROOT" \
    --output "$OUTPUT_DIR/smoke_test.json" \
    --seed 42 \
    --no-pretrained

echo
echo "Smoke test complete."
echo

# ------------------------------------------------------------
# 3. Short loss-scale check
# ------------------------------------------------------------
echo "============================================================"
echo "3/3 Running Stage 1 loss-scale check"
echo "============================================================"

python scripts/verification/loss_scale_check.py \
    --data-root "$DATASET_ROOT" \
    --output "$OUTPUT_DIR/loss_scale_check.csv" \
    --seed 42 \
    --ef-weight 0.5 \
    --seg-weight 0.5 \
    --batches "${AUDIT_LOSS_BATCHES:-10}" \
    --num-workers "${AUDIT_NUM_WORKERS:-0}" \
    --no-pretrained

echo
echo "Loss-scale check complete."
echo

echo "============================================================"
echo " Stage 1 audit finished successfully"
echo "============================================================"
echo
echo "Outputs are in:"
echo "  $OUTPUT_DIR"
echo
echo "Zip that directory and upload it for review."
