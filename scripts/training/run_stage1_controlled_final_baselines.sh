#!/usr/bin/env bash
# HISTORICAL T0 ONLY. Use run_stage1_corrected_training.sh.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || { echo "ERROR: run inside the EchoNet repo"; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -z "${STEP3_GPU:-}" ]]; then
  echo "ERROR: set STEP3_GPU explicitly."
  echo "Example: STEP3_GPU=3 bash scripts/training/run_stage1_controlled_final_baselines.sh"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$STEP3_GPU"

STATUS="$(git status --porcelain --untracked-files=normal)"
[[ -z "$STATUS" ]] || { echo "ERROR: Git working tree is not clean:"; echo "$STATUS"; exit 1; }

COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
TRAIN="scripts/training/train_stage1_controlled_single_task.py"
VERIFY="scripts/verification/verify_stage1_controlled_baselines.py"
W2="output/stage1/T0_two_frame_oracle/weighting/W2"
OUT="output/stage1/T0_two_frame_oracle/final_baselines_controlled"
FORCE_RERUN="${FORCE_RERUN:-0}"

echo "======================================================================"
echo "Stage 1 Step 3 — controlled single-task ablations"
echo "Branch: $BRANCH"
echo "Commit: $COMMIT"
echo "Physical GPU: $STEP3_GPU"
echo "======================================================================"

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"ERROR: expected one visible CUDA device, got {torch.cuda.device_count()}")
print("Logical cuda:0:", torch.cuda.get_device_name(0))
free,total=torch.cuda.mem_get_info(0)
print(f"Free GPU memory: {free/1024**3:.2f}/{total/1024**3:.2f} GiB")
x=torch.randn(64,64,device="cuda")
_ = x @ x
torch.cuda.synchronize()
print("CUDA computation: PASS")
PY

for seed in 42 2026 3407; do
  for f in best.pt run_summary.json best_validation_metrics.json run_config.json; do
    [[ -f "$W2/seed_${seed}/$f" ]] || {
      echo "ERROR: missing W2 artifact $W2/seed_${seed}/$f"
      exit 1
    }
  done
done

echo "Verifying controlled model initialization against W2 architecture..."
python "$VERIFY"

mkdir -p "$OUT"

run_one() {
  local task="$1"
  local seed="$2"
  local out="$3"
  local cfg

  if [[ "$task" == "ef" ]]; then
    cfg="configs/t0_two_frame_oracle/ef_only_controlled.yaml"
  else
    cfg="configs/t0_two_frame_oracle/segmentation_only_controlled.yaml"
  fi

  if [[ -f "$out/run_summary.json" && "$FORCE_RERUN" != "1" ]]; then
    echo "SKIP completed: $task seed $seed"
    return
  fi

  if [[ -e "$out" ]]; then
    if [[ "$FORCE_RERUN" == "1" ]]; then
      rm -rf "$out"
    else
      mv "$out" "${out}_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
  fi

  mkdir -p "$out"
  cp "$cfg" "$out/config.yaml"

  {
    echo "git_branch=$BRANCH"
    echo "git_commit=$COMMIT"
    echo "physical_gpu=$STEP3_GPU"
    echo "task=$task"
    echo "seed=$seed"
  } > "$out/run_identity.txt"

  set +e
  python "$TRAIN"     --task "$task"     --data-root /data/jewettm/dynamic/datasets     --output "$out"     --seed "$seed"     --epochs 25     --batch-size 4     --num-workers 4     --learning-rate 0.0001     --weight-decay 0.0     --regression-hidden-dim 256     --dropout 0.3     2>&1 | tee "$out/console.log"
  rc=${PIPESTATUS[0]}
  set -e

  echo "Exit code: $rc" > "$out/runner_status.txt"
  [[ $rc -eq 0 ]] || exit $rc

  for required in best.pt run_summary.json best_validation_metrics.json training_history.csv run_config.json; do
    [[ -f "$out/$required" ]] || {
      echo "ERROR: expected output missing: $out/$required"
      exit 1
    }
  done
}

for seed in 42 2026 3407; do
  run_one ef "$seed" "$OUT/ef_only/seed_${seed}"
done

for seed in 42 2026 3407; do
  run_one seg "$seed" "$OUT/segmentation_only/seed_${seed}"
done

echo
echo "6/6 controlled single-task runs complete."
echo "W2 was reused unchanged; no W2 retraining was performed."
echo "Next:"
echo "  python scripts/evaluation/evaluate_stage1_controlled_final.py"
