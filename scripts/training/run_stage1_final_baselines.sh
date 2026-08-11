#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT}" ]]; then
  echo "ERROR: run inside the EchoNet Git repository."
  exit 1
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${STEP3_GPU:-0}"

EF_CONFIG="${PROJECT_ROOT}/configs/ef_only.yaml"
SEG_CONFIG="${PROJECT_ROOT}/configs/segmentation_only.yaml"
MTL_CONFIG="${PROJECT_ROOT}/configs/multitask_selected.yaml"
EF_SCRIPT="${PROJECT_ROOT}/scripts/training/train_ef_stage1.py"
SEG_SCRIPT="${PROJECT_ROOT}/scripts/training/train_segmentation_stage1.py"
FORCE_RERUN="${FORCE_RERUN:-0}"

for f in "$EF_CONFIG" "$SEG_CONFIG" "$MTL_CONFIG" "$EF_SCRIPT" "$SEG_SCRIPT"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

GIT_STATUS="$(git status --porcelain --untracked-files=normal)"
if [[ -n "$GIT_STATUS" ]]; then
  echo "ERROR: Git working tree is not clean. Commit/stash changes before official Step 3 runs:"
  echo "$GIT_STATUS"
  exit 1
fi
GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse HEAD)"

echo "Stage 1 Step 3 final baselines"
echo "Git branch: $GIT_BRANCH"
echo "Git commit: $GIT_COMMIT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"ERROR: expected one visible GPU, got {torch.cuda.device_count()}")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(128,128,device='cuda'); _ = x @ x; torch.cuda.synchronize()
print("CUDA computation: PASS")
PY

yaml_value() {
  python - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f: x=yaml.safe_load(f)
for key in sys.argv[2].split('.'):
    x=x[key]
if isinstance(x,bool): print('true' if x else 'false')
else: print(x)
PY
}

DATA_ROOT="$(yaml_value "$EF_CONFIG" data.root)"
EF_OUT="$(yaml_value "$EF_CONFIG" output.root)"
SEG_OUT="$(yaml_value "$SEG_CONFIG" output.root)"
MTL_SOURCE="$(yaml_value "$MTL_CONFIG" reuse.source_root)"
SEEDS=(42 2026 3407)

# Verify locked W2 checkpoints exist and are reusable.
for seed in "${SEEDS[@]}"; do
  for f in best.pt run_summary.json best_validation_metrics.json run_config.json; do
    [[ -f "${MTL_SOURCE}/seed_${seed}/${f}" ]] || {
      echo "ERROR: missing selected W2 artifact ${MTL_SOURCE}/seed_${seed}/${f}"; exit 1;
    }
  done
done

echo "Selected W2 checkpoints found for all three seeds; they will be reused, not retrained."

mkdir -p output/stage1/final_baselines
python - "output/stage1/final_baselines/experiment_manifest.json" "$GIT_BRANCH" "$GIT_COMMIT" "$MTL_SOURCE" <<'PY'
import datetime,json,os,sys,torch
p,branch,commit,mtl_source=sys.argv[1:5]
obj={
  'experiment':'EchoNet-Dynamic Stage 1 final baselines',
  'created_at':datetime.datetime.now().astimezone().isoformat(),
  'git_branch':branch,'git_commit':commit,
  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'seeds':[42,2026,3407],
  'cohort':{'train':7460,'validation':1288,'test':1276},
  'locked_multitask_weight':'W2',
  'multitask_formula':'0.5 * L_EF + 0.5 * L_seg',
  'multitask_reused_from':mtl_source,
  'test_used_for_selection':False,
}
with open(p,'w') as f: json.dump(obj,f,indent=2)
PY

run_one() {
  local task="$1" seed="$2" out="$3"; shift 3
  if [[ -f "$out/run_summary.json" && "$FORCE_RERUN" != "1" ]]; then
    echo "SKIP completed: $task seed $seed"
    return
  fi
  if [[ -e "$out" ]]; then
    if [[ "$FORCE_RERUN" == "1" ]]; then rm -rf "$out";
    else mv "$out" "${out}_incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  fi
  mkdir -p "$out"
  printf '%q ' "$@" > "$out/command.sh"; echo >> "$out/command.sh"
  chmod +x "$out/command.sh"
  echo "git_commit=$GIT_COMMIT" > "$out/run_identity.txt"
  echo "seed=$seed" >> "$out/run_identity.txt"
  echo "task=$task" >> "$out/run_identity.txt"
  set +e
  "$@" 2>&1 | tee "$out/console.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "Exit code: $rc" > "$out/runner_status.txt"
  [[ $rc -eq 0 ]] || exit $rc
  [[ -f "$out/run_summary.json" && -f "$out/best.pt" ]] || {
    echo "ERROR: expected completion files missing for $task seed $seed"; exit 1;
  }
}

# EF-only: exact current EF baseline settings, common traced-video cohort.
EF_EPOCHS="$(yaml_value "$EF_CONFIG" training.epochs)"
EF_FRAMES="$(yaml_value "$EF_CONFIG" training.frames)"
EF_PERIOD="$(yaml_value "$EF_CONFIG" training.period)"
EF_BATCH="$(yaml_value "$EF_CONFIG" training.batch_size)"
EF_WORKERS="$(yaml_value "$EF_CONFIG" training.num_workers)"
EF_LR="$(yaml_value "$EF_CONFIG" training.learning_rate)"
EF_WD="$(yaml_value "$EF_CONFIG" training.weight_decay)"
EF_STEP="$(yaml_value "$EF_CONFIG" training.lr_step_period)"
for seed in "${SEEDS[@]}"; do
  out="${EF_OUT}/seed_${seed}"
  run_one ef_only "$seed" "$out" \
    python "$EF_SCRIPT" --data-root "$DATA_ROOT" --output "$out" --seed "$seed" \
    --epochs "$EF_EPOCHS" --frames "$EF_FRAMES" --period "$EF_PERIOD" \
    --batch-size "$EF_BATCH" --num-workers "$EF_WORKERS" \
    --learning-rate "$EF_LR" --weight-decay "$EF_WD" --lr-step-period "$EF_STEP"
done

# Segmentation-only: exact current segmentation baseline settings.
SEG_EPOCHS="$(yaml_value "$SEG_CONFIG" training.epochs)"
SEG_BATCH="$(yaml_value "$SEG_CONFIG" training.batch_size)"
SEG_WORKERS="$(yaml_value "$SEG_CONFIG" training.num_workers)"
SEG_LR="$(yaml_value "$SEG_CONFIG" training.learning_rate)"
SEG_WD="$(yaml_value "$SEG_CONFIG" training.weight_decay)"
for seed in "${SEEDS[@]}"; do
  out="${SEG_OUT}/seed_${seed}"
  run_one segmentation_only "$seed" "$out" \
    python "$SEG_SCRIPT" --data-root "$DATA_ROOT" --output "$out" --seed "$seed" \
    --epochs "$SEG_EPOCHS" --batch-size "$SEG_BATCH" --num-workers "$SEG_WORKERS" \
    --learning-rate "$SEG_LR" --weight-decay "$SEG_WD"
done

echo "All six new Step 3 training runs completed."
echo "Next: python scripts/evaluation/evaluate_stage1_final.py"
