#!/usr/bin/env bash
# HISTORICAL T0 ONLY. Use run_stage1_corrected_training.sh.
set -euo pipefail

# ==============================================================================
# EchoNet-Dynamic Stage 1: Multi-Task Loss-Weighting Experiment
#
# Runs the nine official Step 2 experiments:
#   W1 = 0.1 * L_EF + 0.9 * L_seg  x seeds 42, 2026, 3407
#   W2 = 0.5 * L_EF + 0.5 * L_seg  x seeds 42, 2026, 3407
#   W3 = 0.9 * L_EF + 0.1 * L_seg  x seeds 42, 2026, 3407
#
# IMPORTANT:
#   - Sequential execution on one visible RTX 3090.
#   - Validation only; the test set is never used by train_multitask.py.
#   - The script refuses to begin if Git has uncommitted tracked/untracked
#     changes (other than ignored files).
#   - Completed runs are skipped unless FORCE_RERUN=1 is set.
#
# Usage:
#   cd /data/jewettm/dynamic
#   bash scripts/training/run_stage1_weighting.sh
#
# Optional:
#   FORCE_RERUN=1 bash scripts/training/run_stage1_weighting.sh
# ==============================================================================

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "ERROR: This script must be run inside the EchoNet Git repository."
    exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Lock official Stage 1 weighting runs to one physical GPU.
export CUDA_VISIBLE_DEVICES=0

CONFIG_DIR="${PROJECT_ROOT}/configs/t0_two_frame_oracle"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/training/train_multitask.py"
FORCE_RERUN="${FORCE_RERUN:-0}"

CONFIGS=(
    "${CONFIG_DIR}/multitask_w1.yaml"
    "${CONFIG_DIR}/multitask_w2.yaml"
    "${CONFIG_DIR}/multitask_w3.yaml"
)

echo "=============================================================================="
echo " EchoNet-Dynamic Stage 1 — Multi-Task Loss-Weighting Experiment"
echo "=============================================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Python:       $(command -v python)"
echo "GPU env:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo

# ------------------------------------------------------------------------------
# Preflight: required files and dependencies
# ------------------------------------------------------------------------------

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Missing training script: ${TRAIN_SCRIPT}"
    exit 1
fi

for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: Missing config: ${cfg}"
        exit 1
    fi
done

python - <<'PY'
try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "ERROR: PyYAML is required. Install it in the echonet environment with: "
        "python -m pip install PyYAML"
    ) from exc
print("PyYAML import: PASS")
PY

# ------------------------------------------------------------------------------
# Preflight: Git must be clean.
# Ignored output files do not count as changes.
# ------------------------------------------------------------------------------

GIT_STATUS="$(git status --porcelain --untracked-files=normal)"
if [[ -n "${GIT_STATUS}" ]]; then
    echo
    echo "ERROR: Git working tree is not clean."
    echo "Commit/stash/remove the following changes before official Stage 1 runs:"
    echo
    echo "${GIT_STATUS}"
    echo
    exit 1
fi

GIT_BRANCH="$(git branch --show-current)"
GIT_COMMIT="$(git rev-parse HEAD)"

echo "Git branch:   ${GIT_BRANCH}"
echo "Git commit:   ${GIT_COMMIT}"
echo "Git status:   clean"
echo

# ------------------------------------------------------------------------------
# Preflight: verify CUDA on the single visible RTX 3090.
# ------------------------------------------------------------------------------

python - <<'PY'
import os
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("Visible CUDA devices:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available.")

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"ERROR: Expected exactly 1 visible CUDA device, got {torch.cuda.device_count()}."
    )

name = torch.cuda.get_device_name(0)
print("GPU:", name)

if "RTX 3090" not in name:
    raise SystemExit(f"ERROR: Expected an RTX 3090, got: {name}")

# Actual CUDA computation, not only device discovery.
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA computation test: PASS")
PY

echo

# ------------------------------------------------------------------------------
# Preflight: verify W1/W2/W3 are identical except for the two intended weights.
# Also verifies seeds and the exact requested weighting values.
# ------------------------------------------------------------------------------

python - "${CONFIGS[@]}" <<'PY'
import copy
import math
import sys
from pathlib import Path

import yaml

paths = [Path(p) for p in sys.argv[1:]]
configs = []
for path in paths:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    configs.append((path, cfg))

expected = {
    "W1": (0.1, 0.9),
    "W2": (0.5, 0.5),
    "W3": (0.9, 0.1),
}
expected_seeds = [42, 2026, 3407]

def without_intended_differences(cfg):
    x = copy.deepcopy(cfg)
    # Weighting is the only experiment-dependent section.
    x.pop("weighting", None)
    return x

reference = without_intended_differences(configs[0][1])
for path, cfg in configs[1:]:
    if without_intended_differences(cfg) != reference:
        raise SystemExit(
            f"ERROR: {path} differs from W1 in settings other than loss weights."
        )

seen = set()
for path, cfg in configs:
    name = str(cfg["weighting"]["name"]).upper()
    efw = float(cfg["weighting"]["ef_weight"])
    segw = float(cfg["weighting"]["seg_weight"])
    seeds = list(cfg["training"]["seeds"])

    if name not in expected:
        raise SystemExit(f"ERROR: Unexpected weighting name {name} in {path}.")
    if name in seen:
        raise SystemExit(f"ERROR: Duplicate weighting name {name}.")
    seen.add(name)

    exp_ef, exp_seg = expected[name]
    if not math.isclose(efw, exp_ef, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"ERROR: {name} EF weight should be {exp_ef}, got {efw}.")
    if not math.isclose(segw, exp_seg, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"ERROR: {name} seg weight should be {exp_seg}, got {segw}.")
    if not math.isclose(efw + segw, 1.0, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"ERROR: {name} weights do not sum to 1.")
    if seeds != expected_seeds:
        raise SystemExit(
            f"ERROR: {name} seeds must be {expected_seeds}, got {seeds}."
        )

if seen != set(expected):
    raise SystemExit(f"ERROR: Expected {set(expected)}, found {seen}.")

print("Configuration consistency check: PASS")
print("Only ef_weight and seg_weight differ across W1/W2/W3.")
PY

echo

# ------------------------------------------------------------------------------
# Helper: read one YAML value.
# ------------------------------------------------------------------------------

yaml_value() {
    local config="$1"
    local expression="$2"

    python - "$config" "$expression" <<'PY'
import sys
import yaml

path = sys.argv[1]
expression = sys.argv[2]

with open(path) as f:
    value = yaml.safe_load(f)

for key in expression.split("."):
    value = value[key]

if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

# ------------------------------------------------------------------------------
# Create one experiment-level manifest before running anything.
# ------------------------------------------------------------------------------

OUTPUT_ROOT="$(yaml_value "${CONFIGS[0]}" "output.root")"
DATA_ROOT="$(yaml_value "${CONFIGS[0]}" "data.root")"
mkdir -p "${OUTPUT_ROOT}"

python - "${OUTPUT_ROOT}/experiment_manifest.json" "${GIT_BRANCH}" "${GIT_COMMIT}" <<'PY'
import datetime
import json
import os
import platform
import subprocess
import sys

import torch
import torchvision
import yaml

out_path, branch, commit = sys.argv[1:4]

manifest = {
    "experiment": "EchoNet-Dynamic Stage 1 multi-task loss weighting",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "git_branch": branch,
    "git_commit": commit,
    "git_status": subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip(),
    "python": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ],
    "weights": {
        "W1": {"ef_weight": 0.1, "seg_weight": 0.9},
        "W2": {"ef_weight": 0.5, "seg_weight": 0.5},
        "W3": {"ef_weight": 0.9, "seg_weight": 0.1},
    },
    "seeds": [42, 2026, 3407],
    "test_set_used": False,
}

with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Wrote experiment manifest: {out_path}")
PY

# ------------------------------------------------------------------------------
# Run W1, W2, W3 x seeds 42, 2026, 3407 sequentially.
# ------------------------------------------------------------------------------

TOTAL_RUNS=9
RUN_NUMBER=0

for CONFIG in "${CONFIGS[@]}"; do
    WEIGHT_NAME="$(yaml_value "${CONFIG}" "weighting.name")"
    EF_WEIGHT="$(yaml_value "${CONFIG}" "weighting.ef_weight")"
    SEG_WEIGHT="$(yaml_value "${CONFIG}" "weighting.seg_weight")"

    EPOCHS="$(yaml_value "${CONFIG}" "training.epochs")"
    BATCH_SIZE="$(yaml_value "${CONFIG}" "training.batch_size")"
    NUM_WORKERS="$(yaml_value "${CONFIG}" "training.num_workers")"
    LEARNING_RATE="$(yaml_value "${CONFIG}" "training.learning_rate")"
    WEIGHT_DECAY="$(yaml_value "${CONFIG}" "training.weight_decay")"

    REGRESSION_HIDDEN_DIM="$(yaml_value "${CONFIG}" "model.regression_hidden_dim")"
    DROPOUT="$(yaml_value "${CONFIG}" "model.dropout")"
    PRETRAINED="$(yaml_value "${CONFIG}" "model.pretrained")"
    DETERMINISTIC="$(yaml_value "${CONFIG}" "training.deterministic")"

    # Read the seed list as whitespace-separated integers.
    mapfile -t SEEDS < <(
        python - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for seed in cfg["training"]["seeds"]:
    print(seed)
PY
    )

    for SEED in "${SEEDS[@]}"; do
        RUN_NUMBER=$((RUN_NUMBER + 1))
        RUN_DIR="${OUTPUT_ROOT}/${WEIGHT_NAME}/seed_${SEED}"

        echo "=============================================================================="
        echo " Run ${RUN_NUMBER}/${TOTAL_RUNS}: ${WEIGHT_NAME}, seed ${SEED}"
        echo "=============================================================================="
        echo "EF weight:          ${EF_WEIGHT}"
        echo "Seg weight:         ${SEG_WEIGHT}"
        echo "Epochs:             ${EPOCHS}"
        echo "Batch size:         ${BATCH_SIZE}"
        echo "Learning rate:      ${LEARNING_RATE}"
        echo "Weight decay:       ${WEIGHT_DECAY}"
        echo "Regression hidden:  ${REGRESSION_HIDDEN_DIM}"
        echo "Dropout:             ${DROPOUT}"
        echo "Pretrained:          ${PRETRAINED}"
        echo "Deterministic:       ${DETERMINISTIC}"
        echo "Output:              ${RUN_DIR}"
        echo

        # Skip a successfully completed run unless explicitly told to rerun.
        if [[ -f "${RUN_DIR}/run_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
            echo "SKIP: ${RUN_DIR}/run_summary.json already exists."
            echo "Use FORCE_RERUN=1 to intentionally rerun completed experiments."
            echo
            continue
        fi

        if [[ -e "${RUN_DIR}" && "${FORCE_RERUN}" == "1" ]]; then
            echo "FORCE_RERUN=1: deleting existing run directory ${RUN_DIR}"
            rm -rf "${RUN_DIR}"
        elif [[ -e "${RUN_DIR}" ]]; then
            # An incomplete run exists. Preserve it instead of silently overwriting.
            TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
            BACKUP_DIR="${RUN_DIR}_incomplete_${TIMESTAMP}"
            echo "Incomplete run directory found."
            echo "Moving it to: ${BACKUP_DIR}"
            mv "${RUN_DIR}" "${BACKUP_DIR}"
        fi

        mkdir -p "${RUN_DIR}"

        # Preserve the exact YAML used with the run.
        cp "${CONFIG}" "${RUN_DIR}/config.yaml"

        CMD=(
            python "${TRAIN_SCRIPT}"
            --data-root "${DATA_ROOT}"
            --output "${RUN_DIR}"
            --seed "${SEED}"
            --ef-weight "${EF_WEIGHT}"
            --seg-weight "${SEG_WEIGHT}"
            --epochs "${EPOCHS}"
            --batch-size "${BATCH_SIZE}"
            --num-workers "${NUM_WORKERS}"
            --learning-rate "${LEARNING_RATE}"
            --weight-decay "${WEIGHT_DECAY}"
            --regression-hidden-dim "${REGRESSION_HIDDEN_DIM}"
            --dropout "${DROPOUT}"
        )

        if [[ "${PRETRAINED}" != "true" ]]; then
            CMD+=(--no-pretrained)
        fi

        if [[ "${DETERMINISTIC}" != "true" ]]; then
            CMD+=(--non-deterministic)
        fi

        # Record the exact executable command in a shell-replayable form.
        {
            echo "#!/usr/bin/env bash"
            echo "export CUDA_VISIBLE_DEVICES=0"
            printf "%q " "${CMD[@]}"
            echo
        } > "${RUN_DIR}/command.sh"
        chmod +x "${RUN_DIR}/command.sh"

        # Record immutable run identity before training starts.
        cat > "${RUN_DIR}/run_identity.txt" <<EOF
weight=${WEIGHT_NAME}
seed=${SEED}
ef_weight=${EF_WEIGHT}
seg_weight=${SEG_WEIGHT}
git_branch=${GIT_BRANCH}
git_commit=${GIT_COMMIT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

        START_TIME="$(date --iso-8601=seconds)"
        echo "Started: ${START_TIME}" | tee "${RUN_DIR}/runner_status.txt"

        # Pipe all stdout/stderr to both screen and a permanent console log.
        set +e
        "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/console.log"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        END_TIME="$(date --iso-8601=seconds)"
        {
            echo "Started: ${START_TIME}"
            echo "Ended: ${END_TIME}"
            echo "Exit code: ${EXIT_CODE}"
        } > "${RUN_DIR}/runner_status.txt"

        if [[ "${EXIT_CODE}" -ne 0 ]]; then
            echo
            echo "ERROR: ${WEIGHT_NAME} seed ${SEED} failed with exit code ${EXIT_CODE}."
            echo "See: ${RUN_DIR}/console.log"
            echo "The runner is stopping so the failure can be investigated."
            exit "${EXIT_CODE}"
        fi

        if [[ ! -f "${RUN_DIR}/run_summary.json" ]]; then
            echo "ERROR: Training exited successfully but run_summary.json is missing."
            exit 1
        fi

        if [[ ! -f "${RUN_DIR}/best_validation_metrics.json" ]]; then
            echo "ERROR: best_validation_metrics.json is missing."
            exit 1
        fi

        echo
        echo "COMPLETED: ${WEIGHT_NAME} seed ${SEED}"
        echo
    done
done

# ------------------------------------------------------------------------------
# Final completeness check
# ------------------------------------------------------------------------------

python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
weights = ["W1", "W2", "W3"]
seeds = [42, 2026, 3407]

missing = []
for weight in weights:
    for seed in seeds:
        run = root / weight / f"seed_{seed}"
        required = [
            "run_summary.json",
            "best_validation_metrics.json",
            "training_history.csv",
            "best.pt",
            "run_config.json",
            "config.yaml",
            "command.sh",
            "console.log",
        ]
        for name in required:
            if not (run / name).exists():
                missing.append(str(run / name))

if missing:
    print("ERROR: Step 2 is incomplete. Missing files:")
    for path in missing:
        print("  ", path)
    raise SystemExit(1)

print("All 9 Stage 1 weighting runs are complete.")
PY

echo
echo "=============================================================================="
echo " Step 2 training runs completed successfully"
echo "=============================================================================="
echo "Output root: ${OUTPUT_ROOT}"
echo
echo "IMPORTANT: Do not evaluate W1/W2/W3 on the test set."
echo "Next: summarize the three seed-level validation results for each weight,"
echo "calculate mean +/- sample SD, and lock one weight using validation only."
