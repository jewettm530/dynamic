#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository root from this script's location:
# dynamic/scripts/training/run_experiments.sh -> dynamic/
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Central output locations
VIDEO_OUTPUT_DIR="$PROJECT_ROOT/outputs/video"
SEGMENTATION_OUTPUT_DIR="$PROJECT_ROOT/outputs/segmentation"
LOG_DIR="$PROJECT_ROOT/outputs/logs"

# Experiment-specific output locations
VIDEO_SWEEP_DIR="$VIDEO_OUTPUT_DIR/hyperparameter_sweep"
VIDEO_TEST_DIR="$VIDEO_OUTPUT_DIR/test_evaluation"
VIDEO_TRAINING_SIZE_DIR="$VIDEO_OUTPUT_DIR/training_size"

SEGMENTATION_BASELINE_DIR="$SEGMENTATION_OUTPUT_DIR/baseline"
SEGMENTATION_TRAINING_SIZE_DIR="$SEGMENTATION_OUTPUT_DIR/training_size"

mkdir -p \
    "$VIDEO_SWEEP_DIR" \
    "$VIDEO_TEST_DIR" \
    "$VIDEO_TRAINING_SIZE_DIR" \
    "$SEGMENTATION_BASELINE_DIR" \
    "$SEGMENTATION_TRAINING_SIZE_DIR" \
    "$LOG_DIR"

cd "$PROJECT_ROOT"

# Make sure the local echonet package is installed and importable.
python3 -c "import echonet" || {
    echo "echonet module not found."
    echo "Run: pip install -e ."
    exit 1
}

echo "Project root: $PROJECT_ROOT"
echo "Video outputs: $VIDEO_OUTPUT_DIR"
echo "Segmentation outputs: $SEGMENTATION_OUTPUT_DIR"
echo "Logs: $LOG_DIR"

# ----------------------------------------------------------------------
# 1. Video-model frame-count sweep
# ----------------------------------------------------------------------

for pretrained in True False; do
    initialization=$(
        if [[ "$pretrained" == "True" ]]; then
            echo "pretrained"
        else
            echo "random"
        fi
    )

    for model in r2plus1d_18 r3d_18 mc3_18; do
        for frames in 96 64 32 16 8 4 1; do
            batch=$((256 / frames))
            batch=$((batch > 16 ? 16 : batch))

            if [[ "$batch" -lt 1 ]]; then
                batch=1
            fi

            run_output="$VIDEO_SWEEP_DIR/frame_count/${model}/${initialization}/frames_${frames}"
            mkdir -p "$run_output"

            cmd="import echonet; echonet.utils.video.run(
                modelname=\"${model}\",
                frames=${frames},
                period=1,
                pretrained=${pretrained},
                batch_size=${batch},
                output=\"${run_output}\"
            )"

            echo "Running frame-count experiment:"
            echo "  Model: $model"
            echo "  Initialization: $initialization"
            echo "  Frames: $frames"
            echo "  Batch size: $batch"
            echo "  Output: $run_output"

            python3 -c "$cmd" 2>&1 |
                tee "$LOG_DIR/video_${model}_${initialization}_frames_${frames}.log"
        done
    done
done

# ----------------------------------------------------------------------
# 2. Video-model sampling-period sweep
# ----------------------------------------------------------------------

for pretrained in True False; do
    initialization=$(
        if [[ "$pretrained" == "True" ]]; then
            echo "pretrained"
        else
            echo "random"
        fi
    )

    for model in r2plus1d_18 r3d_18 mc3_18; do
        for period in 2 4 6 8; do
            frames=$((64 / period))
            batch=$((256 / 64 * period))
            batch=$((batch > 16 ? 16 : batch))

            if [[ "$batch" -lt 1 ]]; then
                batch=1
            fi

            run_output="$VIDEO_SWEEP_DIR/period/${model}/${initialization}/period_${period}"
            mkdir -p "$run_output"

            cmd="import echonet; echonet.utils.video.run(
                modelname=\"${model}\",
                frames=${frames},
                period=${period},
                pretrained=${pretrained},
                batch_size=${batch},
                output=\"${run_output}\"
            )"

            echo "Running period experiment:"
            echo "  Model: $model"
            echo "  Initialization: $initialization"
            echo "  Frames: $frames"
            echo "  Period: $period"
            echo "  Batch size: $batch"
            echo "  Output: $run_output"

            python3 -c "$cmd" 2>&1 |
                tee "$LOG_DIR/video_${model}_${initialization}_period_${period}.log"
        done
    done
done

# ----------------------------------------------------------------------
# 3. Final video-model test evaluation
# ----------------------------------------------------------------------

period=2
pretrained=True
frames=$((64 / period))

for model in r2plus1d_18 r3d_18 mc3_18; do
    run_output="$VIDEO_TEST_DIR/${model}_pretrained"
    mkdir -p "$run_output"

    cmd="import echonet; echonet.utils.video.run(
        modelname=\"${model}\",
        frames=${frames},
        period=${period},
        pretrained=${pretrained},
        run_test=True,
        output=\"${run_output}\"
    )"

    echo "Running final video-model evaluation:"
    echo "  Model: $model"
    echo "  Output: $run_output"

    python3 -c "$cmd" 2>&1 |
        tee "$LOG_DIR/video_${model}_final_test.log"
done

# ----------------------------------------------------------------------
# 4. Baseline segmentation experiment
# ----------------------------------------------------------------------

segmentation_output="$SEGMENTATION_BASELINE_DIR/deeplabv3_resnet50_random"
mkdir -p "$segmentation_output"

cmd="import echonet; echonet.utils.segmentation.run(
    modelname=\"deeplabv3_resnet50\",
    save_segmentation=True,
    pretrained=False,
    output=\"${segmentation_output}\"
)"

echo "Running baseline segmentation experiment:"
echo "  Model: deeplabv3_resnet50"
echo "  Initialization: random"
echo "  Output: $segmentation_output"

python3 -c "$cmd" 2>&1 |
    tee "$LOG_DIR/segmentation_deeplabv3_resnet50_random.log"

# ----------------------------------------------------------------------
# 5. Training-size experiments
# ----------------------------------------------------------------------

pretrained=True
model="r2plus1d_18"
period=2
frames=$((64 / period))

batch=$((256 / 64 * period))
batch=$((batch > 16 ? 16 : batch))

if [[ "$batch" -lt 1 ]]; then
    batch=1
fi

for patients in 16 32 64 128 256 512 1024 2048 4096 7460; do
    epochs=$((50 * (8192 / patients)))

    if [[ "$epochs" -lt 1 ]]; then
        epochs=1
    fi

    if [[ "$epochs" -gt 200 ]]; then
        epochs=200
    fi

    video_run_output="$VIDEO_TRAINING_SIZE_DIR/${patients}_patients"
    segmentation_run_output="$SEGMENTATION_TRAINING_SIZE_DIR/${patients}_patients"

    mkdir -p "$video_run_output"
    mkdir -p "$segmentation_run_output"

    video_cmd="import echonet; echonet.utils.video.run(
        modelname=\"${model}\",
        frames=${frames},
        period=${period},
        pretrained=${pretrained},
        batch_size=${batch},
        num_epochs=${epochs},
        output=\"${video_run_output}\",
        n_train_patients=${patients}
    )"

    echo "Running video training-size experiment:"
    echo "  Patients: $patients"
    echo "  Epochs: $epochs"
    echo "  Output: $video_run_output"

    python3 -c "$video_cmd" 2>&1 |
        tee "$LOG_DIR/video_training_size_${patients}.log"

    segmentation_cmd="import echonet; echonet.utils.segmentation.run(
        modelname=\"deeplabv3_resnet50\",
        pretrained=False,
        num_epochs=${epochs},
        output=\"${segmentation_run_output}\",
        n_train_patients=${patients}
    )"

    echo "Running segmentation training-size experiment:"
    echo "  Patients: $patients"
    echo "  Epochs: $epochs"
    echo "  Output: $segmentation_run_output"

    python3 -c "$segmentation_cmd" 2>&1 |
        tee "$LOG_DIR/segmentation_training_size_${patients}.log"
done

echo "All experiments completed successfully."