# Stage 1 Correction — File and Path Change Log

This project copy has been reorganized and updated for the corrected Stage 1 task:
video-based continuous EF regression with sparse ED/ES LV segmentation supervision.

## Main implementation changes

### New core code

- `echonet/datasets/stage1_video.py`
  - video-only EF loading does not require or open `VolumeTracings.csv`
  - random training clips and deterministic center validation/test clips
  - sparse ED/ES frame/mask loading is enabled only for segmentation
- `echonet/modeling/stage1_video_multitask.py`
  - shared R(2+1)D-18 encoder
  - one video-level EF output after spatiotemporal pooling
  - segmentation decoder trained from expert ED/ES frames
- `echonet/utils/stage1_corrected.py`
- `echonet/utils/stage1_evaluation.py`

### Modified existing dataset code

- `echonet/datasets/echo.py`
  - no longer opens `VolumeTracings.csv` unless a requested target actually needs tracing information
  - supports deterministic center-clip sampling

### Corrected training scripts

- `scripts/training/train_stage1_b1_video_ef.py`
- `scripts/training/train_stage1_b2_segmentation.py`
- `scripts/training/train_stage1_b3_video_multitask.py`
- `scripts/training/run_stage1_corrected_training.sh`

B1 and B3 use the same 32-frame/period-2 video input, R(2+1)D-18 encoder,
spatiotemporal aggregation, preprocessing, optimizer, learning rate, epoch budget,
batch size, split, scheduler, and validation-EF-MAE checkpoint rule.

Because corrected B3 uses a shared R(2+1)D encoder for segmentation rather than
the old DeepLabV3 segmentation architecture, the previous B2 result does **not**
satisfy the professor's unchanged-model reuse rule. Corrected B2 is therefore
implemented as a new 3-seed run.

### Validation selection and final evaluation

- `scripts/analysis/summarize_stage1_corrected_validation.py`
  - W1/W2/W3 selection from validation EF MAE only
  - sample SD uses `ddof=1`
- `scripts/evaluation/evaluate_stage1_corrected_ef.py`
  - B1/B3 EF validation/test path uses video only and does not open tracings
- `scripts/evaluation/evaluate_stage1_corrected_segmentation.py`
  - B2/B3 labeled-frame segmentation evaluation
  - B3 uses the same checkpoint selected by validation EF MAE
- `scripts/evaluation/run_stage1_corrected_final.sh`
- `scripts/analysis/summarize_stage1_corrected_final.py`

### Verification/reproducibility

- `RUN_AUDIT.sh` — corrected Stage 1 audit
- `RUN_T0_AUDIT.sh` — archived historical audit
- `scripts/verification/smoke_test_stage1_corrected.py`
- `scripts/verification/verify_video_only_inference.py`
- `scripts/verification/verify_stage1_corrected_runs.py`
- `scripts/analysis/collect_stage1_reproducibility.py`
- `.gitignore` — protects datasets, checkpoints, local output, tracings, and patient-derived artifacts

## New configuration paths

Corrected configurations:

```text
configs/stage1_corrected/
├── common.yaml
├── B1_video_ef.yaml
├── B2_segmentation.yaml
├── B3_W1.yaml
├── B3_W2.yaml
└── B3_W3.yaml
```

Historical T0 configurations were moved to:

```text
configs/t0_two_frame_oracle/
```

Older unmatched/pre-correction single-task configs were moved to:

```text
configs/archive_pre_correction/
```

## Result path changes already applied in this project copy

Historical two-frame results:

```text
results/stage1/T0_two_frame_oracle/
├── weighting/
└── final_controlled/
```

Older unmatched comparison:

```text
results/stage1/archive_pre_correction/
└── unmatched_comparison/
```

All new corrected results belong under:

```text
results/stage1/corrected/
```

## Local output/checkpoint paths to move in the full working repository

The uploaded zip intentionally omitted `output/`, so those local directories
could not be moved inside this returned project copy. Run this once from the
root of the full repository:

```bash
bash scripts/migration_stage1_correction_paths.sh
```

The helper moves old local folders, if present, to:

```text
output/stage1/T0_two_frame_oracle/
├── weighting/
├── final_baselines_controlled/
└── final_evaluation_controlled/

output/stage1/archive_pre_correction/
├── final_baselines/
└── final_evaluation/
```

New B1/B2/B3 checkpoints and local training state go to:

```text
output/stage1/corrected/
├── B1_video_ef/
├── B2_segmentation/
└── B3_video_mtl/
    ├── W1/
    ├── W2/
    └── W3/
```

## Run order

```bash
# 1. After merging this corrected copy into the full repository, organize old paths.
bash scripts/migration_stage1_correction_paths.sh

# 2. Verify the corrected model and video-only inference path.
bash RUN_AUDIT.sh /data/jewettm/dynamic/datasets outputs/stage1_corrected_audit

# 3. Train B1 (3), B2 (3), and B3 (9) and lock the B3 weight using validation only.
bash scripts/training/run_stage1_corrected_training.sh /data/jewettm/dynamic/datasets

# 4. Only after the weight is locked, run final validation/test evaluation.
bash scripts/evaluation/run_stage1_corrected_final.sh /data/jewettm/dynamic/datasets

# 5. Copy only safe reproducibility artifacts into tracked results/.
python scripts/analysis/collect_stage1_reproducibility.py
```

Do not commit raw videos, extracted frames, masks, tracing files/coordinates,
checkpoints, or patient-derived visualizations.
