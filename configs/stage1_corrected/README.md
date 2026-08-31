# Corrected Stage 1 Configurations

This directory documents the professor-requested **B1/B2/B3 corrected Stage 1 protocol**.

The key correction is that EF must be predicted from a **multi-frame video clip**, not from the expert-selected ED/ES frames. ED/ES labels are reserved for sparse segmentation supervision and segmentation evaluation.

## Files

| File | Purpose |
|---|---|
| `common.yaml` | Shared settings that define the corrected protocol |
| `B1_video_ef.yaml` | Video EF-only baseline |
| `B2_segmentation.yaml` | Segmentation-only baseline |
| `B3_W1.yaml` | B3 with `0.1 L_EF + 0.9 L_seg` |
| `B3_W2.yaml` | B3 with `0.5 L_EF + 0.5 L_seg` |
| `B3_W3.yaml` | B3 with `0.9 L_EF + 0.1 L_seg` |

## Important implementation note

These YAML files are **protocol/configuration records**, not the runtime configuration loader for the training scripts. The current runners pass the same settings through command-line arguments/environment variables and each run writes its effective configuration to `run_config.json`.

For the actual experiment, the runtime source of truth is:

```text
scripts/training/run_stage1_corrected_training.sh
scripts/training/train_stage1_b1_video_ef.py
scripts/training/train_stage1_b2_segmentation.py
scripts/training/train_stage1_b3_video_multitask.py
```

The YAML files should remain synchronized with those scripts so the documented protocol matches the executed protocol.

## Common corrected settings

`common.yaml` records the settings that should remain fixed across the corrected comparison:

- seeds: `42`, `2026`, `3407`
- EF input: 32-frame video clip
- sampling period: 2
- training clip sampling: random valid start
- validation/test clip sampling: deterministic center clip
- encoder: pretrained R(2+1)D-18
- EF aggregation: `AdaptiveAvgPool3d` before one scalar output
- segmentation decoder: FPN-style decoder from shared R(2+1)D features
- segmentation decoder width: 128
- epochs: 45
- batch size: 4
- optimizer: SGD
- learning rate: `1e-4`
- momentum: `0.9`
- weight decay: `1e-4`
- LR scheduler: StepLR
- LR step period: 15
- EF loss: mean MSE
- segmentation loss: mean BCEWithLogitsLoss

## B1 vs. B3 matching rule

B1 and B3 must remain identical for the EF pathway and training conditions. The intended difference is only that B3 also receives segmentation supervision through the shared encoder.

Do not independently change the following for one model:

- video input length/period
- clip sampling
- encoder
- temporal aggregation
- preprocessing
- optimizer or optimizer hyperparameters
- epoch budget
- batch size
- split
- checkpoint rule

If a resource-related setting must change, restart the relevant corrected comparison with the **same setting applied consistently**.

## B2 rerun rationale

The previous segmentation-only baseline used a different segmentation architecture. The corrected B2 uses the same shared R(2+1)D encoder/segmentation decoder family as B3, so previous B2 results are not reused.

B2 checkpoint selection uses **highest validation Mean Dice**.

## B3 weight selection

B3 is trained at all three weights and all three seeds:

```text
W1 × 3 seeds
W2 × 3 seeds
W3 × 3 seeds
```

The final weight is selected using **mean validation EF MAE only**. The test set is not used to choose the loss weight.

The lock is written to:

```text
results/stage1/corrected/validation_weight_selection/LOCKED_WEIGHT.txt
```

## Running the documented protocol

From the repository root:

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets
```

The runner accepts environment overrides such as `BATCH_SIZE`, `EPOCHS`, `NUM_WORKERS`, `FRAMES`, `PERIOD`, and optimizer settings. For the official corrected experiment, leave the documented defaults unchanged unless a necessary protocol-wide correction is made and recorded.

## Reproducibility

Each run writes an effective `run_config.json`. Before final test evaluation, `verify_stage1_corrected_runs.py` checks the required B1/B3 fields for exact agreement.

For the full workflow, see:

```text
README.md
docs/STAGE1_CORRECTED_IMPLEMENTATION.md
results/stage1/corrected/README.md
```
