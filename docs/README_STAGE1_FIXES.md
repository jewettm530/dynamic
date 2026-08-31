# Historical Stage 1 Fix Bundle — Superseded by the Corrected Video Protocol

> **Historical document. Do not use this file as the current Stage 1 run guide.**  
> Current protocol: `docs/STAGE1_CORRECTED_IMPLEMENTATION.md`  
> Current root workflow: `README.md`

This document records an **intermediate Stage 1 correction effort** that occurred before the professor identified the central EF-input problem.

## What this intermediate work fixed

The earlier code audit addressed issues such as:

- consistency of ED/ES trace handling;
- reproducibility utilities;
- dedicated Stage 1 training scripts;
- metric/checkpoint behavior;
- auditing data alignment and split behavior;
- attempts to make EF-only, segmentation-only, and multi-task experiments more controlled.

Those changes were useful, but the resulting controlled multi-task EF setup still used the expert-selected ED and ES frames as two independent 2D EF inputs.

## Why this protocol was superseded

The professor's later review identified that the controlled implementation:

```text
ED frame ─► EF scalar ─┐
                       ├─► average EF
ES frame ─► EF scalar ─┘
```

was not a true video EF model. It did not learn a temporal representation of cardiac motion and required expert ED/ES phase information at EF inference.

The completed controlled results were therefore retained as **T0 — two-frame oracle diagnostic**, while the main Stage 1 experiment was redesigned.

## Current corrected design

The current protocol uses:

```text
B1: multi-frame video → R(2+1)D-18 → temporal/spatial pooling → one EF

B3: same video EF pathway
    + sparse ED/ES segmentation supervision through the shared encoder
```

The EF inference path requires video only and does not load ground-truth ED/ES information.

## Current implementation files

```text
echonet/datasets/stage1_video.py
echonet/modeling/stage1_video_multitask.py

echonet/utils/stage1_corrected.py
echonet/utils/stage1_evaluation.py
echonet/utils/stage1_metrics.py

scripts/training/train_stage1_b1_video_ef.py
scripts/training/train_stage1_b2_segmentation.py
scripts/training/train_stage1_b3_video_multitask.py
scripts/training/run_stage1_corrected_training.sh

scripts/evaluation/evaluate_stage1_corrected_ef.py
scripts/evaluation/evaluate_stage1_corrected_segmentation.py
scripts/evaluation/run_stage1_corrected_final.sh

scripts/verification/smoke_test_stage1_corrected.py
scripts/verification/verify_video_only_inference.py
scripts/verification/verify_stage1_corrected_runs.py
```

## Current audit

Use:

```bash
bash RUN_AUDIT.sh \
    /data/jewettm/dynamic/datasets \
    outputs/stage1_corrected_audit
```

The current audit verifies the corrected architecture and the absence of oracle ED/ES information in the EF inference path.

## Current training

Use:

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets
```

This launches B1, B2, and all B3 weights for seeds 42, 2026, and 3407, then selects a B3 weight using validation EF MAE only.

## Historical file locations

The completed controlled two-frame experiment is preserved under:

```text
configs/t0_two_frame_oracle/
results/stage1/T0_two_frame_oracle/
output/stage1/T0_two_frame_oracle/   # local runtime artifacts if retained
```

Earlier unmatched comparisons are preserved under:

```text
configs/archive_pre_correction/
results/stage1/archive_pre_correction/
output/stage1/archive_pre_correction/   # local runtime artifacts if retained
```

These files are kept for provenance and should not be used to populate the corrected B1/B2/B3 tables.
