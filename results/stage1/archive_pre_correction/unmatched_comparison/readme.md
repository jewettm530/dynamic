# Historical Unmatched Stage 1 Comparison

> **Status:** superseded; retained only for provenance.

This experiment was not sufficiently controlled to serve as the final Stage 1 comparison. Multiple task-related and training-related settings changed simultaneously across EF-only, segmentation-only, and multi-task models.

## Configuration differences in this historical comparison

| Setting | EF only | Segmentation only | Multi-task W2 |
|---|---|---|---|
| Architecture | R(2+1)D-18 | DeepLabV3-ResNet50 | MultiTask DeepLabV3 |
| EF/input type | 32-frame video clip | ED + ES frames | ED + ES frames |
| Epochs | 45 | 50 | 25 |
| Batch size | 20 | 20 | 4 |
| Optimizer | SGD | SGD | Adam |
| Learning rate | `1e-4` | `1e-5` | `1e-4` |
| Weight decay | `1e-4` | `1e-5` | 0 |
| EF representation | Temporal video | — | Independent ED/ES shared features |

Because several variables changed at once, these results cannot isolate the effect of segmentation supervision.

## What happened next

An intermediate controlled experiment was created with a common 2D DeepLab/ResNet-50 setup. That improved experimental control, but EF was still generated from the expert-selected ED/ES frames and therefore still used oracle phase information.

That controlled two-frame experiment is now preserved as:

```text
results/stage1/T0_two_frame_oracle/
```

## Final corrected approach

The professor-requested correction moved the main EF task back to a genuine video setting while preserving a controlled B1/B3 comparison:

```text
B1: multi-frame video → shared R(2+1)D EF pathway → one EF
B3: same video EF pathway + sparse ED/ES segmentation supervision
```

B1 and B3 are matched on the EF input, encoder, temporal aggregation, preprocessing, optimizer, learning rate, epoch budget, split, batch size, scheduler, and checkpoint rule.

The corrected implementation and results live under:

```text
configs/stage1_corrected/
output/stage1/corrected/       # local runtime/checkpoints
results/stage1/corrected/      # safe tracked results
```

Do not use the files in this directory to populate the corrected Stage 1 result tables.
