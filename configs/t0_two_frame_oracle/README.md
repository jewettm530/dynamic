# T0 Configuration Archive — Historical Two-Frame Oracle Diagnostic

This directory contains configurations associated with the **completed pre-correction controlled two-frame Stage 1 experiment**.

T0 is preserved for provenance and interpretation only. It is **not** the corrected video-based Stage 1 protocol and should not be used to launch new B1/B2/B3 runs.

## What T0 did

The T0 EF pathway used the expert-selected ED and ES frames:

```text
ED frame ─► 2D model ─► EF_ED ─┐
                                ├─► average ─► video-level EF
ES frame ─► 2D model ─► EF_ES ─┘
```

This setup demonstrated that segmentation supervision could improve EF regression in that controlled two-frame setting, but it had two important limitations:

1. EF did not learn from the full cardiac motion sequence.
2. EF inference depended on ground-truth ED/ES phase locations.

Because of those limitations, T0 is now treated as an **oracle diagnostic** rather than the main EF baseline.

## Why these files are retained

They provide:

- provenance for the already-completed Stage 1 results;
- a record of the intermediate controlled comparison;
- context for why the corrected video-based B1/B3 experiment was necessary.

## Do not use for corrected Stage 1

Corrected configurations live in:

```text
configs/stage1_corrected/
```

Corrected training is launched with:

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets
```

Historical T0 results are stored in:

```text
results/stage1/T0_two_frame_oracle/
```
