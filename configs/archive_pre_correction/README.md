# Superseded Pre-Correction Configuration Archive

This directory contains configurations from earlier Stage 1 experiments that are retained **only for provenance**.

These experiments should not be used to generate the final corrected Stage 1 results because they did not satisfy the final controlled comparison requirements.

## Why they were superseded

Earlier comparisons mixed differences such as:

- video versus ED/ES-frame EF inputs;
- different architectures;
- different epoch budgets;
- different batch sizes;
- different optimizers and learning rates;
- different checkpoint rules.

Those differences made it difficult to attribute performance changes specifically to segmentation supervision.

A later controlled two-frame experiment corrected many of those mismatches but still required expert ED/ES information for EF. That experiment is preserved separately as **T0**.

The final corrected protocol instead compares:

```text
B1: video EF-only
vs.
B3: matched video EF + sparse segmentation supervision
```

with B2 as the matched segmentation-only baseline.

## Current protocol

Use:

```text
configs/stage1_corrected/
```

and:

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets
```

Do not modify or reuse files in this archive as the basis for new corrected Stage 1 runs unless explicitly reproducing historical behavior.
