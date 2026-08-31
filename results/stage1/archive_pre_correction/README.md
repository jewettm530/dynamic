# Superseded Pre-Correction Results

This directory contains earlier Stage 1 results that are retained for **provenance only**.

They are not part of the final corrected B1/B2/B3 tables.

## Historical sequence

The Stage 1 work progressed through three distinct designs:

1. **Early unmatched comparison** — models differed in architecture, input type, optimizer, epoch budget, and other settings.
2. **Controlled two-frame comparison** — many settings were matched, but EF still used expert-selected ED/ES frames. Those results are preserved separately as **T0**.
3. **Corrected video-based Stage 1** — B1 and B3 use matched video EF pathways, and ED/ES information is restricted to segmentation supervision/evaluation.

This folder contains results from stage 1 of that history: the early unmatched experiments.

## Why these results are not used in final conclusions

When two models differ in several implementation choices at once, a performance difference cannot be cleanly attributed to the intended experimental factor. These comparisons therefore cannot establish that segmentation supervision alone caused an EF improvement or degradation.

## Current results locations

Historical controlled two-frame diagnostic:

```text
results/stage1/T0_two_frame_oracle/
```

Current corrected protocol:

```text
results/stage1/corrected/
```

Do not move corrected B1/B2/B3 files into this archive.
