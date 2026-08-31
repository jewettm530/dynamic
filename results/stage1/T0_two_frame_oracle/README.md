# T0 — Historical Two-Frame Oracle Diagnostic Results

This directory contains the **completed controlled two-frame Stage 1 results** that preceded the professor-requested video-based correction.

These results are intentionally retained and should **not** be deleted or rerun.

## What the T0 model did

The EF branch used expert-selected ED and ES frames as its inputs. Each frame was processed independently, a scalar EF prediction was produced from each frame, and the two predictions were averaged into one video-level EF estimate.

```text
expert ED frame ─► EF prediction ─┐
                                  ├─► average ─► reported EF
expert ES frame ─► EF prediction ─┘
```

The segmentation branch used the associated ED/ES masks.

## Why T0 is no longer the main baseline

T0 does not answer the corrected research question because:

- the EF branch does not receive a temporal video sequence;
- cardiac motion is not modeled before EF prediction;
- EF inference depends on ground-truth ED/ES phase locations.

The results remain scientifically useful as an **oracle/two-frame diagnostic** showing what happened when expert phase information was available.

## Folder contents

- `weighting/` — historical W1/W2/W3 validation weighting results for the two-frame multi-task model.
- `final_controlled/` — historical controlled two-frame EF/segmentation comparison.

Local historical checkpoints/training state, when present, belong under:

```text
output/stage1/T0_two_frame_oracle/
```

They should not be committed to Git.

## How T0 should be reported

T0 may be described as a historical diagnostic, but it should **not** be compared to B1 as though the models had the same EF input setting.

The corrected primary EF comparison is:

```text
B1 video EF-only
vs.
B3 corrected video multi-task
```

Corrected results are stored in:

```text
results/stage1/corrected/
```

See the repository root `README.md` and `docs/STAGE1_CORRECTED_IMPLEMENTATION.md` for the current protocol.
