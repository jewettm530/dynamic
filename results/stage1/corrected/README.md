# Corrected Stage 1 Results

This directory contains the **professor-requested corrected Stage 1 B1/B2/B3 research results and reproducibility evidence**.

The corrected research question is whether sparse LV segmentation supervision improves continuous EF regression when EF is predicted from a **multi-frame echocardiography video clip** and EF inference requires only the video.

Historical two-frame/oracle results do not belong here; they are stored in `results/stage1/T0_two_frame_oracle/`.

## Experimental models

- **B1:** video EF-only baseline
- **B2:** segmentation-only baseline
- **B3:** corrected video multi-task model
- **B3 weights:** W1, W2, W3
- **Seeds:** 42, 2026, 3407

## Directory structure

```text
results/stage1/corrected/
├── validation_weight_selection/
├── evaluation/
├── final/
├── verification/
└── runs/
```

### `validation_weight_selection/`

Generated after all B3 W1/W2/W3 validation runs finish.

Expected files include:

```text
weighting_seed_results.csv
weighting_summary.csv
weighting_selection.json
table_a_validation_weight_selection.md
LOCKED_WEIGHT.txt
```

The selected weight is the setting with the **lowest mean validation EF MAE across seeds 42, 2026, and 3407**.

`LOCKED_WEIGHT.txt` must exist before final test evaluation is allowed. Do not edit it manually to favor test performance.

### `verification/`

Contains evidence that the corrected comparison satisfies the protocol, including the B1/B3 matched-setting check.

The pre-training audit also verifies that:

- EF uses a multi-frame input (`T > 2`);
- temporal/spatial feature aggregation occurs before one EF prediction;
- segmentation loss reaches the shared encoder;
- video-only EF inference does not require `VolumeTracings.csv`, ED/ES indices, or masks.

### `evaluation/`

Contains seed-level final evaluation artifacts generated after the weight is locked. EF and segmentation evaluation are deliberately separated so the EF path remains video-only.

### `final/`

Contains the final Stage 1 summaries corresponding to the professor-requested tables and conclusions.

Expected reporting includes:

**EF regression**

- MAE
- RMSE
- R²
- Pearson r

**LV segmentation**

- Dice ED
- Dice ES
- Mean Dice
- Mean HD95

All final values are reported as **mean ± sample SD (`ddof=1`) across three seeds**.

### `runs/`

Contains safe reproducibility copies from the local runtime directory. The collector may copy:

- `run_config.json`
- `run_summary.json`
- `best_validation_metrics.json`
- `training_history.csv`
- `best_validation_predictions.csv`
- `best_validation_segmentation_metrics.csv`

Checkpoints are never copied here.

## Local runtime results

Training checkpoints and other runtime state remain under:

```text
output/stage1/corrected/
```

and should not be committed.

## How these results are generated

### 1. Validation-stage training

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets
```

This runs 15 sequential experiments:

```text
B1: 3
B2: 3
B3 W1: 3
B3 W2: 3
B3 W3: 3
```

Then it automatically generates the validation-only weight-selection files.

### 2. Final evaluation

After `LOCKED_WEIGHT.txt` is frozen:

```bash
bash scripts/evaluation/run_stage1_corrected_final.sh \
    /data/jewettm/dynamic/datasets
```

The final runner first verifies that B1 and the selected B3 setting match on all required EF/training fields. It then performs video-only EF evaluation, separate segmentation evaluation, and final summary generation.

### 3. Copy safe reproducibility files

```bash
python scripts/analysis/collect_stage1_reproducibility.py
```

## Interpretation rules

The primary EF conclusion should be based on **B1 vs. B3** across MAE, RMSE, R², and Pearson r.

The primary segmentation conclusion should be based on **B2 vs. B3** across Dice and HD95 metrics.

T0 should be discussed only as a historical oracle/two-frame diagnostic because it used a different EF input setting.

## Git/data safety

Do not place any of the following in this directory:

- checkpoints (`*.pt`)
- raw videos
- extracted patient frames
- masks
- tracing files
- patient-derived visualizations

Only safe aggregated/reproducibility files should be committed.
