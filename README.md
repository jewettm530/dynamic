# EchoNet-Dynamic Research Fork

This repository is based on the original **EchoNet-Dynamic** implementation and is being used for a research project on whether **sparse left-ventricular (LV) segmentation supervision improves continuous ejection-fraction (EF) regression**.

The current Stage 1 protocol is a corrected video-based design. The EF task must use a **multi-frame echocardiography clip** and must be able to run at inference using **video only**. Expert ED/ES frame locations and masks are used only for the segmentation task during training/evaluation.

> **Current protocol:** `docs/STAGE1_CORRECTED_IMPLEMENTATION.md`  
> **Correction history:** `STAGE1_CORRECTION_CHANGELOG.md`  
> **Historical two-frame results:** `results/stage1/T0_two_frame_oracle/`  
> **Corrected results:** `results/stage1/corrected/`

---

## 1. Why Stage 1 was corrected

The earlier multi-task implementation predicted EF from the expert-selected end-diastolic (ED) and end-systolic (ES) frames. Each frame was processed independently as a 2D image, an EF value was predicted from each frame, and the two scalar predictions were averaged.

That experiment is useful as an oracle/two-frame diagnostic, but it does **not** test standard video-based EF regression because:

- the EF branch never receives the cardiac motion sequence;
- temporal features are not learned before the EF prediction;
- ground-truth ED/ES phase information is required to construct the EF input.

Those completed results are preserved as **T0 — two-frame oracle diagnostic** and are not rerun.

The corrected research question is:

> **Does sparse LV segmentation supervision improve continuous EF regression when EF is predicted from a multi-frame echocardiography clip and EF inference requires only the video?**

---

## 2. Corrected Stage 1 experiment design

| ID | Model | EF input | Segmentation supervision | Role |
|---|---|---|---|---|
| **T0** | Historical two-frame MTL | Expert-selected ED + ES frames | ED/ES masks | Diagnostic/oracle result only; do not rerun |
| **B1** | Video EF-only baseline | Multi-frame video clip | None | Main EF baseline |
| **B2** | Segmentation-only baseline | Expert-labeled ED/ES frames | ED/ES masks | Main segmentation baseline |
| **B3** | Corrected video MTL | Multi-frame video clip | ED/ES frames/masks during training only | Main multi-task model |

### Primary comparisons

- **EF:** B1 vs. B3
- **Segmentation:** B2 vs. B3
- **T0:** retained for context only because it uses a different, oracle-informed EF input

### Required seeds

All corrected experiments use:

```text
42
2026
3407
```

### B3 loss weights

- **W1:** `0.1 L_EF + 0.9 L_seg`
- **W2:** `0.5 L_EF + 0.5 L_seg`
- **W3:** `0.9 L_EF + 0.1 L_seg`

The selected B3 weight is chosen using **mean validation EF MAE only** across the three seeds. The test set is not used for architecture decisions, debugging, checkpoint selection, or weight selection.

---

## 3. Corrected implementation

### EF pathway

The corrected EF branch uses a video tensor:

```text
[B, 3, T, H, W]
```

with the standard Stage 1 setting:

```text
T = 32 frames
period = 2
```

The EF pathway is:

```text
multi-frame video clip
        ↓
shared R(2+1)D-18 encoder
        ↓
spatiotemporal feature representation
        ↓
AdaptiveAvgPool3d over T/H/W
        ↓
linear EF head
        ↓
one video-level EF prediction
```

The implementation is in:

```text
echonet/modeling/stage1_video_multitask.py
```

`Stage1VideoMultitaskModel.forward_ef(video)` accepts the video tensor only. ED/ES locations, masks, and tracing information are not arguments to the EF forward path.

### Segmentation pathway

B2 and B3 use the same R(2+1)D encoder family and segmentation decoder. Expert-labeled ED/ES frames are passed through the segmentation pathway as one-frame clips, and the segmentation loss updates the shared encoder.

For B3:

```text
video clip ───────────────► shared encoder ─► EF head ─► EF
                                ▲
                                │ shared parameters
                                │
ED/ES labeled frames ─────► shared encoder ─► segmentation decoder ─► masks
```

The ED/ES annotations therefore supervise representation learning during training without becoming EF inputs.

### Dataset implementation

The corrected dataset class is:

```text
echonet/datasets/stage1_video.py
```

Important behavior:

- **B1 EF data path:** loads `FileList.csv` + `Videos/` only and does not load or stat `VolumeTracings.csv`.
- **B3 EF cohort:** uses the same full saved split as B1.
- **B3 segmentation loss:** is computed only for videos that have ED/ES tracings.
- **B2:** requires videos with ED/ES segmentation labels.
- **Training clip sampling:** random valid start.
- **Validation/test clip sampling:** deterministic center clip.

Expected saved-split counts are documented in `configs/stage1_corrected/common.yaml`.

---

## 4. Matched B1/B3 settings

The controlled comparison requires B1 and B3 to remain identical for all EF-related settings except the addition of segmentation supervision in B3.

Current common settings are:

| Setting | Value |
|---|---|
| EF encoder | R(2+1)D-18 |
| Pretrained | Yes |
| Clip length | 32 frames |
| Sampling period | 2 |
| Training clip sampling | Random valid start |
| Validation/test sampling | Center clip |
| Epochs | 45 |
| Batch size | 4 |
| Optimizer | SGD |
| Learning rate | `1e-4` |
| Momentum | `0.9` |
| Weight decay | `1e-4` |
| Scheduler | StepLR |
| LR step period | 15 |
| EF loss | Mean MSE, EF scaled to 0–1 during training |
| EF reporting scale | 0–100 |
| B1/B3 checkpoint rule | Lowest validation EF MAE |
| B2 checkpoint rule | Highest validation Mean Dice |

The runner passes these values to every corrected run. If a resource-related setting such as batch size must be changed, use the **same revised setting for the entire corrected experiment matrix**, rather than changing only one model or seed.

---

## 5. Repository layout

### Corrected implementation

```text
echonet/
├── datasets/stage1_video.py
├── modeling/stage1_video_multitask.py
└── utils/
    ├── reproducibility.py
    ├── stage1_corrected.py
    ├── stage1_evaluation.py
    └── stage1_metrics.py

scripts/
├── training/
│   ├── train_stage1_b1_video_ef.py
│   ├── train_stage1_b2_segmentation.py
│   ├── train_stage1_b3_video_multitask.py
│   └── run_stage1_corrected_training.sh
├── evaluation/
│   ├── evaluate_stage1_corrected_ef.py
│   ├── evaluate_stage1_corrected_segmentation.py
│   └── run_stage1_corrected_final.sh
├── verification/
│   ├── smoke_test_stage1_corrected.py
│   ├── verify_video_only_inference.py
│   └── verify_stage1_corrected_runs.py
└── analysis/
    ├── summarize_stage1_corrected_validation.py
    ├── summarize_stage1_corrected_final.py
    └── collect_stage1_reproducibility.py
```

### Local runtime output — do not commit checkpoints

```text
output/stage1/corrected/
├── B1_video_ef/seed_{42,2026,3407}/
├── B2_segmentation/seed_{42,2026,3407}/
├── B3_video_mtl/
│   ├── W1/seed_{42,2026,3407}/
│   ├── W2/seed_{42,2026,3407}/
│   └── W3/seed_{42,2026,3407}/
└── logs/
```

### Git-tracked research results

```text
results/stage1/
├── T0_two_frame_oracle/
├── archive_pre_correction/
└── corrected/
    ├── validation_weight_selection/
    ├── evaluation/
    ├── final/
    ├── verification/
    └── runs/
```

---

## 6. Data requirements

The project expects the local dataset root to contain at least:

```text
datasets/
├── FileList.csv
├── Videos/
└── VolumeTracings.csv   # required only when segmentation labels are requested
```

On the current Lambda environment the expected root is:

```text
/data/jewettm/dynamic/datasets
```

Raw EchoNet-Dynamic data are not stored in Git.

---

## 7. Environment and installation

From the repository root:

```bash
cd /data/jewettm/dynamic
source /data/jewettm/miniconda3/etc/profile.d/conda.sh
conda activate echonet
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

The original EchoNet-Dynamic package can also be installed from the repository with:

```bash
pip install -e .
```

The corrected Stage 1 scripts explicitly add the repository root to `PYTHONPATH` when launched through the provided shell runners.

---

## 8. Required pre-training audit

Run the corrected audit before launching the full matrix:

```bash
bash RUN_AUDIT.sh \
    /data/jewettm/dynamic/datasets \
    outputs/stage1_corrected_audit
```

The audit verifies:

1. the EF input has `T > 2`;
2. EF features are aggregated temporally/spatially before one scalar prediction;
3. the segmentation loss reaches the shared encoder;
4. the EF inference data path works without `VolumeTracings.csv`;
5. EF inference requests no ground-truth ED/ES indices or masks.

The audit intentionally does not export videos, frames, masks, tracings, or patient-derived visualizations.

---

## 9. Full corrected Stage 1 training

The full validation-stage matrix contains **15 runs**:

```text
B1:       3 seeds
B2:       3 seeds
B3 W1:    3 seeds
B3 W2:    3 seeds
B3 W3:    3 seeds
-----------------
Total:   15 runs
```

The recommended procedure is to run all 15 sequentially on the same physical GPU for consistent hardware conditions.

### Run in tmux

Create one persistent session:

```bash
tmux new -s stage1_corrected
```

Inside tmux:

```bash
cd /data/jewettm/dynamic
source /data/jewettm/miniconda3/etc/profile.d/conda.sh
conda activate echonet
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
mkdir -p output/stage1/corrected/logs
set -o pipefail
```

Then launch the full corrected training matrix:

```bash
bash scripts/training/run_stage1_corrected_training.sh \
    /data/jewettm/dynamic/datasets \
    2>&1 | tee output/stage1/corrected/logs/full_stage1_training.log
```

Detach without stopping training:

```text
Ctrl+B, then D
```

Reconnect later:

```bash
tmux attach -t stage1_corrected
```

Check the log without attaching:

```bash
tail -50 output/stage1/corrected/logs/full_stage1_training.log
```

Do not edit, pull, reset, rebase, switch branches, or otherwise change the model/training code while the sequential matrix is running; later runs would otherwise execute different source code from earlier runs.

---

## 10. Validation-only B3 weight selection

After all nine B3 runs finish, the training runner calls:

```text
scripts/analysis/summarize_stage1_corrected_validation.py
```

This creates:

```text
results/stage1/corrected/validation_weight_selection/
├── weighting_seed_results.csv
├── weighting_summary.csv
├── weighting_selection.json
├── table_a_validation_weight_selection.md
└── LOCKED_WEIGHT.txt
```

`LOCKED_WEIGHT.txt` is generated automatically from the **lowest mean validation EF MAE across seeds 42, 2026, and 3407**. Do not manually choose a weight using test results.

All reported summary values use **mean ± sample SD (`ddof=1`)** across the three independent seeds. Predictions are not pooled across seeds before computing the run-level summary.

---

## 11. Final evaluation

Only after `LOCKED_WEIGHT.txt` exists and the validation choice is frozen, run:

```bash
bash scripts/evaluation/run_stage1_corrected_final.sh \
    /data/jewettm/dynamic/datasets
```

The final runner:

1. verifies that required B1/B3 settings match;
2. performs the explicit **video-only EF evaluation**;
3. evaluates segmentation separately on labeled ED/ES frames;
4. generates the final corrected Stage 1 summaries.

For B3, EF and segmentation metrics are taken from the **same checkpoint**, selected by validation EF MAE.

---

## 12. Reproducibility artifacts

After final evaluation:

```bash
python scripts/analysis/collect_stage1_reproducibility.py
```

The collector copies safe run artifacts from `output/` into `results/stage1/corrected/runs/`, including:

- `run_config.json`
- `run_summary.json`
- `best_validation_metrics.json`
- `training_history.csv`
- validation prediction/metric CSV files

It does **not** copy checkpoints or patient-derived source data.

### Do not commit

- `datasets/`
- AVI videos
- extracted patient frames
- masks/tracings
- `VolumeTracings.csv`
- `*.pt` checkpoints
- patient-derived visualizations
- transient/incomplete runtime output

Before committing:

```bash
git status --short
git diff --cached --name-only
```

A useful reproducibility workflow is to make one commit containing the frozen corrected implementation before/during training, then a later commit containing safe final result summaries and updated documentation.

---

## 13. Historical Stage 1 experiments

### T0 — two-frame oracle diagnostic

```text
results/stage1/T0_two_frame_oracle/
configs/t0_two_frame_oracle/
```

These are completed results from the controlled two-frame model. They are retained because they provide evidence that segmentation supervision helped EF prediction in that oracle-informed setting, but they do not answer the corrected video-based research question.

### Superseded unmatched comparison

```text
results/stage1/archive_pre_correction/
configs/archive_pre_correction/
```

These experiments compared models that differed in architecture, input type, optimizer, epoch budget, and other settings. They are kept only for provenance and should not be used in the final corrected Stage 1 tables.

---

## 14. Metrics and reporting

### EF regression

Report for validation and test:

- MAE ↓
- RMSE ↓
- R² ↑
- Pearson r ↑

### LV segmentation

Report for validation and test:

- Dice ED ↑
- Dice ES ↑
- Mean Dice ↑
- Mean HD95 ↓

Every final summary is reported as **mean ± sample SD across seeds 42, 2026, and 3407**.

---

## 15. Original EchoNet-Dynamic project

EchoNet-Dynamic is an end-to-end deep learning system for:

1. semantic segmentation of the left ventricle;
2. prediction of EF from echocardiography videos/clips; and
3. assessment of cardiomyopathy with reduced EF.

Original paper:

> **Video-based AI for beat-to-beat assessment of cardiac function**  
> David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. *Nature*, 2020.  
> https://doi.org/10.1038/s41586-020-2145-8

Official EchoNet-Dynamic documentation:

https://echonet.github.io/dynamic/

The public EchoNet-Dynamic dataset contains 10,030 deidentified apical four-chamber echocardiogram videos with video-level EF labels and sparse LV tracings at ED/ES for labeled studies.

### Original package usage

The original repository provides commands such as:

```bash
echonet segmentation --save_video
echonet video
```

and the original hyperparameter sweeps are available through the original experiment scripts. These commands are retained for compatibility/history; the research Stage 1 results in this repository should be generated with the dedicated corrected Stage 1 scripts described above.

---

## 16. Stage 1 completion checklist

Before considering corrected Stage 1 complete, confirm:

- [ ] EF audit passes with video-only inference and no oracle ED/ES information.
- [ ] B1 is complete for seeds 42, 2026, and 3407.
- [ ] B2 is complete for seeds 42, 2026, and 3407.
- [ ] B3 W1/W2/W3 are complete for all three seeds.
- [ ] One B3 weight is locked using validation EF MAE only.
- [ ] Final test evaluation is performed only after the lock is frozen.
- [ ] B1/B3 matched-setting verification passes.
- [ ] Tables A, B, and C are complete.
- [ ] All summary values use sample SD (`ddof=1`).
- [ ] Safe reproducibility files are copied into `results/stage1/corrected/runs/`.
- [ ] README/results are updated with the selected weight and final conclusions.
- [ ] The final reproducibility commit hash is recorded.

**Stage 1 reproducibility commit:** `TO_BE_RECORDED_AFTER_FINAL_CODE/RESULTS_COMMIT`
