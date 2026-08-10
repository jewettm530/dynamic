# EchoNet-Dynamic Stage 1 fixed-code bundle

This bundle contains complete replacement/new files for the professor's Stage 1 baseline protocol.

## Replace these existing repository files

- `echonet/datasets/echo.py`
- `echonet/modeling/multitask_deeplab.py`
- `echonet/losses/multitask_loss.py`
- `scripts/training/train_multitask.py`

The `echo.py` replacement is important: the current repository version sorts traced frame **numbers**, which can destroy the official Large/Small ordering. This replacement chooses Large/ED and Small/ES by traced LV polygon area.

## Add these new files

- `echonet/utils/reproducibility.py`
- `echonet/utils/stage1_metrics.py`
- `scripts/training/train_ef_stage1.py`
- `scripts/training/train_segmentation_stage1.py`
- `scripts/verification/audit_stage1.py`
- `scripts/verification/smoke_test_stage1.py`
- `scripts/verification/loss_scale_check.py`
- `RUN_AUDIT.sh`

The old `echonet/utils/video.py` and `echonet/utils/segmentation.py` can stay for historical reproducibility; Stage 1 should use the dedicated new training scripts above so the old checkpoint/metric behavior cannot accidentally leak into the official experiments.

## Before running

From the repository root, make a new branch and copy the files into the matching paths. Do not overwrite the historical commit without preserving it.

```bash
git checkout -b stage1-baselines
```

Confirm the dataset root. In the uploaded project, `echonet/paths.py` points to `<repo>/datasets`, so the likely path on the Lambda machine is:

```text
/data/jewettm/dynamic/datasets
```

If that folder contains `FileList.csv`, `VolumeTracings.csv`, and `Videos/`, it is correct.

## Audit run

```bash
bash RUN_AUDIT.sh /data/jewettm/dynamic/datasets outputs/stage1_audit
```

This creates only a small audit package. It does not export full videos.

Expected output files include:

- `environment.json`
- `split_manifest.csv`
- `split_counts.csv`
- `split_overlap.json`
- `tracing_coverage.csv`
- `sample_manifest.csv`
- `alignment_examples/*.png` (5 train + 5 val + 5 test by default)
- `smoke_test.json`
- `loss_scale_check.csv`
- `loss_scale_check_summary.json`

Review every overlay PNG manually before concluding that ED/ES image, trace, and mask alignment is correct.

## Multi-task weight commands (do not run until the audit is approved)

W1 example:

```bash
python scripts/training/train_multitask.py \
  --data-root /data/jewettm/dynamic/datasets \
  --output outputs/stage1/W1_seed42 \
  --seed 42 --ef-weight 0.1 --seg-weight 0.9
```

W2 uses `--ef-weight 0.5 --seg-weight 0.5`; W3 uses `--ef-weight 0.9 --seg-weight 0.1`. Repeat each with seeds 42, 2026, and 3407.

## GPU note

The shell variable is case sensitive. If you need to restrict visible CUDA devices, use:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

not `export cuda_visible_devices=...`.

However, `nvidia-smi` reporting `Unable to determine the device handle ... Unknown Error` is a driver/host-level problem and is not fixed by `CUDA_VISIBLE_DEVICES`. The audit script separately asks PyTorch whether CUDA devices are usable and records `torch.cuda.get_device_name()` when possible.
