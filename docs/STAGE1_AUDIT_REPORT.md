# Stage 1 code and data audit

Audit date: 2026-08-10. Evidence is in `outputs/stage1_audit/`.

## 1. EF output

- EF-only model: `r2plus1d_18` with `model.fc = Linear(in_features, 1)`.
- Multitask model: global average pooling followed by `Linear(2048, 256)`, ReLU,
  dropout, and `Linear(256, 1)`. There is no final activation, so the output is
  an unconstrained continuous scalar.
- For multitask training, the ED and ES frame scalars are averaged once to make
  one scalar per video. Metrics and MSE use that video scalar, not the two frame
  values.
- Loss: mean MSE. Targets are native EF percentage points (0-100 convention;
  observed local labels 6.9073-96.9672), not fractions. Conversion to the
  reported EF percentage is therefore the identity operation: `EF_percent =
  prediction`. No multiplication by 100, sigmoid, or clipping is applied.

## 2. Segmentation target

- Structure: left-ventricular cavity in the apical four-chamber view.
- Masks: one-channel binary float masks containing only 0 and 1.
- Labels: the two human-traced frames for each traced video. The larger traced
  polygon is designated ED/Large; the smaller is ES/Small. There are 10,025
  traced videos, each with exactly two traced frames.
- Loss: mean `BCEWithLogitsLoss`, computed from raw one-channel logits. Sigmoid
  and a 0.5 threshold are used only for metrics/visualization.

## 3. Reproducibility environment

- Repository: `https://github.com/jewettm530/dynamic.git`
- Branch: `stage1-baselines`
- HEAD: `9e6d93a8b7c6a1c9058dd7893360c1e54212b705`
- The tree has uncommitted Stage 1 changes; the exact status is saved in
  `environment.json`.
- Python 3.9.25; PyTorch 2.7.1+cu118; torchvision 0.22.1+cu118; compiled CUDA
  runtime 11.8; cuDNN 9.1.0.
- NumPy 1.26.4; pandas 2.3.3; SciPy 1.13.1; scikit-learn 1.6.1;
  scikit-image 0.24.0; OpenCV 4.11.0; matplotlib 3.9.4.
- GPU type could not be recorded on this host: `torch.cuda.is_available()` is
  false and `nvidia-smi` cannot communicate with the NVIDIA driver. Do not
  claim a GPU model until the audit is rerun on the training host.

## 4. Splits and leakage

- Saved split counts: train 7,465; validation 1,288; test 1,277 (10,030 total).
- ED/ES-labeled counts usable by segmentation/multitask: train 7,460;
  validation 1,288; test 1,276. Six videos have no tracing (five train, one
  test). EF-only loading was fixed so it no longer incorrectly drops these six.
- The split is the saved `FileList.csv` `Split` column and is never generated
  from a seed. Seeds only affect training order/sample choice, so all seeds use
  the identical manifest. Audit manifest SHA-256:
  `a167250b973120f6ecc225c963c3e5c0873a90b726bbd4fe4255a602292e62c0`.
- Video leakage: zero train/validation, train/test, or validation/test overlap;
  zero duplicated video rows.
- `FileList.csv` has no patient-ID column. The local dataset documentation says
  the 10,030 videos are from individuals, but the deidentified files do not
  provide an independent patient mapping. Thus video-ID leakage is confirmed;
  patient-ID leakage cannot be independently rechecked from this release and
  must not be represented as a direct patient-ID join.

## 5. Sample association check

Fifteen videos were checked: five deterministic random samples from each split.
For every sample, the filename and EF came from the same `FileList.csv` row;
ED/ES indices and traces came from that filename's `VolumeTracings.csv` rows;
masks were regenerated from those trace coordinates. All 30 masks were binary,
all ED masks were larger than their paired ES masks, and manual review of all
15 saved overlays found the masks on the LV cavity at the stated video frames.
No association mismatch was observed. See `sample_manifest.csv` and
`alignment_examples/`.

## 6. Spatial alignment

The maintained Stage 1 EF, segmentation, and multitask scripts set `pad=None`
and apply no crop, rotation, resize, or flip. Consequently there is no spatial
transform that could differ between an image and mask; masks are rasterized in
the native video frame dimensions. This was checked by code inspection and by
the 15 native-resolution before/overlay examples. Any future spatial
augmentation must be implemented as a paired image/mask transform before it is
enabled.

## 7. Loss and evaluation path

- `MultitaskLoss.forward` computes `L_EF = MSE(video_ef, ef_target)` and
  `L_seg = BCEWithLogits(segmentation_logits, masks)`, then exactly
  `L_total = ef_weight * L_EF + seg_weight * L_seg`.
- Only the concatenated labeled ED and ES images/masks enter `L_seg`.
- EF metrics receive one final scalar per video: the mean of the ED and ES EF
  head outputs. On the native scale that scalar is already a percentage.
- Training constructs only train and validation datasets. The best checkpoint
  is selected using validation EF MAE (multitask/EF-only) or validation mean
  Dice (segmentation-only). Test data are not constructed or used for model
  selection. A held-out test evaluation should occur only after loading the
  frozen selected checkpoint.
- The one-batch smoke test confirmed output shapes `[4,1,112,112]` for four
  labeled frames and `[2]` for two final video EF values, and completed backward
  propagation plus an optimizer step.

## 8. Raw loss scale check

A 10-batch, batch-size-2 CPU check (seed 42, random initialization) produced:

| Unweighted loss | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `L_EF` (percentage-point MSE) | 3382.9109 | 3438.8828 | 2085.6062 | 4322.6260 |
| `L_seg` (pixel BCE) | 0.61537 | 0.61112 | 0.57431 | 0.66513 |

The raw EF loss is roughly 5,500 times the raw segmentation loss at this early
check, so nominal task weights do not imply similarly sized contributions.
Stage 1 deliberately fixes EF targets to percentage points and mean MSE for
every experiment; do not switch some runs to 0-1 targets or rescale MSE. Any
loss balancing must be expressed only through the recorded task weights.

## Repairs made during this audit

- Restored package initializers and removed imports of deleted legacy utility
  modules so the maintained scripts import successfully.
- Fixed EF-only loading to retain all split videos when tracings are absent.
- Made `--no-pretrained` genuinely disable ResNet backbone weights, avoiding an
  implicit download.
- Made the audit runner CPU/offline safe and configurable for short loss checks.
- Added labeled split counts, manifest hashing, and explicit seed-independence
  evidence to the audit output.
- Fixed an indentation error in `plot_hyperparameter_sweep.py`.

Verification: all Python files under `echonet/` and `scripts/` compile, shell
syntax for `RUN_AUDIT.sh` passes, `git diff --check` passes, the 15-sample audit
completed, and the forward/backward and 10-batch loss checks completed.
