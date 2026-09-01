# EchoNet-Dynamic Fork:<br/>Interpretable AI for beat-to-beat cardiac function assessment

<p align="center">
  <strong>Deep-learning experiments for echocardiography, ventricular segmentation, and cardiac-function assessment</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Medical%20Imaging-0A7EA4?style=flat-square">
  <img src="https://img.shields.io/badge/Echocardiography-6F42C1?style=flat-square">
</p>

## About This Fork

This repository is a **research fork of [EchoNet-Dynamic](https://github.com/echonet/dynamic)** used to investigate alternative deep-learning approaches for cardiac ultrasound.

The original EchoNet-Dynamic project introduced deep learning for:

* left-ventricular segmentation;
* ejection-fraction estimation;
* video-based cardiac-function assessment.

This fork preserves the upstream implementation while adding experiments involving **multi-task learning, alternative segmentation architectures, and cardiac-function modeling**.

> The EchoNet-Dynamic dataset, original models, and foundational methodology were developed by the original EchoNet authors. This repository contains independent research experiments built on that work.

---

## Research Focus

Current work investigates questions such as:

1. Can segmentation and cardiac-function prediction benefit from shared representations?
2. How do single-task and multi-task models compare under controlled conditions?
3. Which architectures best extract clinically relevant information from echocardiography?
4. How should ejection-fraction prediction use temporal information from the complete video sequence?

---

## Research Extensions

### 🫀 Multi-task DeepLabV3

A custom multi-task architecture jointly supports:

* left-ventricular segmentation of end-diastolic and end-systolic frames;
* cardiac-function prediction from shared learned representations.

Key files include:

```text
echonet/modeling/multitask_deeplab.py
scripts/training/train_multitask.py
```

The goal is to test whether anatomical segmentation can provide useful auxiliary supervision for cardiac-function modeling.

### 🧠 Alternative Segmentation Models

The fork also contains experimental architecture work including:

```text
echonet/modeling/vit_segmentation_model.py
```

for exploring approaches beyond the original convolutional segmentation model.

### 🎥 Video-Based EF Modeling

Additional experiments investigate ejection-fraction prediction directly from echocardiogram video rather than relying only on labeled end-diastolic and end-systolic frames.

---

## Evaluation

Depending on the task, experiments use metrics including:

### EF regression

* MAE
* RMSE
* R²
* Pearson correlation

### Segmentation

* Dice coefficient
* IoU
* accuracy
* precision
* recall
* specificity

### EF classification

* accuracy
* precision
* recall
* F1
* specificity
* ROC-AUC

Controlled comparisons aim to keep dataset splits, preprocessing, optimization settings, and evaluation procedures constant wherever possible.

---

## Dataset

The project uses the **EchoNet-Dynamic dataset**, containing more than 10,000 deidentified echocardiogram videos.

Dataset information and access:

https://echonet.github.io/dynamic

The dataset itself is **not stored in this repository**.

---

## Installation

```bash
git clone https://github.com/jewettm530/dynamic.git
cd dynamic
pip install --user .
```

Core dependencies include PyTorch, Torchvision, NumPy, OpenCV, scikit-image, scikit-learn, and tqdm.

Experimental training workflows are located under:

```text
scripts/training/
```

---

## Repository Structure

```text
dynamic/
├── echonet/
│   ├── datasets/
│   ├── modeling/             # Upstream + experimental architectures
│   ├── losses/
│   └── ...
│
├── scripts/
│   └── training/             # Experimental training workflows
│
├── docs/
├── output/                   # Locally generated experiment outputs
└── README.md
```

---

## Limitations

* This repository is an active research fork.
* Experimental implementations continue to evolve.
* Performance on EchoNet-Dynamic does not establish external clinical generalizability.
* Frame-based models cannot capture all temporal information contained in full videos.
* Models in this repository are research tools and are not intended for clinical use.

---

## Attribution

Original project:

**[echonet/dynamic](https://github.com/echonet/dynamic)**

Original publication:

Ouyang, D., He, B., Ghorbani, A., et al. (2020). *Video-based AI for beat-to-beat assessment of cardiac function.* **Nature**.

https://doi.org/10.1038/s41586-020-2145-8

---

**Skills:** `Python` · `PyTorch` · `Deep Learning` · `Medical Imaging` · `Echocardiography` · `Segmentation` · `Multi-task Learning` · `Video Modeling`

---

<details>
<summary><h1>Technical Details</h1></summary>

### Original EchoNet-Dynamic

The upstream EchoNet-Dynamic system supports:

1. frame-level left-ventricular segmentation;
2. ejection-fraction prediction from video clips;
3. beat-by-beat cardiac-function analysis.

The original commands and implementation remain available in this fork for comparison and reproducibility.

### Multi-task Frame Model

The current frame-based multi-task workflow uses:

```text
LargeFrame  → end-diastolic frame
SmallFrame  → end-systolic frame
LargeTrace  → end-diastolic LV mask
SmallTrace  → end-systolic LV mask
EF          → cardiac-function target
```

Conceptually:

```text
Echocardiogram frame
        │
        ▼
 Shared feature extractor
        │
        ├────────► LV segmentation
        │
        └────────► cardiac-function prediction
```

The segmentation branch learns anatomical information while the second task tests whether those shared features also support cardiac-function assessment.

### Patient-Level Classification Evaluation

For the reduced-EF classification experiment, probabilities from a participant's large and small frames are combined before patient-level classification metrics are calculated.

### Experimental Outputs

Training scripts save artifacts such as:

```text
checkpoint.pt
best.pt
training_history.csv
validation_metrics.csv
```

under experiment-specific output directories.

### Controlled Comparisons

Architecture comparisons should hold the following constant wherever possible:

* train/validation/test split;
* random seeds;
* frame/video sampling;
* preprocessing;
* optimizer settings;
* checkpoint-selection rules;
* evaluation metrics.

This helps distinguish architectural improvements from differences in experimental setup.

### Upstream Usage

The original EchoNet functionality remains available for reproducing upstream tasks such as:

```bash
echonet segmentation --save_video
echonet video
```

For the complete original workflow and data-use requirements, refer to the upstream repository and project documentation.

</details>
