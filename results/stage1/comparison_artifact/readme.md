Experiment was not properly controlled for all runs to use the same saved data split and fixed settings except for the intended loss 
Incorrect configs used:
| Setting           | EF only             | Seg only           | Multi-task W2         |
| ----------------- | ------------------- | ------------------ | --------------------- |
| Architecture      | R(2+1)D-18          | DeepLabV3-ResNet50 | MultiTask DeepLabV3   |
| Input             | 32-frame video clip | ED + ES frames     | ED + ES frames        |
| Epochs            | 45                  | 50                 | 25                    |
| Batch size        | 20                  | 20                 | 4                     |
| Optimizer         | SGD                 | SGD                | Adam                  |
| LR                | 1e-4                | 1e-5               | 1e-4                  |
| Weight decay      | 1e-4                | 1e-5               | 0                     |
| Backbone for EF   | R(2+1)D             | —                  | ResNet50/DeepLab      |
| EF representation | temporal video      | —                  | ED/ES shared features |


The correct shard base config should be:
Backbone: DeepLabV3 / ResNet-50 shared architecture
Pretrained: yes
Data cohort: 7460 train / 1288 val / 1276 test
Input: labeled ED + ES frames
No augmentation
Epochs: 25
Batch size: 4
Optimizer: Adam
Learning rate: 1e-4
Weight decay: 0
Seeds: 42, 2026, 3407
EF scale: 0–1
EF loss: MSE
Seg loss: BCEWithLogitsLoss

Then the only task-related differences are:

EF-only: same shared backbone + EF regression head, segmentation branch removed, optimize only LEF.
Segmentation-only: same shared backbone + segmentation head, EF branch removed, optimize only Lseg.
Multi-task W2: same backbone + both heads, optimize 0.5LE+0.5Lseg.