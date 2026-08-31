# Stage 1 Final Results — Controlled Comparison

## EF regression

| Model | Weight | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |
|---|---|---|---:|---:|---:|---:|
| EF only | N/A | Validation | 5.722 +/- 0.074 | 7.769 +/- 0.060 | 0.601 +/- 0.006 | 0.778 +/- 0.004 |
| EF only | N/A | Test | 6.071 +/- 0.071 | 8.222 +/- 0.037 | 0.548 +/- 0.004 | 0.745 +/- 0.002 |
| Multi-task | W2 | Validation | 5.406 +/- 0.040 | 7.498 +/- 0.195 | 0.629 +/- 0.019 | 0.795 +/- 0.014 |
| Multi-task | W2 | Test | 5.658 +/- 0.111 | 7.789 +/- 0.175 | 0.594 +/- 0.018 | 0.775 +/- 0.010 |

## LV segmentation

| Model | Weight | Split | Dice ED ↑ | Dice ES ↑ | Mean Dice ↑ | Mean HD95 ↓ |
|---|---|---|---:|---:|---:|---:|
| Segmentation only | N/A | Validation | 0.934 +/- 0.001 | 0.908 +/- 0.001 | 0.921 +/- 0.001 | 2.808 +/- 0.043 |
| Segmentation only | N/A | Test | 0.933 +/- 0.000 | 0.908 +/- 0.000 | 0.921 +/- 0.000 | 2.780 +/- 0.029 |
| Multi-task | W2 | Validation | 0.932 +/- 0.002 | 0.907 +/- 0.002 | 0.919 +/- 0.002 | 2.897 +/- 0.073 |
| Multi-task | W2 | Test | 0.931 +/- 0.001 | 0.907 +/- 0.002 | 0.919 +/- 0.001 | 2.830 +/- 0.037 |
