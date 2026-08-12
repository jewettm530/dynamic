# Stage 1 Final Results

## EF regression

| Model | Weight | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |
|---|---|---|---:|---:|---:|---:|
| EF only | N/A | Validation | 8.945 +/- 0.029 | 11.516 +/- 0.053 | 0.124 +/- 0.008 | 0.454 +/- 0.003 |
| EF only | N/A | Test | 9.107 +/- 0.156 | 11.871 +/- 0.245 | 0.057 +/- 0.039 | 0.417 +/- 0.028 |
| Multi-task | W2 | Validation | 5.406 +/- 0.040 | 7.498 +/- 0.195 | 0.629 +/- 0.019 | 0.795 +/- 0.014 |
| Multi-task | W2 | Test | 5.658 +/- 0.111 | 7.789 +/- 0.175 | 0.594 +/- 0.018 | 0.775 +/- 0.010 |

## LV segmentation

| Model | Weight | Split | Dice ED ↑ | Dice ES ↑ | Mean Dice ↑ | Mean HD95 ↓ |
|---|---|---|---:|---:|---:|---:|
| Segmentation only | N/A | Validation | 0.049 +/- 0.056 | 0.019 +/- 0.017 | 0.034 +/- 0.037 | 133.971 +/- 16.064 |
| Segmentation only | N/A | Test | 0.051 +/- 0.058 | 0.020 +/- 0.020 | 0.035 +/- 0.039 | 133.454 +/- 16.953 |
| Multi-task | W2 | Validation | 0.932 +/- 0.002 | 0.907 +/- 0.002 | 0.919 +/- 0.002 | 2.897 +/- 0.073 |
| Multi-task | W2 | Test | 0.931 +/- 0.001 | 0.907 +/- 0.002 | 0.919 +/- 0.001 | 2.830 +/- 0.037 |
