# Figure 9 using finite-volume data only

## Data and workflow

The original `pre-data/*.joblib` OpenFOAM-derived fields are not read by this workflow. `cavity_fvm.py` solves the same square cavity on a fixed staggered finite-volume grid (N=64). `generate_fvm_dataset.py` independently computes every training case at Re = 5, 5.2, ..., 25 and writes `fvm-reference/cavity_fvm_101_N64.npz` (shape 101 × 3 × 64 × 64). The same solver separately computes Re=20.1 and Re=30 as interpolation and extrapolation references; they are not included in training.

`train_fvm_pirn.py` trains a new three-branch residual convolutional model with three blocks in each velocity branch and two in the pressure branch. Inputs are Reynolds numbers. Targets are scaled by lid speed U=Re/10 for velocity and U² for kinematic pressure; pressure is shifted to zero spatial mean. The run uses seed 20261001 and 400 epochs. This corrects the unusable tensor reshape/indexing in the legacy notebook; it is a reproducible replacement model, not a claim of identical weights or every auxiliary-force term of the manuscript network.

```powershell
python generate_fvm_dataset.py --workers 4
python cavity_fvm.py --re 20.1 30 --n 64
python train_fvm_pirn.py
```

## Results

The original `2D-cavity/Untitled.ipynb` contains training but no error-map formula. Other original PIRN `predict.ipynb` notebooks in this repository use `err = abs((reference - prediction) / reference.max())`. The revised figures and `metrics.json` now use that formula for every pixel; the table below reports its spatial mean and maximum. The signed reference maximum is used, not its full range or its maximum absolute value. Pressure prediction and reference are each shifted to zero spatial mean first.

| Re | Role | u mean / max | v mean / max | p mean / max |
|---:|---|---:|---:|---:|
| 15 | Training | 0.0089% / 0.162% | 0.0169% / 0.362% | 0.0085% / 0.481% |
| 20.1 | Interpolation | 0.0093% / 0.163% | 0.0189% / 0.351% | 0.0121% / 0.556% |
| 30 | Extrapolation | 0.0590% / 0.610% | 0.1428% / 1.222% | 0.1230% / 9.106% |

The relative L2 metric from the previous analysis remains available as a separate supplementary statistic:

| Re | Role | u relative L2 | v relative L2 | p relative L2 | p interior relative L2 (outer 5 cells removed) |
|---:|---|---:|---:|---:|---:|
| 15 | Training | 0.053% | 0.065% | 0.56% | 0.71% |
| 20.1 | Interpolation | 0.056% | 0.068% | 0.76% | 1.13% |
| 30 | Extrapolation | 0.40% | 0.51% | 7.84% | 11.15% |

The full metrics, checkpoint, training curve, comparison arrays, and panels are under `fvm-results/`. `Figure9_FVM_original_error_all_cases.png` includes schematic plus all three cases. Prediction and reference in each row share color limits. For the pressure row, the display is deliberately clipped at the **reference 99th percentile of absolute pressure** to show the bulk field; the underlying arrays and all metrics remain unclipped. The revised error maps use the original-project formula and show the complete numerical error range. The older `Figure9_FVM_all_cases.png` used a field-range denominator and is superseded by the file whose name contains `original_error`.

## Numerical scope

The dataset is internally consistent: all 101 training cases satisfy the solver convergence test; the largest native-grid discrete divergence is under 8e-15. Separate N=64 and N=96 FVM runs differ in the interior by at most 0.42% for u, 0.31% for v, and 0.27% for p at Re=15, 20.1, and 30.

The discontinuous moving-lid/no-slip junction creates grid-sensitive pressure point values at the upper corners. This remains true even though all data are now from the same solver. The model-to-FVM evaluation is valid **at fixed N=64 and for the solver's pressure convention**, but the full pressure contour is not a grid-independent continuum benchmark. The extrapolation pressure error is materially larger than the training and interpolation errors and should be reported as such in any manuscript update.
