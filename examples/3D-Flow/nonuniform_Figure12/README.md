# Manuscript Figure 12: nonuniform 3D PIRN

These are the three cases in Figure 12 of the 2026-10-25 manuscript. The experiment used Figure 13 as its working number. The stored predictions come from the validation-selected checkpoint at epoch 41,925 of the nonuniform-grid run on 2026-10-16.

Rows are training `(Gr=10000, Ha=100)`, interpolation validation `(30000, 300)`, and extrapolation test `(40000, 500)`. Columns are PIRN, CFD reference, and normalized absolute error. The slice is `z=0.16 m`; velocity is axial `w`, and temperature is rise above `723.15 K`. The error map is `abs(PIRN-CFD)/max(abs(CFD slice))`. Interpolation and edge extension affect display only; metrics use the original 32×32 sample points.

Run `python render_figure12.py` with NumPy and Matplotlib to redraw the velocity and temperature panels from `three_case_fields.npz`. `snapshot_metadata.json` records the checkpoint SHA-256. `three_case_max_errors.csv` contains the native-grid maximum errors. This script redraws stored predictions; it does not train the model.

The displayed extrapolation case is `(Gr=40000, Ha=500)`. In the later 50,000-step run, `(Gr=10000, Ha=500)` had 53.6516% relative L2 error in axial velocity. The displayed case therefore does not represent all four extrapolation cases.
