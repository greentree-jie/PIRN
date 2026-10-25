# Manuscript Figure 14: uniform 3D PIRN

The uniform-grid run used its validation-selected checkpoint at epoch 37,775 of 38,000. `uniform_predictions.npz` contains the saved fields and `uniform_result.json` contains the run record. `manuscript_Figure14.png` is the panel used in the manuscript.

For `(Gr=40000, Ha=500)` at `z=0.16 m`, the uniform 32×32 prediction is mapped by bilinear interpolation to the original nonuniform 32×32 sample positions. The CFD reference is taken from `../nonuniform_Figure12/three_case_fields.npz` at those positions. The PNG files show PIRN, CFD, and normalized absolute error for axial velocity `w` and temperature rise `T_rise`.

Error is `abs(PIRN - CFD) / max(abs(CFD slice))` on the original sample positions. The display uses a 320×320 interpolation with edge extension. The CSV records native-position maximum error and relative L2 error; for `w`, these are 14.1645% and 13.3621%, respectively.

Run `python render_uniform_on_nonuniform_positions.py` with NumPy and Matplotlib to regenerate the PIRN and error maps, metrics, and provenance. The two CFD PNG files are the shared reference maps from the nonuniform result. The script plots saved predictions; it does not train the model.
