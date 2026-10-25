# Figure 8 cavity results

`manuscript_Figure8.png` is the revised Figure 8 embedded in the 2026-10-25 manuscript. Its field panels match the outputs of `plot_fvm_figure9_panels.py`, and its boxplots match `plot_fvm_figure9_boxplots.py`. Those scripts used Figure 9 as the working number before the manuscript figures were renumbered. The checkpoint, three field-comparison arrays, plotted panels, and boxplot error samples are in `fvm-results/`.

The calculation uses finite-volume reference fields. Run `generate_fvm_dataset.py`, `train_fvm_pirn.py`, then the two plotting scripts to regenerate the component panels; see `FVM_ONLY_REPRODUCTION.md` for the numerical setup and error definition. The previously stored OpenFOAM-derived data are not the reference fields for this revised figure.

The boxplots were arranged differently from the field rows when the final image was assembled. The left rows are training (`Re=15`), interpolation (`Re=20.1`), and extrapolation (`Re=30`); the right panels are training, extrapolation, and interpolation. Thus `(c2)` and `(d2)` do not match the adjacent rows or the caption. The source boxplot images retain their original names: `c2` is training, `d2` is extrapolation, and `b2` is interpolation.
