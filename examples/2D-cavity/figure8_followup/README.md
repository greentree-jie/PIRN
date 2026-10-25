# Figure 8 cavity data and finite-volume check

`manuscript_Figure8.png` is the revised Figure 8 embedded in the 2026-10-25 manuscript. The matching checkpoint and prediction arrays have not been located, so the error values in that image cannot yet be regenerated from this directory.

The scripts provide a separate finite-volume calculation: generate cavity reference fields, train a PIRN model, and plot its predictions. The methods and error definitions are in `FVM_ONLY_REPRODUCTION.md`. These results use finite-volume reference fields, whereas the manuscript figure uses OpenFOAM-derived fields. `Figure9_FVM_original_error_all_cases.png` retains the numbering used when that calculation was made.

In the revised manuscript image, `(c2)` is labeled extrapolation and `(d2)` interpolation; the corresponding field rows and figure caption give the opposite order. These labels need to be reconciled using the matched arrays.
