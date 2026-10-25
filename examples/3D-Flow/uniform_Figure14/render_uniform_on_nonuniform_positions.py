"""Replot the 38k uniform PIRN at the original nonuniform sample positions."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


out = Path(__file__).resolve().parent
nonuniform_dir = out.parent / "nonuniform_Figure12"
uniform = np.load(out / "uniform_predictions.npz")
nonuniform = np.load(nonuniform_dir / "three_case_fields.npz")
uniform_result = json.loads((out / "uniform_result.json").read_text(encoding="utf-8"))
nonuniform_result = json.loads((nonuniform_dir / "snapshot_metadata.json").read_text(encoding="utf-8"))
assert uniform_result["best_epoch"] == 37775
assert nonuniform_result["checkpoint_epoch"] == 41925

ui = np.flatnonzero((uniform["physical_pairs"] == [40000, 500]).all(axis=1)).item()
ni = np.flatnonzero((nonuniform["physical_pairs"] == [40000, 500]).all(axis=1)).item()
ux, uy, uz = (uniform[k].astype(float) for k in ("x", "y", "z"))
nx, ny, nz = (nonuniform[k].astype(float) for k in ("x", "y", "z"))
sl = np.flatnonzero(np.isclose(uz, 0.16, atol=1e-10)).item()
assert np.array_equal(uz, nz) and sl == int(nonuniform["slice_index"])
assert np.all(np.diff(ux) > 0) and np.all(np.diff(uy) > 0)
assert np.all(np.diff(nx) > 0) and np.all(np.diff(ny) > 0)
assert ux[0] <= nx[0] and nx[-1] <= ux[-1]
assert uy[0] <= ny[0] and ny[-1] <= uy[-1]


def resample(field, source_x, source_y, target_x, target_y):
    """Separable bilinear interpolation, preserving axis order [y, x]."""
    along_x = np.stack([np.interp(target_x, source_x, line) for line in field])
    return np.stack([np.interp(target_y, source_y, along_x[:, j])
                     for j in range(len(target_x))], axis=1)


display_x = np.linspace(0.0, 0.04, 320)
display_y = np.linspace(0.0, 0.04, 320)


def display_grid(field):
    """Match the existing nonuniform standalone plot treatment exactly."""
    xq = np.clip(display_x, nx[0], nx[-1])
    yq = np.clip(display_y, ny[0], ny[-1])
    return resample(field, nx, ny, xq, yq)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})
metrics = []
limits = {}
for field_idx, field_name, unit in ((1, "w", "m/s"), (0, "T_rise", "K")):
    uniform_native = uniform["prediction"][ui, field_idx, :, :, sl]
    uniform_at_nonuniform = resample(uniform_native, ux, uy, nx, ny)
    shared_cfd = nonuniform["reference"][ni, field_idx, :, :, sl]
    nonuniform_pred = nonuniform["prediction"][ni, field_idx, :, :, sl]
    assert uniform_at_nonuniform.shape == shared_cfd.shape == (32, 32)
    denominator = max(float(np.abs(shared_cfd).max()), 1e-30)
    error_native = np.abs(uniform_at_nonuniform - shared_cfd) / denominator
    pirn = display_grid(uniform_at_nonuniform)
    error = display_grid(error_native)

    # Retain the nonuniform standalone PIRN/CFD color range for direct visual comparison.
    vmin = float(min(nonuniform_pred.min(), shared_cfd.min()))
    vmax = float(max(nonuniform_pred.max(), shared_cfd.max()))
    if field_idx == 1:
        bound = 1.05 * max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    else:
        padding = 0.05 * (vmax - vmin)
        vmin, vmax = max(0.0, vmin - padding), vmax + padding
    assert uniform_at_nonuniform.min() >= vmin and uniform_at_nonuniform.max() <= vmax
    error_color_max = 1.05 * float(error.max())
    limits[field_name] = dict(PIRN_CFD_vmin=vmin, PIRN_CFD_vmax=vmax,
                              error_color_max=error_color_max,
                              CFD_denominator=denominator)
    metrics.append(dict(field=field_name, Gr=40000, Ha=500, z_m=float(uz[sl]),
                        max_normalized_error_pct=100 * float(error_native.max()),
                        relative_L2_pct=100 * float(np.linalg.norm(uniform_at_nonuniform - shared_cfd)
                                                     / np.linalg.norm(shared_cfd))))

    for kind, array, cmap, low, high, title in (
        ("PIRN", pirn, "jet", vmin, vmax, "PIRN"),
        ("error", error, "Blues", 0.0, error_color_max, "Normalized absolute error"),
    ):
        fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
        image = ax.imshow(array, origin="lower", extent=(0, .04, 0, .04),
                          interpolation="bilinear", cmap=cmap,
                          vmin=low, vmax=high, aspect="equal")
        ax.set_xlim(0, .04)
        ax.set_ylim(0, .04)
        ax.set_xticks([0, .02, .04])
        ax.set_yticks([0, .02, .04])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.tick_params(direction="out", length=3, width=.7)
        ax.set_title(title, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, shrink=.92, pad=.04)
        if kind == "error":
            bar.set_label("fraction of max |CFD slice|", labelpad=8)
        else:
            bar.set_label(unit, labelpad=8)
            if field_idx == 1:
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-5, -5))
                bar.formatter = formatter
                bar.update_ticks()
        bar.outline.set_visible(False)
        stem = f"uniform38k_{field_name}_{kind}_Gr40000_Ha500_nonuniform_positions"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

with (out / "uniform38k_on_nonuniform_positions_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)

provenance = {
    "uniform_experiment": "uniform_predictions.npz",
    "uniform_best_epoch": int(uniform_result["best_epoch"]),
    "nonuniform_experiment": "../nonuniform_Figure12/three_case_fields.npz",
    "nonuniform_best_epoch": int(nonuniform_result["checkpoint_epoch"]),
    "common_positions": "native nonuniform 32x32 x/y at z=0.16 m",
    "uniform_prediction_mapping": "bilinear interpolation of uniform 32x32 prediction to native nonuniform x/y",
    "shared_CFD": "native nonuniform CFD at Gr40000 Ha500 z=0.16 m",
    "error": "abs(interpolated uniform PIRN - nonuniform CFD) / max(abs(nonuniform CFD native slice))",
    "display": "same 320x320 display interpolation and nearest-edge extension as existing nonuniform standalone output",
    "limits": limits,
}
(out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print("Rendered uniform 38k PIRN and error maps on native nonuniform positions")
