"""Render manuscript Figure 12 panels from frozen, selected three-case arrays."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import numpy as np


OUT = Path(__file__).resolve().parent
META = json.loads((OUT / "snapshot_metadata.json").read_text(encoding="utf-8"))
DATA = np.load(OUT / "three_case_fields.npz")
PREDICTIONS = DATA["prediction"]
REF = DATA["reference"]
PAIRS = DATA["physical_pairs"]
X, Y, Z = [DATA[k].astype(float) for k in ("x", "y", "z")]
SL = int(DATA["slice_index"])
assert abs(Z[SL] - 0.16) < 1e-8
assert np.all(np.diff(X) > 0) and np.all(np.diff(Y) > 0)

CASES = [
    (10000, 100, "Training"),
    (30000, 300, "Interpolation validation"),
    (40000, 500, "Extrapolation test"),
]
assert PAIRS.tolist() == [[gr, ha] for gr, ha, _ in CASES]
assert np.all(np.isfinite(PREDICTIONS)) and np.all(np.isfinite(REF))

# Native arrays are ordered (y, x). Interpolate inside the sample range and
# extend nearest edge samples to the 0.04 m physical border for display only.
DX = np.linspace(0.0, 0.04, 320)
DY = np.linspace(0.0, 0.04, 320)
def display_grid(field):
    xq = np.clip(DX, X[0], X[-1])
    yq = np.clip(DY, Y[0], Y[-1])
    x_interp = np.stack([np.interp(xq, X, row) for row in field])
    return np.stack([np.interp(yq, Y, x_interp[:, j])
                     for j in range(len(xq))], axis=1)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.1,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})

def colorbar(fig, mappable, cax, low, high, unit, error=False):
    ticks = np.linspace(low, high, 5)
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    if error:
        exponent = int(np.floor(np.log10(high))) if high < 0.1 else 0
    else:
        exponent = int(np.floor(np.log10(max(abs(low), abs(high)))))
    scale = 10.0 ** exponent
    precision = 3 if error else 2
    cb.formatter = FuncFormatter(
        lambda value, _position: f"{0.0 if abs(value / scale) < 1e-10 else value / scale:.{precision}f}"
    )
    cb.update_ticks()
    cb.ax.tick_params(labelsize=8.0, width=0.6, length=2.2, pad=2)
    cb.outline.set_linewidth(0.7)
    if error:
        cb.set_label("fraction", fontsize=8.8, labelpad=4)
    scale_text = rf"$\times 10^{{{exponent}}}$" if exponent != 0 else ""
    header = f"{scale_text} {unit}".strip() if not error else scale_text
    if header:
        cb.ax.text(0.5, 1.025, header,
                   transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=8.8)

METRICS = []
for field_idx, filename, field_label, unit in [
    (1, "Figure12_velocity", "Axial velocity w", "m/s"),
    (0, "Figure12_temperature", "Temperature rise T − 723.15 K", "K"),
]:
    fig = plt.figure(figsize=(12.8, 11.6), facecolor="white")
    gs = GridSpec(
        3, 6, figure=fig, width_ratios=[1, 0.036, 1, 0.036, 1, 0.036],
        left=0.06, right=0.96, top=0.91, bottom=0.075,
        wspace=0.19, hspace=0.38,
    )
    colorbar_pairs = []
    for row, (gr, ha, split) in enumerate(CASES):
        predicted = PREDICTIONS[row, field_idx, :, :, SL]
        reference = REF[row, field_idx, :, :, SL]
        denominator = max(float(np.max(np.abs(reference))), 1e-30)
        error = np.abs(predicted - reference) / denominator
        low = float(min(np.min(predicted), np.min(reference)))
        high = float(max(np.max(predicted), np.max(reference)))
        if low >= high:
            high = low + 1e-12
        if field_idx == 1:
            # Give both velocity maps a little headroom while keeping a
            # common, zero-centered color scale for PIRN and CFD.
            velocity_limit = 1.05 * max(abs(low), abs(high))
            low, high = -velocity_limit, velocity_limit
        error_high = max(float(np.max(error)), 1e-12)
        METRICS.append(dict(
            Gr=gr, Ha=ha, split=split, field="w" if field_idx else "T_rise",
            native_slice_max_normalized_error_fraction=error_high,
            native_slice_max_normalized_error_pct=100 * error_high,
            native_slice_CFD_denominator=denominator,
        ))
        displays = [
            (predicted, "PIRN", "jet", low, high, False),
            (reference, "CFD", "jet", low, high, False),
            (error, "Normalized absolute error", "Blues", 0.0, error_high, True),
        ]
        for col, (array, title, cmap, vmin, vmax, is_error) in enumerate(displays):
            ax = fig.add_subplot(gs[row, 2 * col])
            cax = fig.add_subplot(gs[row, 2 * col + 1])
            image = ax.imshow(
                display_grid(array), origin="lower", extent=(0, 0.04, 0, 0.04),
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="bilinear", aspect="equal",
            )
            ax.set_xlim(0, 0.04)
            ax.set_ylim(0, 0.04)
            ax.set_xticks([0, 0.02, 0.04])
            ax.set_yticks([0, 0.02, 0.04])
            ax.tick_params(direction="out", length=2.4, width=0.7, labelsize=8.8)
            ax.set_xlabel("x (m)", labelpad=2)
            if col == 0:
                ax.set_ylabel("y (m)", labelpad=2)
            heading = title
            ax.set_title(f"{heading}\n{split}\nGr = {gr}, Ha = {ha}",
                         fontsize=9.1, pad=8, linespacing=1.08)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            colorbar(fig, image, cax, vmin, vmax, unit, is_error)
            colorbar_pairs.append((ax, cax))
    # Equal-aspect maps shrink inside their GridSpec cells. Anchor each colorbar
    # to the actual map edge, so the cloud and colorbar remain visually close.
    fig.canvas.draw()
    for ax, cax in colorbar_pairs:
        box = ax.get_position()
        cax.set_position([box.x1 + 0.012, box.y0, 0.012, box.height])
    fig.suptitle(
        f"{field_label} | nonuniform PIRN validation-best epoch {META['checkpoint_epoch']} | z = {Z[SL]:.2f} m",
        y=0.985, fontsize=11.5,
    )
    fig.text(0.5, 0.018,
             "Normalized absolute error = |PIRN − CFD| / max(|CFD|) on each CFD slice; edge extension is display only.",
             ha="center", va="bottom", fontsize=8.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{filename}.{ext}", dpi=300)
    plt.close(fig)

with (OUT / "three_case_max_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(METRICS[0]))
    writer.writeheader()
    writer.writerows(METRICS)

print("Rendered three-case velocity and temperature Figure 12 at best epoch",
      META["checkpoint_epoch"], flush=True)
