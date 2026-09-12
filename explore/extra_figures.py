"""The two additional verification figures of the report, compared across runs.

E1  forecast zonal power spectrum divided by the ERA5 spectrum at the same valid times.
    RMSE rewards blurring (a smooth field is never far from the truth), so a model can win
    on RMSE while losing its small scales. The spectrum ratio shows that directly:
    1 = the forecast carries as much variance at that scale as reality, below 1 = smoothed.

E2  RMSE change against the baseline for every variable and lead, as a heatmap.
    The required figure covers Z500 and T850 only, both large-scale fields. Hierarchical-scale
    biases should matter most for fields with small-scale structure (humidity, precipitation),
    so this shows where each variant gains and loses across all seventeen variables.

Only runs that have finished (run.json written) are drawn, so the script can be rerun as runs land.

    python explore/extra_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import truth_at

xr.set_options(use_bottleneck=False)

OUT = Path("explore/figures")
RUNS = {  # fixed order and colour: colour follows the model, never its rank
    "baseline17": ("Baseline ViT", "#2a78d6"),
    "multiscale": ("A: multi-scale patches", "#eb6834"),
    "pyramid": ("C: pyramid loss", "#1baf7a"),
}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

SPECTRUM_VARIABLES = ["Z500", "T850", "Q850", "TP6h"]
SPECTRUM_LEADS = [24, 120]
INIT_STRIDE = 4   # every 4th of the 864 initialisations keeps it quick; spectra average smoothly

HEATMAP_ORDER = ["Z850", "Z500", "Z250", "T850", "T500", "T250", "T2M",
                 "U850", "U500", "U250", "V850", "V500", "V250", "Q850", "Q500", "Q250", "TP6h"]

EARTH_CIRCUMFERENCE_KM = 40_075


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def finished_runs() -> list:
    return [r for r in RUNS if (Path("runs") / r / "seed0" / "run.json").exists()]


# ------------------------------------------------------------------------- E1
def zonal_spectrum(field: xr.DataArray) -> np.ndarray:
    """Power per zonal wavenumber k = 1..32, area-weighted over latitude, averaged over time."""
    x = field.transpose(..., "latitude", "longitude").values.astype(np.float64)
    x = x.reshape(-1, *x.shape[-2:])                        # (samples, lat, lon)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2            # (samples, lat, k)
    weights = np.cos(np.deg2rad(field.latitude.values))
    weights = weights / weights.mean()
    return (power * weights[None, :, None]).mean(axis=(0, 1))[1:]


def figure_spectra(runs: list, era5: xr.Dataset) -> pd.DataFrame:
    k = np.arange(1, 33)
    rows = []
    fig, axes = plt.subplots(len(SPECTRUM_LEADS), len(SPECTRUM_VARIABLES),
                             figsize=(11, 5.2), sharex=True, squeeze=False)
    for run in runs:
        forecasts = xr.open_zarr(Path("runs") / run / "seed0" / "forecasts.zarr")
        forecasts = forecasts[SPECTRUM_VARIABLES].isel(time=slice(0, None, INIT_STRIDE))
        for i, lead in enumerate(SPECTRUM_LEADS):
            f = forecasts.sel(prediction_timedelta=np.timedelta64(lead, "h")).load()
            t = truth_at(era5, f).load()   # the scalar lead broadcasts: ERA5 at time + lead
            for j, var in enumerate(SPECTRUM_VARIABLES):
                ratio = zonal_spectrum(f[var]) / zonal_spectrum(t[var])
                name, colour = RUNS[run]
                axes[i, j].plot(k, ratio, color=colour, linewidth=2, label=name)
                rows += [dict(run=run, variable=var, lead_hours=lead, wavenumber=int(kk), power_ratio=float(r))
                         for kk, r in zip(k, ratio)]

    for i, lead in enumerate(SPECTRUM_LEADS):
        for j, var in enumerate(SPECTRUM_VARIABLES):
            ax = axes[i, j]
            style(ax)
            ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
            # the baseline's 4x4 patches repeat 64 / 4 = 16 times round a latitude circle
            ax.axvline(16, color=MUTED, linewidth=1, linestyle=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([1, 2, 4, 8, 16, 32])
            ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(var, color=INK, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(f"lead {lead // 24} d\nforecast / ERA5 power", color=INK_2, fontsize=9)
            if i == len(SPECTRUM_LEADS) - 1:
                ax.set_xlabel("zonal wavenumber k", color=INK_2, fontsize=9)
    axes[0, 0].text(1.1, 1.15, "ERA5 = 1", color=INK, fontsize=8)
    axes[0, 0].text(15, 0.9, "4×4 patch\nperiod", color=MUTED, fontsize=7, ha="right", va="top",
                    transform=axes[0, 0].get_xaxis_transform())

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), labelcolor=INK_2)
    fig.suptitle("Below the dashed line the forecast has lost variance at that scale (blurring); "
                 "above it, it carries variance ERA5 does not (spurious noise). "
                 f"Wavelength at the equator = {EARTH_CIRCUMFERENCE_KM:,} km / k.",
                 y=-0.01, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"E1_spectrum_ratio.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- E2
DIVERGING = LinearSegmentedColormap.from_list(  # blue = better (lower RMSE), red = worse, gray = no change
    "better_worse", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8f8e", "#b73a3a"])


def figure_heatmap(runs: list) -> pd.DataFrame:
    variants = [r for r in runs if r != "baseline17"]
    scores = pd.concat(pd.read_csv(Path("runs") / r / "seed0" / "scores.csv") for r in runs)
    rmse = scores[scores.metric == "rmse"].pivot_table(index=["variable", "lead_hours"], columns="run", values="value")
    leads = sorted(rmse.index.get_level_values("lead_hours").unique())

    fig, axes = plt.subplots(1, len(variants), figsize=(5.4 * len(variants), 5.6), squeeze=False, sharey=True)
    table = []
    limit = 30
    for ax, run in zip(axes[0], variants):
        change = (rmse[run] - rmse["baseline17"]) / rmse["baseline17"] * 100
        grid = change.unstack("lead_hours").reindex(HEATMAP_ORDER)
        table.append(grid.assign(run=run).reset_index())
        image = ax.imshow(grid.values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(RUNS[run][0], color=INK, fontsize=10, loc="left")
        ax.set_yticks(range(len(HEATMAP_ORDER)))
        ax.set_yticklabels(HEATMAP_ORDER, fontsize=8, color=INK_2)
        ticks = [i for i, h in enumerate(leads) if h % 24 == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(leads[i] // 24)}" for i in ticks], fontsize=8, color=INK_2)
        ax.set_xlabel("lead time (days)", color=INK_2, fontsize=9)
        for boundary in (2.5, 6.5, 9.5, 12.5):   # separate Z | T | U | V | Q+TP
            ax.axhline(boundary, color="#ffffff", linewidth=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # label only the 5-day column, where the prediction says the difference should be largest
        last = len(leads) - 1
        for row, value in enumerate(grid.values[:, last]):
            ax.text(last, row, f"{value:+.0f}", ha="center", va="center", fontsize=6,
                    color=INK if abs(value) < 0.6 * limit else "#ffffff")

    bar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, pad=0.02, extend="both")
    bar.set_label("RMSE change vs baseline (%)   blue = better, red = worse", color=INK_2, fontsize=9)
    bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK_2)
    bar.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"E2_rmse_change_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.concat(table)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = finished_runs()
    print(f"finished runs: {runs}")
    era5 = xr.open_zarr("data/era5_eval_5p6.zarr")

    spectra = figure_spectra(runs, era5)
    spectra.to_csv(OUT / "E1_spectrum_ratio.csv", index=False)
    print(f"wrote {OUT}/E1_spectrum_ratio.png/.pdf/.csv")
    summary = spectra[spectra.wavenumber >= 9].groupby(["variable", "lead_hours", "run"]).power_ratio.mean()
    print("\nmean power ratio at small scales (k >= 9); 1 = realistic, < 1 = blurred")
    print(summary.unstack("run").round(3).to_string())

    if "baseline17" in runs and len(runs) > 1:
        table = figure_heatmap(runs)
        table.to_csv(OUT / "E2_rmse_change_heatmap.csv", index=False)
        print(f"\nwrote {OUT}/E2_rmse_change_heatmap.png/.pdf/.csv")
    else:
        print("\nE2 needs the baseline and at least one variant; skipped for now")


if __name__ == "__main__":
    sys.exit(main())
