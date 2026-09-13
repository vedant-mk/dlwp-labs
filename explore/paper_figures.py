"""The two required figures still missing from the report.

F4  a T850 forecast at 1-day lead (brief: "a plot that shows a forecast of T850 after 1 day of lead-time").
    One initialisation (the first of the test period, as experiment.py plots) for ERA5 and the three
    models the paper centres on: the baseline and the two ways, A' (architecture) and C' (loss).
    Top row: the fields on one colour scale; bottom row: forecast minus ERA5 on one diverging scale.

F5  training and validation loss (brief: "a plot that shows the validation and training loss over epochs").
    Seed means with the range over seeds, against training steps with epochs on the top axis.
    Validation loss is plain MSE on 2016 for every model, so it compares across models; the training
    loss of the pyramid runs is a different objective, and the jump at step 4000 is the switch to
    roll-out training.

    python explore/paper_figures.py
"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extra_figures import AXIS, DIVERGING, GRID, INK, INK_2, MUTED, RUNS, forecast_sample, seeds_of
from utils.metrics import truth_at

xr.set_options(use_bottleneck=False)
OUT = Path("explore/figures")
EVAL = Path("data/era5_eval_5p6.zarr")

F4_MODELS = ["baseline17", "multiscale_smooth", "pyramid_w2"]
SHORT = {"baseline17": "Baseline", "multiscale_smooth": "A′ (architecture)", "pyramid_w2": "C′ (loss)"}
WARM = LinearSegmentedColormap.from_list("warm", ["#fdf3ea", "#f3b58a", "#d9622b", "#8a2b0b"])   # one hue, light to dark
ROLLOUT_STEP = 4000
STEPS_PER_EPOCH = 638


def weighted_rmse(error: np.ndarray, latitude: np.ndarray) -> float:
    w = np.cos(np.deg2rad(latitude))[:, None] * np.ones_like(error)
    return float(np.sqrt((w * error ** 2).sum() / w.sum()))


def figure_t850() -> None:
    lead = np.timedelta64(24, "h")
    fields, init = {}, None
    for run in F4_MODELS:
        sample = forecast_sample(run, 0)["T850"].sel(prediction_timedelta=lead)
        init = sample.time.values[0] if init is None else init
        fields[run] = sample.sel(time=init).load()
    era5 = xr.open_zarr(EVAL)
    truth = truth_at(era5, fields["baseline17"].to_dataset().expand_dims(time=[init]))["T850"].isel(time=0).load()

    lat, lon = truth.latitude.values, truth.longitude.values
    # the 64 columns span exactly 360 degrees with edges at -2.8125 .. 357.1875; rolling half a circle
    # puts Greenwich in the middle, and a plate carree centred on the grid's own centre fits it exactly,
    # so the field is drawn as an image with no reprojection (Robinson breaks with this cartopy/shapely)
    half = len(lon) // 2
    centre = -(lon[1] - lon[0]) / 2
    roll = lambda a: np.roll(a, half, axis=1)
    vmin, vmax = np.percentile(truth.values, [1, 99])
    errors = {run: fields[run].values - truth.values for run in F4_MODELS}
    elim = float(np.ceil(np.percentile(np.abs(np.stack(list(errors.values()))), 99)))

    fig = plt.figure(figsize=(6.3, 2.55))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.045], hspace=0.32, wspace=0.07)
    projection = ccrs.PlateCarree(central_longitude=centre)

    def panel(row, col, values, cmap, lo, hi, title):
        ax = fig.add_subplot(grid[row, col], projection=projection)
        mesh = ax.imshow(roll(values), cmap=cmap, vmin=lo, vmax=hi, origin="lower",
                         extent=[-180, 180, lat[0] - 2.8125, lat[-1] + 2.8125], transform=projection,
                         interpolation="nearest")
        ax.set_extent([-180, 180, -90, 90], crs=projection)
        ax.coastlines(linewidth=0.25, color="#52514e")
        ax.set_title(title, fontsize=6, color=INK, loc="left", pad=2)
        return mesh

    field_mesh = panel(0, 0, truth.values, WARM, vmin, vmax, "ERA5 (truth)")
    for col, run in enumerate(F4_MODELS, start=1):
        panel(0, col, fields[run].values, WARM, vmin, vmax, f"{SHORT[run]}")
        error_mesh = panel(1, col, errors[run], DIVERGING, -elim, elim,
                           f"error · RMSE {weighted_rmse(errors[run], lat):.2f} K")

    key = fig.add_subplot(grid[1, 0]); key.axis("off")
    stamp = pd.Timestamp(init)
    key.text(0.0, 0.95, f"T850, 1-day lead\ninitialised {stamp:%Y-%m-%d %H} UTC\nvalid {stamp + pd.Timedelta(hours=24):%Y-%m-%d %H} UTC\nseed 0\n\nerrors: blue = too cold,\nred = too warm",
             va="top", fontsize=5.5, color=INK_2)
    cb1 = fig.colorbar(field_mesh, cax=fig.add_subplot(grid[0, 4]))
    cb1.set_label("T850 (K)", fontsize=6, color=INK_2)
    cb2 = fig.colorbar(error_mesh, cax=fig.add_subplot(grid[1, 4]), extend="both")
    cb2.set_label("error (K)", fontsize=6, color=INK_2)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=5.5, colors=MUTED, labelcolor=INK_2)
        cb.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F4_t850_forecast_24h.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/F4_t850_forecast_24h.png/.pdf  (init {stamp:%Y-%m-%d %H} UTC, error scale ±{elim:.0f} K)")
    for run in F4_MODELS:
        print(f"  {SHORT[run]:20} RMSE {weighted_rmse(errors[run], lat):.3f} K")


def figure_losses() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for run, (name, colour, style_) in RUNS.items():
        seeds = seeds_of(run)
        logs = pd.concat(pd.read_csv(Path("runs") / run / f"seed{s}" / "metrics.csv").assign(seed=s) for s in seeds)

        train = logs.dropna(subset=["train/loss"])
        smooth = (train.sort_values("step").groupby("seed", group_keys=False)
                  .apply(lambda g: g.assign(loss=g["train/loss"].rolling(25, min_periods=1).mean())))
        t = smooth.groupby("step").loss.agg(["mean", "min", "max"])
        axes[0].fill_between(t.index, t["min"], t["max"], color=colour, alpha=0.12, linewidth=0)
        axes[0].plot(t.index, t["mean"], color=colour, linestyle=style_, linewidth=1.0, label=name)

        v = logs.dropna(subset=["val/loss_step1"]).groupby("step")["val/loss_step1"].agg(["mean", "min", "max"])
        axes[1].fill_between(v.index, v["min"], v["max"], color=colour, alpha=0.12, linewidth=0)
        axes[1].plot(v.index, v["mean"], color=colour, linestyle=style_, linewidth=1.0, marker="o", markersize=1.8)

    titles = ["Training loss (own objective)", "Validation loss, 2016 (plain MSE)"]
    for ax, title in zip(axes, titles):
        ax.axvline(ROLLOUT_STEP, color=MUTED, linestyle=":", linewidth=0.8)
        ax.text(ROLLOUT_STEP + 60, 0.97, "roll-out\ntraining", transform=ax.get_xaxis_transform(), fontsize=5.5,
                color=MUTED, va="top")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        # label minor ticks whose leading digit is 2, 3 or 5 (0.02, 0.03, 0.05 ...) so a log axis stays readable
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if f"{y:.0e}"[0] in "235" else ""))
        ax.set_xlim(0, 6000)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=6.5)
        ax.tick_params(which="minor", colors=MUTED, labelcolor=INK_2, labelsize=6.5)
        ax.grid(color=GRID, linewidth=0.4, which="both")
        ax.set_axisbelow(True)
        ax.set_xlabel("training step", color=INK_2, fontsize=7)
        ax.set_ylabel("loss (standardised)", color=INK_2, fontsize=7)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        epochs = ax.secondary_xaxis("top", functions=(lambda s: s / STEPS_PER_EPOCH, lambda e: e * STEPS_PER_EPOCH))
        epochs.set_xlabel("epoch", color=MUTED, fontsize=6)
        epochs.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=5.5)
        epochs.spines["top"].set_color(AXIS)
    axes[0].set_ylim(top=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.12),
               labelcolor=INK_2, handlelength=2.5, columnspacing=1.0)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F5_loss_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/F5_loss_curves.png/.pdf")


if __name__ == "__main__":
    figure_t850()
    figure_losses()
