"""The brief's required RMSE figure, over seeds: Z500 and T850 over a 5-day roll-out against
persistence and the WeatherBench 2 climatology.

experiment.py writes the same comparison with one line per run and seed; for the report each
model is drawn as its seed mean with the seed range shaded, so three seeds read as one result.
Persistence and climatology are the same for every run (checked: identical across all nine).

    python explore/rmse_figure.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extra_figures import AXIS, GRID, INK, INK_2, MUTED, RUNS, seeds_of

OUT = Path("explore/figures")
VARIABLES = {"Z500": "Z500 RMSE (m² s⁻²)", "T850": "T850 RMSE (K)"}


def main() -> None:
    rows = []
    for run in RUNS:
        if not (Path("runs") / run).exists():
            continue
        for seed in seeds_of(run):
            d = pd.read_csv(Path("runs") / run / f"seed{seed}" / "scores.csv")
            rows.append(d[d.variable.isin(list(VARIABLES))].assign(seed=seed))
    scores = pd.concat(rows)
    scores["lead_days"] = scores.lead_hours / 24

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    table = []
    for ax, (var, ylabel) in zip(axes, VARIABLES.items()):
        v = scores[scores.variable == var]
        for run, (name, colour, style_) in RUNS.items():
            m = v[(v.run == run) & (v.metric == "rmse")].groupby("lead_days").value
            if m.ngroups == 0:
                continue
            stats = m.agg(["mean", "min", "max", "std", "count"])
            table.append(stats.reset_index().assign(variable=var, model=run))
            ax.fill_between(stats.index, stats["min"], stats["max"], color=colour, alpha=0.15, linewidth=0)
            ax.plot(stats.index, stats["mean"], color=colour, linewidth=1.8, linestyle=style_,
                    label=f"{name} ({int(stats['count'].max())} seeds)")

        references = v[v.run == "baseline17"].groupby(["metric", "lead_days"]).value.first()
        for metric, style_, label in (("rmse_persistence", "-.", "Persistence"),
                                      ("rmse_climatology", ":", "WB2 climatology")):
            ref = references[metric]
            ax.plot(ref.index, ref.values, style_, color=INK, linewidth=1.3, label=label)
            ax.annotate(label, (ref.index[-1], ref.values[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK_2, va="center")
            table.append(ref.rename("mean").reset_index().assign(variable=var, model=metric))

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 5.9)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(6))
        ax.set_xlabel("lead time (days)", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_title(var, color=INK, fontsize=10, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.12), labelcolor=INK_2)
    fig.text(0.01, -0.02, "Test years 2017–2019, 864 initialisations. Lines: seed mean; shading: range over seeds.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"F1_rmse_z500_t850.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = pd.concat(table)
    out.to_csv(OUT / "F1_rmse_z500_t850.csv", index=False)
    print(f"wrote {OUT}/F1_rmse_z500_t850.png/.pdf/.csv")
    view = out[out.lead_days.isin([0.25, 1, 3, 5])].pivot_table(index=["variable", "model"], columns="lead_days",
                                                                values="mean")
    print(view.round(2).to_string())


if __name__ == "__main__":
    main()
