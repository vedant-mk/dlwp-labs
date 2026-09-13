"""Judge the five pre-registered hypotheses (explore/DATA_NOTES.md) against all seeds.

Rule, fixed before seeds 1 and 2 finished: a difference is real only if every seed agrees in sign
AND |mean| >= 2 x the seed standard deviation. Run explore/extra_figures.py first (H4 reads its CSV).

    python explore/seed_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extra_figures import is_real, seeds_of

OUT = Path("explore/figures")
LEADS = [6, 24, 72, 120]


def rmse_table() -> pd.DataFrame:
    rows = []
    for run in ("baseline17", "pyramid", "multiscale"):
        for seed in seeds_of(run):
            d = pd.read_csv(Path("runs") / run / f"seed{seed}" / "scores.csv")
            rows.append(d[d.metric == "rmse"].assign(seed=seed))
    return pd.concat(rows).pivot_table(index=["variable", "lead_hours", "seed"], columns="run", values="value")


def change(rmse: pd.DataFrame, run: str, variable: str, lead: int) -> np.ndarray:
    g = rmse.xs((variable, lead), level=("variable", "lead_hours"))
    return ((g[run] / g["baseline17"] - 1) * 100).dropna().values


def describe(values: np.ndarray) -> str:
    return f"{values.mean():+.1f}% ± {values.std(ddof=1):.1f} ({', '.join(f'{v:+.1f}' for v in values)})"


def main() -> None:
    rmse = rmse_table()
    lines = ["# Seed analysis against the pre-registered hypotheses", "",
             "Rule: real only if all seeds agree in sign and |mean| >= 2 x seed std. "
             "Values: mean ± std (per-seed values).", ""]
    verdicts = {}

    # H1 -- baseline reproducible: spread within ~1-2%
    lines += ["## H1: baseline is reproducible (seed spread within ~1–2%)", "",
              "| variable | lead (h) | mean RMSE | spread (% of mean) |", "|---|---|---|---|"]
    worst = 0.0
    for var in ("Z500", "T850"):
        for lead in LEADS:
            v = rmse.xs((var, lead), level=("variable", "lead_hours"))["baseline17"].values
            spread = v.std(ddof=1) / v.mean() * 100
            worst = max(worst, spread)
            lines.append(f"| {var} | {lead} | {v.mean():.2f} | {spread:.2f}% |")
    verdicts["H1"] = worst <= 2.0
    lines += ["", f"Largest spread {worst:.2f}% → **{'HOLDS' if verdicts['H1'] else 'FAILS'}**", ""]

    # H2 -- A improves humidity 5-8% at 1-3 days
    lines += ["## H2: A improves humidity RMSE by ~5–8% at 1–3 days", "",
              "| variable | lead (h) | multiscale vs baseline | real & improving? |", "|---|---|---|---|"]
    ok = []
    for var in ("Q850", "Q500", "Q250"):
        for lead in (24, 48, 72):
            c = change(rmse, "multiscale", var, lead)
            good = is_real(c) and c.mean() < 0
            ok.append(good)
            lines.append(f"| {var} | {lead} | {describe(c)} | {'yes' if good else 'no'} |")
    verdicts["H2"] = all(ok)
    lines += ["", f"{sum(ok)}/{len(ok)} cells real improvements → **{'HOLDS' if verdicts['H2'] else 'PARTLY' if any(ok) else 'FAILS'}**", ""]

    # H3 -- A degrades Z500, T850, T2M at 5 days
    lines += ["## H3: A degrades smooth large-scale fields at 5 days", "",
              "| variable | multiscale vs baseline at 120 h | real & worse? |", "|---|---|---|"]
    ok = []
    for var in ("Z500", "T850", "T2M"):
        c = change(rmse, "multiscale", var, 120)
        bad = is_real(c) and c.mean() > 0
        ok.append(bad)
        lines.append(f"| {var} | {describe(c)} | {'yes' if bad else 'no'} |")
    verdicts["H3"] = all(ok)
    lines += ["", f"**{'HOLDS' if verdicts['H3'] else 'FAILS'}**", ""]

    # H4 -- C cuts spurious small-scale power by ~15-30%
    spectra = pd.read_csv(OUT / "E1_spectrum_ratio.csv")
    small = spectra[spectra.wavenumber >= 9].groupby(["run", "seed", "variable", "lead_hours"]).power_ratio.mean()
    lines += ["## H4: C reduces spurious small-scale power (mean ratio at k ≥ 9) by ~15–30%", "",
              "Per-seed pairing: seed k of pyramid against seed k of the baseline.", "",
              "| variable | lead (h) | baseline ratio | pyramid ratio | change | real reduction? |", "|---|---|---|---|---|---|"]
    ok = []
    for var in ("Z500", "T850", "Q850"):
        for lead in (24, 120):
            b = small.xs(("baseline17", var, lead), level=("run", "variable", "lead_hours"))
            p = small.xs(("pyramid", var, lead), level=("run", "variable", "lead_hours"))
            seeds = sorted(set(b.index) & set(p.index))
            c = np.array([(p[s] / b[s] - 1) * 100 for s in seeds])
            good = is_real(c) and c.mean() < 0
            ok.append(good)
            lines.append(f"| {var} | {lead} | {b[seeds].mean():.2f} | {p[seeds].mean():.2f} | {describe(c)} | {'yes' if good else 'no'} |")
    verdicts["H4"] = all(ok)
    lines += ["", "TP6h is excluded: it is blurred (ratio < 1), not noisy, so 'spurious power' does not apply.", "",
              f"{sum(ok)}/{len(ok)} cells real reductions → **{'HOLDS' if verdicts['H4'] else 'PARTLY' if any(ok) else 'FAILS'}**", ""]

    # H5 -- C's Z500/T850 RMSE within seed noise of the baseline
    lines += ["## H5: C's Z500/T850 RMSE is within seed noise of the baseline", "",
              "| variable | lead (h) | pyramid vs baseline | within noise? |", "|---|---|---|---|"]
    ok = []
    for var in ("Z500", "T850"):
        for lead in LEADS:
            c = change(rmse, "pyramid", var, lead)
            noise = not is_real(c)
            ok.append(noise)
            lines.append(f"| {var} | {lead} | {describe(c)} | {'yes' if noise else 'no, a real difference'} |")
    verdicts["H5"] = all(ok)
    lines += ["", f"{sum(ok)}/{len(ok)} cells within noise → **{'HOLDS' if verdicts['H5'] else 'FAILS'}**", ""]

    lines += ["## Summary", "", "| hypothesis | verdict |", "|---|---|"]
    lines += [f"| {h} | {'holds' if v else 'fails (see table)'} |" for h, v in verdicts.items()]
    (OUT / "seed_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
