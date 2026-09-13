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


def power_change(small: pd.Series, run: str, reference: str, variable: str, lead: int) -> np.ndarray:
    a = small.xs((run, variable, lead), level=("run", "variable", "lead_hours"))
    b = small.xs((reference, variable, lead), level=("run", "variable", "lead_hours"))
    seeds = sorted(set(a.index) & set(b.index))
    return np.array([(a[s] / b[s] - 1) * 100 for s in seeds])


def paired_change(rmse: pd.DataFrame, run: str, reference: str, variable: str, lead: int) -> np.ndarray:
    g = rmse.xs((variable, lead), level=("variable", "lead_hours"))
    return ((g[run] / g[reference] - 1) * 100).dropna().values


def round2() -> None:
    rows = []
    for run in ("baseline17", "multiscale", "pyramid", "baseline17_smooth", "multiscale_smooth", "pyramid_w2"):
        for seed in seeds_of(run):
            d = pd.read_csv(Path("runs") / run / f"seed{seed}" / "scores.csv")
            rows.append(d[d.metric == "rmse"].assign(seed=seed))
    rmse = pd.concat(rows).pivot_table(index=["variable", "lead_hours", "seed"], columns="run", values="value")
    spectra = pd.read_csv(OUT / "E1_spectrum_ratio.csv")
    small = spectra[spectra.wavenumber >= 9].groupby(["run", "seed", "variable", "lead_hours"]).power_ratio.mean()

    lines = ["# Round 2: pre-registered predictions P6–P9", "",
             "Rule: real only if all seeds agree in sign and |mean| >= 2 x seed std. Seeds paired by index.", ""]
    verdicts = {}

    lines += ["## P6: A′ fixes A's 5-day damage (seams cause it)", "",
              "| quantity | A′ vs A | real improvement? |", "|---|---|---|"]
    ok = []
    for var in ("Z500", "T850", "T2M"):
        c = paired_change(rmse, "multiscale_smooth", "multiscale", var, 120)
        good = is_real(c) and c.mean() < 0
        ok.append(good)
        lines.append(f"| RMSE {var}, 5 d | {describe(c)} | {'yes' if good else 'no'} |")
    for var in ("Z500", "T850", "Q850"):
        c = power_change(small, "multiscale_smooth", "multiscale", var, 120)
        good = is_real(c) and c.mean() < 0
        ok.append(good)
        lines.append(f"| spurious power k≥9 {var}, 5 d | {describe(c)} | {'yes' if good else 'no'} |")
    verdicts["P6"] = all(ok)
    lines += ["", f"{sum(ok)}/{len(ok)} → **{'HOLDS' if all(ok) else 'PARTLY' if any(ok) else 'FAILS'}**", ""]

    lines += ["## P7: A′ keeps a humidity improvement over the baseline at 1–3 days", "",
              "| variable | lead (h) | A′ vs baseline | real improvement? |", "|---|---|---|---|"]
    ok = []
    for var in ("Q850", "Q500", "Q250"):
        for lead in (24, 48, 72):
            c = paired_change(rmse, "multiscale_smooth", "baseline17", var, lead)
            good = is_real(c) and c.mean() < 0
            ok.append(good)
            lines.append(f"| {var} | {lead} | {describe(c)} | {'yes' if good else 'no'} |")
    verdicts["P7"] = all(ok)
    lines += ["", f"{sum(ok)}/{len(ok)} → **{'HOLDS' if all(ok) else 'PARTLY' if any(ok) else 'FAILS'}**", ""]

    lines += ["## P8: the smoother matters more for A than for the baseline (5 days)", "",
              "| variable | B′ vs baseline | A′ vs A | smaller for the baseline? |", "|---|---|---|---|"]
    ok = []
    for var in ("Z500", "T850"):
        b = paired_change(rmse, "baseline17_smooth", "baseline17", var, 120)
        a = paired_change(rmse, "multiscale_smooth", "multiscale", var, 120)
        good = abs(b.mean()) < abs(a.mean())
        ok.append(good)
        lines.append(f"| {var} | {describe(b)} | {describe(a)} | {'yes' if good else 'no'} |")
    verdicts["P8"] = all(ok)
    lines += ["", "Note: P8 compares magnitudes at 5 days only, as pre-registered. At 1–3 days the smoother's",
              "effect on the baseline is itself large (see the exploratory section).", "",
              f"**{'HOLDS' if all(ok) else 'FAILS'}**", ""]

    lines += ["## P9: milder pyramid weights (C′) cost less, but still reduce spurious power", "",
              "| quantity | comparison | value | as predicted? |", "|---|---|---|---|"]
    ok = []
    c = paired_change(rmse, "pyramid_w2", "pyramid", "Z500", 24)
    good = is_real(c) and c.mean() < 0
    ok.append(good)
    lines.append(f"| RMSE Z500, 1 d | C′ vs C | {describe(c)} | {'yes' if good else 'no'} |")
    for var in ("Z500", "T850"):
        for lead in (24, 120):
            cp = power_change(small, "pyramid_w2", "baseline17", var, lead)
            cc = power_change(small, "pyramid", "baseline17", var, lead)
            good = is_real(cp) and cp.mean() < 0 and abs(cp.mean()) < abs(cc.mean())
            ok.append(good)
            lines.append(f"| spurious power {var}, {lead} h | C′ vs baseline (C: {cc.mean():+.1f}%) | {describe(cp)} | {'yes' if good else 'no'} |")
    verdicts["P9"] = all(ok)
    lines += ["", f"{sum(ok)}/{len(ok)} → **{'HOLDS' if all(ok) else 'PARTLY' if any(ok) else 'FAILS'}**", ""]

    lines += ["## Exploratory (not pre-registered): what multi-scale adds beyond the smoother", "",
              "A′ vs B′, RMSE change. Labelled exploratory because this comparison was chosen after seeing seed 0.", "",
              "| variable | 1 d | 3 d | 5 d |", "|---|---|---|---|"]
    for var in ("Z500", "T850", "T2M", "Q850", "Q500", "Q250", "TP6h", "U250", "V500"):
        cells = []
        for lead in (24, 72, 120):
            c = paired_change(rmse, "multiscale_smooth", "baseline17_smooth", var, lead)
            cells.append(f"{c.mean():+.1f}%{'' if is_real(c) else ' (n.s.)'}")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")
    lines += ["", "(n.s.) = not a robust difference by the rule.", "",
              "## Exploratory: the smoother alone (B′ vs baseline)", "", "| variable | 6 h | 1 d | 3 d | 5 d |", "|---|---|---|---|---|"]
    for var in ("Z500", "T850", "T2M", "Q500", "V500"):
        cells = []
        for lead in (6, 24, 72, 120):
            c = paired_change(rmse, "baseline17_smooth", "baseline17", var, lead)
            cells.append(f"{c.mean():+.1f}%{'' if is_real(c) else ' (n.s.)'}")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")

    lines += ["", "## Summary", "", "| prediction | verdict |", "|---|---|"]
    lines += [f"| {k} | {'holds' if v else 'does not fully hold (see table)'} |" for k, v in verdicts.items()]
    (OUT / "seed_analysis_round2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and "--round2" in sys.argv:
    round2()
elif __name__ == "__main__":
    main()
