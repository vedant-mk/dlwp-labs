# Seed analysis against the pre-registered hypotheses

Rule: real only if all seeds agree in sign and |mean| >= 2 x seed std. Values: mean ± std (per-seed values).

## H1: baseline is reproducible (seed spread within ~1–2%)

| variable | lead (h) | mean RMSE | spread (% of mean) |
|---|---|---|---|
| Z500 | 6 | 184.05 | 0.16% |
| Z500 | 24 | 460.21 | 0.25% |
| Z500 | 72 | 855.68 | 0.31% |
| Z500 | 120 | 1015.64 | 0.59% |
| T850 | 6 | 1.10 | 0.14% |
| T850 | 24 | 2.31 | 0.22% |
| T850 | 72 | 3.85 | 0.17% |
| T850 | 120 | 4.47 | 0.11% |

Largest spread 0.59% → **HOLDS**

## H2: A improves humidity RMSE by ~5–8% at 1–3 days

| variable | lead (h) | multiscale vs baseline | real & improving? |
|---|---|---|---|
| Q850 | 24 | -8.3% ± 0.6 (-7.8, -8.9, -8.1) | yes |
| Q850 | 48 | -9.0% ± 0.5 (-8.8, -9.6, -8.7) | yes |
| Q850 | 72 | -6.7% ± 0.6 (-6.6, -7.3, -6.1) | yes |
| Q500 | 24 | -8.0% ± 0.7 (-7.4, -8.7, -8.0) | yes |
| Q500 | 48 | -8.5% ± 0.5 (-8.0, -9.1, -8.3) | yes |
| Q500 | 72 | -7.6% ± 0.5 (-7.5, -8.2, -7.2) | yes |
| Q250 | 24 | -7.1% ± 0.4 (-6.6, -7.4, -7.3) | yes |
| Q250 | 48 | -8.6% ± 0.4 (-8.2, -8.9, -8.8) | yes |
| Q250 | 72 | -8.4% ± 0.3 (-8.0, -8.6, -8.6) | yes |

9/9 cells real improvements → **HOLDS**

## H3: A degrades smooth large-scale fields at 5 days

| variable | multiscale vs baseline at 120 h | real & worse? |
|---|---|---|
| Z500 | +27.7% ± 5.4 (+27.8, +22.3, +33.1) | yes |
| T850 | +36.1% ± 6.5 (+42.3, +29.3, +36.8) | yes |
| T2M | +58.1% ± 5.5 (+59.8, +52.0, +62.6) | yes |

**HOLDS**

## H4: C reduces spurious small-scale power (mean ratio at k ≥ 9) by ~15–30%

Per-seed pairing: seed k of pyramid against seed k of the baseline.

| variable | lead (h) | baseline ratio | pyramid ratio | change | real reduction? |
|---|---|---|---|---|---|
| Z500 | 24 | 21.74 | 18.16 | -16.5% ± 2.2 (-14.9, -18.9, -15.6) | yes |
| Z500 | 120 | 143.86 | 103.80 | -27.9% ± 2.9 (-30.4, -28.5, -24.8) | yes |
| T850 | 24 | 2.43 | 2.16 | -11.0% ± 3.0 (-8.6, -14.3, -9.9) | yes |
| T850 | 120 | 10.51 | 8.31 | -20.9% ± 4.9 (-15.2, -23.3, -24.2) | yes |
| Q850 | 24 | 1.02 | 0.99 | -2.8% ± 0.3 (-2.6, -2.7, -3.2) | yes |
| Q850 | 120 | 1.57 | 1.37 | -12.8% ± 2.8 (-10.0, -12.9, -15.6) | yes |

TP6h is excluded: it is blurred (ratio < 1), not noisy, so 'spurious power' does not apply.

6/6 cells real reductions → **HOLDS**

## H5: C's Z500/T850 RMSE is within seed noise of the baseline

| variable | lead (h) | pyramid vs baseline | within noise? |
|---|---|---|---|
| Z500 | 6 | +3.0% ± 0.5 (+2.5, +3.4, +3.1) | no, a real difference |
| Z500 | 24 | +3.6% ± 0.3 (+3.2, +3.8, +3.7) | no, a real difference |
| Z500 | 72 | +2.0% ± 0.2 (+1.7, +2.0, +2.1) | no, a real difference |
| Z500 | 120 | +0.8% ± 0.2 (+0.5, +0.8, +1.0) | no, a real difference |
| T850 | 6 | +1.6% ± 0.5 (+1.5, +1.2, +2.1) | no, a real difference |
| T850 | 24 | +2.4% ± 0.5 (+2.5, +1.9, +2.7) | no, a real difference |
| T850 | 72 | +1.2% ± 0.4 (+1.3, +0.8, +1.5) | no, a real difference |
| T850 | 120 | -0.1% ± 0.2 (-0.1, -0.3, +0.1) | yes |

1/8 cells within noise → **FAILS**

## Summary

| hypothesis | verdict |
|---|---|
| H1 | holds |
| H2 | holds |
| H3 | holds |
| H4 | holds |
| H5 | fails (see table) |
