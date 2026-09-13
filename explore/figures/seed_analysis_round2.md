# Round 2: pre-registered predictions P6–P9

Rule: real only if all seeds agree in sign and |mean| >= 2 x seed std. Seeds paired by index.

## P6: A′ fixes A's 5-day damage (seams cause it)

| quantity | A′ vs A | real improvement? |
|---|---|---|
| RMSE Z500, 5 d | -28.8% ± 2.7 (-29.2, -25.9, -31.3) | yes |
| RMSE T850, 5 d | -32.0% ± 4.3 (-36.4, -27.9, -31.9) | yes |
| RMSE T2M, 5 d | -31.1% ± 3.2 (-30.9, -28.0, -34.4) | yes |
| spurious power k≥9 Z500, 5 d | -87.3% ± 1.3 (-88.6, -86.0, -87.4) | yes |
| spurious power k≥9 T850, 5 d | -92.7% ± 1.6 (-94.0, -90.9, -93.1) | yes |
| spurious power k≥9 Q850, 5 d | -25.6% ± 3.1 (-25.2, -22.6, -28.8) | yes |

6/6 → **HOLDS**

## P7: A′ keeps a humidity improvement over the baseline at 1–3 days

| variable | lead (h) | A′ vs baseline | real improvement? |
|---|---|---|---|
| Q850 | 24 | -8.2% ± 0.3 (-8.4, -8.4, -7.9) | yes |
| Q850 | 48 | -8.4% ± 0.4 (-8.5, -8.6, -7.9) | yes |
| Q850 | 72 | -6.0% ± 0.3 (-6.0, -6.3, -5.6) | yes |
| Q500 | 24 | -22.0% ± 0.2 (-21.9, -22.3, -21.9) | yes |
| Q500 | 48 | -18.9% ± 0.4 (-18.5, -19.3, -19.0) | yes |
| Q500 | 72 | -13.2% ± 0.5 (-12.6, -13.3, -13.5) | yes |
| Q250 | 24 | -18.8% ± 0.3 (-18.5, -19.0, -18.9) | yes |
| Q250 | 48 | -17.0% ± 0.3 (-16.6, -17.2, -17.1) | yes |
| Q250 | 72 | -12.7% ± 0.3 (-12.3, -12.7, -13.0) | yes |

9/9 → **HOLDS**

## P8: the smoother matters more for A than for the baseline (5 days)

| variable | B′ vs baseline | A′ vs A | smaller for the baseline? |
|---|---|---|---|
| Z500 | -9.0% ± 1.2 (-9.6, -9.8, -7.6) | -28.8% ± 2.7 (-29.2, -25.9, -31.3) | yes |
| T850 | -7.5% ± 0.6 (-8.0, -6.8, -7.8) | -32.0% ± 4.3 (-36.4, -27.9, -31.9) | yes |

Note: P8 compares magnitudes at 5 days only, as pre-registered. At 1–3 days the smoother's
effect on the baseline is itself large (see the exploratory section).

**HOLDS**

## P9: milder pyramid weights (C′) cost less, but still reduce spurious power

| quantity | comparison | value | as predicted? |
|---|---|---|---|
| RMSE Z500, 1 d | C′ vs C | -2.5% ± 0.1 (-2.5, -2.6, -2.3) | yes |
| spurious power Z500, 24 h | C′ vs baseline (C: -16.5%) | -10.0% ± 1.3 (-8.7, -10.0, -11.3) | yes |
| spurious power Z500, 120 h | C′ vs baseline (C: -27.9%) | -19.8% ± 5.7 (-17.0, -16.2, -26.4) | yes |
| spurious power T850, 24 h | C′ vs baseline (C: -11.0%) | -7.4% ± 2.7 (-5.1, -10.3, -6.7) | yes |
| spurious power T850, 120 h | C′ vs baseline (C: -20.9%) | -17.7% ± 5.8 (-11.7, -23.2, -18.1) | yes |

5/5 → **HOLDS**

## Exploratory (not pre-registered): what multi-scale adds beyond the smoother

A′ vs B′, RMSE change. Labelled exploratory because this comparison was chosen after seeing seed 0.

| variable | 1 d | 3 d | 5 d |
|---|---|---|---|
| Z500 | -4.7% | -2.3% | -0.2% (n.s.) |
| T850 | -5.6% | -2.8% | -0.1% (n.s.) |
| T2M | +8.1% | +7.3% | +8.7% |
| Q850 | -6.1% | -3.6% | +0.7% (n.s.) |
| Q500 | -19.6% | -12.0% | -3.9% |
| Q250 | -17.7% | -13.1% | -7.5% |
| TP6h | -6.9% | +1.4% | +6.2% |
| U250 | -7.5% | -1.4% (n.s.) | +2.8% (n.s.) |
| V500 | -7.8% | -1.5% (n.s.) | +0.5% (n.s.) |

(n.s.) = not a robust difference by the rule.

## Exploratory: the smoother alone (B′ vs baseline)

| variable | 6 h | 1 d | 3 d | 5 d |
|---|---|---|---|---|
| Z500 | -14.4% | -25.3% | -18.2% | -9.0% |
| T850 | -10.1% | -12.4% | -13.5% | -7.5% |
| T2M | -22.3% | -10.2% | -4.7% | +0.1% (n.s.) |
| Q500 | -2.0% | -3.0% | -1.3% | +2.1% |
| V500 | -9.7% | -25.8% | -15.3% | +0.0% (n.s.) |

## Summary

| prediction | verdict |
|---|---|
| P6 | holds |
| P7 | holds |
| P8 | holds |
| P9 | holds |
