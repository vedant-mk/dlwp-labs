# Claims and evidence

What we hoped, what we got, whether it meets Jannik's brief, and what backs each claim.
Use this as the checklist while writing the paper: **every sentence in Results and Discussion should trace to a row here.**

**Last updated:** 2026-09-13, after round 2 (18 runs: 6 models × 3 seeds).

**How to read the evidence columns**
- **Ours**: a number from our own runs, with the file it comes from.
- **Literature**: a key in `explore/references.bib`. ⚠️ All references were written from memory: **verify every DOI and page before submission**; the least certain are marked VERIFY in the `.bib`.
- **Strength**: **Strong** = pre-registered and passes the rule (all 3 seeds agree in sign and |mean| ≥ 2 × seed std). **Moderate** = consistent and measured, but not a pre-registered test. **Exploratory** = chosen after seeing results; report, don't claim as a test. **Weak** = single run, or reasoning without a direct test.

---

## 1. Summary in one paragraph

We hoped a scale-aware ViT would forecast better or more realistically than the course baseline. **We got a clearer answer than expected.** The plain ViT's dominant scale defect is its **patch seams**: a 3×3 layer that lets neighbouring patches interact cuts Z500 error by ~25% at 1 day and extends useful skill (vs climatology) from 2.4 to 3.6 days. With seams fixed, a **multi-scale architecture** adds a large humidity gain (12–20% at 1–3 days) at a consistent surface-temperature cost. A **scale-aware loss** gives a tunable realism/accuracy trade-off. Everything the brief asks for scientifically is done; **the paper itself is not yet written.**

---

## 2. Hoped vs got

### 2.1 Pre-registered hypotheses

| # | We hoped | We got | Verdict | Strength |
|---|---|---|---|---|
| H1 | Baseline reproducible across seeds | Largest spread 0.59% of RMSE | ✅ holds | Strong |
| H2 | A improves humidity 5–8% at 1–3 d | −6.7% to −9.0%, 9/9 cells | ✅ holds | Strong |
| H3 | A worsens smooth fields at 5 d | Z500 +27.7%, T850 +36.1%, T2M +58.1% | ✅ holds | Strong |
| H4 | C cuts spurious small-scale power 15–30% | −11% to −28% (Z500, T850); Q850 −3% to −13% | ✅ holds | Strong |
| H5 | C costs no accuracy | Small real cost: Z500 +0.8% to +3.6% | ❌ **fails** | Strong (a robust negative) |
| P6 | A's damage is caused by its seams | A′ vs A at 5 d: Z500 −28.8%, T850 −32.0%, T2M −31.1%; spurious power −87% to −93% | ✅ holds | Strong |
| P7 | Humidity gain survives the smoother | A′ vs baseline: Q500 −22.0%, Q250 −18.8% at 1 d; 9/9 cells | ✅ holds | Strong |
| P8 | Smoother matters more for A than baseline (5 d) | −28.8% vs −9.0% (Z500); −32.0% vs −7.5% (T850) | ✅ holds | Strong (but see 2.2) |
| P9 | Milder pyramid weights cost less, still cut spurious power | C′ vs C Z500 −2.5% at 1 d; C′ spurious power −10% (1 d), −20% (5 d) | ✅ holds | Strong |

**Ours:** `explore/figures/seed_analysis.md`, `explore/figures/seed_analysis_round2.md`. Criteria fixed in advance in `explore/DATA_NOTES.md`.

### 2.2 Results we did not predict

| Result | Numbers | Strength | Caveat |
|---|---|---|---|
| **The smoother alone transforms the baseline** (B′) | Z500 −14% (6 h), −25% (1 d), −18% (3 d), −9% (5 d); ACC at 3 d 0.35 → 0.59 | Moderate: consistent across 3 seeds, not pre-registered | P8 compared only 5-day magnitudes; at 1–3 d the smoother's effect on the baseline is itself large |
| **Multi-scale adds value beyond the smoother** (A′ vs B′) | Q500 −19.6% / −12.0%, Q250 −17.7% / −13.1% at 1 / 3 d | **Exploratory**: comparison chosen after seeing seed 0 | Robust by the rule, but must be labelled exploratory |
| **Multi-scale costs surface temperature** (A′ vs B′) | T2M +7% to +9% at every lead | Exploratory | Mechanism untested; possibly the strong daily cycle |
| **Useful skill extended by over a day** | Z500 beats climatology until 2.4 d (baseline) → 3.6 d (B′), 3.7 d (A′) | Moderate: measured from seed means | First crossing of seed-mean curves |
| **Baseline shows patch seams** | Spectral spike at k = 16 (= 64 / 4); largest cell-to-cell jumps on patch boundaries | Moderate: two independent measurements agree | Seed-0 physical-space check only |
| Pyramid T2M error alternates with lead | Consistent across seeds | Weak (unexplained) | Likely diurnal; don't interpret |

### 2.3 Protocol decisions, checked on 2016 validation only

| Decision | First justified by | Re-checked on 2016 | Verdict |
|---|---|---|---|
| Predict the tendency (residual) | Dec 2015 (inside training years) ⚠️ | Loss vs persistence: plain 1.31×, residual 0.75× | ✅ Stands |
| Roll-out training | Test-year blow-up ⚠️ | Equal-length control: blow-up was **undertraining**; roll-out still −7% (3 d), −13% (5 d) Z500 | ✅ Stands, **corrected reason** |

**Ours:** `explore/figures/validation2016.md`.

---

## 3. Does it meet Jannik's brief?

| # | Brief asks for | Status | Where |
|---|---|---|---|
| 1 | Choose **one** inductive bias, explain the physics, why valuable | ✅ Met | §4.1 claims; `DATA_NOTES.md` Steps 2–3 |
| 2 | **At least two** ways to introduce it, show and explain code | ✅ Met (architecture: A/A′; loss: C/C′) | `utils/components.py`, `utils/loss_fn.py`, `configs/` |
| 3 | Predict where performance differs, and why | ✅ Met, written before runs | `DATA_NOTES.md` pre-registration sections |
| 4 | Compare both against the baseline ViT | ✅ Met, 3 seeds each | `seed_analysis*.md` |
| 5 | Plot: RMSE Z500 & T850 over 5 d vs persistence and WB2 climatology | ✅ Met | `explore/figures/F1_rmse_z500_t850.png` |
| 6 | Plot: T850 forecast at 1 day | ✅ Met (auto) | `runs/<model>/seed0/forecast_t850_24h.png` |
| 7 | Plot: training and validation loss | ✅ Met (auto, per run) | `runs/<model>/seed*/loss_curves.png` |
| 8 | Runtime and hardware of each run | ✅ Met | `runs/<model>/seed*/run.json` (MacBook Air, Apple M1 GPU via `mps`) |
| 9 | Two additional verification figures + justification | ✅ Met | E1 spectrum ratio, E2 heatmap; justification in `DATA_NOTES.md` |
| 10 | Summarise findings, recommend design choices | 🟡 Evidence ready; **text not written** | §5 below |
| 11 | Report: Intro/Methods/Results/Discussion, ≤2200 words, TCCML LaTeX | ❌ **Not started** | — |
| — | Train ≤ 2015; test exactly 2017–2019; include Z500 & T850; state resolution | ✅ Met | `experiment.py` enforces; 5.625° (the default) |
| — | Use `experiment.py` for reported runs | ✅ Met | all 18 runs |

**Open framing decision:** the brief says "two different ways". Proposed: **A′ (architecture) vs C′ (loss)** as the two, with baseline, A, B′ and C as ablations explaining *why*.

---

## 4. Claims ledger: what the paper may say

### 4.1 Introduction and motivation

| Claim | Ours | Literature | Strength |
|---|---|---|---|
| Predictability depends on scale: small scales lose skill fastest | Indirect: our error growth by variable | `lorenz1969predictability` | Strong (literature) |
| Atmospheric variance falls steeply with wavenumber | Z500 spectrum steeper than k⁻³ | `nastrom1985climatology` | Strong |
| Variables differ widely in small-scale content | Z500 3%, T850 7%, Q700 25%, TP6h 47% of variance at k ≥ 9 (2015) | — | Strong (ours) |
| The baseline ViT treats all scales and variables with one patch size | 4×4 patches, 128 tokens, all 17 variables in one token | `dosovitskiy2021vit` | Strong (from code) |
| Neural nets learn low frequencies first | — | `rahaman2019spectral` | Moderate: general ML, not weather-specific |
| MSE training blurs forecasts; RMSE rewards blur (double penalty) | Baseline Q850/TP6h spectral ratio < 1 | `subich2025doublepenalty` (VERIFY) | Moderate |
| Data-driven weather models: context | — | `lam2023graphcast`, `bi2023pangu`, `pathak2022fourcastnet`, `keisler2022gnn` | Strong |

### 4.2 Methods

| Claim | Ours | Literature | Strength |
|---|---|---|---|
| Data: ERA5 at 5.625°, WB2 protocol | 2009–2015 train, 2016 val, 2017–19 test | `hersbach2020era5`, `rasp2020weatherbench`, `rasp2024weatherbench2` | Strong |
| Latitude-weighted RMSE, ACC, persistence and climatology baselines | Computed by `experiment.py` | `rasp2024weatherbench2` | Strong |
| Residual (tendency) prediction | 1.31× → 0.75× persistence loss on 2016 | `he2016resnet`, `lam2023graphcast` | Strong |
| Multi-step roll-out training | −13% Z500 at 5 d vs equal-length single-step | `lam2023graphcast`, `keisler2022gnn`, `thuemmel2026dlwplabs` | Strong |
| A: parallel patch sizes with shared attention | 2×2 + 8×8, 544 tokens, dim 100, 0.985× params | `chen2021crossvit`, `fan2021mvit` | Strong (design) |
| C: Laplacian-pyramid loss | 3 bands, weights 4/2/1 and 2/1.5/1 | `burt1983laplacian`; motivation `subich2025doublepenalty` (VERIFY) | Strong (design) |
| Seam smoother: 3×3 conv, periodic in longitude, identity-initialised | +2,618 params; identical output at init (tested) | analogy `odena2016checkerboard` | Strong (design) |
| Parameter matching; validation on plain MSE; 3 seeds; pre-registered rule | Documented in configs and notes | — | Strong |

### 4.3 Results

| Claim | Ours | Literature | Strength |
|---|---|---|---|
| Baseline is reproducible (≤0.6% seed spread) | `seed_analysis.md` H1 | — | Strong |
| Baseline shows patch seams at k = 16 | E1 spike; boundary jumps | analogy `odena2016checkerboard` | Moderate |
| Baseline is blurred at synoptic scales and noisy at grid scales | E1 ratio 0.5–0.8 at k = 3–8; ≫1 at k ≥ 10 | — | Strong |
| A improves humidity but degrades smooth fields | H2, H3 | — | Strong |
| A's degradation is caused by its seams | P6 | — | Strong |
| The smoother greatly improves the baseline | B′: Z500 −25% at 1 d; ACC 0.35 → 0.59 at 3 d | — | Moderate (not pre-registered) |
| With seams fixed, multi-scale improves humidity beyond smoothing | A′ vs B′: −12% to −20% at 1–3 d | — | **Exploratory** |
| …at a consistent T2M cost | A′ vs B′: +7% to +9% | — | **Exploratory** |
| C reduces spurious small-scale power | H4 | — | Strong |
| C has a small accuracy cost at 4/2/1 | H5 fails: +0.8% to +3.6% | — | Strong |
| Milder weights remove the cost and keep most of the realism gain | P9 | — | Strong |
| Smoothed models extend skill vs climatology by over a day | 2.4 → 3.6–3.7 d (Z500) | — | Moderate |

### 4.4 Discussion (interpretation: must be worded as interpretation)

| Claim | Support | Strength |
|---|---|---|
| The ViT's biggest scale problem is how patches are written out, not a lack of multi-scale machinery | P6 + B′ results | Moderate |
| Multi-scale helps fields with the most small-scale structure, consistent with the bias | Humidity 25% small-scale variance + A′ humidity gain | Moderate (links two measurements) |
| A's thinner transformer blocks may explain its T2M cost | Parameter split 445k vs 657k | **Weak**: untested hypothesis |
| Recommendation: a seam smoother is a cheap default; multi-scale where moist fields matter; a mild scale-aware loss for realism | All of the above | Moderate |

---

## 5. Draft recommendation (brief item 10)

1. **Always fix the patch seams.** A 3×3 output layer periodic in longitude costs 0.35% more parameters and gives the largest improvement of anything tested.
2. **Use a multi-scale embedding when small-scale-rich fields (humidity) matter**, but only together with the smoother, and check surface temperature.
3. **If realism matters (e.g. downstream use of spectra or extremes), add a scale-aware loss with mild weights** (2/1.5/1): about 20% less spurious power at no measurable accuracy cost.
4. **Do not judge scale-aware changes by RMSE alone**: spectra showed effects RMSE hid.

---

## 6. Weaknesses to state honestly in the paper

| Weakness | Why it matters | How to state it |
|---|---|---|
| All models lose to climatology after 2.4–3.7 days (Z500) | Absolute skill is modest | Small model (~740k params), 5.625°, 7 years, 6000 steps |
| Some early decisions were motivated by test data | Methodological hygiene | Both re-checked on 2016 only; report the corrected roll-out rationale |
| A′ vs B′ comparison chosen after seeing seed 0 | Risk of fishing | Label exploratory; the rule was still applied |
| One smoother design and two pyramid weight settings | Limited exploration | Future work: overlapping un-embedding, weight sweep |
| Deterministic model only; no ensembles | Blurring partly unavoidable | Point to `price2025gencast` (VERIFY) as future direction |
| T2M cost of multi-scale unexplained | Incomplete mechanism | State as observation, hypothesis only |
| References written from memory | Citation accuracy | **Verify all before submission** |

---

## 7. Claims with no literature support in our bibliography

These rest on our own evidence only. That's acceptable for results, but don't overstate novelty:

- The specific finding that **ViT patch un-embedding causes spectral seams in weather forecasts** (analogy only: `odena2016checkerboard`). Worth a literature search before claiming it is new.
- That a **3×3 seam smoother** extends skill this much in a weather ViT.
- The **humidity-specific** benefit of multi-scale embedding.

---

## Changelog

| Date | Change |
|---|---|
| 2026-09-13 | First version after round 2; added `odena2016checkerboard` to the bibliography |
