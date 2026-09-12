# Data notes — DLWP final project

Purpose: understand the data well enough to *defend* the choice of one inductive bias.
Project: compare two ways of adding ONE physics-based inductive bias to the course ViT.
Deadline: 2026-09-21. Report ≤2200 words, TCCML NeurIPS 2026 style.

Rules in force:
- Training data: ERA5 / WeatherBench 2, years **up to and including 2015**.
- Validation: 2016. Test: **exactly 2017, 2018, 2019** — never in training statistics.
- Must include **Z500 and T850** among variables.
- Resolution 5.625° (state explicitly if changed).
- Studies must be run through `experiment.py`.

---

## Step 0 — What is on disk

**Stores**
- `data/` — does not exist yet (gitignored). The training config expects `data/era5_5p6.zarr`,
  eval expects `data/era5_eval_5p6.zarr` and `data/clim_wb2_5p6.zarr`. None built yet.
- `cache/data.nc` — 539 MB NetCDF, the Lab 1 subset. The only weather data currently local.

**`cache/data.nc`, subset to 2015**
- Variables (9): `T2M, U10M, V10M, TP6h, Z500, T850, Q700, U250, V250`
- Grid: **32 lat × 64 lon**, spacing 5.625° in both directions
- Latitude centres: −87.188° … +87.188° (cell *edges* reach the poles, centres do not)
- Longitude: 0° … 354.375°, wraps around
- Time: 2015-01-01 00:00 → 2015-12-31 18:00, **1460 steps** at 6-hourly cadence
- In-memory footprint of the 2015 subset: ~108 MB float32

**Baseline variable list (`configs/baseline.yaml`) vs what we have**
Baseline asks for 16: 5 quantities (T, Z, Q, U, V) × 3 levels (850/500/250 hPa) + `T2M`.
- Present in cache (5): `T2M, Z500, T850, U250, V250`
- **Missing (11)**: `Z850, Z250, T500, T250, Q850, Q500, Q250, U850, U500, V850, V500`
- In cache but not in baseline (4): `U10M, V10M, TP6h, Q700`

Consequence: a data download job is pending before training. Instructor recommends 5+ years
of training data; we currently have 1 year.

---

## Step 1 — What ERA5 / WeatherBench 2 are

1. **ERA5 is a reanalysis** — not raw observations, not a free-running simulation, but the best
   reconstruction of past weather, produced by assimilating millions of real observations
   (satellite, balloon, ship, station) into a physics model that fills every gap.
2. Coverage: **1940–present, hourly, global**, native grid ~31 km, dozens of variables at dozens
   of levels; ~5 PB in total.
3. It is the de facto **ground truth for AI weather models** because it is *complete and physically
   consistent* — real observations have holes, and a network cannot train on holes.
4. **WeatherBench 2** is a benchmark wrapper, not new data: it re-grids ERA5 to workable
   resolutions (ours: 5.625°, the 32×64 grid), fixes train/val/test splits, and standardises
   metrics so papers are comparable.
5. Short version: **ERA5 = the data; WeatherBench 2 = the packaging, the rules, the scoreboard.**

### Evidence this step contributes to the bias decision

| Fact | Bias it supports |
|---|---|
| ERA5 is an *estimate*, not truth; the atmosphere is chaotic on top of that. | Uncertainty quantification |
| Data lives on a **sphere**, re-gridded to a rectangle for convenience — a distortion the model inherits. | Spherical geometry |
| Re-gridding 31 km → ~625 km **averages away all small-scale weather**, leaving large-scale structure only. | Hierarchical scale interactions |

---

## Step 2 — Space: the grid is a flat picture of a round Earth

Script `explore/step2_space.py`, figure `explore/figures/step2_space.png`.

```
grid                : 32 lat x 64 lon, spacing 5.625° x 5.625°
north-south extent  : 625 km      (identical at every latitude)
east-west  equator  : 625 km      (lat -2.81°)
east-west  polemost :  31 km      (lat -87.19°)
cell area  equator  : 390,586 km²
cell area  polemost :  19,188 km²
AREA RATIO eq/pole  : 20.4 x      (= cos(2.81°)/cos(87.19°), exactly)
total area check    : 510,064,472 km² = 4πR² exactly ✓
ViT tokens          : 8 x 16, each 22.5° x 22.5° (patch_size [4,4])
token area ratio    : 5.0 x       (equator token vs polemost token)
```

**Key consequences**
- Cell area follows `cos(latitude)` exactly. Plain MSE over all 2048 cells weights each cell
  equally, so **each km² near the pole carries ~20× the loss weight of a km² at the equator**.
  This is why latitude-weighted RMSE is the standard verification metric (and why the brief
  requires it).
- **Longitude wraps**: column 63 and column 0 are physical neighbours. Panel (d) tiles a Z500
  field twice — the seam at 360°/0° is invisible. The ViT's sin-cos position encoding, built
  from integer `(row, col)` indices, encodes them as maximally distant.
- Panels (a) vs (b) are the same grid: (a) on a sphere, cells converging at the pole;
  (b) flat, every cell identical. **(b) is what reaches the model.**

---

## Step 3 — Variables, and the scales they occupy

Script `explore/step3_variables_scales.py`, figure `explore/figures/step3_variables_scales.png`.

| Var | Level | Altitude | Unit | Mean | Std | Meaning |
|---|---|---|---|---|---|---|
| `Z500` | 500 hPa | ~5.5 km | m²/s² | 55,580 | 3,468 | geopotential — the steering flow; troughs and ridges |
| `T850` | 850 hPa | ~1.5 km | K | 281.4 | 15.5 | temperature above the boundary layer; air-mass marker |
| `Q700` | 700 hPa | ~3.0 km | kg/kg | 0.0033 | 0.0025 | specific humidity — moisture available to storms |
| `TP6h` | surface | 0 km | m/6h | 0.00074 | 0.0013 | precipitation accumulated over 6 h |
| `T2M` | surface | 2 m | K | 287.8 | 21.0 | screen-level temperature; strong daily cycle |
| `U10M`/`V10M` | surface | 10 m | m/s | −0.30 / 0.19 | 5.1 / 4.4 | surface wind components |
| `U250`/`V250` | 250 hPa | ~10.4 km | m/s | 14.7 / −0.04 | 17.2 / 12.7 | jet-stream-level wind |

**Zonal power spectra** (FFT along longitude — exactly periodic, no windowing needed), zonal mean
removed, area-weighted over latitude, averaged over 98 times spanning 2015. Fraction of variance
per band:

| Variable | large (k≤3) | synoptic (k 4–8) | small (k≥9) | dominant |
|---|---|---|---|---|
| **Z500** | 65% | 32% | **3%** | large |
| T2M | 75% | 18% | 8% | large |
| T850 | 60% | 32% | 7% | large |
| U250 | 62% | 32% | 7% | large |
| U10M | 55% | 34% | 12% | large |
| Q700 | 39% | 36% | 25% | large |
| V10M | 25% | 50% | 26% | synoptic |
| **V250** | 19% | **60%** | 22% | synoptic |
| **TP6h** | 27% | 27% | **47%** | **small** |

**Headline number: Z500 keeps 3% of its variance at small scales; TP6h keeps 47% — a ~15×
spread across variables that pass through the same 4×4 patch embedding.**

In the log-log panel, Z500 falls off *steeper than* the `k⁻³` reference at high wavenumber while
TP6h stays nearly flat. They are not the same kind of field.

### Step 3 reflection (answered by Claude at the user's request)

**Smooth:** Z500, T850, U250, T2M — broad ribbons, a few planetary waves round the globe.
**Patchy:** TP6h (speckle), Q700 (filaments); V250 and V10M are eddy-dominated — the
north–south wind *is* the storm-track signal, so it is blobbier than the west–east wind.

**Implication at 5 days.** Error growth rate scales inversely with feature size: a small feature
has a short turnover time so perturbations double in hours, while a planetary wave takes days
(Lorenz). **Predictability is therefore not one number for "the weather" — it is a different
number for every scale.** At 5 days Z500 remains genuinely forecastable (hence its status as
*the* benchmark variable, and its mandatory inclusion in the brief); grid-point precipitation at
5 days is essentially unforecastable, and only a regional probability is honest.

**Why this is the case for hierarchical scales.** A model trained on MSE learns that the optimal
guess for an unpredictable feature is its conditional *mean*, so it blurs — the well-known
smoothing failure of AI weather models at long lead. The baseline applies one `patch_size [4,4]`
and one flat stack of four global-attention blocks identically to a field with 3% small-scale
variance and one with 47%.

---

## Baseline ViT — what it is and what it lacks

Read from `references/components.py` (the course's reference solution; `utils/` is still empty stubs).

Pipeline: patchify (4×4 patches → 8×16 token grid, dim 128) → add fixed sin-cos positions →
4 transformer blocks (attention + gated FFN, RMSNorm, DropPath) → un-patchify back to fields.

| Candidate bias | Lacking? | Code evidence |
|---|---|---|
| **Spherical geometry** | Yes | `init_sincos_positions` builds positions from integer `(row, col)` via `unravel_index` on a flat rectangle. No `cos(lat)` weighting anywhere. Longitude index 0 and 63 encoded as maximally distant despite being neighbours on the sphere. |
| **Uncertainty quantification** | Yes | `ViT.forward` returns a single deterministic field. No variance head, no ensemble member dimension, no distributional output. Objective is plain `mse`. |
| **Heterogeneous variables** | Yes | `EinMix(weight_shape="v p1 p2 d")` blends all 16 variables into a shared token in the first operation. No per-variable embedding, normalisation, or type awareness downstream. |
| **Hierarchical scale interactions** | Yes | Single fixed `patch_size: [4,4]`, one resolution, flat stack of 4 identical blocks. No pyramid, no multi-scale path, no coarse/fine separation. |

All four are genuinely absent — any is a legitimate project.

### Where each could enter the pipeline
Per the project brief, biases can enter via *loss functions, model architectures, dataset
configurations, or forward step methods*.

---

## CRITICAL: what the course labs already cover

Read from the markdown of `03_inductive_biases.ipynb` and `04_ensembles.ipynb`.

| Candidate bias | Covered? | Where |
|---|---|---|
| **Uncertainty quantification** | **Fully — avoid** | All of Lab 4: fair CRPS (`EmpiricalCRPS`), ensemble `forward_step` with `ens_size`, and **three** noise injections (`add`, `concat`, `adaln`) selected by `NetworkConfig.noise_injection`. Verification: rank histogram, spread-skill ratio, information/noise decomposition, ACC. Lab 4's Extension explicitly trains all three injections and compares them — i.e. *one bias, three implementations, vs a deterministic baseline*. That is the final-project structure, pre-built. |
| **Heterogeneous variables** | **Fully — avoid** | Lab 3 Part 3 gives **two** implementations: 3.1 `separable_embed` (per-variable patch embedding, `configs/vit_separable.yaml`) and 3.2 per-variable loss weights (`configs/vit_separable_weighted.yaml`, GraphCast-style weights). Exactly "two different ways to introduce this inductive bias". |
| **Spherical geometry** | **Half — loss only** | Lab 3 Part 1: `WeightedMSE(latitude, variable_weights, latitude_weighting)` with cos(lat) normalised to mean one; `configs/vit_area_mse.yaml`. The lab's own check quotes the ratio cos(2.8°)/cos(87.2°) ≈ 20 — the same number computed in Step 2. **Nothing architectural**: no spherical positions, no equal-area patching, no distance-aware attention, no longitude wrap handling. |
| **Hierarchical scale interactions** | **Not covered at all** | Lab 2 = base pipeline; Lab 3 = loss weights, roll-out fine-tuning, variable separation, time/season embedding; Lab 4 = ensembles. No multi-scale, no pyramid, no windowed/local attention, no spectral decomposition anywhere. |

**Consequence.** Uncertainty quantification and heterogeneous variables would mean re-running
lab exercises with the course's own config files. Spherical geometry is viable only if *both*
implementations are architectural, since the loss-weighting route is Lab 3 Part 1.
Hierarchical scale interactions is the only fully open option.

### Other machinery the labs provide (reusable, not the contribution)
- Roll-out fine-tuning in the training step (`train_rollout_steps`, `pre_steps`) — Lab 3 Part 2.
- Time-of-day / season sin-cos metadata embedding (`metadata_embed`) — Lab 3 Extension.
- Full verification suite in `utils/metrics.py`: `latitude_weights`, `truth_at`,
  `climatology_at`, `rmse_per_initialisation`, `rmse`, `skill_score`, `acc`, `activity`,
  `ensemble_mean`, rank histogram, spread-skill ratio.
- `LossCurve` callback + `plot_metrics` for the train/val loss figure the brief requires.

### Physical case for hierarchical scales (if chosen)
The atmosphere has a scale cascade: large structures (Rossby waves, ~5000 km) evolve slowly and
stay predictable for a week or more; small structures (fronts, convection) evolve fast and lose
predictability within hours. **Different scales have different predictability horizons.** The
baseline ViT applies one fixed `patch_size: [4,4]`, one resolution, and a flat stack of four
identical global-attention blocks — every scale treated identically.

Candidate implementations (need two):
- (a) Multi-scale patching: parallel patch sizes fused, or a U-Net-style ViT with down/upsampling.
- (b) Hierarchical attention: local windowed attention early, global attention late (Swin-style).
- (c) Spectral separation: low-pass/high-pass split, process separately, recombine.
- (d) Multi-scale loss: Laplacian-pyramid loss penalising each scale band.

Evidence still to gather: power spectrum of Z500 (which scales carry variance), and
scale-dependent error growth (Step 7) showing small scales decorrelate first.

---

## Pipeline: adapting references/ into utils/ (done 2026-09-11)

`references/` turned out to be a **different design generation** from what `experiment.py` and
`configs/baseline.yaml` expect (Lab 2's spec). The README calls them "reference implementations
to read" — they are not a drop-in. Mismatches found and fixed:

| Mismatch | Fix applied |
|---|---|
| `WeatherData` vs `WeatherDataset` | class renamed; `WeatherData` kept as an alias |
| `NetworkConfig` had no `name` / `patch_size` | both added (`name: str = "vit"`, `patch_size: tuple = (4, 4)`) |
| `ObjectiveConfig.loss` vs yaml's `objective.name` | field renamed to `name`; module reads `f_{name}` |
| `Config` **required** a `world:` block; `baseline.yaml` has none | `world` now `Optional`, defaulting to `None`; `build_world(num_variables, field_shape, patch_size, separable=False)` added to `utils/config.py`; `ForecastModule.__init__` derives it from the training grid when absent |
| No `Persistence` network (needed by `configs/persistence.yaml`) | `Persistence` + `build_network(network, world)` added to `utils/components.py` |
| `forecasts_to_xarray` missing | lifted from the given code cell in Lab 3 §0.2 |
| `plot_metrics`, `LossCurve` missing | written to the Lab 3 §1.2 spec; `plot_metrics` draws on the current axes because `experiment.plot_loss` calls `plt.legend()`/`savefig` after it |
| `scores` missing everywhere | written: returns `rmse`, `rmse_climatology`, `skill` on a `score` dim (`experiment.py` supplies `rmse_persistence` and `acc` itself) |
| `experiment.py` passes the raw `(hour, dayofyear)` climatology to `acc` | `aligned_climatology()` helper added; `acc` and `scores` call it, so either form works |

**Verified**: all imports resolve, `baseline.yaml` parses, derived world = **8×16 = 128 tokens,
patch (16, 4, 4)** (all 16 variables in one token — confirming the baseline mixes variables
immediately and uses a single spatial scale), ViT maps `(2,16,32,64) → (2,16,32,64)` with
**738,688 parameters**, and `Persistence` round-trips exactly.

**Environment note**: this machine throws `OMP: Error #15` (duplicate `libomp.dylib`) on import.
Workaround in use: `KMP_DUPLICATE_LIB_OK=TRUE`. Worth fixing properly before long runs.

### Still missing before a training run
1. `data/era5_5p6.zarr` — the training store. Only `cache/data.nc` exists; Lab 2's setup cell writes the zarr.
2. `data/stats_train.zarr`, `data/era5_eval_5p6.zarr`, `data/clim_wb2_5p6.zarr` — written/downloaded by `prepare_eval_data.py`.
3. Variable list: `baseline.yaml` asks for 16, we hold 9. **`baseline.yaml` already carries the 9-variable list commented out**, and the README notes the nine "are already in your store, so only their climatology is fetched" — using them avoids the ~3 GB ERA5 download.

---

## Compute

Google Cloud "Billing Account for Education" is active (coupon redeemed, $0.00 spent as of
2026-09-11). GPU VM to be provisioned when training starts.

---

## Extra verification figures (script: `explore/extra_figures.py`)

The brief asks for two additional figures and a justification of why they give views of
performance complementary to the required RMSE curves.

**E1 — forecast zonal power spectrum ÷ ERA5 spectrum** (`E1_spectrum_ratio.png`), for Z500, T850,
Q850, TP6h at 1 and 5 days. *Why complementary:* RMSE is minimised by the conditional mean, so it
rewards smooth forecasts and cannot distinguish a sharp-but-displaced feature from a missing one
(the double penalty). The spectrum ratio is independent of where features sit: it asks only whether
the forecast carries realistic variance at each scale. 1 = realistic, <1 = blurred, >1 = spurious.

**E2 — RMSE change vs baseline, every variable × lead** (`E2_rmse_change_heatmap.png`).
*Why complementary:* the required figure covers only Z500 and T850, both large-scale fields
(3% and 7% small-scale variance). A hierarchical-scale bias should matter most for fields with
small-scale structure (Q, TP6h at 25–47%), so this shows where each variant helps and hurts.

### Finding from E1 on the rollout-trained baseline (2026-09-12)

The baseline is wrong at *both* ends of the spectrum:
- **Blurred at synoptic scales:** power ratio 0.5–0.8 at k = 3–8 (Z500, 5 days). Storm-sized
  features lose 20–50% of their variance.
- **Spurious grid-scale noise:** ratio rises steeply for k ≥ 10, with a **spike exactly at k = 16**
  (Z500 at 5 d: 11.2, 17.1, **50.8**, 31.7, 36.0 for k = 14..18) and at k = 32. 64 longitude
  cells ÷ 4-cell patches = 16 — the patch grid. Confirmed in physical space: mean |jump| between
  neighbouring columns is largest across patch boundaries (606 vs 480–545 inside a patch).
  The spike is absent at 1 day and grows over the rollout, i.e. seams accumulate step by step.
- TP6h is blurred at every scale (ratio 0.1–0.7): MSE regresses the unpredictable field to its mean.

**Sharpened prediction for the variants:** A (2×2 + 8×8 patches) should remove or relocate the
k = 16 spike; C (pyramid loss, weight 4 on 2×2-scale detail) should suppress the grid-scale excess.

---

## Pre-registered criteria for the seed analysis (written 2026-09-12, before seeds 1-2 finished)

**Rule for "real":** a difference counts as real only if all 3 seeds agree in sign AND the mean
difference is at least 2x the seed-to-seed standard deviation. Otherwise it is reported as within
seed noise. This is fixed now so the analysis cannot be tuned to the results.

**Hypotheses, from the seed-0 results:**
1. Baseline is reproducible: Z500/T850 RMSE spread across seeds within ~1-2%.
2. A improves humidity (Q850/Q500/Q250) RMSE by ~5-8% at 1-3 days.
3. A degrades smooth large-scale fields (Z500, T850, T2M) at 5 days.
4. C reduces spurious small-scale power (E1 ratio at k >= 9) by ~15-30% relative to baseline.
5. C's Z500/T850 RMSE is within seed noise of the baseline.

**Expectation before seeing seeds:** 1-3 likely (large effects), 4 probable, 5 uncertain.
Outcomes that fail these are reported as such, not reframed.
