# Learning guide: Deep Learning for Weather Prediction final project

**Purpose:** everything you need to understand, and be able to *explain*, what this project does and why.
Read it top to bottom once, then use it as a reference. Each topic has:

- **In one sentence**: the idea, plainly
- **In your project**: where it shows up, with our real numbers
- **Check yourself**: questions you should be able to answer out loud

> This is a living document. It is updated after each milestone. See the [Changelog](#changelog) at the bottom.

**Last updated:** 2026-09-13 (round 2 in: seam smoother, A′, B′, C′; P6–P9 all hold)

---

## Contents

1. [The project in five sentences](#1-the-project-in-five-sentences)
2. [What Jannik's brief asks for](#2-what-jannik-s-brief-asks-for)
3. [Tools and workflow](#3-tools-and-workflow)
4. [The data: weather basics](#4-the-data-weather-basics)
5. [The data: how it is stored and split](#5-the-data-how-it-is-stored-and-split)
6. [Scales, spectra and predictability: the physics](#6-scales-spectra-and-predictability-the-physics)
7. [Deep learning fundamentals you need](#7-deep-learning-fundamentals-you-need)
8. [Transformers and the Vision Transformer (ViT)](#8-transformers-and-the-vision-transformer-vit)
9. [Turning a network into a forecaster](#9-turning-a-network-into-a-forecaster)
10. [Inductive biases, and why we chose hierarchical scales](#10-inductive-biases-and-why-we-chose-hierarchical-scales)
11. [Our two variants: A (architecture) and C (loss)](#11-our-two-variants-a-architecture-and-c-loss)
12. [How forecasts are judged: verification](#12-how-forecasts-are-judged-verification)
13. [Doing the science honestly: experimental design](#13-doing-the-science-honestly-experimental-design)
14. [Timeline: what we did, in order](#14-timeline-what-we-did-in-order)
15. [Bugs and lessons](#15-bugs-and-lessons)
16. [Results so far](#16-results-so-far)
17. [Open questions and next steps](#17-open-questions-and-next-steps)
18. [How this compares to your ML4TimeSeries project](#18-how-this-compares-to-your-ml4timeseries-project)
19. [Glossary](#19-glossary)
20. [Self-test: explain it out loud](#20-self-test-explain-it-out-loud)
21. [Where things live in the repo](#21-where-things-live-in-the-repo)
- [Changelog](#changelog)

---

## 1. The project in five sentences

1. We train a small neural network, a **Vision Transformer**, to forecast global weather: given the atmosphere now, predict it 6 hours later, then repeat that 20 times to reach 5 days.
2. The standard network treats every size of weather feature the same way, but in reality **large and small features behave very differently** (large ones stay predictable for days, small ones for hours).
3. That idea, that the model should respect differences between scales, is our **inductive bias: hierarchical scale interactions**.
4. We add it in **two different ways**: **A** changes the *architecture* (the model looks at the map at two patch sizes at once), and **C** changes the *loss* (the model is scored separately on each scale).
5. We compare both against the plain baseline using accuracy scores (RMSE, ACC) **and** a spectrum analysis that shows whether forecasts are blurry or noisy, because accuracy scores alone reward blurring.

**Check yourself**
- Can you say those five sentences without reading them?
- What is the difference between *where* A and C add the bias?

---

## 2. What Jannik's brief asks for

| Requirement | Our answer | Status |
|---|---|---|
| Choose **one** inductive bias, explain the physics | Hierarchical scale interactions (Lorenz: predictability depends on scale) | Done |
| **Two** ways to introduce it, show and explain the code | A: multi-scale patches · C: pyramid loss | Done |
| Predict where performance will differ | Written *before* running (see §13) | Done |
| Compare both against the baseline ViT | 3 seeds per model; hypotheses judged (§16.0) | Done |
| Figure: RMSE Z500 & T850 over 5 days vs persistence and climatology | `explore/figures/F1_rmse_z500_t850.png`: seed means with ranges | Done |
| Figure: T850 forecast at 1 day | `runs/*/seed0/forecast_t850_24h.png` (auto) | Done |
| Figure: training and validation loss | `runs/*/seed0/loss_curves.png` (auto) | Done |
| Runtime and hardware of each run | `runs/*/seed*/run.json` (Mac GPU, `mps`) | Done |
| **Two extra** verification figures, with justification | E1 spectrum ratio · E2 per-variable heatmap, both over 3 seeds | Done |
| Summary + recommendation | Evidence in §16.0; wording to write | To do |
| Report: ≤2200 words, Intro/Methods/Results/Discussion, TCCML LaTeX | References ready | To do |
| **Data rules:** train ≤ 2015, test exactly 2017–2019, include Z500 & T850 | Train 2009–2015, val 2016, test 2017–2019 | Done, enforced by `experiment.py` |

**Deadline: 21 September 2026.**

---

## 3. Tools and workflow

### 3.1 Terminal and the conda environment

**In one sentence:** the terminal is a text window where you type commands, and a conda *environment* is a sealed box of Python plus exact library versions, so the code runs the same everywhere.

**In your project:** the environment is `dlwp` (from `environment.yaml`). We run it as `/opt/anaconda3/envs/dlwp/bin/python`. On this Mac every command also needs `KMP_DUPLICATE_LIB_OK=TRUE`, a workaround for two copies of a maths library (OpenMP) loading at once.

**Check yourself:** why use a named environment instead of "whatever Python is installed"?

### 3.2 Git: history for your code

**In one sentence:** git records every change to your files as a *commit* (a snapshot with a message), so you can see what changed, when, and why, and go back if needed.

| Word | Plain meaning |
|---|---|
| **repository (repo)** | A folder whose history git tracks |
| **commit** | A saved snapshot plus a message explaining it |
| **clone** | Download a copy of a repo |
| **fork** | Your own copy of someone else's repo on GitHub |
| **remote** | A named link to an online copy of the repo |
| **push / pull / fetch** | Upload your commits / download and apply others' / download without applying |
| **stash** | Put unsaved changes in a drawer temporarily |
| **merge conflict** | Two edits to the same lines; git asks you to choose |
| **.gitignore** | A list of files git should never track (data, logs, checkpoints) |
| **PR (pull request)** | Asking someone to add your commits to *their* repo. We never do this to Jannik |

**In your project:**
- `upstream` = **Jannik's repo** (`jthuemmel/dlwp-labs`). **Fetch-only.** Its push address is deliberately set to `DISABLED--never-push-to-jannik`, so nothing can ever change his repo.
- `origin` = **your fork** (`vedant-mk/dlwp-labs`), public, where your work is backed up.
- We pulled 5 new labs from Jannik, resolved two conflicts by taking his version (which kept his `use_bottleneck=False` numerical bugfix), and have pushed our work to your fork ever since.
- Big files (data, forecasts, checkpoints) are in `.gitignore`. A committed 8.7 MB checkpoint once made a push fail (§15).

**Check yourself:**
- What is the difference between `origin` and `upstream` here?
- Why is it impossible for this folder to change Jannik's repo?

### 3.3 Google Cloud, and staying at $0

**In one sentence:** Google Cloud rents computers by the hour; you only pay while something *exists*, and your costs come from a university voucher.

| Cloud word | Everyday version | Yours |
|---|---|---|
| **Billing account** | Wallet | Billing Account for Education (no card attached) |
| **Credits** | Gift voucher | $50, "Deep Learning for Weather Forecasting" |
| **Project** | Folder for rented machines | `dlwp-vedant-2026` |
| **API** | A service you switch on (free to enable) | Compute Engine, Cloud Quotas |
| **Quota** | Permission to rent a GPU at all | Raised from 0 to 1 (approved) |
| **Budget alert** | Email warning when spending grows | $10 |

**In your project:** we have **not used the cloud for training**. Your Mac's GPU (`mps`) turned out fast enough. Zero machines exist, and this Mac is logged out of `gcloud` until you reconnect.

**Check yourself:** why can your own money not be charged? (No card; worst case, machines stop when credits run out.)

---

## 4. The data: weather basics

### 4.1 Variables and pressure levels

**In one sentence:** weather is described by a few quantities (temperature, wind, humidity, height of pressure surfaces) measured at several altitudes, which meteorologists label by *pressure* instead of metres.

| Level | Rough altitude | Why it matters |
|---|---|---|
| surface / 2 m | ground | What people feel (`T2M`) |
| 850 hPa | ~1.5 km | Just above the boundary layer; air masses (`T850`) |
| 500 hPa | ~5.5 km | Mid-atmosphere; the "steering flow" (`Z500`) |
| 250 hPa | ~10 km | Jet stream level |

| Variable | Meaning |
|---|---|
| `Z` (geopotential) | Height of a pressure surface. Its highs/lows are ridges and troughs, i.e. weather systems |
| `T` | Temperature |
| `Q` | Specific humidity (kg of water per kg of air) |
| `U`, `V` | West-east and south-north wind |
| `T2M` | Temperature 2 m above ground |
| `TP6h` | Precipitation accumulated over 6 hours |

**In your project:** 17 variables: `T2M`, `Z/T/Q/U/V` at 850/500/250 hPa (the course baseline's 16) plus `TP6h`. The brief requires `Z500` and `T850`.

**Check yourself:** why is Z500 the classic benchmark variable?

### 4.2 ERA5 and WeatherBench 2

**In one sentence:** ERA5 is the best reconstruction of past weather (observations blended into a physics model to fill every gap), and WeatherBench 2 packages it at coarse resolutions with standard rules for fair comparison.

**In your project:** ERA5 at **5.625°** from the WeatherBench 2 public bucket (read anonymously, free). ERA5 natively is ~31 km, hourly, 1940–present.

**Check yourself:** why train on reanalysis rather than raw station data? (No gaps, physically consistent.)

### 4.3 The grid is a flat picture of a round Earth

**In one sentence:** the map is a 32 × 64 rectangle of cells, but on the sphere those cells shrink towards the poles, and the left and right edges touch.

**In your project (Step 2, `explore/step2_space.py`):**
- Cell: 5.625° × 5.625° ≈ 625 km × 625 km at the equator, but only **31 km** wide near the pole.
- An equatorial cell covers **20.4×** the area of a polar one, exactly `cos(latitude)`.
- Longitude wraps: column 63 and column 0 are neighbours.
- That's why scores are **latitude-weighted** (each cell counts by its true area).

**Check yourself:** what goes wrong if you average errors over all 2,048 cells equally?

---

## 5. The data: how it is stored and split

### 5.1 Files: zarr, NetCDF, xarray

**In one sentence:** weather data is a stack of labelled grids (time × latitude × longitude per variable); `xarray` handles those labels in Python, and `zarr` stores them in chunks you can load piece by piece.

**In your project:**

| Store | Contents |
|---|---|
| `data/era5_5p6.zarr` | Training: 2009–2015, 17 variables, 10,224 six-hourly states |
| `data/era5_eval_5p6.zarr` | 2016–2019 (validation + test) |
| `data/clim_wb2_5p6.zarr` | WeatherBench 2 climatology 1990–2019 |
| `data/stats_train.zarr` | Mean and std of each variable over **2009–2015 only** |

**Chunking lesson:** pressure-level variables are stored with all 13 levels in one chunk, so downloading `Z500` alone still downloads all 13 levels. Wire cost is per *variable*, not per level (§15).

### 5.2 Train / validation / test, and leakage

**In one sentence:** you learn on training data, make decisions on validation data, and touch the test data **once** at the end, otherwise you fool yourself.

**In your project:** train 2009–2015 · validate 2016 · test 2017–2019.
- `experiment.py` refuses configs that train past 2015, drop Z500/T850, or compute statistics from the evaluation store.
- **Our own slip (being fixed):** two decisions were made partly after seeing test-year results. See §13.4.

### 5.3 Standardisation

**In one sentence:** before training, each variable is rescaled to mean 0 and standard deviation 1, so temperature (~280 K) and humidity (~0.003) are on comparable scales.

**In your project:** the statistics come from **training years only**. Using test years there would leak future information into the inputs.

**Check yourself:** why must the standardisation statistics exclude 2017–2019?

### 5.4 The two free baselines

| Baseline | Forecast rule | Hard to beat when |
|---|---|---|
| **Persistence** | "The weather stays as it is now" | Short leads (6 h to 1 day) |
| **Climatology** | "The usual weather for this date and hour" | Long leads (days) |

A model is only useful if it beats **both** at the leads it claims.

**In your project:** Z500 persistence RMSE grows from 218 at 6 h to 1,001 at 5 days, while climatology stays flat at ~781.

---

## 6. Scales, spectra and predictability: the physics

### 6.1 Wavenumber and power spectrum

**In one sentence:** any field around a latitude circle can be split into waves; *wavenumber k* is how many times a wave fits around the Earth, and the *power spectrum* says how much variance each wave size carries.

- k = 1: one wave around the globe (planetary scale, ~40,000 km)
- k = 16: 16 waves, each ~2,500 km at the equator
- Wavelength at the equator ≈ 40,075 km ÷ k

We compute it with a Fourier transform (FFT) along longitude. Longitude wraps exactly, so no tricks are needed.

**In your project (Step 3, `explore/step3_variables_scales.py`):** share of variance at small scales (k ≥ 9):

| Variable | Small-scale share |
|---|---|
| Z500 | **3%** |
| T850 | 7% |
| Q700 | 25% |
| TP6h | **47%** |

A **15× spread** across fields that the baseline embeds with identical 4×4 patches.

### 6.2 Predictability depends on scale (Lorenz 1969)

**In one sentence:** small weather features turn over quickly, so their errors double in hours, while large features evolve slowly and stay predictable for days.

**Why it matters:** "how predictable is the weather?" has a *different answer for each scale*. That is the physical basis of our inductive bias.

### 6.3 Why training on MSE makes forecasts blurry

**In one sentence:** when the model can't know where a small feature will be, the guess that minimises squared error is the *average* of all possibilities, which is a smooth smear.

- **Double penalty:** a sharp feature slightly misplaced is punished twice (missing where it is, wrong where it isn't); a blurred one only once. So RMSE rewards blur.
- **Consequence:** a model can win on RMSE while losing realistic detail. That's why we also look at spectra (§12.4).

**Check yourself:** explain the double penalty with a rain cell that's 300 km off.

---

## 7. Deep learning fundamentals you need

| Concept | In one sentence | In your project |
|---|---|---|
| **Neural network** | A function with many adjustable numbers (parameters) that maps input to output | Weather now → change in 6 h |
| **Parameters / weights** | The adjustable numbers | Baseline: **742,784** |
| **Loss** | A number measuring how wrong the output is | MSE, or the pyramid loss for C |
| **Gradient descent** | Nudge every parameter slightly in the direction that lowers the loss | Optimiser **AdamW**, learning rate 1e-3 |
| **Batch** | Samples processed together per update | 16 |
| **Step** | One parameter update | 6,000 per run |
| **Learning-rate schedule** | Change the learning rate over training | Warm up 100 steps, then cosine decay |
| **Validation** | Measuring on unseen data during training, to catch overfitting | Year 2016, logged as `val/loss_step1..4` |
| **GPU / MPS** | Hardware that does the many parallel multiplications fast | Your Mac's chip, `mps:0` |
| **Seed** | The starting number for all randomness in training | 0, 1, 2 |

### 7.1 Seeds, and why we run three

**In one sentence:** the same model trained with different random starts gives slightly different results, so a single run can't tell a real improvement from luck.

**Analogy:** testing a study method on one student vs three. One improved student could be a good day; three is evidence.

**In your project:** every model runs with seeds 0, 1, 2. A difference counts only if all three agree and the mean difference is at least **2× the seed spread** (decided in advance, §13.3).

---

## 8. Transformers and the Vision Transformer (ViT)

### 8.1 Patches and tokens

**In one sentence:** the map is cut into small squares (patches), and each patch is compressed into one vector of numbers (a token) that the network works with.

**In your project (baseline):** 32×64 grid → **4×4 patches** → 8×16 = **128 tokens**, each covering ~2,500 km. All 17 variables in a patch (272 numbers) become **one 128-number token**.

### 8.2 Positional encoding

**In one sentence:** tokens have no built-in sense of location, so each gets a code telling the network where on the map it sits.

**In your project:** a sine/cosine code on the 8×16 token grid. It treats the map as flat, so it doesn't know longitude wraps (§4.3).

### 8.3 Attention

**In one sentence:** each token asks "which other tokens are relevant to me?" and mixes in information from them, weighted by relevance.

- Each token makes a **query** (what I'm looking for), a **key** (what I contain) and a **value** (what I share).
- Relevance = how well my query matches your key.
- **Multi-head:** several attention patterns in parallel. We use **4 heads × 32 numbers**.

### 8.4 A transformer block

**In one sentence:** attention (tokens talk to each other), then a small feed-forward network (each token thinks on its own), each wrapped in normalisation and a skip connection.

**In your project:** 4 blocks; RMSNorm; gated feed-forward (SiLU), 2× expansion; DropPath off.

### 8.5 Where the 742,784 parameters are

| Part | Parameters |
|---|---|
| Patch embedding (17 × 4 × 4 → 128) | 34,816 |
| Positional encoding (128 × 128) | 16,384 |
| 4 transformer blocks (164,160 each) | 656,640 |
| Un-embedding (128 → 17 × 4 × 4) | 34,816 |
| Norm | 128 |
| **Total** | **742,784** |

**Check yourself:** where would you expect patch-boundary seams to come from? (Each token writes its own 4×4 patch independently in the un-embedding.)

---

## 9. Turning a network into a forecaster

### 9.1 Autoregressive rollout

**In one sentence:** predict 6 hours ahead, feed that prediction back in, and repeat, so 20 steps give 5 days.

**Why it's hard:** errors feed into the next step and compound.

### 9.2 Predicting the change (residual / tendency)

**In one sentence:** instead of drawing the whole next map from scratch, the network predicts the *difference* and adds it to the current map: `next = now + net(now)`.

**In your project:** the shipped baseline had no such shortcut and **lost to persistence** (loss 1.26× persistence's). With the residual it **beat persistence** (0.73×). Now applied identically to all three models (`residual: true`).

### 9.3 Rollout training

**In one sentence:** during training, let the model run several steps on its own outputs and learn from the accumulated error, so it learns to recover from its own mistakes.

**In your project:** 4,000 steps predicting one step ahead, then 2,000 steps rolling out 4 steps with gradients through the whole chain (Lab 3 Part 2).

**What it actually fixed. This was corrected after a proper control experiment.** Early on, a model trained for only 2,000 steps exploded at 5 days (Z500 RMSE ~17,000–19,600) and the rollout-trained model didn't, so it *looked* as though rollout training prevented the blow-up. The fair test compares models trained for the **same** number of steps (on 2016 validation data):

| Z500 RMSE, 2016 | 6 h | 3 d | 5 d |
|---|---|---|---|
| single-step, 2,000 steps | 247 | 3,061 | 16,874 |
| single-step, 6,000 steps | **181** | 934 | 1,177 |
| rollout-trained, 6,000 steps | 185 | **864** | **1,027** |

So the blow-up was mostly **undertraining**. Rollout training's real contribution is **~7% better at 3 days and ~13% at 5 days**, for a ~2% cost at 6 h.

**Lesson:** when two things change at once (here, training length *and* rollout training), you can't credit either until you test them separately.

**Check yourself:**
- Why does error compound in a rollout?
- Why was "rollout training fixed the blow-up" an unfair conclusion, and what control fixed the reasoning?

---

## 10. Inductive biases, and why we chose hierarchical scales

### 10.1 What an inductive bias is

**In one sentence:** a built-in assumption that steers what a model learns easily. It's the neural-network version of feature engineering: you encode knowledge in the design instead of in hand-made columns.

**Example:** a CNN assumes nearby pixels matter together; a plain ViT assumes almost nothing about space.

### 10.2 The four candidates, and what the labs already covered

| Candidate | Already in the labs? |
|---|---|
| Uncertainty quantification | **Fully**: Lab 4 (CRPS loss, ensembles, three noise injections) |
| Heterogeneous variables | **Fully**: Lab 3 Part 3 (per-variable embedding + per-variable loss weights) |
| Spherical geometry | **Partly**: Lab 3 Part 1 (cos-latitude loss only) |
| **Hierarchical scale interactions** | **Not at all** → our choice |

**Why:** picking a fully covered bias would mean re-running the instructor's own exercises. Scales was the only open option, and it has strong physics behind it (§6).

**Check yourself:** name the physical argument for hierarchical scales in two sentences.

---

## 11. Our two variants: A (architecture) and C (loss)

### 11.1 Variant A: multi-scale patches (`configs/multiscale.yaml`)

**In one sentence:** cut the map at two patch sizes at once, let fine and coarse tokens attend to each other, and add the two outputs.

```
                ┌─ 2×2 patches → 512 fine tokens   (~1,250 km)
weather map  ───┤
                └─ 8×8 patches →  32 coarse tokens (~5,000 km)
                         │
          one sequence of 544 tokens → 4 transformer blocks
                         │
      fine output + coarse output = predicted change
```

**How it encodes the bias:** separate representations per scale, direct interaction between scales, and an additive large-scale + detail decomposition.

**Fair comparison:** token size reduced from 128 to **100** so parameters match (731,656 = 0.985× baseline).
**Hidden trade-off:** that moves capacity from thinking to reading/writing patches:

| | Baseline | Multiscale |
|---|---|---|
| Patch embed/unembed + positions | 86,016 | 285,600 |
| Transformer blocks | 656,640 | 445,856 |

### 11.2 Variant C: pyramid loss (`configs/pyramid.yaml`, `utils/loss_fn.py`)

**In one sentence:** the network is the unchanged baseline, but the error is scored separately at three scales, with fine detail weighted most.

- **Laplacian pyramid:** split a field into finest detail (deviation from 2×2 averages), medium detail (from 4×4 averages), and the smooth remainder.
- **Weights 4 / 2 / 1:** fine bands carry little variance, so they're weighted up to count comparably.
- **Validation still uses plain MSE**, so all models are compared on the same number.

### 11.3 The research question

> **Does a model need multi-scale *machinery* (A), or is multi-scale *supervision* (C) enough?**

**Check yourself:** why is comparing an architecture change against a loss change more informative than comparing two architecture changes?

---

## 12. How forecasts are judged: verification

### 12.1 RMSE (latitude-weighted)

**In one sentence:** the typical size of the error, with each cell weighted by its true area.
Lower is better. Averaged over 864 forecasts starting through 2017–2019.

### 12.2 ACC (anomaly correlation)

**In one sentence:** how well the forecast's departures from normal line up with reality's departures from normal. 1 = perfect pattern, 0 = no skill.
It checks the *pattern*, not just the size of the error.

### 12.3 Skill score

`skill = 1 − RMSE_model / RMSE_climatology`: 1 = perfect, 0 = no better than climatology, below 0 = worse.

### 12.4 E1: spectrum ratio (our extra figure 1)

**In one sentence:** divide the forecast's power spectrum by ERA5's at the same times. 1 = realistic detail, below 1 = blurred, above 1 = fake noise.
**Why it complements RMSE:** RMSE rewards blur, the spectrum exposes it.

### 12.5 E2: per-variable RMSE heatmap (our extra figure 2)

**In one sentence:** percentage RMSE change vs the baseline for all 17 variables at every lead. Blue = better, red = worse.
**Why it complements the required figure:** the required figure shows only Z500 and T850, both smooth large-scale fields; the scale question matters most for humidity and precipitation.

---

## 13. Doing the science honestly: experimental design

### 13.1 Change one thing at a time

All three models share data, variables, steps, schedule, residual and rollout training. **A** changes only the network. **C** changes only the loss.

### 13.2 Parameter matching

A bigger model might win just from size, so A is shrunk to the same parameter count (§11.1).

### 13.3 Pre-registration

**In one sentence:** write down your predictions and your rule for "real" *before* seeing results, so you can't reshape the story afterwards.

**Ours (in `explore/DATA_NOTES.md`):**
- Real = all 3 seeds agree in sign **and** mean difference ≥ 2× seed spread.
- Hypotheses: (1) baseline reproducible; (2) A improves humidity 5–8% at 1–3 days; (3) A degrades Z500/T850/T2M at 5 days; (4) C cuts spurious small-scale power 15–30%; (5) C's RMSE within seed noise.

### 13.4 Our two methodology fixes (done on 2016 validation data)

Both decisions had first been justified with the wrong data. We re-checked them using **only 2016**, never the test years (`explore/validate_on_2016.py`, `explore/figures/validation2016.md`):

1. **Residual.** It had been checked on December 2015, *inside* the training years. On 2016: plain model **1.31×** persistence's loss, residual **0.75×**. The decision stands.
2. **Rollout training.** It had been motivated by the *test-year* blow-up. On 2016, with an **equal-length control**, it improves 3–5 day skill by 7–13%. The decision stands, but for a different reason than first thought (§9.3).

**Lesson:** decisions belong on validation data. And if you only compare "old setup" against "new setup", a hidden second difference can take the credit.

### 13.5 Negative results are results

A prediction that fails (e.g. "A removes the patch seams" was wrong) is reported, not hidden.

---

## 14. Timeline: what we did, in order

| When | What | Why |
|---|---|---|
| Sep 10 | Found the repo; it was only at Lab 1 | Session started with no folder |
| Sep 10 | Refused two accidental `/create-pr` commands | Would have opened a public PR to Jannik's repo |
| Sep 11 | Pulled Labs 2–5 + reference code from Jannik | Missing files the project needs |
| Sep 11 | Read the brief, the baseline ViT, Labs 3 and 4 | Found 2 of 4 biases already covered |
| Sep 11 | Chose hierarchical scales | Only uncovered option, strong physics |
| Sep 11 | Data tour Steps 0–3 | Grid geometry, variables, spectra |
| Sep 11 | Filled `utils/` by adapting `references/` | Reference code didn't match `experiment.py` |
| Sep 11 | Set up fork backup (`origin`) and GitHub login | Work was only on the laptop |
| Sep 11 | Built training store 2009–2015, 17 variables | Brief recommends 5+ years |
| Sep 11 | Built eval/climatology/stats stores; smoke test passed | Prove the pipeline end to end |
| Sep 12 | Found the baseline loses to persistence → added residual | Controlled diagnostic |
| Sep 12 | Implemented A and C | The two variants |
| Sep 12 | Found 5-day blow-up → rollout training; validation loss fixed to plain MSE | Required figure was meaningless |
| Sep 12 | Google Cloud project, quota approved; then logged out | Option for parallel runs; kept at $0 |
| Sep 12 | Seed-0 runs for all three; extra figures E1 and E2 | First full comparison |
| Sep 12 | Found patch seams at wavenumber 16 | New result about scales |
| Sep 12 | Pre-registered seed criteria; started seeds 1–2 overnight | Separate real effects from luck |
| Sep 12 | Wrote this learning guide | To explain the project properly |
| Sep 13 | All 9 runs done; hypotheses judged: H1–H4 hold, H5 fails | Pre-registered rule |
| Sep 13 | Residual and rollout decisions re-checked on 2016 only | Keep test data out of decisions |
| Sep 13 | Equal-length control showed the blow-up was undertraining | Separate two confounded changes |
| Sep 13 | Required RMSE figure over seeds (F1) | Brief requirement |
| Sep 13 | Round 2: seam smoother, A′, B′ control, C′ milder weights, 9 runs | Is A's failure the idea or the seams? |
| Sep 13 | P6–P9 all hold; smoother extends skill by over a day | Pre-registered rule |

---

## 15. Bugs and lessons

| What went wrong | Lesson |
|---|---|
| Appending years to the zarr store failed mid-way, leaving `T2M` at 2 years and others at 1 | Verify writes; prefer one atomic write over incremental appends |
| A watcher using `pgrep -f name` matched **its own command** and waited forever | Match on log contents or a PID, not on a string in your own command |
| `scores.csv` had a fake variable `time` with timestamps as values | Reference `rmse` kept the per-forecast axis; Lab spec averages over forecasts |
| Validation loss used the training objective | Compare models on one fixed metric (plain MSE) |
| Pushing a model checkpoint made GitHub reject the push | Keep large artefacts out of git |
| Parameter matching first used the wrong head count | Check against the actual config, not a formula |
| Disk would have filled with 6 × 2.4 GB forecasts | Budget disk before long runs; keep only what the paper needs |
| My first "no cloud resources" check miscounted its own error messages | Look at raw output before trusting a count |
| Early "pyramid −16.6% at 5 days" disappeared with proper training | Early results on a broken setup don't transfer |
| "Rollout training fixed the 5-day blow-up" was really "more training fixed it" | Test one change at a time; add an equal-length control |
| A disk "leak" during seed runs was macOS swap (11 GB) | Check what grew before blaming your own files |
| The driver stopped on a disk reading taken seconds after deleting 2.4 GB | Freed space lags; re-check before giving up |
| Six colours failed the colour-blind check | Group by family: hue for family, line style for version |
| "Baseline beats persistence to ~3.5 days" was eyeballed; measured it is 4.5 | Compute crossings, don't read them off a plot |

---

## 16. Results so far

### 16.0 Final verdicts: 3 seeds per model, pre-registered rule

*A difference is real only if all 3 seeds agree in sign and the mean is at least 2× the seed standard deviation. Full tables: `explore/figures/seed_analysis.md`.*

| Hypothesis | Verdict | Evidence (mean ± std over seeds) |
|---|---|---|
| H1: baseline reproducible | ✅ **holds** | Largest seed spread **0.59%** of RMSE |
| H2: A improves humidity 5–8% at 1–3 days | ✅ **holds** | All 9 cells real: **−6.7% to −9.0%** |
| H3: A degrades smooth fields at 5 days | ✅ **holds** | Z500 **+27.7 ± 5.4%**, T850 **+36.1 ± 6.5%**, T2M **+58.1 ± 5.5%** |
| H4: C cuts spurious small-scale power 15–30% | ✅ **holds** | Z500 **−27.9%** at 5 d, T850 **−20.9%**, Q850 −12.8% |
| H5: C's RMSE within seed noise | ❌ **fails** | Small but **real cost**: Z500 +0.8% to +3.6%; T850 +1.2% to +2.4% up to 3 d; tie at 5 d |

**The story these support:**
> *Architecture (A):* ~8% better humidity, but its own patch seams add grid noise that builds up and damages smooth fields by 28–58% at 5 days.
> *Loss (C):* 15–30% less spurious small-scale noise, for a small, consistent accuracy cost (≤3.6%) that shrinks with lead time.

**Why H5 failing matters:** we'd hoped C was free. With a reproducible baseline (spread 0.6%), even +2% is measurable. Writing the rule down first stopped us from calling +2% "basically the same".

*The sections below are the original seed-0 numbers, kept for reference.*

### 16.1 Required variables

| Z500 RMSE | 6 h | 1 d | 3 d | 5 d |
|---|---|---|---|---|
| Baseline | 184 | 461 | 856 | 1,017 |
| C: pyramid | 189 (+2%) | 476 (+3%) | 871 (+2%) | 1,023 (+1%) |
| A: multiscale | 210 (+14%) | 504 (+9%) | 949 (+11%) | 1,300 (+28%) |
| Persistence | 218 | 570 | 900 | 1,001 |
| Climatology | 781 | 781 | 781 | 781 |

| T850 RMSE | 6 h | 1 d | 3 d | 5 d |
|---|---|---|---|---|
| Baseline | 1.10 | 2.31 | 3.84 | 4.46 |
| C: pyramid | 1.12 | 2.36 | 3.89 | 4.46 |
| A: multiscale | 1.17 | 2.46 | 4.46 | 6.35 (+42%) |

### 16.2 All variables (E2)

- **A:** humidity **5–8% better** at 1–3 days; Z, T and T2M **much worse** at 5 days (up to +60% for T2M).
- **C:** within ±3% almost everywhere; slightly better for winds, humidity and precipitation by day 5.

### 16.3 Spectra (E1)

- **Baseline:** blurred at storm scales (k 3–8, ratio 0.5–0.8) **and** fake noise at small scales, with a spike at **k = 16** (64 cells ÷ 4-cell patches). Confirmed by larger jumps across patch boundaries.
- **C:** the least fake noise. Z500 small-scale ratio at 5 days: **94** vs baseline 135.
- **A:** the most fake noise (**793**), with new spikes at its own patch periods (k = 8 and 32); humidity partly *blurred*, which helps its RMSE.

### 16.4 Predictions vs outcomes

| Prediction (written before running) | Outcome |
|---|---|
| C improves spectrum without better RMSE | Held |
| A helps small-scale variables | Humidity yes, precipitation no |
| A helps at short leads | Only humidity and V winds |
| A removes the k = 16 seam | Wrong: it added more |

---

### 16.5 Round 2: fixing the seams, and tuning the loss (3 seeds each)

**Why round 2:** round 1 left one big question: did A fail because of the *multi-scale idea*, or because of *how it wrote its patches out* (the seams)? It also tested only one pyramid weight setting.

**The new ingredient: a seam smoother.**
- **In one sentence:** a tiny 3×3 layer applied to the output map, which blends each grid cell with its neighbours across patch edges, wraps around in longitude, and starts out doing nothing (the identity).
- **Why it helps:** each token writes its patch independently, so patches don't join smoothly. The seams become fake small-scale noise that builds up over 20 forecast steps. The smoother lets neighbouring patches agree at their edges.
- **Cost:** 2,618 extra parameters (0.35%).

**Three new models:** B′ = baseline + smoother (**the control**), A′ = multi-scale + smoother, C′ = pyramid loss with milder weights 2/1.5/1.

**Why the control (B′) was essential:** if we'd only built A′, any improvement could be "smoothing helps anything". B′ separates the two: the gain from smoothing alone, versus what multi-scale adds on top of it.

| Prediction (written before running) | Verdict |
|---|---|
| P6: seams cause A's damage | ✅ holds: A′ vs A at 5 days, Z500 −29%, T850 −32%, T2M −31% |
| P7: humidity gain survives | ✅ holds: A′ vs baseline, Q500 −22%, Q250 −19% at 1 day |
| P8: smoother matters more for A (5 days) | ✅ holds: −29% for A vs −9% for the baseline |
| P9: milder pyramid weights cost less | ✅ holds: RMSE cost gone; spurious power still −20% at 5 days |

**Useful skill, measured** (how long a model stays better than the free baselines):

| Model | Z500 beats climatology until | Z500 beats persistence until |
|---|---|---|
| Baseline | 2.4 days | 4.5 days |
| **B′: baseline + smoother** | **3.6 days** | beyond 5 days |
| **A′: multi-scale + smoother** | **3.7 days** | beyond 5 days |

**What multi-scale adds on top of the smoother** (A′ vs B′; *exploratory*, chosen after seeing seed 0): humidity **12–20% better** at 1–3 days, Z500/T850 ~5% better at 1 day, but **T2M ~8% worse at every lead**.

**The updated story for the paper:**
1. The plain ViT's biggest scale defect is its **patch seams**. A 3×3 layer that lets neighbouring patches interact cuts Z500 error by a quarter at 1 day and adds over a day of useful skill.
2. With seams fixed, a **multi-scale architecture adds a large, robust humidity gain**, the field with the most small-scale structure, exactly as the bias predicts, at a consistent cost for surface temperature.
3. A **scale-aware loss** gives a tunable realism/accuracy trade-off: 4/2/1 buys ~28% less fake noise for up to 3.6% RMSE; 2/1.5/1 buys ~20% for essentially nothing.

**Check yourself:**
- Why did we need B′ even though the question was about A?
- Why is the A′ vs B′ humidity result called "exploratory"?
- What does "beats climatology until 3.6 days" mean, and why is it a good way to summarise skill?

## 17. Open questions and next steps

1. ~~Seeds 1–2~~ ✅ done: H1–H4 hold, H5 fails (§16.0).
2. ~~Methodology fixes on 2016~~ ✅ done: both decisions stand; rollout explanation corrected (§9.3, §13.4).
3. ~~Redraw E1 and E2 with seeds~~ ✅ done (mean lines, seed-range bands, hatching for non-robust cells). The required RMSE figure with seed means and ranges is done too (`explore/figures/F1_rmse_z500_t850.png`).
4. Pyramid's T2M error alternates with lead time, consistently across seeds. Probably the daily cycle; decide whether to mention it.
5. ~~Round 2: seam smoother, A′, B′, C′~~ ✅ done, P6–P9 all hold (§16.5).
6. **Framing decision for the paper:** the brief asks for two ways to add the bias. Candidates: **A′ (architecture)** vs **C′ or C (loss)**, with the baseline, A and B′ as ablations explaining *why*.
7. Heatmap panel titles are slightly crowded; shorten when making paper figures.
8. Next: **write the paper.**
4. **Write the paper:** Intro → Methods → Results → Discussion, ≤2200 words, TCCML LaTeX.
5. **Verify every reference** (DOIs, pages) before submission.
6. Possible discussion point: A's patch artefacts could be reduced with overlapping or smoothed un-embedding (future work).

---

## 18. How this compares to your ML4TimeSeries project

| Concept | Grid-load project | This project |
|---|---|---|
| Target | One number per hour | 17 variables × 32 × 64 grid, every 6 h |
| Split | Train 2015–2024, test 2025 | Train 2009–2015, val 2016, test 2017–2019 |
| Baseline | Ridge regression | Persistence and climatology |
| Model | XGBoost on hand-made features | Neural network learning its own features |
| Multi-step | **Direct**: predict all 24 h at once | **Recursive**: feed predictions back 20 times |
| Design knob | Features | Architecture and loss = inductive bias |
| Question | Best MAE | Does a physics idea help, and why? |

**Same skills:** time splits, baselines, error by horizon, one change at a time.
**New skills:** gridded data, neural architectures, autoregressive rollouts, spectra, seeds.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **ACC** | Anomaly correlation: pattern agreement of departures from normal |
| **Autoregressive** | Each prediction becomes the next input |
| **Climatology** | The average weather for a date and hour |
| **Conditional mean** | The average of all plausible outcomes; what MSE rewards |
| **Double penalty** | Misplaced sharp features punished twice by RMSE |
| **ERA5** | ECMWF reanalysis of global weather since 1940 |
| **FFT** | Fast Fourier transform: splits a signal into waves |
| **hPa** | Hectopascal, a pressure unit; lower = higher altitude |
| **Inductive bias** | A built-in assumption steering what a model learns |
| **Laplacian pyramid** | Splitting a field into detail bands at successive scales |
| **Latitude weighting** | Weighting each cell by `cos(latitude)` = its true area |
| **Lead time** | How far ahead the forecast is |
| **MPS** | Apple's GPU backend used by PyTorch on Mac |
| **Parameter matching** | Giving compared models the same number of weights |
| **Persistence** | Forecasting that nothing changes |
| **Pre-registration** | Stating predictions and criteria before seeing results |
| **Reanalysis** | Observations blended with a physics model to fill gaps |
| **Residual / tendency** | Predicting the change and adding it to the current state |
| **RMSE** | Root mean squared error |
| **Rollout training** | Training through several self-fed prediction steps |
| **Seed** | Starting value for all training randomness |
| **Spectrum ratio** | Forecast power ÷ true power per wavenumber |
| **Standardisation** | Rescaling to mean 0, std 1 |
| **Token** | A vector representing one patch |
| **Wavenumber (k)** | Number of waves around a latitude circle |
| **WeatherBench 2** | Benchmark packaging ERA5 with standard splits and metrics |
| **Zarr** | Chunked storage format for large labelled arrays |

---

## 20. Self-test: explain it out loud

Try each in under a minute, without notes.

1. What is the project, in five sentences?
2. Why hierarchical scales and not uncertainty quantification?
3. What does the Lorenz argument say, and why does it matter here?
4. Why does MSE make forecasts blurry? Use the double penalty.
5. What is a token in the baseline ViT, and how many are there?
6. What does attention do, in plain words?
7. Why did the shipped baseline lose to persistence, and how did we fix it?
8. Why did forecasts explode at 5 days, and what is rollout training?
9. How does A put the bias into the architecture? How does C put it into the loss?
10. Why was A's parameter count matched, and what hidden trade-off did that create?
11. What is the spectrum ratio, and why does it complement RMSE?
12. What is the k = 16 spike, and how did we confirm its cause?
13. Why do we run three seeds, and what is our rule for "real"?
14. Which two methodology issues did we find, and how are we fixing them?
15. Why can nothing in this project change Jannik's repo or charge your money?
16. Which hypothesis failed, and why is that a useful result?
17. What was confounded in "rollout training fixed the blow-up", and how did the control fix it?
18. What is the seam smoother, and why does it help so much?
19. Why was B′ needed to interpret A′?

---

## 21. Where things live in the repo

| Path | What |
|---|---|
| `experiment.py` | Train, test on 2017–19, write scores and required figures (course code) |
| `prepare_eval_data.py` | Build eval, climatology and statistics stores (course code) |
| `utils/` | The pipeline, adapted from `references/` |
| `utils/components.py` | `ViT`, `MultiScaleViT`, `Persistence` |
| `utils/loss_fn.py` | `f_mse`, `f_pyramid_mse`, `laplacian_pyramid` |
| `utils/lightning_module.py` | Training/validation/prediction steps, residual, rollout |
| `utils/metrics.py` | RMSE, ACC, skill, `scores` |
| `configs/baseline17.yaml` · `multiscale.yaml` · `pyramid.yaml` | Round-1 models |
| `configs/baseline17_smooth.yaml` · `multiscale_smooth.yaml` · `pyramid_w2.yaml` | Round-2 models (B′, A′, C′) |
| `explore/figures/seed_analysis_round2.md` | P6–P9 verdicts |
| `runs/<model>/seed<k>/` | Scores, loss curves, run info, forecasts |
| `explore/DATA_NOTES.md` | Detailed data tour, findings and pre-registration |
| `explore/REFERENCES.md` · `references.bib` | Papers and what each supports |
| `explore/extra_figures.py` | E1 and E2 |
| `explore/rmse_figure.py` | F1: required RMSE figure over seeds |
| `explore/diagnose_baseline.py` | Residual vs no-residual diagnostic |
| `explore/run_seeds.py` | Overnight seed driver |
| `explore/seed_analysis.py` · `figures/seed_analysis.md` | Hypotheses judged against 3 seeds |
| `explore/validate_on_2016.py` · `figures/validation2016.md` | Protocol decisions re-checked on 2016 |
| `explore/figures/` | All figures |
| `explore/ablations/` | Earlier runs kept as evidence |
| `explore/CLAIMS_AND_EVIDENCE.md` | Hoped vs got, brief checklist, every claim with its evidence and reference |
| `explore/LEARNING_GUIDE.md` | This guide |

---

## Changelog

| Date | Added |
|---|---|
| 2026-09-12 | First version: sections 1–21, covering everything up to seed-0 results and the start of seeds 1–2 |
| 2026-09-13 | Added `CLAIMS_AND_EVIDENCE.md` (claims ledger for writing); hardware confirmed as Apple M1 |
| 2026-09-13 | Round 2 (§16.5): seam smoother, A′/B′/C′, P6–P9 all hold, measured skill horizons; open questions, timeline, lessons, self-test, repo map updated |
| 2026-09-13 | Required RMSE figure over seeds (F1): models cross climatology at ~2.3 d (Z500); persistence crossing later measured at 4.5 d; §2 status updated |
| 2026-09-13 | §16.0 final seed verdicts; §9.3 rollout explanation corrected with the equal-length control; §13.4 validation fixes done; timeline, lessons, open questions, self-test, repo map updated |
