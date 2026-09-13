# Paper outline

**Working title:** *Seams, scales and supervision: adding hierarchical scale interactions to a Vision Transformer weather model*
(alternative, plainer: *Hierarchical scale interactions in a ViT weather forecaster: architecture versus loss*)

**Deliverable:** PDF via Digicampus by **21 Sep 2026**.
**Limits (Jannik's brief):** Introduction / Methods / Results / Discussion; **≤ 2200 words of text**, not counting figures, captions or references; TCCML NeurIPS 2026 LaTeX template.
**Evidence for every sentence:** `explore/CLAIMS_AND_EVIDENCE.md` (numbers, files, strength, references).

---

## 0. Decisions

**Settled 2026-09-13:** the two ways are **A′ (architecture)** and **C′ (objective)**; write and compile in **Overleaf** (upload `paper/overleaf_upload.zip`); an email to Jannik is drafted to confirm page length and an appendix for code.

### Original options considered

| Decision | Recommendation | Why |
|---|---|---|
| Which models are "the two ways"? | **Way 1 (architecture): A′ = multi-scale embedding + seam smoother. Way 2 (objective): C′ = scale-decomposed pyramid loss.** Baseline, A, B′, C are ablations | One bias, two clearly different entry points; the ablations explain *why* |
| Where does the seam smoother live in the story? | Part of the architectural way: coupling across patch boundaries is the finest scale interaction the ViT lacks. B′ isolates its contribution | Keeps "one bias"; avoids a third idea |
| Author name or anonymous? | Use `\usepackage[preprint]{tackling_climate_workshop_style}` (shows your name) | The default anonymised mode hides who wrote it |
| Code listings in the main text or an appendix? | **Appendix A**, referenced from Methods | Brief says "show the code"; listings would eat the word budget. Ask Jannik if unsure |
| Page length | Follow Jannik's word limit (figures don't count). Expect ~6–7 pages with figures, plus appendix | The workshop's 4-page rule is for the workshop, not the course. **Worth a one-line email to Jannik** |
| Where to compile | **Overleaf** (no install) or BasicTeX locally (`brew install --cask basictex`, needs your password) | `pdflatex` is not installed on this Mac |

---

## 1. Word budget

| Section | Words | Purpose |
|---|---|---|
| Abstract | 150 | Problem, two ways, headline numbers, recommendation |
| 1 Introduction | 350 | Why scales matter physically; what the ViT lacks; what we do |
| 2 Methods | 650 | Data, baseline and protocol, the two ways, evaluation, predictions |
| 3 Results | 650 | Accuracy, spectra, per-variable, the loss trade-off |
| 4 Discussion | 400 | Interpretation, recommendation, limitations |
| **Total** | **2200** | Leave ~100 words of slack when drafting |

---

## Abstract (≈150 words)

- Data-driven weather models built on Vision Transformers cut the globe into fixed patches, treating all spatial scales alike, although atmospheric predictability depends strongly on scale.
- We add **hierarchical scale interactions** to the course ViT in two ways: an **architectural** change (multi-scale patch embedding with a seam smoother coupling neighbouring patches) and an **objective** change (a Laplacian-pyramid loss).
- ERA5 at 5.625°, trained 2009–2015, tested 2017–2019, three seeds, pre-registered predictions.
- Headline: the baseline's dominant scale defect is **patch seams**; coupling patches extends skill over climatology for Z500 from **2.4 to 3.6 days**; multi-scale adds **12–20%** humidity improvement; the pyramid loss trades realism for accuracy along a tunable curve.
- One-sentence recommendation.

---

## 1 Introduction (≈350 words)

**Paragraph 1: context (≈90).** Data-driven global weather models [GraphCast, Pangu, FourCastNet, Keisler] now rival physics models; a ViT [Dosovitskiy] is a simple, common backbone. One sentence on climate relevance: skilful, cheap forecasts support early warning and adaptation.
→ Claims ledger §4.1 "Data-driven weather models: context".

**Paragraph 2: physics of scales (≈120).** Predictability is scale-dependent: small scales lose skill within hours, planetary waves persist for days [Lorenz 1969]. Variance falls steeply with wavenumber [Nastrom & Gage]. In our data, small scales (k ≥ 9) hold **3% of Z500 variance but 47% of precipitation's**. MSE training blurs what cannot be predicted (double penalty) [Subich 2025, VERIFY]; neural nets favour low frequencies [Rahaman].
→ §4.1 rows 1–3, 5–6.

**Paragraph 3: the gap (≈70).** The course ViT embeds all 17 variables in one 4×4-patch token and writes each patch independently: one scale for every field, no interaction below the patch size. Two courses of action exist: change what the model can represent (architecture) or what it is penalised for (loss).
→ §4.1 row 4.

**Paragraph 4: contributions (≈70), as a short list.**
1. Two ways to add scale interactions, compared against the baseline with three seeds and pre-registered predictions.
2. A spectral diagnosis showing the baseline's dominant defect is **patch seams**, and that fixing it extends useful skill by over a day.
3. A recommendation on which design choices to use.

---

## 2 Methods (≈650 words)

### 2.1 Data and protocol (≈120)
- ERA5 [Hersbach] via WeatherBench 2 [Rasp 2020, 2024] at 5.625° (32×64), 6-hourly.
- 17 variables: T2M; Z, T, Q, U, V at 850/500/250 hPa; TP6h.
- Train 2009–2015 (10,224 states), validate 2016, test 2017–2019 (864 initialisations, every 30 h, cycling through all four synoptic hours). Standardisation from training years only.
- All runs through the course `experiment.py`.
→ Ledger §4.2 row 1; compliance audit §3a.

### 2.2 Baseline and shared training protocol (≈150)
- Course ViT: 4×4 patches → 128 tokens, dim 128, 4 blocks, 4 heads, 742,784 parameters [Dosovitskiy; Thümmel labs].
- **Changes from the shipped baseline, applied identically to every model** (state plainly):
  - predict the tendency, x + f(x) [He; GraphCast]: on 2016 validation, loss 1.31× → 0.75× persistence;
  - roll-out training: 4000 single-step then 2000 four-step steps [GraphCast; Keisler; Lab 3]: −13% Z500 RMSE at 5 days vs an equal-length single-step control on 2016;
  - 7 training years and 17 variables.
- AdamW, lr 1e-3, batch 16, warm-up + cosine; validation loss is plain MSE for all models.
→ §4.2 rows 3–4; `explore/figures/validation2016.md`.

### 2.3 Way 1: architecture (≈150)
- **Multi-scale embedding (A):** embed at 2×2 (512 tokens) and 8×8 (32 tokens), concatenate, shared attention, sum of per-branch read-outs [CrossViT; MViT]. Width 100 to match parameters (0.985×).
- **Seam smoother:** 3×3 convolution on the output field, periodic in longitude, identity-initialised (+2,618 parameters). Motivation: independent patch read-out leaves discontinuities at the patch period [analogy: Odena 2016].
- A′ = A + smoother. **B′ = baseline + smoother** as a control.
- Code: Appendix A.
→ §4.2 rows 5, 7.

### 2.4 Way 2: objective (≈120)
- Laplacian pyramid [Burt & Adelson] with three bands (2×2 detail, 4×4 detail, low-pass); squared error per band, weighted, expanded back to the grid so latitude and variable weights still apply.
- Weights **4/2/1 (C)** and **2/1.5/1 (C′)**, rising towards fine scales because their variance is small.
- Network identical to the baseline.
- Equation: L = Σᵦ wᵦ · ‖Pᵦ(ŷ) − Pᵦ(y)‖².
→ §4.2 row 6.

### 2.5 Evaluation and predictions (≈110)
- Latitude-weighted RMSE and ACC against ERA5; persistence and WB2 climatology as references [Rasp 2024].
- Extra verification: zonal power-spectrum ratio forecast/ERA5 (realism: blur vs noise), and per-variable RMSE change (where each change helps).
- Three seeds each; a difference counts only if all seeds agree in sign and |mean| ≥ 2 × seed std, a rule fixed before results.
- Predictions written before running (Table 2), e.g. "multi-scale improves humidity", "the loss reduces spurious small-scale power at no accuracy cost".
→ §4.2 row 8; `DATA_NOTES.md` pre-registration sections.

**Table 1 (Methods): models.** Columns: model · change · parameters · train time per run (Apple M1, MPS) · seeds.

| Model | Change | Params | Train / run |
|---|---|---|---|
| Baseline | course ViT + shared protocol | 742,784 | 17 min |
| A | multi-scale embedding | 731,656 | 76 min |
| B′ | baseline + seam smoother | 745,402 | 14 min |
| A′ | multi-scale + seam smoother | 734,274 | 73 min |
| C | pyramid loss 4/2/1 | 742,784 | 18 min |
| C′ | pyramid loss 2/1.5/1 | 742,784 | 14 min |

(Test time per run: 24–96 s. Brief requirement: runtime and hardware.)

---

## 3 Results (≈650 words)

### 3.1 Accuracy over five days (≈170) · Figure 1 · Table 2
- **Fig. 1 (required):** RMSE of Z500 and T850 vs lead time, seed means with ranges, persistence and climatology. `explore/figures/F1_rmse_z500_t850.pdf`
- Key sentences:
  - Smoothed models (A′, B′) are best at every lead; Z500 beats climatology until **3.6–3.7 d** vs **2.4 d** for the baseline; beats persistence for all 5 days.
  - A alone is worst and degrades with lead (+28% Z500, +36% T850 at 5 d).
  - C and C′ are close to the baseline (C +0.8% to +3.6%; C′ within ~1%, e.g. Z500 +1.0% at 1 d).
- **Table 2:** skill horizons (days beating climatology / persistence) per model, plus RMSE at 1 d and 5 d.
→ §4.3; `explore/DATA_NOTES.md` round-2 skill horizons.

### 3.2 Why: spectra reveal patch seams (≈180) · Figure 2
- **Fig. 2 (extra verification 1):** spectrum ratio for Z500, T850, Q850, TP6h at 1 d and 5 d. `explore/figures/E1_spectrum_ratio.pdf`
- Key sentences:
  - The baseline is blurred at synoptic scales (ratio 0.5–0.8, k = 3–8) **and** noisy at grid scales, with a spike at **k = 16 = 64 / 4**, the patch period; neighbour jumps are largest on patch boundaries.
  - A adds spikes at its own patch periods (k = 8, 32); spurious power at 5 d rises from 144× to ~800× ERA5 (Z500).
  - The smoother removes 87–93% of A's spurious power (P6) and also sharpens precipitation.
- **Justification (brief requires it):** RMSE rewards blur; the spectrum measures realism independently of feature position.

### 3.3 Where: per-variable effects (≈170) · Figure 3
- **Fig. 3 (extra verification 2):** RMSE change heatmaps. For the paper, trim to three panels: **B′ vs baseline**, **A′ vs B′**, **C′ vs baseline** (full five-panel version in the appendix). Hatching marks non-robust cells.
- Key sentences:
  - The smoother alone improves dynamical fields strongly at 1–3 d (Z500 −25%, V500 −26% at 1 d) but barely touches humidity.
  - Beyond the smoother, multi-scale improves humidity by 12–20% at 1–3 d (exploratory comparison, stated as such) at a consistent T2M cost (+7–9%).
- **Justification:** the required figure shows only Z500/T850, which hold little small-scale variance; the bias should matter most for humidity and precipitation.

### 3.4 The loss trade-off (≈130)
- 4/2/1 cuts spurious small-scale power by 11–28% for a small, robust RMSE cost (up to 3.6%): the "no cost" prediction failed.
- 2/1.5/1 keeps ~60–85% of that reduction (Z500: −20% vs −28% at 5 d) with the RMSE cost gone.
- Point to the loss curves figure (Fig. 5) for training behaviour.

**Required figures not yet made in paper form:**
- **Fig. 4 (required): T850 forecast at 1 day.** One row: ERA5 truth · baseline error · A′ error (same initialisation, shared colour scale). Build from `runs/<model>/seed0` forecast slices; the auto-generated per-run PNGs exist as a fallback.
- **Fig. 5 (required): training and validation loss.** Validation (plain MSE, comparable) for all six models, step axis; training loss in a second panel with a caption noting pyramid losses are a different objective and the jump at step 4000 is the switch to roll-out training.

---

## 4 Discussion (≈400 words)

**Paragraph 1: interpretation (≈130).** The ViT's largest scale defect is not missing multi-scale machinery but *how patches are written out*: independent read-out leaves seams whose spurious variance accumulates over the roll-out. Coupling neighbouring patches is the cheapest, largest win. With seams fixed, the multi-scale embedding helps where small-scale structure dominates (humidity, ~25% small-scale variance), consistent with the inductive bias.

**Paragraph 2: predictions scorecard (≈70).** Table or sentence: 8 of 9 pre-registered predictions held; the failure (C costs accuracy) is informative. Name the one prediction that was wrong in round 1 (A removes the k = 16 seam).

**Paragraph 3: recommendation (≈100).** Brief requirement.
1. Always couple neighbouring patches (seam smoother): 0.35% more parameters, largest improvement.
2. Use a multi-scale embedding when moist or small-scale fields matter, together with the smoother; check T2M.
3. For realism, add a mild scale-aware loss (2/1.5/1).
4. Evaluate scale changes with spectra, not RMSE alone.

**Paragraph 4: limitations and future work (≈100).**
- Modest absolute skill: ~740k parameters, 5.625°, 7 years; all models lose to climatology after 2.4–3.7 d.
- Two protocol decisions were first motivated by test data; both re-checked on 2016. The round-2 design was also motivated by test-year spectra; the ranking holds on 2016 validation.
- A′ vs B′ was chosen after seeing seed 0 (exploratory).
- One smoother design, two loss weightings; the T2M cost is unexplained; deterministic only [GenCast, VERIFY].
- Future: overlapping un-embedding, weight sweep, ensembles.

---

## Appendix (not counted in the word limit, but keep short)

- **A. Code listings (brief: "show the code"):** `SeamSmoother`, `MultiScaleViT.forward`, `f_pyramid_mse` + `laplacian_pyramid` (from `utils/components.py`, `utils/loss_fn.py`). Trimmed of docstrings.
- **B. Full five-panel heatmap** (`E2_rmse_change_heatmap.pdf`).
- **C. Pre-registered predictions and verdicts table** (from `CLAIMS_AND_EVIDENCE.md` §2.1).
- **D. Protocol checks on 2016** (residual, equal-length roll-out control).

---

## References (numbered, `unsrt`)

All keys are in `explore/references.bib`. **Verify every entry before submission** (VERIFY-marked first: Subich 2025, GenCast, NeuralGCM).
Expected citations: lam2023graphcast, bi2023pangu, pathak2022fourcastnet, keisler2022gnn, dosovitskiy2021vit, lorenz1969predictability, nastrom1985climatology, subich2025doublepenalty, rahaman2019spectral, hersbach2020era5, rasp2020weatherbench, rasp2024weatherbench2, thuemmel2026dlwplabs, he2016resnet, chen2021crossvit, fan2021mvit, odena2016checkerboard, burt1983laplacian, price2025gencast.

---

## Build list (what must exist before writing is finished)

| Item | Status |
|---|---|
| Fig. 1 RMSE (required) | ✅ exists |
| Fig. 2 spectrum ratio (extra 1) | ✅ exists |
| Fig. 3 heatmap, trimmed to 3 panels | 🟡 make paper version |
| Fig. 4 T850 at 1 day, combined (required) | ⬜ to make |
| Fig. 5 loss curves, combined (required) | ⬜ to make |
| Table 1 models / params / runtime | ✅ numbers ready |
| Table 2 skill horizons | ✅ numbers ready |
| Appendix code listings | ⬜ to extract |
| LaTeX project (template + `main.tex` + `references.bib` + figures) | ⬜ set up in `paper/` or Overleaf |
| Reference verification | ⬜ before submission |

## Suggested schedule

| Date | Work |
|---|---|
| Sep 13–14 | Settle decisions in §0; make Figs 3–5; set up LaTeX project |
| Sep 15 | Methods (easiest: everything is known) |
| Sep 16 | Results |
| Sep 17 | Introduction and Discussion |
| Sep 18 | Abstract, appendix, word count, figure polish |
| Sep 19 | Verify references; full read-through against the claims ledger |
| Sep 20 | Buffer; proofread |
| Sep 21 | Submit on Digicampus |
