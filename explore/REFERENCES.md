# References — what each one supports in the paper

BibTeX: `explore/references.bib`. Written from memory, so **check every DOI, volume and page
before submission**; entries marked VERIFY in the .bib are the least certain.

## Introduction — why scales matter

| Key | Use it for |
|---|---|
| `lorenz1969predictability` | The core physical claim: flows with many scales have finite predictability, and errors at small scales spread upscale. Predictability differs by scale. |
| `nastrom1985climatology` | The observed atmospheric energy spectrum (~k⁻³ at synoptic scales, ~k⁻⁵ᐟ³ at mesoscales). Context for our zonal spectra in Step 3 (Z500 falls off steeper than k⁻³). |
| `rahaman2019spectral` | Neural networks learn low frequencies first ("spectral bias") — a reason a plain network under-represents small scales. |

## Data and benchmark (Methods)

| Key | Use it for |
|---|---|
| `hersbach2020era5` | The ERA5 reanalysis — the training and evaluation data. |
| `rasp2020weatherbench` | Origin of the 5.625° (64×32) grid and of the benchmark framing. |
| `rasp2024weatherbench2` | The data store, the 1990–2019 climatology reference, and the evaluation protocol (latitude-weighted RMSE, ACC). |
| `thuemmel2026dlwplabs` | The course pipeline: `experiment.py`, the baseline ViT, and lab 3's roll-out training. |

## Architecture (Methods, Variant A)

| Key | Use it for |
|---|---|
| `dosovitskiy2021vit` | The baseline: patch embedding + transformer blocks. |
| `he2016resnet` | Residual connection. We predict the tendency `x + f(x)`; justify it here and with GraphCast/Pangu below. |
| `chen2021crossvit` | **Closest precedent for Variant A**: embedding an image at two patch sizes and letting the scales exchange information. |
| `fan2021mvit` | Multi-scale token hierarchies in vision transformers. |
| `odena2016checkerboard` | **Analogy for the patch seams**: networks that write their output in strided blocks leave artefacts at the stride period. About transposed convolutions, not ViT patch un-embedding, so cite as an analogous mechanism, not direct precedent. |
| `liu2021swin` | Hierarchical, windowed attention — the road not taken (Variant B), and the backbone of Pangu-Weather. |

## Objective (Methods, Variant C)

| Key | Use it for |
|---|---|
| `burt1983laplacian` | The Laplacian pyramid decomposition our loss is built on. |
| `subich2025doublepenalty` | **Closest precedent for Variant C**: MSE's "double penalty" makes data-driven forecasts blur; a scale-aware (spherical-harmonic) loss counters it. Directly supports our motivation. VERIFY details. |

## Related data-driven weather models (Introduction / Discussion)

| Key | Use it for |
|---|---|
| `lam2023graphcast` | Tendency (residual) prediction, multi-step roll-out training, per-variable loss weights. Also a multi-mesh (multi-scale) graph — related to Variant A. |
| `bi2023pangu` | Hierarchical 3D Swin transformer with down/up-sampling — hierarchical scales in the architecture. |
| `pathak2022fourcastnet` | Fourier neural operator backbone — scale handling in spectral space. |
| `keisler2022gnn` | Roll-out training for a data-driven global model. Supports our training protocol. |
| `weyn2020cubedsphere` | Earlier CNN forecaster; handling sphere geometry (context for why we did not pick that bias). |
| `price2025gencast` | Ensembles avoid the blurring of deterministic MSE models. Discussion: Variant C attacks the same symptom without an ensemble. |
| `kochkov2024neuralgcm` | Hybrid physics/ML model; discussion of spectral realism of forecasts. |

## Claims from our own experiments to cite in the text (not papers)

- Baseline without residual scores 1.26× the persistence loss at 6 h; with residual 0.73×
  (`explore/diagnose_baseline.py`, 2000 steps).
- Single-step-trained models diverge past a day (baseline Z500 RMSE 19,609 at 120 h vs climatology 781);
  archived under `explore/ablations/*_2000steps_singlestep/`.
- Zonal spectra: Z500 holds ~3% of its variance at k ≥ 9, TP6h ~47% (`explore/DATA_NOTES.md`, Step 3).
