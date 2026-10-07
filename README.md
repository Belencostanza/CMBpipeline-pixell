# CMBpipeline-pixell

A new version of the CMB polarization pipeline [CMBpipeline](https://github.com/Belencostanza/CMBpipeline): Wiener filtering with the DeepWiener neural network, then an optimal quadratic estimator (OQE) of the power spectrum. It uses the [pixell](https://github.com/simonsobs/pixell) library to reduce or remove the approximations of CMBpipeline's projection step.

CMBpipeline projects spherical Q/U maps onto a tangent plane, interpolates them onto a regular grid and works with flat-sky 2D Fourier transforms. This repository works on the sphere: pixell CAR maps and curved-sky spherical-harmonic transforms.

The work is incremental. Each task is described in `TASK*.md`, documented in `docs/`, and analysed in `notebooks/`. CMBpipeline (`../CMBpipeline/source`) is used **read-only**, for its fiducial spectra, SO mask and reference functions. All tests use the Simons Observatory (SO) "rect" configuration.

## Repository layout

```
TASK1.md, TASK2.md            task descriptions
docs/
  01_current_pipeline.md      how CMBpipeline projects, interpolates and Fourier-transforms (with line references)
  02_test_plan.md             Task 1 test design
  03_task2_dataset_plan.md    Task 2 dataset design
  report_task1.html           Task 1 report (self-contained, with figures)
src/
  sky_input.py                reference sky: seeded a_lm, CMBpipeline spectra, beam
  cmbpipeline_a.py            pipeline A: thin wrappers around CMBpipeline functions
  pixell_tools.py             mask transfer to CAR, alm rotation, CAR patches
  spectrum_tools.py           binning, flat and curved-sky estimators, mode-mixing response
  config.dict                 Task 2 configuration (CMBpipeline format, IAS cluster paths)
  config_loader.py            reads config.dict (same logic as CMBpipeline)
  make_dataset.py             Task 2: training/validation set on CAR cut-out maps
  run_dataset.py              Task 2: builds the dataset from config.dict
notebooks/
  01_pixell_validation.ipynb  Task 1 analysis
  02_dataset_check.ipynb      Task 2 local check against a CMBpipeline map
results/                      local outputs (not versioned)
```

## Task 1: can pixell replace the projection + flat-sky Fourier step?

The same input a_ℓm and the same SO coverage go through three pipelines:

| | pipeline | pixels |
|---|---|---|
| **A** | CMBpipeline: HEALPix Q/U → ψ rotation → equidistant projection → Clough–Tocher interpolation → flat FFT → E/B | plane, 704 × 416, 9.6′ |
| **B** | pixell: CAR Q/U → × mask → curved-sky spin-2 `map2alm` → E/B | CAR 9.6′ (full sky 2250 × 1125, or a 1110 × 324 cut-out of the mask region) |
| **C** | intermediate: CAR patch rotated to the equator → flat FFT (`enmap.map2harm`) → E/B | CAR 708 × 397 |

- **Truth:** the exact E and B fields of the input a_ℓm, evaluated on each pipeline's pixels.
- **Comparison:** each pipeline against that truth, and against the same estimator applied to the masked true E/B (the "cut-sky reference", which separates the mask's effect from the pipeline's).
- **Measured:** E→B leakage, spectrum recovery, mode mixing (narrow-band inputs), and paired statistics over 50 realizations.

Main results (signal only, binary mask):

| | A (CMBpipeline) | B (pixell, sphere) |
|---|---|---|
| extra E→B leakage, 24 ≤ ℓ < 1113 | +9 % to +58 % over B (9–82 σ) | reference |
| mode-mixing width at ℓ₀ = 800 | 48.5 | 6.9 (the footprint alone) |
| EE vs the input realization, 24 ≤ ℓ < 773 | within ±11 % | within 3.5 % |
| time from a_ℓm to E/B maps | 9.0 s | 0.45 s |

- **Where A's extra leakage comes from:** pipeline C shows that more than half of it is caused by the flat FFT itself, not by the projection.
- **What no pipeline fixes:** the E/B mixing of the binary mask. It dominates every pipeline at ℓ < 50 and near the mask edge. The recovered BB is 2–3.7 × the input there, all of it from leakage; the genuine B part is recovered to within 0.92–1.03.
- **pixell details:**
  - pixell and healpy use the same Q/U convention (agreement to 10⁻¹²).
  - `map2alm` on a 9.6′ fejer1 grid is exact up to ℓ = 1124.
  - On a cut-out of the full-sky grid, `map2alm(..., method='2d')` is exact to 10⁻¹³; the default `'cyl'` is not.
- **Found in CMBpipeline:**
  - Its E/B have the opposite sign to healpy/pixell.
  - The FFT pixel size (9.553′ × 9.315′) differs from the 9.6′ grid.
  - `proj_2d.deprojection` fails on rectangular grids.
  - The plane mask area is overestimated by 5.6 %.

Details: `docs/report_task1.html`.

## Task 2: training dataset on pixell CAR maps

`src/make_dataset.py` and `src/run_dataset.py` build the training and validation sets for `DeepWiener_threechannels` (`../CMBpipeline/source/network_2d.py`) with pipeline B.

- **Image:** a CAR cut-out of the 9.6′ fejer1 full-sky grid around the SO mask, **nx × ny = 1120 × 320**.
  - Both sides are multiples of 32, as the network's 5 pooling levels require.
  - The border is at least 8 pixels, like CMBpipeline's margin.
  - The maps are never projected: curved-sky transforms stay exact on this cut-out.
- **Tensor:** `(N, 4, 320, 1120)`, float32, with CMBpipeline's channel order:
  - 0, 1: Q_obs, U_obs [μK] = mask × (signal + noise), 0 outside the mask
  - 2: binary mask
  - 3: pixel noise variance σ² [μK²], 0 outside the mask
- **Signal:** `curvedsky.rand_alm` from CMBpipeline's fiducial spectrum (r = 0.034, TE = 0), ℓ_max = 1535, 23′ beam applied to a_E and a_B, then `curvedsky.alm2map(spin=2)` on the cut-out. Seeds are explicit per map.
- **Noise:** CMBpipeline's inhomogeneous white-noise model from the SO hits map, drawn per CAR pixel with σ² = N(n̂) / Ω_pixel.
  - This gives the same noise per steradian as CMBpipeline at every position (2.43 μK-arcmin on average).
  - σ² is analytic, so no `make_variance_map.py` step is needed.
- **Normalization:** none in the file. The training loader subtracts the per-map mean and applies `map_rescale_factor`, as in CMBpipeline.
- **Configuration:** `src/config.dict` keeps CMBpipeline's values (1000 / 100 / 30 maps, noise level 5×10⁻⁷ μK² sr, 23′ beam, SO rect mask). Every difference is listed in `docs/03_task2_dataset_plan.md`.
- **Local check** (`notebooks/02_dataset_check.ipynb`):
  - shapes and channels are correct;
  - the noise matches CMBpipeline's to 10⁻⁷;
  - the analytic σ² matches 100 noise realizations;
  - the spectra are consistent with the theory and with a CMBpipeline map;
  - an untrained network runs a forward pass on the samples;
  - it takes 0.3 s per map, about 6 min for the full set; the training file is 5.7 GB.

Run (from `src/`):

```bash
python run_dataset.py                                     # uses src/config.dict (IAS cluster paths)
WF_CONFIG=/path/to/other_config.dict python run_dataset.py
```

**Status.** The full dataset was generated on the IAS cluster on 2026-10-07 (slurm job 17385111, commit `119a121`, `sbatch slurm/data.sh`, 9 min 50 s): `train_…pt` (1000, 4, 320, 1120) and `valid_…pt` (100, 4, 320, 1120) in `/data/bcostanza/curve-pixell/`, auxiliary files in `/home/bcostanza/wf-curve-pixell/aux/`. Training maps 0–2 are identical to the local test maps (same seeds). Code structure and run instructions: `docs/code_structure.html`. Training still needs the curved-sky counterparts of CMBpipeline's flat-sky loss and OQE steps, and possibly a new `map_rescale_factor`.

## Requirements

Python ≥ 3.10 with numpy, scipy, healpy, pixell (≥ 0.32, with ducc0), astropy, camb, torch, matplotlib and nbformat/jupyter. Also a local copy of [CMBpipeline](https://github.com/Belencostanza/CMBpipeline) next to this repository (`../CMBpipeline/source`), plus its dependencies (nifty7, optuna, torchvision) for the modules imported read-only.
