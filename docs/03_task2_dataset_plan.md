# Task 2 — Plan: training dataset with pixell maps (step 1)

**Goal.** Build the 2D training/validation dataset for `DeepWiener_threechannels` (`../CMBpipeline/source/network_2d.py`)
with pipeline B of Task 1:
- CAR cut-out maps on the sphere, generated with `pixell.curvedsky` and never projected onto a plane;
- SO "rect" configuration.

Plan only. Nothing is implemented yet.

---

## 1. What the network expects (from `network_2d.py` and `training_opt_beam_changed.py`)

| | CMBpipeline |
|---|---|
| input tensor | `(batch, 4, ny, nx)`, float32 |
| channel 0, 1 | Q_obs, U_obs [μK]: `mask · (signal + noise)`, 0 outside the mask. The linear branch reads `x[:, 0:2]` |
| channel 2 | binary mask (1 = observed). The mask branch reads `x[:, 2:3]` |
| channel 3 | pixel noise variance σ² [μK²], 0 outside the mask. The "inho" branch reads `x[:, 2:4]` = (mask, σ²) |
| output | 2 channels: Wiener-filtered Q, U on the same grid |
| map shape | 5 stride-2 average poolings followed by 5 upsamplings, so **ny and nx must be multiples of 32** (CMBpipeline: 416 × 704) |
| normalization | **none in the stored file.** The training loader (`create_dataloader`) subtracts the per-map mean of Q/U and multiplies by `map_rescale_factor` (0.8382 for "rect"). Mask and σ² are used as stored |
| files | `torch.save` of `X_train` `(nsims_train, 4, ny, nx)` and `X_valid` `(nsims_valid, 4, ny, nx)`, named `train_{dataset_tag}.pt` / `valid_{dataset_tag}.pt` in `data_folder` |

The new dataset will keep exactly this layout, channel order, dtype, units and file format. Only the geometry of the
image changes.

## 2. Geometry of the image

- The 9.6′ fejer1 full-sky CAR grid of pipeline B (2250 × 1125), cut to the rows and columns around the SO mask.
- The mask's bounding box is 1090 × 304 pixels (Dec rows 105–408, RA columns 533–1622).
- **Cut-out: nx × ny = 1120 × 320** (full-sky rows 97:417, columns 518:1638).
  - This is the smallest size divisible by 32 that contains the mask.
  - The border is ≥ 8 pixels (8 rows, 15 columns), the same margin as CMBpipeline's `fact = 8`.
  - The Task 1 display cut-out was 1110 × 324.
- Pixels: 9.6′ in Dec and 9.6′ × cos(Dec) in RA (2.6′–8.8′ over Dec −74.4°…−23.4°). Pixel areas are 25–85 arcmin²; the
  plane grid's are 78–92 arcmin².
- The cut-out is a slice of an SHT-compatible grid, so the curved-sky transforms of Task 1 §6b (`map2alm(method='2d')`)
  apply exactly to these images later on.

## 3. How each map is generated (`make_dataset.py`)

For map i, with seeds `cmb_seed0 + i` and `noise_seed0 + i`:

1. **Signal.**
   - Spectra: `utilities.signal_spectrum(r=0.034)` (CMBpipeline, CAMB 'total', TE = 0), cached to a file in `aux_folder`.
   - `curvedsky.rand_alm(ps, lmax=1535, seed)`, then the beam B_ℓ (23′) applied to a_E and a_B with `almxfl`.
   - `curvedsky.alm2map(spin=2)` **directly on the 1120 × 320 cut-out**, giving Q, U.
2. **Noise** (inhomogeneous, white, from the SO hits map; `noise_type = "inho"`):
   - The CMBpipeline model (`nhits_to_sigma2`) defines the noise per solid angle at each sky position:
     N(n̂) = `DEFAULT_NOISE_LEVEL` · (1/hits(n̂)) / ⟨1/hits⟩, with ⟨·⟩ the same mean over the HEALPix mask that CMBpipeline uses.
     `DEFAULT_NOISE_LEVEL` = 5×10⁻⁷ μK² sr, i.e. 2.43 μK-arcmin on average.
   - On CAR the pixel variance is **σ²(pixel) = N(n̂) / Ω_pixel(Dec)**, with hits read at the CAR pixel centre (nearest
     HEALPix pixel, as for the mask). The noise per steradian is thus identical to CMBpipeline's at every sky position.
     Only the pixel size differs.
   - Q and U noise are independent Gaussian draws with this variance, only inside the mask.
3. **Observed maps:** Q_obs = mask · (Q + n_Q), U_obs = mask · (U + n_U).
4. **Channels:** [Q_obs, U_obs, mask, σ²].
   - σ² is known analytically, because the noise is generated per CAR pixel. It replaces CMBpipeline's `inho2d`, which is
     the empirical variance of 100 projected noise realizations (`make_variance_map.py`, needed there because the
     interpolation changes the noise).
   - The local test (step 2) checks the analytic σ² against the variance of 100 noise realizations.

Train maps use indices 0…999 and validation maps 1000…1099, so the seeds never overlap.

## 4. Files (same structure and style as CMBpipeline)

```
src/
  config.dict        # same format as ../CMBpipeline/source/config.dict (only the keys this task uses + CAR keys)
  config_loader.py   # load_config(): parse the dict, resolve folders/paths, create output folders, derive file names
                     #   (same logic as CMBpipeline's config_loader, restricted to these keys)
  make_dataset.py    # class make_dataset (same name and method layout as CMBpipeline):
                     #   __init__(nsims_train, nsims_valid, nsims_test, smooth, apo, nside, fwhm, DEFAULT_NOISE_LEVEL,
                     #            lmax, res_arcmin, npixels_x, npixels_y)
                     #   car_geometry(mask_hp, hits_hp)   -> cut-out (shape, wcs), mask, hits on CAR
                     #   get_QUmaps_car(cls, beam, seed)  -> Q, U on the cut-out
                     #   get_inhomogenous_noise(...)      -> sigma^2 per CAR pixel
                     #   get_inho_noise(variance_map, mask, rng) -> one noise realization
                     #   make_inho_maps_car(...)          -> Q_obs, U_obs of one map
                     #   make_train_dataset(...)          -> X_train, X_valid (torch.save) + aux products
  run_dataset.py     # same as CMBpipeline's: load_config(), make_dataset(...), make_train_dataset(...), print time
slurm/
  data.sh            # step 3 only, same template as ../CMBpipeline/slurm/data.sh
```

- The SO mask and hits are read with CMBpipeline's own `projections.sph_mask.make_SO_mask`, imported read-only; the
  mask definition is therefore identical.
- The Task 1 code (`sky_input.py`, `pixell_tools.py`, ...) stays as it is.

**Outputs:**
- In `data_folder`:
  - `train_{dataset_tag}.pt`, `(1000, 4, 320, 1120)` float32, ≈ 5.7 GB (CMBpipeline: 4.7 GB)
  - `valid_{dataset_tag}.pt`, `(100, 4, 320, 1120)` float32, ≈ 0.57 GB
- In `aux_folder`, the fixed products with their WCS, needed later by the training, the OQE and to go back to the
  sphere:
  - `mask_car.fits`, `variance_car.fits`, `hits_car.fits` (`enmap.write_map`)
  - the spectra (`cls_signal_r0.034.npy`)
  - `seeds_{dataset_tag}.npz`
  - a copy of `config.dict`

## 5. `config.dict`: same values as CMBpipeline unless pipeline B requires a change

| key | CMBpipeline (SO) | new | why |
|---|---|---|---|
| `mask_type` | "rect" | "rect" | same |
| `nside` | 512 | 512 | used only to read the SO mask/hits (HEALPix) |
| `npixels_x`, `npixels_y` | 704, 416 | **1120, 320** | CAR cut-out, multiples of 32 (§2) |
| `fact`, `radius_deg` | 8, 40 | **removed** (margin set by the cut-out: ≥ 8 pixels) | no projection |
| `res_arcmin` | (9.6′ implicit in `proj_2d`) | **9.6** (new key) | CAR resolution, matched to the plane pitch |
| `lmax` (synthesis) | 3·nside − 1 = 1535 (synfast default) | **1535, explicit** (new key `lmax_sim`) | same value, now explicit |
| `r`, `lmin`, `lmax` (analysis) | 0.034, 4, 1000 | same | same |
| `cosmo_fid`, `binning` | … | same | same |
| `fwhm_arcmin` | 23.0 | 23.0 | same value. **Applied to a_E, a_B (spin-2 correct) instead of `hp.smoothing` on Q and U separately** |
| `noise_type` | "inho" | "inho" | same |
| `inho_type` | "inho2" (name of the 2D variance file) | **"so_hits"** | noise from the SO hits map, variance analytic on CAR (§3) |
| `DEFAULT_NOISE_LEVEL` | 5e-7 | 5e-7 | same noise per steradian |
| `smooth`, `apo` | True, False | True, False | same |
| `nsims_train`, `nsims_valid`, `nsims_test` | 1000, 100, 30 | 1000, 100, 30 | same (as in CMBpipeline, `make_train_dataset` writes train and valid only) |
| `map_rescale_factor` | 0.8382 (rect) | 0.8382 | not used to build the dataset; **may need retuning** for training (different pixel statistics) |
| seeds | none (global numpy state) | **`cmb_seed0 = 0`, `noise_seed0 = 100000`** (new keys) | reproducibility |
| `inho2d_file`, `apo_mask_file`, `fisher_modes_file`, … | … | not in this config | not needed to build the dataset |
| `root_folder` | `/home/bcostanza/wf-curve/pol/` | **`/home/bcostanza/wf-curve-pixell/`** | new project, see §6 |
| `data_folder` | `/data/bcostanza/curve/` | **`/data/bcostanza/curve-pixell/`** | new project, see §6 |
| `aux_folder` | `aux/` | `aux/` (under the new root) | |
| `dataset_tag` | `nCMB_r03_geo_beam_proj2d_eq_n512_maskSO_rad40_margin8_inhosim2_nofreq_res_arcmin` | **`nCMB_r034_beam_car9.6_cut1120x320_maskSO_inho_hitsSO`** | |
| `mask_files.so_hits` | local path | path of the hits file **on the cluster** (to be located in step 3) | |
| `cmbpipeline_source` | — | path of `CMBpipeline/source` (new key) | `signal_spectrum`, `make_SO_mask` are imported read-only |

## 6. Output location on the IAS cluster (`bcostanza@ssh.sns.ias.edu`)

- **Proposed:** `root_folder = /home/bcostanza/wf-curve-pixell/` and `data_folder = /data/bcostanza/curve-pixell/`.
  Both are new directories, separate from CMBpipeline's `/home/bcostanza/wf-curve/pol/` and `/data/bcostanza/curve/`,
  so nothing existing can be overwritten.
- `config_loader` only creates folders, and the new file names carry a new `dataset_tag`.
- In step 3, before any run, I will check over ssh that the new directories do not exist yet. I'll also check that the
  cluster's `torch_cmb` env has pixell, and where the SO hits file and `CMBpipeline/source` live there. The ssh itself
  needs your approval then.

## 7. Local test (step 2)

Generate 3 train + 2 valid maps locally, then check:
1. **Shapes and types:** `(3, 4, 320, 1120)` and `(2, 4, 320, 1120)`, float32; 320 and 1120 divisible by 32.
2. **Channels:**
   - Q/U and σ² are exactly 0 outside the mask;
   - the mask is binary, with 286,159 observed pixels;
   - σ² > 0 inside the mask.
3. **Against a CMBpipeline map** (generated locally with CMBpipeline's own functions, the same C_ℓ, beam and noise model):
   - Q/U signal RMS inside the mask;
   - noise level per solid angle (σ²·Ω vs CMBpipeline's σ²·Ω);
   - noise-only and signal E/B spectra of one map (curved-sky transform on the cut-out vs CMBpipeline's flat estimator).

   These should agree up to the known differences: spin-2 beam, and pixel size.
4. **Analytic vs empirical σ²:** the analytic σ² compared with the variance of 100 noise realizations.
5. **Network:** a forward pass of an untrained `DeepWiener_threechannels` on one sample, output `(1, 2, 320, 1120)`, plus
   the training loader's normalization (`create_dataloader` logic) applied to it.
6. **Time and memory** per map, to size the cluster job.

## 8. Not in this task

The training itself. Its loss (`losses.py`) and the OQE (`PowerSpectrum.py`) use flat-sky FFTs with dx, dy and ℓ = |k|,
and will need their curved-sky counterparts on the cut-out. `map_rescale_factor` may need retuning.
