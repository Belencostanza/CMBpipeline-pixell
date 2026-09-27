# 01 — How the current CMBpipeline goes from spherical Q/U to planar E/B

Task 1, Phase 1. Documentation only: no new code.

- **Code:** `../CMBpipeline/` at commit `68ce262`. All paths below are relative to `../CMBpipeline/source/` unless stated otherwise.
- **Config:** `config.dict` in its SO configuration:
  - `mask_type="rect"`, `nside=512`, `npixels_x=704`, `npixels_y=416`, `fact=8`
  - `fwhm_arcmin=23`, `noise_type="inho"`, `inho_type="inho2"`, `smooth=True`, `apo=False`
- **Paper:** `../CMBpipeline/paper_CMBpipeline.pdf` (draft, JCAP format). I cite it by section and equation number.
- **Scope:** only the SO path. The QUBIC/`circ` branches are skipped.
- The shared set-up (mask → projection → dx, dy → ψ) lives in `geometry.build_geometry` (`geometry.py:44-114`).
  Every script (`make_true_maps.py`, `compute_fisher.py`, `compute_bias.py`, `make_variance_map.py`, training) calls it.

### Measured geometry of the SO patch

I ran the pipeline's own functions read-only (a throwaway script outside both repos) with the local SO hits file
`norm_nHits_SA_35FOV_ns512.fits`:

| quantity | value |
|---|---|
| patch centre (lon, lat) | (6.23°, −47.46°) (the pixel with the most hits) |
| observed pixels / f_sky | 349 440 / 0.111 |
| extent of the projected patch, Lx × Ly | 109.4° × 61.9° |
| largest angular distance from the centre | **56.2°** |
| grid (nbins_x, nbins_y) | 704 × 416 (matches `config.dict`) |
| fraction of grid pixels inside mask2d | 0.646 |
| grid pitch actually used to build the grid | 9.6′ × 9.6′ |
| dx, dy passed to the FFTs | **9.553′ × 9.315′** |
| range of the Q/U rotation angle ψ | −75.7° … +78.3° |

The patch is far outside the flat-sky regime: pixels lie up to 56° from the tangent point. Keep this in mind for every
approximation discussed below.

---

## 1. Generating the spherical Q/U maps

| step | code | details |
|---|---|---|
| Theory C_ℓ | `utilities.signal_spectrum` (`utilities.py:187-227`); `utilities.components_spectrum_cosmo` (`utilities.py:29-76`); `spectra.fiducial_spectrum` / `spectra.true_spectrum` (`spectra.py`) | CAMB, `powers['total']` (lensed scalar + tensor) in μK² with `CMB_unit='muK'`. The CAMB output is D_ℓ, which the code divides by ℓ(ℓ+1)/2π for ℓ≥2. `pars.set_for_lmax(7000)`. Fiducial r = 0.034 (tensor tilt from CAMB's consistency relation). |
| Spectrum array | `make_dataset._create_spectrum_array` (`make_dataset.py:107-126`) | order (TT, EE, BB, TE, TB, EB), with **TE = TB = EB = 0** |
| Map synthesis | `utilities.make_maps` (`utilities.py:470-491`) → `hp.synfast(cls, nside, new=True, pol=True)` | healpy defaults: `lmax = 3·nside−1 = 1535`, **`pixwin=False`** (no HEALPix pixel window). A seed, if given, is set through the global `np.random` state and then restored. `config.dict` has `cmb_seed=None`, so the default runs are not reproducible. |
| Beam (training data / "true" sims) | `make_dataset.get_QUmaps` / `get_QUmaps_cl` (`make_dataset.py:149-152`, `174-177`) | `hp.smoothing(qmap, fwhm)` and `hp.smoothing(umap, fwhm)` are called **separately, as two scalar (spin-0) maps** |
| Beam (Fisher / noise-bias sims) | `PowerSpectrum.generate_one_inho_pln` / `generate_inho_pln` (`PowerSpectrum.py:605-647`, `836-890`) | `hp.synalm(C_ℓ, lmax=1600)` (`compute_fisher.py:91-92`), then `hp.almxfl(alm, utilities.bl)` (Gaussian `exp(−ℓ(ℓ+1)σ²/2)`, `utilities.py:229-242`), then `hp.alm2map_spin([E,B], nside=512, spin=2, lmax=1600)` |
| Noise (`inho`) | `make_dataset.get_inhomogenous_noise` (`make_dataset.py:198-227`), `utilities.nhits_to_sigma2` / `sigma2_to_map` (`utilities.py:430-451`) | white noise, σ²_pix ∝ 1/N_hits normalized to mean `DEFAULT_NOISE_LEVEL / Ω_pix`, drawn only inside the mask. Q and U noise are independent. |
| Observed data | e.g. `make_dataset.make_maps_inho_pln_cls` (`make_dataset.py:478-510`) | `dataQ = mask·(Q_sky + n_Q)`, and likewise for U |

**Q/U convention:** healpy/HEALPix **COSMO** convention: Q and U are defined in the local (e_θ, e_φ) basis.
Nothing in the code converts to IAU.

## 2. Selecting the observed sky region (mask, apodization)

- **Spherical mask:** `sph_mask.make_SO_mask` (`projections.py:263-287`).
  - It reads the SO hits map (`read_SO`, `projections.py:100-111`) and takes the max-hits pixel as the centre.
  - It keeps the pixels that are **both** within 100° of the centre **and** have `hits > 0.25` (`projections.py:274-276`).
  - The result is a binary mask. `radius_deg` (40) is **not used** for `rect`.
  - It also returns `valid_index` (the observed pixels) and their unit vectors `vecs`.
- **Apodization:** none. `apo=False` (`config.dict:81`). The mask is binary on the sphere and binary on the plane.
  An apodized mask is only used by NaMaster (3° scale, `config.dict:188-191`), which runs directly on the sphere.
- **Planar mask:** `proj_2d.define_region_mask_from_res_margins` (`projections.py:532-580`).
  - It builds the planar grid (§4) and maps every grid point back to the sphere with the inverse equidistant projection
    (`projections.py:569-577`).
  - It then reads the spherical mask at the **nearest HEALPix pixel** (`hp.ang2pix`, `projections.py:578`).
  - `mask2d` is therefore binary, and its boundary is pixelated at HEALPix resolution.

## 3. Projection onto the plane

**Type:** azimuthal **equidistant** projection, r(θ) = θ, implemented in `proj_2d.proj_conventions` (`projections.py:321-361`).
The same function also returns gnomonic (`tan θ`) and stereographic (`2 tan(θ/2)`) coordinates. Those are unused: only
`z_eq` is used downstream (`geometry.py:86`).

- **Tangent frame at the centre n̂** (`projections.py:325-333`):
  - `north` = the component of the celestial pole ẑ perpendicular to n̂, normalized.
  - `west` = n̂ × north.
- **Coordinates** (`projections.py:335-349`):
  - θ = arccos(v·n̂)
  - φ = atan2(v_⊥·west, v_⊥·north), i.e. measured from north towards west
  - **x = θ sin φ** (pointing west), **y = θ cos φ** (pointing north)
- This matches paper §2.3 (Eq. 2.26).
- As seen from outside the sphere, (x = west, y = north, n̂) is a left-handed frame. Put another way, the plane is the sky
  as seen from inside, with east to the left.

### Q/U rotation (done on the sphere, before projecting)

- `rotate_geo.psi_Q_to_P` (`projections.py:917-939`):
  - For each pixel Q and the centre P it builds m = −e_θ − i e_φ (i.e. north + i·(−east)).
  - It returns **ψ = arg(m_Q · m̄_P)**, the phase of a 3-D dot product between the complex bases at the two points.
- `geometry.py:95-100` computes ψ once per geometry (longitudes wrapped to [−π, π)).
- `make_dataset.make_rot` (`make_dataset.py:312-329`) and the duplicate `PowerSpectrum.make_rot` (`PowerSpectrum.py:564-581`) apply:
  - Q' = Q cos2ψ + U sin2ψ
  - U' = −Q sin2ψ + U cos2ψ
  - that is, (Q+iU)' = e^{−2iψ}(Q+iU), which is paper Eq. 3.1.
- m = −(e_θ + i e_φ) differs from the healpy basis only by an overall sign. That sign is squared away in spin-2, so this
  step re-expresses the healpy (e_θ, e_φ) frame at each pixel in the (e_θ, e_φ) frame of the centre.
- Caveats. These are not tested in the code or in the paper for polarization:
  1. arg(m_Q·m̄_P) is a 3-D projection of one frame onto the other. It is not obviously identical to parallel transport
     along the geodesic, nor to the rotation induced by the projection's Jacobian. The measured ψ reaches ±78° at the edge
     of the patch.
  2. The equidistant map is **not conformal**. At distance θ from the centre, tangential lengths are stretched by
     θ/sin θ ≈ 1.18 at θ = 56°, while radial lengths are unchanged. A pure rotation of Q/U cannot represent this anisotropic
     distortion.

### Unused / inconsistent helpers (not on the SO path)

- `proj_2d.polar_coord`, `get_equidistant`, `get_cart` and `inv_projection` (`projections.py:304-319`, `375-434`) use a
  **different** tangent basis, u = n̂ × ẑ.
- `get_gnomonic` uses healpy's `GnomonicProj`.
- `rotate.psi_Q_to_P` (`projections.py:942-1002`) and `utilities.rotate_QU_gnomonic` (`utilities.py:1494-1556`) are
  older rotation variants.
- None of these are called by the SO pipeline scripts.

## 4. Interpolation onto the 2D grid

Function: `proj_2d.grid_bins_mask_from_res_margin` (`projections.py:751-784`). It is called for Q and U in
`make_dataset.py:502-503` and `PowerSpectrum.py:632-633`/`871-872`, and for the noise variance in `make_variance_map.py:46`.

- **Method:** `scipy.interpolate.griddata(points=z_eq, values, xi=grid, method='cubic', fill_value=0)`.
  - For 2-D scattered data, `method='cubic'` is the **Clough–Tocher piecewise-cubic C¹ interpolant on a Delaunay
    triangulation** of the HEALPix pixel centres.
  - It is not a tensor-product spline.
- **Pixel size:** `pix = reso_arcmin = 9.6′`. This is the `proj_2d` default (`projections.py:293`). `geometry.py:85` does
  not override it, and it is not in `config.dict`.
- **Grid size:**
  - The margin is `fact·pix = 8·9.6′ = 76.8′` on each side.
  - `n = ceil((L + 2·margin)/pix)`, rounded **up to a multiple of 32** (`projections.py:766-770`). This gives 704 × 416.
  - Grid coordinates: `x = x_min − margin + pix·arange(n)` (`projections.py:776-777`), laid out as rows = y, columns = x
    (`np.meshgrid`, shape (ny, nx)).
  - Because of the round-up, the grid extends past `x_max + margin` on the high side only, so the patch is not centred in
    the image.
- **Mask:** grid values where `mask2d == 0` are set to 0 (`projections.py:782`). The 0 fill (`fill_value=0`) and the mask
  cut create a hard edge.
- **Network input** (the dataset tensor):
  - channels = [Q, U, mask2d, σ²_pln]
  - σ²_pln = the pixel variance of 100 projected noise-only realizations (`make_variance_map.py:38-50`)
- **Preprocessing before the network:** the per-map mean over the *whole* grid (zeros included) is subtracted, and the map
  is multiplied by `map_rescale_factor` = 0.8382 (`training_opt_beam_changed.py:94-95`; `PowerSpectrum.py:257-258`). The
  network output is divided by the same factor (`PowerSpectrum.py:275`).

## 5. Fourier transforms (normalization, ell mapping)

### Pixel size used in Fourier space

`geometry.py:89-90`:

```
dx1, dy1 = calculate_res(z_eq, nx, ny)                      # L/n            (utilities.py:1107-1113)
dx, dy   = calculate_res_margin(z_eq, dx1, dy1, nx, ny, 8)  # (L + 2·8·max(dx1,dy1))/(n−1)  (utilities.py:1115-1122)
```

- The result is dx = 9.553′ and dy = 9.315′. These are **not** the 9.6′ pitch used to build the grid in §4.
- Every flat-sky ℓ assigned to a Fourier mode is therefore rescaled by 9.6/9.553 = 1.005 in x and 9.6/9.315 = 1.031 in y,
  so it is anisotropic.
- See §8.

### Normalization

- `tfac = sqrt(dx·dy/(nx·ny))`.
- Forward transforms:
  - `rfft2(map)·tfac` in the loss (`losses.py:41-44`, `82-84`) and in `transf_eb2[_np]`
  - `fft2(map)·tfac` in the OQE (`PowerSpectrum.power`, `PowerSpectrum.py:221-228`)
- Inverse: `irfft2(·)/tfac` (`PowerSpectrum.py:287-288`).
- With this normalization, `|FFT·tfac|²` is the flat-sky power spectrum estimate for a *periodic, unmasked* map
  (units μK²·sr). No f_sky or mask correction is applied; the Fisher matrix absorbs the mask effect in the OQE.
- `irfft2` is called without `s=`. That is fine only because nx = 704 is even.

### ℓ mapping

**ℓ = |k|**, with k = 2π·`fftfreq(n, d)`.

- **OQE** (`PowerSpectrum.flat_spectrum_xy`, `PowerSpectrum.py:104-116`):
  - full `fft2` grid, with (lx, ly) from dx, dy
  - C_ℓ interpolated **linearly in ℓ** with `interp1d(ℓ[2:], C_ℓ[2:])`
  - modes outside [2, ℓ_max] filled with `min(C_ℓ)`
- **Training loss** (`utilities.power_spectrum_flat`, `utilities.py:1075-1093`, called in `training_opt_beam_changed.py:66-67`):
  - rfft grid
  - C_ℓ interpolated **linearly in log ℓ–log C_ℓ**, with C_0 = C_1 = C_2
- The two interpolants are different.
- Estimator Eq. 2.23: `PowerSpectrum.El_fid` (`PowerSpectrum.py:293-320`) computes
  E_b = ½ Σ_{k∈b} |ŝ_k|² / C_pln(k) / C_b.
  - The bins come from `utilities.compute_bins_fractional` (`utilities.py:1301-1315`; `frac=0.2`, `min_width=20`,
    ℓ ∈ [4, 1000]).
  - C_b is the unweighted mean of C_ℓ over ℓ in the bin (`utilities.bineado`, `utilities.py:1348-1370`).

### What the Fourier step does *not* do

- **Boundaries:** periodic boundary conditions are implicit, since both FFTs treat the rectangle as a torus. There is no
  planar apodization (`utilities.apodize_map`, `utilities.py:257-263`, is not called).
- **Pixel windows:** no pixel window is corrected, neither the planar one (`utilities.compute_pixel_window`,
  `utilities.py:1150-1167`, is unused) nor the HEALPix one. The smoothing from the interpolation is not modelled.
- **Beam in the loss:** the planar beam is `utilities.beam_flat` = exp(−k²σ²/2) (`utilities.py:244-254`), evaluated at k = |ℓ|.
  On the sphere the beam is exp(−ℓ(ℓ+1)σ²/2). That is a small but systematic mismatch at low ℓ.

## 6. Q/U → E/B

Function: `utilities.transf_eb2_np` (`utilities.py:564-594`), called from `PowerSpectrum.get_nn_outputs`
(`PowerSpectrum.py:283-290`). Its torch twin `utilities.transf_eb2` (`utilities.py:515-547`) is called in
`losses.fourier_loss` (`losses.py:203-238`). Both use the default `yes=False`:

```
lx = 2π·rfftfreq(nx, dx)      (columns = x = west)
ly = 2π·fftfreq(ny, dy)       (rows    = y = north)
2φ = 2·atan2(ly, −lx)
E  =  cos2φ·Q + sin2φ·U
B  = −sin2φ·Q + cos2φ·U
```

- The E/B Fourier maps go back to real space with `irfft2/tfac`. The OQE then takes `fft2` power of those real maps.
- **Sign / handedness.** This part is an analytic derivation, **not yet verified numerically**. Phase 3 should check it
  against alm-based truth.
  - After the rotation in §3, Q > 0 means polarization along the centre's e_θ, i.e. along the planar **y** axis.
    U > 0 means along the (x + y) diagonal.
  - In the standard flat-sky convention Q > 0 lies along x and φ_ℓ = atan2(ℓ_y, ℓ_x). Relative to that, the code's
    `atan2(ly, −lx)` (equivalently x → −x) gives E_code = −E_std and B_code = −B_std. The "std" frame here is the one
    where Q > 0 lies along y.
  - If this is right, the overall sign has **no effect on power spectra or on the OQE**. It does matter for map-level
    comparisons with healpy/pixell E/B maps, and it has to be checked together with the handedness of the
    (west, north) frame. The healpy COSMO vs IAU convention (sign of U) enters the same check.
- **`utilities.transf_qu_eb`** (`utilities.py:597-615`), the E/B → Q/U inverse, uses a different angle,
  `2·atan2(lx, −ly)`.
  - A numerical round trip `transf_eb2_np` → `transf_qu_eb` does **not** recover Q, U: the maximum error is O(1) for
    unit-variance white noise.
  - It is not the inverse of the forward transform. It is not called by any script or notebook.
- Other E/B helpers (`transf_eb`, `transf_eb_np`, `QU_to_EB_fft`, the `eth2_*` / `spin2_from_potentials` potential-based
  operators, `utilities.py:501-513`, `549-562`, `617-1072`) are not used on the current SO path.

## 7. Approximations and how the paper justifies them

| # | Approximation | Where in code / paper | Paper's justification | Comment |
|---|---|---|---|---|
| A1 | The projection is invertible: **PP† = P†P ≈ I**, so the planar WF is the projected spherical WF | paper Eqs. 2.4-2.7, §3.2 | App. A: WF on the plane vs on the sphere, ρ_ℓ ≈ 1 for equidistant (Fig. 11), with gnomonic worst | The test is **temperature only**, no mask, homogeneous noise, and a **55.8 deg²** region. The SO patch is ~109°×62°, with pixels up to 56° from the centre. Polarization (the ψ rotation) is untested. |
| A2 | **Planar Fourier modes ≈ multipoles** (ℓ ↔ \|k\|), so S_pln is diagonal in k and Eq. 2.23 holds | §2.2 (Eq. 2.22 → 2.23), §4 loss prior | App. B: single-ℓ₀ maps projected. The 16–84% width of the k-distribution is 7.5, 11.0, 17.4 and 23.8 at ℓ₀ = 200, 400, 600, 800 (Table 5), with a shift to lower k. Bins must be wider than this. | Bins from `compute_bins_fractional` have Δℓ ≥ 20 and grow as 0.2ℓ. The *shift* of the kernel (a bias) is not corrected; it is absorbed only through the simulation-based Fisher matrix and noise bias. |
| A3 | Noise covariance diagonal in planar pixels (σ²_i from 100 projected sims) | §4, `make_variance_map.py` | "The interpolation does not introduce additional correlations in the scales of interest" | Asserted, not shown. Cubic interpolation of white noise does correlate neighbouring pixels. |
| A4 | A single frame rotation ψ makes Q/U planar spin-2 fields | §3.2, Eq. 3.1 | None beyond the derivation | The equidistant map is not conformal (θ/sinθ up to 1.18). ψ is not proven to be the geometrically correct angle (§3). |
| A5 | FFT on a periodic torus, applied to a masked, zero-filled, non-periodic rectangle | loss, OQE | App. A mentions boundary effects qualitatively | Causes E→B leakage at mask edges. The paper notes the leakage in the observed BB (§3.2, Fig. 2c); the OQE relies on the Fisher matrix to absorb it. |
| A6 | Interpolation smoothing is negligible (9.6′ pixel vs 23′ beam, HEALPix 6.9′) | §3.2 | "sufficiently sampled, minimizes interpolation-induced artifacts" | No transfer function measured; no pixel window in the model. |
| A7 | Flat beam exp(−k²σ²/2), ℓ = \|k\|, C_ℓ interpolation onto the k grid | `losses.py`, `PowerSpectrum.py` | Not discussed | Two different interpolants (§5). |
| A8 | dx, dy used in Fourier space ≈ the true grid pitch | `geometry.py:89-90` | Paper states "9.5′–9.6′" | Actually 9.55′/9.32′ vs a true pitch of 9.6′ (§8). |

The paper also notes (§2.3, Conclusions) that the patch "extends beyond the strict validity of the flat-sky
approximation". It argues that the equidistant projection has the smallest distortions among the three it tests.

## 8. Code vs paper: differences and open issues

1. **Interpolation method.** Paper §3.2 says "cubic-spline interpolation implemented with SciPy". The code uses
   `griddata(method='cubic')`, which is Clough–Tocher on a Delaunay triangulation (§4).
2. **Pixel size.** The paper says "pixel resolutions between approximately 9.5′ and 9.6′ along x and y".
   - The grid is built with exactly 9.6′ × 9.6′ (`projections.py:763`).
   - The FFTs, the ℓ grid and the beam use dx = 9.553′ and dy = 9.315′ (`geometry.py:89-90`; measured).
   - So every Fourier mode's ℓ is underestimated by ~0.5% (x) and ~3% (y). The y error is ~3% in ℓ, e.g. Δℓ ≈ 15 at ℓ = 500.
   - This looks like a leftover of the older `define_region_mask_nbins_margins` / `grid_bins_mask_margin` grid, which used
     `linspace` with n points and did match `calculate_res_margin`.
3. **ℓ ↔ k relation.** Paper Eq. B.1 writes ℓ(ℓ+1) ↔ |k|². The code uses ℓ = |k| everywhere.
4. **Grid-size consistency.** `config.dict` hard-codes `npixels_x/y = 704/416`. The grid itself is produced independently
   by the round-up rule (§4). They match for this hits map, but nothing enforces it.
5. **Beam and ℓ_max for the Fisher / bias sims vs the training / "true" sims.**
   - Training and "true" sims: `hp.synfast` (ℓ_max 1535), then `hp.smoothing` applied to Q and U separately as scalars.
   - Fisher and noise-bias sims: `synalm` with ℓ_max = 1600 (above 3·nside−1 = 1535, so aliasing is possible), then
     `almxfl` on E/B and a spin-2 `alm2map_spin`.
   - These are two different forward models for "the same" observation. Their equivalence is not documented.
6. **"True" spectrum.** The paper (Table 2) uses a "cosmo" +1σ model with r_true = 0.025. The current `config.dict` uses
   `spectrum_model="alens"`, r_new = 0.028, A_L = 0.975 (`config.dict:154-168`). This is a config choice rather than a bug,
   but the configurations differ.
7. **Loss vs OQE use different C_pln(k).** Log-log interpolation in training, linear in the OQE (§5).
8. **Paper App. A** validates PP† ≈ I with temperature maps on a 55.8 deg² region, then uses that result to justify
   polarization on a ~4600 deg² (f_sky = 0.11) patch.
9. **Repository name.** The paper footnote gives `github.com/Belencostanza/CMBtorch`; the README uses `CMBpipeline`. The
   paper is still a draft (it contains Spanish notes to the authors and "Figure ??" references).
10. **Unresolved, to test in Phases 2–3:**
    - (a) the overall E/B sign / handedness of the planar frame (§6)
    - (b) whether ψ = arg(m_Q·m̄_P) matches parallel transport or the projection Jacobian (§3)
    - (c) the size of the E→B leakage from the periodic FFT on the masked patch
    - (d) the effective transfer function of the Clough–Tocher interpolation

## Notes for later phases

- `pixell` is **not installed** in the `torch_cmb` conda env. That env does have healpy, camb, torch, nifty7, astropy and
  pymaster. pixell needs to be installed before Phase 3.
- The CMBpipeline geometry functions run on the local SO hits file, so pipeline A (the current projection + FFT) can be
  reproduced without the missing products (trained network, Fisher, bias). Those products are only needed for stages
  outside Task 1.
