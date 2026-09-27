# 02 — Test plan (revision 4)

Task 1, Phase 2, following the sections of the updated `TASK1.md` (2.1–2.5). It replaces revisions 1–3.

Changes from revision 3, following the latest `TASK1.md`:
- Pipeline B's power spectrum is **not** a pseudo-C_ℓ of re-masked E/B maps. It is computed directly from the
  spin-2 analysis of the Q/U maps multiplied by the **2D CAR mask**, normalized by that mask's w₂ (§2.3 B, §2.4).
  A and C use the same form: no re-masking of E/B.
- The pixel counts (nx × ny) of A and B are stated explicitly (§2.2, §2.5).

**Objective (TASK1).** Same input a_ℓm^{E,B}, same physical SO coverage. Quantify the *additional* distortion, mode
mixing and E/B leakage that the current projection + interpolation + flat-sky representation (CMBpipeline) introduces,
relative to a CAR representation treated with curved-sky operators (pixell).

**Environment.** `torch_cmb` conda env: pixell 0.32.7, ducc0 0.38.0, healpy 1.18.1, numpy 2.2.5. CMBpipeline
(`../CMBpipeline/source`) is imported read-only. pixell line numbers refer to the installed version.

**What earlier runs already established** (Phase 3 preliminary runs, used here as design input):
- pixell and healpy synthesize identical Q/U: the same polarization convention, to 10⁻¹².
- CMBpipeline's flat E/B has the opposite overall sign (correlation with the truth −0.996).
- CMBpipeline's `proj_2d.deprojection` fails on the rectangular 704 × 416 grid.

---

## 2.1 Common input (reference sky)

**Reference sky.** The band-limited, continuous spherical field defined by one set of a_ℓm:

- a_ℓm^{T,E,B} drawn with `pixell.curvedsky.rand_alm(ps, lmax=1535, seed=S)` from CMBpipeline's fiducial theory
  spectrum `utilities.signal_spectrum(r=0.034)` (CAMB, lensed scalar + tensor, μK²), with TE = TB = EB = 0 as in
  `make_dataset._create_spectrum_array`.
- **lmax = 1535** (= 3·nside − 1), the band limit of CMBpipeline's `hp.synfast`. The field has no power above it.
- **Beam.** One 23′ FWHM Gaussian, applied **once, in harmonic space**, to the input alm:
  `almxfl(a_ℓm, B_ℓ)` with B_ℓ = `hp.gauss_beam(23′)` = exp(−ℓ(ℓ+1)σ²/2), σ = 9.77′ (identical to CMBpipeline
  `utilities.bl`). Both pipelines receive the same beamed alm.
  - CMBpipeline's training sims instead smooth Q and U as scalar maps (`hp.smoothing`). That is not used here, because
    it is not the same operation on a spin-2 field (Phase 1 §8.5).
- **No pixel window** is applied to the input. Every synthesis below *samples* the band-limited field at pixel
  centres (healpy `alm2map` without `pixwin`; pixell `alm2map`). HEALPix and CAR are two samplings of the same field.

**Pixelization and effective resolution of each representation.** These are documented rather than assuming an
identical pixel window:

| representation | used by | sampling | pixel area | band limit it can represent |
|---|---|---|---|---|
| HEALPix nside 512 | A (input of the projection) | equal area, ~6.87′ spacing | 47.2 arcmin² everywhere | synthesis exact at the pixel centres; ℓ ≤ 1535 by convention |
| CMBpipeline plane, 704 × 416 | A (after interpolation) | 9.6′ × 9.6′ on the plane | 92.2 arcmin² nominal; on the sphere 92.2·sinθ/θ = 78–92 arcmin² (θ ≤ 56°) | ℓ_N = π/9.6′ = 1125 along x and y, plus the Clough–Tocher smoothing (transfer ≈ 0.986 at ℓ = 800, 0.978 at ℓ = 1000, measured) |
| CAR fejer1, 9.6′ | B, C | 9.6′ in Dec; 9.6′·cos(Dec) in RA = 8.7′ (Dec −25°) to 2.8′ (Dec −73°) | 92.2·cos(Dec) = 27–84 arcmin² | ℓ_N = 1125 in Dec, higher in RA. `map2alm` quadrature exact up to ℓ ≈ n_rings − 1 = 1124 |

The beam is common to all of them. At ℓ = 1125, B_ℓ = 6.0 × 10⁻³ and B_ℓ² = 3.6 × 10⁻⁵. The beamed EE/BB variance
above ℓ = 1125 is ~6 × 10⁻⁶ of the total. So the 9.6′ representations lose or alias a negligible fraction of the
signal. This is checked directly by a 6′ CAR control run (§2.5).

**Full-sky reference spectrum.** C_ℓ^{ref} = `curvedsky.alm2cl` of the beamed input a_E, a_B, i.e. the realization's own
spectrum before any coverage. The theory C_ℓB_ℓ² is also kept for the ensemble.

## 2.2 Common coverage

**Physical region.** The SO region as CMBpipeline defines it: `sph_mask.make_SO_mask` on the SO SAT hits map
(`norm_nHits_SA_35FOV_ns512.fits`, nside 512). It keeps pixels with hits > 0.25 within 100° of the maximum-hits pixel,
centred at (RA, Dec) = (6.23°, −47.46°), f_sky = 0.1111, 4582.5 deg².

**The mask is defined on the sphere** as this binary HEALPix map. It is transferred to each geometry with one rule:
**each pixel centre takes the value of the HEALPix pixel that contains it** (`hp.ang2pix`). This is the rule
CMBpipeline already uses for its planar mask (`define_region_mask_from_res_margins`, `projections.py:578`).

| | CMBpipeline plane (A) | pixell CAR (B, C) |
|---|---|---|
| how the region is represented | pixel centres inverse-projected (equidistant) onto the sphere → `mask2d` (704 × 416) | CAR pixel centres → HEALPix mask |
| data inside | Clough–Tocher interpolation of the rotated Q/U of the observed HEALPix pixels; 0 outside `mask2d` | exact Q/U samples of the reference sky |
| map size (nx × ny) | **704 × 416** = 292,864 pixels | **2250 × 1125** (RA × Dec, full sky, 9.6′). The patch sits in a 1090 × 304 box |
| observed pixels | 189,097 plane pixels in `mask2d` (from 349,440 HEALPix pixels) | 286,159 CAR pixels |
| area | 4841 deg² if the plane pixels are counted as flat 9.6′ squares; 4583 deg² with the true solid angle | 4583 deg² (`pixsizemap`) |
| boundary | staircase of 9.6′ plane pixels. Their physical size varies with the tangential stretch θ/sinθ (up to 1.18 at the edge), and the edge is non-periodic inside a periodic FFT box | staircase of 9.6′ (Dec) × 2.8′–8.7′ (RA) pixels; no periodicity assumed in B |
| geometric distortion | equidistant: radial distances exact, tangential stretched ≤ 18%; the Q/U frame is rotated by ψ (non-conformal, 1.4° RMS / 4.3° max away from the projection-implied frame) | none in B: CAR is only the sampling of the sphere |

**Apodization.** The primary mask is **binary**, as in CMBpipeline (`apo = False`). An apodized variant is kept as an
*optional* extension, not run by default, because you asked earlier to drop it. If it is enabled:
- it is defined on the sphere as a 3° cosine taper (the NaMaster scale for SO, `config.dict:188-191`), computed from the
  angular distance to the HEALPix mask edge;
- it is transferred by evaluating it at each geometry's pixel centres. It is smooth, so interpolation is benign.

## 2.3 Pipelines

All three pipelines start from the same beamed a_ℓm of §2.1 and use the mask of §2.2.

### A. Current CMBpipeline

a_ℓm → spherical Q/U → projection + interpolation → planar Q/U → flat FFT → E/B → flat power spectrum.

| step | CMBpipeline function (read-only) |
|---|---|
| a_ℓm → Q/U on HEALPix | `hp.alm2map(pol=True, lmax=1535)`, as `synfast` inside `utilities.make_maps` |
| mask, Q/U frame rotation | `make_SO_mask`; `rotate_geo.psi_Q_to_P` + `make_dataset.make_rot` |
| equidistant projection | `proj_2d.proj_conventions` |
| interpolation onto 704 × 416, 9.6′ | `proj_2d.grid_bins_mask_from_res_margin` (`griddata(method='cubic')`) |
| flat FFT, Q/U → E/B | `utilities.transf_eb2_np` with dx, dy from `calculate_res_margin` (as `geometry.py:89-90`), then `irfft2/tfac` (as `PowerSpectrum.get_nn_outputs`) |
| flat power spectrum | `PowerSpectrum.power` (`fft2 · tfac`, \|·\|²) of the E and B maps, with ℓ = \|k\| from the grid of `PowerSpectrum.flat_spectrum_xy`, averaged over the modes in each bin (as `El_fid`), divided by w₂ = ⟨mask2d²⟩ |

- As in CMBpipeline, the E/B maps come from the masked Q/U grid and are **not** masked again before the FFT.
- The division by w₂ is the only addition. CMBpipeline's OQE absorbs the mask through the Fisher matrix instead, and
  both pipelines need the same normalization convention to be compared.
- A's overall E/B sign is −1 relative to the healpy/pixell convention (Phase 1 §6; measured). It is corrected before any
  map-level comparison, and it has no effect on spectra.

### B. pixell: CAR + curved sky

a_ℓm → CAR Q/U → masked spin-2 analysis → E/B → curved-sky power spectrum.

| step | pixell function | why it is the appropriate operation (from the source) |
|---|---|---|
| geometry | `enmap.fullsky_geometry(res=9.6′, variant="fejer1")` (`enmap.py:1713-1740`) | π/9.6′ = 1125 rings exactly, and the function asserts that the resolution divides the sphere. Fejer1 is one of the ring layouts ducc has exact quadrature for (`quad_weights`, `curvedsky.py:492-509`). The patch is handled on this full-sky grid; `map2alm` also pads compatible partial geometries (`curvedsky.py:209-303`, method "2d") |
| a_ℓm → Q/U | `curvedsky.alm2map(alm, map, spin=[0,2])` (`curvedsky.py:83`) | exact spin-2 synthesis at the pixel centres. Q/U are defined in the local (θ, φ) basis with the HEALPix sign convention (healpy's COSMO). Numerically identical to `hp.alm2map` (10⁻¹²) |
| mask | multiply by the transferred mask | cut sky |
| masked Q/U → E/B harmonics | `curvedsky.map2alm(map, lmax=1124, spin=[0,2])` | spin-2 analysis with the fejer1 quadrature weights: the adjoint-exact inverse of `alm2map` for fields band-limited to ℓ ≤ 1124. It returns the a_E, a_B of the masked (cut-sky) Q/U. The ℓ > 1124 part of the input is < 10⁻⁵ of the variance (§2.1); the 6′ control quantifies it |
| E/B maps (for map-level inspection) | `curvedsky.alm2map(a_EB, spin=0)` | E and B are scalar fields |
| power spectrum | `curvedsky.alm2cl(a_E)`, `alm2cl(a_B)` (`curvedsky.py:672`) of the a_E, a_B returned by the spin-2 `map2alm` of the Q/U multiplied by the **2D CAR mask** → bin → / w₂ of that 2D mask (area-weighted: Σ mask² · `pixsizemap` / 4π) | the curved-sky counterpart of A's estimator. A takes \|E(k)\|² of the flat transform of the masked Q/U grid; B takes \|a_E,ℓm\|² of the exact spherical transform of the masked CAR Q/U. Both divide by the w₂ of their own 2D mask. There is no re-masking of E/B maps and no mode-coupling deconvolution (no MASTER/NaMaster) |

Polarization basis:
- pixell and healpy define Q/U in the same local basis and sign convention.
- `curvedsky` handles the spin-2 transform exactly, so no frame rotation is needed in B.
- CMBpipeline instead rotates Q/U to the patch-centre frame (ψ) before projecting. That is an extra, approximate step
  that B does not have.

### C. Intermediate (optional, recommended): CAR + flat FFT

a_ℓm → CAR Q/U → flat FFT (enmap) → E/B → flat power spectrum.

1. **Rotate the sky** so the patch centre goes to (RA, Dec) = (0, 0) with local north along +Dec:
   `curvedsky.rotate_alm(alm, psi=−λ_c, theta=δ_c, phi=0)` (`curvedsky.py:717-742`, ducc Wigner-D rotation, exact for
   E and B). This is needed only because C uses a *flat* FFT:
   - in equatorial CAR the patch spans Dec −73°…−25°, so the RA pixel width cos(Dec) varies by a factor 3.1;
   - after the rotation it spans Dec −38°…+23°, a factor 1.27;
   - the long axis lies along the equator.
2. Q/U: `curvedsky.alm2map(spin=[0,2])` onto a 9.6′ CAR patch sliced from the fejer1 full sky, with the same 76.8′
   margin as A. The CAR grid axes are the local meridian and parallel, so the Q/U frame *is* the grid frame and no ψ
   rotation is needed.
3. Mask, then `enmap.map2harm(normalize="phys", spin=[0,2], iau=False)` (`enmap.py:1358-1372`).
   - `queb_rotmat` (`enmap.py:1391-1400`) rotates with a = 2·atan2(l_x, l_y) and states "This corresponds to the
     Healpix convention".
   - (l_y, l_x) come from `laxes` with the *signed* pixel step (`enmap.py:1275-1299`), so the orientation of the map on
     the sky is taken into account.
   - `normalize="phys"` gives |FFT|² in C_ℓ units, the same as CMBpipeline's tfac.
4. Power spectrum: |E(l)|², |B(l)|² from `map2harm` (no re-masking), binned in `modlmap`, / w₂ of the 2D patch mask.

C shares the flat FFT with A and the exact CAR sampling with B. So:
- **A − C** isolates projection + interpolation + ψ rotation;
- **C − B** isolates the flat-sky Fourier treatment on the same kind of pixels.

What C still approximates:
- the Euclidean FFT on a CAR metric (the RA step varies as cos(Dec); `laxes` uses the mean physical step);
- periodic boundaries.

## 2.4 Quantities to measure

**Reference for every comparison, and how the common mask is separated.** Each pipeline's result is compared with
two references:
1. **Full-sky reference:** the realization's own C_ℓ (`alm2cl` of the input alm), or the theory C_ℓB_ℓ² for ensemble
   means.
2. **Cut-sky reference, pipeline-specific:** the exact true E and B fields (spin-0 synthesis of a_E, a_B), sampled on
   the pipeline's own pixels, multiplied by its own 2D mask, and transformed with the same transform as that pipeline.
   - For A: the true E/B evaluated exactly at the plane grid points (`curvedsky.alm2map_pos`) × `mask2d` → flat FFT →
     \|·\|² / w₂.
   - For B: true E/B on CAR × 2D CAR mask → `map2alm(spin=0)` → `alm2cl` / w₂.
   - For C: true E/B on the rotated CAR × mask → enmap FFT → \|·\|² / w₂.
   - This is what that pipeline would give with a perfect Q/U → E/B separation. It carries all of the coverage effects
     (mode coupling, f_sky normalization, boundary) but no E/B mixing and no projection error.

The ratio (pipeline / its cut-sky reference) is therefore the pipeline's *own* distortion. The difference in that
ratio between A and B is the additional effect of the projection + interpolation + flat sky.

### E/B leakage

1. Input E-only (a_B = 0), same seed, through A, B and C.
2. Recovered BB per bin:
   - absolute;
   - relative to the theory BB (r = 0.034 + lensing), i.e. how large it is compared with the signal of interest;
   - **excess leakage** = BB_A(E-only) / BB_B(E-only) per bin, and the same for C. B's leakage is the pure cut-sky mixing
     of the common footprint, so a ratio > 1 is what A adds.
3. Map level (one realization): the recovered B map vs distance to the mask edge (0–1°, 1–2°, 2–4°, 4–8°, > 8°), to
   show where the leakage sits.
4. Optional: B-only input (a_E = 0) → recovered EE (B→E).

### Power-spectrum recovery

- Input with both E and B (same seeds).
  - A: CMBpipeline's flat estimator (`PowerSpectrum.power` of the E/B maps).
  - B: `alm2cl` of the a_E, a_B from the spin-2 analysis of the Q/U multiplied by the 2D CAR mask, / w₂. This is not a
    pseudo-C_ℓ of re-masked maps.
  - C: the flat enmap estimator.
- Ratios per bin, recovered / full-sky reference and recovered / cut-sky reference, for EE and BB.
- **Binning:**
  - CMBpipeline's bins, `compute_bins_fractional(lmin=4, lmax=1000, frac=0.2, min_width=20)`
    = [4, 24, 44, 64, 84, 104, 125, 150, 180, 216, 259, 311, 373, 447, 537, 644, 773, 927, 1113].
  - Plus a **low-ℓ binning** of width Δℓ = 10 for 2 ≤ ℓ < 102. The fundamental Fourier modes of A's grid are
    Δℓ_x = 2π/112.6° = 3.2 and Δℓ_y = 2π/66.6° = 5.4, so bins narrower than ~10 are not meaningful for the flat
    pipelines.

### Mode mixing

1. **Controlled inputs:** E-only a_ℓm that are non-zero only in a narrow shell ℓ₀ − 2 ≤ ℓ ≤ ℓ₀ + 2, with a flat
   spectrum and random phases, for ℓ₀ = 30, 60, 100, 200, 400, 600, 800. That covers the low-ℓ regime and the ℓ₀ of the
   paper's App. B. 10 realizations per ℓ₀.
2. Output: the recovered EE (and BB) power vs output ℓ. Flat pipelines use Δℓ = 4 bins (about the fundamental mode);
   the curved-sky pipeline uses Δℓ = 1 and is then also shown in Δℓ = 4 bins.
3. **Response shape:**
   - normalized to unit sum;
   - **centroid shift** ⟨ℓ⟩ − ℓ₀;
   - **width** = 84th − 16th percentile (the paper's Table 5 definition);
   - fraction of the response outside ℓ₀ ± 20.
4. **Common footprint coupling** is measured by the same shell run through each pipeline's cut-sky reference (true E
   field × mask, same estimator). The widths are compared: pipeline vs its reference, then A vs B. The difference is the
   extra mixing from projection/interpolation/flat sky.
5. **No-mask control:** A with the whole plane rectangle filled from the sphere (mask2d = 1), and B/C on the full patch
   without a mask. This isolates the projection-induced kernel (paper's App. B) from the footprint. B without a mask
   returns the input exactly, so it is also a correctness check.

### Statistical validation

1. First, one realization (seed 1000), inspected at map and spectrum level for all tests.
2. Then **N = 50 realizations** (seeds 1000–1049) from the same theory spectra, for the leakage and recovery tests.
   The mode-mixing test uses its 10 realizations per ℓ₀.
3. Reported per bin: mean, standard deviation, and standard error σ/√N, for A, B and C.
4. **Paired comparison.** All pipelines see the same realization, so the per-realization difference
   Δ_b = ratio_A − ratio_B is formed. A difference is called systematic only if |⟨Δ_b⟩| > 3·σ(Δ_b)/√N. Pairing removes
   the cosmic variance that the pipelines share.

## 2.5 Configuration and structure

| parameter | value | reason |
|---|---|---|
| nside (A's input) | 512 | CMBpipeline `config.dict` |
| lmax | 1535 | 3·nside − 1, CMBpipeline's synthesis band limit |
| lmax of B's `map2alm` | 1124 | fejer1 exactness at 9.6′ (n_rings − 1). The spectra are compared up to ℓ = 1000 |
| beam | Gaussian, FWHM 23′ | `config.dict:52` |
| SO region | `make_SO_mask` (hits > 0.25, θ < 100°), f_sky 0.111 | CMBpipeline |
| plane grid (A) | **nx × ny = 704 × 416**, 9.6′, margin 8 × 9.6′; 189,097 observed pixels | CMBpipeline |
| CAR grid (B) | **nx × ny = 2250 × 1125** (RA × Dec, 9.6′ fejer1 full sky); patch box 1090 × 304; 286,159 observed pixels | matched to the plane pitch, see below |
| CAR patch (C) | rotated 9.6′ CAR patch, ≈ 708 × 397 including the 76.8′ margin | same pitch as A |
| CAR control | 6′ fejer1, 3600 × 1800 (exact to ℓ = 1799); 732,493 observed pixels | measures any sampling / band-limit effect of 9.6′ |
| mask | binary; 3° cosine apodization optional (off) | §2.2 |
| binning | CMBpipeline bins + Δℓ = 10 low-ℓ bins | §2.4 |
| realizations | seed 1000 (single), 1000–1049 (N = 50), 10 per ℓ₀ (mode mixing) | §2.4 |

**Matching the CAR resolution to the planar grid.** The plane samples every direction at 9.6′, which sets its Nyquist
multipole to ℓ_N = π/9.6′ ≈ 1125.
- A CAR grid with Δ = 9.6′ has exactly this step in Dec, so the same ℓ_N along Dec. Its RA step (9.6′·cos Dec,
  2.8′–8.7′ in the patch) is always finer.
- The CAR representation is therefore never coarser than the plane anywhere in the patch. The shared beam makes the
  content above ℓ_N negligible (§2.1).
- 9.6′ also divides the sphere exactly (1125 rings), which keeps the curved-sky transforms exact up to ℓ = 1124.
- A coarser CAR grid would under-sample relative to A. A finer one would give B more resolution than A has. The 6′
  control shows whether this choice affects any result.

**File structure.** Phase 3 style: plain functions, no classes or configuration framework; parameters defined visibly
in the notebook.

```
src/
  sky_input.py      theory spectra (CMBpipeline utilities), seeded alm, beam, E-only / B-only / shell (l0) alm
  cmbpipeline_a.py  pipeline A as plain functions calling ../CMBpipeline/source:
                    SO geometry, Q/U rotation + projection + interpolation, flat E/B, plane grid positions
  pixell_b.py       CAR geometry, mask transfer, curved-sky E/B (pipeline B)
  pixell_c.py       alm rotation, CAR patch, enmap flat E/B (pipeline C)
  spectra.py        flat estimator (A, C), curved alm2cl of the masked-Q/U analysis (B), binning, response centroid/width, paired statistics
notebooks/
  01_pixell_validation.ipynb   top-to-bottom: parameters -> input sky -> coverage -> A / B / C step by step
                               -> leakage -> recovery -> mode mixing -> ensemble statistics -> summary numbers
results/task1/                 cached arrays and figures (git-ignored)
```

The preliminary Phase 3 code (`config_task1.dict`, `config_loader.py`, `pipeline_current.py`, `pipeline_sphere.py`,
`metrics.py`, `validation.py`) will be refactored into this layout in Phase 3. That includes removing the `Geometry`
class and the config loader, as the Phase 3 instructions require.
