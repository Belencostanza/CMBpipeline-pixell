# Task 1: Validation test — can pixell replace the projection + flat-sky Fourier step?

Objective: given the same underlying spherical a_lm^{E,B} and the same physical
SO coverage, quantify the additional distortion, mode mixing and E/B leakage
introduced by the current projection + interpolation + flat-sky representation,
relative to a CAR representation treated consistently with curved-sky operators
(pixell).

Do NOT implement DeepWiener, loss function, Fisher matrix, noise bias, OQE or
NaMaster. This task is split into phases. STOP at the end of each phase and wait
for my approval before starting the next one.

## Phase 1 — Understand the existing implementation (no new code)
Read `../CMBpipeline/` and the paper. Write `docs/01_current_pipeline.md`
documenting, with file names, function names and line references, exactly how
the current code:
- generates or reads spherical Q/U maps;
- selects the observed sky region (and mask/apodization, if any);
- projects the spherical maps onto the plane (which projection);
- interpolates onto the 2D grid (method, pixel size, grid size);
- performs Fourier transforms (normalization, ell mapping);
- converts Q/U into E/B (angle convention, signs).
Also list the approximations involved and how the paper justifies them.
Note anything in the code that differs from the paper.
STOP.

## Phase 2 — Design the test (no implementation yet)

Write `docs/02_test_plan.md` with the following sections.

### 2.1 Common input (reference sky)

- The reference sky is the band-limited continuous field defined by a single set
  of input a_lm^{T,E,B} (theory spectrum, fixed seed, explicit lmax). It is not
  any pixelized map: HEALPix and CAR are only two ways of sampling the same
  underlying spherical field.
- Every pipeline starts from the same input a_lm.
- Use the same beam in both pipelines. Explicitly document the pixelization and
  effective angular resolution of each representation rather than assuming an
  identical pixel window.
- The input realization itself, through its a_lm and alm2cl, provides the
  full-sky reference spectrum before applying the SO coverage.

### 2.2 Common coverage

- Define one physical sky region within the SO footprint on the sphere, and use
  the same physical coverage in both pipelines.
- Define the mask/apodization on the sphere and transfer it consistently to each
  geometry.
- State explicitly how this same physical region is represented by the current
  CMBpipeline projection and by the pixell CAR geometry.
- Document differences in pixelization, pixel area, angular resolution and
  boundary representation between the two geometries.

### 2.3 Pipelines

- A. Current CMBpipeline: a_lm → spherical Q/U → projection + interpolation →
  planar grid Q/U → flat-sky FFT → E/B → flat-sky power spectrum.
- B. Pixell pipeline: a_lm → CAR Q/U (pixell enmap geometry) → masked
  curved-sky spin-2 analysis (pixell curvedsky) → E/B → curved-sky power
  spectrum using the appropriate pixell operations. 
- C. Intermediate (optional, to separate contributions): a_lm → CAR Q/U →
  flat-sky Fourier transform (enmap) → E/B → flat-sky power spectrum.
  This can help distinguish effects associated with the flat-sky Fourier
  treatment from those associated with projection/interpolation.

For each pixell function used, justify from the documentation/source why it is
the mathematically appropriate operation. Pay particular attention to the Q/U
polarization convention and the local polarization basis.

### 2.4 Quantities to measure

#### E/B leakage

- Start with an E-only realization:
  a_lm^E != 0, a_lm^B = 0.
- Process exactly the same realization through pipelines A and B.
- Measure the recovered B signal in each pipeline.
- Compare the recovered B power as a function of multipole/bin.
- Since the SO footprint itself produces cut-sky E/B mixing, do not interpret
  every recovered B mode as an error of the pipeline. The relevant comparison
  is whether pipeline A introduces additional leakage relative to the
  consistently curved-sky pixell treatment for the same physical coverage.
- Optionally repeat the test with a B-only realization
  (a_lm^E = 0, a_lm^B != 0) to measure B→E mixing.

#### Power-spectrum recovery

- Use realizations containing the desired input E and B spectra and process the
  same input a_lm through both pipelines.
- For pipeline A, calculate the power spectrum using the existing flat-sky
  Fourier implementation.
- For pipeline B, calculate the power spectrum using the appropriate pixell
  curved-sky implementation. Don't do the pseudo-Cl, if it has a mask calculate the power spectrum 
  with the 2D mask.
- Compare both recovered spectra with the spectrum of the same input
  realization.
- Because the maps cover only the SO region, explicitly distinguish differences
  caused by the common sky cut/mask from additional differences introduced by
  projection, interpolation or the flat-sky approximation.
- Compare the recovered spectra as a function of multipole, with particular
  attention to the low-ell regime where the flat-sky approximation is expected
  to be most relevant.

#### Mode mixing

- Generate controlled band-limited inputs, or narrow bands centered at selected
  multipoles ell_0.
- Process the same input through pipelines A and B.
- Measure how power initially localized around ell_0 is redistributed over
  output multipoles.
- Compare the width and shape of the resulting mode-coupling response between
  the two pipelines.
- Separate, as far as possible, the mode coupling common to both pipelines due
  to the finite SO footprint from additional mixing associated with the
  projection/interpolation and flat-sky treatment.

#### Statistical validation

- After validating the procedure with individual realizations, repeat the main
  tests over multiple realizations generated from the same input theory spectra.
- Compare the mean recovered spectra, leakage and scatter for pipelines A and B.
- Use these simulations to determine whether observed differences are
  systematic or consistent with realization-to-realization variance.

### 2.5 Configuration and structure

- Propose values for nside, lmax, SO region, CAR resolution matched as closely
  as possible to the CMBpipeline grid, beam, mask, apodization and multipole
  binning.
- Explicitly justify how the angular resolution of the CAR geometry is matched
  to the current planar grid.
- Proposed file structure (src modules + notebook).

STOP.

## Phase 3 — Implement and run

Implement the approved plan with reusable functions in `src/` and the analysis in
`notebooks/01_pixell_validation.ipynb`.

Keep the notebook **simple, explicit, and easy to follow**. Follow the programming
style of the notebooks in the original `CMBpipeline` repository as much as
possible.

In particular:

- Organize the notebook as a sequence of clear steps that can be executed from
  top to bottom.
- Explicit the number of pixels for maps of pipeline A (nx x ny) and B. 
- Keep important variables such as `nside`, `lmax`, resolution, beam, mask,
  binning, and number of simulations explicitly visible in the notebook.
- Prefer simple function calls and explicit intermediate variables over complex
  abstractions.
- Do not introduce classes, configuration frameworks, dataclasses, command-line
  parsers, or unnecessary helper layers.
- Put functions that are genuinely reused in `src/`, but keep the scientific
  workflow and calculations visible in the notebook.
- Show intermediate maps, spectra, and relevant quantities so that each step can
  be inspected independently.
- Use clear variable names consistent with the original pipeline.
- Add short comments explaining the purpose of each step, but avoid excessive
  markdown or long explanations inside the notebook.
- Do not hide important scientific operations inside large convenience
  functions merely to make the notebook shorter.

Produce the diagnostic plots and quantitative comparisons defined in Phase 2
and summarize the main numerical results at the end.

The priority is that I can open the notebook and easily understand, modify, and
rerun each part of the analysis myself.

STOP.

## Phase 4 — Report
Create `docs/report_task1.html` covering:
1. what the original pipeline does in the projection/Fourier step;
2. which pixell functions replace each operation;
3. the mathematical difference between the approaches;
4. the exact test configuration;
5. plots and quantitative results;
6. which flat-sky/projection approximations are removed, reduced, or still present;
7. discrepancies and unresolved issues.
