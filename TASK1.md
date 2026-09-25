# Task 1: Validation test — can pixell replace the projection + flat-sky Fourier step?

Scientific question: can pixell represent the same sky patch and recover its
E/B polarization information while reducing or avoiding the projection and
flat-sky approximations of the current CMBpipeline?

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
Write `docs/02_test_plan.md` proposing:
- Common input: one set of input alm (T, E, B) from a known theory spectrum and
  fixed seed. All approaches must start from these same alm, so that the true
  E/B maps of the patch and the true spectra are known (ground truth).
- Configuration: nside, lmax, patch center and size, pixel size, beam, mask - matched to what CMBpipeline uses.
- Pipelines to compare, all against ground truth:
  A. current CMBpipeline: sphere → projection/interpolation → grid Q/U → flat FFT → E/B
  B. pixell, CAR geometry: alm2map onto a CAR patch (enmap) → enmap Fourier → E/B
  C. pixell, curved sky: spherical harmonic transforms on the patch (curvedsky)
     → E/B
  For each pixell step, justify from docs/source code which function is
  mathematically appropriate and why. Do not blindly replace one FFT with another.
- Metrics: Q, U, E, B map residuals vs truth; correlation coefficients (map
  level and per ell bin); recovered EE/BB spectra vs input; E→B leakage;
  behavior at low ell and near the boundaries.
- Proposed file structure (src modules + notebook).
STOP.

## Phase 3 — Implement and run
Implement the approved plan: reusable functions in `src/`, analysis in
`notebooks/01_pixell_validation.ipynb`. Produce the diagnostic plots and the
quantitative comparisons. Summarize the main numbers when done.
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
