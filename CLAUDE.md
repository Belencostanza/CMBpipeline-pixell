# CMBpipeline-pixell

## Goal
Develop a new version of the CMB polarization analysis pipeline `CMBpipeline`
using `pixell`, to reduce or remove the approximations introduced by projecting
spherical maps onto a plane and applying flat-sky 2D Fourier transforms.

## Repositories (local)
- `./` — CMBpipeline-pixell: the ONLY place where files may be created or modified.
- `../CMBpipeline/` — existing pipeline. READ-ONLY. Never modify it.
- `pixell` — installed package; source: https://github.com/simonsobs/pixell
- `../CMBpipeline/paper_CMBpipeline.pdf` — scientific paper for the existing
  pipeline. Essential for the mathematical formulation and the approximations
  involved. <!-- adjust this path if the PDF is somewhere else -->

## Existing pipeline (summary)
1. Simulate or read spherical CMB Q/U maps.
2. Project onto a tangent plane and interpolate onto a regular 2D grid.
3. Train DeepWiener to emulate Wiener filtering on the 2D maps.
4. Estimate noise bias and Fisher matrix for the Optimal Quadratic Estimator (OQE).
5. Apply the OQE to estimate the input power spectrum.
6. Compute a pseudo-Cl estimate with NaMaster.
7. Compare OQE and pseudo-Cl.

## Rules
- Development is incremental. Do only the current task; do not start later stages.
- Keep structure and coding style close to CMBpipeline.
- Reusable code in `src/` modules; analysis and plots in `notebooks/`;
  written documentation and reports in `docs/`.
- Always check conventions explicitly: coordinates, Q/U sign convention, E/B convention, normalization, pixel size, pixel window, beam, geometry.
- Use fixed random seeds so results are reproducible.
- Never claim that pixell is better unless numerical tests support it.
  Report discrepancies and unresolved issues honestly.
- When the pixell documentation is unclear, read the pixell source code.
- If running CMBpipeline requires missing data or dependencies, tell me instead
  of reimplementing or working around it.
- Only work and test with SO configuration, skip QUBIC-like configuration.
