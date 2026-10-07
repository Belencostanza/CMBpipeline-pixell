# Training DeepWiener on the CAR dataset — plan

**Goal.** Train `DeepWiener_threechannels` on the Task 2 CAR dataset, following CMBpipeline's procedure
(`../CMBpipeline/source/training_opt_beam_changed.py`, `losses.py`), and change only what the CAR geometry requires.

Plan only: nothing is implemented yet. Line numbers refer to `../CMBpipeline/source/`.

---

## 1. CMBpipeline's training procedure

| step | where | what it does |
|---|---|---|
| config | `training_opt_beam_changed.py:36-53` | `epochs = 200`, `loss_j3 = True`, `batch_size = 1`, `map_rescale_factor`, `fwhm_rad`, file and folder names |
| geometry | `:61-62` | `build_geometry` is called **only to get the plane pixel size dx, dy** of the projected grid |
| prior spectra | `:55, 64-68` | `signal_spectrum(r)`, C_ℓ^EE and C_ℓ^BB multiplied by `factor²`, then interpolated onto the 2D grid of planar modes \|k\| (`utilities.power_spectrum_flat`, log-log interpolation) |
| data | `:85-112` | loads the full `.pt`; Q, U → `(x − mean over the whole image) × factor`; mask and σ² unchanged. **Input** = (Q, U, mask, σ²); **target** = (Q, U), i.e. the observed maps themselves. Also returns `mask[0]`, `inho[0]` for the loss |
| network | `:213-219` | `DeepWiener_threechannels(2, 2, filters)`: Q/U → linear branch; mask → non-linear branch; (mask, σ²) → second non-linear branch (`network_2d.py`, forward at the class's `x[:, 0:2]`, `x[:, 2:3]`, `x[:, 2:4]`) |
| loss | `:229-241` → `losses.lossj3_plane_beam_inho` | J₃ (§2) |
| optimizer | `:222-224` | Adam(lr, betas = (0.5, 0.999), weight decay wd); OneCycleLR(max_lr = 3 lr, pct_start = 0.3, cosine), stepped every batch |
| loop | `:148-194, 249-272` | train / validate each epoch; saves the state dict whenever the validation loss improves; loss histories saved as `.npz` |
| search | `:197-208, 281-297` | Optuna, 10 trials: filters₀…₃ (filters₄ = filters₅ = filters₃), lr ∈ [10⁻⁶, 10⁻⁴], wd ∈ [10⁻⁶, 10⁻³] (paper Table 1); minimizes the best validation loss |

**Cluster log** (`/home/bcostanza/wf-curve/pol/train_plnj3_geo_opt_beam_maskSO_eq_rad40_margin8.out`):
- about **120 s per epoch** on one GPU, so about 6.5 h per 200-epoch trial and about 65 h for 10 trials;
- best validation losses of 1.0–1.4.

## 2. CMBpipeline's loss, written out

`lossj3_plane_beam_inho = realspace_loss_beam_inho + fourier_loss` (`losses.py:325-334`). With ŷ the network output,
d the normalized data (the target), f = `factor`, N_pix = n_x·n_y and N_r = n_y·(n_x/2 + 1) the number of rfft modes:

- **Likelihood term** (`losses.py:134-165`):
  (1/N_pix) Σ_i m_i [d_i − (B ⋆ ŷ)_i]² / (σ_i² f²), for Q and for U.
  - B ⋆ ŷ is the beam applied **with a flat FFT** on the grid: `beam_filter` uses `beam_flat` = exp(−k²σ_b²/2) on \|k\| from dx, dy. The FFT is periodic.
  - σ_i² is the channel-3 variance, estimated from 100 projected noise simulations.
- **Prior term** (`losses.py:203-238`):
  (1/N_r) Σ_{k≠0} \|E_k\|²/C^EE(\|k\|) + (same for B).
  - E_k and B_k come from `transf_eb2`: rfft2 × √(dx dy/N_pix), then rotated by 2φ_k.
  - C(\|k\|) is interpolated from the 1D C_ℓ.

**Check of the normalization.** With this tfac, \|E_k\|² estimates C, and the exact Gaussian prior ŷᵀS⁻¹ŷ equals
Σ_{all k} \|E_k\|²/C. Averaging over the ≈ N_pix/2 rfft modes therefore gives ≈ (1/N_pix) Σ_{all k}. So the code computes

  **J₃ ≈ (1/N_pix) · [ Σ_i m_i (d_i − Bŷ_i)²/σ_i² + Σ_k \|E_k\|²/C_k^EE + \|B_k\|²/C_k^BB ]**,

which is the paper's Eq. 4.1 divided by N_pix. The two terms carry the correct relative weight, apart from the rfft's
self-conjugate columns (k_x = 0 and Nyquist), which are counted once instead of twice. This is a small effect.

**Flat-sky approximations used in the loss** (as in `docs/01_current_pipeline.md`, A2, A5, A7):
1. the beam is applied in planar Fourier space, with ℓ = \|k\|;
2. the prior assumes S diagonal in planar Fourier modes, with C_ℓ interpolated onto \|k\|;
3. both FFTs are periodic on a masked, non-periodic rectangle;
4. the pixel noise is treated as diagonal after interpolation, although it is correlated.

## 3. What changes on the CAR maps

### 3.1 The loss — the main difference (as you expected)

**The flat-FFT parts of J₃ are wrong on CAR pixels and cannot be reused.**
- A CAR pixel is 9.6′ in Dec but 9.6′·cos δ in RA: 2.8′ at −73° and 8.7′ at −25°.
- An FFT with a fixed dx, dy therefore assigns the wrong ℓ to the RA modes, by up to a factor of 3.4 at the southern edge. The beam and C_ℓ would be evaluated at the wrong scale, in a way that depends on position.
- Reusing `losses.py` unchanged would put back a worse version of the approximation this project removes.

**The likelihood term carries over with only one change.**
- The noise is drawn per CAR pixel with the analytic σ², so N is exactly diagonal. Σ_i m_i r_i²/σ_i² is exact here; in CMBpipeline it was an approximation.
- What changes is **the beam**: B ⋆ ŷ must be a curved-sky operation, Y B_ℓ A ŷ, where A = `map2alm`, Y = `alm2map`, and B_ℓ = `hp.gauss_beam` is the same beam that made the data.

**The prior becomes harmonic:** Σ_ℓm \|E_ℓm\|²/C_ℓ^EE + \|B_ℓm\|²/C_ℓ^BB with (E, B)_ℓm = A ŷ.
- ℓ < 2 is excluded, and the m > 0 modes are counted twice (real field).
- This is the exact Gaussian prior on the sphere for the field ŷ.
- Like CMBpipeline's torus, it treats ŷ as zero outside the rectangle. The rectangle's edge is ≥ 8 pixels outside the mask, as in CMBpipeline.

**Proposed curved-sky J₃** (same 1/N_pix scale as CMBpipeline, so that lr and wd ranges and loss values stay comparable):

  a = A ŷ  (spin-2, ℓ ≤ L = 1124)
  J₃ᶜᵃʳ = (1/N_pix) · [ Σ_i m_i (d_i − (Y B a)_i)² / (σ_i² f²) + Σ_{ℓ≥2,m} w_m (\|a^E_ℓm\|²/(C_ℓ^EE f²) + \|a^B_ℓm\|²/(C_ℓ^BB f²)) ],  w_0 = 1, w_{m>0} = 2

**Implementation: one differentiable SHT.**
- pixell has no autograd, but it exposes the exact adjoints (`map2alm(..., adjoint=True)`, `alm2map(..., adjoint=True)`).
- A `torch.autograd.Function` can call A or Y in its forward pass and the adjoint in its backward pass, on the CPU with ducc, moving the tensors between GPU and CPU.
- Dot tests on the 1120 × 320 cut-out, spin-2, L = 1124 (run for this plan):
  - ⟨Ya, f⟩ = ⟨a, Yᵀf⟩ to a relative 3×10⁻¹⁵;
  - ⟨Af, a⟩ = ⟨f, Aᵀa⟩ to 8×10⁻¹⁴.

  The gradients are therefore exact.

**A subtlety specific to CAR: the band limit.**
- At −73°, the RA spacing of 2.8′ resolves local scales up to ℓ ≈ 3900. The exact transform stops at L = 1124 (the fejer1 limit at 9.6′).
- The network can put structure at 1124 < ℓ ≲ 3900 in the RA direction at low Dec. Such modes:
  - are invisible to the likelihood, because the beam has removed them: B_ℓ ≈ 6×10⁻³ at ℓ = 1124 and ≈ 10⁻⁴ at 1500;
  - are not properly penalized by a prior computed at L = 1124 (they alias into it).

  In CMBpipeline the FFT prior reached the Nyquist frequency, so this didn't happen.
- **Proposal:** define the Wiener-filter output as the band-limited map ŷ_WF = Y A ŷ (ℓ ≤ L). The loss sees exactly this map, and the output never contains modes the loss cannot see. It costs nothing extra, because A and Y are already computed.
- To be tested (§5): that Y A behaves as a projector on network outputs, i.e. that ‖YA(YAŷ) − YAŷ‖ is small.

### 3.2 Training script — small changes

| item | CMBpipeline | CAR version |
|---|---|---|
| dx, dy from `build_geometry` | needed for the flat FFTs | **removed**; the CAR geometry (shape, WCS) is read from `aux/mask_car_*.fits` |
| `power_spectrum_flat` | C_ℓ interpolated onto \|k\| | **removed**; C_ℓ used directly on ℓ, plus B_ℓ for ℓ ≤ L |
| `map_rescale_factor` | 0.8382 | **0.75068694** (your new value), in `config.dict` |
| data loading and normalization | `create_dataloader` | **same code**: mean over the whole image, × factor, channels unchanged. This keeps the network input statistics comparable |
| network | `DeepWiener_threechannels` | **same class**, imported read-only from `network_2d.py`. It is fully convolutional, so 320 × 1120 works (forward pass checked in notebook 02) |
| optimizer, scheduler, Optuna ranges, best-model saving | — | **same** |
| `torchsummary`, `optuna` | imports | same environment (`torch_cmb` on the cluster) |

### 3.3 Other differences: no code change, but they can affect the results

1. **Anisotropic pixel scale.**
   - A 3 × 3 convolution covers 29′ × 8′ of sky at −73°, but 29′ × 26′ at −25°.
   - The network must learn a filter that depends on position. The σ² channel encodes cos δ, because σ² ∝ 1/Ω_pix, so the network can see where it is.
   - If training struggles here, the natural fix is an extra cos δ input channel. That would be a change to the architecture, so it is not proposed now.
2. **σ² channel.** It ranges from 0.046 to 0.44 μK², smooth and analytic. CMBpipeline's ranged from 0 to 27 μK², with interpolation spikes at the edges. Because it is unnormalized (`inho_norm = inho`), its scale differs from what CMBpipeline's network saw. That doesn't matter, since we train from scratch.
3. **Mean subtraction over the whole image**, including the zeros outside the mask. This makes the input outside the mask equal to −mean·f, not 0. The behaviour is the same as CMBpipeline's, so it is kept for consistency.
4. **Image size.** 358,400 pixels, against 292,864 (+22 %). GPU memory per sample grows by the same fraction.

## 4. Cost: the real constraint

Measured locally, 1120 × 320 cut-out, spin-2, L = 1124, `method='2d'`:

| threads | A | Y | Aᵀ | Yᵀ |
|---|---|---|---|---|
| 1 | 0.72 s | 0.53 s | 0.77 s | 0.47 s |
| 4 | 0.23 s | 0.17 s | 0.25 s | 0.15 s |
| 8 | 0.19 s | 0.16 s | 0.23 s | 0.12 s |

- **Per training sample:** about 0.35 s forward (A, Y) plus about 0.35 s backward (Aᵀ, Yᵀ) ≈ **0.7 s on the CPU**, against milliseconds for the flat FFTs on the GPU.
- **Per epoch:** ≈ 12 min of SHTs plus ≈ 2 min of network, about 14 min. That is about 7 × CMBpipeline: **≈ 45 h per 200-epoch trial**.
- The cluster's GPU nodes have 256 (apollo01-02) or 64 (h200) CPUs for 8 GPUs, so 8–32 threads per job are realistic. Scaling from 4 to 8 threads was already weak.

Ways to reduce it, to be measured before the large run:
1. **Transform only the 320 rings of the cut-out** instead of the zero-padded 1125 rings, using a ring-by-ring method with the fejer1 quadrature weights. This should be about 3× faster. It must reproduce the `'2d'` result exactly. In Task 1 the default `'cyl'` did not, so the cause has to be found first.
2. **Batch size > 1.** One ducc call on several maps uses the threads better. It changes the optimization, though: CMBpipeline uses batch 1.
3. **Fewer Optuna trials.** Start from the architecture and lr/wd of CMBpipeline's selected model, then run a short search around it.
4. **Optional warm start:** a few epochs with a cheap approximate loss, then the exact loss. Only if 1–3 are not enough.
5. **GPU SHT libraries.** torch-harmonics has no spin-2 transform, and s2fft's torch support is unverified. Not proposed unless 1–3 fail.

## 5. Files and steps

```
src/
  sht_torch.py        # autograd Functions: CarAnalysis (A, backward Aᵀ), CarSynthesis (Y, backward Yᵀ); spin-2, pixell/ducc
  losses_car.py       # realspace_loss_beam_inho_car, harmonic_prior_car, lossj3_car_beam_inho (same names/structure as losses.py)
  training_car.py     # copy of training_opt_beam_changed.py with the changes of §3.2 and the new criterion
  config.dict         # + epochs, loss_j3, batch_size, model/loss/study folders, study_name, sht_lmax, sht_nthreads, factor 0.75068694
slurm/train.sh        # GPU job (template: ../CMBpipeline/slurm/train_opt_beam_changed.sh)
notebooks/03_loss_check.ipynb
```

**Step 1 — loss implementation and validation (local, CPU). STOP after it.**
- `gradcheck` of the SHT Functions in float64, on a small CAR geometry.
- **Prior normalization:** for signal-only maps drawn from C_ℓ, E[prior]/N_modes ≈ 1.
- **Likelihood normalization:** for ŷ equal to the true unbeamed signal, E[likelihood] = N_obs (the number of observed pixels).
- **Consistency with CMBpipeline:** on a CMBpipeline plane map, the flat J₃ and the CAR J₃ evaluated on the same true signal should give comparable per-pixel values. This is not identical by construction; the differences will be reported.
- **Band-limit test** of Y A (§3.1).
- **Timing** of one training step (network + loss) on CPU, then on GPU.

**Step 2 — short training test on the cluster (GPU).** One fixed architecture (CMBpipeline's selected filters, lr, wd), a few epochs on a subset. Checks: the loss decreases, there are no NaNs, the epoch time is as estimated, and the memory fits. STOP.

**Step 3 — full Optuna run**, after your approval of the cost and number of trials.

Not in this task: evaluating the Wiener-filter quality, and the OQE/Fisher steps.

## 6. Decisions for you

1. **Exact curved-sky loss (recommended)**, or accept a faster but approximate loss? The exact one costs about 7× CMBpipeline's training time unless §4.1–4.3 bring it down.
2. **Band-limited output** ŷ_WF = Y A ŷ (recommended), or keep the raw network output as the Wiener-filter map?
3. **Search budget:** a full 10-trial Optuna search, or start from CMBpipeline's selected hyperparameters? If the latter, which study or model was the final one? The cluster log mixes several studies.
4. **`map_rescale_factor` = 0.75068694:** how was it computed? I'll record that in the config comment so it can be reproduced.

---

## 7. Results of steps 1 and 2 (2026-10-07)

**Step 1** (`notebooks/03_loss_check.ipynb`, local CPU):
- **Gradients:** `gradcheck` passes for both transforms (`'cyl'` and `'2d'`) and for the whole J3. The adjoints on the 1120 × 320 cut-out agree to 4×10⁻¹⁴.
- **Data term:** at the true signal of validation maps 0–2 (regenerated exactly from their seeds), term 1 / (2 N_obs/N_pix) = 0.997–0.999 with the signal band-limited to ℓ ≤ 1124. With the full ℓ ≤ 1535 signal it is 1.005–1.006, because ℓ > 1124 power aliases.
- **Prior term:**
  - the E part / f_sky Σ(2ℓ+1) = 0.999;
  - the B part = 1.84, from E→B leakage at the rectangle's edges (an E-only field already gives 0.86 of the expected B prior). CMBpipeline's periodic FFT prior has the same effect at the torus wrap.
- **Band limit:** ‖YA(YAŷ) − YAŷ‖/‖YAŷ‖ ≈ 1 % in the mask (true signal and untrained network), so Y A acts almost as a projector.
- **`'cyl'` instead of `'2d'`:** same prior per ℓ range (to 10⁻⁴) and same beam operator (to 2×10⁻⁴) for band-limited maps; the transforms are 2.2× faster.

**Step 2** (job 17385188, H200 on apollo03, 16 threads; 3 epochs, 100 + 20 maps, CMBpipeline's best trial filters [16, 40, 64, 152], lr 5.03×10⁻⁵, wd 1.61×10⁻⁵):

| epoch | s/step | transforms per step | train loss | valid loss |
|---|---|---|---|---|
| 0 | 0.173 | 0.077 s | 261.0 | 32.2 |
| 1 | 0.133 | 0.076 s (57 %) | 22.8 | 17.6 |
| 2 | 0.131 | 0.074 s (57 %) | 16.2 | 15.8 |

- **Projected cost:** 1000 maps → ≈ 131 s of training + ≈ 9 s of validation ≈ **2.3 min per epoch**, so ≈ 7.8 h per 200-epoch trial and ≈ 78 h for 10 trials. CMBpipeline: ≈ 2 min per epoch, ≈ 65 h (its GPU type is not recorded in the log).
- **Reference loss values** (map 0): J3(ŷ = 0) = 39.5; J3(true signal, ℓ ≤ 1535) = 8.9; J3(true signal band-limited to 1124) = 3.05. The Wiener filter minimizes J3, so a trained network should end below these. They are not comparable to CMBpipeline's 1.0–1.4, because the prior is counted over different modes.
- **Partition:** h200's time limit is 1 day, enough for 2–3 trials per job; apollo's is 7 days. The Optuna study (sqlite, `load_if_exists`) can continue across jobs.
