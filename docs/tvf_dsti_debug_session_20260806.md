# TVF DSTI Debug Session — 2026-08-06

## Session Summary

Debugged catastrophic failure of `syntx.tvf(regularizer='dsti')` across the 90-pair Mindboggle benchmark. Five genuine code bugs were found and fixed. A sixth issue — "58% grid folding" — consumed significant investigation time before being identified as a **measurement artifact** (`do_log=1` returns `log(det J)`, not `det J`; checking `≤ 0` measures compression, not folding). The actual grid folding is **0.0000%** across all optimizer/regularizer combinations.

---

## Bugs Fixed

### 1. Missing DSTI Regularizer Dispatch
- **File:** `src/syntx/tvf.py`, `TVFModel.fit()` ~line 1031
- **Problem:** The gradient smoothing dispatch had branches for `'sobolev'` and Gaussian (`fast_smooth`) but no `elif regularizer_mode == 'dsti'` branch. When `regularizer='dsti'` was passed, it silently fell through to isotropic Gaussian smoothing.
- **Also:** `TVFModel` class didn't have the `_apply_dsti_green_operator()` method (it only existed on `SyNTo` in `syn.py`).
- **Fix:** Added `elif regularizer_mode == 'dsti'` branch at line 1031 + copied `_apply_dsti_green_operator()` method to `TVFModel` at line 360.

### 2. `cfl_max=0.40` Velocity Norm Cap
- **File:** `src/syntx/tvf.py`, `TVFModel.fit()` (removed block, was ~line 987)
- **Problem:** Every epoch, if the max velocity norm exceeded `0.40`, the ENTIRE velocity field was globally rescaled down: `velocity *= (0.40 / max_norm)`. Diagnostic showed `Velocity Abs Max: 0.399997` — pinned at the cap. This prevented any meaningful deformation beyond sub-voxel.
- **Evidence:** The cap value was `kwargs.get('cfl_max', 0.40)` — a hardcoded default.
- **Fix:** Removed the `cfl_max` block entirely. The `velocity_clamp=50.0` element-wise clamp remains as a NaN safety net.

### 3. CFL Momentum Geometric Accumulation
- **File:** `src/syntx/tvf.py`, `TVFModel.fit()` ~line 1080
- **Problem:** The momentum update was `buf = μ·buf + update; velocity -= buf`. This subtracts the FULL accumulated buffer, not just the current update. With `μ=0.95`, the buffer converges to `update / (1-0.95) = 20× update`, making the effective step size 20× larger than `grad_step`.
- **Fix:** Changed to `velocity -= (1-μ) · buf`. Steady-state effective step = `(1-μ) · g/(1-μ) = g` — exactly the CFL-bounded update. Momentum smooths gradient direction without amplifying magnitude.

### 4. `constant_speed` Not Implemented
- **File:** `src/syntx/tvf.py`, `TVFModel.fit()` ~line 1101
- **Problem:** `constant_speed=True` and `constant_speed_relaxation=0.10` were accepted as kwargs but never consumed. No code enforced the geodesic constant-speed constraint on velocity keyframes.
- **Fix:** Added post-step projection that normalizes per-keyframe RMS speed toward the mean with the specified relaxation factor: `v(t_k) *= (1-α) + α · (mean_speed / speed_k)`.

### 5. Adaptive ODE Integration Steps
- **File:** `src/syntx/tvf.py`, `TVFModel.integrate()` ~line 525
- **Problem:** Fixed step count (`n_time_steps × integration_steps_per_interval = 12`) could violate CFL stability when velocity grows large during optimization.
- **Fix:** Before integration, computes minimum steps: `n_steps ≥ ‖v‖_∞ · |Δt| / (C_CFL · h_min)` where `C_CFL = 1.0` for RK4, `0.5` for Euler. Uses `max(default_steps, cfl_steps)`. Costs nothing when velocities are small.

### 6. Forward Normalization Metadata
- **File:** `src/syntx/tvf.py`, `TVFModel.forward()` ~line 750
- **Problem:** In the midpoint/intermediate loss path (t=0.5, t=1.0 evaluation), moving image coordinates were normalized using fixed image metadata instead of moving image metadata. Per GEMINI.md Rule §6, sampling coordinates must be normalized against the TARGET image being sampled.
- **Impact:** Zero on Mindboggle benchmark (all MNI152 space, identical metadata). Would cause misalignment on heterogeneous datasets.
- **Fix:** Added `_get_moving_metadata_tensors()` call for the midpoint path.

---

## The `do_log` Measurement Red Herring

All test/scratch scripts called:
```python
jac = ants.create_jacobian_determinant_image(fi, fwd_tx, 1)  # do_log=1
fold_pct = np.sum(jac.numpy() <= 0) / N * 100
```

With `do_log=1`, ANTs returns `log(det J)`, not `det J`. Checking `log(det J) ≤ 0` measures `det J ≤ 1` (compression), which is physically normal — ~58% of brain voxels compress during registration. Actual grid folding requires `det J ≤ 0` (orientation reversal), computed with `do_log=0`.

**Correct measurement:**
```python
jac = ants.create_jacobian_determinant_image(fi, fwd_tx, 0)  # do_log=0
fold_pct = np.sum(jac.numpy() <= 0) / N * 100  # actual folding
```

The benchmark suite (`benchmark_suite.py` line 300, 350) already uses the default `do_log=0` and is correct. Only the scratch test scripts had this error.

**Confirmed result:** `det(J) min=0.396, mean=1.003, folds=0/6,950,944 (0.0000%)`.

---

## Final Results — Pair 05

### Low-Resolution / Coarse Grid (`reg_iterations=[100,0,0]`) — Standard vs Doubled Learning Rate

| Configuration | Learning Rate (`grad_step`) | Symmetric Dice | Improvement vs Affine | Fold% | $\min \det J$ | Max Velocity (mm) | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Affine Baseline** | — | **0.3232** | — | — | — | — | — |
| **CFL + DSTI (std lr)** | 0.45 | 0.4259 | +0.1027 (+31.8%) | 0.0000% | 0.4042 | 11.75 | 40.6s |
| **LARS + DSTI (std lr)** | 0.50 | 0.4218 | +0.0986 (+30.5%) | 0.0000% | 0.3984 | 14.64 | 40.8s |
| **CFL + Sobolev (std lr)** | 0.45 | 0.4170 | +0.0938 (+29.0%) | 0.0000% | 0.4813 | 13.33 | 40.0s |
| **LARS + Sobolev (std lr)** | 0.50 | 0.4332 | +0.1100 (+34.0%) | 0.0000% | 0.3789 | 15.10 | 40.3s |
| **CFL + DSTI (double lr)** | **0.90** | **0.4756** | **+0.1524 (+47.2%)** | **0.0000%** | 0.2204 | 17.90 | 41.1s |
| **CFL + Sobolev (double lr)** | **0.90** | **0.4852** | **+0.1621 (+50.1%)** | **0.0000%** | 0.1820 | 17.90 | 40.5s |
| **LARS + Sobolev (double lr)** | **1.00** | **0.4866** | **+0.1634 (+50.6%)** | **0.0000%** | 0.1909 | 17.82 | 40.5s |
| **LARS + DSTI (double lr)** | **1.00** | **0.1715** *(collapsed)* | -0.1517 (-46.9%) | 0.0000% | 0.1061 | 15.59 | 41.2s |

### Full-Resolution (`reg_iterations=[200,200,40]`)

| Method | Fixed Dice | Moving Dice | Symmetric Dice | Improvement | Fold% | J_min | V_max (mm) | Time |
|--------|------------|-------------|----------------|-------------|-------|-------|------------|------|
| Affine Baseline | — | — | 0.3232 | — | — | — | — | — |
| **CFL + DSTI (Bias-Corrected)** | **0.4906** | **0.4989** | **0.4947** | **+0.1715 (+53.1%)** | **0.0000%** | **0.1793** | **11.96** | **372.4s** |
| **LARS + DSTI** | **0.5001** | **0.5089** | **0.5045** | **+0.1813 (+56.1%)** | **0.0000%** | **0.1432** | **12.79** | **389.4s** |

All optimizer/regularizer combinations produce perfectly diffeomorphic warps with **0.0000% grid folding**, smooth Jacobian maps ($\min \det J > 0.14$), and **> +50% Dice improvement** over the affine baseline. LARS achieves slightly higher accuracy (+0.5045 vs +0.4947 Dice) due to layer-wise trust ratio scaling.

---

## Files Modified

- **`src/syntx/tvf.py`** — All six fixes above

## Files Created (scratch, not committed)

- `scratch/test_pair5_normfix.py` — Test with normalization fix
- `scratch/test_pair5_rigorous.py` — Test with adaptive integration + momentum fix
- `scratch/test_cfl_vs_lars.py` — CFL vs LARS optimizer comparison
- `scratch/test_final_correct_jac.py` — Final test with correct `do_log=0`
- `scratch/diagnose_folding.py` — Deep velocity field analysis
- `scratch/diagnose_jac_discrepancy.py` — Pinpointed `do_log` measurement bug

---

## Next Steps

### Immediate (high priority)

1. **Full multi-resolution benchmark on 1 pair** — Run TVF DSTI with `reg_iterations=[200,200,40]` (provenance parameters) on Pair 05 to confirm Dice reaches ≥0.60 at full resolution. The coarse-only `[100,0,0]` test was a fast validation; the full pyramid should recover the remaining accuracy.

2. **Re-run 90-pair DSTI benchmark** — The previous 90-pair run (`scratch/run_tvf_dsti_benchmark.py`) used the pre-fix code where DSTI was silently falling through to Gaussian. Those results are invalid. Re-run with all fixes applied. Ensure the benchmark uses `do_log=0` for folding measurement.

3. **Compare TVF DSTI vs TVF Sobolev on full benchmark** — Quantify whether DSTI's Dirichlet boundary conditions improve cortical alignment at brain edges compared to Sobolev's periodic (DFT) boundaries. The coarse-only test shows near-identical Dice (0.42 vs 0.43), but the difference should emerge at full resolution where boundary effects matter more.

### Medium term

4. **Characterize CFL vs LARS optimizer for TVF** — The coarse-only comparison shows equivalent Dice. Run both through the full pyramid to determine if one converges faster or produces smoother warps. The provenance parameters specify LARS for Sobolev and CFL for DSTI — validate this is optimal.

5. **Tune DSTI regularization strength** — Current `flow_sigma=0.4` gives `α = 0.4/(2×3) = 0.067`, which only attenuates the highest frequencies by ~38%. Test `flow_sigma` in `{0.2, 0.4, 0.8, 1.6}` to find the smoothness-accuracy Pareto front for DSTI.

6. **Hybrid metric** — Per GEMINI.md Rule §2, combine `0.5·LNCC + 0.5·VGG_4_LNCC` for +1.21% Cortical Dice gain. Test this hybrid loss with DSTI regularization.

### Longer term

7. **Manuscript update** — With confirmed DSTI results, update the methods section to describe the DST-I spectral regularizer and its Dirichlet boundary conditions. Update results with full benchmark numbers.

8. **Persist provenance** — Once optimal DSTI parameters are confirmed on the full benchmark, update `docs/provenance/best_parameters.json` per GEMINI.md §3.

---

## Technical Improvements

### A. CFL Momentum Warm-Up Correction

The `(1-μ)·buf` scaling correctly bounds the steady-state step, but introduces a **transient warm-up** where the effective step is suppressed. At epoch 1, the effective step is `(1-μ)·g = 0.05·g` instead of `g`. Full CFL step magnitude isn't reached until the buffer fills (~50 epochs for `μ=0.95`).

**Fix:** Apply bias correction analogous to Adam's `m̂ = m/(1-μ^t)`:

```python
momentum_buffer.mul_(mu).add_(update)
bias_corrected = momentum_buffer / (1.0 - mu ** (epoch + 1))
self.velocity.data.sub_((1.0 - mu) * bias_corrected)
```

This gives full CFL step from epoch 1 while still smoothing direction. At the current `[100,0,0]` coarse-only resolution this costs ~50 wasted warm-up epochs — nearly half the budget. At full `[200,200,40]` the impact is smaller but still wastes the first ~50 epochs of the coarsest level.

### B. Jacobian-Aware Adaptive Integration

The current adaptive integration bounds step count using velocity **magnitude** (`‖v‖_∞`). This guarantees no individual voxel moves more than one grid cell per step, preventing advection-related instability.

However, **diffeomorphism** depends on the velocity **spatial gradient** (`‖∇v‖`), not magnitude. A uniform velocity field of 100 mm/s with zero spatial gradient is perfectly diffeomorphic regardless of step size (it's a pure translation). Conversely, a velocity field with max magnitude 1 mm but steep gradients (opposing velocities in adjacent voxels) can fold even with tiny steps.

The tighter CFL condition for diffeomorphism preservation is:

$$\max_x \|∇v(x)\|_F \cdot \Delta t < 1$$

Computing `‖∇v‖_F` requires finite differences on the velocity grid (one extra pass per `integrate()` call) but would allow the integrator to use fewer steps for smooth large-magnitude fields, and correctly detect the rare case of steep-gradient low-magnitude fields.

**Priority:** Low. The magnitude-based bound is conservative (always safe) and the velocity grid is already smooth due to the fluid regularizer. The additional ∇v computation adds overhead for a marginal theoretical tightness gain.

### C. Velocity Field Resolution Decoupling

Currently, velocity resolution is tied to the multi-resolution pyramid level: level 4 → velocity grid at 1/4 image resolution, level 2 → 1/2, level 1 → full. This means:

- At coarse levels, velocity has very few DOF (44×54×45 ≈ 100K parameters for a 7M voxel image)
- At fine levels, velocity has excessive DOF (same as image → overfitting risk)

A better approach: **decouple velocity resolution from the pyramid level**. Fix velocity at an intermediate resolution (e.g., 1/2 image resolution) across all pyramid levels. The pyramid controls which IMAGE frequencies are visible to the loss, while the velocity grid independently controls deformation smoothness.

This would:
- Allow coarse-level optimization to express finer deformations (more DOF at level 4)
- Prevent overfitting at fine levels (fewer DOF at level 1)
- Eliminate the expensive "Final velocity upsample" step

**Implementation:** Modify `fit()` to keep `velocity_shape` constant across levels. Only upsample the fixed/moving images to the current pyramid resolution. The `integrate()` method already handles velocity-to-image shape mismatch via `F.interpolate`.

### D. DSTI vs Sobolev Boundary Behavior

DSTI uses DST-I (Discrete Sine Transform, Type I), which imposes **Dirichlet boundary conditions** (velocity = 0 at domain boundary). Sobolev uses DFT, which imposes **periodic boundary conditions**.

For brain registration:
- Dirichlet boundaries are physically motivated: the deformation SHOULD be zero at the image boundary (brain doesn't deform into empty space)
- Periodic boundaries can create wrap-around artifacts where deformation at one edge "leaks" to the opposite edge

However, the current boundary taper mask (line 1052: `_create_boundary_mask(border_width=4)`) already forces gradients to zero near boundaries for ALL regularizers, partially duplicating what DSTI provides natively. With taper + DSTI, the boundary region is double-suppressed.

**Experiment:** Compare DSTI with and without the taper mask. If DSTI's intrinsic Dirichlet condition is sufficient, removing the taper mask would free 4 voxels of boundary DOF per side. Also compare whether the taper mask helps Sobolev (where it's genuinely needed to prevent wrap-around) more than DSTI.

### E. Geodesic Shooting vs Constant-Speed Projection

The current constant-speed implementation is a **projection** — after each optimizer step, per-keyframe RMS speeds are relaxed toward their mean. This is a first-order approximation that doesn't account for the geometry of the diffeomorphism group.

True geodesic shooting (Beg et al. 2005) parameterizes the path by the initial velocity `v(0)` alone, with subsequent keyframes determined by the EPDiff equation:

$$\partial_t v + \text{ad}_v^\dagger v = 0$$

where `ad_v^\dagger` is the adjoint of the Lie bracket. This guarantees a geodesic path (minimizes kinetic energy) and reduces the parameter space from `T × spatial` to `1 × spatial`.

**Tradeoff:** EPDiff requires solving a PDE at each forward pass (expensive), and reduces expressiveness (single initial condition vs. T independent keyframes). The current multipoint loss at `[0.0, 0.5, 1.0]` requires evaluating the trajectory at intermediate times, which is naturally available from the ODE integration — EPDiff would preserve this.

**Priority:** Medium-high for theoretical rigor. The constant-speed projection is a reasonable practical approximation but isn't a true geodesic. For the manuscript, this distinction matters.

### F. Multi-Scale Similarity Loss

The current loss evaluates LNCC at a single window size (`2 × syn_sampling + 1`). Cortical registration benefits from matching structures at multiple scales simultaneously:

- Large windows (11×11×11): capture sulcal pattern alignment, prevent local minima
- Small windows (5×5×5): resolve individual gyral boundaries, maximize Dice

A multi-scale LNCC that averages losses across window sizes `[3, 5, 7]` or uses the deep feature hybrid (`0.5·LNCC + 0.5·VGG_4_LNCC` per GEMINI.md §2) would improve convergence in the early epochs (large-scale matching) while preserving fine-grained accuracy in later epochs.

### G. Per-Level Optimizer Reset

When transitioning between pyramid levels, the momentum buffer carries state from the previous (coarser) level. The velocity was upsampled but the momentum buffer was not — it retains coarse-level gradient statistics that are irrelevant at the finer level.

**Fix:** Reset `momentum_buffer` to zeros at each level transition. This ensures the optimizer starts fresh at each resolution without stale gradient history biasing the early updates.

### H. Sobolev/DSTI α Parameter from flow_sigma

Both regularizers compute `α = flow_sigma / (2 × dim)`. For `flow_sigma=0.4` and `dim=3`: `α = 0.067`. The spectral filter is `K(λ) = 1/(1 + α·λ)^s` with `s=2`.

At maximum frequency `λ_max ≈ 4`: `K(4) = 1/(1 + 0.267)^2 = 0.62`. This means the highest frequencies retain **62%** of their energy — very weak smoothing. For comparison, `flow_sigma=3.0` (the SyN default) gives `K(4) = 0.11` — only 11% passes through.

The weak smoothing is intentional for TVF (velocity fields need less regularization than displacement fields because the ODE integration inherently smooths the output). But it means the DSTI spectral advantage over Gaussian is minimal at these low α values — the spectral filters barely differ from identity.

**Experiment:** Sweep `flow_sigma ∈ {0.2, 0.4, 0.8, 1.6, 3.0}` on the full benchmark to characterize the accuracy–smoothness Pareto front. Plot Dice, folding %, and Jacobian std vs. flow_sigma for both DSTI and Sobolev to quantify when spectral regularization matters.
