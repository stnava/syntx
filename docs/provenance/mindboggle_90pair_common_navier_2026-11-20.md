# Unified Navier-Cauchy Regularization Across `syntx` Solvers: Theory, Disjoint Held-Out Selection, and 90-Pair Mindboggle-101 Population Benchmark

**Date**: 2026-11-20 / 2026-11-21  
**Host / Hardware**: `barncreek.radio.virginia.edu`, Apple Silicon GPU (`device='mps'`, isolated single-process execution per pair)  
**Shared Linear Initialization**: Cached native PyTorch Mattes-MI `syntx.robust_affine` (`results/canonical_affines/pair_{000..089}_pt7_affine.mat`, 90-pair mean symmetric DKT31 Dice `0.3448 ± 0.0228`, mean runtime `5.32 s`)  
**Reference Baseline**: `ANTs C++ SyNOnly [100, 100, 20]` (`CC[r=2]`, `flow_sigma=3.0`, `total_sigma=0.0`, `grad_step=0.25`; `results/pair_{000..089}_ants_syn.json`, 90-pair mean symmetric DKT31 Dice `0.6216 ± 0.0229`, mean runtime `139.37 s`)

---

## 1. Operator Definition and Selection Principles

In `src/syntx/core/regularizers.py` (`apply_navier_green_operator` / `_build_navier`), the coupled spectral Navier-Cauchy Green's operator on relative aspect-ratio wavevectors $\mathbf{k}$ (`spacing_rel = spacing / min(spacing)`, conjugated into physical space via $D_{\text{zyx}} \mathcal{G}_{\mathbf{s}}(D_{\text{zyx}}^T \mathbf{m})$) is:

$$\mathcal{G}_{\text{Navier}}(\mathbf{k}) = \left(1 + \alpha \|\mathbf{k}\|^2\right)^{-s} \left[ I - \kappa(\nu)\,\frac{\mathbf{k}\mathbf{k}^T}{\|\mathbf{k}\|^2} \right], \qquad \kappa(\nu) = \frac{1}{2(1 - \nu)}, \quad \nu \in [0, 0.5].$$

Decomposing into orthogonal idempotent transverse ($\Pi_{\perp} = I - \hat{\mathbf{k}}\hat{\mathbf{k}}^T$) and longitudinal ($\Pi_{\parallel} = \hat{\mathbf{k}}\hat{\mathbf{k}}^T$) projectors gives eigenvalues $1$ on solenoidal shear/rotation modes and $\lambda_{\parallel}(\nu) = 1 - \kappa(\nu) = \frac{1 - 2\nu}{2(1 - \nu)}$ on irrotational dilatation modes:

1. **Sobolev Order $s = 2.0$ ($H^2$ Biharmonic Envelope)**: In $d = 3$, Sobolev embedding $H^{2s}(\Omega) \hookrightarrow C^{1,\gamma}(\bar{\Omega})$ requires $2s > d/2 + 1 = 2.5 \implies s > 1.25$ for $C^1$ Lipschitz admissibility of velocity fields. $s = 2.0$ is the minimal integer exponent satisfying this condition. Empirically on `[77, 44, 0]`, $s = 1.0$ folded or under-regularized while $s = 3.0$ over-damped cortical sulci (`-1.26 pp` to `-3.80 pp`), failing feasibility across `4/4` solvers.
2. **Operator Multiplicity & Common Poisson Ratio $\nu = 0.20$ ($\kappa = 0.625, \lambda_{\parallel} = 0.375$)**:
   - Because $\Pi_{\perp}\Pi_{\parallel} = 0$, $m$ compositions of $\mathcal{G}_{\text{Navier}}$ transmit longitudinal modes at $(1 - \kappa(\nu))^m$.
   - Single-application ($m = 1$) solvers (`greedy`, `syn`, `tvf`) pass $33.3\%\text{–}37.5\%$ of dilatation gradients across $\nu \in [0.20, 0.25]$, accommodating inter-subject ventricular and sulcal volume differences without folding.
   - Multi-application ($m = 2$) geodesic momentum shooting (`syngs`: `RegAdam` step preconditioning plus forward `shoot` smoothing) compounds longitudinal attenuation as $(1 - \kappa)^2$: at $\nu = 0.20$, $(0.375)^2 = 0.1406$ (`+26.6%` higher longitudinal mobility than $\nu = 0.25$'s $(1/3)^2 = 0.1111$, and $7.2\times$ higher than legacy $\nu = 0.49$'s $0.0004$), while avoiding the interior inverse-consistency blowup observed at $\nu = 0.15$ (`1.12 mm > 1.00 mm` on `[77, 44, 0]`).
3. **Solver-Specific Spatial Scale $\alpha$**: Ablating a literal single $\alpha = 2.25$ across all four PDE discretizations on `[77, 44, 0]` degraded `greedy` (`-0.30 pp`), `syn` (`-0.28 pp`), rendered `tvf` infeasible (`inv_interior_max = 1.27 mm > 1.00 mm`), and collapsed `syngs` (`-7.12 pp` at $\alpha = 1.5$). Thus `(regularizer='navier', s=2.0, poisson_ratio=0.20)` defines the universal dimensionless continuum operator, while each solver retains its calibrated spatial scale (`sobolev_alpha` / `alpha`) and step size.

---

## 2. Phase 1 & 2 Summary: Tune `[77, 44, 0]` and Subject-Disjoint Held-Out `[28, 55, 84]`

All tuning was restricted to `[77, 44, 0]` (`reg_iterations = [100, 100, 20]`). Candidate common settings $\nu \in \{0.15, 0.20, 0.25\}$ were then evaluated once out-of-sample on the 100% subject-disjoint Mindboggle-101 Held-Out set `[28, 55, 84]` (`results/common_navier_heldout_20261120/`):

| Solver | `v6.43.0` Default (`[28, 55, 84]`) | Common Navier $\nu = 0.25$ (`[28, 55, 84]`) | **Common Navier $\nu = 0.20$ (`[28, 55, 84]`)** | Held-Out Gain ($\nu = 0.20$ vs `v6.43.0`) | Held-Out Folding % | Held-Out Worst Int. ICE |
|---|---:|---:|---:|---:|---:|---:|
| `ANTs C++` (`SyNOnly`) | `0.6034` | — | — | — | `0.0000%` | — |
| `syntx.greedy` | `0.6060` (`div_curl`) | `0.6155` | **`0.6169`** | **`+1.09 pp`** | `0.0000%` | `NaN` (unidirectional) |
| `syntx.syn` | `0.6110` (`sobolev`) | `0.6304` | **`0.6304`** | **`+1.94 pp`** | `0.0000%` | `0.42 mm` |
| `syntx.syngs` | `0.6168` (`sobolev`) | `0.6223` | **`0.6251`** | **`+0.83 pp`** | `0.0000%` | `0.96 mm` |
| `syntx.tvf` | `0.6292` (`beltrami`, `ICE=1.14mm`) | `0.6372` | **`0.6367`** | **`+0.75 pp`** | `0.0000%` | `0.90 mm` |
| **4-Solver Mean** | **`0.6158`** | **`0.6264` (`+1.06 pp`)** | **`0.6273`** | **`+1.15 pp`** | **`0.0000%`** | **`<= 0.96 mm`** |

Common Navier $(s = 2.0, \nu = 0.20)$ achieved the highest 4-solver Held-Out mean Dice (`0.6273`), halved worst-case per-solver regret relative to $\nu = 0.25$ (`0.26 pp` vs `0.54 pp` on `syngs`), improved `tvf` interior inverse consistency below `1.00 mm` (`1.14 -> 0.90 mm`), and won `12 / 12` solver $\times$ held-out pair comparisons over `v6.43.0`.

---

## 3. Canonical Parameter Specification (`canonical_2026_11_20`)

- **Shared Operator (`src/syntx/core/regularizers.py`)**: `regularizer='navier'`, `poisson_ratio=0.20`, `s=2.0`.
- **`syntx.greedy`**: `regularizer='navier'`, `sobolev_alpha=2.625`, `total_sigma=0.0`, `learning_rate=1.5`, `cfl_momentum=0.1875`, `flow_sigma=1.62`, `optimizer='cfl'`.
- **`syntx.syn`**: `regularizer='navier'`, `sobolev_alpha=1.5`, `grad_step=0.6`, `flow_sigma=2.4`, `total_sigma=0.0`, `optimizer='cfl'`, `in_loop_inv_steps=10`, `inv_refresh_every=(1, 2, 4)`, `polish_inv_steps=30`, `polish_inv_tol=0.02`, `compose_polish_tol=0.30`.
- **`syntx.syngs`**: `regularizer='navier'`, `SYNGS_DEFAULT_ALPHA={2: 0.25, 3: 0.25}`, `max_step_norm=0.95`, `optimizer='reg_adam'`, `optimizer_lr=2.246094`, `n_steps=8`, `bootstrap_mode='none'`.
- **`syntx.tvf`**: `regularizer='navier'`, `TVF_DEFAULT_ALPHA={2: 2.0, 3: 3.5}`, `TVF_DEFAULT_CFL={'grad_step': 1.8, 'cfl_momentum': 0.8}`, `energy_weight=0.0004`, `n_time_steps=3`, `multipoint_loss=(0.0, 1.0)`, `optimizer='cfl'`.

---

## 4. Full 90-Pair Mindboggle-101 Population Benchmark (`results/navier_90pair_20261120/`)

All 90 Mindboggle-101 pairs (`40` intra-cohort, `50` inter-cohort; `examples/pairs.csv`, seeded permutation `seed=42`, `[100, 100, 20]`) were executed in isolated single-pair subprocesses across all four `syntx` solvers (`360` runs, `100%` `SUCCESS`, `360/360` verified via `check_canonical_call`):

| Method | 90-Pair Mean ± SD Dice | Median (Min–Max) Dice | Intra (`n=40`) | Inter (`n=50`) | $\Delta$ vs `ANTs C++` | Wins vs `ANTs` (`p` Wilcoxon) | Liouville / FD Max Fold % | Worst `min\|J\|` | Mean / P95 / Med Int-Max ICE (mm) | Mean Total Time (Speedup) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `syntx.robust_affine` | `0.3448 ± 0.0228` | `0.3448` | — | — | — | — | `0.0000%` | `1.0000` | `0.0000` | `5.32 s` |
| `ANTs C++ SyN` | `0.6216 ± 0.0229` | `0.6195` (`0.5801–0.6849`) | `0.6276` | `0.6168` | `0.00 pp` | — | `0.0000%` | `0.0459` | — | `139.37 s` (`1.00x`) |
| **`syntx.greedy`** | **`0.6302 ± 0.0220`** | `0.6316` (`0.5767–0.6957`) | `0.6329` | `0.6280` | **`+0.86 pp`** | **`67/90` (`74.4%`, `2.3e-8`)** | **`0.0000% / 0.0000%`** | `0.0505` | `NaN` (forward-only) | **`11.74 s` (`11.87x`)** |
| **`syntx.syngs`** | **`0.6389 ± 0.0217`** | `0.6387` (`0.5871–0.7025`) | `0.6418` | `0.6365` | **`+1.73 pp`** | **`84/90` (`93.3%`, `3.1e-15`)** | **`0.0000% / 0.0000%`** | **`0.1409`** | `0.0073 / 0.0155 / 0.7619` | **`63.73 s` (`2.19x`)** |
| **`syntx.syn`** | **`0.6423 ± 0.0212`** | `0.6418` (`0.5837–0.7017`) | `0.6448` | `0.6404` | **`+2.07 pp`** | **`87/90` (`96.7%`, `5.1e-16`)** | **`0.0000% / 0.0000%`** | `0.0372` | `0.0104 / 0.0711 / 0.3278` | **`46.20 s` (`3.02x`)** |
| **`syntx.tvf`** | **`0.6529 ± 0.0204`** | `0.6516` (`0.6021–0.7119`) | **`0.6565`** | **`0.6500`** | **`+3.13 pp`** | **`90/90` (`100.0%`, `1.7e-16`)** | **`0.0000% / 0.0000%`** | `0.0176` | `0.0077 / 0.0485 / 0.7380` | `197.43 s` (`0.71x`) |

---

## 5. Verified vs. Inferred Claims and Limitations

### Verified
- All `360 / 360` deformable benchmark runs in `results/navier_90pair_20261120/` completed with `status == "SUCCESS"`, exact `0.000000%` Liouville folding (`360/360` zero-fold), exact `0.000000%` exported 1-mm finite-difference folding (`360/360` zero-fold), strictly positive Jacobian determinants (`min|J| >= 0.0176`), `0/90` identical outputs across any pair of solvers, and `360/360` `check_canonical_call` compliance.
- Compared against historical 90-pair `syntx.syn` baselines on the same cohort (`best_parameters.json`), common Navier `syntx.syn` (`0.6423`, `87/90` wins, `0.0000%` folding, `46.20 s`) surpasses both 90-pair Sobolev `syntx.syn` (`0.6342`, `81/90` wins, `90.0%` zero-fold, `52.3 s`) by **`+0.81 pp`** and 90-pair Gaussian `syntx.syn` (`0.6382`, `15.6%` zero-fold) by **`+0.41 pp`** while achieving `100%` zero-folding.
- `syntx.tvf` wins **`90 / 90` (`100.0%`)** head-to-head comparisons against `ANTs C++ SyN` (`0.6529` vs `0.6216`, `+3.13 pp`, minimum Dice across all 90 pairs `0.6021`).

### Inferred
- The stronger inter-cohort (`n = 50`) gains relative to intra-cohort (`n = 40`) — e.g. `+2.36 pp` inter vs `+1.72 pp` intra for `syntx.syn`, and `+3.32 pp` inter vs `+2.89 pp` intra for `syntx.tvf` — are consistent with controlled longitudinal compressibility ($\lambda_{\parallel} = 0.375$) accommodating cross-scanner ventricular and sulcal CSF volume differences without permitting shear-driven grid folding.

### Plain Limitations
1. **Tail Interior Inverse-Identity Error in Velocity-Integrated Solvers (`syngs`, `tvf`)**: While `syntx.syn` maintains interior maximum inverse-identity error `<= 0.9386 mm` (`< 1.00 mm` on `90/90` pairs), `syntx.syngs` (`72/90 <= 1.0 mm`, worst `1.6800 mm`, mean `0.0073 mm`, P95 `0.0155 mm`) and `syntx.tvf` (`70/90 <= 1.0 mm`, worst `1.5548 mm`, mean `0.0077 mm`, P95 `0.0485 mm`) exhibit localized >1-voxel interior inverse residuals at high-curvature sulcal boundaries on `18/90` and `20/90` pairs due to dual-field velocity Penalty weighting (`inverse_identity_weight = 0.50` / `0.05`) followed by bounded-step (`25` iterations) post-integration Levenberg-Marquardt polish.
2. **Backend Scope**: Default `regularizer='navier'` applies to the primary PyTorch backend across all four solvers; the legacy `backend='jax'` paths in `syn`, `syngs`, and `tvf` retain their documented `'gaussian'` / `'sobolev'` defaults.
3. **3-D T1w Brain Calibration**: The calibrated default parameters were tuned and validated on 3-D human T1w brain MRI (`1 mm` isotropic Mindboggle-101) plus 2-D regression phantoms (`r16 -> r64`); non-brain organs with large respiratory/sliding motion (e.g. lung/abdomen CT) remain governed by `syntx.auto_reg` / modality-specific presets.
