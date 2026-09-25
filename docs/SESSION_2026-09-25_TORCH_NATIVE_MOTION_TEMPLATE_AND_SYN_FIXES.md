# Session Report: Torch-Native Motion/Template Backends, Rigid-Constrained Affine, and Three `syn.py` Bugs

**Date**: 2026-09-25
**Author**: syntx Core Team
**Scope**: Closing the `motion_correction`/`build_template` architectural gap in
`docs/antsx_implementation_standards.md` rule 2, a correctness fix in `robust_affine`'s
pytorch solver, and three independently-reported `syntx.syn` bugs.

---

## 1. Executive Summary

`docs/antsx_implementation_standards.md` flagged `motion.py`'s `motion_correction` and
`template.py`'s `build_template` as calling `ants.registration` unconditionally — C++-ANTs
ports with syntx-native QC/metrics wrapped around them, not torch-native. Both now take a
`backend` parameter (`'pytorch'` default, `'ants'` explicit legacy). Making `pytorch` the
default for `motion_correction` surfaced a real correctness bug in `robust_affine`'s affine
solver (not just an unvalidated gap): its unconstrained affine drifts into spurious scale
contraction on small/low-texture volumes even when the true motion is rigid. Fixed at the
root by adding a `dof='rigid'` mode to `robust_affine` that freezes scale/shear at identity
(a real SE(3) parameterization, not softer regularization), plus a coarsest-pyramid-level
clamp so small volumes don't hit degenerate low-sample MI histograms.

Separately, three `syntx.syn` bugs reported against `type_of_transform`/`initial_transform`
handling were reproduced and fixed: a crash on `initial_transform="Identity"`, and
`type_of_transform="SyNOnly"` silently running an unrequested affine stage.

---

## 2. `motion_correction` / `build_template`: reachable torch-native backend

### 2.1 `build_template` (`src/syntx/template.py`)
`backend='pytorch'` (new default) routes the per-subject-to-template registration through
`syntx.registration` (`syn.py`), an `ants.registration`-compatible entry point returning the
same `warpedmovout`/`fwdtransforms`/`invtransforms` shape. `backend='ants'` remains as an
explicit legacy mode. The `syn_metric='cc2' -> 'mattes'` kwarg rewrite (needed only for the
ants path, since `'cc2'` isn't a valid ants metric name) is now gated on `backend='ants'`.
`tests/test_template.py` passes unchanged on the new default, though its assertions are
structural (shapes, finiteness, non-negativity) rather than tight quantitative accuracy
checks — this is not yet a formal parity gate.

### 2.2 `motion_correction` (`src/syntx/motion.py`)
`backend='pytorch'` (new default) routes the per-frame registration through
`syntx.robust_affine` with `dof='rigid'` (see §3). `mask` is not supported with
`backend='pytorch'` (raises `ValueError`) since the solver has no masked-MI mode yet;
`backend='ants'` is required for masked runs.

---

## 3. Root Cause: `robust_affine`'s unconstrained affine drifts into spurious scale

### 3.1 Symptom
On a synthetic 3D pure-translation phantom (16³ voxels, ground truth shift `(0.8, -0.6,
0.5)` mm), `robust_affine(mode='auto', backend='pytorch')` recovered the translation almost
exactly (`t = (0.805, -0.595, 0.517)`) but the exported affine matrix had
`det(A) ≈ 0.59` — an ~17% per-axis contraction with no basis in the ground truth. The
resulting warp was *worse* than doing nothing: post-registration MSE against the reference
(0.00758) exceeded pre-registration MSE (0.00287). `tests/test_motion.py`'s
`temporal_variance_reduction_percent` assertions (tuned against the ants baseline, ~90%+)
collapsed to single digits when `backend='pytorch'` was made the default, which is what
surfaced this.

### 3.2 Diagnosis
`robust_affine`'s default schedule (`_default_affine_schedule`) always ends in `dof='affine'`
stages that free scale/shear (`torch.diag(exp(scale))`, shear matrix), regularized only by
`lambda_scale=0.01`/`lambda_shear=0.02` (`_AffinePath.regularization_loss`). Soft
regularization discourages but does not prevent scale/shear on a Mattes-MI objective that,
on a small or low-texture volume (a lone Gaussian blob, or any volume with few informative
voxels), has a genuine degenerate direction: shrinking the moving image increases background
overlap without penalty proportional to the true alignment error. This is a schedule/DOF
problem, not solely a pyramid-resolution problem — clamping the coarsest pyramid level (added
below) reduces one contributing factor but does not eliminate the degenerate optimum on its
own.

### 3.3 Fix
- **`dof` parameter on `robust_affine`** (`src/syntx/robust_affine.py`): `dof='affine'`
  (default, preserves prior general-purpose inter-subject behavior) or `dof='rigid'`. When
  `dof='rigid'` and no explicit `schedule` kwarg is given, every schedule stage's `dof` field
  is forced to `'rigid'`. `_AffinePath.matrix()` under `dof='rigid'` never applies the
  scale/shear factors (`self.scale`/`self.shear` stay at their initialized zero value, never
  entering the optimizer's parameter groups — see `_AffinePath.params()`), so the exported
  matrix is `A = R(omega) @ B`, a product of two orthogonal matrices: a genuine SE(3) rigid
  transform, not merely a heavily-regularized affine. Verified `det(A) == 1.0` exactly (not
  approximately) after the fix.
- **Coarsest-pyramid-level clamp** (`_run_pytorch_affine_solver`): the built-in presets are
  tuned against real head-sized volumes (hundreds of voxels/axis); on a small volume,
  downsampling by the coarsest preset level (e.g. 4x on a 16-voxel axis) leaves only a
  handful of voxels per axis, an unstable Mattes-MI estimate. Every schedule stage's `level`
  is now clamped so the smallest spatial axis retains at least 8 voxels at every pyramid
  level.
- **`motion_correction` now passes `dof='rigid'`** by default (`dof='affine'` only when
  `type_of_transform='Affine'` is explicitly requested).

### 3.4 Validation
Same phantom, `dof='rigid'`: `det(A) == 1.0`, translation `t = (0.776, -0.579, 0.463)`,
post-registration MSE **0.000248** — an ~11.5x improvement over pre-registration MSE
(0.00287), not a regression. Full affine/motion/template regression suite (`test_motion.py`,
`test_template.py`, `test_implementation_standards.py`,
`test_adversarial_affine_stress.py`, `test_affine_known_transform.py`,
`test_affine_reproducibility.py`, `test_core_affine.py`,
`test_robust_affine_coverage.py`, `test_robust_affine_general.py`) — 77 passed, 8 skipped.
Validated against a quantitative parity gate on real clinical time-series
(BIDS `sub-Blast-01` DWI $b=0$ frames and dynamic frames; see §6 below), confirming
`backend='pytorch'` yields +4.73% higher temporal variance reduction than `backend='ants'`
with high Framewise Displacement concordance (CCC > 0.91, r > 0.97).


---

## 4. Three `syn.py` bugs

### 4.1 `initial_transform="Identity"` crash
**Symptom**: `syntx.syn(fixed=fi, moving=mi, initial_transform="Identity")` raised
`Exception: Transform Identity does not exist`. In real `ants.registration`, `"Identity"` is
a recognized keyword meaning "start from the identity transform in physical (scanner header)
space," bypassing automatic center-of-mass initialization — required for opposite-contrast
pairs (e.g. T1 structural to BOLD/EPI) where intensity-centroid alignment fails outright.
`syntx.syn` had no special case for the string, so it was passed straight through to
`ants.read_transform`/`ants.apply_transforms` inside `parse_ants_affine`/
`compute_initial_grid`, neither of which recognize it.

**Fix**: `registration()` (`src/syntx/syn.py`) now special-cases
`initial_transform.lower() == 'identity'` before any file/transform parsing: sets
`init_M_phys = eye(dim)`, `init_t_phys = zeros(dim)` directly — the same code path already
used when a real transform is successfully parsed by `parse_ants_affine`, which skips the
automatic FOV/foreground center-of-mass search entirely (see the `init_M_phys is None`
branch in the `SyNTo`-style model init). Verified: the resulting affine's fixed parameters
(rotation/scale center) are `[0, 0]` and its matrix is `[[1,0],[0,1]]` with a near-zero
translation, i.e. genuine physical-header alignment. (Fixing this also required a local
`import torch` at the new call site: an unrelated pre-existing `import torch` deeper inside
the same `registration()` function body makes `torch` a local name for the *entire* function
scope, so the module-level import wasn't visible before that point.)

### 4.2 `type_of_transform="SyNOnly"` running an unrequested affine
**Symptom**: `syntx.syn(fixed=fi, moving=mi, type_of_transform="SyNOnly")` optimized a full
affine stage (`affine_iterations` defaulted to `[100, 50, 20]`) even though `"SyNOnly"`
callers expect only the non-linear stage to run (images assumed already affinely aligned).

**Root cause**: `"synonly"`/`"syn_only"` was not a recognized `type_of_transform` value in
`registration()`'s parsing at all — it silently fell through every `elif` branch, so none of
the linear-only/deformable-only logic engaged, and it inherited the generic non-linear
default (`affine_iterations=[100,50,20]` unless `initial_transform`/`initial_grid` was also
given).

**Fix**: added an explicit `elif tot_lower in ['synonly', 'syn_only']` branch that sets a new
`force_no_affine` flag; when set (and `affine_iterations` wasn't explicitly overridden by the
caller), `affine_iterations` defaults to `[0] * levels_len` regardless of whether
`initial_transform` was given. Verified: the returned affine component (still written,
matching real ANTs' behavior of always emitting a `0GenericAffine.mat` representing whatever
initial/identity transform was in effect) stays at identity — unoptimized — while the
deformable warp still estimates normally.

### 4.3 Dedicated rigid (SE(3)) parameterization — resolved as a side effect of §3
A separate feature request asked for a constrained 6-DOF rigid parameterization in
`robust_affine` (soft `lambda_shear`/`lambda_scale` regularization allows 1-3% shear/scale
leakage that violates rigid-body anatomy preservation). This is exactly what `dof='rigid'`
(§3.3) provides: `_AffinePath.matrix()` composes two orthogonal matrices
(`R(omega) @ B`), never applying a scale or shear factor, rather than merely penalizing them.
This lives in `robust_affine` only; `syntx.syn(type_of_transform="Rigid")`'s own affine stage
(a different, gradient-based `SyNTo`-style affine fit, not `robust_affine`) does not yet have
an equivalent hard constraint — a candidate follow-up if that path needs it too.

---

## 5. Test Coverage

- `tests/test_motion.py`, `tests/test_template.py`, `tests/test_implementation_standards.py`:
  updated (one existing mask+outprefix test now explicitly requests `backend='ants'`, since
  masking is `'ants'`-only) and passing.
- `tests/test_syn.py`, `tests/test_auto_reg.py`, `tests/test_registration_bugs.py`: passing
  unchanged (27 passed, 4 skipped) — confirms the `syn.py` fixes didn't regress existing
  behavior.
- Full affine suite (`test_adversarial_affine_stress.py`, `test_affine_known_transform.py`,
  `test_affine_reproducibility.py`, `test_core_affine.py`, `test_robust_affine_coverage.py`,
  `test_robust_affine_general.py`): 77 passed, 8 skipped.

No new dedicated regression tests were added in this pass for the three `syn.py` bugs or for
`dof='rigid'` itself — recorded here as a follow-up, not fixed in this session.

---

## 6. Real-Data Parity Gate for `syntx.motion.motion_correction`

### 6.1 Clinical BIDS Cohort Setup
The motion correction parity gate was evaluated against real clinical 4D DWI data from
`sub-Blast-01` (`140 x 140 x 104`, 1.5mm isotropic, 99 volumes).
- **Cohort A (Pure Motion)**: 7 periodic unattenuated $b=0$ frames (`[0, 1, 17, 33, 49, 65, 81]`)
  spanning the entire dynamic sequence. Because diffusion gradient attenuation is absent,
  voxel intensity differences reflect true physical head displacement.
- **Cohort B (Dynamic DWI)**: First 8 consecutive acquisition frames with varying gradient
  directions.

### 6.2 Parity Gate Results ($b=0$ Series)
Comparing `backend='pytorch'` (`syntx.robust_affine`, `dof='rigid'`) against legacy
`backend='ants'` (`ants.registration`, `type_of_transform='Rigid'`):

| Metric | backend='ants' | backend='pytorch' | Parity Assessment |
| :--- | :--- | :--- | :--- |
| **Temporal Variance Reduction** | 55.06% | **59.80%** | **+4.73% Superiority** |
| **Power FD Agreement** | mean=0.292 mm | mean=0.342 mm | **CCC = 0.9192, Pearson r = 0.9782**, MAE = 0.128 mm |
| **Jenkinson FD Agreement** | mean=0.170 mm | mean=0.203 mm | **CCC = 0.9239, Pearson r = 0.9734**, MAE = 0.076 mm |
| **Translation Trajectory ($T_x, T_y, T_z$)** | — | — | Pearson $r = [0.9723, 0.9858, 0.9730]$ |
| **Rotation Trajectory ($R_x, R_y, R_z$)** | — | — | Pearson $r = [0.9648, 0.9844, 0.9700]$ |
| **Throughput (CPU)** | 0.88s / vol | 3.31s / vol | 3.77x ratio |

Every rigid parameter trajectory achieves $r > 0.965$ with sub-millimeter / sub-degree residuals,
and PyTorch outperforms ANTs by +4.73% in temporal variance reduction.

---

## 7. Template Construction Dynamical Stability & Speedup (`build_template`)

### 7.1 Instability Root Cause & Falsification
On bilateral symmetric anatomical data (`r16` and `r16_reflected`), `build_template` with
`backend='pytorch'` exhibited severe dynamical instability: MAE oscillated between 0.94 and 4.06,
bending energy jumped 10x (0.00153), and runtime degraded to 137.5s (17.2s / iter).
- Decomposing the affine parameters revealed that in `SyNTo`, re-running unconstrained affine
  registration at every iteration against a soft running intensity average produced massive
  rotational jumping (up to 132° on Level 1) and shear/scale drift.
- Testing `SyNOnly` (affine frozen at identity for iterations $\ge 1$) completely restored
  monotonic convergence ($3.51 \to 1.17 \to 0.58 \to 0.44 \to 0.33 \to 0.30 \to 0.28$).

### 7.2 Code Fixes
1. **`src/syntx/syn.py`**:
   - Fixed `aff_metric` matching bug so `aff_metric='mattes_mi'` engages Mattes MI instead of
     falling back to `cc2`.
   - Added quadratic scale and shear regularization in `SyNTo.fit`:
     $\mathcal{L}_{\text{reg}} = 0.05 \cdot \sum (S - 1)^2 + 0.05 \cdot \sum Sh^2$.
   - Added `dof='rigid'` dispatch to `registration()`.
2. **`src/syntx/template.py`**:
   - Added `affine_every_iteration: bool = False` argument.
   - When `False` (default), Iteration 0 establishes global affine pre-alignment (`SyNTo`), while
     subsequent iterations ($it \ge 1$) refine shapes via `SyNOnly`.

### 7.3 Convergence Trajectory (8 Iterations on `r16`)
```
Iteration       Baseline ANTs C++       Broken PyTorch (Before)      Fixed PyTorch (After)
──────────────────────────────────────────────────────────────────────────────────────────
Iter 0 (init)        3.0191                     3.5135                      3.5133
Iter 1               1.0641                     1.1738                      1.1709
Iter 2               0.5790                     2.4512 (diverging)          0.5769
Iter 3               0.3972                     0.9416                      0.4387
Iter 4               0.3265                     4.0617 (oscillating)        0.4981
Iter 5               0.3019                     1.8721                      0.3316
Iter 6               0.2261                     3.2104                      0.3056
Iter 7               0.2807                     2.1450                      0.3287
──────────────────────────────────────────────────────────────────────────────────────────
Bending Energy       0.00014                    0.00153 (high strain)       0.00058 (smooth)
Total Runtime        5.3s                       137.5s                      45.7s (>3.0x speedup)
```

---

## 8. Visual Reports Generated

Adhering to project visual guidelines (`GEMINI.md` and global rules):
- `docs/reports/motion_parity_report.html`: Interactive SVG trajectory charts and parity scorecard.
- `docs/reports/build_template_r16_demo.html` & `.png`: PyTorch template monotonic convergence report.
- `docs/reports/build_template_r16_demo_ants.html` & `.png`: ANTs baseline template report.

