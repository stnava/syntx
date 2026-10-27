# Proposal: one joint optimisation of head motion and acquisition effects over the whole series

Status (2026-10-27, v6.25.0): sections 1-3 implemented as `engine="joint"` (`motion_design._joint`; solver:
`frame_levels`, `init_pose`, `schedule_mode="refine"`, per-reference intensity range, odd-only gauge projection);
decisions of section 4 taken as recorded in section 10; section 5 gates run (phantoms in `docs/MOTION_DESIGN.md`,
real data in section 7 and `benchmarks/motion_cohort_gate.py`). Sections 8 (BOLD field term) and 9 (motion-specific
metric) are not implemented. Follows `docs/MOTION_COMPLETE_MODEL_REVIEW.md` (2026-10-27), which found that the
`unified` engine solves each contrast level separately and gives the free per-frame motion term the residual of
everything the local-difference estimators cannot explain.

## 1. Objective

One objective over all frames `t = 0..T-1`:

```
J(m, beta, C) = sum_t  L_t( I_t o T(m_t + D_t beta) o A(C g_t),  R_t )        image term, per frame
              + lambda sum_t rho( W (m_{t-1} - 2 m_t + m_{t+1}) )              prior on head motion
              + mu sum_j || beta_{j+1} - beta_j ||^2                           drift smoothness of beta
```

- `m_t in R^6` (mm, rad about the shared centre `c0`): head motion, written to the transforms, source of `fd`.
- `D_t beta`: acquisition-locked pose effect that is REGISTRATION BIAS (contrast). Seen by the metric, removed from
  the written pose, never applied to the data.
- `A(C g_t)`: acquisition-locked GEOMETRY (eddy affine along phase encode, tied to `g`), applied to the data and
  excluded from `fd`. This is the existing `pe_affine_covariates` term, unchanged.
- `R_t`: a reference of frame `t`'s own contrast (its level mean; later an SH prediction at its direction).
- `rho`: a sub-quadratic penalty (Charbonnier, `delta` in mm) so a real jump costs linearly and is not smeared.
  `W = diag(1,1,1, 50,50,50)` puts rotations in mm at 50 mm radius.
- `beta_j`: the acquisition coefficients in overlapping windows `j` over the WHOLE acquisition timeline, with a
  global `beta_bar` as the first estimate; `mu` ties the windows (slow drift allowed, jumps not).

Why this gives motion the minimal share: `beta` and `C` are unpenalised except for drift smoothness, so any pose
variation that a design column can explain is free there and costs `lambda rho(.)` in `m`. The solver puts it in
`beta`. What a design column cannot explain is forced into `m` and pays the prior. The split is decided by the
objective, not by a post-hoc projection. The known limit stays: head motion locked to the design (period-2 jitter in
an alternating design) is indistinguishable from `beta` and goes there.

Gauge fixing: level 0 is the anchor (`beta` has no column for it); the eddy translation is odd in `g` and the
contrast columns are even, so the two are separable in the joint solve once both are in one system (replaces the
unconditional projection along `p` that today sends even-in-`g` PE translation to motion).

## 2. Design matrix `D_t` by modality

| modality | levels (anchor first) | contrast columns in `D_t` (unapplied) | geometry (applied) |
|---|---|---|---|
| BOLD | 1 | none; optional physio columns if a belt or slice-phase predictor exists | none |
| ASL | control, label, M0 | level indicators; `s_tc` is the same column | none |
| slow flow | states | level indicators | none (an eddy column per state is a research option) |
| DWI | b0, shells by b | level indicators; per shell the five l = 2 terms of `g`; per shell the even-in-`g` PE translation `(g.p)^2` as a column until the AP/PA test says it is geometry | per shell `C g_t`: translation, two shears, scale along `p` |

The windows `j` are Hann windows on the acquisition timeline (the existing `build_encoding_covariates` basis), not on
a level's sub-timeline. A column is active in a window only where its surround difference has rank (the existing
rank check), so blocked layouts do not produce spurious coefficients.

## 3. Hierarchy (how it is solved)

The objective is non-convex in `m` and the image term dominates the cost, so the solve is staged. Each stage keeps
the whole series in one batch.

```
H0  initialise        per-level low-motion means (existing build_low_motion_reference); m^0 from the existing
                       per-level passes or from the coarse Adam stages of one joint pass
H1  joint pass         all T frames in ONE batched pass; frame t evaluated against R_t (per-frame reference,
                       exists) on its LEVEL's sample pool (new: per-level point pools); unknowns m, beta_j, C
                       updated together by the existing Adam -> L-BFGS -> Newton schedule; after every stage the
                       exact re-split (existing _project_encoding_gauge) runs on the full timeline, so beta_j is
                       always the robust estimate and m carries the rest; the prior term is in the loss
H2  outer loop         recompute R_t from the corrected frames (level means; for DWI optionally the leave-one-out
                       SH prediction at g_t), repeat H1 once or twice (existing reference_iterations)
H3  write              transforms = [eddy(C g_t), rigid(m_t)]; fd from m; beta, C, windows, weights recorded
```

Memory: every real series fits one batch. The chunk limit `_MAX_BATCH_TENSOR_ELEMENTS = 1.5e9` is 850-1770 DWI frames
and 9000 BOLD frames; a 100-volume DWI moving stack at the 2 mm cap is 85 M elements. If a series ever exceeds the
limit, the fallback is overlapping chunks with `beta` and `C` solved on the union (shared parameters across chunks),
not independent chunks.

What already exists and is reused: per-frame references (`reference=<list>`), the Hann window basis and the robust
re-split, the eddy affine and its Jacobian, the convergence-controlled schedule, the sampling menu, the design menu
and record. What is new: per-level sample pools inside one pass, the prior on `m` in the loss with IRLS scale, the
joint odd/even solve for the PE translation, the windows on the acquisition timeline, and the phantoms in section 5.

## 4. Decisions to discuss

1. **Prior on `m`: one term or two.** Charbonnier on the second difference (one `lambda`, one `delta`) versus the
   decomposition `m = s + j` with a quadratic penalty on `Delta^2 s` and an L1 penalty on `j` (trend plus sparse
   jumps). The second is closer to how we describe motion (drift plus events) and gives a usable "jump count" per
   series, at the cost of a second weight. Proposal: start with Charbonnier because it is already wired
   (`surround_weight`, `surround_delta_mm`), and keep the two-term form as the alternative if the phantom gate (5.1)
   shows jumps being smeared.
2. **Where the prior failed before.** The slow-flow phantom regression (tz 0.003 -> 0.019 mm at weights 0.02 / 0.05) was
   measured with per-state passes, where the penalty acts between frames 4 TRs apart and the state offset is not yet
   in the model. It has to be re-measured in the joint configuration before it is used as evidence against the prior.
3. **`beta` drift.** Constant per level (today) or windowed over the series with `mu`. Proposal: constant level
   offsets, windowed l = 2 and eddy terms, because gradient heating changes eddy and direction contrast over 100
   volumes while a tag/control offset does not.
4. **Even-in-`g` PE translation.** Contrast (remove from pose) or geometry (apply). Decided by the AP/PA sign test on
   sub-BLAST032 (plan R2). Until then it is a contrast column, which at least stops it being motion.
5. **Anchor and output space.** Anchoring at b0 makes the b0 frames' pose relative to the shells a prior-determined
   quantity (14 adjacent pairs on a 100-volume table). That is what we want for `fd`, but the output grid is b0's. The
   alternative, anchoring at the largest shell and applying the b0 offset, changes the output space downstream expects.
   Proposal: keep b0 as anchor.
6. **Per-level sample pools.** The reason the current code splits by level is that one shared sample set drawn from
   the mean reference misses high-b edges. In one pass each frame can use the pool of its level's reference. Cost is
   bookkeeping, not compute. Is there a reason this was not done before that I have missed?
7. **Signal-level unification.** The slow-flow GLM takes the poses as nuisance columns. Stage 1 keeps the two layers
   separate but exports `m`, `beta` and `D` so the GLM can refuse motion columns when `beta` was not identifiable, or
   orthogonalise them against the polarity column. A single pose-plus-signal fit is the long-term form and is not in
   this proposal.
8. **Hyper-parameters.** `lambda` and `delta` are set once from the phantom gate and fixed per preset; they are not
   tuned per subject. `delta` in mm (0.25 as today) makes the setting spacing-independent.

## 5. Verification (gates before any default changes)

5.1 **Drifting-acquisition phantom (new).** `simulate_dwi_series` extended with: three shells interleaved after a
    b0 block (the real FPA/SOCOM layout), an eddy matrix `C(t)` that drifts linearly over the series, a direction
    contrast offset that drifts, true motion = drift + 2 jumps + jitter, 5 seeds. Gate: motion error against truth
    lower than the current `unified` engine; jitter FD ratio >= 0.95; jump amplitude recovered within 10 %.
5.2 **No regression on the existing phantoms.** ASL 0.107 deg, slow flow 0.137, single-shell DWI 0.330 deg
    (`docs/MOTION_DESIGN.md`), 5 seeds, same or better.
5.3 **The slow-flow prior re-test (decision 2).** `lambda` in {0, 0.02, 0.05, 0.1} in the joint configuration;
    adopt only if tz error does not regress from the `lambda = 0` baseline.
5.4 **AP/PA sign test (decision 4).** sub-BLAST032 dir-AP and dir-PA: odd component reverses sign, even component
    does not, each to within its seed spread.
5.5 **Real DWI.** sub-BLAST079 AP: tensor-fit residual, cross-shell edge correlation in parenchyma, FD with and
    without b0 switches, against `unified` and `dispatch`. Descriptive, not a gate.
5.6 **Real ASL and BOLD.** sub-BLAST079 ASL perfusion edge ratio and negative fraction; BOLD identical to `bold`
    preset (one level, no columns: the objective reduces to the current one when `lambda = 0`).

## 6. Work packages (for later; none started)

| WP | content | owner module |
|---|---|---|
| 1 | per-level sample pools in one batched pass (`sample_pools: list` keyed by frame) | `motion_batched`, `sampling` |
| 2 | windows on the acquisition timeline; levels + l = 2 + even PE column + eddy in one `D_t`; joint odd/even solve replacing the `p` projection | `motion_batched`, `motion_design` |
| 3 | prior on `m` in the loss (Charbonnier; IRLS scale from the Tukey MAD), `lambda`/`delta` per preset | `motion_batched`, `motion_presets` |
| 4 | `engine="joint"` in `motion_correct_design`: H0-H3 as one call; `unified` kept as the parity reference | `motion_design` |
| 5 | phantoms 5.1 and the gates 5.1-5.3 as tests; the AP/PA script under `benchmarks/` | `motion_simulation`, `tests/` |
| 6 | downstream switch (`antsxdwi`, `antsxfunctional`, `antsxslowflow`) once 5.1-5.4 pass; GLM export of `beta` identifiability | downstream repos |

Each WP ends with its own test and a tracking entry; WP 1-3 are independent of each other and can be reviewed
separately; WP 4 composes them.

## 7. Evidence from `motion_cohort` (2026-10-27, real data, no phantoms)

Question: how much of the raw motion estimate could acquisition terms absorb at all? Method: every series registered to
its own mean with the `bold` preset (no design knowledge, 2 mm cap for DWI), then the poses regressed on the design
columns the metadata provides (ASL: level indicators; DWI: shell indicators, five l = 2 terms, `g.p`; all: `(-1)^t`),
once by least squares (the ceiling, chance correlation included) and once by the surround-Tukey estimator the code
uses. Step MAD is the robust frame-to-frame step per degree of freedom in the acquisition frame, rotations at 50 mm.
Scripts: `scratchpad/misattribution_ledger.py`, `asl_alternation.py`, `bold_pose_spectrum.py` (session scratch, not
committed).

| series | T | FD raw | FD after design (Tukey) | absorbed | step MAD t_pe / t_sl / t_ro / pitch (mm) |
|---|---|---|---|---|---|
| DWI FPA BLAST031 AP (3 shell) | 100 | 1.837 | 1.121 | 39 % | 1.56 / 1.21 / 0.11 / 0.30 |
| DWI FPA BLAST066 AP (3 shell) | 100 | 1.438 | 0.741 | 48 % | 0.97 / 1.40 / 0.13 / 0.31 |
| DWI breacher 4667 (b0 block + 30) | 36 | 0.347 | 0.192 | 45 % | 0.30 / 0.14 / 0.08 / 0.10 |
| DWI breacher 6223 | 36 | 0.431 | 0.283 | 34 % | 0.23 / 0.23 / 0.09 / 0.12 |
| ASL ExpArt EAS009 | 25 | 0.360 | 0.355 | 1 % | 0.07 / 0.22 / 0.06 / 0.06 |
| ASL ExpArt EAS020 | 25 | 0.498 | 0.476 | 4 % | 0.07 / 0.15 / 0.07 / 0.06 |
| ASL FPA BLAST031 | 16 | 0.052 | 0.046 | 11 % | 0.02 / 0.04 / 0.02 / 0.03 |
| ASL FPA BLAST071 | 16 | 0.799 | 0.784 | 2 % | 0.03 / 0.47 / 0.25 / 0.24 |
| BOLD breacher 3315 (TR 1.0) | 300 | 0.311 | 0.310 | 0 % | 0.14 / 0.15 / 0.04 / 0.12 |
| BOLD breacher 6223 (TR 2.5) | 196 | 0.247 | 0.242 | 2 % | 0.22 / 0.08 / 0.02 / 0.05 |

Readings:
- **DWI.** 34-48 % of the raw FD is explainable by metadata columns, and the step anisotropy (PE and slice 10-14x the
  readout step on BLAST031) is the signature of eddy and shell contrast, not of a head. The current `unified` engine
  reports 0.572 mm on BLAST031 (manifest) where columns on a mean-reference registration leave 1.121: the
  contrast-matched reference absorbs more than the columns do. Both are needed.
- **ASL.** A constant tag/control offset absorbs 1-11 %. The bounce exists, in the slice direction (period-2 amplitude
  0.12-0.20 mm in `t_sl` on EAS009/EAS020/BLAST071), but its sign relative to the label is consistent on only 13-43 % of
  frames beyond chance, so a constant column cannot take it. Either the tag/control effect on the pose varies over the
  series, or it is motion. This is the case to resolve with the joint model and a per-frame contrast-matched reference,
  not with a larger column set.
- **BOLD.** Metadata columns explain nothing; `(-1)^t` is 0.002-0.026 mm. The effect is respiratory: on 6223 the PE step
  is 11x the readout step and 6 % of the pose power sits at Nyquist (TR 2.5 aliases breathing there); on 3315 (TR 1.0)
  20-25 % of the power of `t_pe`, `t_sl` and pitch sits in 0.2-0.4 Hz while readout translation is 0.04 mm. 3315 is
  breathing-locked real nodding (slice translation and pitch move with PE); 6223 is a PE-only field shift. The
  discriminator is the DOF pattern, and it needs a data-derived regressor, not a sidecar column.

## 8. BOLD: a physically founded respiratory term (no belt)

A respiratory change of `B0` displaces EPI voxels along the phase-encode axis only, by an amount proportional to the
local field change: non-uniform in space (largest at the skull base and sinuses, with a gradient toward the chest),
and with the same time course everywhere in the head. Rigid head motion is uniform and appears in the B0-immune
degrees of freedom (readout translation, roll, yaw). The proposal adds to each frame's transform one non-rigid term

```
d_t(x) = a_t * phi(x) * p_PE
```

with `phi(x)` a fixed spatial pattern per subject (from the field map when one exists; otherwise the dominant PE-only
mode of the frame residuals after a first rigid pass, orthogonalised to a uniform shift) and `a_t` one scalar per frame
solved with the pose. Because `phi` is non-uniform, `a_t` is identifiable from the images independently of the rigid
PE translation. The uniform part of the same breath is then attributed by the rule already used for eddy: the rigid PE
translation is required to be uncorrelated with `a_t`, so its projection on `a_t` moves to the field term. The
B0-immune DOFs are the null check and must not change. Applied to the data (it is geometry), excluded from `fd`.

This replaces the post-hoc Wiener shrinkage of the reported trajectory (`_shrink_pe_pseudomotion_core`), which has no
image evidence behind it and whose gates were shown to fire on white noise. On 3315 it should leave the nodding in
`fd`; on 6223 it should take most of the PE step. Those two runs are the acceptance pair.

## 9. A motion-specific similarity metric

Mattes MI and correlation treat every intensity difference as a candidate for pose. In a time series we know the
form of the differences that are not pose: the tag/control perfusion pattern, the diffusion signal at direction `g`
given the other directions, the respiratory field shift, the eddy affine. A metric built for motion correction
compares each frame with a prediction of itself from the rest of the series under the forward model:

```
L_t = rho( I_t o T(m_t) o A_t  -  P_t[ {I_s o T(m_s) o A_s}_{s != t} ] )
```

where `P_t` is the design-aware predictor (ASL: the state mean; DWI: the leave-one-out spherical-harmonic prediction at
`g_t`, antsxdwi already has it; BOLD: a local temporal mean plus the field mode of section 8), evaluated on tissue whose
signal does not depend on the design (the existing `sample_mask`), and `rho` robust. The design terms then never
generate pose because they are in the prediction, which is a stronger statement than a column in `D_t` that can
absorb pose after the fact. It is the metric form of "other explanations enter the model".

The earlier negative result with SH-predicted references (`docs/MOTION_SYSTEMATIC_NOISE.md`) used them as the
reference of Mattes MI on the shell sub-timeline with a shared sample set. The proposal differs in three places: a
residual-based `rho` on the predicted intensity (MI discards the predicted scale), the whole timeline with the prior of
section 1, and per-level sample pools. It is the highest-value and highest-risk item in this plan and needs the
BLAST031 / BLAST066 DWI pair and the EAS020 ASL series as its gate before it becomes a default.

## 10. Decisions taken (2026-10-27)

1. **Prior.** Charbonnier on the second differences, knee at the precision floor (0.05 mm), OFF by default. 0.02 was
   best on the structured phantoms (section "engine=joint" of `docs/MOTION_DESIGN.md`) but its weight is absolute and
   competes with the image term's curvature: on a featureless phantom with no motion it displaced frames by 0.1 mm. The
   split is made by the robust re-split after every stage; a curvature-normalised prior (weight in units of the
   measured Newton curvature per frame) is the next work package.
2. **The earlier slow-flow regression** was measured in the split configuration and is superseded by the joint
   phantom numbers; the slow-flow phantom test (`antsxslowflow/tests/test_motion_slowflow_phantom.py`) still gates it.
3. **`beta` is constant per level and per shell** (one window over the series); windowed drift stays an option
   (`window_size`).
4. **Even-in-`g` PE translation** is a contrast term: kept out of the pose by the odd-only projection, never applied
   by the motion model. Residual geometry is the dewarp step's job (antsxdwi reverse-PE), as decided by the user.
5. **The DW-to-b0 link is an explicit registration**, as decided by the user: H0 links a level by a rigid registration
   of the corrected means whenever its offset is not identifiable from its interleaving (DWI's 7 b0 frames), and by
   identity when it is (ASL control / label); the joint pass refines every frame from there and b0 stays the anchor.
6. **Per-level sample pools** implemented as per-frame point weights (`frame_levels`), off by default (neutral or worse
   on the phantoms; the real-data variant is in the gate table).
7. **GLM.** Tested in antsxslowflow (`tests/test_oscillation_motion_orthogonalization.py`): motion nuisance columns
   are orthogonalised to the harmonic columns by default (`motion_orthogonalize=True`); raw poses with a 0.6 mm
   respiration-locked PE shift gave amplitude 1.66 for a truth of 1.0, orthogonalised 1.00. An intensity change locked
   to motion at the frequency is not separable in the GLM (1.23 with a 0.4 gain) and must leave the poses upstream.
8. **Hyper-parameters** fixed per preset from the phantoms, checked on the cohort by the readout-floor gate (the
   B0-immune readout step must not change by more than 15 % between engines).
