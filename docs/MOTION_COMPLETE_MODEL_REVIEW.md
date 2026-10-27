# Review: does the design-driven motion correction implement a complete acquisition model?

Status: review, no code changes. Repo state: syntx v6.24.1 (`3062138`), antsxdwi v0.30.0, antsxfunctional and
antsxslowflow at their current `main`. Reviewed 2026-10-27.

## 1. The goal, stated as a model

The intent is a regression over the whole series

```
y(t) = m(t) + D(t) beta + e(t),            t = 0..T-1
```

where `y(t)` is the pose the image metric wants, `m(t)` is head motion, `D(t)` holds every acquisition predictor
known from metadata (ASL tag/control, DWI shell and gradient direction, slow-flow encoding state, BOLD phase-encode
B0), `beta` is the acquisition-locked effect and `e(t)` is registration noise. The estimator should assign to `m(t)`
only what the acquisition predictors cannot explain, because motion is the free, per-frame term and therefore the one
that absorbs everything else when the two compete.

This makes sense, with one qualification that drives the rest of the review: "minimal motion" is a prior, not a
property of least squares. Without a prior the split between `m(t)` and `D(t) beta` is a gauge choice. The code resolves
the gauge by the opposite convention from the one stated above: `m(t)` is free (6 parameters per frame, no penalty)
and `beta` is the constrained term, estimated only from local second-order differences with Tukey weights. That is
the right convention when the design alternates every frame (the acquisition-locked offset then IS a local
alternation), and the wrong one when the design term varies slowly or continuously (DWI direction terms, drifting
eddy fields), because anything smooth in `D(t) beta` lies in the nullspace of the surround operator and is handed to
`m(t)`.

## 2. What the code does, per modality

The entry point is `motion_correct_design(image, make_design(name, ...), engine="unified")`
(`src/syntx/motion_design.py`). Verified by reading the source and by running the gates on real bval tables.

| modality | design | what is optimised jointly | what is estimated post hoc | `D(t)` terms NOT in the model |
|---|---|---|---|---|
| BOLD | `none`, 1 level | rigid pose per frame to the series mean | opt-in shrinkage of the reported PE/slice/pitch trajectory (`debias_pe_pseudomotion`, never applied to data) | respiratory B0 (no predictor exists), Nyquist alternation |
| ASL | `tag_control`, control / label (/ M0) | each level to its own low-motion mean, separate passes | one constant 6-DOF offset per level over the whole series (`_group_offsets`) | the `s_tc` column is built (`design.encoding`) but not forwarded under `unified`; only the level offset is used; any time variation of the tag-control bias goes to motion |
| slow flow | `encoding_states`, k levels | each state to its own mean, separate passes | one constant offset per state | same as ASL (the state column is categorical, so levels and encoding coincide) |
| DWI | `shells`, b0 + one level per shell | each shell to its own mean, separate passes; per shell: l = 2 direction terms in overlapping Hann windows (`encoding_covariates`), eddy shear/scale tied to `g` (`pe_affine_covariates`) | eddy translation `a0` along PE by surround-Tukey on the shell's translations; one constant offset per shell | even-in-`g` translation along PE (projected out by construction, see 3.3); any drift of the eddy or contrast offset over the series |

Links between levels: identity when both levels are in one connected component and the offset is identifiable
(`offset_identifiable`, 4 mixed frames), otherwise a rigid registration of the corrected level means
(`robust_affine`).

## 3. Findings

Ordered by how much they separate the DWI path from the stated goal. "Verified" = read in the code and, where
stated, run; "inferred" = a consequence I did not measure.

### 3.1 The optimisation is split by contrast, not by time; the whole-series model exists only post hoc (verified)

`motion_correct_grouped` runs one independent `motion_correction` per level. Inside the optimiser there is no term that
couples frame `t` of shell 1 to frame `t+1` of shell 2. The joint model over the acquisition timeline is applied only
after all passes, to the composed rigid chain, and only as one constant per level (`_group_offsets`: a single window,
`window_size = keep.size`).

Consequences:
- Within a shell the surround operator `Y_t - (Y_{t-1} + Y_{t+1}) / 2` runs on the shell's sub-timeline. In the FPA
  and SOCOM tables (99-100 volumes, shells switching on 99 % of steps) adjacent frames of one shell are 2-3 TRs apart.
  Head motion between them is correspondingly larger, so more of it crosses the Tukey cut and the "smooth drift is in
  the nullspace" argument holds at a coarser time scale than the data have (inferred).
- The b0 level (7 frames in all three real tables I checked, not identifiable: 0 active surround rows) is linked by a
  cross-contrast rigid registration of its mean to the shell-1 mean. That is exactly the registration the design was
  built to avoid (`docs/MOTION_DWI.md`: b0-vs-DW bias grows with the b gap). Its error enters the series FD at every
  b0 switch, 14 of 99 steps on the FPA table (verified count, bias magnitude not measured).

The premise in the request that memory forces a sliding window is not what the code does. `batched_rigid_register_pass`
splits a series only above `_MAX_BATCH_TENSOR_ELEMENTS = 1.5e9` elements, which is 1770 frames of a 2 mm-capped DWI
volume and 850 frames of a native 1.5 mm one. Every real series here is one batch. The windows that exist are on the
parameters (`B_win`, one per Hann window, shrunk by ridge toward a global `B_bar` estimated first on all rows of the
pass). That part already implements "theory on the whole series, local deviations as a perturbation". The approximation
that costs accuracy is the split by shell, and memory does not require it.

### 3.2 The prior is on `beta`, not on `m(t)` (verified)

Every estimator of an acquisition effect in the code is of the form "regress second-order surround differences of the
pose on the surround differences of the design, with Tukey weights, then remove the fit from the pose":
`_project_encoding_surround_tukey_np` behind `encoding_covariates`, `bias_covariates`, the eddy `a0` attribution and
`_group_offsets`. Head motion is whatever remains, with no penalty. So:

- A design effect that is constant or linear in time within the window is removed correctly.
- A design effect that varies smoothly across the series (gradient heating, slowly drifting eddy field, contrast
  drift with coil temperature) is in the nullspace of the surround operator and stays in `m(t)`.
- A design effect that is large at a few frames (a frame whose direction happens to make the contrast very different)
  is downweighted by Tukey and stays in `m(t)`.

That is the reverse of the stated priority. It was chosen for a documented reason: plain least squares on the poses
reassigned 0.12-0.22 mm of real head motion to the design (`docs/MOTION_SYSTEMATIC_NOISE.md`), and an in-loop Charbonnier
penalty on `m(t)` made the slow-flow phantom worse (0.003 -> 0.019 mm). Both results were obtained with the split-by-level
passes, where the penalty acts on the sub-timeline. Neither tests the configuration the goal describes: one timeline,
all design columns in `D(t)`, a sparsity prior on the second differences of `m(t)`.

### 3.3 In DWI, even-in-`g` translation along phase encode is assigned to motion by construction (verified)

`_encoding_offset` in `_batched_rigid_register_pass_core` removes the component of the l = 2 offset along `p` on every
evaluation whenever the eddy term is on. The post-fit projection in `_project_encoding_gauge` does the same. The eddy
translation `a0` is linear in `g` (odd). A pose bias along `p` that is even in `g` therefore has no column that can carry
it and lands in `t_param`, i.e. in `fd`.

This was the F14 fix (`docs/MOTION_DESIGN_FINAL_PLAN.md` 4.1): it stops the l = 2 gauge from stealing the physical
eddy translation. The cost is unstated there: `docs/MOTION_DWI.md` (v6.14.0) reports that on sub-BLAST079 "the
predicted references did not reduce the even-in-g pose components", so an even-in-`g` component exists on real data and
is now hard-wired to motion. The plan's own R2 gate (AP/PA sign test on sub-BLAST032) is the experiment that decides
whether that component is geometry (apply) or contrast (remove from the pose); it has not been run. Until it is, the
DWI path over-assigns to motion in exactly the direction the request worries about.

### 3.4 The l = 2 term is a 30-parameter fit per shell on about 46 frames (verified gate, inferred effect)

`FRAMES_PER_BETA_COLUMN = 6` with 5 columns requires 30 frames per shell. On the real tables (SOCOM Blast-09 AP,
FPA BLAST079 AP, ExpArt) each shell has 46-47 frames, so the term is on. With `window_size = 16` on a 46-frame
sub-timeline `build_encoding_covariates` makes about 7 Hann windows, each estimating a (5, 6) matrix from roughly 16
effective rows; the ridge then collapses them to the global `B_bar`. The windowed part is therefore inert on real data
and the term is a single global l = 2 regression per shell. That is not wrong, but the window machinery buys nothing
here and the parameter count against the row count means the Tukey scale is estimated from few residual degrees of
freedom. The 65-volume SOCOM-style series would fall under the gate on one or both shells and lose the term entirely.

### 3.5 The real-data evidence cannot show the goal is met for DWI (verified from the docs)

`docs/MOTION_DESIGN.md` states the injected-motion test re-corrects the method's own output, so the acquisition bias
is common to both runs and cancels; it measures precision, not bias. The remaining DWI evidence is mean FD
0.398 (dispatch) vs 0.395 mm (unified) on one subject, and "FD doubles without the level offset", which shows the
offset is large, not that what remains is motion. The multi-shell phantom flips frames at high b and is declared unable
to rank engines. There is no DWI experiment with known head motion AND a known slowly varying acquisition effect where
the estimate of motion is compared against the truth. The single-shell phantom (`simulate_dwi_series`, b0 every 6
frames, constant eddy matrix `C`) has no drift in its acquisition effects, so it cannot expose 3.2.

### 3.6 Smaller items (verified)

- `pe_info["bias"]` after chunking reports the first chunk only (`batched_rigid_register_pass`). Moot today because
  chunking does not trigger (3.1), but it is a silent partial report if `max_batch_frames` is ever set.
- The F14 test (`test_f14_planted_linear_eddy_shift_recovers_without_leakage`) asserts `max|a0| > 0.25` for a planted
  0.4 mm shift and does not check leakage into `encoding_covariates`, although its name says it does.
- For `tag_control` and `encoding_states`, `_unified` builds `design.encoding` and then does not forward it; only
  `custom` and `shells` reach the in-optimiser term. The level offset covers the constant part. This is consistent
  with the docs but the `Design` object advertises a column that the engine ignores.
- BOLD: `debias_pe_pseudomotion` is a post-hoc shrinkage of the reported trajectory with heuristic gates
  (`_shrink_pe_pseudomotion_core`); nothing in the BOLD model is a physical predictor. The modality "looks ok" because
  its contrast is stationary, not because the model is complete.

## 4. Relation to antsxslowflow

`antsxslowflow.oscillation.design.build_encoding_aware_component_design` is the same regression written at the voxel
level: intercept, drift, a bipolar polarity offset, `g_t * [cos, sin](2 pi f t)` per harmonic, then the 6-DOF or
Friston-24 motion parameters as nuisance columns, all solved by OLS. Motion enters last and unpenalised, so there too
the variance shared between a state-locked pose alternation and the polarity/harmonic columns is split by least
squares, not by priority.

The two layers are consistent today only because the pose fed to the GLM already has the state offset removed by
syntx. When the states are not identifiable (blocked layouts, `not_identifiable_states` in `_state_offset_record`),
the raw poses carry the alternation and are collinear with the polarity column. Nothing in the GLM checks for that;
`design_diagnostics` reports a condition number but no caller gates on it. A complete model would either pass the
design-debiased pose only (and refuse the nuisance columns when the offset was not identifiable), or fit pose and
signal in one system. The first is a small change; the second is the long-term form of the goal.

## 5. Answers

1. **Does the goal make sense.** Yes, as `y = m + D beta + e` over the whole series with a sparsity prior on the second
   differences of `m` and `beta` unpenalised. The code implements the dual (free `m`, local-difference-only `beta`),
   which is equivalent for alternating designs and not for continuous or drifting ones.
2. **How close.** ASL and slow flow: the goal is met for a constant per-state offset, which is what those designs
   produce; the evidence (perfusion edge 0.986 -> 0.955, FD ratio 0.98 on jitter) supports it. BOLD: no acquisition
   predictor is modelled, so the question does not arise. DWI: not met. The series is split by shell inside the
   optimiser, the b0 level is linked by the cross-contrast registration the design was meant to remove, even-in-`g` PE
   translation is forced into motion, and no experiment with a drifting acquisition effect has been run.
3. **What closes the DWI gap** (experiments, not a plan; each needs a gate before it becomes a default):
   - One pass over the acquisition timeline with per-frame references (`reference=<list>` already exists) and every
     design column in one `D(t)`: level indicators, l = 2 per shell, eddy per shell. The documented reason for the
     split (one shared sample set fails high b) is a sampling problem; a per-level point pool inside one pass addresses
     it without giving up temporal adjacency.
   - Replace the unconditional `p`-projection in `_encoding_offset` with the joint constrained l = 1 / l = 2 solve (R3)
     and run the AP/PA sign test (R2) on sub-BLAST032 to decide where the even-in-`g` PE component belongs.
   - Re-test a sparsity prior on `m(t)` in that single-timeline configuration, with a phantom whose eddy matrix and
     contrast offset drift over the series. The earlier negative result was in the split configuration and does not
     transfer.
   - Report FD excluding b0 switches, or estimate the b0 link from the 14 adjacent DW frames instead of the mean
     registration, until the b0 level is identifiable by design (more or interleaved b0s).

## 6. What was and was not run

Run: `pytest tests/test_motion_design.py` (18 passed, 46 s, MPS), the `level_layout` gate on three real bval
tables (SOCOM Blast-09 AP, FPA BLAST079 AP, ExpArt), the auto-chunk size for four typical grids. Not run: any
registration on real data, the AP/PA sign test, any phantom with drifting acquisition effects. Numbers quoted from
`docs/MOTION_DESIGN.md`, `docs/MOTION_DWI.md`, `docs/MOTION_SYSTEMATIC_NOISE.md` and antsxdwi `docs/tracking.md` are
theirs, not re-measured.
