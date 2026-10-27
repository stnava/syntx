# Design-driven motion correction (`syntx.motion_design`)

## Problem

In ASL, DWI, slow-flow and BOLD series, the acquisition design changes the image contrast frame by frame:
- ASL: tag and control frames;
- DWI: b-value shell and gradient direction;
- slow flow: the encoding state;
- BOLD: phase-encode B0.

A registration to a common reference therefore returns head motion plus a pose bias locked to the design. Removing that bias
with a rule that does not use the design does not work. Shrinking any frame-to-frame alternation (the opt-in PE debias) also
removes real head jitter (audit: FD 0.94 -> 0.15 mm). The design labels are what make the bias separable.

## Model

```
pose(t) = m(t) + D(t) beta + e(t)
```

- `m(t)` is head motion (smooth drift plus jumps).
- `D(t)` holds the design columns: the level indicators and, for DWI, the l = 2 direction terms of each shell.
- `beta` is the acquisition-locked offset.
- `e(t)` is residual registration noise.

`beta` is estimated from local (surround) differences of the trajectory with Tukey weights
(`motion_batched._project_encoding_surround_tukey_np`). Smooth drift lies in the nullspace of that estimator, and isolated
jumps are downweighted.

## Design menu

`make_design(name, ...)` builds a `Design` from the metadata, and `DESIGNS` lists the entries.

| name | levels | design columns in the optimizer |
|---|---|---|
| `none` | 1 (BOLD) | none |
| `tag_control` | control (anchor), label, each other type (M0) | none (levels are registered separately) |
| `shells` | `shell_groups(bvals)`: b0 (anchor), shells by ascending b | the l = 2 direction terms per shell; the eddy term along phase encode, tied to the gradient per shell |
| `encoding_states` | the encoding states | none |
| `custom` | caller labels | caller columns |

## Engines

`motion_correct_design(image, design, engine=...)` returns the underlying result plus `design`, which records:
- the engine and the call made;
- the levels and their layout;
- the links between levels;
- the `beta` columns, and those left out;
- the per-level offsets.

### `engine="unified"`, the estimator

1. **Design-specific references and sample points.** Each level is registered in its own pass (`motion_correct_grouped`)
   to its own low-motion mean. Its sample points come from that reference.
   - A single batched pass sharing one sample set fails high-b levels. Its points come from the mean of all references,
     which b0 dominates, so its CSF edges are invisible at high b.
   - DWI phantom, b2000, error within the level: 2.7 deg in one pass vs 0.52 deg in per-level passes.
2. **In-level design columns** are estimated in the optimizer (`encoding_covariates`, `pe_affine_covariates` per level).
   A column set is used only when the level has at least `FRAMES_PER_BETA_COLUMN` (6) frames per column. Columns left out
   are recorded.
3. **Links between levels.**
   - **Identity** where the level's offset is then measured locally. That needs `MIN_MIXED_FOR_BETA` (4) frames with a
     neighbour of another level, or both levels interleaved. A cross-contrast registration would only add its own bias.
   - **Rigid registration of the corrected means** otherwise, i.e. for blocked levels and lone frames.
4. **Per-level offset `beta`.** The composed trajectory is projected on the level indicators, and the offset of each
   non-anchor level is removed by a correction transform at the start of its chain. The data are resampled once.
   - When the levels are not interleaved enough, the offset is not identifiable. It is then not removed, and this is
     recorded (`beta_levels_estimable`).

A one-level design (BOLD) is the plain registration to the series mean, identical to `preset="bold"`.

Options:
- `beta="levels+design" | "levels" | "none"`.
- `iterations=k`: re-register each level to the mean of its corrected frames, k times.

### `engine="joint"`, one optimisation over the whole timeline (v6.25)

Design and evidence: `docs/MOTION_JOINT_OPTIMIZATION_PLAN.md`; review that motivated it: `docs/MOTION_COMPLETE_MODEL_REVIEW.md`.

The `unified` engine solves each level separately and estimates `beta` afterwards from local differences of the
composed trajectory, so any acquisition effect those local differences cannot isolate is left in the pose. The `joint`
engine makes the split part of one objective over all `T` frames:

```
J(m, beta, C) = sum_t L_t(I_t o T(m_t + D_t beta) o A(C g_t), R_t) + lambda sum_t rho(W Delta^2 m_t)
```

- **H0.** `motion_correct_grouped(cross_link="registration")`: every level to its own mean, the level means linked
  by an explicit rigid registration of the corrected means (the DW-to-b0 link is a measurement, not an inference).
- **H1.** One `motion_correction` call over the ORIGINAL frames, each against the corrected mean of its own level in
  the anchor space (`reference=<list>`), started from the H0 poses (`init_pose`, `schedule_mode="refine"`: L-BFGS then
  Newton at full resolution; the coarse Adam stages would walk a good pose away), with every design column of the
  timeline in the optimiser (`joint_design_columns`: one indicator per non-anchor level, the five l = 2 terms per
  shell in their own block, the eddy covariates per shell), the intensity range of each frame taken from its own
  reference, and (opt-in, `surround_weight`) the Charbonnier prior on the second differences of the head trajectory
  with its knee at the registration precision floor (`surround_delta_mm = 0.05`). With the prior off the split between
  motion and the design columns is made by the robust re-split after every stage.
- **Gauge.** The encoding offset may carry a translation along phase encode only in the complement of the eddy
  covariates' span over frames (odd in `g` belongs to the applied eddy term, even in `g` is contrast and stays out of
  the pose); the previous unconditional projection sent the even part to motion.
- Written transforms: `[eddy, rigid]` per frame, `fd` from the rigid part; `design.beta_levels` holds the per-level
  column offsets; `group_results = [the joint result]`, `group_indices = [all frames]`, so antsxdwi's consumers are
  unchanged.

Phantoms with known motion (`simulate_dwi_series` seeds 42 / 7, `simulate_asl_series` seed 2; true mean FD
0.156 / 0.155 / 0.160 mm), FD error in mm:

| engine | DWI s42 | DWI s7 | ASL | mean FD reported (DWI s42 / ASL; truth 0.156 / 0.160) |
|---|---|---|---|---|
| unified | 0.438 | 0.398 | 0.089 | 0.301 / 0.180 |
| joint, lambda 0 (default) | 0.208 | 0.259 | 0.059 | 0.220 / 0.169 |
| joint, lambda 0.02 | 0.113 | 0.203 | 0.050 | 0.184 / 0.154 |
| joint, lambda 0.05 | 0.106 | 0.169 | 0.070 | 0.178 / 0.149 |

`unified` reports twice the true DWI motion; the joint pass with per-reference intensity scaling removes most of
that at `lambda = 0`. The prior at 0.02 takes more on these structured phantoms, but its weight is absolute: its force
(`lambda x delta` per mm) competes with the image term's curvature, and on a featureless Gaussian-blob phantom with no
motion (antsxfunctional `test_asl_tag_control_motion_debias`) 0.02 displaced frames by 0.1 mm where 0 gives exactly
zero. The prior is therefore off by default; a curvature-normalised prior is the next work package.
Per-level point weights (`level_weights=True`: each frame weights the shared pool by its own level's edges) were
neutral or slightly worse on these phantoms and are off by default. Real-data gate: `benchmarks/motion_cohort_gate.py`
(section "motion_cohort gate" below).

### motion_cohort gate (real data, 2026-10-27)

`benchmarks/motion_cohort_gate.py` on the curated `motion_cohort` archive (report:
`docs/reports/motion_cohort_gate_2026-10-27.html`). No truth exists; the columns are physical signatures: mean FD,
the robust frame-to-frame step along phase encode / slice / readout (the readout step is immune to B0 and eddy
effects and must not shrink), and the median relative temporal SD of the corrected series in the foreground
(alignment). `joint` at the default (prior off); 0.02, 0.05 and `level_weights=True` are in the report of the same
day (`docs/reports/motion_cohort_gate_2026-10-27.html`).

| series | T | FD unified -> joint | step t_pe | step t_sl | step t_ro | tsd_rel |
|---|---|---|---|---|---|---|
| ASL ExpArt EAS009 | 25 | 0.367 -> 0.365 | 0.077 -> 0.069 | 0.205 -> 0.200 | 0.055 -> 0.053 | 0.232 -> 0.232 |
| ASL ExpArt EAS020 | 25 | 0.499 -> 0.494 | 0.093 -> 0.103 | 0.206 -> 0.182 | 0.055 -> 0.054 | 0.228 -> 0.228 |
| ASL FPA BLAST031 | 16 | 0.044 -> 0.041 | 0.016 -> 0.016 | 0.029 -> 0.029 | 0.020 -> 0.020 | 0.014 -> 0.014 |
| ASL FPA BLAST071 | 16 | 0.794 -> 0.790 | 0.044 -> 0.044 | 0.481 -> 0.481 | 0.253 -> 0.253 | 0.043 -> 0.043 |
| DWI FPA BLAST031 AP | 100 | 0.606 -> 0.564 | 0.696 -> 0.560 | 0.125 -> 0.100 | 0.057 -> 0.080 | 0.683 -> 0.681 |
| DWI FPA BLAST066 AP | 100 | 0.540 -> 0.452 | 0.601 -> 0.438 | 0.134 -> 0.152 | 0.097 -> 0.072 | 0.684 -> 0.686 |
| DWI breacher 4667 | 36 | 0.150 -> 0.142 | 0.074 -> 0.088 | 0.115 -> 0.105 | 0.029 -> 0.023 | 0.574 -> 0.574 |
| DWI breacher 6223 | 36 | 0.256 -> 0.257 | 0.127 -> 0.145 | 0.136 -> 0.121 | 0.027 -> 0.034 | 0.525 -> 0.525 |

Reading (`docs/reports/motion_cohort_gate_2026-10-27_default.html`; the 0.02 / 0.05 / `level_weights` variants are in
`motion_cohort_gate_2026-10-27.html`): alignment is identical (tsd_rel), so the engines differ in attribution, not in
registration quality. FD falls 1-7 % on ASL and 0-16 % on DWI; on the two FPA DWI series the phase-encode step falls by
0.14-0.16 mm while the readout step moves by at most 0.025 mm in either direction (on DWI the unified readout step also
carries the cross-contrast link bias at every b0 switch, so part of that is not jitter). The ASL readout steps change by
at most 0.002 mm; the large-motion series (BLAST071, FD 0.79) is untouched. With the prior at 0.02 the DWI FD fell a
further 2-5 % and the ASL FD 3-5 %, at the cost described above. Every frame converged. Cost: 1.2-1.5x the unified time
(the H0 pass plus one full-resolution refinement).

### `engine="dispatch"`, today's calls

Reproduces today's calls unchanged:
- ASL: antsxfunctional `syntx_motion_correction`;
- DWI: antsxdwi `motion_correct`;
- slow flow: `preset="slowflow"`;
- BOLD: `preset="bold"`.

## Evidence

Measured on the `motion-design` branch. All runs used MPS in float32, with no concurrent GPU job unless stated.

### Ground-truth phantoms (`syntx.motion_simulation`), 5 seeds

Errors are rotation in degrees / translation in mm.

| case | dispatch | unified | unified, `beta="none"` |
|---|---|---|---|
| ASL, 24 frames | 0.117 / 0.014 | **0.107** / 0.014 | 0.158 / 0.106 |
| slow flow, 4 states x 6 | 0.164 / 0.029 | **0.137** / 0.028 | 0.166 / 0.083 |
| slow flow + eddy 0.4 mm | 0.148 / 0.027 | **0.132** / 0.029 | 0.166 / 0.084 |
| DWI single shell + eddy 0.3 | 0.359 / 0.066 | **0.330 / 0.035** | 0.330 / 0.031 |
| BOLD, BOLD + real period-2 jitter | 0.147, 0.146 | identical | identical |

**Independent real jitter is kept.** With 0.15 mm of jitter, the estimated FD over the true FD is:
- ASL 0.98 (dispatch 0.977);
- slow flow 0.976 (0.979).

**Reference iteration** (`iterations=2`): ASL 0.106, slow flow 0.139, slow flow + eddy 0.136, DWI 0.370 deg. That is no
gain on the phantoms, at 2-3x the time, so it is off by default.

**Multi-shell DWI phantom.** Every engine, dispatch included, registers some high-b frames with a 180 deg flip. The phantom
head is a symmetric ellipsoid, and its edges fade at b 2000-3000.
- Frames more than 10 deg off, over 5 seeds of about 69 frames: 9-158.
- Rotation error on the remaining frames: 1.0-3.1 deg.
- At 1500 / 3000 the unified engine had more flipped frames (158 vs 97).

This phantom cannot rank the engines. Real b3000 data registers without flips.

### Real data, FPA sub-BLAST079 ses-01

The injected-motion test re-corrects the method's own corrected series twice: as is, and after a known rigid trajectory
has been applied. The acquisition bias is common to both runs and cancels. The test therefore measures precision and the
cost of `beta`, not its benefit.

- **ASL** (16 frames, 8 control / 8 label):

  | method | seed spread | injected error | perfusion edge / interior (raw 0.986) | negative perfusion fraction (raw 0.068) |
  |---|---|---|---|---|
  | dispatch | 0.004 mm / 0.011 deg | 0.062 mm / 0.072 deg | 0.955 | 0.064 |
  | unified | 0.003 mm / 0.007 deg | 0.062 mm / 0.071 deg | 0.955 | 0.064 |
  | unified without `beta` | — | 0.046 mm / 0.049 deg | 0.986 | 0.069 |

  Without `beta` the injected error is better, because on 16 frames the offset estimate costs precision, but the perfusion
  edge artifact goes back to the raw level. The tag-control offset is therefore present in real ASL, and removing it
  improves the perfusion map.
- **DWI** (100 volumes, b0 / 1500 / 3000):

  | method | seed spread | injected error | mean FD | time |
  |---|---|---|---|---|
  | dispatch | 0.023 mm / 0.024 deg | 0.085 mm / 0.085 deg | 0.398 mm | 94 s |
  | unified | 0.020 mm / 0.023 deg | 0.095 mm / 0.081 deg | 0.395 mm | 76 s |
  | unified without `beta` | — | 0.168 mm / 0.107 deg | 0.706 mm | — |

  The b0 tSNR and the sharpness of each shell's mean are equal across methods. Without the level offset, FD doubles: the
  shell offsets are large in real DWI, and `beta` removes them.
- **BOLD** (750 volumes): the unified result is identical to dispatch (largest pose difference 0).

**Summary.**
- On the phantoms, the unified engine beats or matches today's calls for ASL, slow flow and single-shell DWI.
- On real ASL and DWI it matches them. These real-data measures do not separate the two.
- It is one code path for every design, and it is 20 % faster on DWI.

## Acquisition axes (`syntx.acquisition_axes`)

The phase-encode B0 debias needs the slice and phase-encode axes. Sidecars often lack them:
- FPA ASL has no `PhaseEncodingDirection`;
- no sidecar on disk has `SliceEncodingDirection`.

`infer_slice_axis` decides from these sources, in order:
1. explicit `slice_axis`;
2. `SliceEncodingDirection`;
3. `SliceTiming` length matching exactly one voxel axis;
4. voxel thickness;
5. slice interleave: `median[corr(i, i+2) - corr(i, i+1)]` of the plane-mean time series, with the design levels removed.

`infer_phase_encode_axis` uses the Nyquist N/2 ghost, cross-checked by the frame-to-frame shift spread.

Both raise when no source decides. The resolver `encoding_frame` replaces the former hard-coded "voxel axis 2 is the slice
axis".

**Validation on real data.** Each series was run with no metadata, in 3 axis orders:
- FPA and ExpArt (ASL, BOLD, DWI) and ds005134 DWI: 21 of 21 decided cases correct.
- ds007693 BOLD (no interleave signal) and travelers DWI (margin 0.018 < 0.02): undecided, 6 runs.
- No wrong answer.

All real data is Siemens with phase encode `j` and axial slices along `k`. The permutations show that the inference follows
the voxel axes. The thresholds are provisional.

## Limits

- **Design-locked real head motion cannot be separated from `beta`.** In every multi-level design here a period-2 jitter
  is in the span of the design columns:
  - ASL tag/control;
  - FPA DWI's alternating shells;
  - slow flow's 4-state cycle.

  On real DWI, 77-82 % of an injected alternation was kept by every engine.
- **Short series.** On 16-frame ASL the offset estimate costs injected-motion precision, 0.046 -> 0.062 mm.
- **`beta` acts on all six degrees of freedom.** No menu entry needs a narrower mask: the DWI eddy term is in the
  phase-encode affine.
- **Multi-shell DWI phantoms with high b** are outside every engine's working range (see above).

## Partially identifiable designs

The level offset is estimated for each level that is interleaved enough: at least 4 active surround rows and at least 4
frames next to another level. A level that is not, for example a lone ASL M0 frame, keeps the pose of its link (the
registration of its mean) and is listed in `cross_group_offset["not_identifiable_groups"]`. The other levels are still
corrected. Before this, a single lone frame switched the offset off for every level.

## Downstream call sites

| repo | call | design |
|---|---|---|
| antsxfunctional | `perfusion.pipeline.syntx_motion_correction` (ASL, rsfMRI, task fMRI) | `tag_control`, `none` |
| antsxdwi | `motion.motion_correct` (with `bvals`) | `shells` |
| antsxslowflow | `motion.api.motion_correct` (encoding states inferred from bval / bvec) | `encoding_states` |

Each uses `engine="unified"`; antsxfunctional and antsxdwi expose `engine=` (`"dispatch"` gives the previous call).
