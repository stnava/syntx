# Tracking Summary
current phase: v6.26.0 (motion model v2 and censoring repair, docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md: S1, S3 done; S2 engine="projected" shipped as available, not default)
next WP: S2 `engine="projected"` (design projected out of the metric, identity level links from the prior, no re-split), gated on the DWI phantom (G1)
blockers: none
active work: R12 below
device: Apple Silicon MPS / CPU (macOS arm64)
last updated: 2026-10-28

---

## R12 (prior in every stage, band-limited FD, resting-BOLD pilot, projected engine, v6.26.0, 2026-10-28)
- **Plan** (`docs/MOTION_JOINT_V2_AND_CENSORING_PLAN.md`): the v2 model (design `D` profiled out of the metric per point, levels
  differ in contrast never in pose, prior inside one objective), six syntx WPs, nine cross-repo censoring WPs, six gates, four
  decisions taken (engine `projected`, 0.5 mm on band-limited FD, automatic band with a recorded trace overriding, field term dropped).
- **Resting-BOLD pilot** (`benchmarks/pilot_bold_resting_joint.py`, `docs/reports/pilot_bold_resting_joint_2026-10-27.html`): one
  L-BFGS objective over all frames (Huber residual about the robust temporal mean at a gradient-weighted point set, per-point scale,
  per-slice-group PE field term, Charbonnier head prior) on breacher sub-6223 (TR 2.5) and sub-3315 (TR 1.0). Edge temporal SD
  0.0323 vs 0.0340 and 0.0515 vs 0.0524 against the `bold` preset while reporting MORE FD (0.48 / 0.87 vs 0.42 / 0.62 mm):
  the near-Nyquist alternation is a real displacement (notching it before resampling raises edge SD 15-20 % and kills the
  DVARS-FD correlation). Field term 0.04-0.06 mm, per-point scale no effect: both dropped from v2.
- **S1, prior in the Newton stage (`motion_batched.py`)**: `_run_converging_stage(reg_hess_fn, reg_frame_fn)`; the Newton gradient
  includes the prior by autograd, its per-frame Hessian block comes from `surround_prior_hessian` (analytic, `w d (s^2 I - u u^T)/s^3`
  summed over the three terms a frame enters, checked against autograd to 1e-15), the line search sees the prior's per-frame
  attribution. Isolation of the "invented motion" on the antsxfunctional zero-motion blob phantom: a near-linear rotation drift
  about the blob centres, i.e. the prior's nullspace crossed with the image plateau of three Gaussian blobs (0.17 mm at the 16 mm
  extent with every frame reporting converged); a smooth texture on the same phantom gives 0.01 mm. G4 re-specified accordingly.
- **S3, band-limited FD (`motion.py`: `band_limited_fd`, `attach_band_limited_fd`; exported as `syntx.band_limited_fd`)**: the pose
  spectrum's peak above 0.1 Hz, widened to a quarter of the hump's height over the white floor and at least a quarter of the
  Nyquist frequency each side, applied only when its power is 1.5x what its width would hold (`band_source` auto | given | none);
  attached to every `motion_correction`, `motion_correct_grouped` and `motion_correct_design` result (`fd_bandlimited`, `fd_band`,
  summary keys) with the TR from the 4-D spacing or `tr=`. Breacher: 0.426 -> 0.246 and 0.622 -> 0.328 mm; frames above 0.5 mm
  27 -> 10 % and 48 -> 16 %. Documented in `docs/MOTION_BOLD.md`.
- **Tests**: `tests/test_band_limited_fd.py` (5), `tests/test_motion_design.py` +2 (Hessian vs autograd; textured zero-motion phantom
  with the prior on), 25 passed; standards allowlist line moved (`motion.py:1212`); `antsxqc.lint` 0. Pre-existing BOLD Nyquist
  window test still fails (unchanged).
- **S2, `engine="projected"` (shipped, not default)**: the design profiled out of the metric at every point (`design_matrix` in
  the batched core, `projected_design_matrix`: intercept per level plus the l = 2 block per shell; design-model references
  refreshed per stage, redraw and Newton iteration; per-point residual scale), identity links and a robust level gauge for
  interleaved levels with `GAUGE_MIN_FRAMES` frames, registration links for blocked levels, eddy applied, the prior in every
  L-BFGS / Newton stage and never in Adam. Phantoms (CPU, head FD error): DWI 0.176 vs joint 0.210 vs unified 0.319; ASL 0.065
  vs 0.055. Prior OFF by default: its weight is not in the image term's units (the loss is a mean over points): at 0.02 it
  stops L-BFGS from correcting a smooth rotation error (plain phantom 0.065 -> 0.208 deg) and on the real DWI cohort it
  doubled the B0-immune readout step with identical alignment (`docs/reports/motion_cohort_gate_projected_2026-10-27.html`);
  the prior-off gate rerun is pending. Tests: `tests/test_motion_design.py` +4 (29 passed).
- Downstream (same release train): antsxqc 0.4.0 catalog and lint rule; antsxfunctional censors on `fd_bandlimited`; antsxdwi,
  antsxslowflow, antsxmm import the one threshold; antsxslowflow's local motion verdict replaced by `antsxqc.grade`.

---

## R11 (joint motion engine, v6.25.0, 2026-10-27)
- **Review and plan** (`docs/MOTION_COMPLETE_MODEL_REVIEW.md`, `docs/MOTION_JOINT_OPTIMIZATION_PLAN.md`): the `unified` engine
  solved each level separately and gave the free motion term everything the local-difference estimators could not isolate;
  measured on the `motion_cohort` archive, 34-48 % of raw DWI FD is explainable by acquisition columns, ASL 1-11 %, BOLD 0 %
  (respiration is not in the metadata).
- **`engine="joint"`** (`src/syntx/motion_design.py`): H0 per-level means linked by explicit registration, H1 one batched
  pass over all frames (per-frame references, `init_pose`, `schedule_mode="refine"`, all design columns of the timeline,
  Charbonnier prior knee 0.05 mm, off by default). Solver (`motion_batched.py`, `motion.py`, `sampling.py`): `init_pose`,
  `refine` schedule, per-reference intensity range, `frame_levels` point weights (`PointSampler.weights_of`), odd-only
  phase-encode projection of the encoding gauge.
- **Evidence**: phantoms (`docs/MOTION_DESIGN.md`, section joint): FD error DWI 0.438 -> 0.208, ASL 0.089 -> 0.059, slow flow
  0.133 -> 0.068 mm; reported mean FD 0.301 -> 0.220 for a truth of 0.156. The prior (0.02) took more on structured
  phantoms but displaced a featureless no-motion phantom by 0.1 mm (absolute weight vs image curvature): off by default,
  curvature-normalised prior = next WP. Cohort (`benchmarks/motion_cohort_gate.py`,
  `docs/reports/motion_cohort_gate_2026-10-27_default.html`): FD -1-7 % ASL, 0-16 % DWI, alignment identical, readout step
  within 0.025 mm, all frames converged.
- **GLM (antsxslowflow 2.10)**: motion nuisance columns orthogonalised to the harmonic columns (raw poses with a 0.6 mm
  respiration-locked PE shift: amplitude 1.66 for a truth of 1.0; orthogonalised 1.00).
- **Tests**: `tests/test_motion_design.py` +5 (joint columns, refine guard, ASL and DWI joint vs unified, odd-only
  projection); motion / sampling / standards suites green except the pre-existing BOLD Nyquist test; `antsxqc.lint` 0.
- Downstream: antsxdwi 0.31.0, antsxfunctional 0.20.0, antsxslowflow 2.10.0 (`engine="joint"` default, `syntx>=6.25.0`).

---

## R10 (Stage 1: In-Solver Gauge Orthogonalization F14, BOLD EPI-Plane Debiasing & Slice-Axis Inference, v6.24.1, 2026-10-27)
- **F14 In-Solver Gauge Orthogonalization (`src/syntx/motion_batched.py`)**:
  - Projected $\ell=2$ contrast gauge in `_batched_rigid_register_pass_core` strictly onto the 5-DOF subspace orthogonal to $\hat{\mathbf{p}}$-translation in both `_encoding_offset` and `_project_encoding_gauge`.
  - Guarantees $\ell=2$ white-matter fiber anisotropy contrast never steals physical linear-in-$\mathbf{g}$ $\ell=1$ PE eddy translation along $\hat{\mathbf{p}}$.
  - Populates `pe_info["bias"]` from `enc_bias_info` in both single-batch and chunked paths.
- **BOLD EPI-Plane Debias Fix & Slice-Axis Inference (`src/syntx/motion_batched.py`, `src/syntx/motion_design.py`, `src/syntx/acquisition_axes.py`)**:
  - Fixed `sig_ref` bug in `separate_bold_pe_pseudomotion` by consuming the returned `meta` record from `_shrink_pe_pseudomotion_core`.
  - Added alias `debias_pe_pseudomotion = separate_bold_pe_pseudomotion`.
  - Routed `motion_correct_design` for BOLD (`"none"` design) with `pe_axis` and `debias_pe_pseudomotion=True`.
  - Enhanced `infer_slice_axis` (`src/syntx/acquisition_axes.py`) to resolve `dcmmeta_slice_dim` from BIDS sidecars and `interleave_and_slab_shape` when both half-series interleave checks agree on the smaller slab axis of a square-in-plane 2D EPI volume.
- **Implementation Standards Exception (`tests/test_implementation_standards.py`)**:
  - Updated line reference in `ALLOWED_REGISTRATION_EXCEPTIONS` from `motion.py:1059` to `motion.py:1062` matching the fallback `ants.registration` call in `src/syntx/motion.py`.
- **Scientific Directive**:
  - Primary `fd` (`dti_FD_mean/max`, `rsf_FD_mean/max`, `asl_FD_mean/max`) remains the single authoritative Framewise Displacement metric produced directly by the solver. No `*_FD_pe_debiased_*` columns are emitted in wide tables.
- **Verification**: All unit tests in `syntx` (`test_motion_design.py`, `test_motion_reference.py`, `test_implementation_standards.py`, `test_stage0_challenger_blast_probes.py`) pass cleanly; static analysis `python -m antsxqc.lint` reports 0 findings.

## R9 (Stage 0: Behavior-Preserving Motion Design Consolidation & Safe Bug Fixes, v6.24.0, 2026-10-26)
- **Safe Bug Fixes BF1–BF7**:
  - BF1 (F4): `_design_tag_control` preserves contiguous level re-indexing even when `"control"` is missing, recording `anchor_volume_type`.
  - BF2 (F5): `_design_shells` enforces explicit `(T, 3)` gradient table shape validation, rejecting `(3, T)`.
  - BF3 (F6): $b_0$ rows zeroed and DW gradient rows unit-normalized in `pe_affine_covariates`.
  - BF4 (F10): `Design.__post_init__` validation enforces contiguous `0..k-1` levels, row-count matching across covariates, and unit-normalized `pe_axis`; `_design_none` supports `pe_axis`.
  - BF5 (F9): `_unified` forwards `encoding_covariates` and `window_size` for multi-level `custom` designs.
  - BF6 (F15): `pe_info["bias"]` populated from `enc_bias_info` in `_batched_rigid_register_pass_core` when `bias_info is None`; downstream fallback to `encoding_pose_bias_mm_rad` in `antsxdwi`.
  - BF7 (F1/F2/F3): Centralized connected-component level graph isolation (`level_adjacency`, `offset_identifiable`); `motion_correct_grouped` links component roots directly, isolating blocked levels.
- **Centralization C1–C3 & Plumbing**:
  - C1: Centralized `level_adjacency` and `offset_identifiable` in `motion_reference.py`, achieving 0 discrepancies across 1,500 random sequences.
  - C2: Centralized robust Tukey trend estimator `_fit_tukey_trend` in `motion_batched.py`.
  - C3: Centralized post-hoc pseudomotion smoother core `_shrink_pe_pseudomotion_core` in `motion_batched.py`.
  - F11: Single 3D reference footprint in `_corrected_level_mean` and broadcast in `reference_predictor`.
  - F7/F8/F13: Plumbed `sidecar` and `slice_axis` through `motion_correction`, `motion_correct_grouped`, and `_batched_allowed_kwargs`; supported path inputs in `antsxdwi.motion_correct`.
  - Deprecated `nuisance_covariates` and `nuisance_axis` in `motion_correct_grouped` with `FutureWarning`.
- **Golden Baseline Verification**: G1–G5 match pre-change baselines within `0.00e+00`; G6 transforms match within `9.54e-07`; G6 FD reflects the BF7 bug fix.
- **Quality & Standards**: 239/239 syntx motion tests pass; downstream test suites in `antsxdwi` (10/10), `antsxfunctional` (3/3), `antsxslowflow` (1/1) pass. Static analysis `python -m antsxqc.lint` reports 0 findings across all 5 repos with zero `.antsxqc-allow` files. Zero `nibabel` imports.

## R8 (Design-Driven Motion Correction `syntx.motion_correct_design`, Acquisition Axes, robust_affine Multi-Start, v6.23.0, 2026-10-26)
- New `src/syntx/motion_design.py`: one entry point `motion_correct_design(image, make_design(name, ...), engine=...)` with a
  design menu (`DESIGNS`: `none` BOLD, `tag_control` ASL, `shells` DWI, `encoding_states` slow flow, `custom`). Engine
  `unified`: each design level registered to its own reference with its own sample points, interleaved levels linked by
  identity (blocked or lone levels by registering their means), the per-level pose offset removed by the surround-Tukey
  projection on the level indicators. BOLD (one level) is identical to `preset="bold"`. `dispatch` reproduces the previous
  per-modality calls. Results keep their type (attribute access). Design and evidence: `docs/MOTION_DESIGN.md`.
- `motion_reference._group_offsets`: the offset is estimated for each level that is interleaved enough (>= 4 active surround
  rows and >= 4 frames next to another level); a level that is not (a lone M0 frame) keeps its linked pose and is listed in
  `cross_group_offset["not_identifiable_groups"]` instead of switching the offset off for every level.
- New `src/syntx/acquisition_axes.py`: slice and phase-encode axes inferred from the image (sidecar, SliceTiming, voxel
  thickness, slice interleave; Nyquist ghost cross-checked by the shift spread), raising when undecidable. Replaces the
  hard-coded slice axis in the opt-in PE debias paths. Real data: 21 / 21 decided cases correct, 6 undecided.
- `robust_affine(mode='ants', multi_start=True)`: rotation candidates no longer double-centred (`t_rot = t_w + C - R C` with
  `fixed_params = C` added `(I - R) C`); candidates scored on images normalised to the (0, 1) histogram range
  (`normalize_robust`, no auto-mask). Known-rotation head phantom (pitch 8 / yaw 12 deg, 2 origins, 3 seeds): the matching
  candidate now wins and the final error is 0.5-1.8 mm (was 3.5-11.4 mm); each fix alone does not change the selection.
- Phantom gate (5 seeds, rotation deg, previous call -> unified): ASL 0.117 -> 0.107, slow flow 0.164 -> 0.137, slow flow
  + eddy 0.148 -> 0.132, single-shell DWI 0.359 -> 0.330 (translation 0.066 -> 0.035 mm). Real FPA sub-BLAST079: ASL and
  DWI tie the previous calls, BOLD identical.
- Downstream moved to the entry point: antsxfunctional (ASL, BOLD), antsxdwi (shells), antsxslowflow (encoding states).
- Tests (targeted): motion_design / motion_reference / acquisition_axes 41 passed; affine set (robust_affine, affine,
  dewarp, new `test_robust_affine_multistart_ants.py`) 95 passed, 1 skipped; standards 10 passed; `antsxqc.lint` 0.
  The full suite was not run for this version.

## R7 (Centralized Metric Sampling `syntx.sampling`, Merged Pilot Ideas, Independent Audits, v6.22.0, 2026-10-25)
- New `src/syntx/sampling.py`: ONE implementation of metric sample-point selection, designed as a menu
  (`SAMPLING_STRATEGIES`: `gradient_floor` default, `gradient`, `uniform`; `get_sampling_strategy`; unknown names raise).
  Candidate region = above the reference's own noise floor (`background_floor`: noise / clipped_noise / zero-filled
  (masked) / no_background, z = 4; zero padding told apart from clipping and from masking) and below the saturated top of
  the foreground. One physical point pool shared by every pyramid scale; device-resident float32 draws (Gumbel top-k);
  at most `MAX_EVAL_POINTS` = 20 000 points per metric evaluation. Design and evidence: `docs/METRIC_SAMPLING.md`.
- Wired into the rigid motion pass, the group-bias pass, `robust_affine` (`fixed_fg` domain, weighted draws, candidate
  scoring foreground) and the JAX MI (`regular_stride`, seeded `uniform_cube_points`; `syn_jax.fit(sample_seed=42)`).
  `sample_fraction` = fraction of the region each evaluation sees (was the pool size).
- Ground-truth motion phantoms (5 seeds, rotation error deg, former sampler -> default): ASL 0.181 -> 0.113, slow flow
  no eddy 0.280 +- 0.138 (max 0.53) -> 0.166 +- 0.022, slow flow eddy 0.183 -> 0.135, DWI 0.672 -> 0.392, BOLD 0.192 ->
  0.138; eddy PE scale 0.0322 for 0.03. Real DWI (FPA sub-BLAST079, MPS): 78 s / mean FD 0.125 mm (former 65 s / 0.163).
  The former ~0.18 deg "phantom rotation floor" and much of the DWI rotation bias were sampling (MOTION_SYSTEMATIC_NOISE.md).
- Merged `pilot/motion-sampling-and-epi-debias`: grouped DWI PE / non-linear-eddy debias of the reported trajectory and
  the BOLD slice / pitch "Nyquist" extension, both OPT-IN (`motion_correct_grouped(debias_pe_pseudomotion=True)`,
  `motion_correction(debias_sagittal_nyquist=True)`): an independent audit showed they remove real jitter from the
  reported FD. The pilot's lever-arm `pe_scale` conditioning and its `> 0.01` / size-dependent pool were not adopted.
- Two independent audits and two re-reviews; findings fixed (cap boundary, thin-slab ITK abort, integer data, padding vs
  clipping vs masking, provenance, robust_affine fallbacks). Open, pre-existing: `_eval_low_res_mi` scores raw
  intensities on a (0, 1) histogram range (changing it broke multi-start selection); known limits in METRIC_SAMPLING.md.
- Dewarp: `_resolve_restrict_transformation` now runs before `assert_comparable_foreground_extent` (foreground-parity
  guard from `bf21dbf`); pyramid-guard allowlist updated for `motion_batched.py` / `dewarp.py`.
- Tests: full suite 2636 passed, 43 skipped, 0 failed (MPS, two processes with no other GPU job; a run concurrent with
  an external MPS cohort job produced non-finite / uninitialised outputs in unrelated transform tests that passed on
  rerun); `antsxqc.lint` 0 findings.

## R6 (`syntx.dewarp` Foreground-Extent Parity Guard, Reference Masks & Regularized Cross-Modal `SyNOnly`, 2026-10-25)
- Deduplicated `assert_comparable_foreground_extent` into `src/syntx/dewarp.py` (exported in `syntx` and `syntx.dewarp`) and added `reference_mask` and `secondary_reference_mask` parameters to `rigid_align_anatomy_to_reference`, `syn_only_reference_to_anatomy_rigid`, and `dewarp_to_anatomical`.
- Defaulted `syn_only_reference_to_anatomy_rigid` and `dewarp_to_anatomical` Stage 1 (`reverse_pe_midspace`) to regularized `total_sigma=1.0`, `levels=[2, 1]`, `reg_iterations=[40, 10]`, preventing whole-head scalp/skull inward pulling (~41-49 mm over-deformation when unmasked EPI/b0 was registered against masked `anatomy_rigid`).
- Tests: `tests/test_dewarp.py` (10 passed, including `test_foreground_mismatch_raises_in_syn_only_and_dewarp` and `test_dewarp_to_anatomical_with_reference_masks`), `tests/test_implementation_standards.py` (10 passed), `antsxqc.lint` 0 findings.

## R5 (Interleaved-Group FD, Encoding Identifiability Limits, v6.21.0, 2026-10-25)
- `motion_correct_grouped` reports motion and FD without the constant per-group pose offset left by cross-group
  registration when the groups are interleaved (every non-anchor group >= 4 active surround rows); `group_offset_fd`
  holds the offsets and `fd_with_group_offsets`. Transforms unchanged (`cross_group_debias=True` also removes it from them).
  Shared estimator `_group_offsets` (surround Tukey on group indicators) behind both. Real 3-shell DWI (FPA sub-BLAST079 AP,
  98 switches / 99 steps): 0.16-0.17 mm / 0.08-0.09 deg per shell, mean FD 0.173 -> 0.163 mm.
- Investigated short-series ASL (T = 16) excess head error (0.026 vs 0.015 mm floor): NOT under-removal and NOT the shared
  parameter freeze (never freezing: identical errors). It is real jitter aligned with `s_t`, unidentifiable from the bias;
  no jitter 0.020 -> 0.002 mm, still head 0.000. Documented in `docs/MOTION_SYSTEMATIC_NOISE.md` with the lone-frame limit
  (an M0 frame as a categorical state is a one-frame spike to the robust projection: register it as a group).
- Tests: new `test_group_offsets_interleaved_shells_recovered_and_blocked_refused`; `test_motion_reference.py` +
  `test_motion_encoding_bias_simulation.py` 28 passed; implementation standards 10 passed; `antsxqc.lint` 0 findings.

## R4 (Encoding-Locked Systematic Noise: Wired Solver, Simulation Benchmark, Robust Estimators, v6.20.0, 2026-10-24)

Tag `v6.20.0`. Remediates the v6.19.0 audit (`docs/HANDOFF_CROSS_REPO_MOTION_AUDIT.md`); recipe and evidence in `docs/MOTION_SYSTEMATIC_NOISE.md`.

- **Simulation (`motion_simulation.py`, new)**: head phantom (GM shell, WM with fibre field, ventricles, texture) moved by a known rigid trajectory; ASL / slow-flow / DWI / BOLD series whose registration bias arises from contrast alone, plus known encoding-locked PE displacement (eddy, respiration); `evaluate_motion_estimate` scores the composed rigid chain against truth. `benchmarks/motion_encoding_bias_simulation.py`: strategy x modality x seed table + HTML report.
- **In-optimizer encoding offset (`motion_batched.py`)**: `B_win` now enters the loss (`_total_pose`), is optimised with the head pose, and is re-estimated robustly (fold back + surround-Tukey projection) after every stage; the written transforms carry head motion only. The v6.19 `B_win` / `_project_encoding_gauge` / `reg_loss_fn` were dead code. In-loop surround penalty kept but default 0 (it degraded the head estimate as its weight grew). Per-chunk encoding offsets concatenated; `delta_Y_mm_rad` padded to all frames.
- **Robust estimators**: eddy translation attribution `C0` and `bias_covariates` use the surround-Tukey estimator instead of least squares on the poses (chance correlation of a head jump with g/l2 terms was reassigned: 0.12 mm and 0.22 mm head error on simulated DWI -> 0.05 / 0.11 mm). Encoding offsets are no longer re-applied as eddy shift.
- **Categorical states**: integer state vectors with > 2 values (`[0, 1, 2, 3]`) are one-hot encoded, not a numeric ramp (slow-flow projection had silently been a single column).
- **Per-frame references** (`build_bilateral_window_references`): whole-series encoding-matched means by default; the local temporal window made poses relative to a moving average (drift never measured). Inclusive state means are the right reference for discrete states (leave-one-out breaks the common space: 0.03 -> 0.12 mm); `exclude_self` added. `encoding_reference_predictor` (new, leave-one-out kernel reference for continuous encodings) is available but not used by default: no reliable gain on simulated DWI.
- **`motion_correct_grouped`**: removed the auto-injection of `pe_affine_covariates` into `encoding_covariates`, the DVARS median rescaling across shells (raw per-shell DVARS returned again), and the post-hoc polynomial PE regression that overwrote `fd`; added opt-in `cross_group_debias`.
- **BOLD**: `separate_bold_pe_pseudomotion` is opt-in (`debias_pe_pseudomotion=True`, needs `pe_axis`), models the PE translation only (slice translation and pitch never touched), no silent `except`.
- `MOTION_PRESETS` test updated for `asl`; raw-registration allowlist line number corrected (already stale at v6.19.0).
- Simulation (3 seeds): ASL 0.150 -> 0.017 mm; slow flow with eddy 0.248 -> 0.031 mm; DWI with eddy 0.385 -> 0.046 mm (direction-dependent rotation bias, about 0.6 deg, remains the open problem); BOLD respiration 0.285 -> 0.078 mm with nodding untouched.
- Tests: 2605 passed, 43 skipped. `antsxqc.lint src/syntx`: 0 findings.

## R3 (Generalized Overlapping-Window Encoding-Bias & Head-Motion Solver Across DWI, ASL, SlowFlow, and BOLD, v6.19.0, 2026-10-24)

Tag `v6.19.0`.

- **What changed.**
  - Added `build_encoding_covariates`, `_project_encoding_surround_tukey_np`, and `separate_bold_pe_pseudomotion` in `src/syntx/motion_batched.py`, decomposing time-series poses over partition-of-unity Hann windows $\psi_w(t)$ with 50%-breakdown second-order surround-difference Tukey biweight IRLS on $\mathfrak{se}(3)$ principal rotation vectors and Empirical Bayes / Wiener EPI-plane noise-floor matching for BOLD respiratory $\Delta B_0(t)$ pseudomotion.
  - Added `build_bilateral_window_references` (`reference="bilateral"`) in `src/syntx/motion_reference.py`, `ASL_MOTION_PRESET` (`"asl"`) and `SLOWFLOW_MOTION_PRESET` (`"slowflow"`) in `src/syntx/motion_presets.py`, and forwarded `encoding_covariates` and `window_size` through `motion_correction` and `motion_correct_grouped` (`src/syntx/motion.py`).
  - Added `tests/test_motion_overlapping_window.py` covering `TestBuildEncodingCovariates`, `TestOverlappingWindowASL`, `TestOverlappingWindowSlowFlow`, `TestOverlappingWindowDWI`, and `TestOverlappingWindowBOLD`.

---

## R2 (Ecosystem Audit Remediation & Zero-Allowlist Consolidation, v6.18.0, 2026-10-23)

Tag `v6.18.0`.


- **What changed.**
  - Fixed `evaluate_registration_qc` in `src/syntx/registration_metrics.py` to aggregate verdicts via `antsxqc.worst_status()` (preserving `"not_assessable"` instead of promoting to `"pass"`), read the `folding_pct` limit dynamically via `antsxqc.hard_limit("folding_pct", side="high")`, and return `float("nan")` instead of fake `min_jacobian=1.0, folding_pct=0.0` when warp loading fails.
  - Fixed `preprocess_for_landmarks` in `src/syntx/landmarks/preprocess.py` to pass `ants.ANTsImage` directly to `antstorch.n4_bias_field_correction` and removed the `ants.n4_bias_field_correction` fallback.
  - Standardized execution status strings in `src/syntx/provenance.py` and `src/syntx/landmarks/optimal_transport.py` to `"success"` and deleted `.antsxqc-allow` (`antsxqc.lint` passes with 0 findings and 0 exceptions).
  - Replaced `reorient_image2("LPI")` in `src/syntx/viz/core.py` and `src/syntx/viz/figures.py` with direct direction-cosine axis permutation/flipping (`AnatomicalVisualizer._orient_lpi`).
  - Consolidated coordinate primitives from `src/syntx/landmarks/spatial.py` into `src/syntx/spatial.py`, delegated energy calculations in `src/syntx/viz/reports.py` to `syntx.deformation_metrics`, added `slots=True` to `src/syntx/contract.py` dataclasses, removed import-time `PYTORCH_MPS_HIGH_WATERMARK_RATIO` mutation from `src/syntx/__init__.py`, and deleted the legacy `src/syntx/reporting.py` shim.
  - Added `AGENTS.md`, `CLAUDE.md`, `docs/antsx_implementation_standards.md`, `docs/contracts.md`, and the 6-check `tests/test_implementation_standards.py` suite.

## R1 (Delegate Registration QC Verdicts to antsxqc, v6.17.0, 2026-10-22)

Tag `v6.17.0`.

- **What changed.**
  - Renamed `src/syntx/qc.py` -> `src/syntx/registration_metrics.py` and `src/syntx/viz/qc_sections.py` -> `src/syntx/viz/metric_sections.py` (`R3-name` compliance).
  - Delegated registration hard validity limits (`min_jacobian`, `folding_pct`, `dice_overlap`, `inverse_identity_error`) and status grading to `antsxqc.hard_limit` and `antsxqc.grade`.
  - Standardized `ModalityReport` (`src/syntx/viz/modality_report.py`) on the canonical 4-status vocabulary (`pass | warn | fail | not_assessable`).
  - Added `tests/test_antsxqc_lint.py` enforcing `antsxqc.lint.assert_clean` (`26 passed`).
