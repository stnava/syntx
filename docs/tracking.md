# Tracking Summary
current phase: v6.22.0 release (centralized metric sampling: syntx.sampling menu; dewarp foreground parity)
next WP: Multi-modal deformable registration benchmarks and GPU memory optimization
blockers: none
active work: none (v6.22.0 released 2026-10-25)
device: Apple Silicon MPS / CPU (macOS arm64)
last updated: 2026-10-25

---

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
