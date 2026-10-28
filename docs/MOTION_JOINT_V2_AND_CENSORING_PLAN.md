# Motion model v2 and censoring repair: plan

Status: proposal (2026-10-27). Nothing in this document is implemented. It supersedes sections 8 and 9 of
`MOTION_JOINT_OPTIMIZATION_PLAN.md` and leaves sections 1 to 7 (the released `engine="joint"`, syntx 6.25.0) in place
until the v2 engine passes the gates in section 5.

Evidence this plan rests on:

- `docs/reports/motion_cohort_gate_2026-10-27_default.html`: the released joint engine on the cohort (FD down 1 to 16 %,
  alignment identical, readout floor unchanged).
- `docs/reports/pilot_bold_resting_joint_2026-10-27.html` and `benchmarks/pilot_bold_resting_joint.py`: one joint
  objective on the two breacher resting series.
- The misattribution ledger (session scratch, to be re-run by WP4): DWI 34 to 48 % of raw FD design-explainable, ASL 1
  to 11 %, BOLD 0 % from metadata.

## 1. What the pilot established

1. A single group-wise objective (robust temporal variance of the warped series about its design-profiled model at a
   common point set, with the head prior inside the objective) solved by L-BFGS from a zero pose aligns resting BOLD
   slightly better than the current engine (edge temporal SD 0.0323 vs 0.0340 and 0.0515 vs 0.0524) in one minute per
   series. This is the core of v2.
2. The breacher FD inflation (0.42 and 0.62 mm mean; 27 to 48 % of frames above 0.5 mm) is a real frame-to-frame
   displacement of the image content at the Nyquist-adjacent physiological frequency. Removing it from the poses
   before resampling raises edge variance by 15 to 20 % and destroys the DVARS to FD correlation. A better estimator
   finds more of it (0.48 and 0.87 mm). It must be applied as a correction and must not be censored.
3. The per-slice-group phase-encode field term and the per-point residual scale are marginal on BOLD (0.04 to 0.06 mm,
   under 1 % alignment change). They stay optional and are not on the critical path.
4. With the alternation band removed from the pose series, FD is 0.14 mm (6223) and 0.19 to 0.28 mm (3315) under every
   estimator, and censoring at 0.5 mm rejects 0 to 5 % of frames instead of 27 to 71 %.
5. The released head prior has three implementation defects (`motion_batched.py`): the Newton stage evaluates the
   image loss alone in its gradient, Hessian and line search, so the prior is dropped in the final stage; no stage
   carries the prior in its curvature; the gauge re-split after each stage is outside the objective. The 0.1 mm
   displacement on the flat phantom has not been isolated to one of these.

## 2. The v2 model

Notation as in `MOTION_JOINT_OPTIMIZATION_PLAN.md` section 1. One point set `x` in the anchor space, warped intensities
`e_t(x) = I_t(A(C g_t) T(m_t) x)` with the eddy geometry `A` applied as now.

```
J(m) = sum_x rho( || P_D^perp e(x) / sigma(x) || )  +  lambda_h sum_t sqrt(delta^2 + |D2 m_t|^2)
P_D^perp = I - D (D' W D)^-1 D' W           (temporal projector, closed form per point)
```

- `D` (T x K) is the known acquisition design, profiled out per point: intercept per level (ASL label, DWI shell, slow
  flow state, BOLD intercept), task regressors convolved with the HRF for task BOLD, and for DWI the real symmetric
  spherical harmonics of `g_t` per shell (order 2 up to b = 1500, order 4 above, never more than a third of the
  directions on the shell). No polynomial drift columns (slow motion would fall into them); scanner drift is a
  spatially constant gain per frame, removed up front.
- Levels differ in contrast, never in pose. There is no per-level pose offset and no cross-contrast level
  registration: a constant pose offset between levels is a nullspace of the within-level metric, and the Charbonnier
  prior resolves it to zero at every level switch. This retires `cross_link` and `_group_offsets` for v2.
- `sigma(x)` is the robust per-point residual scale (optional, default on; measured marginal on BOLD, expected to
  matter for DWI CSF).
- The per-slice-group phase-encode field shift is not part of v2 (decision 4); the pose-dependent field (unwarp) is
  out of scope.
- The prior is a term of the objective in every stage. Newton uses its analytic Hessian. No re-split step exists.
- FD is reported from the rigid part, with the eddy translation and the field term reported separately as now.

Precedent: Orchard et al. 2003 (joint pose and activation), Andersson et al. 2001 (field as a deformation, not
motion), Huizinga et al. 2016 (group-wise registration with a low-rank temporal model; ours uses known temporal
factors), Power et al. 2019 and Fair et al. 2020 (respiratory contamination of FD; band-limited FD for censoring).

## 3. Work packages, syntx

| WP | content | files | gate |
|---|---|---|---|
| S1 (done 2026-10-27) | Prior in the Newton stage: gradient by autograd, analytic per-frame Hessian block `surround_prior_hessian` (checked against autograd), per-frame attribution in the line search; L-BFGS already carried it through its closure. Isolation: the displacement enters in L-BFGS with the prior on as a near-linear rotation drift about the blob centres (prior nullspace x image plateau); Newton cannot leave a plateau it reports as converged; texture removes it (0.17 to 0.01 mm) | `motion_batched.py` (`_surround_reg_loss`, Newton block), test in `tests/test_motion_design.py` | zero-motion blob phantom stays at 0.00 mm at weight 0.02; structured phantoms do not regress |
| S2 | `engine="projected"`: new module `motion_joint.py` built from the pilot (point pool via `PointSampler`, `physical_to_normalized_torch`, Huber, L-BFGS then Newton), design builder `projected_design(design, bvals, bvecs, layout)` with the SH rule, result = `GroupedMotionCorrectionResult` with `design_rank`, `fd`, `fd_raw`, per-point `sigma` summary | `motion_joint.py`, `motion_design.py` (`ENGINES`), `__init__.py`, tests | section 5 |
| S3 (done 2026-10-27) | Band-limited FD: `band_limited_fd(translations, rotations, tr, band=None)` with automatic band = the pose spectrum peak above 0.1 Hz plus or minus a quarter of the Nyquist frequency; result keys `fd_bandlimited`, `fd_alternation_amplitude_mm`, `fd_alternation_hz`, `fd_band_hz`; `motion_correction` and `motion_correct_design` fill them whenever `tr` is known | `motion.py`, `motion_design.py`, tests | on the two breacher series: 0.14 and 0.19 to 0.28 mm; on a synthetic series with a known 0.3 Hz oscillation the band is found within one bin |
| S4 | Misattribution diagnostic: per-series principal angle between the six rigid-derivative images and the design-locked pattern space at the solution; added to the cohort gate | `benchmarks/motion_cohort_gate.py` | reported, no threshold |
| S5 | Gate script extensions: phantom-with-truth table for `projected`; cohort gate with `fd_bandlimited`; pilot BOLD table as a regression test | `benchmarks/`, `docs/reports/` | section 5 |
| S6 | Retire `debias_pe_pseudomotion` as a censoring aid once S3 ships (keep the function; its docstring already warns it under-counts real nodding) | `motion.py` docstrings, `MOTION_BOLD.md` | docs |

## 4. Work packages, censoring repair across repos

Principles, from `antsxqc/RESPONSIBILITIES.md`:

- The censoring series is `fd_bandlimited` for BOLD and ASL (periodic physiology faster than the sampling); for DWI it
  is the rigid `fd` with the design removed (no steady physiological oscillation at a DWI TR). Raw FD stays reported.
- The threshold is an algorithm parameter, defined once (`antsxqc.metrics.FD_SCRUBBING_THRESHOLD_MM = 0.5`) and
  imported everywhere; every local copy and literal goes. Its value is kept at 0.5 mm on the band-limited series until
  gate C5 says otherwise; 0.3 mm is the conservative alternative to be measured, not adopted.
- Verdicts exist only in antsxqc. The one local verdict found (antsxslowflow `scireport_gly.py` `QC_THRESHOLDS` 0.20 /
  0.50 and `_fd_status`) is replaced by `antsxqc.grade`.
- Every new or existing FD key is in the catalog (`antsxqc/specs.py`): `rsf_fd_raw_*`, `rsf_fd_bandlimited_*`,
  `rsf_fd_alternation_amplitude_mm`, `rsf_fd_alternation_hz`, `perf_fd_*`, `perf_fd_bandlimited_*`,
  `perf_tag_control_offset_removed_*`, `dti_FD_raw_mean` note corrected ("full applied pose including the eddy
  translation", not "unfiltered").

| WP | repo | change | files |
|---|---|---|---|
| C1 (done 2026-10-27) | antsxqc | catalog entries and aliases above; `high_motion_fraction` documented as computed on the censoring series; `fd_bandlimited` measurement helper is NOT here (it is an algorithm, syntx S3) | `specs.py`, `metrics.py` docstring |
| C2 (done 2026-10-27) | antsxfunctional rsfMRI | censor and spike on `fd_bandlimited`; raw full poses remain the nuisance regressors (they explain signal; Q7 evidence); fix the `debias_pe_pseudomotion` signature vs docstring mismatch (`pipeline.py:99` vs `:190`); remove the stale "no censoring step" text (`report.py:451-455`); `fd_mean` excludes frame 0 as in syntx; wide-row columns for the new keys | `rsfmri/pipeline.py`, `rsfmri/report.py`, tests |
| C3 (done 2026-10-27) | antsxfunctional task | `fd_spike_threshold_mm` applied to `fd_bandlimited`; import the constant | `task/design.py`, `task/pipeline.py` |
| C4 (done 2026-10-27) | antsxfunctional perfusion | replace the literal 0.5 (`perfusion/pipeline.py:517`) by the import; `fd_pair_scrubbing` wired as an opt-in on `fd_bandlimited`; report line and figure use the same series | `perfusion/pipeline.py`, `outlier_rejection.py`, `figures.py`, `report.py` |
| C5 | antsxfunctional gate | downstream outcome on the cohort and on a repeated-session dataset: grey-matter tSNR after nuisance regression, DVARS to FD slope, test-retest connectivity for censoring on raw vs band-limited FD at 0.3 and 0.5 mm; HTML report; this gate chooses the default threshold | `benchmarks/` |
| C6 (done 2026-10-27) | antsxdwi | delete the local `FD_THRESHOLD = 0.5` (`pipeline.py:64`) and import; flags computed on rigid `fd`, `fd_raw` reported; comment at `motion.py:34` corrected ("flagged, not removed") | `pipeline.py`, `motion.py` |
| C7 (done 2026-10-27; the no-ANTs fallback keeps voxel FD and now records `fd_units`) | antsxslowflow | FD in mm, not voxels (`motion/api.py:95-146`); remove the local verdict (`scireport_gly.py:1810-1843, 2529`, `inference.py:807`) in favour of `antsxqc.grade`; import the constant in `diagnostics/metrics.py:913`; report text reads the threshold from the constant | `motion/api.py`, `reporting/`, `diagnostics/metrics.py` |
| C8 (done 2026-10-27) | antsxmm | `visualize/core.py` threshold imported from antsxqc; report table shows raw and band-limited FD side by side; no new orchestration parameters | `visualize/core.py`, `visualize/report.py` |
| C9 (done 2026-10-27) | antsxqc lint | an R1 rule for cutoffs read from a dict (the `QC_THRESHOLDS` pattern) so C7 cannot recur | `antsxqc/lint.py`, test |

## 5. Gates before any default changes

| gate | criterion | current |
|---|---|---|
| G1 phantom truth | `projected` FD error on the DWI phantom at or below 0.10 mm (joint 0.208, unified 0.438); ASL and slow flow not worse than joint (0.059, 0.068) | joint numbers |
| G2 cohort | readout step within 0.02 mm of unified on every ASL series; PE step not higher than joint; alignment not worse; every frame converged | joint passes |
| G3 BOLD pilot regression | edge temporal SD at or below the pilot (0.0323, 0.0515); `fd_bandlimited` 0.14 and 0.19 to 0.28 mm | pilot |
| G4 zero-motion phantom | textured zero-motion phantom: no point moves by more than the Newton confirm tolerance (0.05 mm at the phantom's 16 mm extent) with the prior on; the untextured three-blob phantom is a documented plateau (rotation about blob centres unconstrained, 0.17 mm drift with every frame reporting converged) and is not a gate | passes (0.01 mm) after S1 |
| G5 downstream (C5) | censoring on `fd_bandlimited` at 0.5 mm keeps at least 90 % of frames on the two breacher series and does not lower test-retest connectivity against raw-FD censoring | not run |
| G6 lint and tests | `antsxqc.lint` 0 findings in all consumer repos; full `pytest tests/` green in each repo, including the pre-existing `test_bold_nyquist_sagittal_ripple_debiased_while_nodding_untouched` (to be fixed or re-specified in S6) | one known failure |

## 6. Order and estimates

1. S1 (one day). Unblocks the prior everywhere and settles whether the flat-phantom displacement was a defect.
2. S3 and C1 (one day). Independent of the engine; removes the breacher inflation from censoring immediately.
3. C2, C3, C4, C6, C7, C8, C9 (two days). Mechanical; each is one commit per repo in the dependency order antsxqc,
   syntx, antsxdwi, antsxfunctional, antsxslowflow, antsxmm.
4. S2 (one to two weeks). The v2 engine, first on the DWI phantom (G1), then the cohort (G2), then BOLD (G3).
5. S4, S5 (two days). Diagnostics and reports.
6. C5 and G5 (one week, needs a repeated-session dataset; `antsxdata` to be checked first).
7. Switch defaults: `engine="projected"` in antsxdwi, antsxfunctional, antsxslowflow only after G1 to G4; the
   censoring threshold only after G5.

## 7. Decisions taken (2026-10-27)

1. Engine name: `projected`. No backward compatibility: `joint` and its gauge machinery (`cross_link`, `_group_offsets`,
   the per-level pose `beta`) are removed when `projected` passes G1 to G4, and the downstream defaults switch in the
   same release train.
2. Censoring default: 0.5 mm on `fd_bandlimited` for BOLD and ASL, 0.5 mm on rigid `fd` for DWI, from
   `antsxqc.metrics.FD_SCRUBBING_THRESHOLD_MM`. G5 may lower it to 0.3 mm later; it does not block C2.
3. Band detection: automatic from the pose spectrum (peak above 0.1 Hz, plus or minus a quarter of the Nyquist
   frequency); a recorded respiratory trace, when present, overrides it; `fd_band_hz` and `fd_band_source` record
   which path was used.
4. The optional phase-encode field term is dropped from v2 (measured 0.04 to 0.06 mm and under 1 % alignment change on
   both breacher series). The pilot script is the record; it is revisited only on a multiband series with a recorded
   breathing trace.
