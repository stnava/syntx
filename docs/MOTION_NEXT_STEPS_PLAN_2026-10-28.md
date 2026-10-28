# Plan, 2026-10-28: one solver, four cases, an ISMRM abstract

Goal (order revised 2026-10-28 afternoon: the work in progress is finished before the abstract is touched again; the draft
and its PDF exist under `docs/abstracts/` and wait): an ISMRM abstract that presents one motion solver for every time-series modality, with the acquisition design in
its objective, a statistical account of what the model explains, and the open software (syntx, antsxdwi,
antsxfunctional, antsxqc). Every claim in it is backed by a number from one of four cases.

| case | series | role |
|---|---|---|
| resting BOLD | breacher sub-3315, TR 1.0 s, 300 volumes | FD inflation, band-limited FD, group-wise model; day-1 / day-2 repeat available for test-retest |
| ASL | FPA sub-BLAST071 ses-01 (FD 0.79 mm; the largest label-locked pose component in the cohort, 0.078 mm along the slice axis) | high motion and the misattribution case: the design absorbs the label-locked jitter, motion explains what is left |
| DWI, high motion | FPA sub-BLAST031 run-001 dir-AP (FD 0.61 mm) | large rotations, eddy, the readout-floor test, pose-dependent dewarp |
| DWI, low SNR | SOCOM sub-Blast-11 run-001 dir-PA (FD 0.52 mm) | b = 3000 near the noise floor, the design-richness test |

Rules: one step at a time; each step is implemented, run on the cases it concerns, and ends with an HTML or PNG that we
look at together before the next step; every number comes from a JSON written by a `benchmarks/` script; targeted
tests only, the full suites after the abstract; no default changes without the step's own gate.

## Decisions taken with the author (2026-10-28, after step 2)

1. Joint stays the DWI default; the projected engine stays available, not default, everywhere (one engine, one story).
2. A per-slice intensity term (odd/even slice parity, interleaved acquisition) gets its own step, run before step 3: measure its
   share and its link to the pose on the two DWI cases, then decide whether it joins the design.
3. Unified stays available as the comparison baseline.
4. The full test suites run before the next release, after the WIP steps.
5. The abstract's ASL row was to use a longer series than the 25-frame EAS020 run. Checked 2026-10-28 on the server: every ASL
   series in the four cohorts is the same protocol length (ExpArt and SOCOM 25 volumes, FPA 16, breacher none), so no longer run
   exists here; the 25-frame row stands unless an outside dataset is brought in.

Decisions 2026-10-28 (afternoon, after step 2d):
6. Scanner drift is modelled in the full model, not only in the measurement: a per-frame global gain inside the solver
   (`frame_gain`, on by default in the joint and projected engines; spatially uniform, so it cannot be confused with slow
   motion), refreshed with the design model; the decomposition credits it as a known-pattern term before every block.
7. Shrinkage toward identity (`pose_shrinkage`, likelihood units) is ON at a small weight, set on the cohort by the
   readout-floor rule (`benchmarks/shrinkage_weight_scan.py`), the gauges staying as the exact minimum-norm step in the
   design's span.
8. Next: the weight scan, then the five cases rerun with gain and shrinkage (new table and figure), then step 3.
   Scan notes (2026-10-28): in likelihood units the shrinkage term is sigma^2 / S times mu per mm^2 (about 1e-7 mu on these
   series), so weights below 1e4 do nothing; the acting range is 1e5 to 1e6. Phantoms (joint, CPU, head-FD error): DWI 0.202 at
   mu = 0, 0.080 at 1e5, 0.171 at 1e6; ASL 0.054, 0.076, 0.142 (rotation drift shrunk). Candidate 1e5; the cohort readout-floor
   rule decides. The per-frame gain alone moved BLAST031's readout step 0.080 -> 0.064 mm (unified 0.057) at unchanged FD and alignment.

Finding 2026-10-28 (late): the Newton stage converted its voxel-unit step to parameter units by dividing by dparam/dvox instead of
multiplying (translations 1/9 of the intended step at 3 mm voxels, rotations hundreds of times too large and always rejected). Found
because a shrinkage weight could not change a real BOLD series at any value; the rule now: verify that a parameter moves the
outcome on a short excerpt before any long scan. Fixed; phantom head-FD errors fell for every engine (unified DWI 0.319 -> 0.242,
joint DWI 0.202 -> 0.160, joint ASL 0.054 -> 0.042). Every number produced under 6.24 to 6.26, the released gates and the parked
abstract's table included, is stale until re-measured.

Decision 2026-10-28 (evening): shrinkage toward identity ON at 0.1 in curvature units for the joint and projected engines
(`JOINT_POSE_SHRINKAGE`), chosen on the five-series cohort scan by the readout-floor rule (`docs/reports/cohort_shrinkage_report_2026-10-28.html`):
FD -11 to -41 %, readout step within 0.023 mm, alignment within 0.7 %; released as syntx 6.27.0 with the Newton unit fix, the
curvature-unit prior scale, the per-frame gain and the decomposition module; downstream pins follow in the same train.

Correctness audit 2026-10-28 (independent agent, harness scripts under the session scratchpad `audit/`): three confirmed
findings. (1) The coupled Newton Hessian counted the roughness prior twice and the shrinkage three times (step too short along
prior directions, early convergence): fixed, the image Hessian now comes from the image gradient alone and the prior's full
Hessian is added once; the audit harness fits multipliers 0.999 / 1.000. (2) The Newton convergence test used the damped step,
so a frame whose line search kept failing was declared converged without moving: fixed, the test uses the undamped step and
Newton rejections are counted in the stage record; the audit's stuck-frame synthetic now reports converged = False. (3) The
written chain `[eddy, rigid]` was resampled rigid-then-eddy while the solver fitted eddy-after-rigid (a transform list acts on
the reference-space point first-listed first, verified with a point source through `ants.apply_transforms` and
`syntx.transforms.apply_transforms`): FIXED 2026-10-28, the chain is written `[rigid, eddy]`, every reader (`motion.py`,
`motion_design._joint`, `motion_reference` forward chains / cross-group debias / nuisance, `motion_simulation`) selects the
rigid by name through `rigid_transform_of` / `is_eddy_transform`, and `test_joint_dwi_eddy_applied_...` and the projected
sibling assert that resampling the written chain equals one resampling through the single matrix of the fitted model and
differs from the opposite order. The same audit found `TransformChain.linear_part` composing in the reverse order of the
point mapper (`_map_points_to_moving_tensor`), which rotated bvecs through the wrong composition for any chain with two
non-commuting linear elements (eddy + rigid, crosses + rigid): fixed, with `tests/test_transform_chain_order.py`. antsxdwi
reads whole chains through `TransformChain` and needs only the syntx pin. Smaller notes: per-level point weights are overwritten by the design-model weights in the projected engine
(dormant unless `level_weights=True`); the one-level init centre differs from the core's when frame 0 is not in the anchor
level (0.02 mm); the core's level gauge has no interleaving guard for direct callers; `fd_raw_power` is misnamed under the
Jenkinson method. All passed checks are listed in the audit report in the session transcript.

## Steps

**1. Baseline (joint engine) table and figure. DONE 2026-10-28** (`docs/reports/ismrm_cases_2026-10-28.html`; `syntx.motion_decomposition` with tests). Run `engine="joint"` on the four cases; per case: FD mean / max,
frames above 0.5 mm on raw and band-limited FD, pose-domain shares on the readout axis (head / applied acquisition
term / removed design-locked term), image-domain variance explained (design, motion after the design, residual) on
raw and corrected, residual-to-FD correlation, time. Figure: one row per case (pose along readout and PE with the
applied term, FD with band-limited FD, variance bars). Compute about 10 minutes. Look at: the table and the figure.
Decide: which numbers carry the story; whether the decomposition becomes a library module now (it must, for the
abstract's software claim: `syntx.motion_decomposition`, unit-tested on a synthetic series).

**2. The DWI design decision. DONE 2026-10-28** (`benchmarks/dwi_design_decision.py`, `docs/reports/dwi_design_decision_2026-10-28.html`): no richer design passes the rule; joint stays the DWI default. The l = 4 terms change nothing (BLAST031 readout step 0.146 -> 0.144 mm with projected, l = 4 share 0.86 -> 0.84 %); the smallest rigid-vs-design angle is 51 to 78 degrees, so the design is not what the pose absorbs. On SOCOM Blast-11 every engine reports the same readout jitter (0.127 to 0.132 mm): it is in the data. On BLAST031 it is engine-dependent (unified 0.057, joint 0.080, projected 0.145): a metric and reference effect on volumes that may not be rigid (slice-to-volume motion, dropout). Step 2b DONE (`benchmarks/slice_residual_diagnostic.py`, `docs/reports/slice_residual_diagnostic_BLAST031_2026-10-28.png`): the slice-stripe index of a volume's residual does NOT follow its readout step (r = -0.15 unified, -0.10 projected); the b = 3000 volumes have the largest steps and the lowest stripe index, so intra-volume motion and dropout are not the cause. The engine-dependent readout jitter is on diffusion-weighted volumes only (b0 0.000, b1500 0.06, b3000 0.10 mm), white in time, and absent from slice structure: per-volume estimator scatter of the Huber metric against a noisier reference at low SNR. A separate finding from the heat maps: an odd/even slice intensity alternation in the residual of every volume (interleaved acquisition, spin history), a per-slice intensity term no engine models. Step 2c DONE: the engine-dependent readout jitter of the projected engine is design-locked after all, a pose component in the span of the direction columns (BLAST031: 63 % of its readout translation explained by the even-in-g terms, joint's 4 %): a design column in the metric is a nullspace of the design-in-metric objective and the pose drifts into it, as it did for the level offsets. Gauging every design column after each stage (`design_gauge`, `full_gauge=True`, gate `METRIC_FRAMES_PER_COLUMN = GAUGE_FRAMES_PER_COLUMN = 8`) halves the readout step on BLAST031 (0.146 -> 0.077 mm, joint 0.080) but the gauge's own estimation noise raises the PE step (0.63 -> 0.91) and FD (0.72 -> 0.84), and on SOCOM Blast-11 it raises the readout step (0.132 -> 0.167). Verdict unchanged: joint stays the DWI default; its architecture (the columns in the pose, pinned by the data, then gauged) is the right one for DWI, and the projected engine's phantom advantage does not transfer to real multi-shell data. Decision record: `docs/reports/dwi_design_decision_2026-10-28.html`. Implement l = 4 per shell for b >= 2500 in the design (`projected_design_matrix` and
the `joint` level columns) plus the misattribution-angle diagnostic. Run unified / joint / joint-l4 / projected-l4 on
the two DWI cases. Look at: readout step, PE step, FD, alignment, l = 4 share after the engine, misattribution angle,
one HTML. Decision rule, fixed now: a richer design wins only if the readout step is within 0.02 mm of unified's on
both cases, the residual share is not higher, and the l = 4 share after the engine falls below the motion share.
Compute about 20 minutes.

**2d. Per-slice intensity term (decision 2). DONE 2026-10-28:** the odd-minus-even slice contrast of the residual is 1 to 2 % of the residual per volume, explains 0.17 % (BLAST031) and 0.12 % (SOCOM Blast-11) of the voxel variance as a free-pattern term, correlates with pitch at r = -0.24 / -0.28 and with FD at -0.04 / -0.09, and leaves the motion share unchanged (0.47 -> 0.43, 0.85 -> 0.85 %). Too small for a design column; recorded, not modelled (`docs/reports/slice_parity_*_2026-10-28.png`). The banding in the heat maps is a static per-slice noise-level difference, not a temporal term.** `benchmarks/slice_parity_term.py`: per volume, the odd-minus-even slice residual contrast c(t) after the shell model; its share of the voxel variance when added as a temporal regressor with a free spatial pattern; its correlation with FD and the pose. Look at: one figure. Decide: a parity column in the design, or out of scope.

**2e. Shrinkage toward identity (author's addition, 2026-10-28).** A penalty mu (|t|^2 + |50 omega|^2) per frame in likelihood units, inside every L-BFGS and Newton stage with its analytic Hessian: the minimum-norm pose in every nullspace the roughness prior ignores (a level offset, anything the design columns absorb, a global offset against the reference). Test: does it replace the level and direction gauges on the ASL and DWI phantoms and on BLAST031 without harming real motion? Then the weight by the readout-floor rule.

**3. Per-volume phase-encode deformation.** Step 3a, diagnostic: on both DWI cases, the per-volume residual PE
displacement (PE-only low-order alignment of each corrected volume to its shell-model prediction) regressed on g,
g g^T and the pose; one figure of the shares. Look at it. Step 3b, only if 3a shows a pose-dependent share: compose
the static susceptibility dewarp per volume with the rigid pose in the antsxdwi chain (one interpolation), optional,
gated on AP/PA b0 edge agreement and the residual share on BLAST031. Step 3c, only if 3a shows a g-quadratic share: a
quadratic eddy basis in the solver's PE term. A free per-volume field is excluded: no physics anchors it.

**4. Band recalibration.** On the breacher 3315 series and the 70-series cohort JSON already written: choose the
prominence threshold and band width so that series with a physiological alternation (lag-one step autocorrelation
below -0.6) get a band and white-jitter series do not; synthetic tests for white, drift, respiratory and cardiac-alias
series. Look at: the cohort histogram of band applied vs lag-one autocorrelation, before and after. Compute minutes.

**5. Downstream gates, one per modality, on these cases.** DWI: tensor fit residual (and FA where a repeat exists)
for unified vs the step-2 winner on both DWI cases. BOLD: grey-matter tSNR after nuisance regression and day-1 vs
day-2 connectivity reproducibility on breacher 3315 under raw-FD vs band-limited censoring at 0.3 and 0.5 mm. Look
at: one HTML per modality. Decide: the censoring threshold and whether the step-2 winner becomes the DWI default.

**6. The abstract.** `docs/abstracts/ISMRM_2027_motion_solver.md`: synopsis, introduction, methods (the objective, the
design per modality, the prior in likelihood units, band-limited FD, the decomposition), results (step 1 table and
figure, steps 2 to 5 as one line each, the phantom-truth line from the released docs), discussion (misattribution,
limits), conclusion, software (repos, versions, one command per modality). Claims proportioned to the evidence.

**7. Land.** The solver work from today (likelihood-unit prior, coupled Newton) and the modules from steps 1 to 4
committed one package per commit with their targeted tests; tracking entries; the full suites run before the release
that follows the abstract.

## Shipping the default solver (2026-10-28, evening)

**What is fixed and ships now (syntx 6.27.0, antsxdwi pin).** The Newton unit conversion, the prior Hessian counted once,
convergence judged on the undamped step, the chain order `[rigid, eddy]` with name-based readers, `TransformChain.linear_part`
in application order, the curvature-unit prior, the shrinkage term, the per-frame gain, band-limited FD consistency and the
decomposition module. The chain-order fix changes resampled DWI volumes and bvec rotation on every eddy-corrected series, so
antsxdwi must pin the new syntx in the same release train; FD and the readout-floor numbers are unchanged (both read the
rigid file alone).

**What the evidence says about the prior defaults.** Real cohort (five series, `docs/reports/cohort_shrinkage_report_2026-10-28.html`):
shrinkage 0.1 lowers FD 11 to 41 % with the readout step within 0.023 mm and alignment within 0.7 %. Phantoms with the
fixed Newton (`scratchpad/phantom_table2`, FD error in mm, ASL / DWI): joint 0.042 / 0.157; shrinkage 0.03 0.100 / 0.081;
shrinkage 0.1 0.129 / 0.113 (rotation error 0.058 -> 0.362 deg); roughness 0.1 0.068 / 0.103; roughness 0.1 + shrinkage 0.03
0.113 / 0.083; roughness 0.3 0.084 / 0.087. Shrinkage at ANY weight pulls a drifting trajectory toward identity: the
ASL phantom (drift on) loses 2 to 3x, the antsxslowflow trajectory test fails at 0.060 mm against its 0.05 budget, and three
syntx phantom tests fail under the 0.1 default. The real-data FD reduction cannot distinguish suppressed jitter from
suppressed drift, so the readout rule alone does not license the default. Roughness penalises second differences only: drift
is free, jitter is damped, and it is the only prior that helps DWI without hurting ASL.

**Decision (author, 2026-10-28 evening): ship 6.27.0 with the critical fixes and the priors AVAILABLE but OFF by default** (`JOINT_POSE_SHRINKAGE = 0.0` set; the rest of this paragraph is the reasoning as recommended) (`JOINT_POSE_SHRINKAGE =
0.0`, `JOINT_SURROUND_WEIGHT = 0.0`): the behaviour every passing phantom test and the released consumer gates were written
against, now on a correct solver. Decide the default prior in its own release (6.28.0) from a gate that can see bias: the
three phantoms (ASL drift, DWI eddy, slow-flow trajectory) must not lose more than 10 % against no prior, AND the real cohort
must keep the readout floor and alignment within the scan's tolerances. Candidate for that gate is roughness 0.1 in curvature
units: run against the three contested syntx phantom tests on 2026-10-28 it passes the two projected tests and the DWI eddy
test but fails the ASL one (FD error 0.085 mm against the 1.1 x unified budget of 0.077; no prior gives 0.042), so it is
not free either; the cohort FD effect is small but in the right direction; shrinkage is a per-call option for
series the user judges drift-free. If the author prefers to keep shrinkage 0.1 as the default now, the three syntx phantom
tests and the antsxslowflow trajectory test must be re-budgeted with a written justification, which the protocol forbids
without instruction.

**Release train once the default is decided.** syntx 6.27.0 (static version in `pyproject.toml` and `src/syntx/__init__.py`;
targeted tests: `test_motion_design`, `test_motion_reference`, `test_transform_chain_order`, `test_band_limited_fd`,
`test_motion_decomposition`, `test_implementation_standards`, lint 0) -> antsxdwi 0.33.0 (pin; `test_motion`,
`test_transforms`, `test_pipeline_e2e` minus the three pre-existing MPS / legacy failures recorded below) -> antsxfunctional
0.23.0 -> antsxslowflow 2.12.0 (`test_motion_slowflow_phantom` must pass) -> antsxmm 4.8.0. Consumer full suites run today
on the pre-fix tree: antsxdwi 3 failed / 562 passed (dewarp foreground-extent check, two legacy-JHU e2e tests whose MPS SyN
loss collapsed: not motion), antsxslowflow 2 failed / 651 passed (the trajectory test above and a simulated-pipeline
quantitative test to be re-run on the fixed tree), antsxfunctional in progress. Targeted syntx run on the fixed tree 2026-10-28 evening: 114 passed; the three
shrinkage-default phantom tests fail; `test_motion_overlapping_window.py::TestOverlappingWindowBOLD::test_bold_nyquist_sagittal_ripple_...`
fails identically at HEAD b90cb1a in a clean worktree (`wiener_shrinkage_eta` list is empty), so it predates this work and is
not a release blocker for the motion fix; it needs its own ticket.

## Engineering rules that hold for every step

Measurements in library modules with unit tests, `benchmarks/` scripts as thin CLIs writing HTML plus JSON under
`docs/reports/`; CPU for numeric tests; no verdict outside antsxqc, every new key in the catalog; physical space
through ANTs with `[warp, affine]` chains and one interpolation; a gate that is not rerun is reported as not run.
