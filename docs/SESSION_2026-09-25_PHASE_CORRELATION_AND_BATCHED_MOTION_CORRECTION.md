# Session Report: FFT Phase-Correlation Candidates, Zero-Failure Large-Jump Capture, and a Validated Batched Multi-Frame Motion-Correction Prototype

**Date**: 2026-09-25
**Author**: syntx Core Team
**Scope**: Follow-on to the same day's earlier torch-native motion/template work
(`docs/SESSION_2026-09-25_TORCH_NATIVE_MOTION_TEMPLATE_AND_SYN_FIXES.md`). Closes the
`motion_correction` real-data parity gap with a ground-truth simulation harness, builds and
validates a batched (whole-time-series-at-once) rigid motion-correction prototype that beats
`ants.registration` on wall-clock, diagnoses and fixes a real large-jump capture-range
failure down to zero observed failures, and generalizes the fix (FFT-based phase
correlation as a translation-init candidate) into `robust_affine.py` production code.

---

## 1. Why: real-data parity was unverified

`motion_correction`'s `backend='pytorch'` default (added in the prior session) was validated
on one real DWI b0 series and one synthetic phantom -- not a controlled, repeatable
ground-truth benchmark. Any conclusion drawn from real acquired frames is inherently
confounded: you can't separate "the algorithm is wrong" from "the ground truth is unknown."

## 2. Ground-truth simulation harness (`scripts/bench_motion_recovery_methods.py`)

Builds cached, reusable test series from REAL clinical mean images (`mean_b0.nii.gz`,
`mean_dwi_aligned.nii.gz`, themselves built from real clinical DWI data via
`syntx.motion.motion_correction(reference='mean', two_pass=True)`), with:
- A KNOWN, randomly injected rigid transform per simulated frame (recoverable exactly, since
  `apply_transforms(fixed=base, moving=base, transformlist=[tx])` warps by a known `tx`; the
  transform any registration method recovers should approximate `tx`'s inverse -- verified
  algebraically and empirically this session after finding and fixing an earlier inversion
  bug).
- Realistic corruption (`ants.simulate_bias_field` + `ants.add_noise_to_image`) so recovery
  is tested against real-texture, real-noise data, not idealized synthetic phantoms.
- Three trajectory modes, all now permanent CLI options: i.i.d. per-frame jitter (default),
  `--smooth` (Ornstein-Uhlenbeck random-walk, temporally correlated motion), and `--jump`
  (small jitter, one large abrupt displacement mid-series, small jitter after -- "the
  participant coughs and shifts, then holds still").
- Optional injected DWI-vs-b0 group-level bias (`--group-trans-mm`/`--group-rot-deg`).
- A fast candidate-registration-method benchmark harness (`--methods`) with a persistent
  cache keyed by a hash of all generation parameters, so new methods/schedules can be
  evaluated in seconds against a fixed ground-truth set instead of regenerating it.

Standard, permanent evaluation step (`fd_from_rigid_params`/`fd_agreement_report` in
`scripts/prototype_batched_motion_correction.py`): computes Power/Jenkinson framewise
displacement from recovered vs. ground-truth rigid parameters using `syntx.motion`'s own
`_extract_rigid_parameters`/`calculate_framewise_displacement` directly (not reimplemented),
so recovery is validated against the actual downstream metric real `motion_correction` users
read, not just raw per-parameter error (which can hide structured mistakes that partially
cancel or compound in the frame-to-frame difference FD measures).

## 3. Batched multi-frame rigid registration (`scripts/prototype_batched_motion_correction.py`)

`motion_correction`/`robust_affine`/`ants.registration` are all sequential: N frames means N
independent calls, each paying a large, roughly-constant per-call overhead (Python dispatch,
autograd graph construction, MPS kernel-launch latency -- established this session as the
dominant cost at this problem size, not raw compute). A batched torch solver processes all N
frames' rigid parameters jointly: one `F.grid_sample` call samples all N moving volumes
against the shared reference in parallel, one batched Mattes-MI evaluation, one backward
pass, one optimizer step per iteration -- the fixed overhead is paid once for the whole
series, not N times.

### 3.1 What didn't work, tried and discarded (recorded so it isn't retried blind)
- Dense (every voxel) coarse pyramid levels: diluted the coarse MI histogram with
  uninformative background/air voxel pairs (no foreground restriction). Fixed by combining
  true `avg_pool3d` downsampling WITH foreground-masked point-sampling, not one or the other.
- A coarse-level axis-order bug (`warp_to_moving_norm` mixed full-resolution spacing with the
  pooled grid's shape when normalizing to `grid_sample` coordinates) caused catastrophic
  (~20-40mm) errors specifically at `level>1`; verified the underlying physical-coordinate
  formula was exact (diff=0 against a manual ground-truth computation) before finding this --
  the bug was in the moving-grid normalization step, not the coordinate math itself.
- Robust p2/p98 percentile intensity normalization (matching `normalize_image`, used by the
  single-frame solver): made things WORSE (0.242mm vs 0.159mm mean error on the same test),
  reverted.
- More Mattes MI histogram bins (44 vs 32, compensating for `parzen_weights`' 2-bin edge
  padding): no improvement, just slower.
- A temporal smoothness prior (penalizing frame-to-frame parameter change): regressed
  accuracy at every tested strength (1.0, 0.05, 0.005) -- **left unresolved**, not shipped.
  Comparing sulceye's independent `mattes_mi_loss_2d` reimplementation against syntx's found
  no algorithmic smoking gun (same B-spline Parzen approach), ruling that out as the cause.

### 3.2 What worked
- Dropping the multi-start cone-search tournament (single Identity_CoM start) -- confirmed
  early: for motion-correction-scale offsets, capture range isn't the bottleneck precision
  is, mirroring the `robust_affine` `dof='rigid'` finding from the prior session.
- Coarse-to-fine schedule: Adam at real `avg_pool3d`-downsampled levels 4 and 2 (broad,
  cheap capture) -> LBFGS at level 1 with point-sampling (precise polish).
- A **dual-metric** correlation term added to (not replacing) Mattes MI at the coarse stages:
  well-motivated specifically for INTRA-subject motion tracking (not the general
  cross-contrast case), since consecutive frames of the same subject/modality share
  near-identical contrast, so correlation's linear-relationship assumption is actually
  satisfied, not approximated, and its loss surface is far smoother (no histogram binning)
  than MI's at coarse resolution.

Result (40-frame series, real clinical b0/DWI contrast, simulated corruption): **~0.27-0.29s
per frame amortized, ~3x faster than `ants.registration('Rigid')` (~0.8s/frame)**, with
0.09-0.16mm / 0.07-0.17deg mean recovery error and FD correlation r=0.97-0.999 against
ground truth. Pressure-tested on a second, entirely fresh subject (sub-Blast-02, never used
during any tuning) and a fresh random seed: held up, even improved slightly -- not an
artifact of overfitting the schedule to one dataset's corruption realization.

## 4. Large-jump capture-range: root-caused to zero failures

Initial jump-scenario testing (small jitter, one large ±15mm/±15deg displacement mid-series,
small jitter after) showed real, measurable degradation vs. the small-jitter baseline: mean
error 0.76-1.41mm (vs 0.09-0.36mm baseline), with 2-3 outlier frames per 30 reaching 1-5mm.

### 4.1 Root cause, diagnosed not assumed
Adding the dual-metric correlation term (even at 8x weight) to the SAME coarse optimization
made no measurable difference -- ruling out "wrong loss landscape shape" as the cause.
Direct measurement: weighted center-of-mass translation init was off by 7-11mm for the
specific outlier frames (vs 1.4-1.5mm for well-behaved ones) -- a real CoM failure, plausibly
from the bias-field corruption skewing the apparent centroid, with no signal downstream that
it happened. No fixed-iteration-budget coarse optimization reliably escapes a start that far
off.

### 4.2 Fix: FFT-based phase correlation as an independent translation-init candidate
A second, globally-exact-for-pure-translation estimator (one real FFT per frame at the
coarsest pyramid level) -- immune to CoM's specific failure mode since it doesn't compute
any weighted centroid. Top-k peaks (not just the best) extracted via simple non-max
suppression, since a single peak can be fooled by real anatomical structure. Candidates
scored via correlation, not MI (diagnosed directly: Mattes MI at the coarsest level failed to
rank the KNOWN true translation above a confidently-wrong CoM guess on at least one outlier
frame, while correlation correctly ranked truth above CoM on the identical candidates --
matching the general "correlation is smoother/more reliable than MI at very coarse
resolution" theme from Sec 3.2, now load-bearing rather than just a nice-to-have).

Reduced outliers substantially (0.842mm -> 0.255mm mean on the hardest combined jump test)
but did not eliminate the last two: further diagnosis found phase correlation's pure-
translation assumption itself breaks down when a frame ALSO has a large rotation (the moving
content isn't a simple shifted copy of fixed anymore) -- confirmed directly by checking the
correlation-surface rank of the true translation shift for an outlier frame: 147th out of
31,850 candidate grid points, i.e. genuinely not findable by widening top-k reasonably.
Fixed by recognizing that ROTATION needs its own capture range too (a 20-iteration
rotation-only fit starting from identity also failed to reliably find a real ~15deg
rotation), and that fitting rotation with translation held frozen at a still-wrong value is
its own chicken-and-egg trap. Final fix: pair each translation candidate (CoM + phase-
correlation peaks) with a small set of rotation seeds (identity + six ±20deg single-axis
rotations -- a narrow, cheap version of the classic cone-search idea, used only for this
init-disambiguation step, not reintroduced into the main optimization), and run a SHORT
JOINT (both t and omega free) correlation-objective refinement from each seed pair, picking
the overall best result per frame.

**Result: zero failures on the combined 15mm/15deg jump test** -- all 30 b0 frames converged
to <=0.31mm error (mean 0.105mm/0.066deg, matching the clean-baseline numbers), FD
correlation r=0.9997 (Power and Jenkinson). DWI settled at ~0.6mm mean / ~1.1mm worst-case,
which matches DWI's own already-established baseline floor (different gradient directions
have genuinely different contrast/SNR, a pre-existing, unrelated finding from the earlier
session) rather than being a residual jump-capture failure -- verified by testing whether
denser rotation seeds helped (they didn't; cost roughly doubled for negligible gain, reverted
rather than shipped).

**ants.registration comparison on the identical jump cache** (apples-to-apples, including FD
this time): ants remains slightly more accurate (b0: 0.084mm/0.026deg vs our 0.105mm/0.066deg;
DWI: 0.530mm/0.101deg vs our 0.615mm/0.238deg; FD r >=0.995 for both on both groups) but our
batched solver is ~22-27% faster per frame (0.64-0.66s vs 0.84-0.87s) while processing the
whole series in one joint call rather than N sequential ones.

## 5. Generalized into production: `robust_affine.py`

`_phase_correlation_translation_candidates()` (new function) and its wiring into
`_run_pytorch_affine_solver`'s real multi-start candidate pool -- the actual production path
used by `motion_correction`, `auto_reg`, and `build_template` alike (note: an existing,
similarly-named `_generate_quick_search_candidates` function was investigated first and found
to be dead code, referenced only by a coverage test, not by any live registration path; the
real candidate generation lives inline in `_run_pytorch_affine_solver`). Each
phase-correlation candidate gets the same rotation-cone sweep as CoM/FOV candidates already
did, not just identity rotation, per the Sec 4.2 lesson.

Zero regression: full existing suite (`test_robust_affine_coverage.py`,
`test_robust_affine_general.py`, `test_affine_known_transform.py`,
`test_affine_reproducibility.py`, `test_adversarial_affine_stress.py`, `test_core_affine.py`,
`test_motion.py`) -- 53/53 passed unchanged. Five new tests added
(`tests/test_phase_correlation_candidates.py`): the estimator's translation-recovery
accuracy, distinctness of top-k candidates, graceful 2D fallback (not implemented for 2D;
must return `[]`, not raise), `robust_affine` integration with no regression on an ordinary
case, and the motivating large-translation-recovery case as a fast, always-run regression
guard (distinct from the slower, real-clinical-data ground-truth scripts under `scripts/`).

## 6. Status / what's NOT done

- The batched multi-frame solver (`scripts/prototype_batched_motion_correction.py`) is
  validated but **not yet wired into `syntx.motion.motion_correction`** as a real backend
  option -- next planned step, not started as of this commit.
- The temporal smoothness prior (Sec 3.1) remains broken; not shipped, not further debugged
  this session.
- `scripts/prototype_batched_joint_group_bias.py` (shared cross-contrast group-bias
  parameter, jointly estimated with per-frame jitter) is validated on synthetic ground truth
  (beats the old two-stage sequential pipeline on translation: 1.24mm vs 2.32mm) but likewise
  not integrated into any production path.
