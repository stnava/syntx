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

## 6. Production integration: `backend='pytorch_batched'`

Following this session's work, the validated batched solver was extracted into a new
production module, `src/syntx/motion_batched.py` (`batched_rigid_register_pass`), and wired
into `syntx.motion.motion_correction` as `backend='pytorch_batched'` -- registers every
non-reference frame in one batched call instead of looping `robust_affine`/
`ants.registration` per frame, reusing the existing two-pass/mean-reference/FD/DVARS
assembly code unchanged. 3D only; `type_of_transform='Rigid'` only; `mask` unsupported
(same restriction as `backend='pytorch'`) -- all three checked and raised clearly at the top
of `motion_correction`, not discovered deep inside the solver.

Caught and fixed one more real bug during integration: the coarse `avg_pool3d` pyramid
levels had no minimum-volume-size safeguard, so a small time-series volume (the exact
16-voxel-per-axis phantom already used throughout `tests/test_motion.py`) would hit the
same degenerate-pyramid failure mode found and fixed in `robust_affine.py`'s single-frame
solver earlier this session. Fixed with the same `MIN_VOXELS_PER_AXIS` clamp, verified on
that exact phantom (translation recovered to within 0.03-0.06mm of the known shift).

Real-clinical-data sanity check (7-frame b0 series): accuracy comparable to
`backend='ants'` (var. reduction 56.7% vs 55.3%, FD correlation 0.977) but MEASURABLY
SLOWER (1.5s/frame vs 0.7s/frame) -- the batching advantage requires enough frames to
amortize the (substantial, ~42-combination) large-jump-capture candidate search that always
runs once per call; documented directly in the `backend` parameter's docstring rather than
left as a surprise. Prefer `backend='pytorch'` or `'ants'` for short series (roughly
<15-20 frames) until the candidate-search cost is tuned to scale down for small
`num_frames`.

8 new tests (`tests/test_phase_correlation_candidates.py` covers the standalone estimator;
`tests/test_motion_batched.py` covers the `motion_correction` integration: known-shift
recovery on the small phantom, identity-reference contract, two-pass mean reference, 2D/
non-Rigid/mask rejection, and `batched_rigid_register_pass`'s own edge cases). Full existing
`test_motion.py` suite (15 tests) passes unchanged.

## 7. What's still NOT done

- The temporal smoothness prior (Sec 3.1) remains broken; not shipped, not further debugged
  this session.
- `backend='pytorch_batched'`'s large-jump candidate-search cost doesn't yet scale down for
  small `num_frames` (Sec 6) -- makes it a net loss on short series even though accuracy is
  fine; worth tuning (e.g. skip or shrink the seed search below some frame-count threshold).
- Only `type_of_transform='Rigid'` and 3D are supported by `backend='pytorch_batched'`.

## 8. Chasing the ~0.09mm accuracy floor (exceed-ants goal) -- confirmed NOT a tuning problem

Goal set explicitly: exceed `ants.registration` accuracy, not just approach it, on both
simulated ground truth and real data. On the fresh-subject b0 ground-truth cache
(`78f2c2ce4910`, 30 frames), the single-group batched solver sits at a genuine floor of
**0.092mm / 0.061-0.067deg** mean recovery error (FD Power r=0.998, MAE~0.14mm), vs. ants'
own **0.067-0.072mm**. Tried and confirmed NOT to move this floor:
- More LBFGS iterations, higher point-sampling (up to 80%), an extra 5th polish stage:
  identical 0.092mm every time.
- Adding a correlation-dominant term (`corr_weight=3.0`) to the fine-level LBFGS polish
  stage (previously correlation was only used at coarse pyramid levels): identical
  0.092mm translation error, and slightly WORSE rotation error (0.078 vs 0.061-0.067deg).
  Reverted; not shipped.

Conclusion: the remaining gap is not a compute/iteration/sampling budget problem, and
isn't fixed by rebalancing the MI/correlation loss mix at the fine stage either. Closing it
would need a structurally different final-stage optimizer (e.g. an analytic Gauss-Newton/
Levenberg-Marquardt step using the closed-form rigid Jacobian, instead of autograd-through-
relaxation with Adam/LBFGS) or a higher-precision similarity metric near convergence --
neither attempted yet.

While investigating this, found and fixed a real (if previously latent) bug in
`get_pyramid_level`'s level==1 branch in `motion_batched.py` and the prototype script: when
`sampling<1.0` subsampled `X_stage`/`w_y_stage` via a random index `sub`, `fixed_vals_stage`
was NOT subsetted the same way, so any fine-level stage with `corr_weight>0.0` would have
computed the correlation loss against a size-mismatched `fixed_vals` tensor. Not triggered in
today's shipped schedules (fine-level stages don't set `corr_weight>0` today), but fixed for
correctness before it silently breaks a future schedule change.

## 9. Group-bias joint estimator: integrated into production

`syntx.motion_batched.batched_group_bias_register_pass` (new function, exported from
`syntx`) productionizes `scripts/prototype_batched_joint_group_bias.py`: jointly estimates
per-frame rigid jitter for every frame in a two-acquisition-group series (e.g. b0 + DWI)
PLUS one shared rigid group-bias transform applied only to the biased group, in a single
batched optimization -- replacing the naive two-stage "register the two mean images"
alternative (2.322mm/2.242deg recovery error on a true 3.24mm bias, vs. 1.238mm/1.934deg
for joint estimation).

Integration was not a direct lift-and-shift -- two things tried and discarded before
landing on the validated design:
1. **Seeding per-frame init from the single-group solver's phase-correlation + coarse
   rotation-seed search (`fit_pair`)**: that machinery fits each frame as an ISOLATED rigid
   transform (no group concept), so for group-masked frames it converges near the TOTAL
   offset (frame jitter + group bias combined). Assigning that directly to `t_param`
   (meant to hold only the frame-local component) while `t_group` is ALSO separately
   nonzero double-counts the shared offset. Attempted a fix (subtract `t_group_init` back
   out) -- still measurably worse than the simple CoM-decomposed init (1.5-2.7mm vs
   1.238mm group-bias error across two attempts). Dropped entirely; not worth the added
   complexity and 2-3x runtime cost for a worse result.
2. **True multi-res `avg_pool3d` pyramid coarse stages** (the genuine improvement
   validated for the single-group solver): confirmed to HURT this decomposition task
   specifically -- averaging pools the frame-local and shared-group signal together at
   exactly the resolution where the optimizer most needs to tell them apart. Reverted to
   the prototype's original point-sampled-at-full-resolution schedule (sampling fraction
   only, `level=1` throughout), which reproduced the prototype's exact validated numbers
   (1.238mm/1.934deg) end to end in the production module.

Net result: production `batched_group_bias_register_pass` matches the prototype's
validated accuracy exactly, with a cleaner API (`group_mask: List[bool]`, returns
`(fwd_transforms, inv_transforms, R_group, t_group, elapsed)` in the same
transform-file-path shape the rest of `motion_batched`/`robust_affine` already use). 5 new
tests in `tests/test_motion_batched_group_bias.py` (known group-bias + per-frame jitter
recovery on a synthetic phantom, all-False mask degenerates to zero bias, 2D rejection,
mismatched-length rejection, empty input).

**Not yet done**: this function is NOT wired into `syntx.motion.motion_correction`'s
`backend` dispatch -- that API has no concept of "two acquisition groups" today (it takes
one 4D image, not labeled groups), so wiring it in is an API design question, not a small
follow-on patch. It's currently a standalone, tested, production-quality function; a caller
building a b0+DWI motion-correction pipeline calls it directly.

## 10. Pushing further on "exceed ants" -- translation reaches parity, rotation does not (yet)

Extensive further investigation (num_bins sweep 8-24, dense/base-fraction sampling sweeps,
5-seed ensemble averaging, an *oracle* best-of-5 using ground truth, a *label-free*
best-of-5 using full-image NCC score, Gaussian pre-smoothing of the moving volume at
various sigmas, a rotation-only micro-polish stage, an explicit Gauss-Newton/SSD
refinement, and radius-biased point sampling to favor rotation-informative peripheral
voxels) on top of Sec 8's investigation, tested on TWO independent ground-truth b0 caches
(`78f2c2ce4910`, 30 frames; `9def7593499b`, 40 frames):

**Real, validated, shipped wins** (now production defaults in `batched_rigid_register_pass`):
- `num_bins=18` (was 32): fewer Mattes MI histogram bins gives a smoother, less
  overfit joint-histogram landscape at this point-sample count. Confirmed on both the b0
  ground-truth cache AND an independent DWI-group registration (rotation error
  0.319->0.248deg there) -- not an artifact of one dataset.
- An added final dense LBFGS stage (`sampling=1.0` over the existing point pool, not a
  larger base sample -- a genuinely larger base sample was tested and found NOT to help).

Net effect on cache `9def7593499b` (40 frames): **trans=0.0742mm vs ants 0.0716mm (within
~4%, essentially at parity)**, rot=0.0524mm vs ants 0.0374deg (still ~40% behind).
**Translation has reached ants parity; rotation has not.**

**Confirmed NOT to help** (each a genuine negative result, not a dead end left untested):
- More iterations/sampling/bins beyond the sweet spot: fully converged, identical output.
- 5-seed ensemble averaging (mean or median): 0.0741mm, barely different from a single
  seed -- the per-seed variation is a SYSTEMATIC bias in which local optimum is found. not
  independent zero-mean noise, so naive averaging can't cancel it.
- Oracle best-of-5 (using ground truth to pick per frame): 0.0625mm/0.0491deg -- THIS is
  below ants' translation number, proving real headroom exists in the seed population.
- Label-free best-of-5 (picking per frame by full-image NCC score instead of ground
  truth): 0.0753mm -- statistically identical to a single seed. The similarity metric
  itself cannot distinguish the good seed from the bad one at this precision; the ambiguity
  is invisible to the objective function, not just hard to search. This is the single most
  informative negative result: it means the residual gap is not a search/optimization
  problem, it's what THIS metric (point-sampled Mattes MI + correlation, trilinear
  interpolation) can perceive at all.
- Gaussian pre-smoothing of the moving volume (sigma 0.15-1.2vox) at the final stage:
  small if any rotation benefit at sigma~0.4, but net worse (blurs away the fine detail
  needed for translation precision); no sigma tested gave a clean win on both axes.
- A genuinely dense (not just re-labeled) base point sample (up to 100% of foreground,
  vs the shipped 15%): no improvement, slightly worse, and ~5x slower.
- Explicit Gauss-Newton/Levenberg-Marquardt SSD refinement on top of the converged
  result (closed-form 6x6 normal equations via `torch.autograd.functional.jacobian`):
  OOM'd at full point count; at a reduced 3000-point subset it converged but to a WORSE
  optimum (0.168mm) than the existing MI+correlation schedule -- SSD's landscape is not
  simply "the same optimum found faster," it has different local optima at this scale.
- Radius-biased point sampling (favoring peripheral, rotation-informative foreground
  voxels via `weight = radius^k`): marginal rotation improvement at k=1 (0.0546 vs
  0.0565deg) but net worse once translation's larger regression is counted (0.0824mm).

**Conclusion**: the residual rotation gap vs ants is very likely an information-theoretic
property of the current similarity metric + point-sampling + trilinear-interpolation
stack at this data's resolution/contrast, not a fixable optimization or hyperparameter
issue -- extensive, methodologically diverse search (12+ distinct approaches) could not
move it. Closing it further would most plausibly require replicating ants' own
MattesMutualInformationImageToImageMetric machinery more literally (its continuous
B-spline-derivative gradient computation and dense Gaussian-smoothed multi-resolution
pyramid, rather than avg-pool downsampling + point-sampled Parzen histograms), which is a
substantially larger reimplementation effort, not a quick follow-on.

## 11. Real-data comparison: genuinely exceeds ants (on the trustworthy case)

Ran the actual real clinical BIDS DWI series (`sub-Blast-01`, the same data
`scripts/benchmark_motion_parity.py` uses) through `backend='ants'` vs
`backend='pytorch_batched'` (with this session's num_bins/dense-final improvements),
using the established honest methodology (variance reduction + FD agreement, trusting a
"we did better" claim only when FD correlation is high):

- **7-frame pure b0 series** (no gradient attenuation, same contrast across frames --
  the case this backend's dual-metric design targets): variance reduction
  **59.4% (batched) vs 53.7% (ants)**, FD(power) agreement r=0.94, MAE=0.16mm. High
  agreement means both methods are tracking the SAME true motion, so the batched solver's
  higher variance reduction is a genuine, trustworthy improvement, not divergent/wrong
  motion estimation. **This exceeds ants on real data.**
- **8-frame consecutive dynamic series** (mixed b-values/gradient directions, i.e.
  different image contrast frame-to-frame): variance reduction 8.2% (batched) vs 4.2%
  (ants), but FD agreement r=0.49 -- LOW, meaning the two methods are estimating
  different, disagreeing motion. Per the established methodology this comparison is NOT
  trustworthy and this "win" is NOT claimed; the mixed-contrast case likely violates both
  methods' same-contrast assumptions differently, in ways that don't validate either as
  more accurate.

**Bottom line on the "exceed ants, both simulated and real" goal**: real data, YES (for
the trustworthy same-contrast case -- this is also the intended use case for
`backend='pytorch_batched'`, a single-subject same-modality time series). Simulated
ground truth, translation YES (parity, ~4% gap, within measurement noise), rotation NO
(a genuine, well-characterized, extensively-tested floor that needs a structurally
different metric implementation to close, not further tuning of this one).

## 12. Breaking the translation/rotation tradeoff: alternating optimization + selective tricubic

Continued past Sec 10's conclusion by testing an aggregate comparison across ALL 8
available ground-truth caches (190 frames total, not just the 2 used so far) with the
then-current production defaults: **rotation was worse than ants on every single cache**
(0.023-0.037deg for ants vs 0.043-0.085deg for batched, no exceptions) while translation
was mixed (batched actually beat ants on 2/8 caches). This much stronger, reproducible
signal ruled out per-cache noise as an explanation and motivated two more targeted
interventions:

1. **Alternating (coordinate-descent) translation-only / rotation-only LBFGS**, replacing
   the single joint final LBFGS stage. Several earlier interventions (tricubic
   interpolation, radius-biased sampling) had shown a suspicious pattern: improving one of
   translation/rotation measurably hurt the other, as if the joint gradient step was
   trading them off against each other rather than both converging independently. Freezing
   one parameter group while optimizing the other, alternating for a few rounds, tests this
   directly -- and it worked: translation improved (0.075->0.072mm on cache
   `78f2c2ce4910`) with rotation essentially unchanged (0.0565->0.0560deg), i.e. no
   tradeoff. 3 rounds vs 2 gave no further translation gain and a marginal rotation gain
   (0.0555 vs 0.0560deg) -- 2 rounds was kept as the better cost/benefit point.

2. **Tricubic (Catmull-Rom) interpolation on the translation-only stage specifically**.
   `torch.nn.functional.grid_sample` has no 3D cubic mode (only trilinear/nearest), unlike
   ants' own smoother interpolant; implemented a standalone separable tricubic sampler
   (`_tricubic_sample`, 64-tap per point) and validated it numerically against
   `grid_sample` at exact grid points (agree to ~1e-6) before using it. Applied only to the
   alternating scheme's translation-only stage (confirmed applying it to the rotation-only
   stage instead made rotation WORSE, not better, while costing much more compute -- tested
   directly, not assumed) gave a large, validated, generalizing translation improvement:

   | cache | before (dense joint) | alternating+tricubic(trans-only) | ants |
   |---|---|---|---|
   | `78f2c2ce4910` (30 frames) | 0.0752mm / 0.0565deg | 0.0705mm / 0.0498deg | 0.0645mm / 0.0332deg |
   | `9def7593499b` (40 frames) | 0.0737mm / 0.0534deg | 0.0529mm / 0.0557deg | 0.0716mm / 0.0374deg |

   On the second cache, translation now **exceeds ants by 26%** (0.053mm vs 0.072mm).
   Rotation improved somewhat on the first cache but not the second -- it remains a real,
   consistent gap (~50% relative) that this intervention does not close, matching Sec 10's
   conclusion that rotation precision needs a more literal replication of ants' own metric
   gradient computation, not another schedule/interpolation variant. Giving the
   rotation-only stage more dedicated iterations (15->30) was tested and made no
   difference (fully converged already, confirming this isn't an iteration-budget gap
   either).

   Cost: tricubic's 64-tap-per-point Python loop is markedly slower than `grid_sample`
   (confirmed to hang/take a very long time when also applied with high iteration counts
   on the rotation-only stage at 40-frame scale -- that combination was abandoned, not
   shipped). Used sparingly (2 short translation-only stages) the cost is acceptable; this
   is why it is NOT used more broadly in the schedule.

**Shipped**: `batched_rigid_register_pass`'s final schedule now alternates translation-only
(tricubic) / rotation-only (trilinear) LBFGS for 2 rounds instead of one joint dense LBFGS
stage. Validated against the production module directly (not just the prototype) on both
ground-truth caches, matching the prototype's numbers. 48/48 relevant tests still pass.

**Updated bottom line**: translation now reaches genuine ants-exceeding accuracy on at
least one independent ground-truth cache (and parity-or-better generally); rotation
remains a real, thoroughly-characterized gap after 15+ distinct structurally different
attempts across this and the prior section -- the most defensible remaining path to close
it is a more literal reimplementation of ants' own Mattes MI gradient/pyramid machinery,
not further hyperparameter search.

## 13. Closing the rotation gap: analytic-gradient backward, not another schedule/interpolation tweak

Sec 10's conclusion was that the rotation gap was an information-theoretic property of
the similarity-metric stack, not a fixable optimization/hyperparameter issue, and that
closing it would need something closer to ants' own metric-gradient machinery. That
conclusion was right about *where to look* (the gradient computation) but wrong about the
scale of effort needed -- the fix turned out to be small and already existed elsewhere in
syntx, just not wired into this module.

**Root cause**: `torch.nn.functional.grid_sample`'s default autograd backward
differentiates *through the discrete bilinear sampling op itself* (interpolation weights
as a function of the grid coordinates). This is a valid gradient, but numerically it is a
coarser, less precise estimate of d(warped image)/d(grid) than computing it analytically:
sampling the moving image's own spatial gradient (dI/dx, dI/dy, dI/dz, via
`_image_spatial_gradient`) at the same grid coordinates and taking the inner product with
the incoming loss gradient. `src/syntx/core/grid.py` already implements exactly this
(`AnalyticalGridSample`, exposed via `grid_sample_nd(..., use_analytical_gradients=True)`),
built earlier for `syn.py`/`tvf.py`, but `motion_batched.py` had its own bare
`F.grid_sample` calls in `compute_loss` (both the per-round-stage version and
`fit_pair`'s coarse eval) and in `batched_group_bias_register_pass`'s `compute_loss`, and
never used it.

Why this hits rotation specifically, not translation: a rotation parameter's effect on
the warped image is a small, spatially-varying displacement that scales with distance
from the rotation center (large at the periphery, ~zero near the center), so its true
gradient signal is a small perturbation riding on top of a much larger per-voxel
intensity-gradient magnitude -- exactly the regime where the discrete bilinear backward's
extra numerical coarseness matters most. Translation's gradient is a uniform shift
independent of position, so the discrete backward was already precise enough for it; this
matches why every earlier intervention in Sec 10/12 could move translation without
touching rotation, but nothing closed the rotation gap until the gradient computation
itself changed.

**Fix applied**: switched all three `compute_loss`/coarse-eval `grid_sample` call sites in
`batched_rigid_register_pass` and `batched_group_bias_register_pass` to
`grid_sample_nd(..., use_analytical_gradients=True)`. No change to the forward pass
(still trilinear) or to the optimization schedule structure -- purely a backward-pass
substitution.

**Caching extension to `core/grid.py`**: `AnalyticalGridSample`'s backward recomputes
`_image_spatial_gradient(input)` on every call by default. In this module's usage the
moving image is fixed across an entire LBFGS/Adam inner loop (only the sampling grid
changes iteration to iteration), so recomputing its spatial gradient every backward call
is pure waste -- confirmed to be the dominant added per-iteration cost when this was first
wired in naively. Added an optional `precomputed_grad_I` parameter to
`AnalyticalGridSample.forward`/`backward` and `grid_sample_nd` (default `None`, fully
backward-compatible with every other existing caller in `syn.py`/`tvf.py`, which pass
nothing and get the old recompute-every-time behavior unchanged). `motion_batched.py`
adds a `get_grad_I` helper that caches `_image_spatial_gradient` by moving-tensor identity
(`_grad_I_cache`, keyed on `id(tensor)`+shape) and passes the cached value into every
`grid_sample_nd` call in the coarse/fine stages.

**Schedule simplification confirmed alongside this**: re-validated the alternating
translation-only/rotation-only LBFGS final-polish stage (Sec 12) now needs only 1 round
(2 stages: translation-only then rotation-only) rather than the 2 rounds (4 stages)
carried over from Sec 12's writeup -- a 2nd round gives numerically identical accuracy
(already fully converged after round 1) for ~40% more wall-clock, since each LBFGS stage's
`strong_wolfe` line search evaluates the loss closure (a full analytic-gradient
forward+backward) many times per outer iteration. Schedule shipped as 1 round / 2 stages.

**Validated result -- aggregate across all 8 ground-truth simulation caches** (190 total
frames, `/tmp/syntx_motion_bench_cache/`, per-frame recovery error vs known injected rigid
transforms, `batched_rigid_register_pass` vs `ants.registration(type_of_transform='Rigid')`):

| | translation (mm) | rotation (deg) |
|---|---|---|
| batched (this fix) | 0.0650 | 0.0191 |
| ants | 0.0719 | 0.0317 |

Batched now exceeds ants on **both** translation (10% better) and rotation (40% better)
in aggregate, and -- unlike every prior round in Sec 10/12, where rotation lost on every
single cache with no exceptions -- on **every one of the 8 individual caches**, not just
in aggregate. This is the first intervention in the session to close the rotation gap
rather than trade it off against translation.

**Real clinical data**: re-ran the same trustworthy same-contrast comparison from Sec 11
(BIDS DWI, `sub-Blast-01`, 7-frame pure-b0 series, FD-agreement r=0.94) with this fix:
variance reduction 59.4% (batched) vs 53.7% (ants) -- consistent with Sec 11's number
(the real-data win was already present before this fix; this fix's contribution is
closing the simulated-ground-truth rotation gap, not the real-data result, which depends
on different aspects of the pipeline).

**Honest caveat -- speed regression, not yet fixed**: the batched solver is currently
SLOWER per-frame than ants at these frame counts (roughly 2-2.2s/frame here vs ants'
~0.7-0.8s/frame), driven by the alternating final-polish stage's `strong_wolfe` line
search evaluating the analytic-gradient loss closure many times per outer LBFGS
iteration. This is a real, known regression from this module's original "batching
amortizes per-call overhead, so it should be faster overall" design goal (see the module
docstring's amortized-overhead framing) -- it is not fixed by anything in this section.
A separate, parallel effort is testing whether replacing the final-polish LBFGS with
plain Adam recovers speed while keeping this section's accuracy win; that work is not
part of what is documented or shipped here.

**Status**: all 26 tests in `tests/test_motion_batched.py`,
`tests/test_motion_batched_group_bias.py`, `tests/test_core_grid.py`,
`tests/test_phase_correlation_candidates.py`, and `tests/test_implementation_standards.py`
pass. `src/syntx/core/grid.py`'s extension is additive and backward-compatible (verified
by `test_core_grid.py` passing unchanged for every other caller).

## 14. `backend='auto'` default, and a documented negative result on `robust_affine.py`

**`motion_correction`'s `backend` default changed from `"pytorch"` to `"auto"`.** Given
Sec 13's result -- the batched solver now provably exceeds `ants.registration`'s own
accuracy on both translation and rotation, in aggregate and on every one of the 8
ground-truth caches individually -- the question was whether/how to make that the
default path for eligible callers, rather than leaving it purely opt-in behind an
explicit `backend='pytorch_batched'`. This was put to the user as an explicit
multiple-choice decision: (a) auto-select the batched backend when eligible, (b) leave it
opt-in only, or (c) make it the unconditional default regardless of eligibility. The user
chose (a), and that is exactly what is implemented: `backend="auto"` (now the default)
resolves at call time to:

- `"ants"` if a `mask` is given -- it is currently the only backend with a masked-MI mode;
- `"pytorch_batched"` if the input is 3D+t, `type_of_transform='Rigid'`, and no mask -- the
  one configuration Sec 13 actually validated as exceeding ants;
- `"pytorch"` otherwise (2D+t, or any non-Rigid transform request) -- the previous default
  behavior, unchanged.

Any explicit `backend=` value ('pytorch', 'pytorch_batched', 'ants') bypasses this
resolution entirely and behaves exactly as before; only callers who omit `backend` (or
pass `backend='auto'` explicitly) get the new resolution logic. Five new tests
(`TestMotionCorrectionAutoBackend` in `tests/test_motion_batched.py`) cover: default
succeeds on the eligible case with correct accuracy, auto falls back to ants with a mask,
auto falls back to pytorch for 2D input, auto falls back to pytorch for an Affine
transform request, and an invalid backend string still raises `ValueError` with the
updated message.

**Negative result: analytic-gradient backward does NOT help `robust_affine.py`'s
single-frame solver.** Given how decisively Sec 13's `grid_sample_nd(...,
use_analytical_gradients=True)` swap closed the rotation gap in `motion_batched.py`, the
natural follow-up question was whether the same substitution would help
`syntx.robust_affine`'s single-frame solver (the `backend='pytorch'` path used elsewhere
in the codebase, e.g. via `motion_correction`'s non-batched fallback). This was tried: in
`_run_pytorch_affine_solver`'s `objective()` function, the two main MI-loss
`F.grid_sample` calls were swapped for `grid_sample_nd(...,
use_analytical_gradients=True)`, with the moving image's spatial gradient cached per
pyramid level (the same caching pattern from Sec 13).

It was then A/B tested cleanly on real Mindboggle brain data
(`tests/test_affine_reproducibility.py`'s test pair), with timing redone back-to-back
after an unrelated CPU-heavy process on the machine had confounded the first measurement
(that first, contended run had suggested ~80% overhead, which was misleading):

- **Overhead**: ~8% (not ~80%) once measured cleanly without contention.
- **Accuracy**: no benefit on the one real-data Dice-score check available -- 0.3267
  without the change vs 0.3218 with it. Essentially flat, marginally worse, not the clear
  win seen in `motion_batched.py`.

Unlike the batched rotation case, this single-frame solver's registration problem does
not appear to sit in the small-perturbation-on-large-background-gradient regime that made
the analytic backward worth its cost in Sec 13 -- or if it does, the effect is too small
to show up against this benchmark's noise floor. Given a real (if modest, ~8%) cost to a
heavily-tested, validated production path and no proven accuracy benefit, the change was
reverted: `git checkout -- src/syntx/robust_affine.py`. Confirmed clean -- `git diff
src/syntx/robust_affine.py` shows no output as of this commit. This is a documented
negative result, not a bug fix, and is not re-attempted this session.

**Side finding: `test_affine_reproducibility.py`'s runtime is not "near instantaneous."**
While investigating the above, direct timing of the fully-reverted, unmodified test file
showed it inherently takes ~154 seconds -- it runs 6 real full-resolution affine
registrations on real Mindboggle brain data at roughly 20-40s each. This is pre-existing
test behavior, independent of anything changed this session, and is not a regression;
it's noted here only because the prior assumption (that the test was fast) was wrong and
worth correcting for future sessions budgeting test-suite time.

**Status**: `tests/test_motion.py`, `tests/test_motion_batched.py`,
`tests/test_motion_batched_group_bias.py`, `tests/test_implementation_standards.py`, and
`tests/test_core_grid.py` -- 41 tests total -- pass with the `backend='auto'` change in
place. `src/syntx/robust_affine.py` carries no diff from HEAD. Version bumped to 5.4.28.

## 15. `robust_affine` unified as the single initial-alignment mechanism across `syn`/`syngs`/`tvf`/`scattered`; three bespoke inline affine optimizers removed

This was an explicit user directive, quoted directly: "the point of robust_affine was to
provide a single choice that would work everywhere -- get rid of other choices. just use
robust_affine." Three different bespoke inline affine optimizers, in three different
files, each redundant with `robust_affine.py` and each with its own separate
maintenance/bug surface, were deleted and replaced with the same mechanism everywhere.

**`src/syntx/syn.py` (`registration()`/`syn()`, plus `auto_reg()`).**
`registration()`'s own inline per-level Adam affine optimizer (previously driven by
`affine_iterations`/`aff_metric`/`aff_sampling`) is removed. When `initial_transform is
None`, `registration()` now always calls `robust_affine(fixed, moving, dof=affine_dof,
mode=affine_mode, seed=affine_seed, verbose=verbose)` internally, feeding the result
through the same `parse_ants_affine`-based absorption code path that already handled a
caller-supplied `initial_transform`. Three new named passthrough params replace the
removed ones: `affine_dof=None`, `affine_mode='pytorch'`, `affine_seed=None`. Passing any
of the three removed params now raises an explicit `TypeError` (a `_removed_affine_params
= {'affine_iterations', 'aff_metric', 'aff_sampling'} & set(kwargs)` guard) rather than
being silently swallowed by `**kwargs` as before.

`auto_reg()` was simplified to match: since `registration()`, `syngs_registration()`, and
`tvf_registration()` all now self-resolve their own alignment via `robust_affine`
internally, `auto_reg()`'s own precomputation step is only still needed for its
`is_affine_only` dispatch branch (which calls `robust_affine` directly, not through one of
the three wrappers). The guiding boolean was renamed from `should_run_affine`'s old
implicit meaning to `is_self_resolving_backend` with a comment explaining why. The
now-stale `'affine_iterations': [...]` entries in `auto_reg()`'s internal
`syngs_params`/`tvf_params` dicts were removed -- these were caught as real `TypeError`s
once the target functions' own rejection guards were added, not merely assumed stale, and
were re-verified clean afterward. Separately, a second, smaller stray reference was found
during final pre-commit review of this same function: inside the `should_run_affine`
block (reachable only via the `is_affine_only` path), two dead lines --
`if aff_mode in ('translation_only', 'com_only') or 'affine_iterations' not in kwargs:` /
`kwargs['affine_iterations'] = [0] * num_levels` -- were setting a now-meaningless kwarg
that flowed only into a direct `run_robust_affine(**aff_params)` call. Because
`robust_affine()` has its own trailing `**kwargs`, this was silently absorbed and inert
(not an active bug, since it never reached one of the three new `TypeError` guards), but
it was vestigial from the old inline-optimizer era and was removed as part of this same
cleanup rather than left as known dead code in a function already being edited.

**`src/syntx/syngs.py` (`syngs_registration()`).** Same pattern:
`GeodesicShootingModel.fit()`'s inline `HierarchicalAffine` Adam optimizer loop is
removed; the same `affine_dof`/`affine_mode`/`affine_seed` params and rejection guard are
added; the `initial_transform`/`T_init`-absorption code path is unified to run identically
whether the transform came from the caller or from `robust_affine`.

**`src/syntx/tvf.py` (`tvf_registration()`).** Same pattern, with one difference: this
file already called `robust_affine` correctly for initialization, but it also ran a
redundant inline `TVFModel.fit()` Adam affine loop on top of the already-absorbed
`T_init`. That redundant second loop is now removed, plus the same
`affine_dof`/`affine_mode`/`affine_seed`/rejection-guard treatment as the other two files.

**`src/syntx/scattered/solver.py` (`ScatteredDiffeomorphicRegistration`) -- behavior
change: affine pre-alignment is now on by default.** This solver had no `robust_affine`
usage at all -- instead a from-scratch, buggy inline LNCC-based pre-alignment
(`_optimize_affine_prealignment`) with a confirmed latent bug (an undefined `spatial`
variable in its 3D branch, never previously exercised because the branch was never hit in
existing tests). That method is deleted and replaced with a new `initial_transform`-driven
mechanism (`None`/`False`/`'identity'`/an explicit transform, matching `greedy.py`'s
established 3-way convention). Because this solver has no physical-space concept, the new
path builds synthetic identity-space ANTs images to call `robust_affine`, then converts
the result into the same dense-displacement-field convention the class already used, via
`greedy.py`'s existing `_build_torch_affine_matrix` helper (reused, not reimplemented).
**This is a real behavior change, not just a refactor**: affine pre-alignment is now ON by
default for this solver (previously off by default, `affine_epochs=0`) -- documented in
the new `initial_transform` config field's docstring.

**Test suite migration (~30 files, mechanical but not trivial).** Removing
`affine_iterations`/`aff_metric`/`aff_sampling` from `syn.py`/`syngs.py`/`tvf.py` broke
every existing test that passed those kwargs to `syn()`/`registration()`/`syngs()`/
`tvf()` -- previously silently absorbed by `**kwargs`, now raising the new explicit
`TypeError` (confirming the guards work as intended, not a regression). Roughly 30 test
files were fixed: the stale kwarg was removed outright where it was just an
iteration-count tuning knob; replaced with `initial_transform='identity'` where the test's
actual intent was "skip affine entirely" (confirmed `syn.py` supports that string
sentinel; `syngs.py`/`tvf.py` do NOT, so those instead got a fixed `affine_seed` pinned
across compared calls, or had the kwarg simply removed). Two dead tests, which tested
internals of the now-deleted optimizer loops directly, were deleted rather than patched.
Two `examples/benchmarks/*.py` scripts that are imported/executed by the test suite
(`benchmark_gpu_performance.py`, `evaluate_all_metrics.py`) were also fixed for the same
reason -- `benchmark_gpu_performance.py` required actually rewriting its "time the affine
stage" benchmark logic to call `robust_affine` directly, since the code path it used to
time no longer exists.

**Explicitly out-of-scope, deferred follow-up: ~20 more files.** A grep across
`scripts/*.py` and `examples/**/*.py` (deliberately excluding `tests/`, which was fully
migrated) for `affine_iterations`/`aff_metric`/`aff_sampling` found 22 more files, none of
which are executed by the test suite (standalone research/demo scripts, not gated by CI):
`examples/benchmarks/benchmark_suite.py`, `examples/benchmarks/compare_metrics.py`,
`examples/benchmarks/generate_ants_2d_comparison_report.py`,
`examples/benchmarks/generate_ants_3d_comparison_report.py`,
`examples/benchmarks/generate_vgg_deep_dive_report.py`,
`examples/benchmarks/run_benchmarks.py`,
`examples/benchmarks/run_comprehensive_benchmarks.py`,
`examples/benchmarks/run_optimizer_sweeps.py`, `examples/evaluate_feature_metrics.py`,
`examples/run_m1_pytorch_tuning.py`, `scripts/affine_repro_harness.py`,
`scripts/benchmark_decathlon.py`, `scripts/benchmark_mbhard_landmark_guidance.py`,
`scripts/benchmark_sulcal_guided_cohort.py`, `scripts/characterize_combinations_2d.py`,
`scripts/characterize_combinations_3d.py`, `scripts/compare_fast_smooth_peak.py`,
`scripts/eval_peak_syn_params.py`, `scripts/master_benchmark_orchestrator.py`,
`scripts/test_6way_fresh.py`, `scripts/test_s3_fresh.py`,
`scripts/tvf_syn_gap_sweep_2d.py`. These were deliberately NOT fixed this session -- real
but low-priority follow-up work that would need someone to actually try running each one
to see what it needs. Listed here by name so this is a documented gap, not a silent one.

**Status**: two overlapping full-suite sweeps this session reached 403 passed + 186
passed in aggregate as fixes landed, with one failure found (a stale `auto_reg()` dict
entry) and fixed, then re-verified clean (7/7 on the specific failing file, then 186
passed on a broader re-run including it). A final pre-commit verification sweep across
every touched-file's test module, run fresh rather than trusted from memory, is the basis
for this commit; see the commit message for its exact pass count. Version bumped to
5.4.30 (5.4.29 was consumed by an unrelated, parallel `robust_affine` phase-correlation
fix that landed in between).

## 16. Registration defaults aligned to this repo's own benchmark data (`sobolev` standardized); pytest ~5x faster

Two unrelated pieces of work landed together in the same commit (v5.4.31) for
expediency. They are documented separately here because they are separately motivated
and separately verified.

### 16.1 `syn`/`syngs`/`tvf`/`greedy` defaults vs. `docs/provenance/best_parameters.json`

A subagent compared each of `syntx.syn`/`syntx.syngs`/`syntx.tvf`/`syntx.greedy`'s
default parameter values against `docs/provenance/best_parameters.json` -- a real
90-pair Mindboggle/mbhard brain-registration benchmark artifact already checked into
this repo -- and updated defaults to match the best-found configuration there, per
explicit user direction to standardize on `regularizer='sobolev'` (chosen for its much
lower folding/topology-violation rate vs. `gaussian`/`dsti1`, despite either winning on
raw dice in some benchmark arms) and to give `greedy` (which has no benchmarked data of
its own) syn's defaults as an interim stand-in where the parameter's physical meaning
actually transfers.

**`src/syntx/syn.py` (`registration()`).**
- `grad_step`: `0.50` -> `0.25`.
- 3D `reg_iterations` default (when `None`): last stage `50` -> `20`, i.e.
  `[100, 100, 50]` -> `[100, 100, 20]`, matching the winning
  `90pair_population_benchmark_sobolev_mps` config. 2D default (`[100, 100, 100, 50]`)
  is unchanged -- no benchmark data exists for 2D.
- `regularizer` was already `'sobolev'` for the pytorch backend -- confirmed, unchanged.

**`src/syntx/syngs.py` (`syngs_registration()` and `integrate_momentum()`).**
- `n_steps`: `6` -> `8`, matching the winning
  `strict_diffeomorphic_zero_folding_mps` config.
- `bootstrap_mode`: `'none'` -> `'antithetic'`.
- Consistency bug found and fixed as a direct consequence: `integrate_momentum()` (a
  standalone reconstruction utility used to rebuild a deformation from a saved momentum
  field) had its own independent `n_steps=6` default, now out of sync with
  `syngs_registration()`'s new `n_steps=8` default -- a caller reconstructing a
  deformation produced under the new default, without passing `n_steps` explicitly,
  would silently get a different ODE discretization than the one that produced it. A
  test caught this mismatch; `integrate_momentum()`'s default was updated to `8` to
  match.

**`src/syntx/tvf.py` (`tvf_registration()`) -- the most involved change, including a
genuine root-cause bug fix.**
- **Root-cause bug**: `tvf_registration()` used to resolve its `regularizer` default
  inconsistently across internal code paths -- the function's own top-level resolution
  defaulted to `'gaussian'` (`kwargs.get('regularizer', 'gaussian')`), while
  `TVFModel.fit()`'s internal `RegAdam` optimizer setup independently defaulted to
  `'sobolev'` -- and the function-level resolved value was never written back into
  `kwargs`, so which default actually took effect depended on which internal path read
  `regularizer` first. Fixed by resolving once
  (`reg_mode = kwargs.get('regularizer', 'sobolev'); kwargs['regularizer'] = reg_mode`)
  so every downstream `model.fit(**kwargs)` call sees the same explicit value. The new
  single default is `'sobolev'`.
- `optimizer`: `'cfl'` -> `'reg_adam'`.
- `optimizer_lr`: `None` -> `1.2`.
- `total_sigma`: `0.02` -> `0.035`.
- `fast_smooth`: `True` -> `False`.
- `cfl_momentum`: `0.95` -> `0.9` (only active when `optimizer='cfl'`; inert but kept
  consistent under the new default `optimizer='reg_adam'`).
- `flow_sigma` deliberately left at `3.0` -- verified via the spectral-regularizer
  on/off-gate mechanism that its exact numeric value is inert for `sobolev`/`dsti`/
  `dsti1`: both `3.0` (this default) and the benchmark file's `1.0` are functionally
  equivalent, since for those regularizers `flow_sigma` only gates smoothing on/off
  (any positive value) rather than setting kernel shape (that's `sobolev_alpha`/
  `dsti_alpha`'s job).
- Several docstrings that were already stale even before this pass were corrected too
  (e.g. one said "Default 0.20" for `grad_step` when the actual default was `0.50`).

**`src/syntx/greedy.py` (`greedy_registration()`).**
- `learning_rate`: `0.50` -> `0.25` (interim stand-in match to syn's new `grad_step`;
  greedy has no benchmarked data of its own).
- 3D `reg_iterations` default: last stage `80` -> `20`, i.e.
  `[100, 100, 80]` -> `[100, 100, 20]`, matching syn's new schedule shape.
- **Architectural finding, and why `regularizer='sobolev'` was correctly NOT applied
  here**: greedy has no spectral `regularizer` selection mechanism at all -- it always
  performs real spatial Gaussian convolution (`gaussian_1d_compact`), ITK-greedy style.
  Applying `regularizer='sobolev'` would be meaningless (no such code path exists), so
  it was deliberately left out rather than added as a no-op or, worse, silently
  misread by a caller as "this now behaves like syn's sobolev path."
- `flow_sigma` (`1.8`) was deliberately left unchanged for the same reason in reverse:
  it is a literal voxel-space sigma for real spatial convolution here, not an
  ITK-variance gate value like syn/tvf's `flow_sigma`. Copying syn's numeric `3.0`
  would be a real, much stronger physical smoothing operation, not a like-for-like
  default alignment -- this reasoning is now documented directly in the docstring.

**`docs/PROJECT_FINDINGS_DETAILED.md` correction.** A stale blanket claim -- "TVF MUST
use `flow_sigma = 0.0`" -- was narrowed. It was written for and remains true of the
legacy `regularizer='gaussian'` path (real per-epoch spatial convolution, genuinely
expensive and dice-degrading at `flow_sigma > 0`), but does not apply to the spectral
paths (`sobolev`/`dsti`/`dsti1`), which use a cheap FFT-based Green's-operator gate
where `flow_sigma=3.0` (the new default) is correct and expected, not a violation of
the original rule. The doc now states both cases explicitly rather than one blanket
claim that was only ever true for one of the two mechanisms.

**`tests/test_reproducibility_fast.py`.** `test_syngs_reproducibility`'s section 1
(testing that `bootstrap_mode='none'` gives seed-independent bit-for-bit determinism)
now passes `bootstrap_mode='none'` explicitly rather than relying on the function
default, since the default changed to `'antithetic'` (intentionally stochastic) as
part of this same pass -- this keeps the actual invariant under test (determinism
under `'none'`) intact rather than having it silently broken by an unrelated default
change.

**Verification.** A 37-file sweep covering every touched function's test dependents
(`test_syn.py`, `test_syngs_*.py`, `test_reproducibility_fast.py`,
`test_bspline_regularizer.py`, `test_auto_reg.py`, `test_coverage_boost_80.py`,
`test_syn_jax.py`, `test_coverage_helpers.py`, `test_e2e_metrics.py`,
`test_challenger_*.py`, `test_registration_bugs.py`, `test_audit_*.py`,
`test_adversarial_*.py`, `test_surface_guided_syn.py`, `test_optimizers.py`,
`test_parameter_sensitivity.py`, `test_bspline_interpolator.py`,
`test_e2e_sobolev_benchmark.py`, `test_gpu_benchmark.py`,
`test_syn_tvf_extended_coverage.py`, `test_tvf_*.py`, `test_scattered_*.py`,
`test_greedy.py`), run with `pytest ... -q -n auto`: **413 passed, 16 skipped, 0
failed** in 161.82s. A previous run of the same sweep (during earlier iteration on
this work) had shown one flaky failure --
`tests/test_syngs_m3_challenger2.py::test_adversarial_performance_benchmark`, a
wall-clock timing-comparison test ("streamlined implementation should be faster than
legacy") on a tiny workload (0.150s vs 0.139s margin) -- confirmed at the time, by
re-running it standalone without `-n auto`, to be CPU-contention flakiness under
parallel execution rather than a real regression; on this final pre-commit run it
passed cleanly even under `-n auto`, consistent with that diagnosis.

### 16.2 pytest performance fix (separate, unrelated to 16.1)

`pyproject.toml`'s `[tool.pytest.ini_options]` used to bake
`--cov=syntx --cov-report=term-missing` into `addopts`, meaning every test invocation
ran full coverage instrumentation by default, measured at roughly 2.5x overhead on a
sample run. Removed from `addopts` (now an empty string) so coverage only runs when
explicitly requested: `pytest --cov=syntx --cov-report=term-missing`.

Also added `pytest-xdist` to the `test` optional-dependency group (it was already
`pip install`ed in this environment), enabling `pytest -n auto` for parallel test
execution across files. This was deliberately NOT added to default `addopts` -- left
as an opt-in flag -- since some tests share the same MPS/GPU device and at least one
timing-comparison test (`test_adversarial_performance_benchmark`, see 16.1) was
observed to flake under parallel contention; forcing `-n auto` on by default for every
invocation was judged not safe without further per-test isolation review.

Combined effect measured on a representative 37-file, ~600-test sweep: roughly 14
minutes (with coverage, sequential) down to about 2 minutes 45 seconds (without
coverage, `-n auto`) -- approximately 5x.

## 17. `'SyNTo'` eliminated as the default `type_of_transform`; `'SyN'` is now the default

The user noticed `registration()`/`syntx.syn()` defaulted to `type_of_transform='SyNTo'`
and asked what it was, correctly suspecting it was redundant. Confirmed: `SyNTo` is the
PyTorch model class name (`class SyNTo(nn.Module)`, likely "SyN Torch," distinguishing it
from the JAX backend's own class), and the string `'SyNTo'` was ALSO accepted as a
`type_of_transform` value -- but `registration()`'s own parsing (`elif tot_lower in ['syn',
'synto']:`) handles `'SyN'` and `'SyNTo'` identically, with zero behavioral difference.
Having the *less* recognizable, ants-inconsistent name be the default (rather than the
standard `'SyN'`, which ants.registration users already expect) was pure redundant naming
with no functional benefit -- exactly the confusion the user flagged.

Per the user's explicit choice (backward-compatible option over a breaking change):
changed the default to `'SyN'` in `registration()`'s signature, docstring, and
`auto_reg()`'s fallback default; `'SyNTo'` remains a fully accepted alias for any existing
caller (internal or external) that already passes it explicitly -- the parsing branch
itself was NOT touched. Verified via the full `test_syn.py`/`test_auto_reg.py`/
`test_reproducibility_fast.py`/`test_audit_2_regression.py`/`test_bulletproof_ants_parity.py`/
`test_challenger_verification.py`/`test_coverage_boost_80.py`/`test_syn_jax.py`/
`test_coverage_helpers.py` sweep that this is a true no-op behaviorally (85/86 passed,
1 skipped-unrelated; the one failure, `test_bulletproof_ants_parity.py::
test_3d_anisotropic_bulletproof_ants_parity`, was confirmed via `git stash` to be a
PRE-EXISTING failure at HEAD (v5.4.31) unrelated to this change -- see Sec 18.

## 18. `test_bulletproof_ants_parity.py::test_3d_anisotropic_bulletproof_ants_parity` -- investigated, confirmed pre-existing, tolerance relaxed 0.960 -> 0.95

Follow-up to Sec 17's discovery. Initial hypothesis (a regression from one of Sec 16's
default changes) was WRONG -- investigated properly rather than guessed:

- The test passes `reg_iterations=[10,10,5]` and `type_of_transform='SyNTo'` explicitly,
  so the only *default*-dependent parameter actually in play here is `grad_step`
  (0.50 -> 0.25 per Sec 16). Testing `grad_step=0.50` (the OLD default) directly on this
  exact test scenario gives `corr_jac = nan` (a `RuntimeWarning: invalid value encountered
  in divide` -- the syntx-computed Jacobian field has ZERO variance, i.e. this few-iteration
  fast config produces an essentially degenerate, near-identity warp under the old
  default). `nan > 0.960` is `False` in Python, so the OLD default would ALSO have failed
  this assertion, just via `nan` instead of a numeric near-miss.
- Confirmed directly via `git worktree add /tmp/syntx_old_check c753f26` (v5.4.28, well
  before ANY of this session's syn/syngs/tvf/scattered/defaults work) and re-running the
  test there: **it already failed at that commit**, with the same symptom shape. This is a
  long-standing, pre-existing borderline test, not a regression from anything this session
  changed.
- Given the tiny gap (0.9598 vs 0.960), the fact this test's OTHER assertions in the same
  function (`corr_disp > 0.999`, `corr_roundtrip > 0.9999`) pass comfortably and are
  unchanged, and that this is a numerically sensitive derived quantity (Jacobian
  determinant, i.e. spatial derivatives of a warp field) on a deliberately fast/coarse
  ([10,10,5] iterations) synthetic 3D anisotropic case -- relaxed the threshold to `0.95`
  per the user's explicit "if it's just a small tolerance issue, adjust safely down to
  0.95" direction. Verified stable across 3 repeated runs (no flakiness at the new
  threshold). This does NOT weaken the test's actual protective value: it still requires
  95%+ correlation with the ants C++ ITK reference Jacobian, and the OTHER 3 assertions in
  the function remain fully intact.

## 19. ITK half-voxel boundary-convention gap in `grid_sample_nd` -- investigated, quantified, ~~WONTFIX~~ **REVERSED, fixed in Sec 20**

> **Update:** the WONTFIX below was reversed. Once measured end-to-end, this gap (plus
> three related defects) broke the syntx <-> ants interoperability contract -- register
> with syntx, apply with ants tools, or vice versa, with no accuracy penalty. See Sec 20.

Prompted by a user-shared comment about a sibling project (ANTsTorch) noting that
PyTorch's MPS backend now has a native `grid_sampler_3d` kernel, which raised the
question of whether syntx has any awareness of related PyTorch/ITK interpolation-
compatibility issues. Investigation, in two parts:

**Native MPS kernel**: confirmed empirically (this environment, torch 2.13.0) that 3D
`grid_sample` on MPS already runs on a genuine native GPU kernel -- forward, backward, and
a 140^3-volume timing test showing 36.7x speedup over CPU (ruling out a silent CPU
fallback). syntx has never needed antstorch's custom Metal-shader workaround or its
`_torch_compat.py` capability-probing layer; it just calls `F.grid_sample` directly via
`core/grid.py`'s `grid_sample_nd`, and that has been adequate. Confirmed via grep: zero
uses of `create_graph=True`/`torch.autograd.grad` anywhere in syntx, so the separate
double-backward gap the antstorch comment also mentions (native MPS grid_sample doesn't
support it, even in the newer torch version discussed) does not currently apply to
anything syntx does either.

**ITK half-voxel boundary convention**: a real, separate, and previously undocumented gap.
Verified directly against actual `ants.apply_transforms` output (not just inherited from
the antstorch comment) with a controlled 1D-isolated test (an image varying along a single
axis, a known 4.3-voxel translation): ITK/ants correctly interpolates sample points that
fall within half a voxel beyond the last voxel center (its `OutsideValue`-bounded extended
domain), while neither of PyTorch's `grid_sample` padding modes reproduces this alone --
`'zeros'` cuts off too early (giving 6.65 vs the true 9.50 at the affected voxel in the
test, a ~30% error), `'border'` gets that one voxel right but then wrongly keeps
replicating past the true zero-crossing point. The effect is precisely confined to a
single voxel-shell at whatever boundary a warp pushes sampling toward -- confirmed exact
agreement with ants at every interior voxel and every voxel further outside the extended
margin. Affects every syntx registration method, since all route through the same
`grid_sample_nd`.

**Cost/benefit assessed, decision: WONTFIX for now.**
- *Cost*: high. The fix belongs in `core/grid.py` (`grid_sample_nd` and
  `AnalyticalGridSample`'s backward, which samples the image too), needs correct 2D/3D
  handling of the extended-domain clamp + edge-replicate-pad + explicit outside-zeroing,
  and -- since every accuracy benchmark this session ran (the 8 ground-truth
  motion-correction caches, the real Mindboggle Dice/Jacobian tests, `test_
  bulletproof_ants_parity.py`) was tuned on top of the CURRENT boundary behavior -- would
  need a full re-validation sweep comparable in scope to the `robust_affine` unification
  work (Sec 15), this session's single largest undertaking.
- *Benefit*: assessed as small for the metrics already validated this session. All of
  this session's benchmark content sits well inside the field of view (8-12+ voxels of
  margin in the anisotropic parity test specifically), where the affected single-voxel
  boundary shell falls in background/zero regions -- unlikely to explain the Jacobian
  threshold near-miss addressed in Sec 18. Where it WOULD plausibly matter: tight-FOV
  EPI/DWI acquisitions where real anatomy extends close to the slice edge, and
  large-motion/large-rotation cases (the "participant coughs and shifts" scenario from
  earlier this session, Sec 4-ish) that can genuinely push content toward the boundary.
- *Decision*: not worth implementing speculatively given the cost/benefit skew. Documented
  here as a known, precisely-quantified, understood limitation rather than an unknown one.
  Revisit if a concrete use case (tight-FOV data, or a large-motion scenario) is found to
  actually depend on it -- do not preemptively build the fix before that need is real.

## 20. ants <-> syntx interoperability contract: edge-inverse breakdown, native inverse, out-of-domain ingestion -- fixed

**Goal (user):** register with syntx, then use ants tools to warp images/transfer labels
(or register with ants and consume in syntx) with *no accuracy penalty, in either
direction, including at the field-of-view boundary*. Encoded as an executable contract in
`tests/test_ants_interop_contract.py` (18 tests; an "interior" anatomy case, a "tight
FOV" case where content crosses every face, a (0,0,0)-inside-the-image header, and a
distinct non-cubic fixed/moving header case). Initially 8/14 failed; now 18/18 pass.

### 20.1 Inverse transform broke down at the edges (most important)
- *Symptom:* round trip `fwd -> inv` through `ants.apply_transforms_to_points`: edge-shell
  p95 0.889 voxel (ants' own SyN: 0.006); tight FOV 0.352 (ants 0.034); max ~1.1 voxel.
- *Root cause:* ANTs SyN holds displacement at exactly 0 on every face (ITK
  `EnforceStationaryBoundary`; measured face max 0.0000 for ants, ~1.1 voxel for syntx).
  A syntx face point was pushed outside the domain, where the ITK inverse field is 0 by
  definition, so the round trip cannot close -- no inverse solver can fix that.
- *Fix:* `SyNTo(stationary_boundary=True)` (default; `syntx.syn(..., stationary_boundary=False)`
  restores the old behaviour). Faces of all four half fields + in-loop inverses are
  re-zeroed after every update step (main and retry loops) and on the final composed
  fields. Because the constraint holds from the zero initial field, only one step's
  update is removed at the face -- the "8.0 -> 0.0 in one voxel" discontinuity that got
  the earlier after-the-fact Dirichlet masking removed does not occur (field decays
  smoothly: |u| 0.00 / 0.19 / 0.37 / 0.57 at 0 / 1 / 2 / 4 voxels from the face).
- *Result:* edge p95 0.889 -> 0.004 (ants 0.005); tight FOV 0.352 -> 0.013 (ants 0.034);
  min Jacobian improved (0.545 -> 0.631, 0.320 -> 0.401); tight-FOV registration improved
  (corr(warped, fixed) 0.804 -> 0.863); interior case essentially unchanged (0.99970 -> 0.99952).
- *Residual, understood:* interior round-trip p95 ~0.03 voxel vs ants ~0.01-0.02. The
  fixed-point solver gives an exact *right* inverse (`fwd o inv`, max 1e-4) but
  `inv o fwd` at grid points is limited by trilinear discretisation of syntx's somewhat
  larger fields; the algebraic inverse balances both identities (0.033 / 0.027) better
  than solver-polishing (0.050 / 0.000), so no polish step was added.

### 20.2 Native `SyNTo.forward_inverse` disagreed with the exported inverse
- *Symptom:* up to 5% of intensity range in the image interior; corr with truth 0.99766
  (native) vs 0.99973 (ants applying the exported inverse).
- *Root cause:* it resampled `warp_r2l` (defined on the *fixed* grid, pre-affine frame) onto
  the moving grid and applied the inverse affine afterwards -- wrong domain and wrong order.
- *Fix:* rewritten to evaluate exactly the exported `[affine^-1, warp_inv]` chain:
  `z = A^-1(y)`, sample `warp_r2l` at z on its own grid (zero outside, ITK), sample the
  fixed image at `z + v(z)`. Now matches ants to ~1e-6. With a non-affine
  `initial_transform` (`initial_grid`) it raises instead of silently ignoring it.

### 20.3 `compute_initial_grid` sent out-of-domain points to physical (0,0,0)
- *Root cause:* the transform was "evaluated" by resampling per-axis coordinate images with
  `ants.apply_transforms`, which returns 0 outside the moving FOV -> coordinate (0,0,0).
  Reproduced with a header whose (0,0,0) lies inside the moving image: out-of-FOV points
  sampled tissue (33.4) instead of background.
- *Fix:* antsApplyTransforms now composes the list into one displacement field on the fixed
  grid (`compose=`), i.e. the transform itself is evaluated everywhere; points that leave
  the moving image get normalized coordinates outside [-1, 1] and sample background.

### 20.4 Also found and fixed along the way
- **ITK half-voxel sampling (Sec 19):** `grid_sample_nd(padding_mode='itk')` -- edge-clamped
  interpolation to half a voxel beyond the edge centres, 0 beyond (`itk_inside_mask`).
  Used by `SyNTo.forward`, `forward_inverse` and `SyNToTransform.apply`; native warps now
  match `ants.apply_transforms` to ~1e-6 including the edge shell (was 0.47 of range in
  the tight-FOV edge). *Scope note:* the optimiser's internal sampling is unchanged; with
  stationary boundaries the fields are 0 at the faces, so zeros-vs-ITK for the *fields*
  no longer differs, and the image-sampling difference is confined to one boundary shell.
- **`SyNTo.forward` axis-order bug:** passed the moving shape to
  `grid_to_physical_affine_torch` in xyz while `fit()` uses zyx; invisible on cubic
  images, 50% of range error on non-cubic moving images (the distinct-header test fails
  with the old line, verified).
- **`SyNToTransform.apply`:** ignored `moving_spacing/origin/direction` recorded by
  `to_transform` (assumed moving header == fixed) and used `'border'` padding.
- **`genericLabel` interpolator:** `grid_sample_nd(interpolator='genericLabel')` -- ITK
  `LabelImageGenericInterpolateImageFunction` (per-label linear weights, argmax, ties to
  the lower label). syntx-native label transfer now agrees 100% with ants for both
  `nearestNeighbor` and `genericLabel` (was 96.6-98.5% for genericLabel).
- `test_bulletproof_ants_parity.py` 3D case: its box-vs-shifted-box pair needs no
  deformation (affine alone: Dice 1.0, MSE 3e-5; the old non-zero field made it worse,
  MSE 8e-5), so with the stationary boundary the best-loss field is exactly zero and the
  field-conversion correlation was NaN. The moving image now has a non-affine bump.

### 20.5 Benchmark JSON never recorded inverse consistency (bug)
`evaluate_mindboggle_pair` read `inverse_identity_errors['phi_1']['mean'/'p95']`, but
`registration()` returns `{'max_error', 'mean_error', 'error_map'}` -- so `syntx_inv_mean` /
`syntx_inv_p95` were NaN in every benchmark record. Now summarised from the error map
(`_inverse_error_stats`), with new `syntx_inv_max`, `syntx_inv_interior_mean`,
`syntx_inv_interior_max` (same 5-voxel-eroded mask as the standard report).

On mbhard (pair 44) the report shows interior max inverse error 6.65 mm (mean 0.054 mm);
the same value with `stationary_boundary=False` (6.58 mm) and before this work (5.29 mm), so
it is not caused by these changes. The run used `src/syntx/benchmark/config.py`'s
`syn_config` (grad_step 0.35, flow_sigma 2.5, fast_smooth on), which has drifted from the
canonical `docs/provenance/best_parameters.json` record (0.25 / 3.0 / fast_smooth off,
in_loop_inv_steps 10); reconciling that is the next piece of work.

### 20.6 Not covered
- Other engines (`greedy`, `syngs`, `tvf`) were not audited against this contract.
- `SyNToTransform.apply`'s legacy normalized-grid (non-physical) path still pads with `'border'`.

## 21. Canonical `syntx.syn` parameters -- one source of truth, enforced by test

**Decision (user):** the canonical SyN parameters are `docs/provenance/best_parameters.json`,
`"90pair_population_benchmark_sobolev_mps"`, with `fast_smooth=False`. `syntx.syn()`'s
defaults must reproduce it and every benchmark entry point must use those defaults unless a
caller explicitly overrides them.

### 21.1 Audit: five places, four different parameter sets
| Parameter | Record | `syntx.syn` (before) | `benchmark/config.py` (before) | `run_standard_report_demo` | `high_level_benchmark_run` | `run_config.json` |
|---|---|---|---|---|---|---|
| grad_step | 0.25 | 0.25 | **0.35** | 0.25 | **0.5** | 0.25 |
| flow_sigma | 3.0 | 3.0 | **2.5** | 3.0 | 3.0 | 3.0 |
| fast_smooth | False | None->False | **True** | False | **True** | **true** |
| in_loop_inv_steps | 10 | **6** | not passed (6) | 6 | 6 | absent |
| sobolev_alpha | 1.5 (see 21.2) | **None -> sqrt(flow_sigma)/2 = 0.866** | 1.5 | **0.866** | **0.866** | 1.5 |
| analytic gradients | False | False | False | False | **True** | False |
| reg_iterations (3D) | [100,100,20] | same | same | **[80,80,20]** | same | same |
| default model | -- | -- | -- | **gaussian** | -- | -- |

`kernel_type` 'bessel' vs 'sobolev' is cosmetic (same filter branch). `flow_sigma` is an ITK
variance in `registration()`; the model stores sigma = sqrt(3.0).

### 21.2 Provenance trace
- **Record (2026-08-17 `9761f69`, numbers updated 2026-08-18 `d9eaefa`)**: commit message states
  "Sobolev SyN (alpha=1.5)"; the evaluator of that date passed `sobolev_alpha=1.5`,
  `fast_smooth=False`, step 0.25, flow 3.0 explicitly -- but **not** `in_loop_inv_steps`, so the
  90-pair numbers were most likely produced with the then-default **6**. The record's 10 traces
  to the `.agents/teamwork_preview_worker_m4_1` ablation ("Fix 3", single pair 0: Dice 0.5990
  with 10 vs 0.6007 with 6, 0% folding both). Kept at 10 per the canonical record; **open
  question** whether it should be 6.
- **Most recent cohort result (2026-09-20 `3e862c9`, 5-arm 90-pair, syn Dice 0.6350, folding
  0.0066%)**: the record stores **no parameters**. Per-pair data
  (`results/cohort_90pair_fullres_random_summary.csv`) for ANTs/syn/tvf were written from
  2026-09-19 19:35, 28 min after `9acc5fa` changed `config.py` syn_config to step 0.35 / flow 2.5
  (a change not mentioned in that commit's message). The generating script is not in the repo,
  git history, `~/Documents/code`, iCloud, any Antigravity session, or any Claude transcript;
  companion scripts (`scripts/add_{syngs,greedy}_to_cohort_benchmark.py`) use
  `evaluate_mindboggle_pair`, so the syn arm almost certainly ran the drifted config. Its
  numbers should not be attributed to the canonical parameters.

### 21.3 Changes
- `syntx.syn`: `in_loop_inv_steps=10` is now an explicit parameter (was a hidden `kwargs`
  default of 6); Sobolev `sobolev_alpha` defaults to 1.5 when not given (was the implicit
  sqrt(flow_sigma)/2 = 0.866; dsti/dsti1 fall-backs unchanged); `fast_smooth` docstring fixed
  (it does switch the Sobolev post-filter).
- `benchmark/config.py` `syn_config` = canonical values; `inverse_steps` (never applied)
  replaced by `in_loop_inv_steps`; new `syn_config_to_syn_kwargs()` raises on unmapped keys.
- `evaluate_mindboggle_pair` (sobolev/syn): calls `syntx.syn` with its defaults unless a
  `config` or explicit keyword is supplied. `run_standard_report_demo`: method defaults, default
  model `sobolev` (CLI fallback too). `high_level_benchmark_run`: SyN parameters default to
  `None` = syntx.syn defaults (its ANTs arm now uses ANTs' own grad_step unless given; was 0.5).
- `run_config.json`: `syn_fast_smooth` false, `in_loop_inv_steps` 10. Record: added
  `fast_smooth: false`, `sobolev_alpha: 1.5`.
- `tests/test_canonical_syn_parameters.py`: captures the *effective* `syntx.syn` defaults
  (intercepting `SyNTo.fit`) and asserts the record, `config.py` and `run_config.json` all agree.

### 21.4 Not done / follow-ups
- The `flow_sigma` "has no effect with spectral regularizers" warning is inaccurate when
  `fast_smooth=False`: flow_sigma sets the Sobolev post-filter width.
- Result records should store the resolved parameter set and commit hash so a lost generator
  script cannot make a cohort result unattributable again.

## 22. Machine-captured benchmark provenance (`syntx.provenance`)

The 2026-09-20 5-arm cohort's parameters were published without a manifest and its
generator script was never committed; a reconstruction written later on the executing
machine (`docs/provenance/mindboggle_90pair_5arm_benchmark_2026-09-20_parameters.md`)
conflicts with the committed code for the syntx.syn arm (Sec 21.2). Fix, so this cannot
recur (details: docs/BENCHMARKING_GUIDE.md Section 7):

- `syntx.provenance`: `capture_registration_calls()` records every registration call
  (syntx syn/tvf/syngs/greedy/robust_affine/auto_reg, ants.registration) with explicit
  arguments **and** the resolved parameters captured at each model's `fit()` (so hidden
  defaults such as the old `in_loop_inv_steps=6` / `sobolev_alpha=0.866` are visible);
  `build_manifest()` adds git commit + full uncommitted diff + untracked package sources,
  the invoking script's full text, environment (host, versions, devices, load).
- `evaluate_mindboggle_pair` / `evaluate_pair` / `evaluate_msd_pair` return
  `result["provenance"]` (decorator `with_provenance`); `worker.py` keeps it.
- Published records: `record_result()` derives provenance from per-run manifests and
  refuses mixed commits/diffs/parameter sets; `tests/test_provenance_records.py` fails for
  any record in `best_parameters.json` without it (27 pre-existing records frozen as legacy).
- The 2026-09-20 record is marked "reconstructed, not captured" with its conflicts listed.

## 23. mbhard `syntx.syn` parameter sweep (first run with captured provenance)

8 variants on pair 44 from clean commit `1dba81e`; full table and findings:
`docs/provenance/syn_param_sweep_mbhard_2026-09-28.md`. Summary: the canonical
configuration (fast_smooth off, alpha 1.5, step 0.25) gives Dice 0.6083-0.6085, 0 %
folding, interior max inverse error 0.79-0.82 mm in 68-69 s. Reducing smoothing
(`fast_smooth` on, or the old implicit alpha 0.866) buys +0.003-0.004 Dice at the cost of
folding (0.003-0.007 %), Jacobian min 0 and a ~6 mm interior inverse outlier -- this, not the
inverse solver, is the source of the 6.7 mm max seen earlier (Sec 20.5). The committed
2026-09-20 config is the least regularised (folding 0.009 %, inverse 6.7 mm).
`in_loop_inv_steps` 6 vs 10 makes no difference. Follow-ups: confirm on more pairs; the
121 s runtime with `stationary_boundary=False` (vs 69 s) is unexplained; warnings raised
inside `syntx.syn` with `stacklevel=2` now point at the provenance wrapper (cosmetic).

## 24. Automated tuning (`syntx.benchmark.tune`) -- greedy and syn defaults updated

Tuner: baseline twice -> noise -> margin; one-at-a-time screen; refinement with combinations,
bracketing and **compensating pairs** (Dice-gaining infeasible move x topology-clean move, needed
when the incumbent sits on the Dice/topology edge); every accepted winner confirmed by a repeat;
objective = mean Dice over Mindboggle pairs 77/44/0 with per-pair constraints relative to the
defaults (Dice drop <= 0.001, folding, Jacobian min, interior inverse cap -- the latter only where
the defaults are fold-free); reg_iterations held at [100, 100, 20]; resumable cache keyed by the
registration-code fingerprint; live.md / --table monitoring; codify on a branch.

| method | change | mean Dice | per pair (77 / 44 / 0) | topology / inverse |
|---|---|---|---|---|
| greedy | learning_rate 0.25 -> 0.375 | +0.0105 | +0.0139 / +0.0069 / +0.0107 | 0 % folding everywhere (no inverse) |
| syn | grad_step 0.25 -> 0.4, flow_sigma 3.0 -> 2.4, sobolev_alpha 1.5 -> 2.25 | 0.6214 -> 0.6235 | 0.6133->0.6163 / 0.6084->0.6126 / 0.6426->0.6416 | folding 0.0042 -> 0.0007 % on 77, 0 % on 44/0; global max inverse 3.90->3.71, 2.98->2.79, 4.86->2.91 mm |

The syn default is the user-selected near-tie of the confirmed winner (grad_step 0.5, alpha 2.25:
+0.0023, 0 % folding everywhere, ~20 % slower); both recorded (`syntx.syn/tuned_syn_2026_09_29`,
`syntx.syn/canonical_2026_09_29`). `sobolev_alpha` is now an explicit `syntx.syn` parameter;
`auto_reg`'s syn preset now uses the defaults (it forced fast_smooth=True). Reports:
`docs/provenance/tuning/{greedy,syn}_2026-09-29.md`. Canonical-parameter tests follow the new
records. The syn default rests on 1 run per pair (the winner on 2).

**Robustness caveat (found by the full suite):** on a tiny synthetic 28^3 blocky pair
(`tests/test_restrict_transformation.py`), the tuned step (grad_step 0.4, flow_sigma 2.4)
oscillates -- the loss is best at epoch 0 of each level and the best-loss logic keeps the
near-zero field -- while the previous step (0.25, 3.0) descends monotonically. The defaults
were tuned on three real T1 brain pairs only; that mechanism test now pins its parameters.
Worth checking on more pairs / other anatomy before relying on the defaults broadly.

## 25. Canonicalisation / codify generalised (one declaration per method)

Until now each method was canonicalised by hand (three near-duplicate canonical tests; codify
knew only greedy; syn's new defaults were applied by hand). Now `tune.METHODS` declarations
(`_CANONICAL`) drive one generic test (`tests/test_canonical_parameters.py`, replacing
`test_canonical_{syn,greedy,syngs}_parameters.py`) and codify (`targets_for`, placement of each
default: signature / dict constant / config.py / run_config.json; `record_canonical` adds a
record + `docs/provenance/canonical.json` pointer). Found immediately: `run_config.json` had
drifted for greedy (grad_step 0.5 vs 0.375) and syngs (alpha 0.35 / max_step_norm 0.20 vs
0.45 / 0.19) -- fixed. Guide: docs/BENCHMARKING_GUIDE.md Section 7.3.

## 26. MPS run-to-run non-determinism (syngs, tvf) -- fixed at the source

Symptom: identical syngs runs on MPS differed by up to 0.03 Dice on Mindboggle pairs (pair 77:
0.5578 vs 0.5640 even after v5.4.65's RegAdam relative eps); CPU was bit-exact. Debugged at the
coarse level only (pair 77, `reg_iterations=[N,0,0]`) by hashing grads/params per optimiser step
and, where grads first differed, every ATen op of a repeated backward over the SAME graph
(`retain_graph`). Four non-deterministic GPU/MPS backward paths, all float-atomic accumulation:

| # | op | where | replacement |
|---|---|---|---|
| 1 | `grid_sampler_3d_backward`, input gradient | sampling the trainable velocity (syngs shooting, tvf) | `core.grid.DeterministicGridSample` |
| 2 | `avg_pool3d_backward` | LNCC (`cc2`) box filter | `core.losses.box_mean_nd`: stride-1 zero-padded box sum is self-adjoint, grad = pool(g / count) via the deterministic forward kernel |
| 3 | `grid_sampler_3d_backward`, grid gradient (MPS only) | same sampler | analytic, per-point fixed-order sum |
| 4 | stock `F.grid_sample` in `compose_grids` | syngs inverse-consistency term | routed through #1/#3 |

`DeterministicGridSample` on MPS is a Metal kernel (`core/mps_kernels.py`, `torch.mps.compile_shader`):
one thread per sample point; the grid gradient summed in-thread; input-gradient contributions
added as fixed-point integers (31-bit, split into two int32 atomic limbs -- integer addition is
associative, so the sum is exact and order independent; 65536 contributions per voxel before a
limb could overflow). Elsewhere (CUDA) a PyTorch int64 fixed-point fallback. Gradients equal the
stock ones to float32 rounding (~1e-7 relative). `sample_field_cf` / `grid_sample_nd` /
`compose_grids` use it automatically when a differentiable field is sampled on MPS/CUDA
(`DETERMINISTIC_SAMPLE_DEVICES`, `DETERMINISTIC_POOL_DEVICES` switch it off).

Result: syngs 3 x 100 coarse iterations and tvf 2 x 100 -- bit-identical (tvf with the stock
kernels: grads differ from step 0). Speed, end to end, interleaved, `[30,0,0]`, best of 2-3
(machine loaded by another job): syngs 21.05 s stock vs 20.64 s deterministic, tvf 9.58 vs
9.70 s (+1.3 %); kernel alone at the full 160x256x256 size is on par with stock (noisy: 48 vs 69,
59 vs 47 ms). A first pure-PyTorch version cost +60 % end to end -- hence the Metal kernel.

RegAdam `eps_rel` (v5.4.65) is back to default 0 (the tuned behaviour); it only damped the
noise, and is kept as an opt-in (`adam_eps_rel`). Tests: `tests/test_mps_determinism.py`.

## 27. SyNGS tune on deterministic code -> new defaults; GPU contention; flow_sigma / fast_smooth

Tune `results/tune_syngs_20260929_det` (start `max_step_norm=0.3`; baseline noise 0.00000, margin
0.0005): winner `max_step_norm` 0.19 -> 0.3, `alpha` 0.45 -> 0.675, +0.0159 mean Dice, 0 % folding
on 77 / 44 / 0 (0.5842 / 0.5926 / 0.6203), inverse error at the defaults' level -- now the syngs
default (codify branch `tune/syngs-20260930`, merged; record `syntx.syngs/canonical_2026_09_30`).
Larger steps gain up to +0.033 but fold, even with stronger alpha.

**GPU contention corrupts MPS results.** syngs is bit-reproducible across processes on an idle GPU,
but with another process on the GPU, stock MPS occasionally computes / reads back wrong values
(pure torch: `(ones*k).sum().item()` wrong once in ~23k under load) and registrations diverge.
`scripts/audit_tune_runs.py` re-runs cached evaluations (pinning the tune-time defaults) and checks
bit-identity: baseline, winner and flow_sigma=1.5 reproduced 9/9; the start point's pair-77/44
runs did not -- contaminated in the first tune process. That contamination, not a code path, was
the apparent flow_sigma "step change" (1.5 == 2.25 == 3.0 == 3.75: flow_sigma's value is unused with
spectral regularisers). Run tunes with nothing else on the GPU.

Fixes: `fast_smooth` removed from syngs (it never read it; passing it now raises TypeError) and
from the tune space; `flow_sigma` no longer searched for syngs; `flow_sigma=0` now really
disables the velocity smoothing (it was silently replaced by 3.0), None = default, negative
raises; tune's syngs `alpha` default read live from `default_alpha(3)` (was a stale 0.45 literal
after codify); cohort provenance ignores per-image model geometry; auto_reg's syngs preset uses
the syngs defaults (it hard-coded alpha 0.35 / lr 1.2 / max_step 0.25).

**Follow-up (v5.4.72): one strength parameter per regulariser.** syngs: `sobolev` / `dsti` / `dsti1`
use `alpha` (0 = no smoothing) and reject `flow_sigma` / `gaussian_sigma`; `gaussian` / `bspline` use
`flow_sigma` (default None = 3.0, 0 = no smoothing) and reject `alpha` / `sobolev_alpha` /
`dsti_alpha`; an unknown regulariser raises. `flow_sigma` default is now None and it is gone from
the syngs canonical records / config / run_config / tuner spec. Default syngs results unchanged
(winner pair 0 re-run bit-identical). Checked the other methods (flow_sigma 1.5 / 3.0 / 4.5): syn,
greedy and tvf (every regulariser) genuinely use the value -- only syngs had the gotcha.
Found on the way, **RegAdam bug**: any regulariser other than 'sobolev' with `gaussian_sigma > 0`
(RegAdam's default is 1.5) took the Gaussian branch, so `dsti` / `dsti1` were never applied --
TVF's benchmark configuration (regularizer dsti1, gaussian_sigma = flow_sigma) was Gaussian-smoothing
its RegAdam steps (and warning that flow_sigma "has no effect"). Fixed; this changes TVF results
with dsti / dsti1 -- to be re-baselined in the TVF canonicalisation / tune.
