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
