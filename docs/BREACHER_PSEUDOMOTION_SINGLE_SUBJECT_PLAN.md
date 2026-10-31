# breacher pseudomotion: single-subject identification and model (plan, 2026-10-31)

Status: plan with measurements. No package code changed. Companions: `BREACHER_PSEUDOMOTION_REVIEW_PROPOSAL.md`
(E1–E4, D1–D4) and `BREACHER_PSEUDOMOTION_GROUP_MODEL_PLAN.md` (S1–S3). The question here: can the breathing-locked
displacement be identified and modelled in one run, without DICOMs, physio logs or the rest of the cohort?

## Answer in one paragraph

Partly, yes. One run contains its own clock: the interleaved acquisition records the odd-index slices about 1 s before
the even-index slices (sidecars: means 0.47 s and 1.46 s). Registering the two half-volumes separately and comparing
them in the breathing band separates a displacement that changes *between* volumes (the scanner moved the field of
view: PACE) from one that changes *continuously* (bed or head). On ten breacher runs the coherent part of the
displacement is volume-locked in eight (phase 3–33° between the halves, against 69–74° predicted for continuous
motion and 3–5° measured for a volume-locked synthetic), continuous in one (sub-2026, which was acquired with a
different sequence variant) and mixed in one (sub-0102). A single-subject model that removes the volume-locked share
from FD is therefore feasible with two extra registrations per run. What the two-set clock cannot do is recover the
head trajectory itself: that needs three or more slice sets and a solver that fits per-set poses jointly, which is
the second, larger stage proposed below. A sizeable per-set component (30–60 % of band variance) that is coherent
with neither half is not rigid head motion either; its origin is open.

## Measurements (ten runs; `scripts/single_subject_signatures.py`, outputs `ss_real.json`, `ss_calib*.json`)

Runs: the eight TR-2.0 runs with real EPI sidecars (sub-1239, 2026, 4667, 4786, 5315, 6522, 8910, 9162; early set =
odd k in every one, 0.47 s vs 1.46 s) plus sub-3315 and sub-0102 (field-map sidecars; same timing assumed). Full
volumes: the production call (joint engine, PE debias; raw pose reconstructed). Half-volumes: the same joint engine
on the 18-slice stacks (7.32 mm spacing). Band 0.15–0.25 Hz. Rotations as mm at 50 mm.

### M1. Calibration of the clock test (synthetic, from the runs' own mean images and noise)

| planted displacement (z, 0.15 mm, 0.21 Hz) | phase late vs early, t_sl | coherence | r(early, late) band | sd of late − early |
|---|---|---|---|---|
| volume-locked (both sets shifted together) | 3–5° | 0.69–0.78 | 1.00 | 0.011 mm |
| continuous (each set at its own time) | 69° (predicted 72–74°) | 0.53–0.62 | 0.26 | 0.122 mm (= 1.2 × planted) |

Image noise alone leaves the two half-sets agreeing to 0.011 mm in the band. The aliasing branch does not rescue a
continuous reading: breathing at 0.30 Hz aliases to 0.20 Hz with a predicted phase of −107°, and at exactly
0.25 Hz the halves are in anti-phase; no true frequency gives phase 0 for continuous motion.

### M2. Clock test on real runs (t_sl, the dominant axis)

| run | sequence | phase late vs early | coherence | r(early, late) | slope late~early | sd(late − early) | reading |
|---|---|---|---|---|---|---|---|
| sub-3315 | fm2d2 sidecar | +3° | 0.68 | 0.83 | 0.74 | 0.116 | volume-locked |
| sub-5315 | ep2d_pace | −3° | 0.52 | 0.62 | 0.52 | 0.118 | volume-locked |
| sub-4667 | ep2d_pace | −6° | 0.49 | 0.74 | 0.64 | 0.191 | volume-locked |
| sub-6522 | pace_physio | +17° | 0.84 | 0.82 | 0.65 | 0.054 | volume-locked |
| sub-8910 | ep2d_pace | +18° | 0.60 | 0.63 | 0.55 | 0.170 | volume-locked |
| sub-9162 | ep2d_pace | +22° | 0.75 | 0.74 | 0.79 | 0.130 | volume-locked (+ some continuous) |
| sub-4786 | pace_physio | +24° | 0.81 | 0.76 | 0.92 | 0.116 | volume-locked (+ some continuous) |
| sub-1239 | pace_physio | +33° | 0.57 | 0.62 | 0.68 | 0.093 | mixed |
| sub-0102 | fm2d2 sidecar | −43° | 0.61 | 0.36 | 0.32 | 0.266 | mixed |
| sub-2026 | pace_physio2 | +73° | 0.33 | 0.01 | 0.01 | 0.161 | continuous |

The same holds on t_pe (phase −21° to +9°, coherence 0.55–0.89, r 0.76–0.91 in nine runs) and t_ro (phase
2–29°, coherence 0.45–0.77). The field of view is displaced on all axes at volume boundaries, which is what a
prospective correction does. The PE displacement that `debias_pe_pseudomotion` attributes to the respiratory field is
itself mostly volume-locked here; the FD outcome is the same (not head motion), the mechanism is different.

### M3. What is not rigid head motion, by construction

A rigid head displacement h(t) sampled by the two sets gives coherent half-set series: in phase if volume-locked,
lagged if continuous. The real runs have r(early, late) of only 0.6–0.8 while the calibration gives 1.00 for the same
shift under image noise. So 35–60 % of each half-set's band variance (sd 0.06–0.19 mm) is band-limited but
incoherent between the halves. It cannot be a rigid displacement of the whole head. Candidates: a non-rigid or
slice-dependent displacement (a z-dependent stretch, field-gradient-dependent PE shift), interleaved spin-history
modulation that the 18-slice registration reads as z-translation, or aliased cardiac pulsation (1.0–1.2 Hz aliases to
0.0–0.2 Hz at TR 2.0 s and is sampled 1 s apart by the two sets, hence incoherent). The last one is real tissue
motion; the first two are not. This component is not separated by the two-set clock.

### M4. Direction and pivot (per run, band-limited 6-DOF steps)

PC1 holds 54–92 % of band step variance (white noise: 21 %) and points along t_sl in every run (loading 0.68–0.99),
with pitch loadings of −0.72 to +0.45 and t_pe −0.07 to −0.35. A rotation about a fixed pivot explains 1–38 % of the
band-limited translation (relative residual 0.62–0.99), and the fitted pivots scatter (|r| 18–184 mm). The
displacement is a translation along the slice axis, not a nod about the occiput.

### M5. PE field shift as a breathing reference

The PE B0 shift estimated by the production debias is breathing-band (85–98 % of its step power) in every run, and
coherent with t_sl at only 0.19–0.65, mostly in anti-phase. It is a usable per-run breathing clock for the band
choice, not a regressor for the displacement.

### M6. Jerk response (pooled, full-volume poses)

Only six isolated jerks exist across the ten runs (the alternation itself defeats the isolation criterion). The
pooled displacement along the jerk direction is 0.81 of the jerk at lag 0, then 0.52–0.66 at lags 1–6: neither the
full reversal a lag-1 prospective correction would give nor a clean step. Three control runs (ExpArt sub-EAS024,
SOCOM sub-Blast-01, FPA sub-BLAST071; first 300 frames, same call) yield one isolated jerk in total, with a response
of 0.99 then 0.41–0.64. The jerk test is underpowered in both groups and decides nothing here.

## Single-subject model

### Stage 1 (two slice sets; feasible now; +50 s per run)

Per run:
1. Register the early and late half-volumes with the joint engine (as above). Express in the acquisition frame.
2. Clock statistic per axis: coherence-weighted phase and coherence between the halves in the band chosen from the
   run's own PE B0 spectrum (M5), or 0.15–0.25 Hz when that is flat.
3. Decision, calibrated on the run's own synthetic pair (M1, rebuilt from its mean image): phase within the
   volume-locked calibration ± 2 SE and coherence ≥ 0.4 → the coherent band component is volume-locked; phase within
   the continuous prediction → continuous; else mixed. Runs that fail the calibration (no coherent band) are left
   unchanged.
4. FD for censoring: subtract from the band-limited full-volume pose the in-phase share held in common by the two
   halves, scaled by r²(early, late), before differencing. Translations, rotations: all six DOF, each with its own r².
   The incoherent per-set component (M3) stays in FD (conservative: it may contain pulsation). The applied transforms
   are untouched.

Gates (pre-registered):
- Isotropic white jitter (0.05 and 0.12 mm per axis): the two halves are incoherent, r² ≈ 0, nothing removed; FD
  retained ≥ 95 %.
- Continuous synthetic nod (0.15 mm, 0.2–0.3 Hz, each set at its own time): phase reads continuous; nothing removed.
- Volume-locked synthetic (0.15 mm): FD after removal within 10 % of the FD of the planted residual (zero plus noise).
- The two readings on sub-2026 and sub-0102 are reported as mixed/continuous and get no removal; that is the
  correct outcome for the model, whatever those runs turn out to be.

Expected effect on breacher: removal of 36–69 % of the band variance in eight runs; FD mean falls by roughly a third
on the strongest runs (to be measured in the gate, not estimated here).

### Stage 2 (three or more slice sets; the full model; syntx joint engine)

Model each slice set s of volume t as pose h(t_s) + F_t: h a continuous head (or bed) trajectory on a temporal basis
limited to 0.5 Hz, F_t one free 6-DOF offset per volume (the field of view). Identifiability: with n sets per TR
and h ≤ 0.5 Hz, unknowns per 600 s run are 600 (h) + 300 (F) and observations 300·n·6; n = 3 is critical, n = 4
over-determined. Sets of 12 or 9 slices (11 or 14.6 mm spacing) are poorly conditioned for rotations, so h's rotations
need the surround prior and F's rotations may be tied to F's translations through the pivot of the previous volume.
This is a new design in `motion_design` (`slice_sets` with per-set acquisition times) solved in the joint engine
against the full-series mean. Outputs: h (head), F (volume-locked), per-set residual (M3's component). FD from h.
Gates: the three slice-timed generators (H1 lagged FOV feedback, H2 continuous bed, H3 occiput nod) at ≥ 90 % correct
attribution of band variance; the white-jitter null; applied transforms (h + F) reproduce the current ones within
0.02 mm. Cost: several days of solver work; it is the only route to a per-volume head trajectory.

### What stays open at single-subject level

- Whether the volume-locked displacement is PACE. The signature (all six DOF moved at volume boundaries) fits it; the
  protocol flag is in the DICOM CSA header only. sub-2026's continuous reading coincides with its distinct sequence
  (`ep2d_pace_physio2`, syngo B18P), n = 1.
- The incoherent per-set component (M3): rigid-motion-free by construction, origin unknown; Stage 2 isolates it,
  neither stage explains it.
- The jerk test is underpowered on these runs (six events).
- Ten runs, one per subject, half of them sharing one protocol variant.


## Stage 1 built (WIP, 2026-10-31) and applied to three unseen runs

`src/syntx/motion_slice_sets.py` (`syntx.slice_set_clock`, `slice_set_clock_from_poses`; opt-in, nothing in the default
path changes; tests `tests/test_motion_slice_sets.py`: white jitter untouched, volume-locked removed, continuous kept).
Per-run calibration is not implemented; fixed thresholds (volume-locked |phase| <= 35 deg, continuous >= 55 deg,
cross-power-weighted coherence >= 0.4). Benchmark `benchmarks/breacher_slice_set_clock_wip.py`; report
`docs/reports/breacher_slice_set_clock_wip_2026-10-31.{png,json}`. Cost: +32 to 38 s per run on top of 26 to 31 s.

| run (fm2d2 sidecar, early set assumed odd) | t_sl phase / coh / reading | t_pe | t_ro | FD mean before -> after | > 0.5 mm before -> after |
|---|---|---|---|---|---|
| sub-0519 | +52 deg / 0.61 / mixed, 0 removed | VL, share 0.86 | VL, 0.55 | 0.435 -> 0.318 | 22.4 -> 9.4 % |
| sub-0879 | -105 deg / 0.74 / continuous, 0 removed | VL, 0.51 | VL, 0.43 | 0.240 -> 0.205 | 5.4 -> 3.0 % |
| sub-1008 | -22 deg / 0.42 / VL, share 0.14 | VL, 0.67 | VL, 0.62 | 0.344 -> 0.253 | 13.7 -> 8.7 % |

Unlike the ten calibration runs, the slice-axis displacement is not volume-locked in two of these three (sub-0879's
-105 deg is the prediction for continuous motion at a true 0.30 Hz aliased to 0.20 Hz). The PE and readout
displacements are volume-locked in all three, so the FD reduction here comes from those axes. Jerks are preserved
(the spikes in the figure are unchanged); the reduction is in the baseline. The production `fd_bandlimited`
(Jenkinson) is close to `fd_head_vl` on these runs. Reading on the dominant axis therefore varies by subject; a
per-run decision, which is what Stage 1 makes, is required rather than a cohort rule.


## Generality and automatic use (answered 2026-10-31; not implemented)

Opt-in: strictly. Nothing calls `slice_set_clock`; it adds `fd_head_vl` and `slice_set_clock` to a result it is given.

Applies to: interleaved 2-D acquisitions whose odd and even slices occupy disjoint halves of the TR, slice axis third,
>= 64 frames, breathing band below Nyquist (default 0.15-0.25 Hz needs TR <= 2 s; pass `band_hz` otherwise),
single-contrast series (half-sets use the "none" design; ASL/DWI would need the design passed through). Refused, by
the sidecar timing, for sequential and multiband orders (parities overlap in time; a multiband-2 ascending pattern
passes a parity-mean test and fails the disjoint-range test, which is now the criterion). Thresholds are fixed from the
breacher calibration; per-run synthetic calibration is what makes the decision portable across geometries.

Automatic cohort mode (`slice_set_clock="auto"` in antsxfunctional, proposed): per run, (1) applicability gate:
sidecar SliceTiming parity-disjoint -> run; no SliceTiming -> run only if `MultibandAccelerationFactor` is absent or 1
and the data interleave score along the slice axis is high; else skip with a recorded reason; (2) the decision uses
|phase| and r^2, neither of which depends on which parity is early, so a missing SliceTiming does not change the FD
outcome, only the sign labels; (3) censoring uses `fd_head_vl` when present, else `fd_bandlimited`, with the reading,
shares, `early_set_source` and gate reason in provenance; (4) a cohort QC table of readings per DOF. Reliability
limits: the data interleave score is a weak gate on its own (breacher sub-0102 0.89 but sub-1239 0.19, against
ExpArt 0.02 and FPA 0.05 on 120 frames), so without SliceTiming the protocol must be known to be interleaved; the
failure direction of a wrong reading is conservative (nothing removed) except for an in-phase non-motion component
common to both half-sets, which on a new cohort is unverified; the censoring policy itself still awaits G5.

## Work packages (SS1 built as WIP; others not started)

| WP | content | where | cost |
|---|---|---|---|
| SS0 | commit the probe scripts under `benchmarks/breacher_single_subject/` with a dated report of M1–M6 | syntx | small |
| SS1 | Stage 1 as `syntx.motion.slice_set_clock(image, sidecar)` returning the clock table, calibration and the r²-scaled band removal as a separate FD series (`fd_head_vl`), never a default | syntx | 1 day |
| SS2 | gates for SS1 (white jitter, continuous nod, volume-locked), run on the ten runs; report | syntx benchmarks | 0.5 day |
| SS3 | antsxfunctional: optional censoring on `fd_head_vl`, provenance of the clock reading per run | antsxfunctional | 0.5 day |
| SS4 | Stage 2 design and solver; generators; gates | syntx | days |
| SS5 | G5 downstream gate (test-retest connectivity) with Stage 1 censoring vs current | antsxfunctional / antsxmm | after the TR fix re-run |

## Scripts and outputs (session scratchpad)

`/private/tmp/claude-501/-Users-stnava-data-repos-syntx/bd7342f1-d24a-46cc-a87f-cc8030bd6349/scratchpad/`:
`scripts/single_subject_signatures.py` (real and calib modes), `scripts/jerk_response.py`, `scripts/control_probe.py`;
outputs `ss_real.json`, `ss_calib.json`, `ss_calib2.json`, `control.json`.
