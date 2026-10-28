# Plan, 2026-10-28: one solver, four cases, an ISMRM abstract

Goal: an ISMRM abstract that presents one motion solver for every time-series modality, with the acquisition design in
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

## Steps

**1. Baseline (joint engine) table and figure.** Run `engine="joint"` on the four cases; per case: FD mean / max,
frames above 0.5 mm on raw and band-limited FD, pose-domain shares on the readout axis (head / applied acquisition
term / removed design-locked term), image-domain variance explained (design, motion after the design, residual) on
raw and corrected, residual-to-FD correlation, time. Figure: one row per case (pose along readout and PE with the
applied term, FD with band-limited FD, variance bars). Compute about 10 minutes. Look at: the table and the figure.
Decide: which numbers carry the story; whether the decomposition becomes a library module now (it must, for the
abstract's software claim: `syntx.motion_decomposition`, unit-tested on a synthetic series).

**2. The DWI design decision.** Implement l = 4 per shell for b >= 2500 in the design (`projected_design_matrix` and
the `joint` level columns) plus the misattribution-angle diagnostic. Run unified / joint / joint-l4 / projected-l4 on
the two DWI cases. Look at: readout step, PE step, FD, alignment, l = 4 share after the engine, misattribution angle,
one HTML. Decision rule, fixed now: a richer design wins only if the readout step is within 0.02 mm of unified's on
both cases, the residual share is not higher, and the l = 4 share after the engine falls below the motion share.
Compute about 20 minutes.

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

## Engineering rules that hold for every step

Measurements in library modules with unit tests, `benchmarks/` scripts as thin CLIs writing HTML plus JSON under
`docs/reports/`; CPU for numeric tests; no verdict outside antsxqc, every new key in the catalog; physical space
through ANTs with `[warp, affine]` chains and one interpolation; a gate that is not rerun is reported as not run.
