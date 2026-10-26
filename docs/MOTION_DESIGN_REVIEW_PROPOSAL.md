# Design-driven motion correction: correctness review and fix proposal

Status: proposal (2026-10-26). Scope: `syntx.motion_design`, `syntx.motion_reference` (level links and
per-level offsets), `syntx.phase_encode`, `syntx.acquisition_axes`, and the call sites in antsxfunctional
0.19.0, antsxdwi 0.30.0 and antsxslowflow 2.9.0. Reviewed at syntx v6.23.0 (`613c826`).

Evidence: code reading, synthetic pose probes against `motion_reference._group_offsets` and
`motion_design.level_layout` (no registration), and the targeted baseline run before writing
(`antsxqc.lint src/syntx` 0 findings; pytest motion_design / acquisition_axes / phase_encode /
phase_encode_guard 202 passed). Downstream working trees were clean at antsxdwi `77732c3`,
antsxfunctional `7b3985d`, antsxslowflow `f12c83f`.

## 1. What the module does

`make_design(name, ...)` turns acquisition metadata into a `Design`: per-frame integer levels (frames of a
level share a contrast and a reference; level 0 is the anchor) and optional per-frame columns that can
carry an acquisition-locked pose bias. `motion_correct_design(image, design, engine="unified")` registers
every level to a reference of its own contrast (`motion_correct_grouped`), links adjacent levels either by
identity (levels interleaved in time) or by a rigid registration of their corrected means, and removes the
per-level offset `beta` estimated from surround differences of the composed trajectory with Tukey weights.
DWI adds the eddy term along phase encode tied to the gradient and the five l = 2 direction terms as
in-level columns. BOLD is one level and reduces to registration to the series mean.

The phase-encode axis comes from `resolve_phase_encode` (sidecar, label, entity, explicit name; no default
axis; conflicts raise) or, with `allow_inference=True`, from a caller-supplied callable such as
`acquisition_axes.infer_phase_encode_axis` (Nyquist ghost on the two in-plane axes, cross-checked by the
frame-to-frame shift spread; slice axis from metadata, voxel thickness or slice interleave).

## 2. Findings

Ordered by severity. "Probe" means a synthetic trajectory run through the function named, with no
registration.

### F1 (high). Offset identifiability is judged against any other level, not against the anchor

`level_layout` marks a level `beta_estimable` when at least 4 of its frames have a neighbour of *any*
other level; `_group_offsets` uses the same criterion. `_unified` then links the level by identity and
leaves the level-vs-anchor difference to the local offset estimate.

For the common multi-shell layout "b0 block first, then interleaved shells" (`[0]*5 + [1,2]*n`) both
shells are "estimable" because they alternate with each other, while the b0-versus-shell difference is
supported by one boundary. Probe (three seeds, planted offsets 1.0 and 1.3 mm, a real 1.5 mm head jump
at the block boundary): the jump is attributed in full to both shell offsets (1.5 mm each). In the real
pipeline the same jump is absorbed by the separate level means, the surround fit sees only the single
boundary, and the shell space stays misaligned to the b0 anchor by an unestimated constant. Either way the
outcome is wrong for this layout. Single-shell b0-block protocols are safe (one mixed frame, registration
link); multi-shell b0-block protocols are not, and antsxdwi now defaults to `engine="unified"`.

### F2 (medium). "identity" means "same as the previous level"; the engine assumes "same as the anchor"

`motion_correct_grouped` builds each frame's chain as `[cross_1, ..., cross_g, own]`; an identity entry adds
nothing, so level g inherits every earlier registration link. `_unified` chooses identity on an
anchor-based argument (an interleaved level shares the series' mean head position). With a registered
level before an identity-linked one (b0 anchor, blocked shell 1, interleaved shell 2) the later level
carries the earlier cross registration. Occurs only with F1-type layouts; fix together.

### F3 (medium). Two identifiability rules that disagree

`_group_offsets` adds `n_frames >= 8`, at least 8 frames among anchor plus estimable levels, and at least
4 active surround rows; `level_layout` knows none of these. Random-layout probe: 41 disagreements in
about 1500 valid layouts (for example T = 9, levels `[2,1,1,0,2,0,1,0,1]`: the layout says level 1 is
estimable, the estimator returns nothing). The engine has already linked the level by identity on the
offset premise; the level is left uncorrected and the result only records `not_identifiable_groups`.

### F4 (medium). `tag_control` levels are not 0-based without a `control` volume

`_design_tag_control` hard-codes control = 0, label = 1, other types from 2. `["label"]*8 + ["m0scan"]`
gives levels {1, 2}; `["deltam", "m0scan"]` gives {2, 3}. `motion_correct_grouped` raises
"groups must use the labels 0..k-1". antsxfunctional avoids the deltam case by not running motion
correction on it; the design itself must re-index.

### F5 (medium). A (3, T) gradient table is accepted silently

`_design_shells` does `np.asarray(bvecs).reshape(len(bvals), 3)`. Probe: a transposed table is accepted
and yields different (meaningless) covariates with no error. antsxdwi's `gradients.read_bvals_bvecs`
transposes before the call, so production is safe today; the entry point is not.

### F6 (medium). b0 rows enter the eddy covariates; gradient rows are not normalised

`pe_affine_covariates = g` uses every raw row. Converters that write non-zero vectors for b0 volumes
(probe: `[1, 0, 0]`) give the b0 level a non-zero eddy design, so an eddy term is fitted on frames that
have no diffusion gradient. The l = 2 columns already zero those rows; the eddy rows must too and be
unit-normalised (the term is fitted per shell, so amplitude does not matter, but norm-encoded b-values
would otherwise leak in).

### F7 (medium). Slice-axis metadata does not reach the PE debias; the inference hook cannot take levels

`motion_correction(debias_pe_pseudomotion=True)` calls `separate_bold_pe_pseudomotion` without `sidecar`
although that function accepts one; `motion_correct_grouped` calls `infer_slice_axis(image, slice_axis,
levels=groups)` without a sidecar. A BOLD run whose sidecar has `SliceTiming` goes to image inference and
can raise `AcquisitionAxisUndecidable` where metadata would have decided. `infer_phase_encode_axis(image)`,
the `resolve_phase_encode(inference=)` hook, cannot receive `levels`, `slice_axis` or `sidecar`, although
the module's own thresholds say DWI and ASL interleave detection needs the level means removed.

### F8 (low). The inferred axis loses its evidence and goes through a lossy round trip

`infer_phase_encode_axis` returns the physical axis (argmax of the direction column);
`resolve_phase_encode` maps it back to a voxel axis by argmax. The ghost identifies the *voxel* axis
exactly; the round trip is exact only for aligned headers. The scores (ghost ratio, shift spread, slice
axis source) are dropped from the record.

### F9 (low). `custom` designs lose their encoding columns in the unified engine

The docstring promises caller-supplied levels and encoding columns; `_unified` forwards
`encoding_covariates` for `shells` only.

### F10 (low). `pe_axis` validation

The debias path accepts a voxel index; the eddy path (`motion_correct_grouped`, the batched solver) needs
a 3-vector and fails with a shape error deep in torch for an index. `Design` has no validation of
`levels`, column row counts or `pe_axis`.

### F11 (low). The reference iteration stacks T copies of one image

`_corrected_level_mean` returns `[mean] * T`; `motion_correction(reference=list)` stacks every reference on
the device.

### F12 (low). BOLD's systematic error is not part of its design

`make_design("none")` cannot carry `pe_axis`; the respiratory PE shift is handled only by the opt-in
`debias_pe_pseudomotion` kwarg, outside the design record.

### F13 (docs). antsxdwi `motion_correct` docstring

`img4d` is documented as "or path" (`motion_correct_design` needs `.shape`); `backend` is documented as
used but is ignored in the grouped path.

Verified and left alone: `_real_sh_l2` spans the traceless quadratics with rank 5; the eddy translation
gauge is resolved by the surround-Tukey projection within shell and is orthogonal to the even l = 2 space;
`_debias_cross_group_offsets` places the correction first in the chain and FD counts the rigid part only;
`phase_encode` precedence, conflict and vocabulary rules; LPS conventions and `physical_vector` for BIDS
labels; `ghost_scores` and `interleave_scores` work in voxel index order, matching ANTsPy `.numpy()`;
antsxslowflow `_state_offset_record` and antsxdwi `bias_by_group` consume the unified result correctly.

## 3. Defaults for the three debias switches

| switch | changes | evidence | proposed default |
|---|---|---|---|
| `cross_group_debias` (unified: `beta != "none"`) | the transforms: a per-level constant offset removed from the chain | real 3-shell DWI (FPA sub-BLAST079): 0.16-0.17 mm / 0.08-0.09 deg per shell, mean FD 0.173 -> 0.163 mm; ASL phantom: tag-control bias removed. Valid only for levels interleaved with the anchor-connected set (F1). | On in the design engine (as now), gated by the adjacency rule of WP1 so blocked anchors fall back to registration. `motion_correct_grouped(cross_group_debias=False)` stays the low-level default (dispatch parity; `group_offset_fd` still reported). |
| `debias_pe_pseudomotion` | reported motion and FD only; transforms unchanged; needs `pe_axis` and a decided slice axis | BOLD phantom with a 0.4 mm respiratory PE shift: FD error 0.61 -> 0.19 mm, nodding untouched. The shrinkage cannot separate independent real PE jitter from B0 jitter (audit: PE jitter FD 0.53 -> 0.07 mm). Raises when the PE or slice axis is undecidable. | Off as the value of `fd`: antsxqc grades FD (scrubbing at 0.5 mm) and series without PE metadata must not fail by default. Compute it whenever `pe_axis` is known and the slice axis is decided, and report it beside the raw FD (`fd_pe_debiased`, `pe_b0_shift_mm`, the `pe_pseudomotion_debias` record; `skipped` with a reason otherwise). Producers measure; antsxqc decides which FD it grades. |
| `debias_sagittal_nyquist` | reported slice translation and pitch | audit: real nodding jitter FD 0.94 -> 0.15 mm removed; the lag-1 gate (< -0.25) is met by independent real jitter | Off and not computed by default; explicit per-study opt-in with the caveat in the docstring. |

Sources: `MOTION_SYSTEMATIC_NOISE.md` (phantom table and notes), `METRIC_SAMPLING.md` ("Debias of the
reported trajectory"), `HANDOFF_CROSS_REPO_MOTION_AUDIT.md` item 3, tracking R7.

## 4. Work packages

syntx first; downstream after a syntx release. One package per commit, on instruction.

### WP1 One identifiability rule on the level-adjacency graph (F1, F2, F3)

- `motion_reference.level_adjacency(levels)`: a (k, k) count of frames of level a with a neighbour of
  level b. `motion_reference.offset_identifiable(levels, min_mixed=4)`: walking from the anchor, a level is
  identifiable when it has at least `min_mixed` adjacencies to levels already reached (the anchor or
  identified levels); it records which level it is identified against; unreached levels get `None`. The
  frame-count guards now inside `_group_offsets` move into this helper so both callers return the same
  verdict.
- `_group_offsets`: estimate offsets for identified levels only, columns = their indicators, frames of
  unidentified levels excluded (as now).
- `_unified`: identity when the level is identified against the previous level and every earlier link is
  identity; a new `"anchor"` link (empty chain prefix; `motion_correct_grouped` gains the entry) when the
  level is identified but an earlier level was registered; registration otherwise. After the grouped call,
  raise `RuntimeError` if a level linked by identity on the offset premise appears in
  `not_identifiable_groups` (an invariant guard once the rule is shared).
- Tests: the b0-block-then-two-shell probe above (shell 1 registration, shell 2 identified against shell 1,
  boundary jump not absorbed); random-layout agreement of `level_layout` and `_group_offsets` (0
  mismatches); the F2 chain case.

### WP2 Design validation (F4, F5, F6, F9, F10, F12)

- `Design.__post_init__`: levels are 0..k-1 with every label present; column row counts equal T; `pe_axis`
  is a non-zero 3-vector stored unit-normalised; `pe_axis` is a field of every design (`none` included).
- `_design_tag_control` re-indexes (control as anchor when present, else the first type present; the
  anchor recorded in `detail`).
- `_design_shells`: `bvecs` must be exactly (T, 3); eddy rows = unit-normalised g on diffusion-weighted
  rows, zero elsewhere.
- `_unified` forwards `encoding_covariates` (with the preset's `window_size`) for multi-level `custom`
  designs as it does for `shells`.
- Tests: label + m0 and deltam + m0 designs build with levels {0, 1}; (3, T) bvecs raise; b0 rows of the
  eddy covariates are zero for a `[1, 0, 0]` b0 vector; custom multi-level designs pass their encoding to
  the grouped call.

### WP3 Axis inference hook and slice-axis plumbing (F7, F8)

- `acquisition_axes.phase_encode_inference(*, levels=None, slice_axis=None, sidecar=None)` returns a
  callable `image -> PhaseEncode` (exact `voxel_axis`, `sign=None`, `physical_vector` = unit voxel column,
  `source="inferred"`, evidence = the scores and the slice-axis source). `infer_phase_encode_axis` stays
  and is implemented through it. `resolve_phase_encode` validates a hook-returned `PhaseEncode`
  (`voxel_axis < ndim`) and attaches its `skipped` list.
- `sidecar` kwarg threaded through `motion_correction` to `separate_bold_pe_pseudomotion` and through
  `motion_correct_grouped` to `infer_slice_axis`; `motion_correct_design` passes `sidecar` and
  `slice_axis` through unchanged.
- Tests: oblique header + inferred axis resolves to the ghost's voxel axis with the ghost ratio in the
  evidence; a debias call with a `SliceTiming` sidecar records `slice_axis.source == "slice_timing"`.

### WP3b PE pseudomotion as a side-channel report (section 3)

- With `pe_axis` given and `debias_pe_pseudomotion=False`, `motion_correction` and
  `motion_correct_grouped` run the separation into `fd_pe_debiased`, `pe_b0_shift_mm` and
  `pe_pseudomotion_debias` without touching `fd` or `motion_parameters`; an undecidable slice axis records
  `{"method": "skipped", "reason": ...}` and does not raise. `debias_pe_pseudomotion=True` keeps today's
  behaviour (replaces `fd`, keeps `fd_raw`).
- `motion_correct_design` forwards `pe_axis` from the `Design`; antsxfunctional passes the resolved PE
  vector whenever it has one, not only with the switch on.
- Tests: a default call with `pe_axis` leaves `fd` identical to a call without it and populates
  `fd_pe_debiased`; an undecidable slice axis yields the `skipped` record.

### WP4 Reference iteration memory (F11)

`reference_predictor` may return a single 3-D image, passed as `reference=` directly;
`_corrected_level_mean` returns the mean image.

### WP5 Documentation and release

`MOTION_DESIGN.md` (adjacency rule, `"anchor"` link, design validation, inference factory, the
side-channel FD); `tracking.md` R9 entry drafted; version 6.24.0 after the full suite.

### WP6 Downstream (after syntx 6.24.0)

- antsxdwi `motion.py`: pin `syntx>=6.24.0`; docstring fixes (F13); pass the unit's sidecar as `sidecar=`.
  Pipeline: when the sidecar has no `PhaseEncodingDirection` and inference is allowed, use
  `phase_encode_inference(levels=shell_groups(bvals), sidecar=...)` as one candidate beside the
  T1-anchored estimator in `antsxdwi.pe`, both recorded.
- antsxfunctional `perfusion.pipeline.syntx_motion_correction`: forward `json_sidecar` as `sidecar=`
  and the resolved PE vector always; record `design["links"]` and the anchor level in provenance.
- antsxslowflow `motion/api.py`: no code change expected (`offset_mm_rad_by_group` keeps its shape);
  pin bump.
- Each repo: targeted tests, `antsxqc.lint`, tracking entry drafted.

## 5. Verification

1. syntx: `python -m antsxqc.lint src/syntx` (0 findings); pytest on `test_motion_design.py`,
   `test_motion_reference.py`, `test_acquisition_axes.py`, `test_phase_encode.py`,
   `test_phase_encode_guard.py`, `test_motion_dwi_preset.py`; the full suite before the release.
2. Phantom gate (5 seeds) from `MOTION_DESIGN.md` for ASL, slow flow and 2-shell DWI with interspersed
   b0s (must not regress), plus a new b0-block-first 2-shell phantom (must beat the current unified result;
   numbers reported against the dispatch engine). HTML report under `reports/`.
3. Real FPA sub-BLAST079 DWI, ASL and BOLD: FD and offsets within the same-engine noise floor for
   interspersed layouts.
4. Downstream: antsxdwi `tests/test_motion.py`; antsxfunctional
   `tests/test_asl_tag_control_motion_debias.py` and the perfusion tests; antsxslowflow
   `tests/test_motion_slowflow_phantom.py`.
