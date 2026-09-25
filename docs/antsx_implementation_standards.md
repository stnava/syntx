# ANTsX implementation standards

Applies to every antsxmm, antsxdwi, antsxslowflow, and syntx module (and any future
antsx-family repository). Written after finding concrete violations of all three rules
below in real, committed code across these repos during review — this is not a
hypothetical checklist.

A copy of this file lives in each repo's `docs/`. It is read the same way reviewer notes
are read: before starting work, and before declaring anything done. Any agent working in
these repos treats a violation of these rules the same as a failing test.

**syntx-specific note on rule 2:** in antsxmm/antsxdwi/antsxslowflow, rule 2 means
"call `syntx.*`, not `ants.registration`/`ants.motion_correction`." Inside syntx itself
there is no more-fundamental layer to defer to — syntx *is* the thing those repos are told
to route through. Here rule 2 instead means: syntx's own torch-native primitives
(`syn`, `robust_affine`'s default backend, `build_template`'s deformable stage, `auto_reg`)
should not silently regress to legacy `ants.registration`/`ants.motion_correction` calls as
their real working path, and any place they still do must be an explicit, reasoned,
narrow exception (a documented legacy mode, a benchmark baseline, or a genuine
not-yet-migrated primitive) — not an unmarked default.

---

## 1. Respect ANTsPy's own conventions; don't fight physical space

**Prefer ANTsImage-native operations over drop-to-numpy-and-rebuild.** `ants.slice_image`,
`ants.crop_indices`, `ants.resample_image_to_target`, `ants.copy_image_info`, etc. carry
origin/spacing/direction through automatically. Reconstructing an ANTsImage from a bare
numpy array (`ants.from_numpy(arr, origin=..., spacing=..., direction=...)`) is sometimes
unavoidable (this codebase does real numeric work in torch/numpy), but every such
reconstruction is a place a geometry bug can be introduced silently — a copy-pasted
`origin=(0,0,0)` default, a transposed spacing tuple, a stale direction matrix from before
a crop. When rebuilding from an array that came from an existing ANTsImage, use
`ants.copy_image_info(source_image, new_image)` to copy geometry from the actual source,
not by re-typing origin/spacing/direction values by hand.

**Don't call `ants.reorient_image2` (or any `ants.reorient_image*`) superfluously.**
Reorientation resamples the image — it is not free, and it is not lossless. It is legitimate
exactly once, at a documented boundary, when a specific downstream consumer genuinely
requires a canonical orientation (e.g. a viewer or file format that assumes RAI/LPI). It is
**not** a defensive habit to sprinkle at the start of every function "just in case" the
input isn't canonically oriented — code that correctly uses the image's own direction
matrix does not need its inputs pre-reoriented at all. Every `reorient_image*` call site
must have a one-line comment naming the specific downstream consumer that requires it. If
you can't name one, delete the call.

**Displaying in physical space means respecting spacing (aspect ratio) and direction
(radiological convention), not just plotting the raw array.** A montage or slice view built
directly from `image.numpy()` without accounting for `image.spacing` will silently distort
anisotropic-voxel data, and without accounting for `image.direction` will silently mirror
left/right or flip anterior/posterior. Figure/report code must read spacing and direction
from the image and apply them, never assume isotropic identity-direction data.

## 2. Registration and motion quantification: syntx's own torch-native path is the default

`ants.registration(...)` and `ants.motion_correction(...)` are the old path. For syntx's
consumers (antsxmm, antsxdwi, antsxslowflow), the standard path is `syntx.syn`,
`syntx.robust_affine`, `syntx.motion_correction`, `syntx.auto_reg`, and
`syntx.template.build_template`.

Known current findings inside syntx itself (found during this review, not fixed here —
recorded so they aren't silently rediscovered or reintroduced):
- `src/syntx/motion.py` (`motion_correction`'s inner per-frame registration loop) and
  `src/syntx/template.py` (`build_template`, whose docstring says "Direct port of
  ants.build_template") call `ants.registration` unconditionally as their real working
  path — they are C++-ANTs ports with syntx-native QC/metrics wrapped around them, not
  torch-native reimplementations. This is an accepted, documented architectural gap (see
  `ALLOWED_REGISTRATION_EXCEPTIONS` in `tests/test_implementation_standards.py`), not
  something to silently fix in passing — migrating either to a torch-native deformable
  solver is its own project with its own parity/gate requirements, matching the rigor
  applied to `antsxdwi`'s dewarp module.
- `src/syntx/robust_affine.py`'s `mode='ants'`/`'ants_fast'` branch calls
  `ants.registration(type_of_transform='Affine')` as an explicitly named, non-default
  legacy mode (see the function's own docstring comparing it against the default
  torch path) — a deliberate, labeled alternative, not a silent fallback.
- `src/syntx/benchmark/high_level.py` and `src/syntx/benchmark/evaluate.py` call
  `ants.registration` only when the benchmark harness is explicitly asked to run an
  `"ants"`/`"ants_syn"` baseline model for comparison against syntx — the entire point of
  that code path is to measure syntx against legacy ANTs, so calling it there is correct,
  not a violation.

A plain-ANTs call remains legitimate only as an explicit, narrow, labeled alternative (a
named legacy mode, a benchmark baseline, or a genuine not-yet-migrated primitive with a
recorded justification) — never as the unmarked default path, and never silently dropping
a feature (axis restriction, provenance) that the syntx-native call would have provided.

For the downstream repos (antsxmm/antsxdwi/antsxslowflow), known current violations are
tracked in each of those repos' own copy of this document.

## 3. Use antstorch's versions of standard preprocessing steps

Where antstorch provides a torch-native equivalent, use it instead of the antspynet/ANTs
version: `antstorch.brain_extraction`, `antstorch.n4_bias_field_correction`,
`antstorch.denoise_image`, and `antstorch.preprocess_brain_image`.

Known current finding: `src/syntx/landmarks/preprocess.py`'s N4 and denoise steps already
try `antstorch` first and log a warning before falling back to `ants.n4_bias_field_correction`
/ `ants.denoise_image` on any exception (including ones unrelated to antstorch's actual
availability, e.g. a shape mismatch) — this is the "junk fallback" pattern flagged and
removed elsewhere (antsxslowflow, antsxdwi): a bare `except Exception` swallows real bugs
and silently degrades to the legacy path instead of surfacing them. Not fixed here (out of
the scope of this pass); recorded as a documented, narrow exception so it is not
rediscovered as new, and so a future pass tightens it to the specific antstorch-unavailable
exception type, matching the antsxslowflow/antsxdwi fix.

---

## Enforcement

A prose rule that nobody checks is not a rule. Each repo has a grep-based test analogous to
antsxdwi's `tests/test_implementation_standards.py`: `ALLOWED_*_EXCEPTIONS` dicts requiring
a documented reason for every matched call site, failing on anything unlisted. See this
repo's `tests/test_implementation_standards.py`.

A reviewer (human or agent) finding a new violation during any review adds it to this
document's "known current findings" list before it can be considered someone else's problem
to track — an unrecorded violation gets silently reintroduced.
