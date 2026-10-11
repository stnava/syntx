# syntx 6.4.0

New central phase-encode module `syntx.phase_encode` (design and migration guide: `docs/PHASE_ENCODE.md`). Correctness over backward compatibility: the silent defaults are removed.

## Added
* `PhaseEncode` (frozen dataclass), `resolve_phase_encode`, `restrict_for_image`, `restrict_with_provenance`, `restrict_transformation`, `relate_phase_encodes`, `pe_to_dict` / `pe_from_dict`, `parse_pe_label`, `pe_entity_from_path`, `PE_UNRESTRICTED`; errors `PhaseEncodeError`, `PhaseEncodeUndecidable`, `PhaseEncodeConflict`, `PhaseEncodeVocabularyError`, `PhaseEncodeObliqueError`, `PhaseEncodeRelationError`. Exported from `syntx`, `syntx.spatial`.
* Header-aware (flipped, permuted, oblique, 2-D, 4-D), signed, LPS-documented; sources must agree after header mapping; DICOM `ROW`/`COL` and bare `x/y/z` are vocabulary errors; labels are never matched by substring.
* `ModalityUnit.phase_encode` (structured records) and length validation of `phase_encoding` / `phase_encode`.
* Tests: `tests/test_phase_encode.py` (grids x labels, oblique, vocabulary, conflicts, undecidable, relation, round trip, shim parity, random flip/permutation known-shift phantom, a restricted SyN on a flipped permuted grid, removed defaults), `tests/test_phase_encode_guard.py` (no new hard-coded PE defaults in `src/syntx`).

## Changed (breaking on purpose)
* `syn_only_reference_to_anatomy_rigid`, `syn_only_motion_reference_to_t1_rigid`, `dewarp_to_anatomical`: `restrict_transformation` is required (the silent `(0, 1, 0)` default is gone); `None` raises; `'unrestricted'` is the explicit opt-out; string labels go through the header; `x/y/z` raise.
* `restriction_from_orientation` is a shim over the module: 4-D images give 3 weights (was 4); sidecar/BIDS errors are the named subclasses of ValueError; a missing sidecar file is reported as missing; the obliquity warning text changed.
* `ModalityUnit` raises ValueError for a partial `phase_encoding` / `phase_encode` list.

## Not in this release
No migration of the other repos (antsxdwi, antsxslowflow, antsxfunctional, antsxmm, antsxbids): see the guide in `docs/PHASE_ENCODE.md`. The data-driven axis estimator stays in antsxdwi (`infer_pe_axis`) and is passed in as `inference=`.
