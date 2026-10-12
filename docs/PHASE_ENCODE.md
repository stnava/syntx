# Phase-encode direction in syntx (`syntx.phase_encode`)

Single place where a phase-encode (PE) direction is parsed, resolved against an image header, compared and turned into a
`restrict_transformation`. Introduced in 6.4.0 after the 2026-10-11 cross-repo audit (silent `(0, 1, 0)` / `"j"` defaults,
voxel axes treated as physical axes, DICOM `ROW`/`COL` written under the BIDS key, entity/sidecar conflicts never checked).
Policy: no silent fallback, no default axis, failures raise named errors, opting out is explicit and recorded.

## Conventions
* Physical space is ITK/ANTs **LPS**: +x Left, +y Posterior, +z Superior. `image.direction` columns are the voxel axes i, j, k in LPS.
* BIDS `i j k` with optional `-`: encoding along increasing (`+`) or decreasing (`-`) voxel index. Physical vector = `sign * direction[:, axis]`.
* Anatomical names mean "acquired from X toward Y": `AP` = +y, `PA` = -y, `LR` = -x, `RL` = +x, `SI` = -z, `IS` = +z. They are mapped back to a voxel axis through the header and give the same `PhaseEncode` as the equivalent BIDS label (AP on a RAS NIfTI is `j-`).
* A leading `dir-` is stripped. Labels are matched as whole strings, never by substring (`map`, `path`, `para`, `space` do not match).
* Rejected on purpose: DICOM `ROW`/`COL` (not BIDS, no sign), `x`/`y`/`z` (an axis with no direction; BIDS has no such value), empty, `None`. Uppercase `I` is not the BIDS `i`.
* `dir-UNK` in a file name means "no statement". A name without a `dir-` entity is no statement. Only the `dir-` entity of the **real file's basename** is read; never read a staged `DTILR`/`DTIRL` name. The entity value is alphanumeric only (BIDS), so `dir-LR_dwi`, `dir-LR-dwi` (PPMI style) and `dir-LR.nii.gz` all read `LR`; `dir-run-01` reads `run` and `dir-none` reads `none`, which are not phase-encode words and raise `PhaseEncodeVocabularyError` in `resolve_phase_encode` (v6.4.1).
* Restriction weights: one-hot in physical (x, y, z) order, length = spatial dimensions (a 4-D image gives 3, 2-D gives 2), `1` = free axis, the layout `syntx.syn(restrict_transformation=...)` consumes.

## API
```python
from syntx.phase_encode import (PhaseEncode, resolve_phase_encode, restrict_for_image, restrict_with_provenance,
    restrict_transformation, relate_phase_encodes, pe_to_dict, pe_from_dict, parse_pe_label, pe_entity_from_path,
    PE_UNRESTRICTED)            # also importable from syntx and syntx.spatial

pe = resolve_phase_encode(b0, sidecar="sub-01_dwi.json", entity_path=real_path)   # or label=..., anatomical=...
w, prov = restrict_with_provenance(b0, sidecar=...)     # w = one-hot tuple; prov = JSON-safe record of what was applied
w = restrict_for_image(b0, pe=pe)                       # from a carried record (PhaseEncode or its dict)
w = restrict_for_image(b0, pe_policy="unrestricted")    # explicit opt-out: returns None, prov says so
relate_phase_encodes(pe_a, pe_b)                        # "opposed" | "same" | raises PhaseEncodeRelationError
```
`PhaseEncode` (frozen): `voxel_axis`, `sign` (+1/-1; `None` only for an inferred axis), `physical_vector` (unit LPS), `physical_axis`, `alignment` (|cos| with the nearest axis), `label` (canonical `i,i-,j,j-,k,k-`), `anatomical` (`AP,PA,LR,RL,SI,IS` when aligned), `source` (`sidecar|label|entity|inferred|explicit`), `source_detail`, `evidence` (every source examined, what each mapped to, direction matrix).

Rules of `resolve_phase_encode`: precedence of the decisive source `sidecar > label > explicit > entity`; **all stated sources must agree** after the header mapping (same voxel axis, sign, physical axis); a missing sidecar file or bad JSON raises; a sidecar without the key is "no statement". Obliquity: below `oblique_tol` (default 0.95, about 18 degrees) `on_oblique='raise'` (default) or `'warn'`. `allow_inference=True` only calls a caller-supplied `inference(image) -> physical axis | PhaseEncode`; syntx contains no estimator (antsxdwi keeps `infer_pe_axis` and passes it in). An inferred axis is unsigned and cannot be related.

## Errors (all `ValueError` subclasses of `PhaseEncodeError`)
| Error | Raised when |
|---|---|
| `PhaseEncodeUndecidable` | no source states the direction and no inference; sidecar without the key and nothing else |
| `PhaseEncodeConflict` | two sources disagree after the header mapping; `pe_policy='unrestricted'` given together with sources |
| `PhaseEncodeVocabularyError` | `ROW`, `COL`, `x/y/z`, `''`, `None`, any unknown word (also a bad value inside a sidecar) |
| `PhaseEncodeObliqueError` | alignment below tolerance with `on_oblique='raise'` |
| `PhaseEncodeRelationError` | `relate_phase_encodes` on orthogonal, ambiguous or unsigned encodes |
| `PhaseEncodeError` | base: missing/unreadable sidecar, axis absent in a 2-D image, malformed record |

## Changes inside syntx 6.4.0
* `syntx.spatial.restriction_from_orientation`: same signature, now a shim over the module. A 4-D image returns 3 weights (was 4). Sidecar/BIDS paths raise the named errors (a missing sidecar file now says so instead of "no PhaseEncodingDirection found"). Obliquity still warns, at `obliquity_warn_threshold`; the warning text changed ("oblique"). `anatomical_axis=` remains the axis-only legacy path (accepts `x/y/z`, no direction).
* `syntx.dewarp.syn_only_reference_to_anatomy_rigid` (alias `syn_only_motion_reference_to_t1_rigid`) and `dewarp_to_anatomical`: `restrict_transformation` is **required** (no default). Accepts a `PhaseEncode` / dict, a weight tuple, a `.json` sidecar path, a label, or `'unrestricted'`; `None` raises `TypeError`. A label is now resolved through the header (previously `AP` and `x/y/z` skipped it); `x/y/z` now raise.
* `syntx.contract.ModalityUnit`: new `phase_encode: list[dict]` (`pe_to_dict` records); `phase_encoding` / `phase_encode` must be empty or the length of `input_paths` (ValueError otherwise).
* Guard: `tests/test_phase_encode_guard.py` forbids new hard-coded `(0, 1, 0)`, `[0, 1, 0]`, `pe_axis=1`, bare `'j'` defaults and `anatomical_axis='..'` defaults in `src/syntx` outside `phase_encode.py` (allow-list with a reason per entry).

## Migration guide for Pass B (file:line as of 2026-10-11)
* **antsxdwi**: replace `dewarp.py:188-273` (`phase_encode_vector`, `restriction_from_phase_encode`, `check_opposed_phase_encoding`) and `series_pair.py:41-50` (`_relation_from_labels`) with `resolve_phase_encode` / `relate_phase_encodes` (keep thin re-exports one release); `dewarp.py:279,532` `pe_axis=1` defaults -> required argument; `io.py:46-53` `DwiSidecar.pe_axis/pe_sign` delete or route through the resolver; `scripts/run_dwi_standalone.py:121-168` build a positional list with an entry per input (the B-only label at :153-156 is now caught by the `ModalityUnit` length check); `pipeline.py:409-436, 624-646` resolve with `inference=infer_pe_axis`. Callers of `dewarp_to_anatomical` already pass `restrict_transformation` (`dewarp.py:356,436,626`, `anatomy_refine.py:170`, scripts `experiment_hybrid_phantom.py:206`, `dewarp_playground.py:195,376`); no change required. Tests constructing units with matching lists are unaffected.
* **antsxslowflow**: delete `spaces/api.py:42-78 derive_pe_restriction`; replace `api.py:496` with `restrict_with_provenance(motion_reference_isotropic, sidecar=<b0 sidecar>)` or an explicit `pe_policy="unrestricted"` parameter (see the slowflow note); replace the constants at `api.py:657, 694, 727` with the applied tuple / `prov`; rewrite `tests/test_canonical_phase_and_tx_contracts.py:163-190`. Callers of `syn_only_motion_reference_to_t1_rigid` now MUST pass `restrict_transformation=`: `tests/test_canonical_contract_draft.py:125`, `tests/_simulated_contracts.py:298`, `tests/test_syntx_fallback_narrowing.py:86`, `experiments/simulated_canonical_transform_path.py:323`, `experiments/simulated_transform_contract.py:197`.
* **antsxfunctional**: `registration.py:249-276` replace the cascade (j default, AP fallbacks, `(0.0, 1.0, 0.0)`) with `restrict_with_provenance(fixed, pe=..., sidecar=..., label=..., pe_policy=...)`; add `phase_encode` / `pe_policy` parameters (default require); thread them through `rsfmri/pipeline.py:90,166-169,245-253` and `perfusion/pipeline.py:213`.
* **antsxmm**: `selectors.py:304-335 read_phase_direction` and `modalities/dispatch.py:440-505` build one `PhaseEncode` per input from the real file (entity + sidecar), fill `ModalityUnit.phase_encode` and `phase_encoding` (canonical labels) at full length (no clearing; a missing one raises naming the path); pass PE to the rsfMRI (`dispatch.py:1090-1160`), perfusion (`:1008-1085`) and slowflow (`:681-715`) bridges; `staging.py:36-52` keeps the legacy names but nothing reads them.
* **antsxbids**: `json_sidecar.py:25 make_json` must never write `ROW`/`COL`/`None` under `PhaseEncodingDirection`; `link/profiles/generic.py:83-87,100-104` use the strict entity pattern (see `pe_entity_from_path`).

## antsxslowflow: what its restriction is actually applied to (user note 2026-10-11)
In slowflow the phase-contrast images are in exactly the same space as the b0 (no distortion between them), so no b0 <-> phase-contrast mapping may be restricted by a PE axis. In `canonical_space` (`spaces/api.py`) `restrict_transformation` is used at exactly one registration: `motion_reference_isotropic_brain` (the b0/magnitude-derived motion reference, an EPI) **to the rigidly aligned T1** (`t1_r`), SyNOnly, at `api.py:533` (syntx.syn) and `:552` (plain ANTs fallback). Phase is never registered; it follows the per-frame chain native -> motion reference (same grid) -> T1 rigid -> that SyN field. The motion-reference -> anatomy step is the one place the EPI geometric distortion along the b0's own PE axis matters, so that is where a restriction belongs.
* Policy for that call: PE **resolved from the b0/magnitude sidecar** (the sidecar of the EPI the motion reference was built from, not a guess), `restrict_with_provenance(motion_reference_isotropic, sidecar=...)`, header-aware. If no b0 PE is known, either fail with `PhaseEncodeUndecidable` or let the caller pass the explicit `pe_policy="unrestricted"` (recorded); never a default y.
* Policy for any b0 <-> phase-contrast mapping (none exists today): do not restrict; if one is ever added, it is explicit `pe_policy="unrestricted"` (or identity). The central API supports both (`restrict_for_image(..., pe_policy="unrestricted")` returns `None` and `restrict_with_provenance` records it).
* Note for Pass B: the bridge passes `_bids_sidecar_json(phase_path)` (`dispatch.py:681-715`), the sidecar of the **phase** file; the PE must come from the b0/magnitude sidecar (or the phase sidecar only if it is verified to carry the same PE). Record `prov` in the `TransformStep` payloads.
