# `syntx` Contracts

## 1. Shared Modality Plugin Contract (`syntx.contract`)
- `ModalityUnit`, `SessionContext`, `ComputeContext`, `ProvenanceEntry`, and `ModalityResult` are `@dataclass(slots=True)` types defining the interface between `antsxmm` and modality producers (`antsxstructural`, `antsxdwi`, `antsxfunctional`, `antsxslowflow`).
- Stage execution statuses use `"success" | "failed" | "skipped"` exclusively (`syntx.contract.Status`), distinct from `antsxqc` quality verdicts (`"pass" | "warn" | "fail" | "not_assessable"`).

## 2. Registration & Spatial Geometry Contract
- All registrations (`syntx.robust_affine`, `syntx.syn`, `syntx.tvf`, `syntx.syngs`, `syntx.greedy`) preserve physical LPS coordinates (`origin`, `spacing`, `direction`) and enforce the single-interpolation policy via `syntx.transforms.apply_transforms`.
- Coordinate conversions (`vox_to_physical`, `physical_to_vox`, `ortho_view_spec`, `slice_from_spec`) live in `syntx.spatial` (and are re-exported in `syntx.landmarks.spatial`).

## 3. Registration Metrics & Visualization Bridge
- `syntx.registration_metrics.evaluate_registration_qc` computes numeric registration metrics (`folding_pct`, `min_jacobian`, `harmonic_energy`, `bending_energy`, `inverse_consistency_error_max_mm`, `target_dice_symmetric`) and delegates all grading and status aggregation to `antsxqc.grade()`, `antsxqc.hard_limit()`, and `antsxqc.worst_status()`.
- Failed warp loads or missing warps produce `NaN` metrics and `"not_assessable"` verdicts — never synthetic `min_jacobian=1.0, folding_pct=0.0` pass values.
