# syntx 6.6.0

Shared probe-displacement helper `syntx.pose` for rigid and affine registration pose verification, anti-duplication guards, and complete deduplication of local implementations across tests and benchmarks.

## Added
* `syntx.pose` module exporting:
  - `probe_points(fixed, kind='default')`: Generates standardized physical probe points around the fixed center of mass (default 9 probes: 8 corners at $\pm 15, \pm 15, \pm 5$ mm plus center; also supports `kind='slab_thick'` with $z$ lever arm up to 1.5 slices from slab boundaries).
  - `pose_error(fixed, tx_a, tx_b, probes=None) -> float`: Calculates the maximum displacement error $\max_i \|\text{disp}_A(q_i) - \text{disp}_B(q_i)\|$ in mm. Supports ANTsTransform objects, transform file paths, and $4 \times 4$ / $3 \times 4$ / $3 \times 3$ affine matrices. Follows standard ITK fixed $\rightarrow$ moving coordinate conventions.
  - `pose_error_report(fixed, tx_a, tx_b, probes=None) -> PoseErrorReport`: Returns detailed per-probe displacement statistics (`max_error_mm`, `mean_error_mm`, `z_mean_error_mm`, `probe_errors_mm`, `displacements_a`, `displacements_b`, `worst_probe`).
  - `probe_displacements(fixed, tx, probes=None) -> np.ndarray`: Returns displacement vectors $\text{disp}(q_i) = T(q_i) - q_i$ in mm across probe points.
  - `PoseErrorReport`: Dictionary subclass with properties and human-readable formatting.
* Exported symbols in `syntx` top-level namespace: `pose`, `probe_points`, `pose_error`, `pose_error_report`, `probe_displacements`, `PoseErrorReport`.
* Unit tests in `tests/test_pose.py`: Tests identical transforms ($0.0$ mm), known $3.0$ mm translation along all axes, known $90^\circ$ rotation about center, ITK direction convention, report structure, file path loading, list unwrapping, and named input error handling.
* Anti-duplication guard in `tests/test_pose_guard.py`: Scans `src/syntx`, `tests`, and `benchmarks` to forbid local re-implementations of probe corner generation, `apply_to_point` displacement loops, and local `pose_error` definitions.
* Documentation in `docs/POSE.md` covering coordinate conventions, mathematical formulas, and API examples.

## Changed
* Deduplicated local probe-displacement calculation in `tests/test_robust_affine_anisotropy.py` and `tests/test_slab_capture.py` to call `syntx.pose_error`.
* Deduplicated `benchmarks/pyramid_compare.py` and `benchmarks/slab_setting_eval.py` to route through `syntx.pose`.
* Updated phase-encode migration documentation in `docs/PHASE_ENCODE.md`.
