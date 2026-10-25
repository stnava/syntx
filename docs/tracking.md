# Tracking Summary
current phase: v6.19.0 release (Generalized overlapping-window encoding-bias + head-motion solver across DWI, ASL, SlowFlow, and BOLD EPI)
next WP: Multi-modal deformable registration benchmarks and GPU memory optimization
blockers: none
active work: none (v6.19.0 released 2026-10-24)
device: Apple Silicon MPS / CPU (macOS arm64)
last updated: 2026-10-24

---

## R3 (Generalized Overlapping-Window Encoding-Bias & Head-Motion Solver Across DWI, ASL, SlowFlow, and BOLD, v6.19.0, 2026-10-24)

Tag `v6.19.0`.

- **What changed.**
  - Added `build_encoding_covariates`, `_project_encoding_surround_tukey_np`, and `separate_bold_pe_pseudomotion` in `src/syntx/motion_batched.py`, decomposing time-series poses over partition-of-unity Hann windows $\psi_w(t)$ with 50%-breakdown second-order surround-difference Tukey biweight IRLS on $\mathfrak{se}(3)$ principal rotation vectors and Empirical Bayes / Wiener EPI-plane noise-floor matching for BOLD respiratory $\Delta B_0(t)$ pseudomotion.
  - Added `build_bilateral_window_references` (`reference="bilateral"`) in `src/syntx/motion_reference.py`, `ASL_MOTION_PRESET` (`"asl"`) and `SLOWFLOW_MOTION_PRESET` (`"slowflow"`) in `src/syntx/motion_presets.py`, and forwarded `encoding_covariates` and `window_size` through `motion_correction` and `motion_correct_grouped` (`src/syntx/motion.py`).
  - Added `tests/test_motion_overlapping_window.py` covering `TestBuildEncodingCovariates`, `TestOverlappingWindowASL`, `TestOverlappingWindowSlowFlow`, `TestOverlappingWindowDWI`, and `TestOverlappingWindowBOLD`.

---

## R2 (Ecosystem Audit Remediation & Zero-Allowlist Consolidation, v6.18.0, 2026-10-23)

Tag `v6.18.0`.


- **What changed.**
  - Fixed `evaluate_registration_qc` in `src/syntx/registration_metrics.py` to aggregate verdicts via `antsxqc.worst_status()` (preserving `"not_assessable"` instead of promoting to `"pass"`), read the `folding_pct` limit dynamically via `antsxqc.hard_limit("folding_pct", side="high")`, and return `float("nan")` instead of fake `min_jacobian=1.0, folding_pct=0.0` when warp loading fails.
  - Fixed `preprocess_for_landmarks` in `src/syntx/landmarks/preprocess.py` to pass `ants.ANTsImage` directly to `antstorch.n4_bias_field_correction` and removed the `ants.n4_bias_field_correction` fallback.
  - Standardized execution status strings in `src/syntx/provenance.py` and `src/syntx/landmarks/optimal_transport.py` to `"success"` and deleted `.antsxqc-allow` (`antsxqc.lint` passes with 0 findings and 0 exceptions).
  - Replaced `reorient_image2("LPI")` in `src/syntx/viz/core.py` and `src/syntx/viz/figures.py` with direct direction-cosine axis permutation/flipping (`AnatomicalVisualizer._orient_lpi`).
  - Consolidated coordinate primitives from `src/syntx/landmarks/spatial.py` into `src/syntx/spatial.py`, delegated energy calculations in `src/syntx/viz/reports.py` to `syntx.deformation_metrics`, added `slots=True` to `src/syntx/contract.py` dataclasses, removed import-time `PYTORCH_MPS_HIGH_WATERMARK_RATIO` mutation from `src/syntx/__init__.py`, and deleted the legacy `src/syntx/reporting.py` shim.
  - Added `AGENTS.md`, `CLAUDE.md`, `docs/antsx_implementation_standards.md`, `docs/contracts.md`, and the 6-check `tests/test_implementation_standards.py` suite.

## R1 (Delegate Registration QC Verdicts to antsxqc, v6.17.0, 2026-10-22)

Tag `v6.17.0`.

- **What changed.**
  - Renamed `src/syntx/qc.py` -> `src/syntx/registration_metrics.py` and `src/syntx/viz/qc_sections.py` -> `src/syntx/viz/metric_sections.py` (`R3-name` compliance).
  - Delegated registration hard validity limits (`min_jacobian`, `folding_pct`, `dice_overlap`, `inverse_identity_error`) and status grading to `antsxqc.hard_limit` and `antsxqc.grade`.
  - Standardized `ModalityReport` (`src/syntx/viz/modality_report.py`) on the canonical 4-status vocabulary (`pass | warn | fail | not_assessable`).
  - Added `tests/test_antsxqc_lint.py` enforcing `antsxqc.lint.assert_clean` (`26 passed`).
