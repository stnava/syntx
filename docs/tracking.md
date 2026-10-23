# Tracking Summary
current phase: v6.17.0 release (Registration QC thresholds and verdicts delegated to antsxqc; qc modules renamed to registration_metrics and metric_sections; convergence-driven registration engine)
next WP: Multi-modal deformable registration benchmarks and GPU memory optimization
blockers: none
active work: none (v6.17.0 released 2026-10-22)
device: Apple Silicon MPS / CPU (macOS arm64)
last updated: 2026-10-22

---

## R1 (Delegate Registration QC Verdicts to antsxqc, v6.17.0, 2026-10-22)

Tag `v6.17.0`.

- **What changed.**
  - Renamed `src/syntx/qc.py` -> `src/syntx/registration_metrics.py` and `src/syntx/viz/qc_sections.py` -> `src/syntx/viz/metric_sections.py` (`R3-name` compliance).
  - Delegated registration hard validity limits (`min_jacobian`, `folding_pct`, `dice_overlap`, `inverse_identity_error`) and status grading to `antsxqc.hard_limit` and `antsxqc.grade`.
  - Standardized `ModalityReport` (`src/syntx/viz/modality_report.py`) on the canonical 4-status vocabulary (`pass | warn | fail | not_assessable`).
  - Added `tests/test_antsxqc_lint.py` enforcing `antsxqc.lint.assert_clean` (`26 passed`).
