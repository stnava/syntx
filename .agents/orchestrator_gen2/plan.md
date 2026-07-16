# Execution Plan: Syntx 2D Parity and Deep Feature Triggering

## Milestones

| # | Milestone Name | Scope | Dependencies | Status |
|---|----------------|-------|--------------|--------|
| 1 | Exploration & Diagnostics | Explore codebase, analyze existing 2D parameters/DICE scores, check existing unit tests. | None | PLANNED |
| 2 | Baseline 2D Parity | Tune default parameters in `syntx` to match/exceed `ants.registration` DICE on 2D phantoms within 1%. | M1 | PLANNED |
| 3 | Deep Feature Degeneracy Trigger | Implement a dynamic triggering/fallback mechanism in `syntx` for ResNet-10/VGG19 metrics. | M2 | PLANNED |
| 4 | Visual Comparison Report | Generate `docs/parity_report.html` containing required visual maps and registration metrics. | M3 | PLANNED |
| 5 | Test Verification & Audit | Run all 78 unit tests and run the Forensic Auditor to ensure all constraints are met. | M4 | PLANNED |

## Verification Plan

### Milestone 1 (Exploration)
- Verify baseline metrics for `ants.registration` on `r16`, `r27`, `r64`.
- Enumerate all 78 unit tests and verify they run successfully in the current workspace.

### Milestone 2 (Baseline 2D Parity)
- Verify `syntx` LNCC/Mattes-MI registrations achieve mean DICE score parity (within 1%) with `ants.registration`.
- Record metrics for each phantom.

### Milestone 3 (Triggering Mechanism)
- Run registrations using the triggering mechanism.
- Confirm it deactivates deep feature metrics at degenerate resolutions (coarser stages) and falls back to intensity metrics.
- Confirm this improves DICE / reduces folding compared to always-on baseline.

### Milestone 4 (Report Generation)
- Verify `docs/parity_report.html` exists and displays:
  - Edge and/or region overlap
  - Deformed grids
  - Jacobian determinant maps
  - Warped images side-by-side next to target/fixed images
  - Detailed summary tables of DICE scores, runtimes, and folding rates.

### Milestone 5 (Test Verification & Audit)
- Verify all 78 unit tests pass.
- Run Forensic Auditor to confirm clean status (no cheating, single-interpolation compliance, VGG 3D LNCC Layer 4 compliance).
