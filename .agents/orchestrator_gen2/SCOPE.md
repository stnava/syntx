# Scope: Syntx 2D Parity and Deep Feature Triggering

## Architecture
- **Feature Extractor Layer**: PyTorch-based networks (VGG19, DINOv2, ResNet-10) in `src/syntx/features.py`.
- **SyN Registration Loop**: PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`).
- **Trigger Mechanism**: Dynamic check in `src/syntx/syn.py` / `src/syntx/syn_jax.py` during resolution levels to deactivate deep feature metrics when degenerate (falling back to intensity-based LNCC/Mattes-MI).
- **Parity Report**: Located at `docs/parity_report.html`.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Diagnostics | Find baseline metrics, identify current 2D parameters and existing tests. | None | DONE |
| 2 | Baseline 2D Parity | Establish default parameters for `syntx` to match/exceed `ants.registration` DICE scores on 2D phantoms (`r16`, `r27`, `r64`). | M1 | DONE |
| 3 | Deep Feature Degeneracy Triggering | Implement and evaluate triggering heuristic for ResNet-10/VGG19 features at coarse resolution stages. | M2 | DONE |
| 4 | Reporting & Visual Dashboards | Generate `docs/parity_report.html` with edge overlap, deformed grids, Jacobian determinants, and side-by-side images. | M3 | DONE |
| 5 | Verification & Forensic Audit | Verify all 78 unit tests pass and run the Forensic Auditor. | M4 | DONE |

## Interface Contracts
- **Single Interpolation Policy**: Composed transforms applied directly to native-space images in a single step (e.g. `ants.apply_transforms`).
- **Trigger Heuristic**: Resolution-based or variance/gradient-based check inside the registration optimization loop to fallback to raw intensity similarity.
