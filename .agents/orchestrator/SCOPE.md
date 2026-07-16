# Scope: syntx Deep Feature Benchmarking and 3D Parity

## Architecture
- **Feature Extractor Layer**: PyTorch-based networks (VGG19, DINOv2, ResNet-10, Swin UNETR) in `src/syntx/features.py` that extract deep multi-scale features from 2D/3D images.
- **SyN Backends**: PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`) registration optimization loops.
- **Evaluation suite**: Python scripts to run sweeps, evaluate metrics, and plot correlations.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Verification | Verify 2D phantoms and cached 3D scans, inspect backends and defaults. | none | DONE |
| 2 | 2D Systematic Sweep | Run 2D registrations comparing raw intensity LNCC vs ResNet-10, VGG19, DINOv2. Measure DICE, folding, speed. | M1 | PLANNED |
| 3 | 3D Parity Achievement | Establish 3D parameter configurations in PyTorch/JAX to match/exceed ANTs baseline DICE under equivalent LNCC/Mattes-MI. | M1 | PLANNED |
| 4 | 3D Deep Features Impact | Evaluate 3D VGG LNCC Layer 4, DINOv2, ResNet-10 vs standard 3D intensity baselines. | M3 | PLANNED |
| 5 | Comprehensive Reporting | Compile HTML report at `docs/deep_feature_impact_report.html` with required visualizations (edges, grids, Jacobians, side-by-side). | M2, M4 | PLANNED |
| 6 | Validation & Forensic Audit | Run all unit tests, execute forensic integrity auditor to verify correctness. | M5 | PLANNED |

## Interface Contracts
- **Registration function**: `syntx.syn(...)` or `syntx.registration(...)`.
- **Single Interpolation Policy**: No intermediate pre-warping prior to optimization. All transforms composed and applied in a single step.
- **VGG 3D Mode Requirement**: Only VGG 3D LNCC with Layer 4 (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) is acceptable for accurate registration.
