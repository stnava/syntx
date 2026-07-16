# Project: syntx Deep Feature Registration Benchmarking & Quality Characterization

## Architecture
- **Feature Extractor Layer**: PyTorch-based networks (VGG19, DINOv2, ResNet-10, Swin UNETR) in `src/syntx/features.py` that extract deep multi-scale features from 2D/3D images.
- **SyN JAX Backend**: JAX-based SyN optimization loop in `src/syntx/syn_jax.py` or PyTorch-based loop in `src/syntx/syn.py` that performs registration using deep feature space similarity.
- **Grader Component**: `antspyt1w.resnet_grader` evaluates input image scan quality.
- **Evaluation/Benchmarking Track**: Executed via a dedicated benchmarking suite that profiles 2D and native-resolution 3D registrations, calculates DICE overlap on DKT segmentations, and computes Jacobian-based folding rates.
- **Visualization and Dashboard**: Generating warped grids, edge maps, Jacobian determinants, and side-by-side comparisons, formatted into an HTML dashboard.

## Code Layout
- `src/syntx/features.py`: Deep feature network extractors and similarity losses.
- `src/syntx/syn_jax.py` & `src/syntx/syn.py`: JAX and PyTorch SyN registration loops.
- `examples/`: Code templates, sweeps, and utilities (e.g. `examples/vgg_sweep_2d.py`, `examples/vgg_sweep_3d.py`).
- `docs/benchmarks.html`: Output dashboard for benchmark reporting.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Verification | Locate datasets, verify feature networks and grader functionality, and assess hardware backend. | none | DONE |
| 2 | 2D Comparative Benchmarks | Run registrations on `r16`, `r27`, `r64` using VGG19, DINOv2, ResNet-10, and standard SyN. Analyze landscape. | M1 | IN_PROGRESS |
| 3 | 3D Native Resolution Benchmarks | Grade low-quality scans. Run 3D registrations at native resolution and evaluate DKT labels overlap (DICE). | M2 | IN_PROGRESS |
| 4 | Reporting & Visual Dashboards | Plot correlations, construct metric summary tables, compile visual maps, and output `docs/benchmarks.html`. | M3 | IN_PROGRESS |
| 5 | Validation & Forensic Audit | Perform code verification, enforce single interpolation policy, and execute the forensic audit. | M4 | PLANNED |

## Interface Contracts
- **`antspyt1w.resnet_grader`**: Used to grade image scan quality.
- **Cortical/DKT DICE Evaluation**: `ants.label_overlap_measures` or custom label DICE computation on target template's DKT label map.
- **Single Interpolation Policy**: Under NO circumstances should pre-warped inputs be passed to optimization steps. Multi-transform composition must occur in a single execution step.
- **VGG 3D Mode Requirement**: Only VGG 3D LNCC with Layer 4 (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) is permitted for accurate registrations.
