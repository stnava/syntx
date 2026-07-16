# Context

## Codebase Paths
- Feature Extractor: `src/syntx/features.py`
- PyTorch Registration Loop: `src/syntx/syn.py`
- JAX Registration Loop: `src/syntx/syn_jax.py`
- Transformations: `src/syntx/transform.py`
- ResNet Grader: `src/syntx/resnet.py`

## Benchmarks & Sweeps
- 2D Sweep: `examples/vgg_sweep_2d.py`
- 3D Sweep: `examples/vgg_sweep_3d.py`
- 2D Comparison Report Generator: `examples/generate_ants_2d_comparison_report.py`

## User Rules & Guardrails
- **GEMINI.md**:
  - Single Interpolation Policy: No intermediate pre-warping. Compose all transforms and apply them in a single step (e.g. `ants.apply_transforms`).
  - Cortical label maps: Mean DICE drop >= 0.01 is unacceptable.
  - VGG 2D mode LNCC is not an acceptable substitute. Only VGG 3D LNCC with Layer 4 (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) meets the intensity-based LNCC performance.
  - Reports must show spatial images (overlaps, grids, Jacobians, side-by-side warped/target).
