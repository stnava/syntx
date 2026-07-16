# Context: JAX Modular Feature-Space Metrics & Swin UNETR

## Environment and Codebase Context
- **Syntx Core**: PyTorch and JAX based registration library. 
- **PyTorch backend (`src/syntx/syn.py`)**: Uses PyTorch for optimization. Includes `FeatureSpaceLoss` in `src/syntx/features.py` for feature-based registration.
- **JAX backend (`src/syntx/syn_jax.py`)**: Uses JAX for optimization. Currently only supports native JAX metrics (`local_ncc_loss_nd_jax` and `mattes_mi_loss_nd_jax`).
- **Feature Extractors (`src/syntx/features.py`)**: Defines `FeatureExtractor` base and `VGG19Extractor`, `DINOv2Extractor`, `ResNet10Extractor` subclasses.
- **Target task**:
  - Implement JAX/Flax support for PyTorch feature extractors in `src/syntx/syn_jax.py` using DLPack tensor sharing.
  - Implement Swin UNETR 3D encoder in `src/syntx/features.py`.
  - Create comparative evaluation script `examples/evaluate_all_metrics.py` utilizing a real brain template (T1w) and a real T2w-like scan.
  - Keep test coverage >= 90% and ensure all unit tests pass.

## Reference Paths
- Fixed Template: `/Users/stnava/.antspyt1w/T_template0.nii.gz`
- Moving T2w Volume: `/Users/stnava/.antspymm/I1499279_Anon_20210819142214_5.nii.gz` (b0 volume)
