# Scope: JAX Feature-Space Metrics & Swin UNETR Integration

## Architecture
- PyTorch-based features in `src/syntx/features.py`, containing:
  - Base `FeatureExtractor` and subclasses (VGG19, DINOv2, ResNet-10, and now `SwinUNETRExtractor`).
  - `FeatureSpaceLoss` calculating LNCC or other similarity metrics over features.
- JAX-based SyN optimizer in `src/syntx/syn_jax.py`, containing:
  - `SyNTo` JAX optimization class.
  - A JAX-PyTorch zero-copy bridge via DLPack (`jax.dlpack` and `torch.utils.dlpack`) to compute gradients of PyTorch loss on JAX tensors without copying.
- Benchmarking/Evaluation script `examples/evaluate_all_metrics.py` testing registration.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Implement Swin UNETR 3D Encoder | Implement MONAI Swin UNETR 3D encoder (`SwinUNETRExtractor`) in `features.py` with lazy loading and cached weight loading. | none | PLANNED |
| 2 | DLPack sharing JAX-PyTorch bridge | Implement Flax/JAX support for modular feature-space metrics using DLPack tensor sharing in `src/syntx/syn_jax.py`. | M1 | PLANNED |
| 3 | Evaluation / Benchmarking Script | Implement `examples/evaluate_all_metrics.py` testing T1w-to-B0 and T1w-to-DWI registrations. | M2 | PLANNED |
| 4 | Unit Test Verification & Code Coverage | Ensure all unit tests pass, total code coverage remains >= 90%. | M3 | PLANNED |
| 5 | E2E Integration (Tiers 1-4) | Wait for TEST_READY.md and verify the final implementation against all E2E tests. | M4 | PLANNED |
| 6 | Adversarial Coverage Hardening (Tier 5) | Perform adversarial coverage hardening using a Challenger -> Worker -> Reviewer loop. | M5 | PLANNED |

## Interface Contracts
### `SwinUNETRExtractor` interface in `features.py`:
- Inherits from `FeatureExtractor`.
- `__init__(self, feature_layers=[4])` with lazy loading of MONAI Swin UNETR weights.
- `is_3d = True`
- `in_channels = 1`
- `normalize(self, x)`: identity mapping or custom normalization.
- `extract(self, x)`: extracts features at the specified layers.

### DLPack bridge in `syn_jax.py`:
- Wrapper `dlpack_feature_loss(loss_fn, img_fixed_jax, img_moving_jax, transform_params)` that converts JAX tensors to PyTorch tensors using `jax.dlpack.to_dlpack` and `torch.from_dlpack`, computes loss and gradients in PyTorch (utilizing PyTorch autograd), and converts the loss value and gradients back to JAX using DLPack.
- Integrate into JAX's optimizer loop (e.g., using `jax.custom_vjp` or by returning value and grad directly to custom fit loops).
