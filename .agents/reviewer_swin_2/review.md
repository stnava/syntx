## Review Summary

**Verdict**: REQUEST_CHANGES

## Findings

### Critical Finding 1: SwinUNETR Initialization TypeError
- **What**: The initialization of `SwinUNETRExtractor` raises a `TypeError` because MONAI's `SwinUNETR` does not accept `img_size` in its constructor.
- **Where**: `src/syntx/features.py:204`
- **Why**: In MONAI 1.6.0, `SwinUNETR` constructor has no `img_size` parameter. This causes all initializations of `SwinUNETRExtractor` to fail immediately.
- **Suggestion**: Remove `img_size=self.img_size` from the `SwinUNETR` instantiation.

### Major Finding 2: Incorrect Downsampling Factor and Spatial Discontinuity
- **What**: The downsampling factor calculation uses `2 ** layer` instead of `2 ** (layer + 1)` when computing the expected size for output interpolation.
- **Where**: `src/syntx/features.py:281`
- **Why**: Swin Transformer features are downsampled by factors of 2 (projection/stage 0), 4 (stage 1), 8 (stage 2), 16 (stage 3), and 32 (stage 4). The stage indices mapped to `layer` (1, 2, 3, 4) correspond to downsampling factors of 4, 8, 16, and 32 respectively, which is `2 ** (layer + 1)`. Using `2 ** layer` (yielding 2, 4, 8, 16) causes:
  1. Output feature dimensions to mismatch other models (e.g. ResNet-10 layer 4 outputs size 2 for input size 64, but SwinUNETR outputs size 4).
  2. A massive spatial resolution discontinuity. For instance, an input of size 96 bypasses interpolation and outputs stage 4 feature map at `/32` (size 3). An input of size 95 gets interpolated to 96, runs through Swin, and gets interpolated back to `95 // 16 = 5`, meaning a 1-voxel input size difference results in a size difference of 2 voxels in the feature space (3 vs 5).
- **Suggestion**: Update the expected shape calculation to: `expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]`

### Major Finding 3: SwinViT ImportError in Unit Tests
- **What**: The test `test_swin_unetr_weights_download_and_key_cleaning` tries to import `SwinViT` from `monai.networks.nets` and raises an `ImportError`.
- **Where**: `tests/test_feature_networks.py:257`
- **Why**: Class `SwinViT` is not exposed in `monai.networks.nets` in MONAI 1.6.0. The actual class name is `SwinTransformer`, which is located in `monai.networks.nets.swin_unetr`.
- **Suggestion**: Import `SwinTransformer` from `monai.networks.nets.swin_unetr` (or mock the model class appropriately).

### Minor Finding 4: Missing `einops` Environment Dependency
- **What**: Running the SwinUNETRExtractor results in a `ModuleNotFoundError: No module named 'einops'` because `einops` is not installed.
- **Where**: Environment level / runtime import of MONAI's SwinUNETR.
- **Why**: MONAI's SwinUNETR implementation depends on `einops`, but it is not listed in `pyproject.toml` or installed in the current environment.
- **Suggestion**: Add `einops` to `pyproject.toml` dependencies or test dependencies.

### Minor Finding 5: Hardcoded `window_size` in 3D Perceptual Loss
- **What**: In `FeatureSpaceLoss`, both `_forward_3d` and `_forward_2d_reconstruct_3d` ignore the user-defined `lncc_window` and hardcode `window_size=5`.
- **Where**: `src/syntx/features.py:323, 445-447`
- **Why**: While 5x5x5 is a standard local window, ignoring the constructor argument `lncc_window` when `is_3d` is True or mode is `lncc_3d` is inconsistent.
- **Suggestion**: Use `self.lncc_window` instead of hardcoding `5`.

## Verified Claims

- SwinUNETRExtractor lazy loading warning fallback → verified via running tests → PASS (it raises a warning as expected)
- ResNet-10 shape tests → verified via pytest → PASS
- DINOv2 small shape tests → verified via pytest → PASS
- VGG19 shape tests → verified via pytest → PASS

## Coverage Gaps

- SwinUNETRExtractor E2E registration performance is currently untested due to the inability to run the extractor without `einops` and the `TypeError` constructor bug. High risk — must be re-evaluated after changes are made.

## Unverified Items

- Final registration accuracy of SwinUNETR compared to intensity LNCC — not verified because tests crash before execution.
