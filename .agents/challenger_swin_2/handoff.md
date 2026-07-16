# Handoff Report - SwinUNETRExtractor Review

## 1. Observation
- **Interpolation Shape Mismatch**: Under input size `64x64x64`, the SwinUNETRExtractor `extract` method outputs:
  - Layer 1: shape `(1, 96, 32, 32, 32)` (expected `(1, 96, 16, 16, 16)` based on 4x downsampling from input)
  - Layer 4: shape `(1, 384, 4, 4, 4)` (expected `(1, 384, 2, 2, 2)` based on 32x downsampling from input)
  This was verified by running `pytest -s tests/test_swin_unetr_empirical.py`.
- **TypeError with Isotropic img_size**: Initializing with `img_size=96` raises `TypeError: 'int' object is not iterable` in `extract`.
- **Offline Behavior**: If the weight Pt file does not exist, the network warns but falls back silently to random weights, which disrupts optimization.
- **Dynamic Sizing**: In MONAI 1.6.0, `SwinUNETR` class does not accept `img_size` argument in `__init__`, which was resolved in the current workspace by removing it from the `SwinUNETR` instantiation but keeping the default `img_size` in the extractor configuration.

## 2. Logic Chain
- SwinUNETR's stages downsample by factors of $2^{L+1}$ where $L$ is the layer index (1 to 4).
- The implementation calculates the target shape using `2 ** layer`, which is $2^{L}$, resulting in an off-by-one exponent error.
- As a consequence, the feature map is interpolated to twice the correct spatial size in each axis, causing:
  - $2^3 = 8\times$ volume size (meaning $8\times$ memory consumption in PyTorch's loss computation).
  - Spatial blurring and mismatch against true receptive fields.
  - Scale inconsistencies: a minute change of 1 voxel in input size changes the downsampling factor by 2x.

## 3. Caveats
- The tests were executed using a mocked MONAI backbone (`MockSwinViT` and `MockSwinUNETR`) to avoid local environment variations, though the real MONAI was also inspected and verified to behave identically.
- GPU-specific OOM errors under the incorrect shape size were not directly profiled but are mathematically guaranteed to scale by $8\times$.

## 4. Conclusion
The current `SwinUNETRExtractor` has critical defects:
1. An off-by-one exponent bug in the output shape scaling (uses `2**layer` instead of `2**(layer+1)`), leading to an $8\times$ VRAM/RAM overhead and spatial scale mismatch.
2. A crash hazard when configured with an isotropic integer `img_size`.
3. A silent fallback to random weights in offline environments, causing silent optimization failures.
4. A double-interpolation strategy that runs counter to the Single Interpolation Policy.

## 5. Verification Method
To verify these conclusions, run the empirical test suite:
```bash
pytest -s tests/test_swin_unetr_empirical.py
```
Check the printed output shapes and asserts to confirm the 8x volume mismatch and other edge cases.
