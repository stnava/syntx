## Challenge Summary

**Overall risk assessment**: HIGH

Through systematic, empirical testing, we have verified that the `SwinUNETRExtractor` implementation in `src/syntx/features.py` exhibits three major functional and architectural flaws. Two of these are critical code bugs that cause significant shape inconsistency, memory overhead, and crash risks, while the third is a reliability vulnerability in offline environments.

---

## Challenges

### [High] Challenge 1: Exponent Off-by-One Bug in Feature Interpolation
- **Assumption challenged**: The feature map downsampling factor at layer $L$ in SwinUNETR is assumed to be $2^{L}$.
- **Attack scenario**: When the input size does not match the configured `img_size` (e.g. 64x64x64 input with 96x96x96 default `img_size`), the extractor interpolates the input to 96x96x96, runs SwinViT, and then interpolates the intermediate feature maps back to the input scale.
  However, it calculates the downsampling factor using:
  `expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]`
  In SwinTransformer/SwinUNETR, the patch embedding first downsamples by 2, and each subsequent stage downsamples by 2. Thus:
  - Layer 1 (index 1) actually downsamples by 4x.
  - Layer 2 (index 2) actually downsamples by 8x.
  - Layer 3 (index 3) actually downsamples by 16x.
  - Layer 4 (index 4) actually downsamples by 32x.
  The code uses `2 ** layer`, which computes downsampling factors of 2x, 4x, 8x, and 16x respectively (off-by-one in exponent).
- **Blast radius**:
  - **Memory Overhead**: The feature maps are $2^3 = 8\times$ larger in volume than expected. For 3D registration, this causes a major increase in VRAM/RAM consumption, making it highly prone to Out-Of-Memory (OOM) failures.
  - **Spatial Resolution Inconsistency**: The output feature size is inconsistent. For example:
    - Input size 96 (no interpolation) -> Layer 4 output size is **3x3x3** (32x downsampling).
    - Input size 95 (triggers interpolation) -> Layer 4 output size is **5x5x5** (16x downsampling).
    A reduction in input size by 1 voxel causes the feature map to *increase* in size.
- **Mitigation**: Change the downsampling factor exponent from `2 ** layer` to `2 ** (layer + 1)` in `expected_shape` calculation:
  `expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]`

### [Medium] Challenge 2: TypeError on Integer or Non-Isotropic `img_size`
- **Assumption challenged**: The `img_size` configuration is assumed to be checked and normalized to a tuple.
- **Attack scenario**: If the user initializes `SwinUNETRExtractor(img_size=96)` (standard practice for isotropic dimensions), initialization succeeds. However, calling `extract` immediately crashes with `TypeError: 'int' object is not iterable` at line 265:
  `if spatial_shape != tuple(original_img_size):`
- **Blast radius**: Sudden runtime crash when running feature extraction with valid isotropic configurations.
- **Mitigation**: Normalize `img_size` in `__init__` to a 3-tuple if passed as an integer or a sequence of different length.

### [Medium] Challenge 3: Silent Fallback to Random Weights in Offline Environments
- **Assumption challenged**: Registration loss relies on pretrained/meaningful feature representations.
- **Attack scenario**: In an offline or network-restricted environment, the model fails to download the MONAI Swin ViT weights (`model_swinvit.pt`). The exception is caught, and a warning is printed. However, the program execution continues silently.
- **Blast radius**: The extractor falls back to randomly initialized weights. The registration optimization will optimize against random noise features, causing the registration to fail silently or produce garbage alignments without raising an error.
- **Mitigation**: Add a boolean flag `allow_random_fallback` (default: False). If False and the weight file is missing/download fails, raise a descriptive `FileNotFoundError` or `ConnectionError` instead of a warning.

### [Medium] Challenge 4: Double Interpolation Policy Violation
- **Assumption challenged**: Input must be interpolated to `img_size` to use SwinUNETR features.
- **Attack scenario**: When the input spatial shape mismatches `img_size`, the input volume is first interpolated to `img_size`, and the extracted feature map is then interpolated back. This violates the Single Interpolation Policy (`GEMINI.md`) and introduces double-interpolation blurring.
- **Blast radius**: Spatial blurring and loss of high-frequency registration alignment details.
- **Mitigation**: Instead of interpolating the input, pad the input to the nearest multiple of 32 (like DINOv2Extractor does for patch size 14), run the model dynamically, and crop the output feature maps to the correct downsampled scale.

---

## Stress Test Results

- **Input 96^3, layer 1** -> Expected shape: (1, 96, 24, 24, 24) -> Actual shape: (1, 96, 24, 24, 24) -> **PASS**
- **Input 96^3, layer 4** -> Expected shape: (1, 384, 3, 3, 3) -> Actual shape: (1, 384, 3, 3, 3) -> **PASS**
- **Input 64^3, layer 1 (with interpolation)** -> Expected shape: (1, 96, 16, 16, 16) -> Actual shape: (1, 96, 32, 32, 32) -> **FAIL** (Inconsistent 2x scale)
- **Input 64^3, layer 4 (with interpolation)** -> Expected shape: (1, 384, 2, 2, 2) -> Actual shape: (1, 384, 4, 4, 4) -> **FAIL** (Inconsistent 16x scale)
- **Offline mode weight load fail** -> Expected: raise Error -> Actual: raise Warning and proceed with random weights -> **FAIL**
- **`img_size=96` (int)** -> Expected: handle isotropic size -> Actual: `TypeError: 'int' object is not iterable` -> **FAIL**

---

## Unchallenged Areas

- **VGG19 & DINOv2 Extractors**: Out of scope for this challenge request, but their shapes and simple 2D triplanar behaviors were briefly run and verified as part of the overall test suites.
