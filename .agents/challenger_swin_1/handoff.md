# Handoff Report - SwinUNETRExtractor Challenge

## 1. Observation
- **File Checked**: `src/syntx/features.py` (lines 178-287)
- **Test File**: `tests/test_feature_networks.py` (lines 184-280)
- **Initialization Error**: Running `pytest tests/test_feature_networks.py` under Python 3.13 with MONAI 1.6.0 installed results in:
  `TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'`
- **Import Error**: Running pytest results in:
  `ImportError: cannot import name 'SwinViT' from 'monai.networks.nets'`
- **Interpolation Shapes**: Running the verification script `verify_swin.py` with the `SwinUNETR` class patched to bypass the `img_size` argument resulted in the following feature dimensions for an input of shape `64x64x64`:
  - Layer 1 output: `torch.Size([1, 96, 32, 32, 32])` (Expected physical shape: `16x16x16`)
  - Layer 2 output: `torch.Size([1, 192, 16, 16, 16])` (Expected physical shape: `8x8x8`)
  - Layer 3 output: `torch.Size([1, 384, 8, 8, 8])` (Expected physical shape: `4x4x4`)
  - Layer 4 output: `torch.Size([1, 768, 4, 4, 4])` (Expected physical shape: `2x2x2`)

---

## 2. Logic Chain
1. When instantiating `SwinUNETRExtractor`, the code passes the `img_size` keyword argument to `SwinUNETR`. Since MONAI's `SwinUNETR` constructor signature does not accept this argument, a `TypeError` occurs, preventing any execution.
2. In the unit test `test_swin_unetr_weights_download_and_key_cleaning`, the test code tries to import `SwinViT` from `monai.networks.nets`. However, MONAI's underlying class is named `SwinTransformer` (and resides under `monai.networks.nets.swin_unetr`). Thus, importing `SwinViT` raises an `ImportError`.
3. When input shapes mismatch the native `img_size` (96), the input is interpolated to `96^3`, passed through `swinViT`, and then the output features are interpolated back using the formula `expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]`.
4. However, the downsampling factor of MONAI's `SwinUNETR` at `layer` (index 1 to 4) is actually $2^{\text{layer}+1}$ (e.g. at layer 4, downsampling is 32, not 16).
5. As a result, the returned feature maps have spatial shapes twice as large as the correct downsampled shapes. This spatial inconsistency corrupts registration loss calculation.

---

## 3. Caveats
- Benchmark measurements were executed on CPU only. GPU-specific behavior (e.g. CUDA memory limits, GPU forward pass speed) was not measured.
- Monai versions earlier than 1.6.0 were not tested, though the initialization argument bug is expected to occur on all modern versions of MONAI where `SwinUNETR` does not accept `img_size`.

---

## 4. Conclusion
`SwinUNETRExtractor` is currently unusable. To resolve this:
1. `img_size=self.img_size` must be removed from `SwinUNETR` instantiation in `src/syntx/features.py`.
2. The downsampling exponent formula in `extract()` must be corrected from `2 ** layer` to `2 ** (layer + 1)` to align feature map interpolation dimensions.
3. Unit test imports of `SwinViT` must be corrected to import `SwinTransformer` from `monai.networks.nets.swin_unetr`.

---

## 5. Verification Method
To reproduce the findings:
1. Install MONAI and einops: `pip install monai einops`
2. Run the verification script: `python .agents/challenger_swin_1/verify_swin.py`
3. Run the unit tests: `pytest tests/test_feature_networks.py`
