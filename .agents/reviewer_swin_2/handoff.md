# Handoff Report — Swin UNETR 3D Encoder Review

## 1. Observation

### Unit Test Outputs
Running `pytest tests/test_feature_networks.py` produced the following errors:

- **TypeError during instantiation of `SwinUNETR`**:
```
E       TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'
src/syntx/features.py:204: TypeError
```

- **ImportError for SwinViT**:
```
FAILED tests/test_feature_networks.py::test_swin_unetr_weights_download_and_key_cleaning
E           ImportError: cannot import name 'SwinViT' from 'monai.networks.nets' (/Users/stnava/miniconda3/lib/python3.13/site-packages/monai/networks/nets/__init__.py)
```

### PyTorch Code Inspection
- In `src/syntx/features.py` lines 204-210, `SwinUNETR` is initialized as:
```python
        self.model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=14,  # default out channels in SSL pretrained zoo
            feature_size=48,
            spatial_dims=3
        )
```
- In `src/syntx/features.py` lines 280-282, output features expected shape is calculated as:
```python
            if spatial_shape != tuple(original_img_size):
                expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]
                feat = F.interpolate(feat, size=expected_shape, mode='trilinear', align_corners=True)
```
- In `tests/test_feature_networks.py` line 257, `SwinViT` is imported as:
```python
        from monai.networks.nets import SwinViT
```

### Package Inspection
- Running python inspect on the environment package MONAI 1.6.0 yields:
```
TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'
```
and:
```
ModuleNotFoundError: No module named 'einops'
```
while listing keys inside `monai.networks.nets.swin_unetr` returned `['SwinTransformer', 'SwinTransformerBlock', 'SwinUNETR']` but not `SwinViT`.

---

## 2. Logic Chain

1. **TypeError in SwinUNETR**:
   - *Observation*: Pytest output shows `SwinUNETR.__init__() got an unexpected keyword argument 'img_size'` when calling `SwinUNETR(...)` in `src/syntx/features.py` at line 204.
   - *Reasoning*: MONAI 1.6.0 has removed the `img_size` argument from `SwinUNETR.__init__` signature, leaving it to accept fully convolutional variable dimensions. Thus, attempting to construct it with `img_size` throws a `TypeError`.

2. **ImportError in Unit Tests**:
   - *Observation*: Pytest output shows `ImportError: cannot import name 'SwinViT' from 'monai.networks.nets'` at `test_feature_networks.py` line 257.
   - *Reasoning*: The class `SwinViT` is not exposed in MONAI 1.6.0's nets module. The actual backbone module class is `SwinTransformer` from `monai.networks.nets.swin_unetr`. The mock in the unit test refers to a non-existent name, causing the test to fail.

3. **Missing `einops` dependency**:
   - *Observation*: Forward pass execution of `SwinUNETR` raised `OptionalImportError: from einops import rearrange (No module named 'einops')`.
   - *Reasoning*: MONAI's implementation of Swin Transformer relies on `einops`, which is not listed in `pyproject.toml` or installed in the target environment.

4. **Downsampling Discontinuity**:
   - *Observation*: Expected shape in `src/syntx/features.py` line 281 computes `s // (2 ** layer)`.
   - *Reasoning*: Swin ViT downsamples by 2 at the patch embedding stage (stage 0), then by 2 at stages 1, 2, 3, 4. Thus:
     - `layer = 1` downsamples by 4.
     - `layer = 2` downsamples by 8.
     - `layer = 3` downsamples by 16.
     - `layer = 4` downsamples by 32.
     - The factor should be `2 ** (layer + 1)`. Since the code uses `2 ** layer`, it calculates downsampling as `/16` for layer 4. Consequently:
       - For native `96x96x96` inputs, no interpolation is done, and layer 4 returns `3x3x3` features (downsampled by 32).
       - For input `95`, interpolation is done, and it calculates `expected_shape = 95 // 16 = 5`. It returns `5x5x5` features.
       - A smaller input size (95 vs 96) yields a larger feature grid (5 vs 3). This introduces dimension mismatch and discontinuity bugs.

---

## 3. Caveats

- E2E tests have multiple other failures in `test_e2e_metrics.py` because the JAX-PyTorch bridge module (`dlpack_feature_loss`) is not implemented yet. This was not investigated in detail, as it is outside the scope of Milestone 3 Swin UNETR review, but it prevents the entire E2E test suite from passing.
- We did not evaluate registration accuracy of the Swin UNETR encoder empirically due to the inability to run the extractor.

---

## 4. Conclusion

The work product created by worker_swin contains several critical correctness bugs and failing tests:
- `SwinUNETRExtractor` cannot be initialized due to `img_size` TypeError.
- Unit tests fail due to importing the non-existent `SwinViT` class.
- The downsampling factor logic is mathematically incorrect, introducing spatial resolution discontinuities.

Verdict: **REQUEST_CHANGES**

---

## 5. Verification Method

To verify the changes independently:
1. Run `pytest tests/test_feature_networks.py`. It must pass all 11 tests once the SwinUNETR constructor, `SwinTransformer` import mock, and downsampling calculations are fixed.
2. Confirm `einops` is installed (`pip install einops`) to allow actual forward passes.
