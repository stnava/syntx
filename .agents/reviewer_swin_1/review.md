# Review Report: Swin UNETR 3D Encoder Implementation

**Verdict**: REQUEST_CHANGES

---

## Findings

### [Critical] Finding 1: SwinUNETR Initialization TypeError (img_size parameter)
- **What**: Passing the `img_size` parameter to `SwinUNETR`'s constructor throws a `TypeError`.
- **Where**: `src/syntx/features.py`, line 204
- **Why**: In modern MONAI versions (e.g., MONAI 1.5.0/1.6.0), `SwinUNETR`'s `__init__` constructor does not accept `img_size` as a keyword argument (it is fully convolutional/deformable and accepts arbitrary spatial input dimensions that are divisible by $2^5$). Passing `img_size` results in a crash: `TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'`.
- **Suggestion**: Remove `img_size` from the arguments passed to `SwinUNETR` in `SwinUNETRExtractor.__init__`.

### [Major] Finding 2: SwinViT ImportError in Unit Tests
- **What**: The unit tests attempt to import `SwinViT` from `monai.networks.nets`, which does not exist, causing an `ImportError`.
- **Where**: `tests/test_feature_networks.py`, line 257 (inside `test_swin_unetr_weights_download_and_key_cleaning`)
- **Why**: In MONAI, `SwinViT` is not exposed directly in `monai.networks.nets`. Instead, `SwinTransformer` is used as the backbone class (though it is referenced as `swinViT` inside the model). Attempting to import `SwinViT` fails.
- **Suggestion**: Import `SwinTransformer` instead of `SwinViT` for unit testing and state-dict loading checks.

### [Major] Finding 3: Logic and Scaling Mismatch in Output Feature Interpolation
- **What**: The output spatial size calculation during interpolation introduces a scale mismatch and discontinuity.
- **Where**: `src/syntx/features.py`, lines 281-282 (inside `SwinUNETRExtractor.extract`)
- **Why**:
  1. **Scale Mismatch**: `hidden_states[layer]` contains feature maps with downsampling factor $2^{layer + 1}$ relative to the input shape `original_img_size` (since `hidden_states[0]` is downsampled by 2, `hidden_states[1]` by 4, etc.). However, the expected shape is calculated using `2 ** layer`: `expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]`. For example, for `layer = 4`, the actual feature map is downsampled by 32, but the code scales it to `s // 16`, making the output feature map twice the correct spatial size.
  2. **Size Discontinuity**: If `spatial_shape == original_img_size` (e.g., 96), the interpolation is skipped, and the output shape is `96 // 32 = 3` (for layer 4). If `spatial_shape = 95`, the interpolation is executed and scales to `95 // 16 = 5`. A tiny change in input size (from 96 to 95) results in a sudden jump in feature map size from 3 to 5.
- **Suggestion**: Change the scale factor in `expected_shape` to `2 ** (layer + 1)` (or `2 ** (idx + 1)` depending on the actual extracted index) to align the downsampling and scaling factors correctly.

### [Minor] Finding 4: Ineffective Mocking in Lazy Import Test
- **What**: `test_swin_unetr_extractor_lazy_import` fails to raise `ImportError` under environments where MONAI is installed because deleting keys from `sys.modules` does not prevent standard Python imports from finding the library in `sys.path`.
- **Where**: `tests/test_feature_networks.py`, lines 157-182 (inside `test_swin_unetr_extractor_lazy_import`)
- **Why**: Deleting `monai` from `sys.modules` does not stop subsequent import statements from re-importing the library from the installed environment. Hence, the test does not raise an `ImportError` and instead raises a `TypeError` due to the `img_size` bug, leading to test failure.
- **Suggestion**: Use `unittest.mock.patch.dict("sys.modules", {"monai": None})` or similar mocking strategies to correctly simulate the absence of the library, rather than manually deleting keys from `sys.modules`.

---

## Verified Claims

- **DINOv2 Extractor Shape and Pruning**: Verified via `test_extractors_and_loss_shapes` (passed when torch hub is cached / simulated). -> **PASS**
- **ResNet10 2D and 3D Feature Extractor Shapes**: Verified via `test_resnet10_architectures` and `test_extractors_and_loss_shapes` -> **PASS**
- **SwinUNETRExtractor Lazy Import Failure**: Verified via running `pytest` which showed that `test_swin_unetr_extractor_lazy_import` failed to raise `ImportError` under MONAI installed environments. -> **FAIL**
- **SwinUNETRExtractor Shapes and Interpolation**: Verified via running `pytest` which showed that these tests threw `TypeError` during model instantiation. -> **FAIL**

---

## Coverage Gaps

- **Offline weight availability**: The weight loading mechanism uses `urllib.request.urlretrieve` which fails in offline/restricted network environments. Although it catches exceptions and warns, there is a risk that the model will default silently to randomly initialized weights, which would severely degrade registration performance.
- **SwinUNETR integration in JAX**: JAX support is currently unverified as all tests relating to Swin UNETR integration in JAX fail with `ImportError` due to the missing `dlpack_feature_loss` in `syntx.syn_jax` (which is outside the current Swin UNETR scope but blocks overall verification).

---

## Unverified Items

- **Real registration alignment with Swin UNETR**: Cannot be verified because the unit tests and registration pipeline fail during `SwinUNETR` model instantiation.
