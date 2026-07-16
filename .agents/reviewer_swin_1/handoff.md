# Swin UNETR 3D Encoder Review Handoff

## 1. Observation
I directly observed the following from running `pytest` and inspecting the codebase:
- **Observation 1 (SwinUNETR TypeError)**: In `src/syntx/features.py` line 204, the constructor for `SwinUNETR` is initialized with the `img_size` parameter:
  ```python
  self.model = SwinUNETR(
      img_size=self.img_size,
      in_channels=self.in_channels,
      out_channels=14,  # default out channels in SSL pretrained zoo
      feature_size=48,
      spatial_dims=3
  )
  ```
  This causes a crash when run in the environment:
  ```
  TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'
  ```
- **Observation 2 (SwinViT ImportError)**: In `tests/test_feature_networks.py` line 257, the test attempts to import `SwinViT` directly:
  ```python
  from monai.networks.nets import SwinViT
  ```
  This causes the following error:
  ```
  ImportError: cannot import name 'SwinViT' from 'monai.networks.nets'
  ```
- **Observation 3 (Ineffective Lazy Import Test Mocking)**: In `tests/test_feature_networks.py` lines 157-182, the test deletes keys from `sys.modules` to mock the absence of `monai`:
  ```python
  if 'monai' in sys.modules:
      del sys.modules['monai']
  ```
  However, this does not prevent a subsequent `import monai` call from reloading it from `sys.path`. Since MONAI is installed in the test environment, the class does not raise `ImportError` but instead runs through `__init__` and crashes with `TypeError`.
- **Observation 4 (Logic and Scaling Discontinuity)**: In `src/syntx/features.py` line 281-282, output interpolation calculation computes target spatial sizes as:
  ```python
  expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]
  ```
  However, the feature map extracted at `hidden_states[layer]` is downsampled by `2 ** (layer + 1)` relative to the input resolution of the backbone, leading to a scale factor mismatch (e.g. outputs of size $2\times$ what is expected). When `spatial_shape == img_size`, no interpolation is done, so for `layer=4` the size is `96 // 32 = 3`. When `spatial_shape = 95`, interpolation scales it to `95 // 16 = 5`.
- **Observation 5 (E2E Test Errors)**: Running `pytest` returned 19 failed tests, many of which failed with:
  ```
  ImportError: cannot import name 'dlpack_feature_loss' from 'syntx.syn_jax'
  ```

## 2. Logic Chain
1. From Observation 1, the `SwinUNETRExtractor` implementation directly violates the constructor signature of `SwinUNETR` in the installed MONAI package version. This makes the extractor unusable under standard installations.
2. From Observation 2, `test_swin_unetr_weights_download_and_key_cleaning` cannot compile or run because it references a non-existent name `SwinViT` in the `monai.networks.nets` package, leading to a direct test suite failure.
3. From Observation 3, the lazy-import check logic fails to test what it intends because deleting `sys.modules` keys does not hide an installed library from being re-imported.
4. From Observation 4, `expected_shape` uses a scale factor of `2 ** layer` while the corresponding extracted feature map `hidden_states[layer]` is downsampled by `2 ** (layer + 1)`. This results in a scale mismatch and severe discontinuities when switching between exact match sizes and non-matching sizes.
5. Therefore, the implementation has significant correctness, logic, and compatibility issues.

## 3. Caveats
- I did not fix any of the implementation or test code because the reviewer role prohibits writing to implementation directories.
- The `dlpack_feature_loss` import error was not investigated in detail, as it is outside the scope of Swin UNETR (it relates to DLPack JAX integration).

## 4. Conclusion
The implementation of the Swin UNETR 3D Encoder has critical design and runtime bugs. It fails to initialize due to parameter mismatches with MONAI (`img_size`), it fails to run its test suite due to invalid package imports (`SwinViT`), and it has a mathematical scaling error in its feature interpolation layer. The verdict is **REQUEST_CHANGES**.

## 5. Verification Method
- Execute the unit tests using `pytest -v tests/test_feature_networks.py`.
- If the implementation is corrected, all tests should pass without `TypeError` or `ImportError`.
