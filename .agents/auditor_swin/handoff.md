# Swin UNETR Audit Handoff Report

## 1. Observation

- **Command executed**: `pytest tests/test_feature_networks.py`
  - **Result**: Failed 4 out of 11 tests.
  - **Verbatim Error 1**:
    ```
    E       TypeError: SwinUNETR.__init__() got an unexpected keyword argument 'img_size'
    src/syntx/features.py:204: TypeError
    ```
  - **Verbatim Error 2**:
    ```
    E           ImportError: cannot import name 'SwinViT' from 'monai.networks.nets' (/Users/stnava/miniconda3/lib/python3.13/site-packages/monai/networks/nets/__init__.py)
    tests/test_feature_networks.py:257: ImportError
    ```
- **Command executed**: `pytest tests/test_e2e_metrics.py`
  - **Result**: Failed 19 out of 27 tests.
  - **Verbatim Error 3**:
    ```
    E   ImportError: cannot import name 'make_pytorch_loss_jax' from 'syntx.syn_jax' (/Users/stnava/code/syntx/src/syntx/syn_jax.py)
    ```
- **Source Code Verification**:
  - `src/syntx/features.py` lines 178–287:
    Implements `SwinUNETRExtractor` inheriting from `FeatureExtractor`. It uses a lazy import of `SwinUNETR` from `monai.networks.nets`, verifies arguments, instantiates the architecture, downloads pre-trained weights if not cached, handles state_dict key stripping (e.g. `module.` and `swinViT.`), and applies trilinear resizing for mismatching spatial shapes.
  - `src/syntx/syn.py` lines 885–890:
    Integrates `'swinunetr'` or `'swin_unetr'` metric keys, mapping to `SwinUNETRExtractor` wrapped in `FeatureSpaceLoss`.
  - `tests/test_feature_networks.py` lines 157–280:
    Includes tests for lazy imports, shape matching, interpolation, and weights cleaning.
  - `tests/test_swin_unetr_empirical.py` lines 1–129:
    A separate test suite that fully mocks MONAI imports and passes all tests.

## 2. Logic Chain

- **Observation on Code Structure**: The implementation in `src/syntx/features.py` and integration in `src/syntx/syn.py` represents a real, complete integration of the Swin UNETR metric framework. The parameters, weights cleaning, lazy loading, and interpolation logic are actively implemented.
- **Observation on Mocking**: The test suite `test_swin_unetr_empirical.py` validates the internal routing and interpolation maths by mocking the MONAI dependencies.
- **Observation on Failures**: The unit tests in `test_feature_networks.py` and `test_e2e_metrics.py` fail because:
  - The installed MONAI version is 1.6.0.
  - In MONAI 1.6.0, `SwinUNETR` does not accept the `img_size` argument (which is passed in `features.py`).
  - In MONAI 1.6.0, `SwinViT` is not directly exported in `monai.networks.nets`.
- **Verdict Assessment**: Since the failure is entirely due to runtime compatibility issues with MONAI 1.6.0's updated constructor signature and exports, rather than the presence of dummy mocks or hardcoded test values designed to deceive, this is a genuine but buggy implementation.
- **Conclusion Support**: Under the lenient Development Mode (as specified in `.agents/ORIGINAL_REQUEST.md`), wrappers and library usage are allowed, and only fabricated/dummy features are violations. Therefore, the verdict is CLEAN.

## 3. Caveats

- We did not modify any source code or test files to resolve the MONAI 1.6.0 compatibility issues, as the audit protocol requires an audit-only role.
- We did not deep-dive into resolving `syn_jax.py`'s missing functions (`to_torch_tensor`, `make_pytorch_loss_jax`, `dlpack_feature_loss`) since they were not in the designated files to audit.

## 4. Conclusion

- **Verdict**: CLEAN.
- The Swin UNETR implementation is genuine, does not hardcode test results, and does not use facade or dummy patterns.
- There are active runtime type/import errors due to incompatibility with MONAI 1.6.0, which must be addressed by updating the constructor call in `src/syntx/features.py` (removing `img_size` or updating it to match MONAI 1.6.0) and the import in `tests/test_feature_networks.py`.

## 5. Verification Method

- **Files to Inspect**:
  - `src/syntx/features.py`
  - `/Users/stnava/code/syntx/.agents/auditor_swin/audit.md`
- **Commands to Run**:
  - `pytest tests/test_swin_unetr_empirical.py` (passes with mocks).
  - `pytest tests/test_feature_networks.py` (demonstrates the MONAI 1.6.0 signature and import errors).
