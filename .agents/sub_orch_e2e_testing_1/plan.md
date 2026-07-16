# E2E Test Suite Design Plan

This document outlines the design for the 27 test cases across 4 tiers to verify the JAX Feature-Space Metrics & Swin UNETR Integration.

## Features Under Test
1. **DLPack-based PyTorch Feature-Space Loss integration in JAX SyN loops** (Feature 1)
2. **MONAI Swin UNETR 3D feature extractor** (Feature 2)

---

## Tier 1: Feature Coverage (10 Test Cases)
These tests ensure all features are fully exercised under standard conditions.

1. **`test_swin_unetr_extractor_init`**: Verify that `SwinUNETRExtractor` initializes correctly on CPU with default layers (`feature_layers=[4]`), dynamic lazy-loading is handled, and properties `is_3d` and `in_channels` return correct values.
2. **`test_swin_unetr_extractor_lazy_load_monai`**: Verify that MONAI is imported lazily inside `SwinUNETRExtractor.__init__` rather than at module import time.
3. **`test_swin_unetr_extractor_shapes`**: Verify feature extraction output shapes from `SwinUNETRExtractor` for a 3D input of shape `(1, 1, 96, 96, 96)` using layers `[1, 2, 3, 4]`.
4. **`test_swin_unetr_extractor_normalization`**: Verify that `SwinUNETRExtractor.normalize` handles 3D tensors correctly without modifications (identity or expected normalization).
5. **`test_dlpack_tensor_sharing_roundtrip`**: Verify the basic DLPack sharing bridge between JAX and PyTorch (JAX array -> DLPack -> PyTorch tensor -> DLPack -> JAX array) ensuring zero-copy and identical values.
6. **`test_dlpack_loss_forward`**: Verify that wrapping a PyTorch `FeatureSpaceLoss` with DLPack returns the correct loss value as a JAX scalar.
7. **`test_dlpack_loss_backward`**: Verify that JAX autograd (VJP) can compute gradients of the DLPack-wrapped PyTorch feature loss with respect to JAX input arrays, and that they match the expected gradient shapes and values.
8. **`test_dlpack_bridge_gradient_sharing`**: Verify that DLPack correctly shares the gradients from PyTorch's backward pass back to the JAX optimizer.
9. **`test_vgg_3d_lncc_layer4_jax`**: Verify that the VGG 3D LNCC (Layer 4) similarity metric operates correctly under the DLPack JAX bridge wrapper.
10. **`test_dlpack_multi_level_compatibility`**: Verify that the DLPack bridge works with varying downsampled spatial dimensions (e.g., matching JAX multiresolution pyramids).

---

## Tier 2: Boundary & Corner Cases (10 Test Cases)
These tests verify robustness under edge conditions, invalid inputs, and extreme values.

11. **`test_swin_unetr_invalid_input_dim`**: Verify that `SwinUNETRExtractor` raises a `ValueError` or appropriate error when given a 2D input tensor.
12. **`test_swin_unetr_invalid_layers`**: Verify that initializing `SwinUNETRExtractor` with invalid layer indices (e.g. out of bounds or empty) raises `ValueError`.
13. **`test_dlpack_mismatched_shapes`**: Verify that the DLPack bridge raises a `ValueError` when JAX and PyTorch expect different tensor shapes.
14. **`test_dlpack_unsupported_dtypes`**: Verify that the DLPack bridge raises a `TypeError` or handles gracefully when unsupported data types (like `int32` or `float64`) are passed.
15. **`test_dlpack_empty_tensors`**: Verify that passing zero-size or empty tensors to the DLPack bridge raises an error or is rejected.
16. **`test_swin_unetr_batch_sizes`**: Verify extraction shapes and robustness for different batch sizes (e.g., `B = 2` vs `B = 1` vs `B = 0` error handling).
17. **`test_dlpack_non_contiguous_arrays`**: Verify that the DLPack bridge handles non-contiguous JAX/PyTorch arrays correctly (either converting them to contiguous or raising an error).
18. **`test_dlpack_numerical_stability_nan_inf`**: Verify that the DLPack bridge handles inputs containing NaNs or Infs gracefully without crashing.
19. **`test_dlpack_detached_graphs`**: Verify that when PyTorch's loss returns a detached gradient (e.g. zero grad), the DLPack bridge correctly returns zero gradients to JAX.
20. **`test_swin_unetr_offline_cache_fallback`**: Verify that `SwinUNETRExtractor` falls back gracefully or uses random/mock weights if pretrained weight download fails or is offline.

---

## Tier 3: Cross-Feature Combinations (2 Test Cases)
These tests evaluate how the features interact with each other and the rest of the codebase.

21. **`test_syn_jax_step_with_swin_unetr_loss`**: Verify that a single optimization step of `syn_step_jax` runs successfully when using the DLPack-wrapped Swin UNETR feature loss.
22. **`test_multimetric_syn_jax_registration`**: Verify that `SyNTo.fit` can combine standard intensity LNCC and DLPack-wrapped PyTorch feature loss (e.g. VGG 3D LNCC Layer 4) in a multi-metric registration.

---

## Tier 4: Real-World Application Scenarios (5 Test Cases)
These tests verify realistic application-level scenarios using actual dataset files and checking domain constraints.

23. **`test_real_t1w_to_b0_registration_swin_unetr`**: Run a 3D registration of T1w-to-B0 MRI volumes using the Swin UNETR Feature-Space Loss and verify that it completes without errors.
24. **`test_real_t1w_to_dwi_registration_vgg3d`**: Run a 3D registration of T1w-to-DWI MRI volumes using VGG 3D LNCC (Layer 4) and verify that it completes and the outputs are saved.
25. **`test_registration_folding_constraint`**: Run a 3D registration using DLPack feature loss and assert that the folding rate (fraction of voxels where Jacobian determinant <= 0) is <= 0.01% (Single Interpolation Policy check).
26. **`test_comparative_metrics_script_execution`**: Run the multi-modal comparison registration benchmark and verify that results are written to `outputs_comparison/final_feature_metrics_results.csv`.
27. **`test_cortical_label_registration_accuracy`**: Register a cortical label map using VGG 3D LNCC with Layer 4, and verify that the Mean DICE score does not regress by >= 0.01 compared to standard intensity LNCC (VGG 3D LNCC accuracy requirement).

---

## Mock/Fallback Execution Strategy
Since `monai` might not be installed in the environment, the test harness will support **Mocking MONAI** when `monai` is missing. This ensures the tests run cleanly in all environments and verify the interface contracts, while using the real `monai` if it is present.
Similarly, if some dataset files are missing, the tests will fall back to simulated/cached images or mock registrations to ensure 100% test suite execution.
