## 2026-07-14T18:55:00Z
You are the worker responsible for optimizing the syntx registration pipeline (Milestones 2 & 3).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please implement the following optimizations and changes in the codebase:

1. Optimize JAX-PyTorch DLPack Bridge (src/syntx/syn_jax.py):
   - Wrap make_pytorch_loss_jax so it attaches '_is_pytorch_loss = True' and '_pytorch_loss_fn = pytorch_loss_fn' to the returned function.
   - Redesign the SyN registration step loop in SyNTo.fit() to separate PyTorch similarity metric execution from JAX JIT-compiled updates:
     * Write JIT-compiled helper 'prepare_mid_images_and_gradients_jax' to warp fixed/moving images to the middle space and compute physical image spatial gradients.
     * Write JIT-compiled helper 'syn_update_step_jax' to perform displacement field updates, smoothing, and double inversion.
     * For each registration epoch:
       - Warp images and compute image gradients using 'prepare_mid_images_and_gradients_jax'.
       - For each loss function in self.loss_functions:
         a. If the function is a PyTorch loss (has '_is_pytorch_loss'), execute the forward and backward passes eagerly on GPU device arrays using DLPack (to_torch_tensor and to_jax_array_dl) and PyTorch autograd. Bypassing JAX value_and_grad avoids JAX tracers, JIT compiling, and CPU fallback.
         b. If native JAX, evaluate using a JIT-compiled or eager JAX gradient helper.
       - Sum losses and gradients.
       - If 'use_analytical_gradients' is True, compute raw warp gradients using the image gradients.
       - If 'use_analytical_gradients' is False, use jax.vjp on the warping function (which is eager/JIT-compiled) to propagate gradients of the loss back to the warps.
       - Update the warps using 'syn_update_step_jax'.
     * This avoids compile cache invalidation and completely avoids CPU fallback during registrations.

2. Optimize Feature Extractors & Loss (src/syntx/features.py):
   - In SwinUNETRExtractor.extract: Instead of trilinear upsampling the input volume to (96, 96, 96) and downsampling the features back, compute the nearest multiple of 32 for each spatial dimension, pad the input volume to this size, run the Swin ViT backbone, and crop the output feature maps to the expected resolution. This completely avoids interpolation and significantly reduces voxels processed for smaller inputs (e.g. 16x16x16).
   - In FeatureSpaceLoss._forward_2d_triplanar: Avoid redundant upscaling to 'target_size' if possible, or optimize padding.

3. Align with GEMINI.md (src/syntx/syn.py):
   - Single Interpolation Policy: Do not pre-warp the moving image via 'ants.apply_transforms' prior to optimization when 'initial_transform' is provided.
   - Instead, compute an 'initial_grid' (representing the mapping from fixed space to moving space under the initial transform) using coordinate warping (helper function that warps moving physical coords to fixed space and normalizes to [-1, 1]). Pass 'initial_grid' to model.fit(..., initial_grid=initial_grid).
   - In SyNTo.fit(): If 'initial_grid' is provided, downsample/resize it for each resolution level using linear interpolation, and compose it on the fly with the learned transforms when warping moving images (for both affine pre-alignment and SyN).
   - Ensure the final warped output images still apply all composed transforms in a single interpolation step at the end.
   - In syn.py, update default parameter signatures from 'vgg_layers=[8]' and 'vgg_mode="patch_walk"' to 'vgg_layers=[4]' and 'vgg_mode="lncc_3d"'.

4. Verification:
   - Run pytest to ensure all 77 unit tests continue to pass.
   - Run python examples/evaluate_all_metrics.py to verify it saves outputs to outputs_comparison/final_feature_metrics_results.csv.

Write a handoff report at /Users/stnava/code/syntx/.agents/worker_perf_optimization/handoff.md when complete.

## 2026-07-14T19:01:43Z
[Message] sender=f21b20dc-e4b4-4894-9c5b-2f32499326d4 priority=MESSAGE_PRIORITY_HIGH content=**Context**: Performance Optimizations and Code Coverage
**Content**: The parent agent reported that total repository code coverage has dropped to 87% due to the new functions. Please ensure you add comprehensive unit test cases in the test suite (e.g., tests/test_syn_jax.py or tests/test_e2e_metrics.py) covering all new JAX functions (prepare_mid_images_and_gradients_jax, syn_update_step_jax, upscale_initial_grid) and composition grid functions in syn_jax.py and syn.py. This is required to restore and maintain the total repository code coverage to >= 90%.
**Action**: Implement these unit tests, verify they pass, and ensure code coverage is >= 90% before delivering the final handoff.
