## 2026-07-14T18:32:16Z
You are worker_implementation, a worker agent.
Your working directory is `/Users/stnava/code/syntx/.agents/worker_implementation`. Please create this directory and write all coordination/metadata files there.

Your tasks are:
1. Fix all Swin UNETR 3D encoder bugs:
   - Handle integer/tuple `img_size` at initialization of `SwinUNETRExtractor` (e.g. if `img_size` is an integer, convert it to a 3-tuple).
   - In `src/syntx/features.py`, update the downsampling formula in `SwinUNETRExtractor.extract()` from `2**layer` to `2**(layer+1)`.
   - Update `tests/test_feature_networks.py` to import/mock `SwinTransformer` (from `monai.networks.nets.swin_unetr`) instead of `SwinViT` to resolve the `ImportError` under MONAI 1.6.0.
   - Update `tests/test_swin_unetr_empirical.py` to assert the correct expected downsampled shapes using `2**(layer+1)`.
   - Use `self.lncc_window` instead of hardcoded `5` in `FeatureSpaceLoss` 3D mode calculations.

2. Implement Flax/JAX support for modular feature-space metrics using DLPack:
   - In `src/syntx/syn_jax.py`, implement a JAX-PyTorch DLPack bridge using `jax.pure_callback` and `jax.custom_vjp` to wrap PyTorch-based `FeatureSpaceLoss` functions so JAX optimization can compute values and gradients without copying memory.
   - Update JAX's `SyNTo` and `fit` loop to accept, configure, and optimize with PyTorch feature losses (e.g., `'vgg19'`, `'dinov2'`, `'resnet10'`, `'swinunetr'`), including support for combined multi-metric loss lists and weights.
   - Update `src/syntx/syn.py`'s `registration` call to JAX backend to forward all VGG-related parameters (`vgg_layers`, `vgg_mode`, `vgg_lncc_window_size`, etc.).

3. Implement the comparative evaluation script `examples/evaluate_all_metrics.py`:
   - Test T1w-to-B0 and T1w-to-DWI registrations.
   - Compare backends and metrics.
   - Display a summary table and write output to `outputs_comparison/final_feature_metrics_results.csv`.
   - Include spatial/structural visualization generation (overlap, deformed grids, Jacobian determinants, and side-by-side deformed vs target images) to satisfy GEMINI.md guidelines.

4. Run all tests via pytest to verify they pass, and ensure total coverage is >= 90%.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## 2026-07-15T03:18:36Z
You are a Worker subagent. Your task is to implement the optimizer choices (specifically Adam, SGD, L-BFGS) alongside the current step-based/CFL update for deformable registration fields in both PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`), run systematic sweeps in 2D and 3D, and generate the required dashboard report.

Please follow these instructions:
1. Read the Explorer's handoff report at `/Users/stnava/code/syntx/.agents/explorer_exploration_1/handoff.md` to understand the designed architecture.
2. In `src/syntx/syn.py` (PyTorch backend):
   * Add support for `optimizer_type` (e.g. `cfl`, `adam`, `sgd`, `lbfgs`) and `optimizer_lr` in the `fit` method.
   * Integrate PyTorch's native `Adam`, `SGD` (with momentum), and `LBFGS` (with strong_wolfe line search) optimizers to update the `warp_l2r` and `warp_r2l` parameters.
   * Ensure that for `adam` and `sgd`, fluid regularization (smoothing of gradients using `separable_gaussian_filter` and boundary masking) is applied to the gradients before `optimizer.step()`.
   * For `lbfgs`, wrap the loss calculation and gradient smoothing inside a PyTorch `closure()`, and call `optimizer.step(closure)`.
   * Keep the standard CFL step-based update for `cfl`.
   * Apply boundary masking, elastic regularization, and diffeomorphic projection (inversion updates) at the end of each iteration for all optimizers.
3. In `src/syntx/syn_jax.py` (JAX backend):
   * Add similar `optimizer_type` and `optimizer_lr` parameters to the `fit` method.
   * Implement SGD and Adam parameter updates in JAX (compiled using JAX JIT).
   * For JAX L-BFGS, bridge JAX with Scipy's `minimize(..., method='L-BFGS-B', jac=True)` on CPU, wrapping parameter packing/unpacking and loss/gradient evaluation. Apply boundary masking and fluid gradient smoothing during gradient evaluation.
4. Implement a comprehensive benchmark sweep script `examples/run_optimizer_sweeps.py` (or extend `examples/run_benchmarks.py` / `examples/run_comprehensive_benchmarks.py`) that:
   * Systematically sweeps the 4 optimizers (`cfl`, `adam`, `sgd`, `lbfgs`) across deep feature metrics (VGG19, ResNet-10, DINOv2) and standard intensity baselines (`lncc`, `mattes_mi`) in both 2D and 3D.
   * Records Dice scores, folding rates (Jacobian determinant <= 0), optimization speeds, and convergence history (losses per iteration).
   * Verifies 3D baseline parity (within 1%) with `ants.registration`.
   * Compiles the rich HTML performance dashboard at `docs/optimizer_and_deep_feature_report.html` containing structural overlays, warp grids, Jacobian maps, convergence plots, and side-by-side deformed/target comparisons (adhering to GEMINI.md guidelines).
5. Add unit tests in `tests/test_optimizers.py` to verify the functionality of all new optimizers on both backends. Run the test suite and ensure all tests pass.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Return a message with your handoff report and the paths of modified/created files when done.
