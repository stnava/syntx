# Explorer Exploration 1 Handoff Report — Optimizer Integration Design

## 1. Observation
We examined the following files and directories in the `syntx` codebase:
* **`src/syntx/syn.py`**:
  * **Line 767**: Class `SyNTo(nn.Module)` constructor storing `warp_l2r`, `warp_r2l`, `warp_l2r_inv`, and `warp_r2l_inv` as PyTorch `nn.Parameter` tensors initialized to zero:
    ```python
    self.warp_l2r = nn.Parameter(torch.zeros(1, *grid_shape, dim))
    self.warp_r2l = nn.Parameter(torch.zeros(1, *grid_shape, dim))
    ```
  * **Line 828**: `fit` method running registration hierarchy.
  * **Lines 1111-1224**: PyTorch registration loop. Step calculation is based on either analytical gradients or autograd backward passes. The updates are applied greedily using composition:
    ```python
    grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, self.fluid_sigma, spacing=curr_physical_spacing)
    ...
    delta_l = lr_l * grad_l
    ...
    coords_l = identity - delta_l
    ...
    warp_l2r.copy_(warp_l2r_sampled - delta_l)
    ```
    Dirichlet zero boundaries are enforced via `warp_l2r.mul_(b_mask)`. Elastic regularization uses `separable_gaussian_filter` on the warp fields if `elastic_sigma > 0.0`. Diffeomorphic projection uses `update_inverse_field_nd` iterations.
* **`src/syntx/syn_jax.py`**:
  * **Line 845**: `syn_update_step_jax` compiles update logic using `jax.jit`.
  * **Lines 920-955**: JAX `SyNTo` constructor with similar grid field layouts in NumPy arrays.
  * **Lines 1280-1350**: JAX registration loop. In each step, midpoint images are warped via `prepare_mid_images_and_gradients_jax` or `warp_images_jax` / `jax.vjp`. Updates are compiled into `syn_update_step_jax`.
* **`examples/run_benchmarks.py`**:
  * **Lines 188-290**: `run_2d_benchmarks` registers phantom image pairs (`r16`, `r27`, `r64`) using metrics like `vgg19`, `dinov2`, and standard metrics, comparing with ANTs SyN.
  * **Lines 293-490**: `run_3d_benchmarks` registers native T1w brain images, extracts 2D LAI slices, generates Edge Overlays, Warped Grid Overlays, Jacobian Determinant maps, and calculates multiregion Dice metrics, comparing with ANTs SyN.
* **`tests/test_challenger_verification.py`**:
  * **Line 207**: `test_parameter_tuning_dice_parity` verifies that PyTorch and JAX `SyNTo` implementations achieve Dice score parity with the classic ANTs SyN baseline within 1%.

## 2. Logic Chain
1. **Diffeomorphic Composition vs. Additive Updates**:
   * For gradient descent and Adam optimizers, updates can be composed diffeomorphically (matching standard SyN, where `warp_new = warp(identity - delta) - delta`) or additively (standard deep learning optimizer update `warp_new = warp - delta`).
   * For L-BFGS, standard line searches and step length scaling assume that the parameters live in a vector space and are updated additively. Thus, L-BFGS must be implemented using **additive updates** during its line search evaluations.
2. **Spatial Regularization in Optimizers**:
   * Fluid regularization (smoothing of updating velocities/gradients) must be applied either to the raw gradients before the optimizer step (`fluid_regularization='gradient'`) or to the optimizer step direction output before scaling (`fluid_regularization='step'`).
   * In practice, smoothing the gradient *before* passing it to Adam is necessary to avoid numerical instability in the second-moment estimate ($v_t$), preventing high-frequency noise from inflating the denominator.
   * Elastic regularization (smoothing of the displacement field itself) must be applied to the final composed/updated displacement fields at the end of each iteration, regardless of the optimizer.
3. **Closure and Line Searches for L-BFGS**:
   * PyTorch's native `torch.optim.LBFGS` is highly optimized and includes strong Wolfe line search functionality. To integrate it, we can create an L-BFGS optimizer instance and define a `closure()` that performs image warping, computes loss, runs backpropagation, and optionally filters the gradients.
   * After the L-BFGS optimizer takes a step (`optimizer.step(closure)`), we can enforce boundary masks, apply elastic smoothing, and project back to the invertible space using `update_inverse_field_nd`.
4. **JAX Backend Optimizers**:
   * Since `optax` and `jaxopt` are not project dependencies, we can implement standard SGD (with optional momentum) and Adam update formulas directly as JAX `jit`-compiled functions, preserving JAX execution speeds.
   * For JAX L-BFGS, we can bridge JAX with standard `scipy.optimize.minimize(..., method='L-BFGS-B', jac=True)`. We define a CPU-compatible wrapper function that flattens JAX parameters, runs JAX autograd (`value_and_grad` on the midpoint loss), filters gradients, and returns CPU scalar losses and flat NumPy arrays for Scipy.
5. **Parity and Sweep Validation**:
   * Incorporating sweeps into `examples/run_benchmarks.py` will allow direct benchmarking of `sgd`, `adam`, and `lbfgs` configurations against the standard step-based `cfl` updates.
   * Standard ANTs baselines in `test_challenger_verification.py` can be used to verify that the integrated optimizers do not regress the registration performance and maintain parity within 1%.

## 3. Caveats
* Using Scipy's L-BFGS-B in JAX transfers parameters and gradients between CPU memory and GPU/TPU devices, which might add a small data-transfer overhead compared to native JAX-JIT loops. However, given that SyN optimization loops run for a limited number of iterations (e.g. 10 to 100), this overhead is negligible.
* L-BFGS updates parameters directly and might violate invertibility if the step size is too large. Consequently, applying boundary enforcement and inversion projection at the end of each line-search step is critical to regularize the deformation.

## 4. Conclusion
We have designed a robust integration plan for Adam, SGD, and L-BFGS optimizers in both PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`) backends, supporting both additive and compositional updates. This plan:
1. Customizes gradient smoothing and parameter composition.
2. Integrates PyTorch's native `LBFGS` optimizer with custom closures.
3. Integrates Scipy's `minimize` for JAX-based L-BFGS.
4. Formulates sweeps in `run_benchmarks.py` to compare optimizers across deep feature spaces and standard metrics while complying with all guardrails.

## 5. Verification Method
### Independent Verification Steps:
1. **Linting and compilation check**: Run `pytest tests/test_syn.py` and `pytest tests/test_syn_jax.py` to ensure current tests pass.
2. **Optimizer functional verification**: Propose adding a new unit test suite `tests/test_optimizers.py` verifying that:
   * Adam, SGD, and L-BFGS run without throwing exceptions in both PyTorch and JAX.
   * Registration losses decrease over iterations under all optimizers.
   * Minimum Jacobian values remain within valid limits (>= -1e-5), ensuring invertibility is preserved.
3. **DICE score parity**: Verify that registration results using the new optimizers achieve Dice scores matching or exceeding the baseline step-based SyN updates within 1% (e.g., using `pytest tests/test_challenger_verification.py`).
