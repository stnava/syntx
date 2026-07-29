# Handoff Report: TVF Velocity Gradient Smoothing Fix, GEMINI.md Rule 8 Compliance, and PyTorch/JAX Parity

## 1. Observation
- **Code Changes Made**:
  1. `src/syntx/tvf.py` (lines 386–396):
     - Removed permuted channel-first layout (`grad.permute(...)`) in `TVFModel.fit()`. `self.velocity.grad[t]` of shape `(1, *velocity_shape, dim)` (channel-last format) is passed directly to `separable_gaussian_filter(self.velocity.grad[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel')`.
     - Updated `grid_sample_nd` calls in `forward()` (lines 257, 278) and `fit()` (line 344) to explicitly set `padding_mode='zeros'` for intensity image warping, strictly complying with GEMINI.md Rule 8.
  2. `src/syntx/tvf_jax.py`:
     - Implemented `TVFModelJAX` mirroring PyTorch `TVFModel` algorithmically.
     - Uses RK4/Euler continuous keyframe velocity ODE integration (`integrate()`) with `padding_mode='border'` for displacement field sampling.
     - Uses `local_ncc_loss_nd_jax` and `jax_grid_sample` with `padding_mode='zeros'` for midpoint-symmetric intensity image LNCC loss (`forward()` and `fit()`), strictly complying with GEMINI.md Rule 8.
     - Uses `jax.grad` and `separable_gaussian_filter_jax` for multi-resolution pyramid optimization and fluid regularization gradient smoothing (`fit()`).
  3. `src/syntx/syn_jax.py`:
     - Updated `box_filter_jax` to divide box sums by unpadded element counts (`out_sum / jnp.maximum(out_count, 1e-5)`), matching PyTorch `avg_pool` behavior with `count_include_pad=False` on zero-padded intensity images.
  4. `src/syntx/__init__.py`:
     - Imported and exported `TVFModel` and `TVFModelJAX` in `__all__`.
  5. `tests/test_tvf.py`:
     - Expanded test suite with:
       - `test_tvf_model_2d_forward_and_warp`
       - `test_tvf_model_3d_forward_and_warp`
       - `test_tvf_velocity_gradient_smoothing_isotropic`: Verifies 3D impulse in $V_z$ does not leak into $V_y$ or $V_x$ ($V_y = 0.0, V_x = 0.0$), and smooths isotropically across Z, Y, and X axes ($val_z == val_y == val_x > 0.0$) in both PyTorch and JAX.
       - `test_tvf_model_fit_2d_and_3d`: Verifies multi-res `fit()` optimization reduces LNCC loss in 2D/3D for both PyTorch and JAX.
       - `test_tvf_pytorch_jax_parity`: Verifies forward LNCC loss match ($\le 0.001$), forward displacement warp match ($\le 0.001$), and inverse displacement warp match ($\le 0.001$) between PyTorch and JAX.

- **Test Execution & Outcome Commands**:
  - `pytest tests/test_tvf.py`: 5 passed in 79.23s. Coverage: `src/syntx/tvf.py` 83%, `src/syntx/tvf_jax.py` 73%.
  - `pytest`: Full repository test suite passed (29 passed in 241.05s).

## 2. Logic Chain
1. **Diagnosis & Solution for Channel Permutation Bug**:
   - `separable_gaussian_filter` in `syn.py` and `syn_jax.py` requires channel-last layout `(B, *spatial, dim)`.
   - `self.velocity.grad[t]` is shape `(1, *velocity_shape, dim)`.
   - Permuting to `(1, 3, D, H, W)` caused `separable_gaussian_filter` to interpret `dim=3` as spatial dimension 0 ($Z$) and $W$ as channel count, zeroing spatial smoothing along $W$ and coupling velocity components $V_z \rightarrow V_y, V_x$.
   - Passing `self.velocity.grad[t]` directly as channel-last `(1, *velocity_shape, dim)` restores correct per-component isotropic Gaussian filtering.

2. **GEMINI.md Rule 8 Compliance & Unpadded Box Filter Parity**:
   - GEMINI.md Rule 8 mandates `padding_mode='zeros'` for intensity images and `padding_mode='border'` for displacement fields.
   - Setting `padding_mode='zeros'` explicitly in both `tvf.py` and `tvf_jax.py` prevents artificial edge-color stripes when warping images.
   - Updating `box_filter_jax` in `syn_jax.py` to divide zero-padded box sums by unpadded element counts aligns JAX LNCC loss calculation with PyTorch's `avg_pool` (`count_include_pad=False`).
   - Parity verification confirms forward loss difference $\approx 9.3 \times 10^{-10}$ and displacement warp max absolute difference $\approx 1.9 \times 10^{-6}$, far exceeding the $\le 0.001$ tolerance requirement.

## 3. Caveats
- No caveats. All 5 TVF unit tests pass cleanly and all 29 tests in the syntx repository pass.

## 4. Conclusion
- PyTorch `TVFModel.fit()` velocity gradient smoothing fix complete and verified.
- `TVFModelJAX` implemented and fully verified against PyTorch backend with strict GEMINI.md Rule 8 compliance.
- `TVFModel` and `TVFModelJAX` exported in `syntx.__init__`.
- All tests in `tests/test_tvf.py` and across the entire codebase pass.

## 5. Verification Method
1. Run `pytest tests/test_tvf.py`
2. Run `pytest`
