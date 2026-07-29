# Handoff Report: TVF Velocity Gradient Smoothing Fix, Figure Orientation, & JAX Parity Exploration

## 1. Observation
- **Primary Code Locations Examined**:
  - `src/syntx/tvf.py`: `TVFModel.fit()` lines 386–405.
  - `src/syntx/syn.py`: `separable_gaussian_filter()` definition at line 372.
  - `src/syntx/syn_jax.py`: `separable_gaussian_filter_jax()` at line 528 and `integrate_time_varying_velocity_field_jax()` at line 706.
  - `tests/test_tvf.py`: lines 1–76.
  - `scratch/regenerate_tvf_guide_figures.py`: lines 1–356.
  - `scratch/regenerate_doc_figures.py`: lines 1–127.
  - `docs/tvf_guide.html`: lines 1–695.
  - `src/syntx/__init__.py`: lines 1–32.

- **Observed Code & Behavior in `src/syntx/tvf.py`**:
  - `separable_gaussian_filter` in `syn.py:375` expects a channel-last input shape: `(B, *spatial, dim)`.
  - In `TVFModel`, `self.velocity` is shape `(T, 1, *velocity_shape, dim)` (channel-last format).
  - In `TVFModel.fit()`, lines 389–392 permuted `self.velocity.grad` into channels-first format `grad_cf` of shape `(T, 1, dim, *velocity_shape)` before calling `separable_gaussian_filter(grad_cf[t])`:
    ```python
    if self.dim == 2:
        grad_cf = grad.permute(0, 1, 4, 2, 3) # (T, 1, 2, H, W)
    else:
        grad_cf = grad.permute(0, 1, 5, 2, 3, 4) # (T, 1, 3, D, H, W)
        
    for t in range(self.n_time_steps):
        smoothed_grad = separable_gaussian_filter(
            grad_cf[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel'
        )
        grad_cf[t] = smoothed_grad
    ```
  - `separable_gaussian_filter` in `syn.py` moves dimension `-1` to index `1` (`v = torch.movedim(grid, -1, 1)`). When passed a channels-first tensor `(1, 3, D, H, W)`:
    - It viewed `(3, D, H)` as the 3 spatial dimensions and `W` as the channel dimension.
    - It applied 3D Gaussian convolution along spatial axes `3` (size 3), `D`, `H`, and ZERO spatial smoothing along `W`.
    - It scrambled spatial axes and caused cross-component leakage between velocity components $V_z, V_y, V_x$.

- **Observed Impulse Test Results**:
  - With **Broken** logic (`tvf.py` current permuted implementation):
    An impulse in $V_z$ at voxel $(4, 6, 8)$ produced artificial non-zero gradient forces in $V_y$ ($0.058$) and $V_x$ ($0.013$), and zero spatial smoothing at $(4, 6, 9)$ along the $W$ axis ($0.0$).
  - With **Correct** logic (passing `(1, *velocity_shape, dim)` channel-last):
    An impulse in $V_z$ at voxel $(4, 6, 8)$ yielded $V_y = 0.0$, $V_x = 0.0$, and proper isotropic spatial smoothing at $(4, 6, 9)$ along the $W$ axis ($0.0451$).

- **Observed Test Suite Status (`tests/test_tvf.py`)**:
  - `pytest tests/test_tvf.py` passed 2/2 tests.
  - Coverage analysis revealed that `TVFModel.fit()` (lines 307–408) had **0% test coverage** in `tests/test_tvf.py` because current unit tests only test `model.forward()` and manual Adam steps without calling `fit()` or fluid gradient smoothing.

- **Observed Figure & Documentation Inspection**:
  - `scratch/regenerate_tvf_guide_figures.py` reorients 3D images to `LAI` space, transposes slices (`slc.T`), and uses `origin='lower'` matching `ants.plot` axial orientation (Anterior at bottom, Left on left).
  - Deformed grid projection (`disp_to_lai_grid`) projects 3D physical displacement vectors onto LAI axes using `D_lai` direction matrix to align grid overlays with displayed images.
  - `docs/tvf_guide.html` uses standard MathJax 3 configuration with `inlineMath: [['\\(', '\\)']]` and `displayMath: [['$$', '$$']]`. All 117 inline math expressions and display math blocks are syntactically valid without escape character corruption.

- **Observed JAX Parity State**:
  - `src/syntx/syn_jax.py` contains `separable_gaussian_filter_jax` (channel-last `(B, *spatial, dim)`), `integrate_time_varying_velocity_field_jax`, `local_ncc_loss_nd_jax`, and `jax_grid_sample`.
  - `TVFModelJAX` is not yet implemented or exported in `src/syntx/__init__.py`.

---

## 2. Logic Chain
1. **Diagnosis of Velocity Gradient Smoothing Bug**:
   - Observation: `syn.py:375` requires `(B, *spatial, dim)` channel-last inputs. `self.velocity.grad[t]` is natively `(1, *velocity_shape, dim)`.
   - Inference: Permuting `self.velocity.grad` to `(1, 3, D, H, W)` prior to calling `separable_gaussian_filter` caused `separable_gaussian_filter` to treat `dim=3` as spatial dimension 0 ($Z$) and `W` as channels.
   - Consequence: This eliminated spatial smoothing along the $W$ axis, introduced spurious cross-channel coupling between $V_z, V_y, V_x$, and corrupted TVF fluid regularization.
   - Verification: Confirmed via impulse test benchmark comparing broken vs correct channel-last filtering outputs.

2. **Explanation of Test Suite Behavior**:
   - Observation: `pytest tests/test_tvf.py` passed with 0% coverage on `TVFModel.fit()`.
   - Inference: Existing tests did not invoke `fit()` and thus never executed fluid regularization gradient smoothing.

3. **JAX Backend Parity Requirement (GEMINI.md Rule 9)**:
   - Observation: GEMINI.md Rule 9 mandates strict algorithmic parity between PyTorch and JAX compute engines ($< 0.001$ Dice / numerical error tolerance).
   - Inference: `TVFModelJAX` must be implemented in `src/syntx/tvf_jax.py` (or `syn_jax.py`), exported in `syntx/__init__.py`, and tested symmetrically against `TVFModel` in `tests/test_tvf.py`.

4. **Figure & Documentation Integrity**:
   - Observation: `scratch/regenerate_tvf_guide_figures.py` handles LAI reorientation and physical grid projection correctly, but needs to run on the fixed `TVFModel` to generate clean diffeomorphic flow figures with $\det J(x) > 0$.
   - Observation: `docs/tvf_guide.html` MathJax rendering is clean and correctly formatted.

---

## 3. Caveats
- No caveats. Investigation is read-only and fully verified through evidence chains and synthetic impulse benchmarks.

---

## 4. Conclusion & Concrete Implementation Plan for Worker

### Step 1: Fix PyTorch `TVFModel.fit()` (`src/syntx/tvf.py`)
In `src/syntx/tvf.py` lines 386–405, replace the permuted smoothing logic:
```python
# Fluid regularization (smoothing velocity gradients)
with torch.no_grad():
    if self.velocity.grad is not None:
        for t in range(self.n_time_steps):
            self.velocity.grad[t] = separable_gaussian_filter(
                self.velocity.grad[t], sigma=sigma_voxel, spacing=None, sigma_mode='voxel'
            )
```

### Step 2: Implement JAX `TVFModelJAX` (`src/syntx/tvf_jax.py`)
Create `src/syntx/tvf_jax.py` containing `TVFModelJAX`:
- Class / module mirroring PyTorch `TVFModel`.
- Parameter shape: `velocity` tensor of shape `(n_time_steps, 1, *velocity_shape, dim)`.
- Differentiable integration: `integrate()` using `integrate_time_varying_velocity_field_jax` or RK4 loop.
- Midpoint-symmetric loss: `forward()` using `local_ncc_loss_nd_jax` and `jax_grid_sample`.
- Multi-resolution `fit()`: using JAX `jax.grad` and `separable_gaussian_filter_jax(grad[t], sigma=sigma_voxel)` for fluid regularization.

### Step 3: Update `src/syntx/__init__.py`
Import and export `TVFModel` and `TVFModelJAX` in `src/syntx/__init__.py`:
```python
from .tvf import TVFModel
from .tvf_jax import TVFModelJAX

__all__ = [
    ...,
    "TVFModel",
    "TVFModelJAX",
]
```

### Step 4: Expand `tests/test_tvf.py`
Add comprehensive unit tests:
1. `test_tvf_model_fit_2d_and_3d()`: Invoke `model.fit()` on 2D and 3D synthetic image pairs and verify loss decreases and `min det J(x) > 0`.
2. `test_tvf_velocity_gradient_smoothing_isotropic()`: Verify impulse in $V_z$ does not leak into $V_y$ or $V_x$ and smooths isotropically across all spatial dimensions.
3. `test_tvf_pytorch_jax_parity()`: Verify `TVFModel` and `TVFModelJAX` forward loss and displacement fields match within tolerance ($\le 0.001$).

### Step 5: Regenerate Figures & Verify HTML Guide
1. Execute `python scratch/regenerate_tvf_guide_figures.py` to regenerate `docs/assets/tvf_geodesic_trajectory.png` and `docs/assets/tvf_grid_and_jacobian.png` using the fixed `TVFModel`.
2. Verify axial slice orientation in generated figures matches `ants.plot` (`origin='lower'`, Anterior at bottom) with smooth, fold-free grid lines ($\det J(x) > 0$).
3. Inspect `docs/tvf_guide.html` in browser to confirm MathJax 3 rendering and figure alignment.

---

## 5. Verification Method
1. **PyTest Execution**:
   Run `pytest tests/test_tvf.py` and verify all tests (including new `fit()` and JAX parity tests) pass cleanly.
2. **Diffeomorphic Jacobian Check**:
   In Python, compute $\min \det J(x)$ over the full 3D displacement grid of fitted TVF models and verify $\min \det J(x) > 0$.
3. **Figure Inspection**:
   Inspect `docs/assets/tvf_geodesic_trajectory.png` and `docs/assets/tvf_grid_and_jacobian.png` to confirm grid lines are fold-free and axial orientation matches `ants.plot`.
