# TVF Implementation Technical Review

**File:** `src/syntx/tvf.py` (1990 lines)  
**Date:** 2026-08-15  
**Scope:** Deep correctness audit — bugs, logic errors, and concrete improvement recommendations.

---

## 1. Confirmed Bugs

### 1.1 `cfl_max_val` Variable Shadowing (Line 1424)

**Severity: Medium — silently changes behavior depending on call path.**

At line 1044, `cfl_max_val` is computed once from the constructor default:

```python
cfl_max_val = float(kwargs.get('cfl_max', self.cfl_max if self.cfl_max is not None else 0.0))
```

But inside the per-epoch loop (line 1424), the variable is **re-assigned** from `kwargs` with a different default:

```python
cfl_max_val = kwargs.get('cfl_max', None)
if cfl_max_val is not None and float(cfl_max_val) > 0:
```

If `cfl_max` is NOT in kwargs (the common case when calling from `tvf_registration`, which does not pass `cfl_max` through `**kwargs` to `fit()`), the outer assignment evaluates to `self.cfl_max = 0.40`, but the inner assignment evaluates to `None`. This means the velocity magnitude clamping that the outer code expects to be active is silently **disabled** every epoch.

**Fix:** Delete the inner re-assignment at line 1424. Use the outer `cfl_max_val` computed at line 1044.

---

### 1.2 LARS `p_norm` / `g_norm` Double Computation (Lines 141–145)

**Severity: Low — wasted compute, no correctness impact.**

```python
p_norm = torch.norm(p)
g_norm = torch.norm(g)

p_norm = torch.norm(p)  # exact duplicate
g_norm = torch.norm(g)  # exact duplicate
```

Lines 141–142 compute `p_norm` and `g_norm`, then lines 144–145 compute them again identically. This is dead code duplication from a copy-paste.

**Fix:** Delete lines 144–145.

---

### 1.3 Duplicate Imports (Lines 24–38)

**Severity: Low — no runtime effect, code hygiene.**

The following symbols are imported twice:
- `get_physical_grid_torch` (lines 25, 29)
- `physical_to_normalized_torch_cached` (lines 26, 30)
- `grid_to_physical_affine_torch` (lines 27, 33)
- `grid_sample_nd` (lines 28, 36)

**Fix:** Remove the duplicate entries.

---

### 1.4 Analytical Gradient Collapses TVF to SVF (Lines 1247–1265)

**Severity: High — mathematically incorrect for time-varying fields.**

When `use_analytical_gradients=True`, the code computes the spatial similarity gradient at the midpoint $t=0.5$, then distributes it uniformly:

```python
for t in range(self.n_time_steps):
    self.velocity.grad[t, 0] = combined_grad[0]
```

Every keyframe $v(x, t_k)$ receives the **identical** gradient. This forces all velocity keyframes to evolve identically, collapsing the time-varying parameterization into a stationary velocity field. The correct TVF analytical gradient requires pulling the similarity gradient back through the flow Jacobian $\partial\phi_{t_k\to 0.5}/\partial v(t_k)$ independently for each keyframe — a computation that autograd handles correctly but analytical mode does not.

**Impact:** Any run with `use_analytical_gradients=True` will produce a degenerate TVF where $v(x, t_0) = v(x, t_1) = \ldots = v(x, t_{T-1})$, yielding results equivalent to (but slower than) standard SyN.

**Recommendation:** Either (a) remove analytical gradient support from TVF entirely, or (b) add a guard that raises `NotImplementedError("Analytical gradients are not supported for TVF with n_time_steps > 1")`.

---

### 1.5 `device_str` Unbound When `backend='jax'` (Lines 1913–1918)

**Severity: Medium — runtime crash on JAX backend.**

After the `if backend == 'pytorch': ... elif backend == 'jax': ...` block, the cleanup code references `device_str`:

```python
if device_str == 'mps':
    torch.mps.synchronize()
```

When `backend='jax'`, `device_str` is set to `'cpu'` (line 1801), so this specific line won't crash. However, at lines 1923–1924:

```python
W_fwd_tensor = torch.from_numpy(fwd_np).to(device_str).float()
W_inv_tensor = torch.from_numpy(inv_np).to(device_str).float()
```

This sends tensors to `'cpu'` (correct) but then calls `calculate_inverse_identity_error` using PyTorch on what should be a JAX-only path. While this happens to work because fwd/inv are already numpy, it is fragile and semantically incorrect.

**Fix:** Guard the PyTorch cleanup and inverse error computation with `if backend == 'pytorch':`.

---

## 2. Correctness Concerns

### 2.1 `curr_spacing` in `fit()` Uses XYZ, But Velocity Grid Is ZYX

In `fit()` at line 1091:

```python
curr_spacing = [sp * level for sp in self.spacing]
```

`self.spacing` is in XYZ order (set directly from `fixed.spacing` at line 1727 via `spacing=spacing`). But `self.image_shape` is in ZYX order (set from `grid_shape_zyx` at line 1724). This means the CFL step normalization at line 1358–1360:

```python
sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
grad_voxel = grad / sp_t  # convert to voxel units
```

divides the gradient's ZYX-ordered last dimension by XYZ-ordered spacing. For isotropic images (1mm³), this produces correct results. For anisotropic images (e.g., 1×1×3mm), the Z and X spacing components are swapped, producing incorrect voxel-space normalization.

**Impact:** CFL step size computation is wrong for anisotropic images. The velocity magnitude is over- or under-estimated along the Z vs X axes, which can cause excessive folding along one axis.

**Recommendation:** Either reverse `self.spacing` when computing `curr_spacing` in `fit()`, or explicitly document and enforce the convention. The `integrate()` method handles this correctly via its `reversed()` calls, but `fit()` does not.

---

### 2.2 `shrink_ratio` Uses Only First Axis (Line 1083)

```python
shrink_ratio = float(curr_vel_shape[0]) / float(max_vel_shape[0])
```

This computes the ratio using only the first spatial dimension. For anisotropic velocity grids where axes are shrunk differently (e.g., `max_vel_shape=(96,96,48)` → `curr_vel_shape=(48,48,24)`), the ratio should be the geometric mean across all axes. Using axis 0 alone produces correct results only when all axes have equal resolution.

**Impact:** The `math.sqrt(shrink_ratio)` scaling applied to the CFL step and LARS learning rate at coarse levels is off for anisotropic grids, causing over- or under-stepping.

**Recommendation:** Use `shrink_ratio = math.prod(curr_vel_shape) / math.prod(max_vel_shape)` and take the cube root (3D) or square root (2D).

---

### 2.3 Convergence Check Samples at Irregular Intervals

At line 1459:
```python
if epoch % 5 == 0 or epoch == epochs - 1:
    recent_losses.append(loss_val)
```

The `recent_losses` list is only appended every 5 epochs, but the linear regression at line 1468 uses `np.arange(convergence_window)` as the x-axis (i.e., equally spaced integers 0, 1, 2, ...). This is mathematically correct — the slope of the loss sampled every 5th epoch still indicates convergence. No bug here, but the slope threshold `convergence_threshold = 1e-6` should be understood as "per sample" not "per epoch."

---

### 2.4 `interp_mode` Variable Unused (Line 1046)

```python
interp_mode = 'trilinear' if self.dim == 3 else 'bilinear'
```

This variable is computed but never referenced anywhere in `fit()`. The actual interpolation mode selection happens locally inside the fast_smooth block (line 1297) and in `_resize_velocity`.

**Fix:** Remove the dead variable.

---

## 3. Performance Recommendations

### 3.1 Redundant `import math` and `import gc` Inside Loop

`import math` appears at line 22 (top-level), then again at lines 1095 and 1363 inside the hot loop. `import gc` appears at lines 1480 and 1491. While Python caches module imports, the `import` statement still has overhead from the module lookup and the `sys.modules` dict access on every iteration.

**Fix:** Remove all inner `import math` and `import gc` statements; use the top-level imports.

---

### 3.2 `_create_boundary_mask` Called Twice Per Epoch (Lines 1306, 1344)

In the gradient smoothing block, `_create_boundary_mask` is called at line 1306 for `bmask_pre` (on the possibly-downsampled shape), and again at line 1344 for the full-resolution taper. The full-resolution mask is cached, but the downsampled one is not.

**Fix:** Cache both masks using the existing `_bmask_cache` pattern with distinct keys.

---

### 3.3 `torch.tensor()` Allocation Every Epoch for `sp_t` (Lines 1358, 1426)

Inside the per-epoch CFL block:
```python
sp_t = torch.tensor(curr_spacing, device=device, dtype=dtype)
```

This allocates a new small tensor every epoch. Since `curr_spacing` is constant within a pyramid level, this should be hoisted outside the epoch loop.

**Fix:** Compute `sp_t` once per pyramid level, alongside `curr_spacing`.

---

### 3.4 Antisymmetric Mode Does Not Mirror Velocity

The docstring at line 188 says:
> `antisymmetric : bool — Enforce anti-symmetry v(t_k) = -v(t_{K-1-k})`

But the actual implementation (line 838) only ensures both `t=0` and `t=1` are included in `eval_points`. There is no code that actually constrains or mirrors the velocity keyframes. This is not a bug per se — the docstring describes a possible future feature — but the documentation is misleading.

**Recommendation:** Either implement the velocity mirroring constraint, or update the docstring to accurately describe the current behavior (symmetric evaluation points, not anti-symmetric velocity fields).

---

## 4. Robustness Recommendations

### 4.1 `affine_inv_file` Written But Never Used (Line 1889)

```python
affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
...
ants.write_transform(tx_inv, affine_inv_file)
```

The inverse affine transform is written to disk but `affine_inv_file` is never included in `inv_transforms` or anywhere else. The `whichtoinvert_inv = [True, False]` mechanism re-inverts `affine_file` at apply time instead.

**Impact:** Orphaned temp file accumulates on disk across runs.

**Fix:** Remove the `affine_inv_file` creation and the `ants.write_transform(tx_inv, affine_inv_file)` call, or include it in the output dict for callers who need it.

---

### 4.2 No Gradient Clipping Despite `image_grad_clip` Parameter

The constructor accepts `image_grad_clip=6.0` (line 209) and stores it (line 222), but this value is never used anywhere in the codebase. Similarly, `velocity_clamp` is stored but never applied.

**Fix:** Either implement gradient clipping using the parameter, or remove it from the constructor signature.

---

## 5. Summary

| # | Issue | Severity | Type |
|---|-------|----------|------|
| 1.1 | `cfl_max_val` shadowed — velocity clamping silently disabled | Medium | Bug |
| 1.2 | LARS double `p_norm`/`g_norm` computation | Low | Bug |
| 1.3 | Duplicate imports | Low | Hygiene |
| 1.4 | Analytical gradient collapses TVF to SVF | High | Bug |
| 1.5 | `device_str` scoping fragile on JAX path | Medium | Bug |
| 2.1 | XYZ/ZYX spacing mismatch in CFL norm (anisotropic) | Medium | Correctness |
| 2.2 | `shrink_ratio` single-axis only | Low | Correctness |
| 2.4 | `interp_mode` unused variable | Low | Dead code |
| 3.1 | Redundant in-loop imports | Low | Performance |
| 3.2 | Uncached boundary mask at downsampled resolution | Low | Performance |
| 3.3 | Per-epoch tensor allocation for `sp_t` | Low | Performance |
| 3.4 | Antisymmetric docstring misleading | Low | Documentation |
| 4.1 | `affine_inv_file` written but never used | Low | Leak |
| 4.2 | `image_grad_clip` / `velocity_clamp` never used | Low | Dead code |

**Priority fixes:** Items 1.1, 1.4, and 2.1 should be addressed before any production benchmark run on anisotropic data.
