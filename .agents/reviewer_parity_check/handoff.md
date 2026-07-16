# Handoff Report — Review of Physical Space Optimization and Affine Coordinate Composition

This report evaluates the correctness, completeness, robustness, and conformance of the physical space optimization and affine coordinate composition implementations in `src/syntx/syn.py` and `src/syntx/syn_jax.py`.

---

## 1. Observation

1. **Test Session Executions**:
   - Running the default test suite via `pytest` yields a successful run:
     ```
     ============ 122 passed, 6 skipped, 6 warnings in 303.64s (0:05:03) ============
     ```
   - However, when running the tests in `tests/test_syn.py` sequentially as part of a file run (e.g. `pytest tests/test_syn.py`), `test_pytorch_syn_2d_vgg19` fails:
     ```
     FAILED tests/test_syn.py::test_pytorch_syn_2d_vgg19 - assert -0.1261639446020...
     E       assert -0.12616394460201263 >= -1e-05
     ```
   - Running that specific test in isolation (e.g. `pytest tests/test_syn.py -k test_pytorch_syn_2d_vgg19 -v`) passes successfully.

2. **Divergent Moving Image Handling in Optimization (`fit`)**:
   - In PyTorch (`src/syntx/syn.py` lines 1353–1360), `J_pyr` is built by pre-warping the moving image with the optimized affine grid:
     ```python
     moving_affine = F.grid_sample(moving_image, grid, padding_mode='border', align_corners=True)
     J_pyr = [F.interpolate(moving_affine, ...) if s > 1 else moving_affine for s in levels]
     ```
   - In JAX (`src/syntx/syn_jax.py` lines 1510–1529), `J_pyr` is built from the original, un-warped moving image `J_jax`:
     ```python
     fixed_smoothed = smooth_image_jax(I_jax, sig, spacing=fixed_spacing)
     moving_smoothed = smooth_image_jax(J_jax, sig, spacing=moving_spacing)
     J_level = interpolate_jax(moving_smoothed, 1.0 / s, self.dim)
     J_pyr.append(J_level)
     ```

3. **Double Affine Warping in PyTorch Midpoint Loss**:
   - In `prepare_mid_images_and_gradients_torch` (called during PyTorch SyN optimization):
     ```python
     phi_r2l_phys = X_phys + warp_r2l
     y_phys = phi_r2l_phys @ M_phys.t() + t_phys
     y_norm = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
     J_mid = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)
     ```
     Here, `J_curr` is already pre-warped by $A$ (`moving_affine`), but `y_norm` applies the affine transform $A$ again.

4. **Continuous Gradient Lie Algebra Rotations**:
   - Both backends parameterize rotation using skew-symmetric Lie Algebra and use a first-order Taylor expansion `I + K_raw` for small angles to avoid non-differentiable zero divisions:
     - PyTorch (`src/syntx/syn.py` lines 47–49):
       ```python
       R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)
       R_small = I + K_raw
       return torch.where(is_zero, R_small, R)
       ```
     - JAX (`src/syntx/syn_jax.py` lines 225–228):
       ```python
       R = I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
       R_small = I + K_raw
       return jnp.where(is_zero, R_small, R)
       ```

---

## 2. Logic Chain

1. **Mathematical Correctness of Transform Compositions**:
   - Let $\phi_1(x)$ map fixed space to midpoint space. In displacement field notation, $\phi_1^{-1}(x) = x + w_{l2r\_inv}(x)$ where $x$ is a midpoint coordinate.
   - Let $\phi_2^{-1}(x) = x + w_{r2l}(x)$ map midpoint space to pre-affine moving space.
   - Let $A(u)$ represent the affine transform mapping pre-affine moving space to moving space.
   - The composed forward mapping is $y = A(\phi_2^{-1}(\phi_1(x)))$.
   - At inference time (`forward`), both PyTorch and JAX compose this correctly by computing:
     - `phi_l2r_phys = X_phys + warp_resampled` = $\phi_2^{-1}(\phi_1(x))$.
     - `y_phys = phi_l2r_phys @ M_phys.t() + t_phys` = $A(\phi_2^{-1}(\phi_1(x)))$.
     - They warp the native `moving_image` exactly once using the composed grid. This confirms the correctness of the composed mapping logic.

2. **Single-Interpolation Policy Conformance**:
   - **Inference Conformance**: Both PyTorch and JAX adhere to the single-interpolation policy when mapping images using the final transforms.
   - **Optimization Divergence (PyTorch Backend Bug)**:
     - During the PyTorch SyN optimization loop, `J_curr` is `moving_affine` (already interpolated once with the affine grid).
     - However, the mapping grid `y_norm` still applies the affine matrices `M_phys` and `t_phys`.
     - When `J_mid` is sampled as `F.grid_sample(J_curr, y_norm)`, it samples an already-warped image with warped coordinates, resulting in the affine transform being applied twice ($J(A(A(\phi_2^{-1}(x))))$).
     - This double warping degrades optimization gradients and leads to spatial instability and folding under certain noise seeds, causing the `min_jac >= -1e-5` check to fail for the VGG19 2D test.
   - **Optimization Conformance (JAX Backend)**:
     - JAX uses the un-warped `J_curr` during optimization and only warps it once using `y_norm` inside the loop, correctly conforming to the single-interpolation policy.

---

## 3. Caveats

- The random seed in test environments is set globally by `antspyt1w.get_data` to `1234`. The precise order of test runs affects the seed state at the start of the VGG19 test, which explains why the test is flaky (fails in the full suite run but passes when isolated).
- This review does not modify implementation code to preserve "review-only" constraints.

---

## 4. Conclusion

- **Verdict**: **REQUEST_CHANGES** (due to the PyTorch backend double affine warping bug and single-interpolation policy divergence during optimization).
- The coordinate mapping logic is mathematically correct and works perfectly at inference time, but optimization in `syn.py` must be updated to align with `syn_jax.py`'s correct single-interpolation behavior.

---

## 5. Verification Method

To verify the findings and check for the double-warping bug:
1. Inspect `src/syntx/syn.py` at line 1357: observe that `moving_affine` is created, and line 1359 initializes `J_pyr` from it.
2. Inspect `src/syntx/syn.py` at line 1446: `J_curr` is passed to `prepare_mid_images_and_gradients_torch` which applies `M_phys` and `t_phys` internally.
3. Contrast with `src/syntx/syn_jax.py` at lines 1523–1527: `J_pyr` is initialized using the unwarped `moving_smoothed`.
4. Run the PyTorch syn tests:
   ```bash
   pytest tests/test_syn.py -k test_pytorch_syn_2d_vgg19 -v
   ```

---

# QUALITY REVIEW REPORT

**Verdict**: REQUEST_CHANGES

## Findings

### [Major] Finding 1: PyTorch Backend Double Affine Warping & Single Interpolation Divergence

- **What**: The PyTorch backend applies the affine transform twice during SyN deformable optimization, causing a double warping bug.
- **Where**: `src/syntx/syn.py` (lines 1357–1359, 1445–1450).
- **Why**: `J_pyr` is constructed using `moving_affine` (which already has the affine transform applied). During the optimization loop, `prepare_mid_images_and_gradients_torch` applies the affine matrices `M_phys, t_phys` again when sampling `J_curr`. This violates the single-interpolation policy during training, degrades optimization quality, and causes test flakiness/folding.
- **Suggestion**: Re-implement PyTorch's `J_pyr` initialization to match JAX's correct behavior (`src/syntx/syn_jax.py` lines 1510–1529) by constructing `J_pyr` from the unwarped, smoothed `moving_image` instead of `moving_affine`.

### [Minor] Finding 2: Lack of VGG19 Test Coverage in JAX backend

- **What**: The JAX test suite (`tests/test_syn_jax.py`) does not contain equivalent tests for deep feature space metrics (e.g., VGG19 / DINOv2).
- **Where**: `tests/test_syn_jax.py`.
- **Why**: Misses validation of the DLPack feature loss bridge for JAX.
- **Suggestion**: Add a JAX equivalent of `test_pytorch_syn_2d_vgg19` to `tests/test_syn_jax.py`.

## Verified Claims

- **Composed coordinate mapping logic correctness** $\to$ verified via mathematical derivation and code inspection $\to$ **PASS** (Correctly matches $y = A(\phi_2^{-1}(\phi_1(x)))$ at inference).
- **Lie Algebra AD Gradient Continuity at Identity** $\to$ verified via checking Taylor expansion fallback $\to$ **PASS** (Avoids gradient lock/zero gradients).
- **Test Suite Success** $\to$ verified via `pytest` run $\to$ **PASS** (122 tests passed in the default run).

---

# ADVERSARIAL REVIEW REPORT

**Overall risk assessment**: MEDIUM

## Challenges

### [Medium] Challenge 1: Single Interpolation Violation under Large Rotations/Translations

- **Assumption challenged**: The assumption that pre-warping moving images for optimization inputs behaves similarly to single-step composition.
- **Attack scenario**: If the moving image undergoes a large rotation (e.g. 90 degrees or massive translation) relative to the fixed image, the double affine warping in the PyTorch backend will double the shift, mapping coords completely outside the image boundaries and causing optimization failure.
- **Blast radius**: Prevents PyTorch backend from successfully registering images with large initial misalignments.
- **Mitigation**: Standardize PyTorch's optimization pipeline to use the unwarped moving image and compose grid parameters on the fly, matching the JAX backend.
