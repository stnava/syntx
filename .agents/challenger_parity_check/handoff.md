# Verification Report & Handoff

## 1. Observation

### Coordinate-Mapping DICE Score Verification
- Run `python scratch/test_internal_dice.py`:
  - Command: `python scratch/test_internal_dice.py`
  - Output:
    ```
    Computed DICE between ANTs and PyTorch physical warped grids: 1.000000
    Physical coordinate mapping verification PASSED!
    ```

### Unit Test Suite Execution
- Run `pytest`:
  - Command: `pytest`
  - Output: `122 passed, 6 skipped, 6 warnings in 301.66s`
  - Verification: Standard unit test suite passes fully on Darwin/macOS.

### 2D Parity and 3D Registration Verification
- Run PyTorch 3D registration tests (slow):
  - Command: `pytest --runslow -k "test_pytorch_syn_3d"`
  - Output: `4 passed, 124 deselected, 1 warning in 102.42s` (All 4 PyTorch 3D tests passed).
- Run JAX 2D registration tests:
  - Command: `pytest -v -k "test_jax_syn_2d"`
  - Output:
    ```
    tests/test_syn_jax.py::test_jax_syn_2d_lncc PASSED
    tests/test_syn_jax.py::test_jax_syn_2d_mattes_mi PASSED
    ```
- Run JAX 3D registration tests (slow):
  - Command: `pytest --runslow -k "test_jax_syn_3d"`
  - Output:
    ```
    FAILED tests/test_syn_jax.py::test_jax_syn_3d_lncc - assert -8.26388931274414 > 0.0
    1 failed, 1 passed, 126 deselected, 1 warning in 25.71s
    ```
  - Note: `test_jax_syn_3d_mattes_mi` PASSED, but `test_jax_syn_3d_lncc` FAILED due to a negative Jacobian determinant (`min_jac = -8.26` / `-6.36`).
  - Investigation: A custom comparison script `scratch/compare_torch_jax_3d.py` was run:
    - PyTorch 3D SyNTo (LNCC) min Jacobian: `0.8011` (stable, no folding).
    - JAX 3D SyNTo (LNCC) min Jacobian: `-6.3644` (grid folds severely, failing the stability test).

### Runtime & GPU/MPS Profiling
- Run profiling script `scratch/profile_gpu_overhead.py` on the GPU accelerator (`mps` backend):
  - Output:
    ```
    Profiling on device: mps
    Time for 100 physical space conversions: 0.5030s (0.005030s per iteration)
    Time for 100 registration epochs (simulated): 0.6987s (0.006987s per epoch)
    Physical space conversion overhead: 71.98%
    ```
- Run caching mitigation profiling script `scratch/profile_caching_mitigation.py`:
  - Output:
    ```
    Profiling on device: mps
    Time for 100 cached physical space conversions: 0.1067s (0.001067s per iteration)
    Time for 100 registration epochs (simulated, cached): 0.2187s (0.002187s per epoch)
    Physical space conversion overhead (cached): 48.81%
    Expected speedup in registration epoch runtime: 219.55%
    ```

---

## 2. Logic Chain

1. **Coordinate-Mapping Correctness**: Since the DICE score computed between the ANTs C++ implementation and the PyTorch physical warped grids via `test_internal_dice.py` is exactly `1.000000`, the coordinate transformation equations from physical space back-and-forth to normalized spaces are mathematically correct and conform to ANTs coordinate mapping conventions.
2. **PyTorch Parity & 3D Registration**: The passing of standard pytest suite and `--runslow` PyTorch 3D tests confirms PyTorch registration parity is fully preserved and 3D registration is stable.
3. **JAX 3D Registration Defect**:
   - The JAX 2D registration suite passes cleanly.
   - JAX 3D registration with Mattes MI passes, but JAX 3D with LNCC fails.
   - The minimum Jacobian determinant in JAX 3D LNCC is `-6.36` (meaning severe grid folding occurred), whereas PyTorch 3D LNCC maintains a minimum Jacobian determinant of `0.80`.
   - Therefore, there is a stability/regularization defect specific to the JAX 3D LNCC implementation (potentially in the interaction between 3D gradients and the JAX 3D LNCC metric, or JAX separable Gaussian filter padding behavior).
4. **Physical Space Conversion GPU Overhead**:
   - Every registration epoch, `prepare_mid_images_and_gradients_torch` calls `get_physical_grid_torch` and `physical_to_normalized_torch` on the active GPU device.
   - This allocates tensors and copies constants (`spacing`, `origin`, `direction`, `target_shape`) from CPU memory to GPU memory, triggering high GPU kernel submission and transfer latency.
   - Empirically, this accounts for **71.98%** of the entire registration epoch duration.
   - Pre-computing and caching the physical space grid `X_phys` and pre-transferring the normalization constant tensors once per level reduces this overhead by 5x (from `0.5030s` to `0.1067s` for 100 epochs), providing a simulated **~220% speedup** in overall epoch runtime.

---

## 3. Caveats

- All profiling tests were conducted on macOS using the `mps` (Metal Performance Shaders) backend, as no CUDA GPU was available in the local environment. Performance on CUDA may differ slightly, but the CPU-to-GPU transfer bottleneck and memory allocation latency for static coordinates remain.
- JAX tests were executed on `CpuDevice`, since JAX does not natively support the Apple Silicon MPS backend.

---

## 4. Conclusion

- **Coordinate Mapping Accuracy**: High-accuracy coordinate mapping is verified (DICE = 1.0).
- **Parity Status**: PyTorch 2D/3D and JAX 2D registration parity are verified.
- **Defects Discovered**:
  - **Bug 1: JAX 3D LNCC Instability / Grid Folding**: `test_jax_syn_3d_lncc` fails due to a negative Jacobian determinant (`-6.3644`), indicating grid folding. (PyTorch equivalent is stable at `0.8011`).
  - **Bug 2: Redundant Physical Space Conversion Overhead**: Physical space conversion introduces a **~72% GPU overhead** because grids and conversion tensors are rebuilt/reallocated on every optimization epoch. Caching `X_phys` and pre-allocating/pre-transferring normalization constants once per level yields a **~220% (~3.2x) speedup** in epoch execution.

---

## 5. Verification Method

To verify these results independently, run:
```bash
# 1. Run internal coordinate DICE verification
python scratch/test_internal_dice.py

# 2. Run standard unit tests
pytest

# 3. Run PyTorch 3D registration tests (all pass)
pytest --runslow -k "test_pytorch_syn_3d"

# 4. Run JAX 3D registration tests (LNCC fails)
pytest --runslow -k "test_jax_syn_3d"

# 5. Run physical conversion profiling
python scratch/profile_gpu_overhead.py
python scratch/profile_caching_mitigation.py
```
