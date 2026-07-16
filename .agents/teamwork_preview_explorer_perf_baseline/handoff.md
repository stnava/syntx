# Handoff Report: Baseline Profiling & Diagnostics (Milestone 1)

This report details the baseline performance profiling, diagnostic findings, and GEMINI.md compliance analysis for `syntx` registration workflows.

---

## 1. Observation

### Unit Tests
Running the command `pytest` collected 83 items, where 77 passed, 6 were skipped, and 0 failed:
```
tests/test_coverage_helpers.py ........................                  [ 28%]
tests/test_e2e_metrics.py ...............
...
============= 77 passed, 6 skipped, 6 warnings in 75.03s (0:01:15) =============
```

### Benchmark Runtimes
Running the baseline command `python examples/evaluate_all_metrics.py` generated the following output:
```
=== Running Multi-modal Perceptual Similarity Benchmark ===
[T1w-to-B0 | VGG19] Completed in 1.944s. Folding Rate: 0.0000%
[T1w-to-B0 | SwinUNETR] Completed in 1.551s. Folding Rate: 0.0000%
[T1w-to-DWI | VGG19] Completed in 1.633s. Folding Rate: 0.0000%
[T1w-to-DWI | SwinUNETR] Completed in 1.534s. Folding Rate: 0.0000%
Results saved to outputs_comparison/final_feature_metrics_results.csv
```

### Profiling Bottlenecks
Using `cProfile` to analyze `evaluate_all_metrics.py` (saved to `.agents/teamwork_preview_explorer_perf_baseline/profile.txt`), we observed:
* **JAX JIT Compilation:** `compiler.py:292(backend_compile_and_load)` was called 389 times and consumed **5.058s** (over 40% of the entire 12.592s program execution time, and ~65% of registration logic execution).
* **VJP Callbacks:** `syn_jax.py:48(py_forward)` and `syn_jax.py:66(py_backward)` were called 4 times each, taking `0.074s` and `0.136s` respectively.

### JAX-PyTorch Bridge & JIT Trace
In `src/syntx/syn_jax.py` (lines 92-97), the bridge explicitly checks for JAX tracer types:
```python
        if isinstance(m, jax.core.Tracer) or isinstance(f, jax.core.Tracer):
            return jax.pure_callback(
                py_forward,
                jax.ShapeDtypeStruct((), jnp.float32),
                m, f
            )
```
In the fallback function `py_forward` (lines 48-52) and `py_backward` (lines 66-71), host-to-device data movement is explicitly performed via NumPy:
```python
    def py_forward(m_np, f_np):
        m_np = np.asarray(m_np)
        f_np = np.asarray(f_np)
        m_torch = torch.from_numpy(m_np)
        f_torch = torch.from_numpy(f_np)
        ...
```

### Redundant Interpolation & Scaling Operations
* **Triplanar Slicing (`src/syntx/features.py:361`):**
  ```python
  slices_in.append(F.interpolate(input_nd[:, 0, z-1:z+2], size=(target_size, target_size), mode='bilinear', align_corners=True))
  ```
* **SwinUNETR Feature Extractor (`src/syntx/features.py:268-285`):**
  ```python
  if spatial_shape != tuple(original_img_size):
      x_input = F.interpolate(x, size=original_img_size, mode='trilinear', align_corners=True)
  ...
  if spatial_shape != tuple(original_img_size):
      expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]
      feat = F.interpolate(feat, size=expected_shape, mode='trilinear', align_corners=True)
  ```

### GEMINI.md Compliance Check
* **Single Interpolation Policy:** `GEMINI.md` states:
  > *Constraint: No intermediate file-based pre-warping (e.g., calling `ants.apply_transforms` to generate a pre-aligned image for optimization inputs).*
  
  However, in `src/syntx/syn.py` (lines 1394-1396), `ants.apply_transforms` is applied directly to the inputs if `initial_transform` is supplied, and this pre-warped `moving_reg` image is subsequently optimized:
  ```python
      if initial_transform is not None:
          tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
          moving_reg = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_list)
  ```
* **Accuracy Thresholds & Default Configurations:** `GEMINI.md` states:
  > *Do not recommend or default to VGG 2D or coarser layers (like Layer 8) when accuracy is the target.*
  
  However, `src/syntx/syn.py` defaults to `vgg_layers=[8]` (lines 832, 1337) and `vgg_mode='patch_walk'`.

---

## 2. Logic Chain

### JAX JIT Compilation Bottleneck
1. cProfile logs show `backend_compile_and_load` being called 389 times, contributing 5.058s to runtime.
2. In `syn_jax.py`, the registration optimizer calls `fit` at each multi-resolution scale. Inside `fit`, it instantiates the feature loss function (e.g., `VGG19Extractor`) and wraps it via `make_pytorch_loss_jax`.
3. Every invocation of `make_pytorch_loss_jax` returns a new Python function object (`jax_loss_fn` closure).
4. `syn_step_jax` is JIT-compiled with `similarity_metric` specified as a static argument (`static_argnums=(..., 16)`).
5. Because a fresh function object is generated on every registration run (even if parameters and sizes are identical), its object reference changes. JAX treats this as a change in static arguments, invalidating its cache and triggering a full recompilation of `syn_step_jax`.

### JAX-PyTorch Bridge Fallback
1. Under `jax.jit`, variables are passed to JAX operations as `jax.core.Tracer` objects.
2. Checking `isinstance(m, jax.core.Tracer)` redirects execution into the `if` branch of `jax_loss_fn` in `syn_jax.py`.
3. The function returns a `jax.pure_callback`, which runs on the host CPU.
4. JAX converts the device array to a CPU NumPy array (`m_np`, `f_np`).
5. The callback invokes `np.asarray` and `torch.from_numpy` on the host CPU, which copies memory to PyTorch on the host, before moving it to the GPU via `m_torch.to(device)`.
6. This completely bypasses the zero-copy DLPack bridge (`torch.from_dlpack`) when tracing under `jax.jit`, causing multiple device-to-host and host-to-device copies on every single optimization epoch.

### Redundant Interpolation logic
1. In triplanar VGG loss, slices of shape `(H, W)`, `(D, W)`, and `(D, H)` are upscaled to `(target_size, target_size)` where `target_size = max(D, H, W)`. For asymmetric volumes, this forces unnecessary upscaling (e.g., upscaling a size-16 dimension to 128), increasing compute and memory overhead and introducing spatial blurring.
2. In `SwinUNETRExtractor`, if the input volume size is not `(96, 96, 96)`, the volume is upsampled to `(96, 96, 96)` using trilinear interpolation, passed through Swin ViT, and then the resulting feature maps are downsampled back to the expected target shapes.
3. For a `(16, 16, 16)` synthetic volume, this upsampling increases the number of voxels from 4,096 to 884,736 (a $216\times$ increase), significantly slowing down forward passes.

### GEMINI.md Non-Compliance
1. GEMINI.md prohibits file-based pre-warping prior to optimization to prevent spatial blurring.
2. `syn.py` uses `ants.apply_transforms` on the input `moving` image prior to optimization if `initial_transform` is provided, violating this rule.
3. GEMINI.md advises against defaulting to VGG Layer 8.
4. `syn.py` defaults to `vgg_layers=[8]` and `vgg_mode='patch_walk'`, violating the default configuration target.

---

## 3. Caveats

* **Hardware Specifics:** Profiling was conducted on Apple Silicon hardware (Darwin macOS); memory transfers may show different characteristics on CUDA devices due to unified memory vs. discrete PCIe bus copies.
* **Volume Dimensions:** Runtimes recorded in `examples/evaluate_all_metrics.py` utilize a small mock dataset size `(16, 16, 16)` due to downsampling / synthetic fallback. The overhead of SwinUNETR and triplanar upscaling is expected to scale exponentially with larger clinical volume sizes.
* **Read-Only Constraints:** No source code modifications were implemented during this investigation.

---

## 4. Conclusion

The primary performance bottleneck in JAX-backend registrations is JAX JIT cache invalidation caused by dynamic loss function instantiations, followed closely by the DLPack-to-CPU callback fallback under `jax.jit`. Additionally, redundant interpolations in `SwinUNETRExtractor` (upscaling to 96x96x96) and triplanar VGG slicing degrade execution efficiency.

### Actionable Recommendations for Implementer:
1. **JIT Cache Reusability:** Define or cache wrapped loss metrics outside of the `fit` loop, or implement custom hashing for PyTorch wrapped losses so JAX can reuse compiled XLA graphs.
2. **DLPack Bridge under JIT:** Replace `jax.pure_callback` with `jax.experimental.io_callback` or implement a custom JAX primitive to share GPU device pointers directly using DLPack without host round-trips.
3. **Interpolation Clean-up:**
   * Optimize `SwinUNETRExtractor` to support variable-size inputs or pad instead of upsampling.
   * Remove the mandatory `(target_size, target_size)` upscale in triplanar slicing; process each projection batch at its native resolution (or pad instead of scaling).
4. **Guardrail Alignment:**
   * Modify `syn.py` to compose initial transforms dynamically on the coordinate grid instead of pre-warping the moving image via `ants.apply_transforms`.
   * Update the default signature in `syn.py` to `vgg_layers=[4]` and `vgg_mode='lncc_3d'` to match GEMINI.md target recommendations.

---

## 5. Verification Method

To verify these diagnostics independently:
1. **Tests Execution:**
   ```bash
   pytest
   ```
   Verify that all 77 tests pass.
2. **Benchmark Execution:**
   ```bash
   python examples/evaluate_all_metrics.py
   ```
   Verify that the runtimes match the ~1.5s - 2.0s baseline.
3. **Trace Analysis:**
   Inspect the file `.agents/teamwork_preview_explorer_perf_baseline/profile.txt` to confirm that `backend_compile_and_load` represents the major runtime driver.
