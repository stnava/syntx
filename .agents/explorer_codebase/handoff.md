# Handoff Report — explorer_codebase Codebase Investigation

## 1. Observation

### 1.1 Existing Test Suite Baseline Run
We successfully ran `pytest` on the codebase to establish a baseline.
* **Command:** `pytest` in directory `/Users/stnava/code/syntx`
* **Result output:**
  ```
  tests/test_coverage_helpers.py ........................                  [ 51%]
  tests/test_feature_networks.py .......                                   [ 65%]
  tests/test_syn.py ..ss..ss.                                              [ 85%]
  tests/test_syn_jax.py ..ss.                                              [ 95%]
  tests/test_transform.py ..                                               [100%]
  ...
  Name                     Stmts   Miss  Cover   Missing
  ------------------------------------------------------
  src/syntx/__init__.py        7      0   100%
  src/syntx/features.py      232     13    94%   97, 129-130, 169, 251-252, 260-261, 269-270, 296, 305, 314
  src/syntx/resnet.py         75      0   100%
  src/syntx/syn.py           933     72    92%   ...
  src/syntx/syn_jax.py       667     70    90%   ...
  src/syntx/transform.py      96      0   100%
  ------------------------------------------------------
  TOTAL                     2010    155    92%
  ================== 41 passed, 6 skipped, 4 warnings in 46.38s ==================
  ```

### 1.2 Feature Extractor Structure in `src/syntx/features.py`
We inspected `src/syntx/features.py` (lines 9-25) and verified that all feature extractors must subclass `FeatureExtractor`:
```python
class FeatureExtractor(nn.Module):
    """Abstract base for all feature extractors."""
    @property
    def is_3d(self) -> bool:
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        raise NotImplementedError

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract(self, x: torch.Tensor) -> list:
        raise NotImplementedError
```

### 1.3 JAX Registration Loop in `src/syntx/syn_jax.py`
We viewed `src/syntx/syn_jax.py` and observed:
* Hierarchical image pyramids are created at lines 854-866.
* Affine pre-alignment is optimized using `affine_step_jax` (lines 560-623) called inside `SyNTo.fit` (lines 897-949).
* SyN deformable registration is run using `syn_step_jax` (lines 627-768) called inside `SyNTo.fit` (lines 1007-1021).
* Inside `syn_step_jax`, the similarity loss is computed in `loss_mid_fn` under `use_analytical_gradients=True` (lines 675-684) and in `loss_warp_fn` under `use_analytical_gradients=False` (lines 686-696).

### 1.4 JAX and PyTorch DLPack Support
We confirmed that JAX arrays and PyTorch tensors natively support the modern PEP 518/python array API `__dlpack__` protocol for zero-copy memory access via `torch.utils.dlpack.from_dlpack` and `jax.dlpack.from_dlpack` (Observation 1.3 in teamwork_preview_explorer_exploration_1 handoff report).

---

## 2. Logic Chain

1. **Test Suite Stability:** Observation 1.1 shows that the test suite runs and passes cleanly with 92% coverage. Any future implementations of `SwinUNETRExtractor` or the JAX DLPack bridge must preserve these tests and maintain this level of test coverage.
2. **Dynamic / Lazy Loading of MONAI:** Since `monai` is not listed under `dependencies` in `pyproject.toml` (Observation 1.2) and is not installed in the system environments, globally importing MONAI in `src/syntx/features.py` would cause an `ImportError` for users who do not use SwinUNETR. Therefore, MONAI must be imported dynamically inside `SwinUNETRExtractor.__init__`.
3. **Weights Download and Cache:** To ensure that pre-trained SSL weights are available for feature extraction, we can download the official `model_swinvit.pt` from MONAI's model zoo. We will cache it locally at `~/.syntx_cache/model_swinvit.pt` and load it into the `swinViT` backbone using `strict=False` since SSL checkpoints do not contain decoder parameters.
4. **JAX-PyTorch DLPack VJP Bridge:**
   - Because PyTorch functions cannot run inside JAX JIT directly, they must be wrapped as a host callback via `jax.pure_callback`.
   - To support automatic differentiation of this PyTorch-based loss w.r.t inputs `im` and `jm` in JAX, we can define a JAX custom VJP (`jax.custom_vjp`) wrapper.
   - The custom VJP will use `jax.pure_callback` to execute PyTorch's forward and backward passes.
   - PyTorch tensors and JAX arrays can be shared in a zero-copy manner using `from_dlpack`.
5. **Loss Injection:**
   - The wrapped loss function can be directly substituted in `loss_mid_fn` and `loss_warp_fn` inside `syn_step_jax` when `similarity_metric` specifies a feature space metric.

---

## 3. Caveats

* **Offline Environments:** In restricted network modes where download fails, the `SwinUNETRExtractor` must fail gracefully and report that weights must be placed at `~/.syntx_cache/model_swinvit.pt` manually.
* **Device Compatibility:** While DLPack works zero-copy, it is critical that both JAX and PyTorch tensors reside on the same device (e.g. CPU or the same GPU card). If there is a mismatch, the DLPack bridge will raise an error.
* **Analytical Gradients:** When using `use_analytical_gradients=True`, the loss function's gradients w.r.t the image intensities are explicitly mapped to the spatial gradients. The wrapped PyTorch loss must provide correct gradients w.r.t `im` and `jm`.

---

## 4. Conclusion

* MONAI SwinUNETR integration is fully feasible. It should be subclassed under `FeatureExtractor` in `features.py`, use lazy imports, and support downloading/caching of weights.
* A JAX custom VJP bridge using `jax.pure_callback` and `from_dlpack` is the optimal way to integrate PyTorch-based FeatureSpaceLoss into the JAX registration loop.
* We have produced a detailed analysis document at `/Users/stnava/code/syntx/.agents/explorer_codebase/analysis.md` summarizing these findings and including proposed implementation structures.

---

## 5. Verification Method

To independently verify these conclusions and baseline stability:
1. Run the test suite:
   ```bash
   pytest
   ```
2. Verify JAX and PyTorch DLPack functionality on CPU:
   ```bash
   python -c "import jax; import torch; import torch.utils.dlpack; x_jax = jax.numpy.ones(5); x_torch = torch.utils.dlpack.from_dlpack(x_jax); assert x_torch.sum() == 5.0"
   ```
