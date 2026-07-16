# Challenge Report

## Challenge Summary

**Overall risk assessment**: HIGH (due to a critical component swap discrepancy in `SyNToTransform` file-based exports)

---

## Challenges

### [Critical] Challenge 1: Component Swap Discrepancy in `SyNToTransform` Exports

* **Assumption challenged**: That the displacement fields exported by `SyNToTransform.to_composite_warp()` and `SyNToTransform.export_classic()` correctly represent physical space translations for ANTs/ITK without component order swapping.
* **Attack scenario**: A user performs registration, retrieves the `SyNToTransform` object from the backend (e.g., via JAX `get_forward_transform()`), and calls `.to_composite_warp(filename)` or `.export_classic(prefix)`. When warping the moving image using these exported files (e.g. via `ants.apply_transforms`), ANTs/ITK maps the displacement components incorrectly (swapping X and Z in 3D, and X and Y in 2D). This shifts the image along the wrong axes and degrades the registration quality.
* **Blast radius**: High. Downstream workflows that load the warp fields directly for analysis or application will use incorrect, coordinate-swapped fields. In a 3D sphere translation test, applying the non-swapped warp resulted in a DICE overlap of `0.2289` (failure), compared to `1.0000` (success) when components were swapped.
* **Mitigation**: Update `_to_physical_displacement` in `src/syntx/transform.py` to reverse the components before passing the array to `ants.from_numpy`. Specifically:
  ```python
  if self.dim == 2:
      phys_disp = phys_disp[..., [1, 0]]
  elif self.dim == 3:
      phys_disp = phys_disp[..., [2, 1, 0]]
  ```

### [Medium] Challenge 2: Fragile Metric Lookup in PyTorch Degeneracy Fallback

* **Assumption challenged**: That `self.metrics.index(metric)` in `syn.py` is a robust way to map metrics to loss functions.
* **Attack scenario**: If a user configures `self.metrics` with multiple metrics of the same type (e.g., duplicate custom metrics), `index(metric)` always returns the index of the first occurrence. This can lead to mismatching loss functions if some but not all of the duplicate metrics are deep metrics.
* **Blast radius**: Low. Duplicate metrics in a single SyN optimization loop are extremely rare.
* **Mitigation**: Replace `for metric in self.metrics:` with `for metric_idx, metric in enumerate(self.metrics):` and retrieve the loss function using `metric_idx`, mirroring the robust implementation in JAX (`syn_jax.py`).

---

## Stress Test Results

* **Deep Feature Degeneracy Fallback (Shape < 32)**:
  * *Scenario*: Setup PyTorch and JAX registration on inputs with spatial shape 16.
  * *Expected behavior*: Extractors for VGG19 should not be called, falling back to local LNCC.
  * *Actual/predicted behavior*: Extractors bypassed (call count = 0).
  * *Result*: **PASS**
* **Displacement Field Non-Folding**:
  * *Scenario*: Register 2D phantoms `r16` and `r64` and calculate Jacobian determinant map of the exported forward warp.
  * *Expected behavior*: Strictly positive Jacobian determinants and zero folding rate.
  * *Actual/predicted behavior*: Minimum Jacobian > 0, folding rate = 0.
  * *Result*: **PASS**
* **Parameter Tuning & Parity**:
  * *Scenario*: Compare registration of `r16` and `r27` with tuned parameters (`grad_step=0.75`, `flow_sigma=1.732`) against baseline ANTs SyN.
  * *Expected behavior*: Mean DICE score parity within 1% drop.
  * *Actual/predicted behavior*: PyTorch and JAX backend DICE scores matched or exceeded the baseline (within 1%).
  * *Result*: **PASS**
* **Warp Export Application (Ground Truth)**:
  * *Scenario*: Create a 3D volume with a sphere translated along the X axis by +4 voxels. Export the translation warp using `SyNToTransform` (no swap) vs `syn.py` (component swap), apply them via `ants.apply_transforms`, and evaluate DICE overlap.
  * *Expected behavior*: The warped moving image aligns perfectly with the target (DICE = 1.0000) for both exports.
  * *Actual/predicted behavior*: NOSWAP export (SyNToTransform) fails to align (DICE = 0.2289); SWAPPED export (syn.py) aligns perfectly (DICE = 1.0000).
  * *Result*: **FAIL** (for `SyNToTransform` exports)

---

## Unchallenged Areas

* **Swin UNETR Extractor Weights** — Not challenged under custom weights since we used the default extractor architecture.
* **Cortical label maps natively in JAX** — Due to the lack of JAX-native label overlap metrics, label overlap was evaluated after converting to ANTs/numpy.
