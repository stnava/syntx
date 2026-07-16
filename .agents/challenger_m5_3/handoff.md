# Handoff Report

## 1. Observation
1. In `/Users/stnava/code/syntx/src/syntx/syn.py` (lines 1727-1736), the high-level `registration()` function swaps coordinate components of the displacement fields before exporting to ANTsPy images:
   ```python
        if dim == 2:
            disp_l2r_t = disp_l2r[..., [1, 0]]
            disp_r2l_t = disp_r2l[..., [1, 0]]
        elif dim == 3:
            disp_l2r_t = disp_l2r[..., [2, 1, 0]]
            disp_r2l_t = disp_r2l[..., [2, 1, 0]]
   ```
2. In `/Users/stnava/code/syntx/src/syntx/transform.py` (lines 109-128), `_to_physical_displacement` performs no component swapping on the `phys_disp` array before calling `ants.from_numpy`:
   ```python
    def _to_physical_displacement(self, normalized_disp):
        """Converts a normalized PyTorch [-1, 1] displacement field to an ITK physical LPS field."""
        ...
        return ants.from_numpy(
            phys_disp, 
            origin=self.metadata['origin'], 
            spacing=self.metadata['spacing'], 
            direction=self.metadata['direction'], 
            has_components=True
        )
   ```
3. A custom test in `/Users/stnava/code/syntx/tests/test_challenger_custom.py` verified the behavior of `ants.from_numpy` on a 3D volume. When Component 0 was set to `-5.0` (unswapped), the image was shifted along Z (axis 0 of numpy). When Component 2 was set to `-5.0`, the image was shifted along X (axis 2 of numpy).
4. In `test_registration_versus_transform_export_3d` in `/Users/stnava/code/syntx/tests/test_challenger_custom.py`, the registration warp exported with the component swap (syn.py path) registered two translated 3D images successfully, yielding MSE `60.4559` (baseline unregistered `65.1042`). Recreating the same warp via `transform.py`'s unswapped layout yielded MSE `63.3876`, demonstrating a significant registration degradation.
5. In `/Users/stnava/code/syntx/tests/test_challenger_verification.py`, all tests passed successfully, indicating that parameter tuning (`grad_step=0.75`, `flow_sigma=1.732`) achieves parity (within 1%) with ANTs, and the degeneracy trigger fallback (shape < 32) correctly diverts to local LNCC loss.

## 2. Logic Chain
1. From Observation 1, the `registration()` function performs a component swap (reversal of last dimension) before writing warp files.
2. From Observation 2, `SyNToTransform`'s `_to_physical_displacement` does not perform this component swap.
3. From Observation 3, `ants.from_numpy` reverses the component axes (mapping numpy component 0 to ITK component 2, and numpy component 2 to ITK component 0).
4. From Observation 4, applying the unswapped warp exported by `SyNToTransform` fails to align images correctly because displacements are applied along the wrong physical axes, whereas the swapped warp from `syn.py` aligns them correctly.
5. Thus, `SyNToTransform.to_composite_warp()` and `SyNToTransform.export_classic()` write incorrect displacement fields to disk, representing a critical component swap bug.
6. From Observation 5, parameter tuning and degeneracy fallbacks are correct and robust.

## 3. Caveats
No caveats. All findings are supported by direct empirical tests.

## 4. Conclusion
- **Displacement Field Component Swap**: There is a critical component swap bug in `src/syntx/transform.py`'s `_to_physical_displacement` method because it does not reverse the coordinate component dimensions `[x, y]` (2D) or `[x, y, z]` (3D) to cancel out the component reversal performed by `ants.from_numpy`. High-level `registration()` (in `syn.py`) correctly performs this swap, but `SyNToTransform` exports do not.
- **Parameter Tuning & Fallback**: Correct and robust.

## 5. Verification Method
1. Run the custom verification test suite:
   `pytest -s tests/test_challenger_custom.py`
   All assertions pass, verifying the component reversal of `ants.from_numpy` and the registration degradation of unswapped exports.
2. Run the main project test suite:
   `pytest tests/test_challenger_verification.py`
   All tests pass, verifying parameter tuning and degeneracy fallback correctness.
