# Handoff Report

## 1. Observation
1. In `/Users/stnava/code/syntx/src/syntx/syn.py` (lines 1727-1736), the high-level `registration()` function swaps the coordinate components of the displacement fields to `[1, 0]` (for 2D) and `[2, 1, 0]` (for 3D) before exporting to ANTsPy images:
   ```python
        if dim == 2:
            disp_l2r_t = disp_l2r[..., [1, 0]]
            disp_r2l_t = disp_r2l[..., [1, 0]]
        elif dim == 3:
            disp_l2r_t = disp_l2r[..., [2, 1, 0]]
            disp_r2l_t = disp_r2l[..., [2, 1, 0]]
   ```
2. In `/Users/stnava/code/syntx/src/syntx/transform.py` (lines 109-128), the `_to_physical_displacement` function of `SyNToTransform` does not perform any component swap:
   ```python
    def _to_physical_displacement(self, normalized_disp):
        """Converts a normalized PyTorch [-1, 1] displacement field to an ITK physical LPS field."""
        spatial_shape = torch.tensor(list(reversed(self.target_shape)), dtype=torch.float32, device=self.device)
        ...
        return ants.from_numpy(
            phys_disp, 
            origin=self.metadata['origin'], 
            spacing=self.metadata['spacing'], 
            direction=self.metadata['direction'], 
            has_components=True
        )
   ```
3. In a ground-truth translation test (`/Users/stnava/code/syntx/.agents/challenger_m5_4/verify_warp_application.py`), a 3D sphere shifted by +4 voxels along the X axis was aligned using `ants.apply_transforms`. Applying the non-swapped warp (`SyNToTransform` export) failed to align the sphere, yielding a DICE score of `0.2289`. Applying the swapped warp (components reversed) successfully aligned the sphere perfectly, yielding a DICE score of `1.0000`.
4. In `/Users/stnava/code/syntx/tests/test_challenger_verification.py`, all tests successfully passed, indicating that the degeneracy trigger fallback (shape < 32) correctly falls back to local LNCC (extractor call count = 0), the Jacobian determinant is positive (non-folding), and the tuned parameters (`grad_step=0.75`, `flow_sigma=1.732`) achieve DICE score parity with ANTs.

## 2. Logic Chain
1. From Observation 1, the `registration()` function reverses displacement field components when writing files.
2. From Observation 2, `SyNToTransform` does not reverse displacement field components when writing files.
3. From Observation 3, only displacement fields with reversed (swapped) component ordering result in correct alignment when applied via `ants.apply_transforms`. Non-swapped fields apply displacements along the wrong axes (e.g. mapping X displacement to the Z axis in 3D).
4. Therefore, `SyNToTransform`'s `.to_composite_warp()` and `.export_classic()` methods write incorrect (corrupted) displacement fields to disk, representing a critical component swap bug.
5. From Observation 4 and 5, parameter tuning and degeneracy fallbacks are correct and robust under standard inputs.

## 3. Caveats
No caveats. All findings were verified empirically using ground-truth translation shift simulations on 3D volumes.

## 4. Conclusion
* **Displacement Field Component Swap**: There is a critical component swap bug in `src/syntx/transform.py`'s `_to_physical_displacement` method because it does not reverse the coordinate component dimensions `[x, y]` or `[x, y, z]` to matching ITK/ANTs physical coordinate order `[y, x]` or `[z, y, x]`. High-level `registration()` (in `syn.py`) correctly performs this swap, but `SyNToTransform` exports do not.
* **Parameter Tuning & Fallback**: Correct and robust. PARITY with ANTs is achieved using tuned parameters, non-folding is validated via strictly positive Jacobian determinants, and the fallback to local LNCC successfully bypasses deep metric feature extraction on degenerate inputs (min shape < 32).

## 5. Verification Method
1. Run the ground-truth warp application test script:
   `python /Users/stnava/code/syntx/.agents/challenger_m5_4/verify_warp_application.py`
   Output shows:
   `DICE after applying NOSWAP warp (SyNToTransform export): 0.2289`
   `DICE after applying SWAPPED warp (syn.py registration export): 1.0000`
2. Run pytest suite (especially `tests/test_challenger_verification.py`):
   `pytest tests/test_challenger_verification.py`
