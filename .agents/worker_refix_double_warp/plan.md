# Refix Double Warp - Implementation & Verification Plan

This plan details the steps to resolve the double affine warping bug in the PyTorch backend (`src/syntx/syn.py`) and verify that all test requirements are successfully met.

## 1. Edit Target Code
- **Target File**: `src/syntx/syn.py`
- **Location**: Lines 1353-1360
- **Action**: Remove the pre-warping block:
  ```python
  with torch.no_grad():
      grid = self.get_affine_grid(spatial_shape, device)
      if initial_grid is not None:
          grid = compose_grids(initial_grid, grid)
      moving_affine = F.grid_sample(moving_image, grid, padding_mode='border', align_corners=True)
      
  J_pyr = [F.interpolate(moving_affine, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=False) if s > 1 else moving_affine for s in levels]
  ```
  This prevents `J_pyr` from being overwritten with pre-warped images, matching JAX's correct single-interpolation behavior.

## 2. Run Target Test in Sequence
- Run the specific failing/flaky test in sequence:
  ```bash
  pytest tests/test_syn.py -k test_pytorch_syn_2d_vgg19
  ```
- Ensure the test passes without flakiness.

## 3. Run Internal Dice Test
- Run `scratch/test_internal_dice.py` and verify the output DICE score is >= 0.999.
  ```bash
  python scratch/test_internal_dice.py
  ```

## 4. Run Full Test Suite
- Run the entire test suite using pytest to ensure all 122 tests pass:
  ```bash
  pytest
  ```

## 5. Produce Handoff Report
- Write `handoff.md` with the required five sections: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
