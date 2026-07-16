"""Pin down the exact bug in physical_to_normalized_torch_cached."""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

# The bug is in the flip. Let me trace both functions step by step.

# Input: physical coords in YX order (from get_physical_grid_torch)
spacing_xy = (2.0, 3.0)  # ANTs XY
origin_xy = (10.0, 20.0)  # ANTs XY
direction = np.eye(2)
shape_yx = (4, 6)  # YX

# === Non-cached path ===
print("=== Non-cached: physical_to_normalized_torch ===")
spacing_rev = tuple(reversed(spacing_xy))  # (3.0, 2.0) = YX
origin_rev = tuple(reversed(origin_xy))    # (20.0, 10.0) = YX
direction_rev = np.asarray(direction)[::-1, ::-1].copy()  # still eye(2)
print(f"  spacing_rev (YX): {spacing_rev}")
print(f"  origin_rev (YX): {origin_rev}")

# Then _yfirst receives these in YX order and processes:
# Input phys_coords are in YX order: e.g., (20, 10) for the origin
phys_yx = torch.tensor([20.0, 10.0])  # origin
print(f"  phys_yx (input): {phys_yx}")

diff = phys_yx - torch.tensor(list(origin_rev))  # [20,10] - [20,10] = [0,0]
print(f"  diff = phys - origin_rev: {diff}")

rotated = diff @ torch.tensor(np.array(direction_rev), dtype=torch.float32)  # [0,0] @ I = [0,0]
print(f"  rotated: {rotated}")

voxel = rotated / torch.tensor(list(spacing_rev))  # [0,0] / [3,2] = [0,0]
print(f"  voxel: {voxel}")

shape_t = torch.tensor(list(shape_yx), dtype=torch.float32)
norm = (voxel / (shape_t - 1)) * 2.0 - 1.0  # [0,0] / [3,5] * 2 - 1 = [-1,-1]
print(f"  normalized (YX order): {norm}")
# _yfirst returns this directly - NO reversal
# So the output is in YX order... but grid_sample expects XY!
# Actually, grid_sample with align_corners=True expects the last dim to be (x, y) for 2D
# For a (B, H, W, 2) grid, dim=-1 is (x_coord, y_coord)
print(f"  This means output is in YX order")
print(f"  For grid_sample, this means component [0] indexes rows (Y), [1] indexes cols (X)")
print(f"  grid_sample EXPECTS [0]=X, [1]=Y")
print(f"  SO THE NON-CACHED VERSION IS ALSO SWAPPED!")

# Wait, let me re-check. What does grid_sample actually expect?
# Per PyTorch docs: grid[..., 0] = width (x), grid[..., 1] = height (y)
# So grid_sample expects XY order in the last dimension.
# _yfirst returns YX order.
# physical_to_normalized_torch calls _yfirst with reversed params but doesn't flip.
# So physical_to_normalized_torch ALSO returns YX order!

# But the actual test showed origin -> (-1, -1) for both, which is symmetric.
# Let me test with the far corner.

print(f"\n=== Far corner test ===")
# Far corner in YX: (20 + 3*3, 10 + 5*2) = (29, 20)
far_yx = torch.tensor([29.0, 20.0])

# Non-cached path (reversed params):
diff2 = far_yx - torch.tensor(list(origin_rev))  # [29,20] - [20,10] = [9,10]
rotated2 = diff2  # identity direction
voxel2 = rotated2 / torch.tensor(list(spacing_rev))  # [9,10] / [3,2] = [3, 5]
norm2 = (voxel2 / (shape_t - 1)) * 2.0 - 1.0  # [3/3, 5/5] * 2 - 1 = [1, 1]
print(f"  Non-cached far corner (YX): {norm2}")
print(f"  This is (1,1) which is correct in YX but grid_sample reads as X=1, Y=1")

# But the actual test showed (2.333, 0.200) for non-cached! Let me re-check.
# Oh wait - the grid at [-1, -1] is NOT what I expected.
# Let me print the actual grid values

print(f"\n=== Actually checking get_physical_grid_torch ===")
from syntx.syn import get_physical_grid_torch
grid = get_physical_grid_torch(shape_yx, spacing_xy, origin_xy, direction)
print(f"  Grid[0,0]: {grid[0, 0, 0, :]} (should be origin in YX = (20, 10))")
print(f"  Grid[-1,-1]: {grid[0, -1, -1, :]} (should be far corner in YX)")
print(f"  Grid[0,-1]: {grid[0, 0, -1, :]} (last column, first row)")
print(f"  Grid[-1,0]: {grid[0, -1, 0, :]} (first column, last row)")

# The grid output says:
# Grid[0,0]: (20, 10)   -- origin in YX: correct
# Grid[0,-1]: (35, 10)  -- last column should be Y=20, X=10+(6-1)*2=20 but got (35, 10)!
# That means Y changed by 15 = 5*3 = (W-1)*spacing_y???
# This is WRONG. Walking along columns (dim=1) should change X, not Y.
# But the physical grid seems to be mapping dim=1 to Y and dim=0 to X.

# Let me look at get_physical_grid_torch more carefully
print(f"\n  For shape (H=4, W=6), spacing_xy=(2,3):")
print(f"  Grid[i,j] should be: Y = origin_y + i*spacing_y, X = origin_x + j*spacing_x")
print(f"  Grid[0,0] = (Y=20, X=10) ✓")
print(f"  Grid[0,5] should be (Y=20, X=10+5*2=20) but got {grid[0, 0, 5, :]}")
print(f"  Grid[3,0] should be (Y=20+3*3=29, X=10) but got {grid[0, 3, 0, :]}")

