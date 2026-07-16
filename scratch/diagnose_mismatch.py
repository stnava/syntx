"""Investigate the mismatch between cached and non-cached physical_to_normalized."""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
from syntx.syn import (physical_to_normalized_torch, physical_to_normalized_torch_cached,
                        _physical_to_normalized_torch_yfirst)

# Simple test: 1D case with known values
spacing = (2.0, 3.0)  # XY order (ANTs physical space convention)
origin = (10.0, 20.0)
direction = np.eye(2)
shape = (4, 6)  # HW = YX order

# A physical point at origin should map to normalized (-1, -1)
phys = torch.tensor([[[10.0, 20.0]]])  # Origin point

# Non-cached version:
# spacing_rev = (3.0, 2.0), origin_rev = (20.0, 10.0), direction_rev = eye(2)
# Then _yfirst computes:
#   diff = [10, 20] - [20, 10] = [-10, 10]  (but origin is now reversed)
# Wait, the input phys_coords are in... what order?

print("=== Physical coordinate order analysis ===")
print(f"ANTs spacing: {spacing} (X, Y)")
print(f"ANTs origin: {origin} (X, Y)")
print(f"Internal shape: {shape} (H=Y, W=X)")

print(f"\nphysical_to_normalized_torch:")
print(f"  Reverses spacing to {tuple(reversed(spacing))}")
print(f"  Reverses origin to {tuple(reversed(origin))}")
print(f"  Then calls _yfirst which expects YX-ordered parameters")

# So _yfirst gets spacing=(3,2), origin=(20,10), and shape=(4,6)
# The input phys_coords should be in... what order?
# get_physical_grid_torch outputs YX order

# Let me trace get_physical_grid_torch
from syntx.syn import get_physical_grid_torch
grid = get_physical_grid_torch(shape, spacing, origin, direction)
print(f"\nget_physical_grid_torch output shape: {grid.shape}")
print(f"Grid corner [0,0]: {grid[0, 0, 0, :]}")  # Should be origin in YX
print(f"Grid corner [0,-1]: {grid[0, 0, -1, :]}")  # Should be (origin_y, origin_x + (W-1)*spacing_x)
print(f"Grid corner [-1,0]: {grid[0, -1, 0, :]}")  # Should be (origin_y + (H-1)*spacing_y, origin_x)

# For identity direction:
# Physical grid at [i, j] should be (origin_y + i*spacing_y, origin_x + j*spacing_x)
expected_00 = (origin[1], origin[0])  # (Y, X) = (20, 10)
expected_0W = (origin[1], origin[0] + (shape[1]-1)*spacing[0])  # (20, 10+5*2) = (20, 20)
expected_H0 = (origin[1] + (shape[0]-1)*spacing[1], origin[0])  # (20+3*3, 10) = (29, 10)
print(f"\nExpected grid[0,0]: {expected_00}")
print(f"Expected grid[0,-1]: {expected_0W}")
print(f"Expected grid[-1,0]: {expected_H0}")

print(f"\n=== Applying physical_to_normalized_torch (non-cached) ===")
# Test with grid origin point
test_point = grid[0:1, 0:1, 0:1, :]  # Origin
norm_noncached = physical_to_normalized_torch(test_point, shape, spacing, origin, direction)
print(f"Origin -> normalized (non-cached): {norm_noncached}")
print(f"Expected: (-1, -1) in grid_sample XY order")

test_end = grid[0:1, -1:, -1:, :]  # Far corner
norm_end_nc = physical_to_normalized_torch(test_end, shape, spacing, origin, direction)
print(f"Far corner -> normalized (non-cached): {norm_end_nc}")
print(f"Expected: (1, 1) in grid_sample XY order")

print(f"\n=== Applying physical_to_normalized_torch_cached ===")
# Cached version uses pre-reversed parameters
spacing_rev = tuple(reversed(spacing))
origin_rev = tuple(reversed(origin))
direction_rev = np.asarray(direction)[::-1, ::-1].copy()

shape_t = torch.tensor(list(shape), dtype=torch.float32)
spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
origin_t = torch.tensor(origin_rev, dtype=torch.float32)
direction_t = torch.tensor(direction_rev, dtype=torch.float32)

norm_cached = physical_to_normalized_torch_cached(test_point, shape_t, spacing_t, origin_t, direction_t)
print(f"Origin -> normalized (cached): {norm_cached}")

norm_end_c = physical_to_normalized_torch_cached(test_end, shape_t, spacing_t, origin_t, direction_t)
print(f"Far corner -> normalized (cached): {norm_end_c}")

print(f"\nDifference at origin: {(norm_cached - norm_noncached).abs().max().item():.10f}")
print(f"Difference at far corner: {(norm_end_c - norm_end_nc).abs().max().item():.10f}")

