"""Quick verification that the coordinate fixes are correct."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np
import torch
sys.path.insert(0, 'src')
import ants
from syntx.syn import get_physical_grid_torch, physical_to_normalized_torch, physical_to_normalized_torch_cached

# Test 1: Grid correctness with anisotropic spacing
shape = (4, 6)  # H=4, W=6 (YX)
spacing_xy = (2.0, 3.0)  # ANTs XY
origin_xy = (10.0, 20.0)  # ANTs XY
direction = np.eye(2)

grid = get_physical_grid_torch(shape, spacing_xy, origin_xy, direction)
print("=== Grid correctness (anisotropic) ===")
print(f"Grid[0,0]: {grid[0,0,0,:].tolist()}  (expect [20, 10] = YX origin)")
print(f"Grid[0,5]: {grid[0,0,5,:].tolist()}  (expect [20, 20] = Y=20, X=10+5*2)")
print(f"Grid[3,0]: {grid[0,3,0,:].tolist()}  (expect [29, 10] = Y=20+3*3, X=10)")
assert torch.allclose(grid[0,0,0,:], torch.tensor([20.0, 10.0])), "Origin wrong!"
assert torch.allclose(grid[0,0,5,:], torch.tensor([20.0, 20.0])), "Grid[0,5] wrong!"
assert torch.allclose(grid[0,3,0,:], torch.tensor([29.0, 10.0])), "Grid[3,0] wrong!"
print("✓ Grid is correct")

# Test 2: Normalization consistency (cached vs non-cached)
spacing_rev = tuple(reversed(spacing_xy))
origin_rev = tuple(reversed(origin_xy))
direction_rev = np.asarray(direction)[::-1, ::-1].copy()
shape_t = torch.tensor(list(shape), dtype=torch.float32)
spacing_t = torch.tensor(spacing_rev, dtype=torch.float32)
origin_t = torch.tensor(origin_rev, dtype=torch.float32)
direction_t = torch.tensor(direction_rev, dtype=torch.float32)

norm_nc = physical_to_normalized_torch(grid, shape, spacing_xy, origin_xy, direction)
norm_c = physical_to_normalized_torch_cached(grid, shape_t, spacing_t, origin_t, direction_t)
max_diff = (norm_nc - norm_c).abs().max().item()
print(f"\n=== Cached vs non-cached consistency ===")
print(f"Max difference: {max_diff:.10f}")
assert max_diff < 1e-6, f"Cached/non-cached mismatch: {max_diff}"
print("✓ Cached and non-cached are consistent")

# Test 3: Identity mapping (origin → (-1,-1), far corner → (1,1))
origin_pt = grid[0:1, 0:1, 0:1, :]
far_pt = grid[0:1, -1:, -1:, :]
norm_origin = physical_to_normalized_torch(origin_pt, shape, spacing_xy, origin_xy, direction)
norm_far = physical_to_normalized_torch(far_pt, shape, spacing_xy, origin_xy, direction)
print(f"\n=== Identity normalization ===")
print(f"Origin normalized: {norm_origin.squeeze().tolist()}  (expect [-1, -1])")
print(f"Far corner normalized: {norm_far.squeeze().tolist()}  (expect [1, 1])")
assert torch.allclose(norm_origin.squeeze(), torch.tensor([-1.0, -1.0])), "Origin norm wrong!"
assert torch.allclose(norm_far.squeeze(), torch.tensor([1.0, 1.0])), "Far corner norm wrong!"
print("✓ Normalization produces correct [-1,1] range")

# Test 4: grid_sample identity
import torch.nn.functional as F
fi = ants.image_read(ants.get_ants_data('r16'))
I_t = torch.tensor(fi.numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
X_phys = get_physical_grid_torch(fi.shape, fi.spacing, fi.origin, fi.direction)
coords = physical_to_normalized_torch(X_phys, fi.shape, fi.spacing, fi.origin, fi.direction)
I_id = F.grid_sample(I_t, coords, align_corners=True)
diff = (I_t - I_id).abs().max().item()
print(f"\n=== grid_sample identity ===")
print(f"Max pixel diff from identity warp: {diff:.8f}")
assert diff < 1e-5, f"Identity warp is wrong: {diff}"
print("✓ Identity warp reproduces original image")

# Test 5: Quick SyN registration
import syntx
mi = ants.image_read(ants.get_ants_data('r64'))
tx_aff = ants.registration(fi, mi, 'Affine', reg_iterations=[100, 100, 20])
mi_aff = ants.apply_transforms(fi, mi, tx_aff['fwdtransforms'])
mi_before = ants.image_mutual_information(fi, mi_aff)

reg_pt = syntx.syn(fi, mi_aff, 'SyNTo', backend='pytorch',
                    reg_iterations=[20, 0, 0], affine_iterations=[0, 0, 0],
                    similarity_metric='mattes_mi', verbose=False, grad_step=0.2)
warped = ants.apply_transforms(fi, mi_aff, reg_pt['fwdtransforms'])
mi_after = ants.image_mutual_information(fi, warped)

reg_ants = ants.registration(fi, mi_aff, 'SyNOnly', reg_iterations=[20, 0, 0], syn_metric='mattes')
mi_ants = ants.image_mutual_information(fi, reg_ants['warpedmovout'])

print(f"\n=== SyN registration test ===")
print(f"MI before SyN: {mi_before:.4f}")
print(f"MI after syntx SyN: {mi_after:.4f}")
print(f"MI after ANTs SyN: {mi_ants:.4f}")
print(f"Δ MI (syntx): {mi_after - mi_before:.4f}  {'BETTER' if mi_after < mi_before else 'WORSE'}")
print(f"Δ MI (ANTs):  {mi_ants - mi_before:.4f}")
