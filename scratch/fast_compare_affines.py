"""Compare torch vs numpy affine conversions."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
import numpy as np, torch
sys.path.insert(0, 'src')
import ants
from syntx.syn import grid_to_physical_affine, grid_to_physical_affine_torch

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# Simple test: identity T_grid
T_id = np.eye(3, dtype=np.float32)
M_np, t_np = grid_to_physical_affine(T_id, fi, mi)
M_pt, t_pt = grid_to_physical_affine_torch(
    torch.tensor(T_id), fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction)

print("Identity T_grid:")
print(f"  numpy M:\n{M_np}")
print(f"  numpy t: {t_np}")
print(f"  torch M:\n{M_pt.numpy()}")
print(f"  torch t: {t_pt.numpy()}")
print(f"  Match M: {np.allclose(M_np, M_pt.numpy(), atol=1e-5)}")
print(f"  Match t: {np.allclose(t_np, t_pt.numpy(), atol=1e-5)}")

# Test with a non-trivial affine
T_test = np.array([[1.05, -0.01, 0.1], [-0.01, 0.92, -0.2], [0, 0, 1]], dtype=np.float32)
M_np2, t_np2 = grid_to_physical_affine(T_test, fi, mi)
M_pt2, t_pt2 = grid_to_physical_affine_torch(
    torch.tensor(T_test), fi.shape, fi.spacing, fi.origin, fi.direction,
    mi.shape, mi.spacing, mi.origin, mi.direction)

print(f"\nNon-trivial T_grid:\n{T_test}")
print(f"  numpy M:\n{M_np2}")
print(f"  numpy t: {t_np2}")
print(f"  torch M:\n{M_pt2.numpy()}")
print(f"  torch t: {t_pt2.numpy()}")
print(f"  Match M: {np.allclose(M_np2, M_pt2.numpy(), atol=1e-4)}")
print(f"  Match t: {np.allclose(t_np2, t_pt2.numpy(), atol=1e-4)}")

# The torch version outputs YX but numpy outputs XY
# Let's check if numpy output == P @ torch_output @ P
P = np.eye(2)[::-1]
M_pt2_xy = P @ M_pt2.numpy() @ P
t_pt2_xy = P @ t_pt2.numpy()
print(f"\n  torch→XY M:\n{M_pt2_xy}")
print(f"  torch→XY t: {t_pt2_xy}")
print(f"  Match after perm: {np.allclose(M_np2, M_pt2_xy, atol=1e-4)}")
