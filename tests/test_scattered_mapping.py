"""
tests.test_scattered_mapping — Verification Suite for Bidirectional Coordinate Mapping
=====================================================================================

Covers Milestone M2 (Features F8, F9, F20):
- Differentiable Eulerian displacement field evaluation at scattered coordinates (2D and 3D).
- Forward and backward coordinate warping: phi(x) = x + u(x) and phi^-1(y) = y + v(y).
- Exact linear field reproduction to machine precision (float64).
- Exact grid vertex recovery.
- Padding modes ('border', 'zeros') and out-of-bounds safety (zero NaNs).
- Coordinate conventions ('xyz' vs 'zyx') and physical domain bounds.
- Anderson inverse consistency criterion: ||phi o phi^-1 - id||_inf < 1e-3.
- Autograd gradcheck in double precision (float64) for coordinates and field.
- Edge cases (empty point clouds, single points, dimension mismatches).
- ScatteredWarper reusable nn.Module.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx.core.inverse import update_inverse_field_nd_anderson
from syntx.scattered.mapping import (
    evaluate_field_at_scattered,
    warp_scattered_coordinates,
    ScatteredWarper,
)


def test_evaluate_field_shapes_2d_and_3d():
    """Test 1: Output tensor shapes across 2D/3D, batched/unbatched, and asymmetric grids."""
    # 2D Batched
    field_2d_b = torch.randn(2, 16, 20, 3)
    coords_2d_b = torch.randn(2, 50, 2)
    out1 = evaluate_field_at_scattered(field_2d_b, coords_2d_b)
    assert out1.shape == (2, 50, 3)

    # 2D Unbatched
    field_2d_u = torch.randn(16, 20, 3)
    coords_2d_u = torch.randn(50, 2)
    out2 = evaluate_field_at_scattered(field_2d_u, coords_2d_u)
    assert out2.shape == (50, 3)

    # 3D Batched
    field_3d_b = torch.randn(2, 8, 12, 16, 3)
    coords_3d_b = torch.randn(2, 40, 3)
    out3 = evaluate_field_at_scattered(field_3d_b, coords_3d_b)
    assert out3.shape == (2, 40, 3)

    # 3D Unbatched
    field_3d_u = torch.randn(8, 12, 16, 3)
    coords_3d_u = torch.randn(40, 3)
    out4 = evaluate_field_at_scattered(field_3d_u, coords_3d_u)
    assert out4.shape == (40, 3)


def test_evaluate_field_exact_linear_reproduction_2d():
    """Test 2: Bilinear interpolation of an affine field is exact everywhere to machine precision."""
    H, W = 16, 16
    y = torch.linspace(-1, 1, H, dtype=torch.float64)
    x = torch.linspace(-1, 1, W, dtype=torch.float64)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    # Affine field: f1(x, y) = 2.0*x - 3.0*y + 1.0, f2(x, y) = -0.5*x + 4.0*y
    f_x = 2.0 * gx - 3.0 * gy + 1.0
    f_y = -0.5 * gx + 4.0 * gy
    field = torch.stack([f_x, f_y], dim=-1).unsqueeze(0)  # (1, 16, 16, 2)

    torch.manual_seed(42)
    pts = (torch.rand(25, 2, dtype=torch.float64) * 1.7 - 0.85)  # in (-0.85, 0.85)^2

    sampled = evaluate_field_at_scattered(field, pts)  # (25, 2)

    expected_x = 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 1.0
    expected_y = -0.5 * pts[:, 0] + 4.0 * pts[:, 1]
    expected = torch.stack([expected_x, expected_y], dim=-1)

    diff = torch.abs(sampled - expected).max()
    assert diff < 1e-12, f"Linear interpolation error {diff} exceeds machine precision"


def test_evaluate_field_exact_linear_reproduction_3d():
    """Test 3: Trilinear interpolation of a 3D linear field is exact to machine precision."""
    D, H, W = 8, 10, 12
    z = torch.linspace(-1, 1, D, dtype=torch.float64)
    y = torch.linspace(-1, 1, H, dtype=torch.float64)
    x = torch.linspace(-1, 1, W, dtype=torch.float64)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')

    f3_x = 2.0 * gx - 1.5 * gy + 0.5 * gz + 1.0
    f3_y = -gx + 3.0 * gy - 2.0 * gz
    f3_z = 0.5 * gx + 0.5 * gy + gz - 0.5
    field3 = torch.stack([f3_x, f3_y, f3_z], dim=-1).unsqueeze(0)

    torch.manual_seed(42)
    pts3 = (torch.rand(20, 3, dtype=torch.float64) * 1.6 - 0.8)

    sampled3 = evaluate_field_at_scattered(field3, pts3)

    exp3_x = 2.0 * pts3[:, 0] - 1.5 * pts3[:, 1] + 0.5 * pts3[:, 2] + 1.0
    exp3_y = -pts3[:, 0] + 3.0 * pts3[:, 1] - 2.0 * pts3[:, 2]
    exp3_z = 0.5 * pts3[:, 0] + 0.5 * pts3[:, 1] + pts3[:, 2] - 0.5
    expected3 = torch.stack([exp3_x, exp3_y, exp3_z], dim=-1)

    diff3 = torch.abs(sampled3 - expected3).max()
    assert diff3 < 1e-12, f"3D Linear interpolation error {diff3} exceeds machine precision"


def test_evaluate_field_grid_nodes_exact_recovery():
    """Test 4: Exact recovery of field values when querying precisely on grid nodes."""
    grid = torch.tensor([
        [1.0, 2.0],
        [5.0, 6.0]
    ], dtype=torch.float32).unsqueeze(-1)  # (2, 2, 1)

    # 4 corners of normalized grid [-1, 1]^2
    pts = torch.tensor([
        [-1.0, -1.0],  # top-left (row 0, col 0) -> 1.0
        [1.0, -1.0],   # top-right (row 0, col 1) -> 2.0
        [-1.0, 1.0],   # bottom-left (row 1, col 0) -> 5.0
        [1.0, 1.0],    # bottom-right (row 1, col 1) -> 6.0
    ], dtype=torch.float32)

    sampled = evaluate_field_at_scattered(grid, pts)
    expected = torch.tensor([[1.0], [2.0], [5.0], [6.0]], dtype=torch.float32)
    assert torch.allclose(sampled, expected, atol=1e-6)


def test_evaluate_field_padding_modes_and_out_of_bounds():
    """Test 5: Verify out-of-bounds safety, zero NaNs, and padding_mode behavior."""
    field = torch.ones(10, 10, 2, dtype=torch.float32) * 5.0
    pts_out = torch.tensor([[-2.5, 0.0], [0.0, 3.0], [5.0, 5.0]], dtype=torch.float32)

    # Border padding clamps to boundary value (5.0)
    out_border = evaluate_field_at_scattered(field, pts_out, padding_mode='border')
    assert torch.isfinite(out_border).all()
    assert torch.allclose(out_border, torch.full_like(out_border, 5.0), atol=1e-5)

    # Zeros padding zeroes out-of-bounds entries
    out_zeros = evaluate_field_at_scattered(field, pts_out, padding_mode='zeros')
    assert torch.isfinite(out_zeros).all()
    assert torch.allclose(out_zeros, torch.zeros_like(out_zeros), atol=1e-5)


def test_evaluate_field_coordinate_conventions():
    """Test 6: Verify coordinate convention handling ('xyz' vs 'zyx')."""
    H, W = 16, 20
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    field = gx.unsqueeze(-1)  # (16, 20, 1) has gradient strictly along X
    pts = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

    # In 'xyz': coords[0] is X=0.5
    out_xyz = evaluate_field_at_scattered(field, pts, coord_convention='xyz')
    assert torch.isclose(out_xyz[0, 0], torch.tensor(0.5), atol=1e-5)

    # In 'zyx': coords[1] is X, so coords=[[-0.5, 0.5]] has X=0.5 at index 1
    pts_zyx = torch.tensor([[-0.5, 0.5]], dtype=torch.float32)
    out_zyx = evaluate_field_at_scattered(field, pts_zyx, coord_convention='zyx')
    assert torch.isclose(out_zyx[0, 0], torch.tensor(0.5), atol=1e-5)


def test_evaluate_field_physical_domain_bounds():
    """Test 7: Verify field evaluation with physical millimeter domain bounds."""
    field = torch.linspace(-1, 1, 33).view(33, 1, 1).expand(33, 33, 1).contiguous()
    bounds = ([0.0, 0.0], [100.0, 100.0])

    # Point at physical center (50, 50) mm should correspond to normalized (0, 0)
    pts_phys = torch.tensor([[50.0, 50.0]], dtype=torch.float32)
    sampled = evaluate_field_at_scattered(field, pts_phys, domain_bounds=bounds)
    assert torch.isclose(sampled[0, 0], torch.tensor(0.0), atol=1e-5)


def test_forward_coordinate_warp_identity_and_translation():
    """Test 8: Verify forward coordinate warping phi(x) = x + u(x)."""
    coords = torch.randn(30, 2)

    # Identity warp (u = 0)
    u_zero = torch.zeros(16, 16, 2)
    warped_id = warp_scattered_coordinates(coords, u_zero, direction='forward')
    assert torch.allclose(warped_id, coords, atol=1e-7)

    # Constant translation (u = c)
    c = torch.tensor([0.1, -0.2])
    u_const = c.view(1, 1, 2).expand(16, 16, 2).contiguous()
    warped_const = warp_scattered_coordinates(coords, u_const, direction='forward')
    assert torch.allclose(warped_const, coords + c, atol=1e-6)


def test_backward_coordinate_warp_inversion():
    """Test 9: Verify backward coordinate warping phi^-1(y) = y + v(y)."""
    coords = torch.randn(25, 2)
    shift = torch.tensor([0.12, -0.08])
    u = shift.view(1, 1, 2).expand(16, 16, 2).contiguous()
    v = (-shift).view(1, 1, 2).expand(16, 16, 2).contiguous()

    # Forward warp
    warped = warp_scattered_coordinates(coords, u, direction='forward')
    # Backward warp with inverse field
    recovered = warp_scattered_coordinates(warped, v, direction='backward')
    assert torch.allclose(recovered, coords, atol=1e-6)


def test_anderson_inverse_consistency_scattered_criterion():
    """Test 10: Mandatory acceptance criterion: ||phi o phi^-1 - id||_inf < 1e-3 with Anderson acceleration."""
    H, W = 64, 64
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    # Smooth non-linear displacement field vanishing at boundaries
    envelope = (1.0 - gy**2) * (1.0 - gx**2)
    u_x = 0.05 * torch.sin(np.pi * gx) * envelope
    u_y = 0.05 * torch.cos(np.pi * gy) * envelope
    W_disp = torch.stack([u_x, u_y], dim=-1).unsqueeze(0)  # (1, 64, 64, 2)

    # Compute inverse via Anderson acceleration (steps=20, m=5)
    v = update_inverse_field_nd_anderson(W_disp, None, steps=20, m=5)

    # Sample N=500 scattered points in [-0.8, 0.8]^2
    torch.manual_seed(42)
    pts = (torch.rand(500, 2) * 1.6 - 0.8)

    # Forward warp: Y = phi(X)
    Y = warp_scattered_coordinates(pts, W_disp.squeeze(0), direction='forward')

    # Backward warp: X_rec = phi^-1(Y)
    X_rec = warp_scattered_coordinates(Y, v.squeeze(0), direction='backward')

    # Calculate L_inf error across all points
    err = torch.norm(X_rec - pts, p=float('inf'), dim=-1)
    max_err = float(err.max())
    mean_err = float(err.mean())

    assert max_err < 1.0e-3, f"Anderson inverse consistency failed: max error {max_err:.6e} >= 1e-3"
    assert mean_err < 2.0e-4, f"Anderson inverse mean error {mean_err:.6e} too high"


def test_autograd_gradcheck_evaluate_field_2d_and_3d():
    """Test 11: Verify analytical autograd gradients vs finite differences in double precision."""
    # 2D Gradcheck: Points positioned in cell interiors
    pts2d = torch.tensor([
        [-0.35, 0.15],
        [0.25, -0.45],
        [0.05, 0.05]
    ], dtype=torch.float64, requires_grad=True)

    field2d = torch.randn(1, 8, 8, 2, dtype=torch.float64, requires_grad=True) * 0.1

    def func2d(p, f):
        return evaluate_field_at_scattered(f, p)

    assert torch.autograd.gradcheck(func2d, (pts2d, field2d), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 3D Gradcheck: Interior points
    pts3d = torch.tensor([
        [0.12, -0.23, 0.34],
        [-0.45, 0.15, -0.22]
    ], dtype=torch.float64, requires_grad=True)

    field3d = torch.randn(1, 6, 6, 6, 3, dtype=torch.float64, requires_grad=True) * 0.1

    def func3d(p, f):
        return evaluate_field_at_scattered(f, p)

    assert torch.autograd.gradcheck(func3d, (pts3d, field3d), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_mapping_edge_cases_and_error_handling():
    """Test 12: Robustness against empty inputs, single points, dimension mismatches, and ScatteredWarper."""
    field = torch.randn(8, 8, 2)

    # Empty points (N = 0)
    empty_pts = torch.empty(0, 2)
    out_empty = evaluate_field_at_scattered(field, empty_pts)
    assert out_empty.shape == (0, 2)

    # Single point (N = 1)
    single_pt = torch.tensor([[0.0, 0.0]])
    out_single = evaluate_field_at_scattered(field, single_pt)
    assert out_single.shape == (1, 2)

    # Dimension mismatch: 3D points with 2D field raises ValueError
    pts_3d = torch.randn(10, 3)
    with pytest.raises(ValueError):
        evaluate_field_at_scattered(field, pts_3d)

    # Test ScatteredWarper module
    warper = ScatteredWarper(displacement_field=field)
    warped_mod = warper(single_pt)
    warped_func = warp_scattered_coordinates(single_pt, field, direction='forward')
    assert torch.allclose(warped_mod, warped_func, atol=1e-6)
