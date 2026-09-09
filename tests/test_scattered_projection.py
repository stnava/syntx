"""
tests.test_scattered_projection — Verification Suite for Scattered Data Projection
==================================================================================

Covers Milestone M1 (Features F1–F7, F19):
- Shape & dimensionality parity across 2D/3D, scalar/multichannel, batched/unbatched.
- Autograd differentiability (torch.autograd.gradcheck) in double precision (float64) for coordinates and features.
- Zero NaNs/Infs under unit, physical, auto bounds, and empty regions.
- Binary and continuous Eulerian domain masking.
- Point confidence weights with gradients.
- Chunk size invariance (unchunked vs chunked forward and backward).
- Batch independence.
- Pre-cached ScatteredProjector module equivalence and dtype conversion.
- Drop-in backward compatibility with sulceye.flatmap.utils.differentiable_grid_projection.
- Coordinate ordering and orientation ('xyz' vs 'zyx').
- Single-point exact value recovery.
- Edge cases (empty point clouds, degenerate points).
"""

import numpy as np
import pytest
import torch

from syntx.scattered.projection import (
    ProjectionConfig,
    project_scattered_to_grid,
    ScatteredProjector,
    differentiable_grid_projection,
    compute_adaptive_sigma,
)


def test_projection_shapes_2d():
    """Test 1: Output tensor dimensionality and shape conventions for 2D inputs."""
    points_unbatched = torch.randn(50, 2)
    values_scalar = torch.randn(50)
    values_multi = torch.randn(50, 3)
    points_batched = torch.randn(2, 50, 2)
    values_batched = torch.randn(2, 50, 4)

    # Unbatched scalar -> (1, 1, 32, 32)
    out1 = project_scattered_to_grid(points_unbatched, values_scalar, grid_shape=32)
    assert out1.shape == (1, 1, 32, 32)

    # Unbatched multi-channel with asymmetric grid -> (1, 3, 24, 32)
    out2 = project_scattered_to_grid(points_unbatched, values_multi, grid_shape=(24, 32))
    assert out2.shape == (1, 3, 24, 32)

    # Batched multi-channel -> (2, 4, 16, 16)
    out3 = project_scattered_to_grid(points_batched, values_batched, grid_shape=(16, 16))
    assert out3.shape == (2, 4, 16, 16)

    # return_density=True
    out4, den4 = project_scattered_to_grid(points_batched, values_batched, grid_shape=(16, 16), return_density=True)
    assert out4.shape == (2, 4, 16, 16)
    assert den4.shape == (2, 1, 16, 16)


def test_projection_shapes_3d():
    """Test 2: Output tensor shapes for 3D inputs mapping to 5D Eulerian volumes (B, C, D, H, W)."""
    points_3d = torch.randn(60, 3)
    vals_scalar_3d = torch.randn(60)
    vals_multi_3d = torch.randn(60, 2)
    pts_batch_3d = torch.randn(2, 60, 3)
    vals_batch_3d = torch.randn(2, 60, 1)

    # Scalar 3D -> (1, 1, 16, 16, 16)
    out1 = project_scattered_to_grid(points_3d, vals_scalar_3d, grid_shape=16)
    assert out1.shape == (1, 1, 16, 16, 16)

    # Multi-channel asymmetric 3D -> (1, 2, 8, 12, 16)
    out2 = project_scattered_to_grid(points_3d, vals_multi_3d, grid_shape=(8, 12, 16))
    assert out2.shape == (1, 2, 8, 12, 16)

    # Batched 3D with return_density
    out3, den3 = project_scattered_to_grid(pts_batch_3d, vals_batch_3d, grid_shape=(10, 10, 10), return_density=True)
    assert out3.shape == (2, 1, 10, 10, 10)
    assert den3.shape == (2, 1, 10, 10, 10)


def test_projection_finite_and_bounds():
    """Test 3: Guarantee zero NaNs, zero Infs under unit, physical, auto, and out-of-bound coordinates."""
    # Case A: Unit bounds with points outside
    pts_out = torch.tensor([[-2.5, -2.5], [0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)
    vals_out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out_unit = project_scattered_to_grid(pts_out, vals_out, grid_shape=16, domain_bounds=(-1.0, 1.0))
    assert torch.isfinite(out_unit).all() and not torch.isnan(out_unit).any()

    # Case B: Asymmetric physical millimeter bounding box
    pts_phys = torch.tensor([[10.0, 20.0], [30.0, 80.0]], dtype=torch.float32)
    vals_phys = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    out_phys = project_scattered_to_grid(
        pts_phys, vals_phys, grid_shape=(16, 16), domain_bounds=([0.0, 0.0], [50.0, 100.0])
    )
    assert torch.isfinite(out_phys).all() and not torch.isnan(out_phys).any()

    # Case C: Auto bounds
    out_auto = project_scattered_to_grid(pts_phys, vals_phys, grid_shape=(16, 16), domain_bounds='auto')
    assert torch.isfinite(out_auto).all() and not torch.isnan(out_auto).any()


def test_projection_empty_voxel_stability():
    """Test 4: Mathematical stability when voxels have zero density, check fill_value and autograd."""
    # Cluster points strictly in quadrant [0.6, 0.9] x [0.6, 0.9]
    torch.manual_seed(42)
    pts = torch.rand(20, 2) * 0.3 + 0.6
    vals = torch.ones(20, 1)

    # Default fill_value=0.0
    out, den = project_scattered_to_grid(
        pts, vals, grid_shape=16, domain_bounds=(-1.0, 1.0), sigma=0.05, return_density=True
    )
    # Voxel at (-0.8, -0.8) is index (1, 1)
    assert den[0, 0, 1, 1] < 1e-10
    assert out[0, 0, 1, 1] == 0.0
    assert torch.isfinite(out).all()

    # Custom fill_value=-999.0
    out_fill = project_scattered_to_grid(
        pts, vals, grid_shape=16, domain_bounds=(-1.0, 1.0), sigma=0.05, fill_value=-999.0
    )
    assert out_fill[0, 0, 1, 1] == -999.0

    # Autograd stability on empty voxels
    pts_grad = pts.clone().requires_grad_(True)
    vals_grad = vals.clone().requires_grad_(True)
    out_g = project_scattered_to_grid(pts_grad, vals_grad, grid_shape=16, domain_bounds=(-1.0, 1.0), sigma=0.05)
    loss = out_g.sum()
    loss.backward()
    assert torch.isfinite(pts_grad.grad).all()
    assert torch.isfinite(vals_grad.grad).all()


def test_projection_autograd_gradcheck_coordinates_and_features_2d():
    """Test 5: Verify analytical autograd gradients vs finite differences in double precision 2D."""
    pts = torch.tensor([
        [-0.4, -0.3], [0.2, 0.5], [-0.1, 0.2],
        [0.3, -0.4], [-0.5, 0.1], [0.4, 0.1]
    ], dtype=torch.float64, requires_grad=True)
    vals = torch.tensor([[1.2], [0.5], [2.1], [1.8], [0.7], [1.5]], dtype=torch.float64, requires_grad=True)

    def func(p, v):
        return project_scattered_to_grid(
            p, v, grid_shape=(6, 6), domain_bounds=(-1.0, 1.0), sigma=0.25, epsilon=1e-6
        )

    assert torch.autograd.gradcheck(func, (pts, vals), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)


def test_projection_autograd_gradcheck_3d():
    """Test 6: Verify 3D autograd differentiability for coordinates and features in double precision."""
    pts_3d = torch.tensor([
        [-0.3, 0.2, -0.1], [0.4, -0.2, 0.3],
        [-0.1, -0.4, 0.2], [0.2, 0.3, -0.3]
    ], dtype=torch.float64, requires_grad=True)
    vals_3d = torch.tensor([[1.0], [2.5], [0.8], [1.9]], dtype=torch.float64, requires_grad=True)

    def func_3d(p, v):
        return project_scattered_to_grid(
            p, v, grid_shape=(4, 4, 4), domain_bounds=(-1.0, 1.0), sigma=0.35, epsilon=1e-6
        )

    assert torch.autograd.gradcheck(func_3d, (pts_3d, vals_3d), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)


def test_projection_domain_masks():
    """Test 7: Verify binary and continuous Eulerian domain mask behavior."""
    torch.manual_seed(42)
    pts = torch.randn(30, 2) * 0.4
    vals = torch.randn(30, 1)

    y, x = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16), indexing='ij')
    bin_mask = ((x ** 2 + y ** 2) <= 0.36).float()
    cont_mask = torch.clamp(1.0 - torch.sqrt(x ** 2 + y ** 2), 0.0, 1.0)

    # Binary mask
    out_bin = project_scattered_to_grid(pts, vals, grid_shape=16, mask=bin_mask)
    assert (out_bin[:, :, bin_mask == 0] == 0.0).all()

    # Continuous mask
    out_unmasked = project_scattered_to_grid(pts, vals, grid_shape=16)
    out_cont = project_scattered_to_grid(pts, vals, grid_shape=16, mask=cont_mask)
    assert torch.allclose(out_cont, out_unmasked * cont_mask, atol=1e-6)


def test_projection_point_weights():
    """Test 8: Verify point confidence weights and weight differentiability."""
    pts = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    vals = torch.tensor([[10.0], [20.0]])

    # Equal weights -> 15.0
    w_eq = torch.tensor([1.0, 1.0])
    out_eq = project_scattered_to_grid(pts, vals, grid_shape=11, point_weights=w_eq)
    assert torch.isclose(out_eq[0, 0, 5, 5], torch.tensor(15.0), atol=1e-3)

    # Unequal weights (3, 1) -> (3*10 + 20)/4 = 12.5
    w_uneq = torch.tensor([3.0, 1.0])
    out_uneq = project_scattered_to_grid(pts, vals, grid_shape=11, point_weights=w_uneq)
    assert torch.isclose(out_uneq[0, 0, 5, 5], torch.tensor(12.5), atol=1e-3)

    # Zero weight omission -> 10.0
    w_zero = torch.tensor([1.0, 0.0])
    out_zero = project_scattered_to_grid(pts, vals, grid_shape=11, point_weights=w_zero)
    assert torch.isclose(out_zero[0, 0, 5, 5], torch.tensor(10.0), atol=1e-3)

    # Differentiability with respect to weights
    w_grad = torch.tensor([1.0, 2.0], requires_grad=True)
    out = project_scattered_to_grid(pts, vals, grid_shape=11, point_weights=w_grad)
    out.sum().backward()
    assert torch.isfinite(w_grad.grad).all()


def test_projection_chunk_size_invariance():
    """Test 9: Verify that chunk size does not affect forward or backward numerical output."""
    torch.manual_seed(42)

    # Double precision (float64) check: bitwise exact invariance
    pts1_64 = torch.randn(40, 2, dtype=torch.float64, requires_grad=True)
    vals1_64 = torch.randn(40, 2, dtype=torch.float64, requires_grad=True)
    pts2_64 = pts1_64.clone().detach().requires_grad_(True)
    vals2_64 = vals1_64.clone().detach().requires_grad_(True)
    pts3_64 = pts1_64.clone().detach().requires_grad_(True)
    vals3_64 = vals1_64.clone().detach().requires_grad_(True)

    out_un_64 = project_scattered_to_grid(pts1_64, vals1_64, grid_shape=24, chunk_size=576)
    out_17_64 = project_scattered_to_grid(pts2_64, vals2_64, grid_shape=24, chunk_size=17)
    out_1_64 = project_scattered_to_grid(pts3_64, vals3_64, grid_shape=24, chunk_size=1)

    assert torch.allclose(out_17_64, out_un_64, atol=1e-12, rtol=1e-10)
    assert torch.allclose(out_1_64, out_un_64, atol=1e-12, rtol=1e-10)

    out_un_64.sum().backward()
    out_17_64.sum().backward()
    assert torch.allclose(pts2_64.grad, pts1_64.grad, atol=1e-10, rtol=1e-8)
    assert torch.allclose(vals2_64.grad, vals1_64.grad, atol=1e-10, rtol=1e-8)

    # Single precision (float32) check: tolerance for BLAS tiling roundoff
    pts1 = torch.randn(40, 2, requires_grad=True)
    vals1 = torch.randn(40, 2, requires_grad=True)
    pts2 = pts1.clone().detach().requires_grad_(True)
    vals2 = vals1.clone().detach().requires_grad_(True)
    pts3 = pts1.clone().detach().requires_grad_(True)
    vals3 = vals1.clone().detach().requires_grad_(True)

    out_un = project_scattered_to_grid(pts1, vals1, grid_shape=24, chunk_size=576)
    out_17 = project_scattered_to_grid(pts2, vals2, grid_shape=24, chunk_size=17)
    out_1 = project_scattered_to_grid(pts3, vals3, grid_shape=24, chunk_size=1)

    assert torch.allclose(out_17, out_un, atol=1e-4, rtol=1e-3)
    assert torch.allclose(out_1, out_un, atol=1e-4, rtol=1e-3)

    out_un.sum().backward()
    out_17.sum().backward()
    assert torch.allclose(pts2.grad, pts1.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(vals2.grad, vals1.grad, atol=1e-3, rtol=1e-3)


def test_projection_batch_handling():
    """Test 10: Verify multi-batch handling (B > 1) and batch independence."""
    torch.manual_seed(42)
    pts0, vals0 = torch.randn(20, 2), torch.randn(20, 1)
    pts1, vals1 = torch.randn(20, 2) + 0.5, torch.randn(20, 1) * 2.0
    pts2, vals2 = torch.randn(20, 2) - 0.5, torch.randn(20, 1) + 1.0

    pts_b = torch.stack([pts0, pts1, pts2], dim=0)
    vals_b = torch.stack([vals0, vals1, vals2], dim=0)

    out_b = project_scattered_to_grid(pts_b, vals_b, grid_shape=16)
    out0 = project_scattered_to_grid(pts0, vals0, grid_shape=16)
    out1 = project_scattered_to_grid(pts1, vals1, grid_shape=16)
    out2 = project_scattered_to_grid(pts2, vals2, grid_shape=16)

    assert torch.allclose(out_b[0], out0[0], atol=1e-6)
    assert torch.allclose(out_b[1], out1[0], atol=1e-6)
    assert torch.allclose(out_b[2], out2[0], atol=1e-6)


def test_scattered_projector_caching():
    """Test 11: Verify ScatteredProjector module buffering, .to(dtype), and functional equivalence."""
    torch.manual_seed(42)
    projector = ScatteredProjector(grid_shape=(20, 20), domain_bounds=(-1.0, 1.0), sigma=0.05)
    pts = torch.randn(30, 2)
    vals = torch.randn(30, 2)

    assert isinstance(projector, torch.nn.Module)
    assert hasattr(projector, 'grid_coords')
    assert hasattr(projector, 'Y_scaled')

    out_mod = projector(pts, vals)
    out_fn = project_scattered_to_grid(pts, vals, grid_shape=(20, 20), domain_bounds=(-1.0, 1.0), sigma=0.05)
    assert torch.allclose(out_mod, out_fn, atol=1e-6)

    # Dtype conversion
    projector_64 = projector.to(torch.float64)
    out_64 = projector_64(pts.to(torch.float64), vals.to(torch.float64))
    assert out_64.dtype == torch.float64


def test_differentiable_grid_projection_dropin_alias():
    """Test 12: Verify drop-in backward compatibility with sulceye signature and numpy arrays."""
    torch.manual_seed(42)
    u_torch = torch.randn(25, 2)
    v_torch = torch.randn(25)

    # Positional args
    res_pos = differentiable_grid_projection(u_torch, v_torch, 16, 0.05)
    assert res_pos.shape == (1, 1, 16, 16)

    # Keyword args and return_density
    res, den = differentiable_grid_projection(u_torch, v_torch, grid_res=16, sigma=0.05, return_density=True)
    assert res.shape == (1, 1, 16, 16) and den.shape == (1, 1, 16, 16)

    # Numpy input compatibility
    np.random.seed(42)
    u_np = np.random.randn(25, 2).astype(np.float32)
    v_np = np.random.randn(25).astype(np.float32)
    res_np = differentiable_grid_projection(u_np, v_np, grid_res=16)
    assert isinstance(res_np, torch.Tensor)
    assert res_np.shape == (1, 1, 16, 16)


def test_projection_coordinate_conventions():
    """Test 13: Verify that 'xyz' maps X to columns and Y to rows, and 'zyx' matches tensor indices."""
    # Point at (x=0.8, y=0.0) on [-1, 1]^2 with grid_shape=(21, 21)
    pt = torch.tensor([[0.8, 0.0]])
    val = torch.tensor([[10.0]])

    # Under 'xyz':
    # x=0.8 is column index 18 (20 * 1.8 / 2.0 = 18)
    # y=0.0 is row index 10 (20 * 1.0 / 2.0 = 10)
    out_xyz = project_scattered_to_grid(
        pt, val, grid_shape=21, domain_bounds=(-1.0, 1.0), sigma=0.05, coord_convention='xyz'
    )
    max_idx = torch.argmax(out_xyz[0, 0])
    row = max_idx.item() // 21
    col = max_idx.item() % 21
    assert row == 10
    assert col in (18, 19)

    # Under 'zyx':
    # Coordinate 0 is row (0.8 -> row 18)
    # Coordinate 1 is col (0.0 -> col 10)
    out_zyx = project_scattered_to_grid(
        pt, val, grid_shape=21, domain_bounds=(-1.0, 1.0), sigma=0.05, coord_convention='zyx'
    )
    max_idx_zyx = torch.argmax(out_zyx[0, 0])
    row_zyx = max_idx_zyx.item() // 21
    col_zyx = max_idx_zyx.item() % 21
    assert row_zyx in (18, 19)
    assert col_zyx == 10


def test_projection_exact_single_point_recovery():
    """Test 14: Center point placed on grid node recovers exact feature value."""
    pt = torch.tensor([[0.0, 0.0]])
    val = torch.tensor([[7.5]])
    out = project_scattered_to_grid(pt, val, grid_shape=11, domain_bounds=(-1.0, 1.0), sigma=0.02)
    # Grid 11x11 on [-1, 1] has center voxel (5, 5) at exactly (0.0, 0.0)
    assert torch.isclose(out[0, 0, 5, 5], torch.tensor(7.5), atol=1e-3)


def test_projection_edge_cases_and_config():
    """Verify empty point sets, degenerate points, adaptive sigma, and ProjectionConfig."""
    # Empty point cloud (N=0)
    pts_empty = torch.empty((0, 2), dtype=torch.float32)
    vals_empty = torch.empty((0, 1), dtype=torch.float32)
    out_empty = project_scattered_to_grid(pts_empty, vals_empty, grid_shape=16)
    assert out_empty.shape == (1, 1, 16, 16)
    assert (out_empty == 0.0).all()
    assert torch.isfinite(out_empty).all()

    # Degenerate zero-spread points
    pts_degen = torch.full((5, 2), 0.5, dtype=torch.float32)
    vals_degen = torch.ones((5, 1), dtype=torch.float32)
    out_degen = project_scattered_to_grid(pts_degen, vals_degen, grid_shape=16, domain_bounds='auto')
    assert torch.isfinite(out_degen).all()

    # Adaptive sigma
    pts_cloud = torch.randn(100, 2) * 0.3
    vals_cloud = torch.randn(100, 1)
    sigma_val = compute_adaptive_sigma(pts_cloud)
    assert 0.01 <= sigma_val <= 0.20
    out_auto_sigma = project_scattered_to_grid(pts_cloud, vals_cloud, grid_shape=16, sigma='auto')
    assert torch.isfinite(out_auto_sigma).all()

    # ProjectionConfig passing
    cfg = ProjectionConfig(grid_shape=16, domain_bounds=(-1.0, 1.0), sigma=0.04, fill_value=1.5)
    out_cfg = project_scattered_to_grid(pts_cloud, vals_cloud, config=cfg)
    assert out_cfg.shape == (1, 1, 16, 16)


def test_projection_validation_and_errors():
    """Verify input validation, error handling, and coverage edge cases."""
    pts = torch.randn(10, 2)
    vals = torch.randn(10)

    # 1. epsilon <= 0
    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        project_scattered_to_grid(pts, vals, epsilon=0.0)

    # 2. sigma <= 0
    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        project_scattered_to_grid(pts, vals, sigma=0.0)
    with pytest.raises(ValueError, match="All components of sigma must be strictly positive"):
        project_scattered_to_grid(pts, vals, sigma=(-0.1, 0.1))

    # 3. invalid coord_convention
    with pytest.raises(ValueError, match="Unknown coord_convention"):
        project_scattered_to_grid(pts, vals, coord_convention='bad')

    # 4. grid_shape dimension mismatch
    with pytest.raises(ValueError, match="grid_shape length"):
        project_scattered_to_grid(pts, vals, grid_shape=(16, 16, 16))

    # 5. values length mismatch
    with pytest.raises(ValueError, match="values length"):
        project_scattered_to_grid(pts, torch.randn(5))
    with pytest.raises(ValueError, match="values shape"):
        project_scattered_to_grid(pts, torch.randn(5, 2))

    # 6. point_weights length mismatch
    with pytest.raises(ValueError, match="point_weights length"):
        project_scattered_to_grid(pts, vals, point_weights=torch.ones(5))

    # 7. Unsupported domain_bounds
    with pytest.raises(ValueError, match="Unsupported domain_bounds specification"):
        project_scattered_to_grid(pts, vals, domain_bounds=12345)

    # 8. Adaptive sigma edge cases (N=1 and large N)
    assert compute_adaptive_sigma(torch.randn(1, 2)) == 0.01
    s_large = compute_adaptive_sigma(torch.randn(2500, 2))
    assert 0.01 <= s_large <= 0.20

    # 9. 1D point_weights
    out_pw = project_scattered_to_grid(pts, vals, grid_shape=16, point_weights=torch.ones(10))
    assert out_pw.shape == (1, 1, 16, 16)

    # 10. Dynamic ScatteredProjector with domain_bounds='auto'
    proj_dyn = ScatteredProjector(grid_shape=16, domain_bounds='auto')
    out_dyn = proj_dyn(pts, vals)
    assert out_dyn.shape == (1, 1, 16, 16)

