"""
tests.test_scattered_pullback_pushforward — Verification Suite for Feature Transport
===================================================================================

Covers Milestone M2 (Features F10, F11, F12, F21):
- Grid-to-scattered pullback: Phi^* G_B(x) = G_B(phi(x)).
- Scattered-to-grid pushforward: Phi_* F_A via warped coordinates + Nadaraya-Watson regression.
- Point-to-point transport: Direct Lagrangian transport vs Eulerian lattice bridge.
- Chunking invariance and memory-bounded execution.
- Cycle consistency criterion: pushforward -> pullback -> assert Pearson correlation r > 0.98.
- Autograd gradcheck in double precision (float64) for pullback, pushforward, and transport.
- Binary and continuous Eulerian domain masking and physical bounds.
"""

import numpy as np
import pytest
import torch

from syntx.scattered.transport import (
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
)


def test_pullback_grid_to_scattered_shapes_2d_and_3d():
    """Test 1: Output tensor shapes for grid-to-scattered pullback across 2D/3D and channels."""
    # 2D Batched Multi-Channel
    grid_2d = torch.randn(2, 4, 32, 32)
    coords_2d = torch.randn(2, 60, 2)
    disp_2d = torch.randn(2, 32, 32, 2) * 0.02
    out1 = pullback_grid_to_scattered(grid_2d, coords_2d, disp_2d)
    assert out1.shape == (2, 60, 4)

    # 2D Unbatched
    grid_u = torch.randn(4, 32, 32)
    coords_u = torch.randn(60, 2)
    out2 = pullback_grid_to_scattered(grid_u, coords_u)
    assert out2.shape == (60, 4)

    # 3D Batched
    grid_3d = torch.randn(2, 3, 16, 16, 16)
    coords_3d = torch.randn(2, 40, 3)
    disp_3d = torch.randn(2, 16, 16, 16, 3) * 0.02
    out3 = pullback_grid_to_scattered(grid_3d, coords_3d, disp_3d)
    assert out3.shape == (2, 40, 3)


def test_pullback_identity_and_translation():
    """Test 2: Pullback under identity map and constant translations."""
    # Linear feature grid: G(x, y) = x + y
    H, W = 16, 16
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    grid = (gx + gy).unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 16)

    coords = torch.tensor([[0.2, 0.3]], dtype=torch.float32)

    # Identity pullback: G(0.2, 0.3) = 0.5
    out_id = pullback_grid_to_scattered(grid, coords)
    assert torch.isclose(out_id[0, 0], torch.tensor(0.5), atol=1e-5)

    # Pullback under translation c = [0.1, -0.1]: G(0.3, 0.2) = 0.5
    disp = torch.zeros(1, 16, 16, 2)
    disp[..., 0] = 0.1
    disp[..., 1] = -0.1
    out_trans = pullback_grid_to_scattered(grid, coords, disp)
    assert torch.isclose(out_trans[0, 0], torch.tensor(0.5), atol=1e-5)


def test_pushforward_scattered_to_grid_shapes_2d_and_3d():
    """Test 3: Output tensor shapes for scattered-to-grid pushforward."""
    # 2D Pushforward
    coords_2d = torch.randn(2, 50, 2)
    feats_2d = torch.randn(2, 50, 3)
    disp_2d = torch.randn(2, 24, 24, 2) * 0.02

    out2d, den2d = pushforward_scattered_to_grid(
        coords_2d, feats_2d, disp_2d, grid_shape=(24, 24), return_density=True
    )
    assert out2d.shape == (2, 3, 24, 24)
    assert den2d.shape == (2, 1, 24, 24)

    # 3D Pushforward
    coords_3d = torch.randn(2, 40, 3)
    feats_3d = torch.randn(2, 40, 2)
    out3d = pushforward_scattered_to_grid(coords_3d, feats_3d, grid_shape=(12, 12, 12))
    assert out3d.shape == (2, 2, 12, 12, 12)


def test_pushforward_mass_and_mean_conservation():
    """Test 4: Pushforward preserves constant intensive scalar feature across dense regions."""
    torch.manual_seed(42)
    coords = torch.rand(400, 2) * 1.4 - 0.7  # dense in [-0.7, 0.7]^2
    feats = torch.full((400, 1), 3.5, dtype=torch.float32)

    grid, density = pushforward_scattered_to_grid(
        coords, feats, grid_shape=(32, 32), sigma=0.08, return_density=True
    )

    # In voxels with substantial point support (density > 1e-3), value must equal 3.5
    dense_mask = density[0, 0] > 1e-3
    assert dense_mask.sum() > 50
    assert torch.allclose(grid[0, 0][dense_mask], torch.tensor(3.5), atol=1e-3)


def test_transport_scattered_to_scattered_direct_and_bridge():
    """Test 5: Compare Method A (direct Lagrangian) and Method B (grid bridge) mutual correlation."""
    torch.manual_seed(42)
    coords_src = torch.rand(200, 2) * 1.4 - 0.7
    feats_src = torch.sin(np.pi * coords_src[:, 0:1]) * torch.cos(np.pi * coords_src[:, 1:2])
    coords_tgt = torch.rand(150, 2) * 1.2 - 0.6

    disp = torch.randn(1, 32, 32, 2) * 0.02

    f_direct = transport_scattered_to_scattered(
        coords_src, feats_src, coords_tgt, disp, sigma=0.05, method='direct'
    )
    f_bridge = transport_scattered_to_scattered(
        coords_src, feats_src, coords_tgt, disp, sigma=0.05, method='grid_bridge', grid_shape=(64, 64)
    )

    # Assert high Pearson correlation between both implementations
    fd_c = f_direct.squeeze() - f_direct.mean()
    fb_c = f_bridge.squeeze() - f_bridge.mean()
    r = torch.sum(fd_c * fb_c) / (torch.norm(fd_c) * torch.norm(fb_c))
    assert r > 0.95, f"Direct vs Bridge transport correlation {r:.4f} <= 0.95"


def test_transport_scattered_to_scattered_shapes_and_chunking():
    """Test 6: Chunking invariance and batched support for point-to-point transport."""
    coords_src = torch.randn(50, 2)
    feats_src = torch.randn(50, 3)
    coords_tgt = torch.randn(40, 2)

    # Chunked evaluation
    out_chunked = transport_scattered_to_scattered(
        coords_src, feats_src, coords_tgt, sigma=0.1, chunk_size=10
    )
    # Unchunked evaluation
    out_unchunked = transport_scattered_to_scattered(
        coords_src, feats_src, coords_tgt, sigma=0.1, chunk_size=0
    )

    assert out_chunked.shape == (40, 3)
    assert torch.allclose(out_chunked, out_unchunked, atol=1e-6)


def test_cycle_consistency_pushforward_pullback_identity():
    """Test 7: Mandatory acceptance criterion: Pushforward -> Pullback Pearson r > 0.98 under identity."""
    torch.manual_seed(42)
    N = 1000
    coords = torch.rand(N, 2) * 1.4 - 0.7  # in [-0.7, 0.7]^2
    feats = (torch.sin(2.0 * np.pi * coords[:, 0]) + torch.cos(2.0 * np.pi * coords[:, 1])).unsqueeze(-1)

    grid = pushforward_scattered_to_grid(coords, feats, grid_shape=(64, 64), sigma=0.04)
    recovered = pullback_grid_to_scattered(grid, coords)

    f_orig = feats.squeeze()
    f_rec = recovered.squeeze()
    fo_c = f_orig - f_orig.mean()
    fr_c = f_rec - f_rec.mean()
    r = float(torch.sum(fo_c * fr_c) / (torch.norm(fo_c) * torch.norm(fr_c)))

    assert r > 0.98, f"Cycle consistency identity correlation {r:.4f} <= 0.98"


def test_cycle_consistency_pushforward_pullback_warped():
    """Test 8: Mandatory acceptance criterion: Pushforward -> Pullback Pearson r > 0.98 under smooth warp."""
    torch.manual_seed(42)
    N = 1000
    coords = torch.rand(N, 2) * 1.4 - 0.7
    feats = (torch.sin(2.0 * np.pi * coords[:, 0]) + torch.cos(2.0 * np.pi * coords[:, 1])).unsqueeze(-1)

    # Smooth non-zero displacement field
    H, W = 64, 64
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    u_x = 0.05 * torch.sin(np.pi * gx)
    u_y = 0.05 * torch.cos(np.pi * gy)
    disp = torch.stack([u_x, u_y], dim=-1).unsqueeze(0)  # (1, 64, 64, 2)

    # Push forward to warped grid
    grid_warp = pushforward_scattered_to_grid(coords, feats, disp, grid_shape=(64, 64), sigma=0.04)
    # Pull back from warped grid
    recovered_warp = pullback_grid_to_scattered(grid_warp, coords, disp)

    f_orig = feats.squeeze()
    f_rec = recovered_warp.squeeze()
    fo_c = f_orig - f_orig.mean()
    fr_c = f_rec - f_rec.mean()
    r = float(torch.sum(fo_c * fr_c) / (torch.norm(fo_c) * torch.norm(fr_c)))

    assert r > 0.98, f"Cycle consistency warped correlation {r:.4f} <= 0.98"


def test_cycle_consistency_3d():
    """Test 9: 3D Pushforward -> Pullback cycle consistency (r > 0.95)."""
    torch.manual_seed(42)
    N = 600
    coords = torch.rand(N, 3) * 1.4 - 0.7
    feats = (torch.sin(np.pi * coords[:, 0]) * torch.cos(np.pi * coords[:, 1]) * torch.sin(np.pi * coords[:, 2])).unsqueeze(-1)

    grid = pushforward_scattered_to_grid(coords, feats, grid_shape=(32, 32, 32), sigma=0.06)
    recovered = pullback_grid_to_scattered(grid, coords)

    f_orig = feats.squeeze()
    f_rec = recovered.squeeze()
    fo_c = f_orig - f_orig.mean()
    fr_c = f_rec - f_rec.mean()
    r = float(torch.sum(fo_c * fr_c) / (torch.norm(fo_c) * torch.norm(fr_c)))

    assert r > 0.95, f"3D cycle consistency correlation {r:.4f} <= 0.95"


def test_autograd_gradcheck_pullback():
    """Test 10: Autograd gradcheck for grid-to-scattered pullback."""
    pts = torch.tensor([
        [-0.35, 0.15],
        [0.25, -0.45],
        [0.05, 0.05]
    ], dtype=torch.float64, requires_grad=True)

    grid_feat = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    disp = torch.randn(1, 8, 8, 2, dtype=torch.float64, requires_grad=True) * 0.05

    def func(g, p, d):
        return pullback_grid_to_scattered(g, p, d)

    assert torch.autograd.gradcheck(func, (grid_feat, pts, disp), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_autograd_gradcheck_pushforward():
    """Test 11: Autograd gradcheck for scattered-to-grid pushforward."""
    pts = torch.tensor([
        [-0.35, 0.15],
        [0.25, -0.45],
        [0.05, 0.05]
    ], dtype=torch.float64, requires_grad=True)

    feats = torch.tensor([
        [1.0, 2.0],
        [0.5, 1.5],
        [2.0, 0.2]
    ], dtype=torch.float64, requires_grad=True)

    disp = torch.randn(1, 6, 6, 2, dtype=torch.float64, requires_grad=True) * 0.05

    def func(p, f, d):
        return pushforward_scattered_to_grid(p, f, d, grid_shape=(6, 6), sigma=0.3, epsilon=1e-5)

    assert torch.autograd.gradcheck(func, (pts, feats, disp), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_domain_mask_and_bounds_in_transport():
    """Test 12: Verify Eulerian masking and physical domain bounds in pushforward and pullback."""
    coords = torch.rand(100, 2) * 1.6 - 0.8
    feats = torch.ones(100, 1)

    # Circular mask: radius <= 0.5
    y = torch.linspace(-1, 1, 32)
    x = torch.linspace(-1, 1, 32)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    mask = ((gx**2 + gy**2) <= 0.5**2).float()

    grid = pushforward_scattered_to_grid(coords, feats, grid_shape=(32, 32), domain_mask=mask, sigma=0.05)
    # Outside mask must be strictly 0.0
    assert (grid[:, :, mask == 0.0] == 0.0).all()

    # Pullback respects masked values
    out_pull = pullback_grid_to_scattered(grid, coords)
    assert torch.isfinite(out_pull).all()


# ===========================================================================
# Milestone M2 Remediation: Batch Dimension Broadcasting Verification Suite
# ===========================================================================


def test_pullback_grid_to_scattered_batch_broadcasting():
    """Test 13: Pullback batch broadcasting across unbatched coords, batched fields, and batched grids."""
    # Case A: 2D unbatched coords (N, d) + batched displacement (B, H, W, d) + unbatched grid (C, H, W) -> (B, N, C)
    coords_2d = torch.randn(25, 2)
    disp_2d = torch.randn(3, 16, 16, 2) * 0.02
    grid_2d_u = torch.randn(4, 16, 16)
    out_a = pullback_grid_to_scattered(grid_2d_u, coords_2d, disp_2d)
    assert out_a.shape == (3, 25, 4), f"Expected (3, 25, 4), got {out_a.shape}"

    # Numerical equivalence: each batch slice must match individual unbatched slice evaluation
    for b in range(3):
        slice_out = pullback_grid_to_scattered(grid_2d_u, coords_2d, disp_2d[b:b+1])
        assert torch.allclose(out_a[b], slice_out, atol=1e-6)

    # Case B: 2D unbatched coords (N, d) + batched grid (B, C, H, W) + disp=None -> (B, N, C)
    grid_2d_b = torch.randn(3, 4, 16, 16)
    out_b = pullback_grid_to_scattered(grid_2d_b, coords_2d, None)
    assert out_b.shape == (3, 25, 4), f"Expected (3, 25, 4), got {out_b.shape}"

    for b in range(3):
        slice_grid = pullback_grid_to_scattered(grid_2d_b[b], coords_2d, None)
        assert torch.allclose(out_b[b], slice_grid, atol=1e-6)

    # Case C: 2D unbatched coords (N, d) + batched displacement (B, H, W, d) + batched grid (B, C, H, W) -> (B, N, C)
    out_c = pullback_grid_to_scattered(grid_2d_b, coords_2d, disp_2d)
    assert out_c.shape == (3, 25, 4), f"Expected (3, 25, 4), got {out_c.shape}"

    # Case D: 2D unbatched coords (N, d) + batched displacement (B, H, W, d) + unchanneled grid (H, W) -> (B, N)
    grid_scalar = torch.randn(16, 16)
    out_d = pullback_grid_to_scattered(grid_scalar, coords_2d, disp_2d)
    assert out_d.shape == (3, 25), f"Expected (3, 25), got {out_d.shape}"

    # Case E: 3D unbatched coords (N, 3) + batched displacement (B, D, H, W, 3) + unbatched grid (C, D, H, W) -> (B, N, C)
    coords_3d = torch.randn(20, 3)
    disp_3d = torch.randn(2, 8, 8, 8, 3) * 0.02
    grid_3d_u = torch.randn(3, 8, 8, 8)
    out_3d = pullback_grid_to_scattered(grid_3d_u, coords_3d, disp_3d)
    assert out_3d.shape == (2, 20, 3), f"Expected (2, 20, 3), got {out_3d.shape}"

    # Case F: Error handling on conflicting batch dimensions (e.g. grid batch 3 vs disp batch 4)
    disp_mismatch = torch.randn(4, 16, 16, 2)
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        pullback_grid_to_scattered(grid_2d_b, coords_2d, disp_mismatch)


def test_pushforward_scattered_to_grid_batch_broadcasting():
    """Test 14: Pushforward batch broadcasting across unbatched coords, batched features, and batched fields."""
    coords = torch.randn(30, 2)
    feats_b = torch.randn(3, 30, 2)

    # Case A: unbatched coords (N, d) + batched features (B, N, C) + disp=None -> (B, C, *grid_shape)
    grid_a, den_a = pushforward_scattered_to_grid(
        coords, feats_b, displacement_field=None, grid_shape=(20, 20), return_density=True
    )
    assert grid_a.shape == (3, 2, 20, 20), f"Expected (3, 2, 20, 20), got {grid_a.shape}"
    assert den_a.shape == (3, 1, 20, 20), f"Expected (3, 1, 20, 20), got {den_a.shape}"

    # Numerical equivalence against individual slices
    for b in range(3):
        single_grid, _ = pushforward_scattered_to_grid(
            coords, feats_b[b], displacement_field=None, grid_shape=(20, 20), return_density=True
        )
        assert torch.allclose(grid_a[b], single_grid, atol=1e-6)

    # Case B: unbatched coords (N, d) + batched features (B, N, C) + batched displacement (B, H, W, d)
    disp_b = torch.randn(3, 20, 20, 2) * 0.02
    grid_b = pushforward_scattered_to_grid(
        coords, feats_b, displacement_field=disp_b, grid_shape=(20, 20)
    )
    assert grid_b.shape == (3, 2, 20, 20), f"Expected (3, 2, 20, 20), got {grid_b.shape}"

    # Case C: unbatched coords (N, d) + unbatched features (N, C) + batched displacement (B, H, W, d)
    feats_u = torch.randn(30, 2)
    grid_c = pushforward_scattered_to_grid(
        coords, feats_u, displacement_field=disp_b, grid_shape=(20, 20)
    )
    assert grid_c.shape == (3, 2, 20, 20), f"Expected (3, 2, 20, 20), got {grid_c.shape}"

    # Case D: 3D unbatched coords (N, 3) + batched features (B, N, C) -> (B, C, D, H, W)
    coords_3d = torch.randn(25, 3)
    feats_3d_b = torch.randn(2, 25, 4)
    grid_3d = pushforward_scattered_to_grid(
        coords_3d, feats_3d_b, displacement_field=None, grid_shape=(10, 10, 10)
    )
    assert grid_3d.shape == (2, 4, 10, 10, 10), f"Expected (2, 4, 10, 10, 10), got {grid_3d.shape}"


def test_transport_scattered_to_scattered_batch_broadcasting():
    """Test 15: Point-to-point transport batch broadcasting and autograd warning hygiene."""
    import warnings

    torch.manual_seed(42)
    coords_s = torch.rand(20, 2) * 1.4 - 0.7
    feats_s = torch.randn(20, 3)
    coords_t = torch.rand(15, 2) * 1.4 - 0.7
    disp_tr = torch.randn(3, 16, 16, 2) * 0.02

    # Case A: Direct Lagrangian transport with unbatched coords + batched displacement
    tr_dir = transport_scattered_to_scattered(
        coords_s, feats_s, coords_t, disp_tr, method='direct', sigma=0.05
    )
    assert tr_dir.shape == (3, 15, 3), f"Expected (3, 15, 3), got {tr_dir.shape}"

    for b in range(3):
        slice_dir = transport_scattered_to_scattered(
            coords_s, feats_s, coords_t, disp_tr[b:b+1], method='direct', sigma=0.05
        )
        assert torch.allclose(tr_dir[b], slice_dir, atol=1e-6)

    # Case B: Grid bridge transport with unbatched coords + batched displacement
    tr_br = transport_scattered_to_scattered(
        coords_s, feats_s, coords_t, disp_tr, method='grid_bridge', grid_shape=(16, 16),
        domain_bounds=(-1.0, 1.0), sigma=0.05
    )
    assert tr_br.shape == (3, 15, 3), f"Expected (3, 15, 3), got {tr_br.shape}"

    for b in range(3):
        slice_br = transport_scattered_to_scattered(
            coords_s, feats_s, coords_t, disp_tr[b:b+1], method='grid_bridge', grid_shape=(16, 16),
            domain_bounds=(-1.0, 1.0), sigma=0.05
        )
        assert torch.allclose(tr_br[b], slice_br, atol=1e-6)

    # Case C: Batched source features (B, N_src, C) with unbatched coordinates
    feats_s_b = torch.randn(3, 20, 3)
    tr_feats = transport_scattered_to_scattered(
        coords_s, feats_s_b, coords_t, displacement_field=None, method='direct', sigma=0.05
    )
    assert tr_feats.shape == (3, 15, 3), f"Expected (3, 15, 3), got {tr_feats.shape}"

    # Case D: 3D point-to-point transport
    coords_s_3d = torch.rand(20, 3) * 1.4 - 0.7
    feats_s_3d = torch.randn(20, 2)
    coords_t_3d = torch.rand(12, 3) * 1.4 - 0.7
    disp_3d = torch.randn(2, 8, 8, 8, 3) * 0.02
    tr_3d = transport_scattered_to_scattered(
        coords_s_3d, feats_s_3d, coords_t_3d, disp_3d, method='direct', sigma=0.08
    )
    assert tr_3d.shape == (2, 12, 2), f"Expected (2, 12, 2), got {tr_3d.shape}"

    # Case E: Autograd UserWarning hygiene check
    coords_s_g = torch.randn(10, 2, requires_grad=True)
    feats_s_g = torch.randn(10, 2, requires_grad=True)
    coords_t_g = torch.randn(8, 2, requires_grad=True)
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter('always')
        out_grad = transport_scattered_to_scattered(
            coords_s_g, feats_s_g, coords_t_g, method='grid_bridge', grid_shape=(16, 16)
        )
        out_grad.sum().backward()

    # Must emit zero UserWarnings regarding scalar conversion of requires_grad tensors
    grad_warnings = [
        w for w in recorded_warnings
        if issubclass(w.category, UserWarning) and "requires_grad=True" in str(w.message)
    ]
    assert len(grad_warnings) == 0, f"Unexpected autograd UserWarning detected: {grad_warnings}"


def test_autograd_gradcheck_batch_broadcasting():
    """Test 16: Analytical autograd gradchecks for batched/unbatched broadcasting in double precision."""
    # Points placed safely in cell interiors of an 8x8 grid
    pts_grad = torch.tensor([
        [-0.35, 0.15],
        [0.25, -0.45],
        [0.05, 0.05]
    ], dtype=torch.float64, requires_grad=True)

    # 1. Pullback gradcheck: unbatched coords (N, d) + batched displacement (B, H, W, d) + unbatched grid (C, H, W)
    grid_grad_u = torch.randn(2, 8, 8, dtype=torch.float64, requires_grad=True)
    disp_grad = torch.randn(2, 8, 8, 2, dtype=torch.float64, requires_grad=True) * 0.02

    def func_pb_disp(g, p, d):
        return pullback_grid_to_scattered(g, p, d)

    assert torch.autograd.gradcheck(func_pb_disp, (grid_grad_u, pts_grad, disp_grad), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 2. Pullback gradcheck: unbatched coords (N, d) + batched grid (B, C, H, W) + disp=None
    grid_grad_b = torch.randn(2, 2, 8, 8, dtype=torch.float64, requires_grad=True)

    def func_pb_grid(g, p):
        return pullback_grid_to_scattered(g, p, None)

    assert torch.autograd.gradcheck(func_pb_grid, (grid_grad_b, pts_grad), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 3. Pushforward gradcheck: unbatched coords (N, d) + batched features (B, N, C)
    feats_grad_b = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)

    def func_pf(p, f):
        return pushforward_scattered_to_grid(p, f, grid_shape=(6, 6), sigma=0.3, epsilon=1e-5)

    assert torch.autograd.gradcheck(func_pf, (pts_grad, feats_grad_b), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 4. Direct transport gradcheck: unbatched coords_src, coords_tgt, feats_src + batched displacement
    pts_t_grad = torch.tensor([[-0.1, 0.2]], dtype=torch.float64, requires_grad=True)
    feats_s_grad = torch.tensor([[1.0, 0.5], [0.2, 1.2], [0.8, 0.3]], dtype=torch.float64, requires_grad=True)

    def func_tr_dir(ps, fs, pt, d):
        return transport_scattered_to_scattered(ps, fs, pt, d, sigma=0.4, method='direct')

    assert torch.autograd.gradcheck(func_tr_dir, (pts_grad, feats_s_grad, pts_t_grad, disp_grad), eps=1e-6, atol=1e-4, rtol=1e-3)

    # 5. Grid bridge transport gradcheck: unbatched coords_src, coords_tgt, feats_src + batched displacement
    def func_tr_br(ps, fs, pt, d):
        return transport_scattered_to_scattered(ps, fs, pt, d, sigma=0.4, method='grid_bridge', grid_shape=(8, 8), domain_bounds=(-1.0, 1.0))

    assert torch.autograd.gradcheck(func_tr_br, (pts_grad, feats_s_grad, pts_t_grad, disp_grad), eps=1e-6, atol=1e-4, rtol=1e-3)
