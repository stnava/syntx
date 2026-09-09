"""
tests.test_scattered_transport_stress — Empirical Stress Test Suite for Milestone M2
===================================================================================

Adversarial and empirical stress tests for feature transport:
- 2D/3D cycle consistency (r > 0.98) under identity and non-rigid diffeomorphisms.
- Multi-channel feature cycle consistency (r > 0.98 per channel).
- Direct Lagrangian vs Eulerian grid bridge mutual fidelity (r > 0.99).
- Memory safety and auto-chunking with large point counts (N=10,000 to 25,000).
- Numerical invariance across chunk sizes.
- Physical domain bounds and coordinate conventions ('xyz' vs 'zyx').
- End-to-end autograd gradient flow through displacement fields and coordinates.
- Adversarial edge cases: N=1, extreme out-of-bounds coordinates, extreme bandwidths.
"""

import numpy as np
import pytest
import torch

from syntx.core.smoothing import separable_gaussian_filter
from syntx.scattered.transport import (
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
)


def compute_pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient between two tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = torch.norm(a_c) * torch.norm(b_c)
    if denom < 1e-8:
        return 0.0
    return float(torch.sum(a_c * b_c) / denom)


def test_stress_cycle_consistency_2d_identity_and_nonrigid():
    """Stress Test 1: 2D cycle consistency r > 0.98 across identity and non-rigid warps."""
    torch.manual_seed(42)
    N = 1500
    coords = torch.rand(N, 2) * 1.4 - 0.7  # in [-0.7, 0.7]^2
    feats = (torch.sin(2 * np.pi * coords[:, 0]) + torch.cos(2 * np.pi * coords[:, 1])).unsqueeze(-1)

    # 1. Identity Warp
    grid_id = pushforward_scattered_to_grid(coords, feats, grid_shape=(64, 64), sigma=0.04)
    rec_id = pullback_grid_to_scattered(grid_id, coords)
    r_id = compute_pearson_r(feats, rec_id)
    assert r_id > 0.98, f"2D Identity cycle consistency failed: r = {r_id:.5f} <= 0.98"

    # 2. Sinusoidal Non-rigid Diffeomorphism
    H, W = 64, 64
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    ux = 0.05 * torch.sin(np.pi * gx) * torch.cos(np.pi * gy)
    uy = 0.05 * torch.cos(np.pi * gx) * torch.sin(np.pi * gy)
    disp_sin = torch.stack([ux, uy], dim=-1).unsqueeze(0)

    grid_sin = pushforward_scattered_to_grid(coords, feats, disp_sin, grid_shape=(64, 64), sigma=0.04)
    rec_sin = pullback_grid_to_scattered(grid_sin, coords, disp_sin)
    r_sin = compute_pearson_r(feats, rec_sin)
    assert r_sin > 0.98, f"2D Sinusoidal warp cycle consistency failed: r = {r_sin:.5f} <= 0.98"

    # 3. Random Gaussian-Smoothed Non-rigid Diffeomorphism
    noise = torch.randn(1, 2, H, W)
    disp_rand = separable_gaussian_filter(noise, sigma=3.0).permute(0, 2, 3, 1)
    disp_rand = 0.08 * disp_rand / disp_rand.abs().max()

    grid_rand = pushforward_scattered_to_grid(coords, feats, disp_rand, grid_shape=(64, 64), sigma=0.04)
    rec_rand = pullback_grid_to_scattered(grid_rand, coords, disp_rand)
    r_rand = compute_pearson_r(feats, rec_rand)
    assert r_rand > 0.98, f"2D Random smooth warp cycle consistency failed: r = {r_rand:.5f} <= 0.98"


def test_stress_cycle_consistency_3d_identity_and_nonrigid():
    """Stress Test 2: 3D cycle consistency r > 0.98 across identity and non-rigid warps."""
    torch.manual_seed(42)
    N = 2500
    coords_3d = torch.rand(N, 3) * 1.4 - 0.7  # in [-0.7, 0.7]^3
    feats_3d = (torch.sin(np.pi * coords_3d[:, 0]) * torch.cos(np.pi * coords_3d[:, 1]) * torch.sin(np.pi * coords_3d[:, 2])).unsqueeze(-1)

    # 1. 3D Identity Warp
    grid_3d_id = pushforward_scattered_to_grid(coords_3d, feats_3d, grid_shape=(48, 48, 48), sigma=0.05)
    rec_3d_id = pullback_grid_to_scattered(grid_3d_id, coords_3d)
    r_3d_id = compute_pearson_r(feats_3d, rec_3d_id)
    assert r_3d_id > 0.98, f"3D Identity cycle consistency failed: r = {r_3d_id:.5f} <= 0.98"

    # 2. 3D Sinusoidal Non-rigid Diffeomorphism
    D = H = W = 48
    z = torch.linspace(-1, 1, D)
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    ux = 0.04 * torch.sin(np.pi * gx) * torch.cos(np.pi * gy)
    uy = 0.04 * torch.sin(np.pi * gy) * torch.cos(np.pi * gz)
    uz = 0.04 * torch.sin(np.pi * gz) * torch.cos(np.pi * gx)
    disp_3d_sin = torch.stack([ux, uy, uz], dim=-1).unsqueeze(0)

    grid_3d_sin = pushforward_scattered_to_grid(coords_3d, feats_3d, disp_3d_sin, grid_shape=(D, H, W), sigma=0.05)
    rec_3d_sin = pullback_grid_to_scattered(grid_3d_sin, coords_3d, disp_3d_sin)
    r_3d_sin = compute_pearson_r(feats_3d, rec_3d_sin)
    assert r_3d_sin > 0.98, f"3D Sinusoidal warp cycle consistency failed: r = {r_3d_sin:.5f} <= 0.98"

    # 3. 3D Random Gaussian-Smoothed Non-rigid Diffeomorphism
    noise_3d = torch.randn(1, 3, 32, 32, 32)
    disp_3d_rand = separable_gaussian_filter(noise_3d, sigma=2.0).permute(0, 2, 3, 4, 1)
    disp_3d_rand = 0.06 * disp_3d_rand / disp_3d_rand.abs().max()

    grid_3d_rand = pushforward_scattered_to_grid(coords_3d, feats_3d, disp_3d_rand, grid_shape=(32, 32, 32), sigma=0.04)
    rec_3d_rand = pullback_grid_to_scattered(grid_3d_rand, coords_3d, disp_3d_rand)
    r_3d_rand = compute_pearson_r(feats_3d, rec_3d_rand)
    assert r_3d_rand > 0.98, f"3D Random smooth warp cycle consistency failed: r = {r_3d_rand:.5f} <= 0.98"


def test_stress_cycle_consistency_multichannel_2d_and_3d():
    """Stress Test 3: Multi-channel (C=4) feature cycle consistency r > 0.98 per channel."""
    torch.manual_seed(42)
    N = 1500
    coords = torch.rand(N, 2) * 1.4 - 0.7
    f1 = torch.sin(2 * np.pi * coords[:, 0])
    f2 = torch.cos(2 * np.pi * coords[:, 1])
    f3 = coords[:, 0]**2 - coords[:, 1]**2
    f4 = torch.exp(-(coords[:, 0]**2 + coords[:, 1]**2) / 0.2)
    feats = torch.stack([f1, f2, f3, f4], dim=-1)

    noise = torch.randn(1, 2, 64, 64)
    disp = separable_gaussian_filter(noise, sigma=3.0).permute(0, 2, 3, 1)
    disp = 0.06 * disp / disp.abs().max()

    grid = pushforward_scattered_to_grid(coords, feats, disp, grid_shape=(64, 64), sigma=0.04)
    rec = pullback_grid_to_scattered(grid, coords, disp)

    assert grid.shape == (1, 4, 64, 64)
    assert rec.shape in [(N, 4), (1, N, 4)]
    rec_eval = rec.squeeze(0) if rec.dim() == 3 else rec

    for c in range(4):
        r_c = compute_pearson_r(feats[:, c], rec_eval[:, c])
        assert r_c > 0.98, f"Channel {c} cycle consistency failed: r = {r_c:.5f} <= 0.98"


def test_stress_transport_scattered_direct_vs_bridge_fidelity():
    """Stress Test 4: Mutual fidelity between Direct and Grid Bridge transport r > 0.99."""
    torch.manual_seed(42)
    # 2D case with non-equal point counts
    N_src, N_tgt = 800, 1200
    src_pts = torch.rand(N_src, 2) * 1.4 - 0.7
    tgt_pts = torch.rand(N_tgt, 2) * 1.4 - 0.7
    f_src = torch.sin(2 * np.pi * src_pts[:, 0:1]) * torch.cos(2 * np.pi * src_pts[:, 1:2])

    noise = torch.randn(1, 2, 64, 64)
    disp = separable_gaussian_filter(noise, sigma=3.0).permute(0, 2, 3, 1)
    disp = 0.05 * disp / disp.abs().max()

    f_dir = transport_scattered_to_scattered(src_pts, f_src, tgt_pts, disp, sigma=0.05, method='direct')
    f_brg = transport_scattered_to_scattered(src_pts, f_src, tgt_pts, disp, sigma=0.05, method='grid_bridge', grid_shape=(64, 64))

    r_2d = compute_pearson_r(f_dir, f_brg)
    assert r_2d > 0.99, f"2D Direct vs Bridge transport failed: r = {r_2d:.5f} <= 0.99"

    # 3D case
    N_src_3d, N_tgt_3d = 1000, 1200
    src_3d = torch.rand(N_src_3d, 3) * 1.4 - 0.7
    tgt_3d = torch.rand(N_tgt_3d, 3) * 1.4 - 0.7
    f_src_3d = (torch.sin(np.pi * src_3d[:, 0]) * torch.cos(np.pi * src_3d[:, 1])).unsqueeze(-1)

    noise_3d = torch.randn(1, 3, 32, 32, 32)
    disp_3d = separable_gaussian_filter(noise_3d, sigma=2.0).permute(0, 2, 3, 4, 1)
    disp_3d = 0.05 * disp_3d / disp_3d.abs().max()

    f_dir_3d = transport_scattered_to_scattered(src_3d, f_src_3d, tgt_3d, disp_3d, sigma=0.06, method='direct')
    f_brg_3d = transport_scattered_to_scattered(src_3d, f_src_3d, tgt_3d, disp_3d, sigma=0.06, method='grid_bridge', grid_shape=(48, 48, 48))

    r_3d = compute_pearson_r(f_dir_3d, f_brg_3d)
    assert r_3d > 0.99, f"3D Direct vs Bridge transport failed: r = {r_3d:.5f} <= 0.99"


def test_stress_memory_safety_large_n_auto_chunking():
    """Stress Test 5: Verify memory safety with large point counts (N=10,000+)."""
    torch.manual_seed(42)

    # 1. 2D N=10,000 Pushforward & Pullback
    c_10k = torch.rand(10000, 2) * 1.4 - 0.7
    f_10k = torch.randn(10000, 3)
    g_10k = pushforward_scattered_to_grid(c_10k, f_10k, grid_shape=(128, 128), sigma=0.03, target_memory_mb=64.0)
    assert g_10k.shape == (1, 3, 128, 128)
    assert torch.isfinite(g_10k).all()

    p_10k = pullback_grid_to_scattered(g_10k, c_10k)
    assert p_10k.shape[-2:] == (10000, 3)
    assert torch.isfinite(p_10k).all()

    # 2. 2D N=25,000 Pushforward
    c_25k = torch.rand(25000, 2) * 1.4 - 0.7
    f_25k = torch.randn(25000, 2)
    g_25k = pushforward_scattered_to_grid(c_25k, f_25k, grid_shape=(128, 128), sigma=0.02, target_memory_mb=64.0)
    assert g_25k.shape == (1, 2, 128, 128)
    assert torch.isfinite(g_25k).all()

    # 3. 3D N=10,000 on 64x64x64 grid (262,144 voxels -> 10.5 GB unchunked GEMM)
    c_3d_10k = torch.rand(10000, 3) * 1.4 - 0.7
    f_3d_10k = torch.randn(10000, 1)
    g_3d_10k = pushforward_scattered_to_grid(c_3d_10k, f_3d_10k, grid_shape=(64, 64, 64), sigma=0.04, target_memory_mb=64.0)
    assert g_3d_10k.shape == (1, 1, 64, 64, 64)
    assert torch.isfinite(g_3d_10k).all()

    # 4. Direct Point-to-Point Transport N_src=10,000 -> N_tgt=10,000
    c_tgt_10k = torch.rand(10000, 3) * 1.4 - 0.7
    tr_10k = transport_scattered_to_scattered(
        c_3d_10k, f_3d_10k, c_tgt_10k, sigma=0.04, method='direct', target_memory_mb=32.0
    )
    assert tr_10k.shape == (10000, 1)
    assert torch.isfinite(tr_10k).all()


def test_stress_chunking_numerical_invariance():
    """Stress Test 6: Verify exact numerical invariance across different chunk sizes."""
    torch.manual_seed(42)
    src_pts = torch.rand(1000, 2) * 1.4 - 0.7
    src_feats = torch.randn(1000, 2)
    tgt_pts = torch.rand(800, 2) * 1.4 - 0.7

    out_chunk_25 = transport_scattered_to_scattered(
        src_pts, src_feats, tgt_pts, sigma=0.05, method='direct', chunk_size=25
    )
    out_chunk_200 = transport_scattered_to_scattered(
        src_pts, src_feats, tgt_pts, sigma=0.05, method='direct', chunk_size=200
    )
    out_unchunked = transport_scattered_to_scattered(
        src_pts, src_feats, tgt_pts, sigma=0.05, method='direct', chunk_size=0
    )

    diff_25_200 = (out_chunk_25 - out_chunk_200).abs().max().item()
    diff_25_un = (out_chunk_25 - out_unchunked).abs().max().item()

    assert diff_25_200 < 1e-6, f"Chunking invariance failure (25 vs 200): {diff_25_200}"
    assert diff_25_un < 1e-6, f"Chunking invariance failure (25 vs unchunked): {diff_25_un}"


def test_stress_physical_domain_bounds_and_conventions():
    """Stress Test 7: Verify transport under physical millimeter domain bounds and coordinate conventions."""
    torch.manual_seed(42)
    # Physical brain bounding box: X in [-50, 50], Y in [0, 100]
    bounds_2d = ([-50.0, 0.0], [50.0, 100.0])
    N = 600
    x_pts = torch.rand(N, 1) * 80.0 - 40.0
    y_pts = torch.rand(N, 1) * 80.0 + 10.0
    coords_phys = torch.cat([x_pts, y_pts], dim=-1)
    feats = (torch.sin(coords_phys[:, 0:1] / 15.0) + torch.cos(coords_phys[:, 1:2] / 15.0))

    grid_phys = pushforward_scattered_to_grid(
        coords_phys, feats, grid_shape=(64, 64), domain_bounds=bounds_2d, sigma=3.0
    )
    rec_phys = pullback_grid_to_scattered(
        grid_phys, coords_phys, domain_bounds=bounds_2d
    )
    r_phys = compute_pearson_r(feats, rec_phys)
    assert r_phys > 0.98, f"Physical bounds cycle consistency failed: r = {r_phys:.5f} <= 0.98"

    # Coordinate convention fidelity: 'xyz' vs 'zyx'
    H, W = 20, 30
    y_lin = torch.linspace(-1, 1, H)
    x_lin = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(y_lin, x_lin, indexing='ij')
    test_grid = (5.0 * gx + 2.0 * gy).unsqueeze(0).unsqueeze(0)

    # In 'xyz': query is (x, y) = (0.4, -0.2) -> 5*(0.4) + 2*(-0.2) = 1.6
    q_xyz = torch.tensor([[0.4, -0.2]])
    v_xyz = pullback_grid_to_scattered(test_grid, q_xyz, coord_convention='xyz')
    assert torch.isclose(v_xyz[0, 0], torch.tensor(1.6), atol=1e-4)

    # In 'zyx': query is (y, x) = (-0.2, 0.4) -> 1.6
    q_zyx = torch.tensor([[-0.2, 0.4]])
    v_zyx = pullback_grid_to_scattered(test_grid, q_zyx, coord_convention='zyx')
    assert torch.isclose(v_zyx[0, 0], torch.tensor(1.6), atol=1e-4)


def test_stress_autograd_gradient_propagation_end_to_end():
    """Stress Test 8: Verify backward pass gradients for displacement field, points, and features."""
    coords = torch.tensor([[0.25, -0.15], [-0.35, 0.45]], dtype=torch.float64, requires_grad=True)
    feats = torch.tensor([[1.8], [2.6]], dtype=torch.float64, requires_grad=True)
    disp = (torch.randn(1, 8, 8, 2, dtype=torch.float64) * 0.05).requires_grad_(True)
    tgt = torch.tensor([[0.1, 0.1], [-0.1, -0.1]], dtype=torch.float64, requires_grad=True)

    # 1. Pullback autograd
    dummy_grid = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    out_pull = pullback_grid_to_scattered(dummy_grid, coords, disp)
    out_pull.sum().backward()
    assert disp.grad is not None and disp.grad.norm() > 0
    assert coords.grad is not None and coords.grad.norm() > 0

    # 2. Pushforward autograd
    disp_pf = (torch.randn(1, 8, 8, 2, dtype=torch.float64) * 0.05).requires_grad_(True)
    c_pf = torch.tensor([[0.25, -0.15], [-0.35, 0.45]], dtype=torch.float64, requires_grad=True)
    f_pf = torch.tensor([[1.8], [2.6]], dtype=torch.float64, requires_grad=True)
    out_pf = pushforward_scattered_to_grid(c_pf, f_pf, disp_pf, grid_shape=(8, 8), sigma=0.1)
    out_pf.sum().backward()
    assert disp_pf.grad is not None and disp_pf.grad.norm() > 0
    assert c_pf.grad is not None and c_pf.grad.norm() > 0
    assert f_pf.grad is not None and f_pf.grad.norm() > 0

    # 3. Direct Transport autograd
    disp_tr = (torch.randn(1, 8, 8, 2, dtype=torch.float64) * 0.05).requires_grad_(True)
    c_tr = torch.tensor([[0.25, -0.15], [-0.35, 0.45]], dtype=torch.float64, requires_grad=True)
    f_tr = torch.tensor([[1.8], [2.6]], dtype=torch.float64, requires_grad=True)
    out_tr = transport_scattered_to_scattered(c_tr, f_tr, tgt, disp_tr, sigma=0.1, method='direct')
    out_tr.sum().backward()
    assert disp_tr.grad is not None and disp_tr.grad.norm() > 0
    assert c_tr.grad is not None and c_tr.grad.norm() > 0
    assert f_tr.grad is not None and f_tr.grad.norm() > 0
    assert tgt.grad is not None and tgt.grad.norm() > 0


def test_stress_edge_cases_and_adversarial_inputs():
    """Stress Test 9: Test N=1, extreme coordinates, extreme sigmas, and error handling."""
    # N = 1
    c1 = torch.tensor([[0.3, -0.4]])
    f1 = torch.tensor([[5.0]])
    g1 = pushforward_scattered_to_grid(c1, f1, grid_shape=(16, 16), sigma=0.1)
    p1 = pullback_grid_to_scattered(g1, c1)
    assert torch.isclose(p1[0, 0], torch.tensor(5.0), atol=1e-4)

    # N = 0
    c0 = torch.zeros(0, 2)
    f0 = torch.zeros(0, 1)
    g0 = pushforward_scattered_to_grid(c0, f0, grid_shape=(16, 16), sigma=0.1)
    p0 = pullback_grid_to_scattered(g1, c0)
    assert g0.shape == (1, 1, 16, 16)
    assert p0.shape[-2:] == (0, 1)

    # Out of bounds coordinates: points far outside [-1, 1]
    c_oob = torch.tensor([[500.0, -1000.0]])
    f_oob = torch.tensor([[2.0]])
    g_oob = pushforward_scattered_to_grid(c_oob, f_oob, grid_shape=(16, 16), sigma=0.1)
    assert torch.isfinite(g_oob).all()
    assert (g_oob == 0.0).all()

    p_oob_border = pullback_grid_to_scattered(g1, c_oob, padding_mode='border')
    assert torch.isfinite(p_oob_border).all()
    p_oob_zeros = pullback_grid_to_scattered(g1, c_oob, padding_mode='zeros')
    assert torch.isfinite(p_oob_zeros).all()
    assert (p_oob_zeros == 0.0).all()

    # Extreme sigmas
    g_tiny_sig = pushforward_scattered_to_grid(c1, f1, grid_shape=(16, 16), sigma=1e-5)
    assert torch.isfinite(g_tiny_sig).all()
    g_huge_sig = pushforward_scattered_to_grid(c1, f1, grid_shape=(16, 16), sigma=100.0)
    assert torch.isfinite(g_huge_sig).all()

    # Invalid negative sigma must raise ValueError
    with pytest.raises(ValueError):
        pushforward_scattered_to_grid(c1, f1, grid_shape=(16, 16), sigma=-0.05)
