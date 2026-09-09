"""
tests.test_scattered_batch_challenge — Adversarial Challenge Suite for Milestone M2 Remediation
=============================================================================================

Empirically challenges batch broadcasting and numerical slice invariance across:
- Batch sizes B in {1, 3, 7}
- Spatial dimensions: 2D and 3D (including non-square and non-cubic grids)
- pullback_grid_to_scattered
- pushforward_scattered_to_grid
- transport_scattered_to_scattered
- Autograd gradient check and slice gradient invariance

Exposes 5 empirical bugs/invariances:
1. C == B heuristic collision in _standardize_grid_features (pullback corrupts multi-channel grid into scalar batch).
2. Pushforward crash with ValueError on B=1 batched scalar features (1, N) with unbatched coordinates.
3. Pushforward misinterpretation of B == N batched scalar features as unbatched multi-channel.
4. Batch dimension dropped when B=1 with unbatched coordinates despite batched grid/field.
5. Inconsistent unchanneled scalar squeezing between B=1 and B > 1.
"""

import pytest
import torch

from syntx.scattered.transport import (
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
)


# ===========================================================================
# 1. Robust Multi-Channel Pullback Batch Broadcasting & Slice Invariance
# ===========================================================================

@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dim,grid_shape", [(2, (18, 24)), (3, (10, 14, 12))])
def test_challenge_pullback_batched_coords_and_batched_disp(B, dim, grid_shape):
    """Verify pullback when coordinates are explicitly batched (B, N, d) and disp is (B, *spatial, d).

    Slice invariance: evaluating batch slice b must equal evaluating slice b individually.
    """
    torch.manual_seed(100 + B * 10 + dim)
    N = 35
    C = 4
    coords = torch.rand(B, N, dim) * 1.4 - 0.7
    disp = torch.randn(B, *grid_shape, dim) * 0.02
    grid = torch.randn(C, *grid_shape)  # unbatched multi-channel

    out_batched = pullback_grid_to_scattered(grid, coords, disp)
    assert out_batched.shape == (B, N, C)

    for b in range(B):
        out_single = pullback_grid_to_scattered(grid, coords[b:b+1], disp[b:b+1])
        assert out_single.shape == (1, N, C)
        assert torch.allclose(out_batched[b], out_single[0], atol=1e-6)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dim,grid_shape", [(2, (20, 16)), (3, (8, 12, 10))])
def test_challenge_pullback_batched_grid_and_batched_coords(B, dim, grid_shape):
    """Verify pullback when coordinates (B, N, d) and grid (B, C, *spatial) are both batched."""
    torch.manual_seed(200 + B * 10 + dim)
    N = 28
    C = 4
    coords = torch.rand(B, N, dim) * 1.4 - 0.7
    grid = torch.randn(B, C, *grid_shape)

    out_batched = pullback_grid_to_scattered(grid, coords, displacement_field=None)
    assert out_batched.shape == (B, N, C)

    for b in range(B):
        out_single = pullback_grid_to_scattered(grid[b:b+1], coords[b:b+1], displacement_field=None)
        assert out_single.shape == (1, N, C)
        assert torch.allclose(out_batched[b], out_single[0], atol=1e-6)


@pytest.mark.parametrize("B", [3, 7])
def test_challenge_pullback_channels_last_and_non_contiguous(B):
    """Challenge pullback: channels-last input tensor (B, H, W, C) and non-contiguous views."""
    torch.manual_seed(500 + B)
    H, W = 16, 18
    C = 4
    N = 20
    coords = torch.rand(B, N, 2) * 1.2 - 0.6
    disp = torch.randn(B, H, W, 2) * 0.02

    # Channels-last tensor
    grid_cl = torch.randn(B, H, W, C)
    out_cl = pullback_grid_to_scattered(grid_cl, coords, disp, channel_dim=-1)
    assert out_cl.shape == (B, N, C)

    # Channels-first equivalent
    grid_cf = torch.movedim(grid_cl, -1, 1)
    out_cf = pullback_grid_to_scattered(grid_cf, coords, disp, channel_dim=1)
    assert torch.allclose(out_cl, out_cf, atol=1e-6)

    # Non-contiguous slicing
    grid_strided = torch.randn(B, C, H * 2, W)[..., ::2, :]
    assert not grid_strided.is_contiguous()
    disp_strided = torch.randn(B, H * 2, W, 2)[..., ::2, :, :]
    assert not disp_strided.is_contiguous()

    out_strided = pullback_grid_to_scattered(grid_strided, coords, disp_strided)
    assert out_strided.shape == (B, N, C)


# ===========================================================================
# 2. Pushforward Batch Broadcasting & Slice Invariance (3D multi-channel)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dim,grid_shape", [(2, (18, 22)), (3, (8, 12, 10))])
def test_challenge_pushforward_batched_features_unbatched_coords_slice_invariance(B, dim, grid_shape):
    """Challenge pushforward: unbatched coords (N, d) + batched features (B, N, C) + identity disp.

    Verifies slice invariance for both features and density grids.
    """
    torch.manual_seed(600 + B * 10 + dim)
    N = 35
    C = 3
    coords = torch.rand(N, dim) * 1.4 - 0.7
    feats_b = torch.randn(B, N, C)

    grid_b, den_b = pushforward_scattered_to_grid(
        coords, feats_b, displacement_field=None, grid_shape=grid_shape, sigma=0.06, return_density=True
    )
    assert grid_b.shape == (B, C, *grid_shape)
    assert den_b.shape == (B, 1, *grid_shape)

    for b in range(B):
        # Single slice evaluation
        grid_single, den_single = pushforward_scattered_to_grid(
            coords, feats_b[b:b+1], displacement_field=None, grid_shape=grid_shape, sigma=0.06, return_density=True
        )
        assert torch.allclose(grid_b[b:b+1], grid_single, atol=1e-6)
        assert torch.allclose(den_b[b:b+1], den_single, atol=1e-6)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dim,grid_shape", [(2, (16, 20)), (3, (8, 10, 10))])
def test_challenge_pushforward_batched_features_and_batched_disp_slice_invariance(B, dim, grid_shape):
    """Challenge pushforward: unbatched coords (N, d) + batched features (B, N, C) + batched disp (B, *spatial, d).

    Verifies that simultaneous feature and displacement batching satisfies exact slice invariance.
    """
    torch.manual_seed(700 + B * 10 + dim)
    N = 40
    C = 2
    coords = torch.rand(N, dim) * 1.4 - 0.7
    feats_b = torch.randn(B, N, C)
    disp_b = torch.randn(B, *grid_shape, dim) * 0.02

    grid_b, den_b = pushforward_scattered_to_grid(
        coords, feats_b, displacement_field=disp_b, grid_shape=grid_shape, sigma=0.05, return_density=True
    )
    assert grid_b.shape == (B, C, *grid_shape)
    assert den_b.shape == (B, 1, *grid_shape)

    for b in range(B):
        grid_single, den_single = pushforward_scattered_to_grid(
            coords, feats_b[b:b+1], displacement_field=disp_b[b:b+1], grid_shape=grid_shape, sigma=0.05, return_density=True
        )
        assert torch.allclose(grid_b[b:b+1], grid_single, atol=1e-6)
        assert torch.allclose(den_b[b:b+1], den_single, atol=1e-6)


@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("dim,grid_shape", [(2, (16, 16)), (3, (8, 8, 8))])
def test_challenge_pushforward_unbatched_features_batched_disp_slice_invariance(B, dim, grid_shape):
    """Challenge pushforward: unbatched coords (N, d) + unbatched features (N, C) + batched disp (B, *spatial, d).

    Verifies broadcasting of a single feature set across multiple deformation fields.
    """
    torch.manual_seed(800 + B * 10 + dim)
    N = 30
    C = 3
    coords = torch.rand(N, dim) * 1.4 - 0.7
    feats_u = torch.randn(N, C)
    disp_b = torch.randn(B, *grid_shape, dim) * 0.02

    grid_b, den_b = pushforward_scattered_to_grid(
        coords, feats_u, displacement_field=disp_b, grid_shape=grid_shape, sigma=0.05, return_density=True
    )
    assert grid_b.shape == (B, C, *grid_shape)
    assert den_b.shape == (B, 1, *grid_shape)

    for b in range(B):
        grid_single, den_single = pushforward_scattered_to_grid(
            coords, feats_u, displacement_field=disp_b[b:b+1], grid_shape=grid_shape, sigma=0.05, return_density=True
        )
        assert torch.allclose(grid_b[b:b+1], grid_single, atol=1e-6)
        assert torch.allclose(den_b[b:b+1], den_single, atol=1e-6)


# ===========================================================================
# 3. Transport Point-to-Point (Batched coords & features)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 3, 7])
@pytest.mark.parametrize("method", ['direct', 'grid_bridge'])
def test_challenge_transport_batched_coords_and_disp_slice_invariance(B, method):
    """Verify point-to-point transport when coords are batched (B, N, d)."""
    torch.manual_seed(1000 + B)
    N_s, N_t, C = 30, 20, 3
    coords_s = torch.rand(B, N_s, 2) * 1.2 - 0.6
    feats_s = torch.randn(B, N_s, C)
    coords_t = torch.rand(B, N_t, 2) * 1.2 - 0.6
    disp = torch.randn(B, 16, 16, 2) * 0.02

    kwargs = {"grid_shape": (16, 16), "domain_bounds": (-1.0, 1.0)} if method == 'grid_bridge' else {}

    tr_b = transport_scattered_to_scattered(
        coords_s, feats_s, coords_t, displacement_field=disp, method=method, sigma=0.06, **kwargs
    )
    assert tr_b.shape == (B, N_t, C)

    for b in range(B):
        tr_single = transport_scattered_to_scattered(
            coords_s[b:b+1], feats_s[b:b+1], coords_t[b:b+1], displacement_field=disp[b:b+1],
            method=method, sigma=0.06, **kwargs
        )
        assert torch.allclose(tr_b[b:b+1], tr_single, atol=1e-6)


# ===========================================================================
# 4. Error Handling & Autograd Gradient Slice Invariance
# ===========================================================================

def test_challenge_batch_mismatch_exceptions():
    """Challenge error handling: verifying explicit ValueError is raised on batch size mismatches."""
    coords_b3 = torch.randn(3, 20, 2)
    disp_b5 = torch.randn(5, 16, 16, 2)
    grid_b4 = torch.randn(4, 2, 16, 16)
    feats_b2 = torch.randn(2, 20, 2)

    # Pullback: coords batch 3 vs disp batch 5
    with pytest.raises(ValueError):
        pullback_grid_to_scattered(grid_b4, coords_b3, disp_b5)

    # Pullback: grid batch 4 vs disp batch 5
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        pullback_grid_to_scattered(grid_b4, coords_b3[0], disp_b5)

    # Pushforward: coords batch 3 vs features batch 2
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        pushforward_scattered_to_grid(coords_b3, feats_b2, grid_shape=(16, 16))

    # Transport: source batch 3 vs target batch 5
    coords_tgt_b5 = torch.randn(5, 15, 2)
    with pytest.raises(ValueError, match="Batch dimension mismatch"):
        transport_scattered_to_scattered(coords_b3, torch.randn(3, 20, 1), coords_tgt_b5)


def test_challenge_autograd_batch_slice_gradient_invariance():
    """Challenge autograd: verify analytical gradient slice invariance across batches in float64."""
    B = 3
    pts = torch.tensor([
        [-0.2, 0.3],
        [0.4, -0.1],
        [-0.3, -0.4],
    ], dtype=torch.float64)

    feats_b = torch.randn(B, 3, 2, dtype=torch.float64, requires_grad=True)
    disp_b = (torch.randn(B, 8, 8, 2, dtype=torch.float64) * 0.02).requires_grad_(True)

    # Forward batch
    grid_b = pushforward_scattered_to_grid(pts, feats_b, disp_b, grid_shape=(8, 8), sigma=0.3)
    loss_b = grid_b.sum()
    loss_b.backward()

    grad_feats_batched = feats_b.grad.clone()
    grad_disp_batched = disp_b.grad.clone()

    # Compare with individual slice gradients
    for b in range(B):
        feat_single = feats_b.data[b:b+1].clone().requires_grad_(True)
        disp_single = disp_b.data[b:b+1].clone().requires_grad_(True)

        grid_s = pushforward_scattered_to_grid(pts, feat_single, disp_single, grid_shape=(8, 8), sigma=0.3)
        loss_s = grid_s.sum()
        loss_s.backward()

        assert torch.allclose(grad_feats_batched[b], feat_single.grad[0], atol=1e-8), (
            f"Gradient slice invariance failure in features for b={b}"
        )
        assert torch.allclose(grad_disp_batched[b], disp_single.grad[0], atol=1e-8), (
            f"Gradient slice invariance failure in displacement field for b={b}"
        )


# ===========================================================================
# 5. EMPIRICAL BUG REPRODUCTIONS (Remediated: now standard passing tests)
# ===========================================================================

def test_bug_reproduce_c_equals_b_heuristic_collision():
    """Bug 1: Pullback of unbatched 3-channel grid (C=3, H, W) through a batch of 3 displacement fields (B=3).

    Instead of evaluating all 3 channels for all 3 subjects -> (3, N, 3),
    the code misinterprets C=3 as B=3 scalar images, applying disp[0] only to channel 0,
    disp[1] only to channel 1, disp[2] only to channel 2, returning (3, N).
    """
    coords = torch.randn(10, 2)
    rgb_image = torch.ones(3, 16, 16)
    rgb_image[0] *= 10.0
    rgb_image[1] *= 20.0
    rgb_image[2] *= 30.0

    disp_b3 = torch.zeros(3, 16, 16, 2)
    out_b3 = pullback_grid_to_scattered(rgb_image, coords, disp_b3)
    # Expected: (3, 10, 3) where every subject has [10., 20., 30.]
    # Actual bug: shape is (3, 10) where subject 0 has 10., subject 1 has 20., subject 2 has 30.!
    assert out_b3.shape == (3, 10, 3), f"Bug: output shape is {out_b3.shape} instead of (3, 10, 3)"
    assert torch.allclose(out_b3[0, 0], torch.tensor([10.0, 20.0, 30.0]))


def test_bug_reproduce_pushforward_b1_batched_scalar_crash():
    """Bug 2: Pushforward crashes when features is (1, N) and coords is (N, d)."""
    coords = torch.randn(20, 2)
    features_b1 = torch.randn(1, 20)
    # Raises: ValueError: values shape torch.Size([1, 20]) does not match points count 20
    grid = pushforward_scattered_to_grid(coords, features_b1, grid_shape=(16, 16))
    assert grid.shape == (1, 1, 16, 16)


def test_bug_reproduce_pushforward_b_equals_n_channel_misinterpretation():
    """Bug 3: Pushforward with N=3 points and B=3 subjects with scalar features (3, 3)."""
    coords = torch.randn(3, 2)
    features_b3 = torch.randn(3, 3)  # 3 subjects, 3 points each
    grid = pushforward_scattered_to_grid(coords, features_b3, grid_shape=(16, 16))
    # Expected: (3, 1, 16, 16) [3 subjects, 1 channel]
    # Actual bug: (1, 3, 16, 16) [1 subject, 3 channels]
    assert grid.shape == (3, 1, 16, 16), f"Bug: output shape is {grid.shape} instead of (3, 1, 16, 16)"


def test_bug_reproduce_pullback_strips_batch_dim_when_b1():
    """Bug 4: Pullback strips batch dimension to (N, C) when displacement has shape (1, H, W, d)."""
    coords = torch.randn(25, 2)
    grid = torch.randn(4, 16, 16)
    disp1 = torch.randn(1, 16, 16, 2) * 0.02
    out = pullback_grid_to_scattered(grid, coords, disp1)
    # Expected: (1, 25, 4) since disp was explicitly batched as (1, 16, 16, 2)
    # Actual bug: returns (25, 4) because unbatched_coords and B==1
    assert out.shape == (1, 25, 4), f"Bug: output shape is {out.shape} instead of (1, 25, 4)"


def test_bug_reproduce_transport_strips_batch_dim_when_b1():
    """Bug 5: Point-to-point transport strips batch dimension to (N_tgt, C) when displacement has shape (1, H, W, d)."""
    coords_s = torch.randn(20, 2)
    coords_t = torch.randn(15, 2)
    feats_s = torch.randn(20, 3)
    disp1 = torch.randn(1, 16, 16, 2) * 0.02
    out = transport_scattered_to_scattered(coords_s, feats_s, coords_t, disp1)
    # Expected: (1, 15, 3) since disp was explicitly batched as (1, 16, 16, 2)
    # Actual bug: returns (15, 3) because unbatched_tgt and B_eff==1
    assert out.shape == (1, 15, 3), f"Bug: output shape is {out.shape} instead of (1, 15, 3)"
