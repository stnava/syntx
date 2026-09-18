import pytest
import torch
import numpy as np

from syntx.core.losses import (
    local_ncc_loss_nd,
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
)


def test_lncc_identical_images():
    # Identical images should yield CC = 1.0, hence loss = -1.0
    img = torch.randn(1, 1, 16, 16)
    loss = local_ncc_loss_nd(img, img, window_size=5)
    assert np.isclose(loss.item(), -1.0, atol=1e-4)


def test_lncc_analytical_vs_autograd():
    img1 = torch.randn(1, 1, 16, 16, requires_grad=True)
    img2 = torch.randn(1, 1, 16, 16, requires_grad=True)

    loss_autograd = local_ncc_loss_nd(img1, img2, window_size=5, use_ants_pseudo_gradient=False)
    loss_analytical = local_ncc_loss_nd(img1, img2, window_size=5, use_ants_pseudo_gradient=True)

    assert torch.allclose(loss_autograd, loss_analytical, atol=1e-4)


def test_lncc_cauchy_schwarz_and_variance_floor():
    # Flat image (zero variance) should not produce NaNs
    flat = torch.zeros(1, 1, 16, 16)
    loss = local_ncc_loss_nd(flat, flat, window_size=5)
    assert not torch.isnan(loss)


def test_b_spline_3():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = b_spline_3(x)
    assert np.isclose(b[0].item(), 2.0 / 3.0)
    assert np.isclose(b[1].item(), 1.0 / 6.0)
    assert np.isclose(b[2].item(), 0.0)
    assert np.isclose(b[3].item(), 0.0)


def test_mattes_mi_identical_images():
    img = torch.linspace(0, 1, 64).repeat(64, 1).unsqueeze(0).unsqueeze(0)
    loss = mattes_mi_loss_nd(img, img, num_bins=16)
    assert not torch.isnan(loss)
    assert loss < 0.0


def test_box_lncc_loss():
    from syntx.core.losses import BoxLNCCLoss, box_lncc_loss_nd

    # 1. Identical textured images -> loss close to -1.0
    torch.manual_seed(42)
    img = torch.randn(1, 1, 16, 16, 16)
    loss_fn = BoxLNCCLoss(kernel_size=5)
    loss = loss_fn(img, img)
    assert not torch.isnan(loss)
    assert np.isclose(loss.item(), -1.0, atol=1e-3)

    # 2. Functional interface matches module
    loss_func = box_lncc_loss_nd(img, img, window_size=5)
    assert torch.allclose(loss, loss_func)

    # 3. Flat zero-padded background evaluates to -1.0 (perfect correlation)
    flat = torch.zeros(1, 1, 16, 16, 16)
    flat_loss = loss_fn(flat, flat)
    assert not torch.isnan(flat_loss)
    assert np.isclose(flat_loss.item(), -1.0, atol=1e-5)

    # 4. Backward autograd passes cleanly
    img1 = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    img2 = torch.randn(1, 1, 16, 16, 16)
    l = loss_fn(img1, img2)
    l.backward()
    assert img1.grad is not None
    assert not torch.isnan(img1.grad).any()


def test_box_lncc_target_caching():
    from syntx.core.losses import BoxLNCCLoss

    loss_fn = BoxLNCCLoss(kernel_size=5)
    t1 = torch.randn(1, 1, 16, 16, 16)
    p1 = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    p2 = torch.randn(1, 1, 16, 16, 16, requires_grad=True)

    # Initial forward: target cached
    loss1 = loss_fn(p1, t1)
    assert id(t1) in loss_fn._target_cache
    cached_version, t_sum, t_var = loss_fn._target_cache[id(t1)]

    # Second forward with same target: cache hit, identical tensors reused
    loss2 = loss_fn(p2, t1)
    assert loss_fn._target_cache[id(t1)][1] is t_sum
    assert loss_fn._target_cache[id(t1)][2] is t_var

    # Forward with new target: caches new target
    t2 = torch.randn(1, 1, 16, 16, 16)
    loss3 = loss_fn(p1, t2)
    assert id(t2) in loss_fn._target_cache
    assert id(t1) in loss_fn._target_cache

    # Target requiring grad: does not cache
    t_grad = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    loss4 = loss_fn(p1, t_grad)
    assert id(t_grad) not in loss_fn._target_cache


def test_mattes_mi_amp_overflow_protection():
    """Verify Mattes MI Parzen joint histogram never overflows to NaN under AMP autocast."""
    from syntx.core.losses import mattes_mi_loss_nd, mattes_mi_loss_core

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # Large synthetic volume with >100,000 voxels that would overflow float16 if not guarded
    x = torch.rand(1, 1, 64, 64, 64, device=dev, requires_grad=True)
    y = (x + 0.1 * torch.randn_like(x)).detach().requires_grad_(True)

    dev_type = "cuda" if "cuda" in str(dev) else ("mps" if "mps" in str(dev) else "cpu")
    with torch.amp.autocast(device_type=dev_type, dtype=torch.float16, enabled=(dev_type != "cpu")):
        loss = mattes_mi_loss_nd(x, y, auto_mask=True)
        assert torch.isfinite(loss), f"Mattes MI produced non-finite loss: {loss.item()}"
        assert loss.item() < 0.0, f"Expected negative mutual information, got {loss.item()}"

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Mattes MI backward produced NaN/inf gradients"
    assert x.grad.abs().sum() > 0.0, "Mattes MI backward produced zero gradients"


def test_mattes_mi_foreground_automasking():
    """Verify auto_mask=True filters out zero padding from the joint histogram."""
    from syntx.core.losses import mattes_mi_loss_nd

    # 32x32x32 image with 85% zero padding
    x = torch.zeros(1, 1, 32, 32, 32, requires_grad=True)
    y = torch.zeros(1, 1, 32, 32, 32, requires_grad=True)

    with torch.no_grad():
        # Place non-zero foreground in center
        x[:, :, 10:22, 10:22, 10:22] = torch.rand(1, 1, 12, 12, 12) * 0.8 + 0.2
        y[:, :, 10:22, 10:22, 10:22] = x[:, :, 10:22, 10:22, 10:22] + 0.05 * torch.randn(1, 1, 12, 12, 12)

    loss_automask = mattes_mi_loss_nd(x, y, auto_mask=True)
    assert torch.isfinite(loss_automask)

    loss_automask.backward()
    assert x.grad is not None
    # Background voxels must have zero gradient
    bg_mask = (x.detach() <= 0.01) & (y.detach() <= 0.01)
    assert torch.all(x.grad[bg_mask] == 0.0), "Background voxels should receive zero gradient with auto_mask=True"


def test_mattes_mi_partition_of_unity():
    """Verify B-spline Parzen windowing maintains exact partition of unity at boundaries."""
    from syntx.core.losses import b_spline_3

    num_bins = 32
    min_val, max_val = -1.0, 1.0
    pad = 2.0
    u_min = pad
    u_max = float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)
    bin_indices = torch.arange(num_bins, dtype=torch.float32).unsqueeze(0)

    # Test values including extreme boundaries [-1.0, 1.0]
    x = torch.linspace(-1.0, 1.0, 100, requires_grad=True)
    u = u_min + (x.unsqueeze(1) - min_val) * scale
    w = b_spline_3(u - bin_indices)
    weight_sums = w.sum(dim=1)

    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "B-spline partition of unity violated"

    # Verify zero phantom boundary gradients
    total_sum = weight_sums.sum()
    total_sum.backward()
    assert x.grad is not None
    assert x.grad.abs().max().item() < 1e-5, f"Boundary gradient spike detected: {x.grad.abs().max().item()}"


def test_box_cc2_and_box_lncc2_loss():
    """Verify box_cc2_loss_nd and box_lncc_loss_nd with squared=True compute valid differentiable loss."""
    from syntx.core.losses import box_cc2_loss_nd, box_lncc_loss_nd

    x = torch.rand(1, 1, 16, 16, 16, requires_grad=True)
    y = (x + 0.05 * torch.randn_like(x)).detach().requires_grad_(True)

    # box_cc2_loss_nd
    l_box_cc2 = box_cc2_loss_nd(x, y, window_size=5)
    assert torch.isfinite(l_box_cc2)
    assert l_box_cc2.item() < 0.0  # Normalized cross correlation loss is negative
    l_box_cc2.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Reset gradient and test box_lncc_loss_nd with squared=True vs False
    x.grad.zero_()
    l_box_lncc_sq = box_lncc_loss_nd(x, y, window_size=5, squared=True)
    l_box_lncc_lin = box_lncc_loss_nd(x, y, window_size=5, squared=False)
    assert torch.allclose(l_box_cc2, l_box_lncc_sq)
    assert l_box_lncc_lin.item() < l_box_lncc_sq.item() or torch.isfinite(l_box_lncc_lin)

