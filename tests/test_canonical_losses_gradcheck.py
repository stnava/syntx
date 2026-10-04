"""Float64 torch.autograd.gradcheck compliance suite across all canonical loss functionals.

Specifications (per ORIGINAL_REQUEST.md & PROJECT.md):
- Float64 torch.autograd.gradcheck across all canonical loss functionals:
  - local_ncc_loss_nd (autograd path, 2D and 3D, odd kernel sizes 3, 5)
  - BoxLNCCLoss and box_lncc_loss_nd (squared=True and squared=False, 2D and 3D)
  - CanonicalMattesMIFunction and mattes_mi_loss_nd (fixed_range=(0.0, 1.0))
  - soft_dice_loss_nd (single and multi-channel, with/without mask)
  - mse_loss_nd, l2_loss_nd, mae_loss_nd (with/without mask)
  - compute_soft_distance_transform (2D and 3D)
- Use eps=1e-6, atol=1e-4, rtol=1e-3.
"""

import pytest
import torch
import torch.autograd

from syntx.core.losses import (
    BoxLNCCLoss,
    CanonicalMattesMIFunction,
    box_lncc_loss_nd,
    compute_soft_distance_transform,
    local_ncc_loss_nd,
    mattes_mi_loss_nd,
    soft_dice_loss_nd,
)

# Optional imports for pointwise family pending M1 package export
try:
    from syntx.core.losses import l2_loss_nd, mae_loss_nd, mse_loss_nd
    HAS_POINTWISE = True
except ImportError:
    HAS_POINTWISE = False


@pytest.mark.parametrize("window_size", [3, 5])
@pytest.mark.parametrize("squared", [True, False])
def test_gradcheck_local_ncc_loss_2d(window_size, squared):
    """Verify float64 gradcheck for local_ncc_loss_nd in 2D with odd kernel sizes."""
    torch.manual_seed(42)
    I = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)

    def func(x, y):
        return local_ncc_loss_nd(
            x, y,
            window_size=window_size,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("squared", [True, False])
def test_gradcheck_local_ncc_loss_3d(squared):
    """Verify float64 gradcheck for local_ncc_loss_nd in 3D."""
    torch.manual_seed(43)
    I = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64, requires_grad=True)

    def func(x, y):
        return local_ncc_loss_nd(
            x, y,
            window_size=3,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("squared", [True, False])
def test_gradcheck_box_lncc_loss_2d(squared):
    """Verify float64 gradcheck for BoxLNCCLoss and box_lncc_loss_nd in 2D."""
    torch.manual_seed(44)
    I = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True)

    loss_fn = BoxLNCCLoss(kernel_size=3, squared=squared)

    # Test module class
    assert torch.autograd.gradcheck(lambda a, b: loss_fn(a, b), (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)

    # Test functional wrapper
    assert torch.autograd.gradcheck(
        lambda a, b: box_lncc_loss_nd(a, b, window_size=3, squared=squared),
        (I, J),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )


@pytest.mark.parametrize("squared", [True, False])
def test_gradcheck_box_lncc_loss_3d(squared):
    """Verify float64 gradcheck for BoxLNCCLoss and box_lncc_loss_nd in 3D."""
    torch.manual_seed(45)
    I = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64, requires_grad=True)
    J = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64, requires_grad=True)

    loss_fn = BoxLNCCLoss(kernel_size=3, squared=squared)

    assert torch.autograd.gradcheck(lambda a, b: loss_fn(a, b), (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert torch.autograd.gradcheck(
        lambda a, b: box_lncc_loss_nd(a, b, window_size=3, squared=squared),
        (I, J),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )


@pytest.mark.parametrize("num_bins", [16, 32])
def test_gradcheck_canonical_mattes_mi_function(num_bins):
    """Verify float64 gradcheck for CanonicalMattesMIFunction analytical backward pass."""
    torch.manual_seed(46)
    N = 32
    x = torch.linspace(-0.8, 0.8, N, dtype=torch.float64, requires_grad=True)
    y = (torch.linspace(-0.7, 0.7, N, dtype=torch.float64) + 0.05 * torch.randn(N, dtype=torch.float64))
    y = y.detach().requires_grad_(True)

    def func(a, b):
        return CanonicalMattesMIFunction.apply(a, b, num_bins, -1.0, 1.0, 2.0, 1024, None)

    assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_mattes_mi_loss_nd_2d():
    """Verify float64 gradcheck for mattes_mi_loss_nd in 2D with fixed_range=(0.0, 1.0)."""
    torch.manual_seed(47)
    I = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.8 + 0.1
    J = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.8 + 0.1
    I.requires_grad_(True)
    J.requires_grad_(True)

    def func(a, b):
        return mattes_mi_loss_nd(a, b, num_bins=16, fixed_range=(0.0, 1.0), auto_mask=False)

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_mattes_mi_loss_nd_3d():
    """Verify float64 gradcheck for mattes_mi_loss_nd in 3D with fixed_range=(0.0, 1.0)."""
    torch.manual_seed(48)
    I = torch.rand(1, 1, 4, 4, 4, dtype=torch.float64) * 0.8 + 0.1
    J = torch.rand(1, 1, 4, 4, 4, dtype=torch.float64) * 0.8 + 0.1
    I.requires_grad_(True)
    J.requires_grad_(True)

    def func(a, b):
        return mattes_mi_loss_nd(a, b, num_bins=16, fixed_range=(0.0, 1.0), auto_mask=False)

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("use_mask", [False, True])
def test_gradcheck_soft_dice_loss_single_channel(use_mask):
    """Verify float64 gradcheck for single-channel soft_dice_loss_nd with and without mask."""
    torch.manual_seed(49)
    I = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.6 + 0.2
    J = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.6 + 0.2
    I.requires_grad_(True)
    J.requires_grad_(True)

    if use_mask:
        mask = (torch.rand(1, 1, 6, 6, dtype=torch.float64) > 0.3).to(torch.float64)
    else:
        mask = None

    def func(a, b):
        return soft_dice_loss_nd(a, b, mask=mask)

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("use_mask", [False, True])
def test_gradcheck_soft_dice_loss_multi_channel(use_mask):
    """Verify float64 gradcheck for multi-channel (C=3) soft_dice_loss_nd with and without mask."""
    torch.manual_seed(50)
    I = torch.rand(1, 3, 5, 5, dtype=torch.float64) * 0.6 + 0.2
    J = torch.rand(1, 3, 5, 5, dtype=torch.float64) * 0.6 + 0.2
    I.requires_grad_(True)
    J.requires_grad_(True)

    if use_mask:
        mask = (torch.rand(1, 1, 5, 5, dtype=torch.float64) > 0.3).to(torch.float64)
    else:
        mask = None

    def func(a, b):
        return soft_dice_loss_nd(a, b, mask=mask)

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("loss_name", ["mse", "l2", "mae"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_gradcheck_pointwise_losses(loss_name, use_mask):
    """Verify float64 gradcheck for canonical pointwise loss family."""
    if not HAS_POINTWISE:
        pytest.skip("Pointwise loss family (mse_loss_nd, l2_loss_nd, mae_loss_nd) not yet exported in syntx.core.losses")

    torch.manual_seed(51)
    if loss_name == "mae":
        # Keep I and J separated to avoid non-differentiable subgradient cusp at zero
        I = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.3 + 0.1
        J = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.3 + 0.6
    else:
        I = torch.randn(1, 1, 6, 6, dtype=torch.float64)
        J = torch.randn(1, 1, 6, 6, dtype=torch.float64)

    I.requires_grad_(True)
    J.requires_grad_(True)

    if use_mask:
        mask = (torch.rand(1, 1, 6, 6, dtype=torch.float64) > 0.2).to(torch.float64)
    else:
        mask = None

    fn_map = {
        "mse": mse_loss_nd,
        "l2": l2_loss_nd,
        "mae": mae_loss_nd,
    }
    loss_fn = fn_map[loss_name]

    def func(a, b):
        return loss_fn(a, b, mask=mask)

    assert torch.autograd.gradcheck(func, (I, J), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("spatial_dim", [2, 3])
def test_gradcheck_compute_soft_distance_transform(spatial_dim):
    """Verify float64 gradcheck for differentiable compute_soft_distance_transform."""
    torch.manual_seed(52)
    shape = (1, 1, 6, 6) if spatial_dim == 2 else (1, 1, 5, 5, 5)
    img = torch.rand(shape, dtype=torch.float64) * 0.8 + 0.1
    img.requires_grad_(True)

    def func(a):
        # threshold=0.5 ensures smooth infinitely differentiable sigmoid foreground mapping
        dt = compute_soft_distance_transform(a, sigma=2.0, threshold=0.5)
        return dt.sum()

    assert torch.autograd.gradcheck(func, (img,), eps=1e-6, atol=1e-4, rtol=1e-3)
