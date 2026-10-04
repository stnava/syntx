"""Empirical Adversarial Stress Testing Suite for Canonical Losses Core (Milestone 1).

Adversarial Challenger Suite verifying:
1. Flat / Zero-variance images (constant volumes): zero NaNs, zero division-by-zero crashes.
2. Extreme intensity ranges ([-10000, 10000], [0, 1e-5], etc.) under float32 and AMP.
3. Empty / all-zero spatial masks: graceful reduction and finite loss.
4. Non-contiguous strided tensor inputs: strict parity against contiguous equivalents (GEMINI.md Rule 4).
5. Edge-case shapes (non-square 2D, non-cubic 3D, odd kernels) and float64 gradcheck.
"""

import pytest
import torch
import torch.autograd

from syntx.core.losses import (
    BoxLNCCLoss,
    l2_loss_nd,
    local_ncc_loss_nd,
    mae_loss_nd,
    mattes_mi_loss_nd,
    mse_loss_nd,
    soft_dice_loss_nd,
)

# ==============================================================================
# 1. Flat / Zero-Variance Stress Tests
# ==============================================================================

@pytest.mark.parametrize(
    "name,I_val,J_val,dim",
    [
        ("both_ones_2d", 1.0, 1.0, 2),
        ("both_zeros_2d", 0.0, 0.0, 2),
        ("diff_constants_2d", 5.0, -2.0, 2),
        ("both_ones_3d", 1.0, 1.0, 3),
        ("diff_constants_3d", 10.0, 0.0, 3),
    ],
)
def test_flat_zero_variance_loss_functionals(name, I_val, J_val, dim):
    """Verify flat / zero-variance volumes produce finite losses and finite gradients."""
    shape = (1, 1, 16, 16) if dim == 2 else (1, 1, 8, 8, 8)
    I = torch.full(shape, I_val, dtype=torch.float32, requires_grad=True)
    J = torch.full(shape, J_val, dtype=torch.float32, requires_grad=True)

    # 1.1 Local NCC (autograd)
    loss_ncc = local_ncc_loss_nd(I, J, window_size=5, squared=True)
    assert torch.isfinite(loss_ncc), f"Local NCC loss non-finite on {name}: {loss_ncc}"
    loss_ncc.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.2 Local NCC (pseudo / analytical)
    I.grad = None
    J.grad = None
    loss_pseudo = local_ncc_loss_nd(I, J, window_size=5, squared=True, use_ants_pseudo_gradient=True)
    assert torch.isfinite(loss_pseudo), f"Local NCC pseudo loss non-finite on {name}: {loss_pseudo}"
    loss_pseudo.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.3 Box LNCC
    I.grad = None
    J.grad = None
    box_loss_fn = BoxLNCCLoss(kernel_size=5, squared=True)
    loss_box = box_loss_fn(I, J)
    assert torch.isfinite(loss_box), f"Box LNCC loss non-finite on {name}: {loss_box}"
    loss_box.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.4 Mattes MI (fixed range)
    I.grad = None
    J.grad = None
    loss_mi = mattes_mi_loss_nd(I, J, num_bins=16, fixed_range=(-5.0, 15.0), auto_mask=False)
    assert torch.isfinite(loss_mi), f"Mattes MI fixed range loss non-finite on {name}: {loss_mi}"
    loss_mi.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.5 Mattes MI (auto mask / dynamic range)
    I.grad = None
    J.grad = None
    loss_mi_auto = mattes_mi_loss_nd(I, J, num_bins=16, fixed_range=None, auto_mask=True)
    assert torch.isfinite(loss_mi_auto), f"Mattes MI auto mask loss non-finite on {name}: {loss_mi_auto}"
    loss_mi_auto.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.6 MSE, L2, MAE
    for fn in (mse_loss_nd, lambda a, b: l2_loss_nd(a, b, squared=True), mae_loss_nd):
        I.grad = None
        J.grad = None
        l_pw = fn(I, J)
        assert torch.isfinite(l_pw)
        l_pw.backward()
        assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 1.7 Soft Dice
    I.grad = None
    J.grad = None
    l_dice = soft_dice_loss_nd(torch.clamp(I, 0.0, 1.0), torch.clamp(J, 0.0, 1.0))
    assert torch.isfinite(l_dice)
    l_dice.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()


def test_flat_one_random_one_constant():
    """Verify local NCC and Mattes MI when one image is constant and the other is random."""
    shape = (1, 1, 16, 16)
    I = torch.full(shape, 3.0, requires_grad=True)
    J = torch.randn(shape, requires_grad=True)

    # Autograd NCC
    loss = local_ncc_loss_nd(I, J, window_size=5)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # Box LNCC
    I.grad = None
    J.grad = None
    box_fn = BoxLNCCLoss(kernel_size=5)
    loss = box_fn(I, J)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # Mattes MI
    I.grad = None
    J.grad = None
    loss = mattes_mi_loss_nd(I, J, num_bins=16, fixed_range=(-5.0, 5.0), auto_mask=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()


# ==============================================================================
# 2. Extreme Dynamic Ranges & AMP Float16
# ==============================================================================

@pytest.mark.parametrize(
    "range_name,low,high",
    [
        ("huge_ct", -10000.0, 10000.0),
        ("ultra_huge", -1e5, 1e5),
        ("tiny_micro", 0.0, 1e-5),
        ("subnormal", 0.0, 1e-9),
    ],
)
def test_extreme_ranges_float32(range_name, low, high):
    """Verify all loss functions operate stably under extreme dynamic ranges in float32."""
    torch.manual_seed(201)
    shape = (1, 1, 16, 16)
    I = (torch.rand(shape) * (high - low) + low).requires_grad_(True)
    J = (torch.rand(shape) * (high - low) + low).requires_grad_(True)

    # Test each functional
    funcs = [
        ("mse", lambda a, b: mse_loss_nd(a, b)),
        ("l2", lambda a, b: l2_loss_nd(a, b)),
        ("mae", lambda a, b: mae_loss_nd(a, b)),
        ("local_ncc", lambda a, b: local_ncc_loss_nd(a, b, window_size=5)),
        ("box_lncc", lambda a, b: BoxLNCCLoss(kernel_size=5)(a, b)),
        ("mattes_mi", lambda a, b: mattes_mi_loss_nd(a, b, num_bins=16, fixed_range=(low, high), auto_mask=False)),
        ("soft_dice", lambda a, b: soft_dice_loss_nd(torch.sigmoid(a), torch.sigmoid(b))),
    ]

    for fname, fn in funcs:
        I_t = I.clone().detach().requires_grad_(True)
        J_t = J.clone().detach().requires_grad_(True)
        loss = fn(I_t, J_t)
        assert torch.isfinite(loss), f"{fname} produced non-finite loss on {range_name}: {loss.item()}"
        loss.backward()
        assert torch.isfinite(I_t.grad).all() and torch.isfinite(J_t.grad).all(), f"{fname} gradient non-finite on {range_name}"


def test_float16_normalized_inputs():
    """Verify all loss functions handle float16 inputs safely when normalized to [0, 1] per GEMINI.md Rule 2."""
    torch.manual_seed(202)
    shape = (1, 1, 16, 16)
    I = torch.rand(shape, dtype=torch.float16, requires_grad=True)
    J = torch.rand(shape, dtype=torch.float16, requires_grad=True)

    # Under normalized [0, 1] range, all metrics including MSE remain well below 65504
    l_mse = mse_loss_nd(I, J)
    assert torch.isfinite(l_mse)
    l_mse.backward()
    assert torch.isfinite(I.grad).all()

    I.grad = None
    l_mae = mae_loss_nd(I, J)
    assert torch.isfinite(l_mae)
    l_mae.backward()
    assert torch.isfinite(I.grad).all()

    I.grad = None
    l_ncc = local_ncc_loss_nd(I, J, window_size=5)
    assert torch.isfinite(l_ncc)
    l_ncc.backward()
    assert torch.isfinite(I.grad).all()

    I.grad = None
    l_dice = soft_dice_loss_nd(I, J)
    assert torch.isfinite(l_dice)
    l_dice.backward()
    assert torch.isfinite(I.grad).all()


@pytest.mark.parametrize(
    "range_name,low,high",
    [
        ("ct_hounsfield", -1000.0, 1000.0),
        ("ct_bone_window", -1000.0, 3000.0),
        ("high_dynamic_range", -10000.0, 10000.0),
        ("boundary_fp16_range", -60000.0, 60000.0),
        ("micro_range", 0.0, 1e-5),
        ("subnormal_range", 0.0, 1e-9),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_unnormalized_dynamic_ranges(range_name, low, high, dtype):
    """Verify canonical loss functionals handle unnormalized dynamic ranges in half-precision."""
    torch.manual_seed(203)
    shape = (1, 1, 16, 16)
    I_raw = (torch.rand(shape) * (high - low) + low).to(dtype)
    J_raw = (torch.rand(shape) * (high - low) + low).to(dtype)

    funcs = [
        ("mse", lambda a, b: mse_loss_nd(a, b), True),
        ("l2", lambda a, b: l2_loss_nd(a, b), True),
        ("mae", lambda a, b: mae_loss_nd(a, b), True),
        ("local_ncc_autograd", lambda a, b: local_ncc_loss_nd(a, b, window_size=5, squared=True, use_ants_pseudo_gradient=False), False),
        ("local_ncc_pseudo", lambda a, b: local_ncc_loss_nd(a, b, window_size=5, squared=True, use_ants_pseudo_gradient=True), False),
        ("box_lncc", lambda a, b: BoxLNCCLoss(kernel_size=5)(a, b), False),
        ("mattes_mi", lambda a, b: mattes_mi_loss_nd(a, b, num_bins=16, fixed_range=(low, high), auto_mask=False), False),
    ]

    for fname, fn, is_pointwise in funcs:
        I = I_raw.clone().requires_grad_(True)
        J = J_raw.clone().requires_grad_(True)
        loss = fn(I, J)
        assert torch.isfinite(loss), f"{fname} produced non-finite loss on {range_name} ({dtype}): {loss.item()}"
        if is_pointwise:
            assert loss.dtype == torch.float32, f"{fname} must return float32 for {dtype} input, got {loss.dtype}"
        loss.backward()
        assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all(), f"{fname} gradient non-finite on {range_name} ({dtype})"


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_device_half_precision_unnormalized_dynamic_ranges(device_name):
    """Verify half-precision unnormalized dynamic ranges across CPU, MPS, and CUDA devices."""
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device_name)
    torch.manual_seed(204)
    shape = (1, 1, 16, 16)
    # CT Hounsfield range [-1000, 1000]
    I = ((torch.rand(shape, device=device) * 2000.0 - 1000.0).to(torch.float16)).requires_grad_(True)
    J = ((torch.rand(shape, device=device) * 2000.0 - 1000.0).to(torch.float16)).requires_grad_(True)

    # 1. Pointwise MSE
    loss_mse = mse_loss_nd(I, J)
    assert torch.isfinite(loss_mse), f"MSE non-finite on {device_name}"
    assert loss_mse.dtype == torch.float32, f"MSE return dtype should be float32 on {device_name}"
    loss_mse.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 2. Local NCC Analytical / Pseudo on device
    I.grad = None
    J.grad = None
    loss_pseudo = local_ncc_loss_nd(I, J, window_size=5, squared=True, use_ants_pseudo_gradient=True)
    assert torch.isfinite(loss_pseudo), f"Local NCC pseudo non-finite on {device_name}"
    loss_pseudo.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()

    # 3. Local NCC Autograd on device
    I.grad = None
    J.grad = None
    loss_autograd = local_ncc_loss_nd(I, J, window_size=5, squared=True, use_ants_pseudo_gradient=False)
    assert torch.isfinite(loss_autograd), f"Local NCC autograd non-finite on {device_name}"
    loss_autograd.backward()
    assert torch.isfinite(I.grad).all() and torch.isfinite(J.grad).all()


# ==============================================================================
# 3. Empty & Sparse Masks
# ==============================================================================

@pytest.mark.parametrize("dim", [2, 3])
def test_all_zero_spatial_masks(dim):
    """Verify all loss functions gracefully handle all-zero masks without NaN or division by zero."""
    shape = (1, 1, 16, 16) if dim == 2 else (1, 1, 8, 8, 8)
    I = torch.rand(shape, requires_grad=True)
    J = torch.rand(shape, requires_grad=True)
    mask = torch.zeros(shape)

    # Pointwise
    for fn in (mse_loss_nd, l2_loss_nd, mae_loss_nd):
        loss = fn(I, J, mask=mask)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0

    # Cross Correlation
    l_ncc = local_ncc_loss_nd(I, J, mask=mask, window_size=3)
    assert torch.isfinite(l_ncc)

    l_box = BoxLNCCLoss(kernel_size=3)(I, J, mask=mask)
    assert torch.isfinite(l_box)

    # Mattes MI
    l_mi = mattes_mi_loss_nd(I, J, mask=mask, auto_mask=False, fixed_range=(0.0, 1.0))
    assert torch.isfinite(l_mi)
    assert l_mi.item() == 0.0

    # Soft Dice
    l_dice = soft_dice_loss_nd(I, J, mask=mask)
    assert torch.isfinite(l_dice)
    assert l_dice.item() == 1.0


def test_single_active_voxel_mask():
    """Verify robust execution when exactly one voxel in the mask is active."""
    shape = (1, 1, 16, 16)
    I = torch.rand(shape, requires_grad=True)
    J = torch.rand(shape, requires_grad=True)
    mask = torch.zeros(shape)
    mask[0, 0, 8, 8] = 1.0

    # MSE
    l_mse = mse_loss_nd(I, J, mask=mask)
    assert torch.isfinite(l_mse)
    l_mse.backward()
    assert torch.isfinite(I.grad).all()

    # Mattes MI
    I.grad = None
    l_mi = mattes_mi_loss_nd(I, J, mask=mask, auto_mask=False, fixed_range=(0.0, 1.0))
    assert torch.isfinite(l_mi)
    l_mi.backward()
    assert torch.isfinite(I.grad).all()

    # Soft Dice
    I.grad = None
    l_dice = soft_dice_loss_nd(I, J, mask=mask)
    assert torch.isfinite(l_dice)
    l_dice.backward()
    assert torch.isfinite(I.grad).all()


# ==============================================================================
# 4. Non-Contiguous Strided Tensor Parity (GEMINI.md Rule 4)
# ==============================================================================

def test_non_contiguous_strided_tensor_parity_2d():
    """Verify non-contiguous sliced/permuted/transposed inputs match contiguous outputs bitwise."""
    torch.manual_seed(401)
    base = torch.randn(2, 3, 16, 20)
    # Permuted non-contiguous view
    I_nc = base.permute(0, 2, 1, 3)
    assert not I_nc.is_contiguous()
    J_nc = I_nc.clone()

    I_c = I_nc.contiguous()
    J_c = J_nc.contiguous()

    # MSE
    diff_mse = (mse_loss_nd(I_nc, J_nc) - mse_loss_nd(I_c, J_c)).abs().item()
    assert diff_mse < 1e-6

    # Box LNCC
    box_fn = BoxLNCCLoss(kernel_size=3)
    diff_box = (box_fn(I_nc, J_nc) - box_fn(I_c, J_c)).abs().item()
    assert diff_box < 1e-6

    # Local NCC
    diff_ncc = (local_ncc_loss_nd(I_nc, J_nc, window_size=3) - local_ncc_loss_nd(I_c, J_c, window_size=3)).abs().item()
    assert diff_ncc < 1e-6

    # Mattes MI
    diff_mi = (
        mattes_mi_loss_nd(I_nc, J_nc, num_bins=16, fixed_range=(-3.0, 3.0), auto_mask=False)
        - mattes_mi_loss_nd(I_c, J_c, num_bins=16, fixed_range=(-3.0, 3.0), auto_mask=False)
    ).abs().item()
    assert diff_mi < 1e-6


def test_non_contiguous_strided_tensor_parity_3d():
    """Verify 3D non-contiguous strided slice inputs match contiguous outputs."""
    torch.manual_seed(402)
    base = torch.randn(1, 1, 16, 16, 16)
    I_nc = base[:, :, ::2, ::2, ::2]
    assert not I_nc.is_contiguous()
    J_nc = I_nc.clone()

    I_c = I_nc.contiguous()
    J_c = J_nc.contiguous()

    assert (mse_loss_nd(I_nc, J_nc) - mse_loss_nd(I_c, J_c)).abs().item() < 1e-6
    assert (BoxLNCCLoss(kernel_size=3)(I_nc, J_nc) - BoxLNCCLoss(kernel_size=3)(I_c, J_c)).abs().item() < 1e-6
    assert (local_ncc_loss_nd(I_nc, J_nc, window_size=3) - local_ncc_loss_nd(I_c, J_c, window_size=3)).abs().item() < 1e-6


# ==============================================================================
# 5. Edge-Case Shapes and Float64 Gradcheck
# ==============================================================================

@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 7, 13),      # Non-square 2D
        (1, 1, 5, 23),      # Extreme aspect ratio 2D
        (2, 2, 7, 9),       # Multi-batch, multi-channel 2D
        (1, 1, 5, 9, 7),    # Non-cubic 3D
        (1, 1, 4, 11, 6),   # Non-cubic 3D
    ],
)
def test_gradcheck_edge_shapes_local_ncc(shape):
    """Verify float64 gradcheck passes for local_ncc_loss_nd on anisotropic shapes."""
    torch.manual_seed(501)
    I = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    J = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: local_ncc_loss_nd(a, b, window_size=3, squared=True),
        (I, J),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 7, 13),
        (1, 1, 5, 23),
        (1, 1, 5, 9, 7),
    ],
)
def test_gradcheck_edge_shapes_box_lncc(shape):
    """Verify float64 gradcheck passes for BoxLNCCLoss on anisotropic shapes."""
    torch.manual_seed(502)
    I = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    J = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    box_fn = BoxLNCCLoss(kernel_size=3, squared=True)
    assert torch.autograd.gradcheck(
        lambda a, b: box_fn(a, b),
        (I, J),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_window_size_larger_than_spatial_extent():
    """Verify loss functionals safely handle window sizes larger than spatial image extent."""
    shape = (1, 1, 3, 5)
    I = torch.randn(shape, requires_grad=True)
    J = torch.randn(shape, requires_grad=True)

    # local_ncc dynamically adapts kernel to fit domain
    l_ncc = local_ncc_loss_nd(I, J, window_size=9)
    assert torch.isfinite(l_ncc)
    l_ncc.backward()
    assert torch.isfinite(I.grad).all()

    # Box LNCC with kernel_size=9 on shape 3x5
    I.grad = None
    box_fn = BoxLNCCLoss(kernel_size=9)
    l_box = box_fn(I, J)
    assert torch.isfinite(l_box)
    l_box.backward()
    assert torch.isfinite(I.grad).all()
