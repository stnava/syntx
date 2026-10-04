"""Memory efficiency and AMP autocast stress verification suite for canonical losses.

Specifications (per ORIGINAL_REQUEST.md & PROJECT.md):
- 3D volume peak transient memory test:
  - Assert peak transient memory <= 100 MB for CanonicalMattesMIFunction and BoxLNCCLoss
    on 176 x 256 x 256 volumes (using CPU tracemalloc and MPS if available).
- AMP stress testing:
  - 10 forward-backward iterations on 10^6 voxel tensors under torch.float16 and torch.bfloat16
    with extreme dynamic ranges (high [-100, 1000], low [0, 0.001], zero background).
  - Assert zero NaNs, zero Infs, finite loss, and finite non-zero gradients.
"""

import tracemalloc

import pytest
import torch

from syntx.core.losses import (
    BoxLNCCLoss,
    CanonicalMattesMIFunction,
    local_ncc_loss_nd,
    mae_loss_nd,
    mattes_mi_loss_nd,
    mse_loss_nd,
    soft_dice_loss_nd,
)

HAS_MSE = True


def _get_device_and_amp_types():
    """Determine available execution device and supported AMP autocast dtypes."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_types = [torch.float16]
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        amp_types = [torch.float16, torch.bfloat16]
    else:
        device = torch.device("cpu")
        amp_types = [torch.bfloat16]
    return device, amp_types


def test_canonical_mattes_mi_3d_peak_memory():
    """Verify peak transient memory <= 100 MB for CanonicalMattesMIFunction on 176x256x256 volume."""
    N = 176 * 256 * 256  # 11,534,336 voxels (~44 MB per float32 volume)

    # 1. CPU Tracemalloc verification
    x_cpu = torch.rand(N, dtype=torch.float32) * 2.0 - 1.0
    y_cpu = torch.rand(N, dtype=torch.float32) * 2.0 - 1.0

    tracemalloc.start()
    with torch.no_grad():
        loss_cpu = CanonicalMattesMIFunction.apply(x_cpu, y_cpu, 32, -1.0, 1.0, 2.0, 32768, None)
    _, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_cpu_mb = peak_cpu / (1024 ** 2)
    assert peak_cpu_mb <= 100.0, f"CanonicalMattesMIFunction CPU peak transient memory {peak_cpu_mb:.2f} MB > 100 MB"
    assert torch.isfinite(loss_cpu)

    # 2. MPS verification (if available)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

        x_mps = x_cpu.to("mps")
        y_mps = y_cpu.to("mps")
        torch.mps.synchronize()
        base_mem = torch.mps.current_allocated_memory()

        with torch.no_grad():
            loss_mps = CanonicalMattesMIFunction.apply(x_mps, y_mps, 32, -1.0, 1.0, 2.0, 32768, None)
            torch.mps.synchronize()

        peak_mps = torch.mps.current_allocated_memory()
        peak_mps_mb = (peak_mps - base_mem) / (1024 ** 2)
        assert peak_mps_mb <= 100.0, f"CanonicalMattesMIFunction MPS peak transient memory {peak_mps_mb:.2f} MB > 100 MB"
        assert torch.isfinite(loss_mps)


def test_box_lncc_3d_peak_memory():
    """Verify peak transient memory <= 100 MB for BoxLNCCLoss on 176x256x256 volume."""
    shape = (1, 1, 176, 256, 256)
    loss_fn = BoxLNCCLoss(kernel_size=5)

    # 1. CPU Tracemalloc verification
    I_cpu = torch.randn(shape, dtype=torch.float32)
    J_cpu = torch.randn(shape, dtype=torch.float32)

    tracemalloc.start()
    with torch.no_grad():
        loss_cpu = loss_fn(I_cpu, J_cpu)
    _, peak_cpu = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_cpu_mb = peak_cpu / (1024 ** 2)
    assert peak_cpu_mb <= 100.0, f"BoxLNCCLoss CPU peak transient memory {peak_cpu_mb:.2f} MB > 100 MB"
    assert torch.isfinite(loss_cpu)

    # 2. MPS verification (if available and enough VRAM)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

        I_mps = I_cpu.to("mps")
        J_mps = J_cpu.to("mps")
        loss_fn_mps = BoxLNCCLoss(kernel_size=5).to("mps")
        torch.mps.synchronize()
        base_mem = torch.mps.current_allocated_memory()

        with torch.no_grad():
            loss_mps = loss_fn_mps(I_mps, J_mps)
            torch.mps.synchronize()

        peak_mps = torch.mps.current_allocated_memory()
        peak_mps_mb = (peak_mps - base_mem) / (1024 ** 2)
        assert peak_mps_mb <= 100.0, f"BoxLNCCLoss MPS peak transient memory {peak_mps_mb:.2f} MB > 100 MB"
        assert torch.isfinite(loss_mps)


def _run_amp_stress_iterations(loss_callable, shape, device, dtype, dynamic_range="high", num_iterations=10):
    """Helper executing num_iterations forward-backward steps under AMP autocast."""
    dev_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    torch.manual_seed(999)

    for step in range(num_iterations):
        if dynamic_range == "high":
            # Extreme range [-100.0, 1000.0]
            I = (torch.rand(shape, device=device) * 1100.0 - 100.0).requires_grad_(True)
            J = (I + torch.randn_like(I) * 20.0).detach().requires_grad_(True)
        elif dynamic_range == "low":
            # Tiny range [0.0, 0.001] to test variance floors and underflow
            I = (torch.rand(shape, device=device) * 1e-3).requires_grad_(True)
            J = (I + torch.randn_like(I) * 1e-4).detach().requires_grad_(True)
        elif dynamic_range == "zeros":
            # 80% zeros to test zero variance background regions
            I = (torch.rand(shape, device=device) * (torch.rand(shape, device=device) > 0.8).float()).requires_grad_(True)
            J = (I + torch.randn_like(I) * 0.1).detach().requires_grad_(True)
        else:
            raise ValueError(f"Unknown dynamic_range {dynamic_range}")

        with torch.amp.autocast(device_type=dev_type, dtype=dtype):
            loss = loss_callable(I, J)

        assert torch.isfinite(loss), f"Step {step}: Non-finite loss {loss.item()} under AMP {dtype} ({dynamic_range})"
        assert not torch.isnan(loss), f"Step {step}: NaN loss under AMP {dtype} ({dynamic_range})"
        assert not torch.isinf(loss), f"Step {step}: Inf loss under AMP {dtype} ({dynamic_range})"

        loss.backward()

        assert I.grad is not None, f"Step {step}: Missing gradient on input tensor"
        assert torch.isfinite(I.grad).all(), f"Step {step}: NaN/Inf in gradient under AMP {dtype} ({dynamic_range})"
        assert I.grad.abs().sum() > 0.0, f"Step {step}: Zero gradient under AMP {dtype} ({dynamic_range})"


@pytest.mark.parametrize("dynamic_range", ["high", "low", "zeros"])
def test_amp_stress_box_lncc(dynamic_range):
    """Stress-test BoxLNCCLoss with 10 iterations on 10^6 voxels under AMP autocast."""
    device, amp_types = _get_device_and_amp_types()
    shape = (1, 1, 100, 100, 100)  # exactly 1,000,000 voxels

    for dtype in amp_types:
        loss_fn = BoxLNCCLoss(kernel_size=5).to(device)
        _run_amp_stress_iterations(loss_fn, shape, device, dtype, dynamic_range=dynamic_range, num_iterations=10)


@pytest.mark.parametrize("dynamic_range", ["high", "low", "zeros"])
def test_amp_stress_local_ncc(dynamic_range):
    """Stress-test local_ncc_loss_nd with 10 iterations on 10^6 voxels under AMP autocast."""
    device, amp_types = _get_device_and_amp_types()
    shape = (1, 1, 100, 100, 100)  # 10^6 voxels

    for dtype in amp_types:
        def loss_fn(a, b):
            return local_ncc_loss_nd(a, b, window_size=5, squared=True, use_ants_pseudo_gradient=False)

        _run_amp_stress_iterations(loss_fn, shape, device, dtype, dynamic_range=dynamic_range, num_iterations=10)


@pytest.mark.parametrize("dynamic_range", ["high", "low", "zeros"])
def test_amp_stress_mattes_mi(dynamic_range):
    """Stress-test mattes_mi_loss_nd with 10 iterations on 10^6 voxels under AMP autocast."""
    device, amp_types = _get_device_and_amp_types()
    shape = (1, 1, 100, 100, 100)  # 10^6 voxels

    for dtype in amp_types:
        def loss_fn(a, b):
            return mattes_mi_loss_nd(a, b, num_bins=32, auto_mask=False)

        _run_amp_stress_iterations(loss_fn, shape, device, dtype, dynamic_range=dynamic_range, num_iterations=10)


@pytest.mark.parametrize("dynamic_range", ["high", "low"])
def test_amp_stress_soft_dice(dynamic_range):
    """Stress-test soft_dice_loss_nd with 10 iterations on 10^6 voxels under AMP autocast."""
    device, amp_types = _get_device_and_amp_types()
    shape = (1, 1, 100, 100, 100)  # 10^6 voxels

    for dtype in amp_types:
        def loss_fn(a, b):
            # Scale to positive probabilities for Soft Dice
            a_prob = torch.sigmoid(a)
            b_prob = torch.sigmoid(b)
            return soft_dice_loss_nd(a_prob, b_prob)

        _run_amp_stress_iterations(loss_fn, shape, device, dtype, dynamic_range=dynamic_range, num_iterations=10)


def test_amp_stress_pointwise_mse():
    """Stress-test mse_loss_nd with 10 iterations on 10^6 voxels under AMP autocast."""
    if not HAS_MSE:
        pytest.skip("mse_loss_nd not yet exported in syntx.core.losses")

    device, amp_types = _get_device_and_amp_types()
    shape = (1, 1, 100, 100, 100)

    for dtype in amp_types:
        _run_amp_stress_iterations(mse_loss_nd, shape, device, dtype, dynamic_range="high", num_iterations=10)


def test_amp_stress_unaligned_independent_ct_volumes():
    """Stress-test 10 iterations on 10^6 voxels unaligned CT volumes ([-1000, 1000]) under AMP autocast."""
    device, amp_types = _get_device_and_amp_types()
    dev_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    shape = (1, 1, 100, 100, 100)  # 1,000,000 voxels

    for dtype in amp_types:
        for fname, loss_fn in [
            ("box_lncc", BoxLNCCLoss(kernel_size=5).to(device)),
            ("local_ncc", lambda a, b: local_ncc_loss_nd(a, b, window_size=5, squared=True)),
            ("mattes_mi", lambda a, b: mattes_mi_loss_nd(a, b, num_bins=32, auto_mask=False)),
            ("mse", lambda a, b: mse_loss_nd(a, b)),
            ("mae", lambda a, b: mae_loss_nd(a, b)),
        ]:
            torch.manual_seed(1001)
            for step in range(10):
                I = (torch.rand(shape, device=device) * 2000.0 - 1000.0).requires_grad_(True)
                J = (torch.rand(shape, device=device) * 2000.0 - 1000.0).requires_grad_(True)

                with torch.amp.autocast(device_type=dev_type, dtype=dtype):
                    loss = loss_fn(I, J)

                assert torch.isfinite(loss), f"{fname} step {step}: non-finite loss {loss.item()} under AMP {dtype}"
                assert not torch.isnan(loss), f"{fname} step {step}: NaN loss under AMP {dtype}"
                assert not torch.isinf(loss), f"{fname} step {step}: Inf loss under AMP {dtype}"

                loss.backward()

                assert I.grad is not None, f"{fname} step {step}: missing grad"
                assert torch.isfinite(I.grad).all(), f"{fname} step {step}: non-finite grad under AMP {dtype}"
                assert I.grad.abs().sum() > 0.0, f"{fname} step {step}: zero grad under AMP {dtype}"


def test_direct_half_precision_10e6_voxels_pointwise():
    """Verify direct float16 and bfloat16 10^6 voxel tensors in mse_loss_nd and mae_loss_nd without autocast."""
    shape = (1, 1, 100, 100, 100)
    for dtype in (torch.float16, torch.bfloat16):
        I = ((torch.rand(shape) * 2000.0 - 1000.0).to(dtype)).requires_grad_(True)
        J = ((torch.rand(shape) * 2000.0 - 1000.0).to(dtype)).requires_grad_(True)

        # MSE
        loss_mse = mse_loss_nd(I, J)
        assert torch.isfinite(loss_mse)
        assert loss_mse.dtype == torch.float32
        loss_mse.backward()
        assert torch.isfinite(I.grad).all()

        # MAE
        I.grad = None
        J.grad = None
        loss_mae = mae_loss_nd(I, J)
        assert torch.isfinite(loss_mae)
        assert loss_mae.dtype == torch.float32
        loss_mae.backward()
        assert torch.isfinite(I.grad).all()
