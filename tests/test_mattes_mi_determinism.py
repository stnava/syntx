"""
Mattes MI must be deterministic and device-consistent, including on Apple MPS where a single
large matmul over Parzen weights is not (see core.losses._parzen_joint_histogram).
"""
import numpy as np
import pytest
import torch

from syntx.core.losses import _parzen_joint_histogram, b_spline_3, mattes_mi_loss_nd

MPS = torch.backends.mps.is_available()


def _weights(v, num_bins=32):
    u = 2.0 + (v.view(-1, 1) + 1.0) * ((num_bins - 1 - 4.0) / 2.0)
    return b_spline_3(u - torch.arange(num_bins, device=v.device, dtype=torch.float32)[None])


def _concentrated(n=725_370, seed=0):
    torch.manual_seed(seed)
    x = torch.full((n,), -1.0); x[: n // 3] = torch.rand(n // 3) * 2 - 1
    y = torch.rand(n) * 2 - 1
    return x, y


def test_parzen_histogram_matches_float64_reference():
    x, y = _concentrated(200_000)
    ref = _weights(x).double().t() @ _weights(y).double()
    h = _parzen_joint_histogram(_weights(x), _weights(y)).double()
    assert float((h - ref).abs().max()) < 5e-3            # fp32 accumulation only
    assert abs(float(h.sum()) - 200_000) < 0.5             # partition of unity


def test_parzen_histogram_partition_of_unity_and_grad():
    x, y = _concentrated(50_000)
    wx = _weights(x).requires_grad_(True)
    h = _parzen_joint_histogram(wx, _weights(y))
    h.sum().backward()
    assert wx.grad is not None and torch.isfinite(wx.grad).all()


@pytest.mark.skipif(not MPS, reason="MPS only")
def test_parzen_histogram_deterministic_on_mps_across_allocations():
    x, y = _concentrated()
    ref = (_weights(x).double().t() @ _weights(y).double())
    rng = np.random.default_rng(1)
    hs = []
    for _ in range(4):
        junk = [torch.empty(int(rng.integers(1, 8_000_000)), device="mps") for _ in range(int(rng.integers(1, 4)))]
        hs.append(_parzen_joint_histogram(_weights(x.to("mps")), _weights(y.to("mps"))).cpu().double())
        del junk
    for h in hs[1:]:
        assert torch.equal(h, hs[0])
    assert float((hs[0] - ref).abs().max()) < 5e-3


@pytest.mark.skipif(not MPS, reason="MPS only")
def test_mattes_mi_mps_matches_cpu():
    torch.manual_seed(0)
    I = torch.rand(1, 1, 40, 96, 96); J = (0.7 * I + 0.3 * torch.rand_like(I)) * (I > 0.2)
    I = I * (I > 0.2)
    l_cpu = mattes_mi_loss_nd(I, J, num_bins=32, sampling_percentage=0.5).item()
    l_mps = [mattes_mi_loss_nd(I.to("mps"), J.to("mps"), num_bins=32, sampling_percentage=0.5).item() for _ in range(3)]
    assert l_mps[0] == l_mps[1] == l_mps[2]
    assert abs(l_mps[0] - l_cpu) < 1e-4


def test_fixed_range_makes_mi_independent_of_padding_value():
    """With fixed bounds, changing the (masked-out) background does not move the histogram axes."""
    torch.manual_seed(0)
    I = torch.rand(1, 1, 24, 48, 48); J = torch.rand(1, 1, 24, 48, 48)
    mask = I > 0.3
    I = I * mask; J = J * mask
    J2 = J.clone(); J2[~mask] = 5.0                          # different background outside the mask
    l1 = mattes_mi_loss_nd(I, J, mask=mask, auto_mask=False, fixed_range=(0.0, 1.0)).item()
    l2 = mattes_mi_loss_nd(I, J2, mask=mask, auto_mask=False, fixed_range=(0.0, 1.0)).item()
    assert l1 == l2
    # per-call bounds: shrinking the warped image's range inside the mask changes the value
    l3 = mattes_mi_loss_nd(I, J * 0.5, mask=mask, auto_mask=False, fixed_range=(0.0, 1.0)).item()
    assert l3 != l1
