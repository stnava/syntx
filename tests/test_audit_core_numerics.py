"""Regression tests for the docstring-audit behaviour issues in core/losses.py,
core/smoothing.py, core/optimizers.py and core/mps_kernels.py. CPU only, tiny tensors."""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------------------
# core/losses.py
# ---------------------------------------------------------------------------------------

def test_mattes_sampling_uses_the_requested_fraction(monkeypatch):
    import syntx.core.losses as L
    seen = {}
    real = L.parzen_weights

    def spy(v, *a, **k):
        seen.setdefault("n", v.numel())
        return real(v, *a, **k)
    monkeypatch.setattr(L, "parzen_weights", spy)
    x = torch.rand(1000) * 2 - 1
    L.mattes_mi_loss_core(x, x.clone(), sampling_percentage=0.6)
    assert seen["n"] == 600                                   # was 1000 (stride int(1/0.6) = 1)


def test_mattes_empty_input_keeps_the_graph():
    from syntx.core.losses import mattes_mi_loss_nd
    I = torch.zeros(4, 4, requires_grad=True)
    loss = mattes_mi_loss_nd(I, torch.zeros(4, 4))           # auto_mask selects nothing
    loss.backward()
    assert I.grad is not None


def test_mattes_fixed_weights_mismatch_raises():
    from syntx.core.losses import mattes_mi_loss_core, parzen_weights
    x = torch.rand(100) * 2 - 1
    with pytest.raises(ValueError, match="fixed_weights"):
        mattes_mi_loss_core(x, x, fixed_weights=parzen_weights(x[:50]))


def test_parzen_weights_ignore_nan_samples():
    from syntx.core.losses import parzen_weights
    w = parzen_weights(torch.tensor([0.0, float("nan"), 0.5]))
    assert torch.allclose(w[1], torch.zeros_like(w[1]))      # no mass in the middle bin
    assert torch.allclose(w[0].sum(), torch.tensor(1.0))


def test_mattes_auto_mask_is_scale_relative():
    from syntx.core.losses import mattes_mi_loss_nd
    rng = np.random.default_rng(0)
    a = torch.tensor(rng.random((16, 16)), dtype=torch.float32)
    b = torch.tensor(rng.random((16, 16)), dtype=torch.float32)
    a[:4] = 0.005; b[:4] = 0.005                               # background (below 1 % of max)
    v1 = mattes_mi_loss_nd(a, b)
    v2 = mattes_mi_loss_nd(a * 1000, b * 1000)               # same image, raw-scale units
    assert torch.allclose(v1, v2, atol=1e-5)


def test_local_ncc_even_window_raises():
    from syntx.core.losses import local_ncc_loss_nd
    with pytest.raises(ValueError, match="odd"):
        local_ncc_loss_nd(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16), window_size=4)


def test_analytic_lncc_border_windows_use_true_counts():
    import torch.nn.functional as F
    from syntx.core.losses import local_ncc_loss_nd
    torch.manual_seed(0)
    I = torch.rand(1, 1, 12, 12, requires_grad=True)
    J = torch.rand(1, 1, 12, 12)
    local_ncc_loss_nd(I, J, window_size=5, use_ants_pseudo_gradient=True, squared=True).backward()
    # reference: the ITK pseudo-derivative with N = number of in-image voxels of each window
    box = lambda x: F.avg_pool2d(x, 5, 1, 2, count_include_pad=False)
    Id = I.detach()
    Fc, Mc = Id - box(Id), J - box(J)
    sFF = box(Fc ** 2).clamp(min=1e-6); sMM = box(Mc ** 2).clamp(min=1e-6); sFM = box(Fc * Mc)
    N = F.avg_pool2d(torch.ones_like(Id), 5, 1, 2, count_include_pad=True) * 25
    ref = -2.0 / N * (sFM / (sFF * sMM + 1e-8)) * (Mc - sFM / (sFF + 1e-8) * Fc) / Id.numel()
    assert torch.allclose(I.grad, ref, atol=1e-6, rtol=1e-4)


def test_box_lncc_even_kernel_raises():
    from syntx.core.losses import BoxLNCCLoss
    with pytest.raises(ValueError, match="odd"):
        BoxLNCCLoss(kernel_size=4)


def _disk(r=5, n=24, c=12):
    yy, xx = np.mgrid[:n, :n]
    return torch.tensor(((yy - c) ** 2 + (xx - c) ** 2 <= r * r).astype(np.float32))[None, None]


def test_sdf_mse_is_signed_and_differs_from_edt():
    from syntx.core.losses import distance_transform_loss, _soft_signed_distance
    a = _disk(5)
    d = _soft_signed_distance(a, sigma=1.0)
    assert d[0, 0, 12, 12] < 0 < d[0, 0, 0, 0]               # negative inside, positive outside
    b = _disk(7)
    assert not torch.allclose(distance_transform_loss(a, b, mode="sdf_mse", tau=0.1),
                              distance_transform_loss(a, b, mode="edt_mse", tau=0.1))


def test_distance_loss_requires_positive_tau():
    from syntx.core.losses import distance_transform_loss
    a = _disk(5)
    for tau in (0.0, None, -1.0):
        with pytest.raises(ValueError, match="tau"):
            distance_transform_loss(a, a, mode="potential_mse", tau=tau)


def test_image_distance_transform_float_output_and_signed_full_foreground():
    from syntx.core.losses import compute_image_distance_transform
    m = torch.zeros(10, 10, dtype=torch.bool); m[:, 5:] = True
    d = compute_image_distance_transform(m)
    assert d.dtype == torch.float32 and d[0, 0] == 5.0        # was truncated to bool
    full = torch.ones(6, 6)
    s = compute_image_distance_transform(full, signed=True)
    assert (s < 0).all()                                      # inside everywhere
