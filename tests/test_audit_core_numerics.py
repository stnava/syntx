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


# ---------------------------------------------------------------------------------------
# core/smoothing.py
# ---------------------------------------------------------------------------------------

def test_boundary_mask_rim_zero_is_all_ones():
    from syntx.core.smoothing import get_boundary_mask
    m = get_boundary_mask((5, 6), "cpu", torch.float32, rim_size=0)
    assert torch.all(m == 1)


@pytest.mark.parametrize("fn_name", ["separable_gaussian_filter", "fast_separable_gaussian_filter"])
def test_gaussian_filters_validate_sigma_and_spacing(fn_name):
    import syntx.core.smoothing as sm
    fn = getattr(sm, fn_name)
    g = torch.rand(1, 8, 9, 10, 3)
    with pytest.raises(ValueError, match="one sigma per spatial axis"):
        fn(g, [1.0, 1.0])                                     # 2 values for 3 axes
    with pytest.raises(ValueError, match="spacing"):
        fn(g, 1.0, spacing=(1.0, 2.0, 3.0))                  # spacing would be ignored
    with pytest.raises(ValueError, match="sigma_mode"):
        fn(g, 1.0, sigma_mode="mm")
    assert fn(g, 1.0, spacing=(1.0, 2.0, 3.0), sigma_mode="physical").shape == g.shape


def test_gaussian_filter_kernel_type_validation():
    from syntx.core.smoothing import separable_gaussian_filter
    g = torch.rand(1, 8, 9, 2)
    a = separable_gaussian_filter(g, 1.0)
    assert torch.equal(a, separable_gaussian_filter(g, 1.0, kernel_type="gaussian"))  # alias
    with pytest.raises(ValueError, match="kernel_type"):
        separable_gaussian_filter(g, 1.0, kernel_type="besel")


def test_separable_1d_filter_rejects_wrong_kernel_count():
    from syntx.core.smoothing import separable_1d_filter
    k = torch.tensor([0.25, 0.5, 0.25])
    with pytest.raises(ValueError, match="kernels"):
        separable_1d_filter(torch.rand(1, 1, 6, 6), [k])


def test_bspline_spline_distance_follows_coord_convention():
    pytest.importorskip("antstorch")
    from syntx.core.smoothing import smooth_displacement_field_bspline
    torch.manual_seed(0)
    f = torch.randn(1, 20, 30, 2)                             # tensor (y, x) grid
    a = smooth_displacement_field_bspline(f, spacing=(1.0, 1.0), spline_distance=(4.0, 12.0),
                                          coord_convention="xyz")
    b = smooth_displacement_field_bspline(f, spacing=(1.0, 1.0), spline_distance=(12.0, 4.0),
                                          coord_convention="zyx")
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------------------
# core/optimizers.py
# ---------------------------------------------------------------------------------------

def _regadam_step(**kw):
    from syntx.core.optimizers import RegAdam
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(1, 12, 14, 2))
    opt = RegAdam([p], lr=0.5, max_step_norm=1e9, **kw)
    p.grad = torch.randn_like(p)
    opt.step()
    return p.detach().clone()


def test_regadam_dsti_uses_dsti_alpha_and_spacing():
    a = _regadam_step(regularizer="dsti", sobolev_alpha=0.5)
    b = _regadam_step(regularizer="dsti", sobolev_alpha=0.5, dsti_alpha=2.0)
    assert not torch.allclose(a, b)                           # dsti_alpha now takes effect
    c = _regadam_step(regularizer="dsti", sobolev_alpha=0.5, spacing=(1.0, 3.0))
    assert not torch.allclose(a, c)                           # and spacing does too


def test_regadam_unknown_regularizer_raises():
    from syntx.core.optimizers import RegAdam
    with pytest.raises(ValueError, match="regularizer"):
        RegAdam([torch.nn.Parameter(torch.zeros(1, 4, 4, 2))], regularizer="sobolv")


def test_regadam_step_bound_without_host_sync(monkeypatch):
    from syntx.core.optimizers import RegAdam
    calls = []
    real_item = torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "item", lambda self: calls.append(1) or real_item(self))
    p = torch.nn.Parameter(torch.zeros(1, 8, 8, 2))
    opt = RegAdam([p], lr=1.0, regularizer="none", max_step_norm=0.1)
    p.grad = torch.randn_like(p) * 10
    opt.step()
    assert not calls                                          # bound computed on the device
    step = p.detach().norm(dim=-1).max()
    assert step <= 0.1 + 1e-6


# ---------------------------------------------------------------------------------------
# core/mps_kernels.py (validation only; the kernels need MPS)
# ---------------------------------------------------------------------------------------

def test_mps_backward_rejects_unsupported_padding_mode():
    from syntx.core.mps_kernels import grid_sample_backward_mps
    x = torch.zeros(1, 1, 4, 4); g = torch.zeros(1, 4, 4, 2)
    with pytest.raises(ValueError, match="padding_mode"):
        grid_sample_backward_mps(torch.zeros(1, 1, 4, 4), x, g, "reflection")
