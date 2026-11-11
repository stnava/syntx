"""Exact parity, MPS determinism, and SyN half-field projection/cadence unit tests."""

from __future__ import annotations

import ants
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import syntx
from syntx.core.losses.cross_correlation import (
    _box_pool,
    _window_counts,
    box_mean_nd,
    local_ncc_loss_nd,
)


def _ref_avg_pool_nd(x: torch.Tensor, window_size: int, count_include_pad: bool) -> torch.Tensor:
    pool_fn = F.avg_pool2d if x.dim() == 4 else F.avg_pool3d
    return pool_fn(
        x,
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2,
        count_include_pad=count_include_pad,
    )


def _ref_local_ncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None,
    window_size: int,
    squared: bool,
    var_floor: float = 1e-6,
) -> torch.Tensor:
    I_mean = _ref_avg_pool_nd(I, window_size, count_include_pad=False)
    J_mean = _ref_avg_pool_nd(J, window_size, count_include_pad=False)
    F_c = I - I_mean
    M_c = J - J_mean
    I_var = torch.clamp(_ref_avg_pool_nd(F_c * F_c, window_size, count_include_pad=False), min=0.0)
    J_var = torch.clamp(_ref_avg_pool_nd(M_c * M_c, window_size, count_include_pad=False), min=0.0)
    IJ_cov = _ref_avg_pool_nd(F_c * M_c, window_size, count_include_pad=False)
    safe_I = torch.clamp(I_var, min=var_floor)
    safe_J = torch.clamp(J_var, min=var_floor)
    if squared:
        cc = torch.clamp((IJ_cov**2) / (safe_I * safe_J + 1e-8), min=0.0, max=1.0)
    else:
        cc = torch.clamp(IJ_cov / (torch.sqrt(safe_I * safe_J) + 1e-6), min=-1.0, max=1.0)
    if mask is not None:
        active = ((I_var > var_floor) & (J_var > var_floor) & (mask > 0.5)).to(I.dtype)
        return -torch.sum(cc * active) / (torch.sum(active) + 1e-8)
    return -torch.mean(cc)


@pytest.mark.parametrize("spatial", [(11, 14), (7, 9, 12), (3, 4, 5)])
@pytest.mark.parametrize("window_size", [1, 3, 5, 7])
@pytest.mark.parametrize("count_include_pad", [False, True])
def test_separable_box_pool_and_counts_exact_float64_parity(
    spatial: tuple[int, ...],
    window_size: int,
    count_include_pad: bool,
) -> None:
    torch.manual_seed(101)
    x = torch.randn((2, 3, *spatial), dtype=torch.float64)
    eff_k = min(window_size, min(spatial))
    if eff_k % 2 == 0:
        eff_k = max(1, eff_k - 1)
    y_sep = _box_pool(x, eff_k, count_include_pad=count_include_pad)
    y_ref = _ref_avg_pool_nd(x, eff_k, count_include_pad=count_include_pad)
    assert torch.allclose(y_sep, y_ref, atol=1e-13, rtol=1e-12)

    if not count_include_pad:
        x_req = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        g_up = torch.randn_like(x)
        y1 = box_mean_nd(x_req, eff_k)
        y2 = _ref_avg_pool_nd(x_ref, eff_k, count_include_pad=False)
        (g_sep,) = torch.autograd.grad((y1 * g_up).sum(), x_req)
        (g_ref,) = torch.autograd.grad((y2 * g_up).sum(), x_ref)
        assert torch.allclose(g_sep, g_ref, atol=1e-13, rtol=1e-12)

        cnt_sep = _window_counts(x, eff_k)
        ones = torch.ones_like(x[:, :1])
        cnt_ref = _ref_avg_pool_nd(ones, eff_k, count_include_pad=True) * (eff_k ** len(spatial))
        assert torch.allclose(cnt_sep, cnt_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("spatial", [(15, 13), (11, 10, 9)])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_local_ncc_loss_exact_parity_vs_reference_3d_pool(
    spatial: tuple[int, ...],
    squared: bool,
    with_mask: bool,
) -> None:
    torch.manual_seed(202)
    I = torch.rand((1, 2, *spatial), dtype=torch.float64, requires_grad=True)
    J = torch.rand((1, 2, *spatial), dtype=torch.float64, requires_grad=True)
    mask = (torch.rand((1, 1, *spatial), dtype=torch.float64) > 0.25).to(torch.float64) if with_mask else None

    loss_new = local_ncc_loss_nd(I, J, mask=mask, window_size=5, squared=squared)
    gi_new, gj_new = torch.autograd.grad(loss_new, (I, J))

    I_r = I.detach().clone().requires_grad_(True)
    J_r = J.detach().clone().requires_grad_(True)
    loss_ref = _ref_local_ncc_loss_nd(I_r, J_r, mask=mask, window_size=5, squared=squared)
    gi_ref, gj_ref = torch.autograd.grad(loss_ref, (I_r, J_r))

    assert torch.allclose(loss_new, loss_ref, atol=1e-13, rtol=1e-12)
    assert torch.allclose(gi_new, gi_ref, atol=1e-13, rtol=1e-12)
    assert torch.allclose(gj_new, gj_ref, atol=1e-13, rtol=1e-12)


def test_single_pass_5ch_gradcheck_and_translation_invariance() -> None:
    torch.manual_seed(303)
    I = torch.rand((1, 1, 6, 7, 5), dtype=torch.float64, requires_grad=True)
    J = torch.rand((1, 1, 6, 7, 5), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: local_ncc_loss_nd(a, b, window_size=3, squared=True, single_pass=True),
        (I, J),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    l0 = local_ncc_loss_nd(I, J, window_size=3, squared=True, single_pass=True)
    l_shift = local_ncc_loss_nd(I + 500.0, J - 250.0, window_size=3, squared=True, single_pass=True)
    assert torch.allclose(l0, l_shift, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS required for accelerator determinism check",
)
def test_mps_lncc_bitwise_determinism_and_cpu_parity() -> None:
    torch.manual_seed(404)
    shape = (1, 1, 48, 64, 64)
    I_cpu = torch.rand(shape, dtype=torch.float32, requires_grad=True)
    J_cpu = torch.rand(shape, dtype=torch.float32, requires_grad=True)
    l_cpu = local_ncc_loss_nd(I_cpu, J_cpu, window_size=5, squared=True)
    gi_cpu, gj_cpu = torch.autograd.grad(l_cpu, (I_cpu, J_cpu))

    I_m1 = I_cpu.detach().to("mps").requires_grad_(True)
    J_m1 = J_cpu.detach().to("mps").requires_grad_(True)
    l_m1 = local_ncc_loss_nd(I_m1, J_m1, window_size=5, squared=True)
    gi_m1, gj_m1 = torch.autograd.grad(l_m1, (I_m1, J_m1))

    I_m2 = I_cpu.detach().to("mps").requires_grad_(True)
    J_m2 = J_cpu.detach().to("mps").requires_grad_(True)
    l_m2 = local_ncc_loss_nd(I_m2, J_m2, window_size=5, squared=True)
    gi_m2, gj_m2 = torch.autograd.grad(l_m2, (I_m2, J_m2))

    assert torch.equal(l_m1, l_m2)
    assert torch.equal(gi_m1, gi_m2)
    assert torch.equal(gj_m1, gj_m2)
    assert torch.allclose(l_m1.cpu(), l_cpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gi_m1.cpu(), gi_cpu, atol=2e-6, rtol=1e-5)


def test_syn_cfl_projection_and_cadence_polish_wiring() -> None:
    rng = np.random.default_rng(12)
    zz, yy, xx = np.mgrid[-1:1:24j, -1:1:24j, -1:1:24j]
    fi_np = np.exp(-((xx**2 + yy**2 + zz**2) / 0.35)).astype(np.float32) + 0.02 * rng.standard_normal(
        (24, 24, 24), dtype=np.float32
    )
    mi_np = np.exp(-(((xx - 0.12) ** 2 + (yy + 0.08) ** 2 + zz**2) / 0.35)).astype(np.float32)
    fi = ants.from_numpy(np.clip(fi_np, 0.0, 1.0), spacing=(1.0, 1.0, 1.0))
    mi = ants.from_numpy(np.clip(mi_np, 0.0, 1.0), spacing=(1.0, 1.0, 1.0))

    reg_base = syntx.syn(
        fi,
        mi,
        type_of_transform="SyNOnly",
        reg_iterations=[4, 3],
        device="cpu",
        return_warped=False,
    )
    reg_proj = syntx.syn(
        fi,
        mi,
        type_of_transform="SyNOnly",
        reg_iterations=[4, 3],
        device="cpu",
        return_warped=False,
        cfl_project_inverse=True,
        projection_frequency=1,
    )
    reg_cad = syntx.syn(
        fi,
        mi,
        type_of_transform="SyNOnly",
        reg_iterations=[4, 4],
        device="cpu",
        return_warped=False,
        inv_refresh_every=(1, 2),
        polish_inv_steps=15,
    )

    reg_comb = syntx.syn(
        fi,
        mi,
        type_of_transform="SyNOnly",
        reg_iterations=[4, 4],
        device="cpu",
        return_warped=False,
        lncc_single_pass=True,
        inv_refresh_every=(1, 2),
        polish_inv_steps=15,
        polish_project_inverse=True,
    )

    w_base = reg_base["fwd_warp_image"].numpy()
    w_proj = reg_proj["fwd_warp_image"].numpy()
    w_comb = reg_comb["fwd_warp_image"].numpy()
    assert not np.allclose(w_base, w_proj, atol=1e-7)
    assert np.isfinite(w_comb).all()
    ice = reg_cad["inverse_identity_errors"]
    mean_err = ice.get("mean_error_mm", ice.get("mean_error", 0.0))
    assert mean_err < 0.05
    ice_comb = reg_comb["inverse_identity_errors"]
    mean_err_comb = ice_comb.get("mean_error_mm", ice_comb.get("mean_error", 0.0))
    assert mean_err_comb < 0.05
