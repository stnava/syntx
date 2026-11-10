"""Focused regression tests for microCT-scale SyN memory housekeeping, slab-chunked LNCC,
memory-minimal fixed-point/Anderson field inversion, MPS 5-D reductions, and slab-chunked
TransformChain resampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ants
import syntx
from syntx.core.inverse import (
    _batch_reduce_amax,
    _batch_reduce_mean,
    _batch_reduce_sum,
    calculate_inverse_identity_error,
    update_inverse_field_nd,
)
from syntx.core.losses.cross_correlation import local_ncc_loss_nd
from syntx.core.losses.registry import get_similarity_loss
from syntx.transforms import TransformChain, _reference_grid_points_lps


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("use_pseudo", [False, True])
@pytest.mark.parametrize("metric_chunks", [2, 4, 7])
def test_chunked_lncc_exact_parity_loss_and_grad(
    ndim: int,
    squared: bool,
    with_mask: bool,
    use_pseudo: bool,
    metric_chunks: int,
) -> None:
    torch.manual_seed(17)
    spatial = (23, 19) if ndim == 2 else (19, 15, 13)
    img_i = torch.rand((1, 1, *spatial), dtype=torch.float64, requires_grad=True)
    img_j = torch.rand((1, 1, *spatial), dtype=torch.float64, requires_grad=True)
    mask = (torch.rand((1, 1, *spatial), dtype=torch.float64) > 0.2).to(torch.float64) if with_mask else None

    loss_ref = local_ncc_loss_nd(
        img_i,
        img_j,
        window_size=5,
        mask=mask,
        use_ants_pseudo_gradient=use_pseudo,
        squared=squared,
        metric_chunks=1,
    )
    gi_ref, gj_ref = torch.autograd.grad(loss_ref, (img_i, img_j))

    img_i_c = img_i.detach().clone().requires_grad_(True)
    img_j_c = img_j.detach().clone().requires_grad_(True)
    loss_chunk = local_ncc_loss_nd(
        img_i_c,
        img_j_c,
        window_size=5,
        mask=mask,
        use_ants_pseudo_gradient=use_pseudo,
        squared=squared,
        metric_chunks=metric_chunks,
    )
    gi_chunk, gj_chunk = torch.autograd.grad(loss_chunk, (img_i_c, img_j_c))

    assert torch.allclose(loss_chunk, loss_ref, atol=1e-13, rtol=1e-12)
    assert torch.allclose(gi_chunk, gi_ref, atol=1e-13, rtol=1e-12)
    assert torch.allclose(gj_chunk, gj_ref, atol=1e-13, rtol=1e-12)

    with torch.no_grad():
        loss_ng = local_ncc_loss_nd(
            img_i.detach(),
            img_j.detach(),
            window_size=5,
            mask=mask,
            use_ants_pseudo_gradient=use_pseudo,
            squared=squared,
            metric_chunks=metric_chunks,
        )
    assert torch.allclose(loss_ng, loss_ref.detach(), atol=1e-13, rtol=1e-12)


def test_chunked_lncc_gradcheck_and_invalid_chunks() -> None:
    torch.manual_seed(29)
    img_i = torch.rand((1, 1, 11, 9, 8), dtype=torch.float64, requires_grad=True)
    img_j = torch.rand((1, 1, 11, 9, 8), dtype=torch.float64, requires_grad=True)
    mask = (torch.rand((1, 1, 11, 9, 8), dtype=torch.float64) > 0.15).to(torch.float64)

    assert torch.autograd.gradcheck(
        lambda a, b: local_ncc_loss_nd(a, b, window_size=3, mask=mask, squared=True, metric_chunks=3),
        (img_i, img_j),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    with pytest.raises(ValueError, match="metric_chunks"):
        local_ncc_loss_nd(img_i, img_j, window_size=3, metric_chunks=0)

    loss_mod = get_similarity_loss("cc2", window_size=3, metric_chunks=3)
    val_c = loss_mod(img_i, img_j, mask=mask)
    val_1 = get_similarity_loss("cc2", window_size=3, metric_chunks=1)(img_i, img_j, mask=mask)
    assert torch.allclose(val_c, val_1, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("method", ["fixed_point", "anderson"])
def test_inverse_field_batch_independence_and_warmstart_stats(method: str) -> None:
    torch.manual_seed(41)
    spatial = (18, 20, 16)
    spacing = (0.5, 0.6, 0.7)
    origin = (-4.0, -5.0, -3.0)
    direction = np.eye(3)

    # Smooth diffeomorphic displacement pair
    raw = torch.randn((2, 3, *spatial), dtype=torch.float32) * 0.25
    smooth = torch.nn.functional.avg_pool3d(
        torch.nn.functional.pad(raw, (2, 2, 2, 2, 2, 2), mode="replicate"),
        kernel_size=5,
        stride=1,
    )
    fwd_pair = smooth.movedim(1, -1).contiguous()
    # Zero boundary rim
    fwd_pair[:, 0] = 0.0
    fwd_pair[:, -1] = 0.0
    fwd_pair[:, :, 0] = 0.0
    fwd_pair[:, :, -1] = 0.0
    fwd_pair[:, :, :, 0] = 0.0
    fwd_pair[:, :, :, -1] = 0.0

    stats_pair: dict = {}
    inv_pair = update_inverse_field_nd(
        fwd_pair,
        None,
        steps=12,
        method=method,
        spacing=spacing,
        origin=origin,
        direction=direction,
        max_error_threshold=1e-3,
        mean_error_threshold=1e-4,
        stats=stats_pair,
    )
    assert "steps" in stats_pair
    assert stats_pair["steps"] >= 1

    inv_0 = update_inverse_field_nd(
        fwd_pair[0:1],
        None,
        steps=12,
        method=method,
        spacing=spacing,
        origin=origin,
        direction=direction,
        max_error_threshold=1e-3,
        mean_error_threshold=1e-4,
    )
    inv_1 = update_inverse_field_nd(
        fwd_pair[1:2],
        None,
        steps=12,
        method=method,
        spacing=spacing,
        origin=origin,
        direction=direction,
        max_error_threshold=1e-3,
        mean_error_threshold=1e-4,
    )
    # When batched, stopping waits until BOTH fields converge; individual errors must both be below tolerance.
    ic_0 = calculate_inverse_identity_error(fwd_pair[0:1], inv_pair[0:1], spacing, origin, direction)
    ic_1 = calculate_inverse_identity_error(fwd_pair[1:2], inv_pair[1:2], spacing, origin, direction)
    assert ic_0["mean_error"] < 1e-3
    assert ic_1["mean_error"] < 1e-3
    assert torch.allclose(inv_pair[0:1], inv_0, atol=2e-3)
    assert torch.allclose(inv_pair[1:2], inv_1, atol=2e-3)

    # Warm-started with converged inverse -> exits on step 1 without allocating history
    warm_stats: dict = {}
    inv_warm = update_inverse_field_nd(
        fwd_pair,
        inv_pair.clone(),
        steps=10,
        method=method,
        spacing=spacing,
        origin=origin,
        direction=direction,
        max_error_threshold=1e-2,
        mean_error_threshold=1e-3,
        stats=warm_stats,
    )
    assert warm_stats["steps"] == 1
    assert warm_stats["converged"] is True
    assert torch.allclose(inv_warm, inv_pair, atol=1e-3)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS device required for 5-D Metal reduction parity test",
)
def test_mps_5d_batch_reductions_and_anderson_parity() -> None:
    torch.manual_seed(73)
    x_cpu = torch.rand((2, 48, 52, 44, 3), dtype=torch.float32)
    x_cpu[0, 21, 17, 9, 2] = 9.75
    x_cpu[1, 3, 40, 31, 0] = 14.25
    x_mps = x_cpu.to("mps")

    assert torch.allclose(_batch_reduce_amax(x_mps).cpu(), _batch_reduce_amax(x_cpu), atol=1e-6)
    assert torch.allclose(_batch_reduce_mean(x_mps).cpu(), _batch_reduce_mean(x_cpu), rtol=1e-5, atol=1e-6)
    assert torch.allclose(_batch_reduce_sum(x_mps).cpu(), _batch_reduce_sum(x_cpu), rtol=1e-5, atol=1e-3)

    spatial = (32, 36, 28)
    spacing = (0.4, 0.4, 0.4)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    raw = torch.randn((2, 3, *spatial), dtype=torch.float32) * 0.15
    fwd_cpu = torch.nn.functional.avg_pool3d(
        torch.nn.functional.pad(raw, (2, 2, 2, 2, 2, 2), mode="replicate"),
        kernel_size=5,
        stride=1,
    ).movedim(1, -1).contiguous()
    fwd_mps = fwd_cpu.to("mps")

    inv_cpu = update_inverse_field_nd(
        fwd_cpu, None, steps=8, method="anderson", spacing=spacing, origin=origin, direction=direction
    )
    inv_mps = update_inverse_field_nd(
        fwd_mps, None, steps=8, method="anderson", spacing=spacing, origin=origin, direction=direction
    )
    assert torch.allclose(inv_mps.cpu(), inv_cpu, atol=1e-4, rtol=1e-4)


def test_syn_metric_chunks_and_tied_inverse_storage_parity() -> None:
    rng = np.random.default_rng(101)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        np.linspace(-1.0, 1.0, 30, dtype=np.float32),
        np.linspace(-1.0, 1.0, 26, dtype=np.float32),
        indexing="ij",
    )
    r2_f = xx**2 + yy**2 + zz**2
    r2_m = (xx - 0.08) ** 2 + (yy + 0.06) ** 2 + zz**2
    arr_f = np.exp(-3.0 * r2_f).astype(np.float32) + 0.01 * rng.standard_normal(xx.shape, dtype=np.float32)
    arr_m = np.exp(-3.0 * r2_m).astype(np.float32) + 0.01 * rng.standard_normal(xx.shape, dtype=np.float32)

    fixed = ants.from_numpy(arr_f, origin=(-5.0, -6.0, -4.0), spacing=(0.4, 0.4, 0.4))
    moving = ants.from_numpy(arr_m, origin=(-5.0, -6.0, -4.0), spacing=(0.4, 0.4, 0.4))

    reg_ref = syntx.syn(
        fixed,
        moving,
        type_of_transform="SyNOnly",
        syn_metric="cc2",
        syn_sampling=2,
        reg_iterations=[4, 3],
        levels=[2, 1],
        inverse_method="fixed_point",
        metric_chunks=1,
        device="cpu",
        seed=7,
    )
    reg_chunk = syntx.syn(
        fixed,
        moving,
        type_of_transform="SyNOnly",
        syn_metric="cc2",
        syn_sampling=2,
        reg_iterations=[4, 3],
        levels=[2, 1],
        inverse_method="fixed_point",
        metric_chunks=3,
        device="cpu",
        seed=7,
    )

    model = reg_chunk["model"]
    assert model.warp_l2r_inv.data.data_ptr() == model.warp_r2l.data.data_ptr()
    assert model.warp_r2l_inv.data.data_ptr() == model.warp_l2r.data.data_ptr()

    np.testing.assert_allclose(
        reg_chunk["fwd_warp_image"].numpy(),
        reg_ref["fwd_warp_image"].numpy(),
        atol=1e-5,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        reg_chunk["warpedmovout"].numpy(),
        reg_ref["warpedmovout"].numpy(),
        atol=1e-5,
        rtol=1e-4,
    )

    jac = syntx.liouville_determinant(reg_chunk, fixed)
    assert jac.shape == fixed.shape
    assert float(jac.numpy().min()) > 0.0
    assert reg_chunk["inverse_identity_errors"]["phi_1"]["mean_error"] < 0.05


def test_transform_chain_slab_resampling_parity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import syntx.transforms as tx_mod

    rng = np.random.default_rng(131)
    arr = rng.random((20, 18, 16), dtype=np.float32)
    disp = (rng.standard_normal((20, 18, 16, 3)) * 0.15).astype(np.float32)
    img = ants.from_numpy(arr, origin=(1.0, -2.0, 0.5), spacing=(0.5, 0.6, 0.7))
    disp_img = ants.from_numpy(disp, origin=(1.0, -2.0, 0.5), spacing=(0.5, 0.6, 0.7), has_components=True)
    warp_path = str(tmp_path / "warp.nii.gz")
    ants.image_write(disp_img, warp_path)

    # Verify x_slice grid construction matches full grid slice-for-slice
    full_pts = _reference_grid_points_lps(img, "cpu").reshape(20, 18, 16, 3)
    slab_pts = _reference_grid_points_lps(img, "cpu", x_slice=(5, 13)).reshape(8, 18, 16, 3)
    assert torch.allclose(slab_pts, full_pts[5:13], atol=1e-6)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        return

    chain = TransformChain.from_paths([warp_path])
    out_single = chain.apply(img, img, interpolator="linear", device=dev)

    monkeypatch.setattr(tx_mod, "_MAX_GPU_CACHE_VOXELS", 1024)
    tx_mod._DISP_FIELD_CACHE.clear()
    tx_mod._MAPPED_GRID_CACHE.clear()
    out_slab = chain.apply(img, img, interpolator="linear", device=dev)
    assert len(tx_mod._DISP_FIELD_CACHE) == 0
    assert len(tx_mod._MAPPED_GRID_CACHE) == 0
    np.testing.assert_allclose(out_slab.numpy(), out_single.numpy(), atol=1e-5, rtol=1e-5)
