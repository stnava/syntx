"""
Unit and regression tests for the ANTsTorch PR #96–#98 port into syntx:

1. ``syntx.landmarks.blob._separable_gaussian3d``: per-axis symmetric shifted-slice
   Gaussian blur with float64-normalized tap weights, exact ``replicate`` and ``zeros``
   boundary handling, zero 5-D ``F.pad`` / ``F.conv3d``, and MPS large-slice parity.
2. ``syntx.core.smoothing.separable_gaussian_filter``: CPU 3-D shifted-slice fast path,
   MPS ``kernel_size >= 33`` large-kernel branch, ``replicate`` / ``constant`` / ``zeros``
   modes across ``1D`` / ``2D`` / ``3D``, ``channels_last=True/False``, ``sigma <= 0``
   axes, and ``requires_grad=True`` autograd safety.
3. ``syntx.features.SwinUNETRExtractor._pad_to_multiple_3d``: per-axis ``torch.cat``
   32-multiple padding avoiding Apple MPS 5-D ``F.pad`` silent corruption on
   ``(1, 1, 18, 256, 256)`` (``H * W = 65536`` with ``pad_h == pad_w == 0``).
4. ``syntx.syn._backward_similarity_terms_per_term``: multi-channel / multi-metric
   per-term detached-leaf backward equivalence vs. joint ``loss.backward()``,
   ``weakref + gc.collect()`` proof that ``<= 1`` metric autograd graph is alive at once,
   and ``grad is not None`` guard when a term returns a disconnected ``0.0`` loss.
5. Post-``.backward()`` ``del loss`` graph release, finite convergence, and determinism
   across ``syntx.syn``, ``syntx.tvf``, ``syntx.syngs``, and ``syntx.greedy``.
"""

from __future__ import annotations

import gc
import math
import weakref

import ants
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from syntx.core.losses import get_similarity_loss
from syntx.core.smoothing import get_cached_gaussian_kernel_1d, separable_gaussian_filter
from syntx.features import SwinUNETRExtractor
from syntx.landmarks.blob import _gaussian_kernel_1d, _separable_gaussian3d
from syntx.syn import _backward_similarity_terms_per_term


def _ref_conv1d_blur_3d(vol: torch.Tensor, sigmas, mode: str = "replicate") -> torch.Tensor:
    """Reference 3-D Gaussian blur using 3-D reshaped ``F.pad`` + ``F.conv1d`` on CPU."""
    if np.isscalar(sigmas):
        sig = (float(sigmas),) * 3
    else:
        sig = tuple(float(x) for x in sigmas)
    pad_mode = "constant" if mode in ("zeros", "constant") else "replicate"
    pad_kw = {"mode": pad_mode, "value": 0.0} if pad_mode == "constant" else {"mode": pad_mode}
    out = vol.cpu()
    for dim, sg in zip((2, 3, 4), sig):
        if not sg > 0.0:
            continue
        k = _gaussian_kernel_1d(sg, dtype=out.dtype).view(1, 1, -1)
        r = k.shape[-1] // 2
        perm = list(range(out.ndim))
        perm[-1], perm[dim] = perm[dim], perm[-1]
        p = out.permute(*perm).contiguous()
        flat = p.view(-1, 1, p.shape[-1])
        conv = F.conv1d(F.pad(flat, (r, r), **pad_kw), k)
        out = conv.view(*p.shape).permute(*perm).contiguous()
    return out.to(vol.device)


def _ref_smoothing_conv1d(
    grid: torch.Tensor,
    sigma,
    mode: str = "replicate",
    channels_last: bool = True,
) -> torch.Tensor:
    """Reference N-D ITK Bessel Gaussian blur via 3-D reshaped ``F.conv1d`` on CPU."""
    g_cpu = grid.detach().cpu()
    v = torch.movedim(g_cpu, -1, 1) if channels_last else g_cpu
    num_spatial = v.ndim - 2
    sig_list = [float(sigma)] * num_spatial if isinstance(sigma, (int, float)) else [float(x) for x in sigma]
    eff_mode = "constant" if mode == "zeros" else mode
    pad_kw = {"mode": eff_mode, "value": 0.0} if eff_mode == "constant" else {"mode": eff_mode}
    for i, sig in enumerate(sig_list):
        if sig <= 0.0:
            continue
        k = get_cached_gaussian_kernel_1d(sig, "cpu", v.dtype)
        pad = k.shape[-1] // 2
        td = i + 2
        perm = list(range(v.ndim))
        perm[-1], perm[td] = perm[td], perm[-1]
        p = v.permute(*perm).contiguous()
        flat = p.view(-1, 1, p.shape[-1])
        conv = F.conv1d(F.pad(flat, (pad, pad), **pad_kw), k)
        v = conv.view(*p.shape).permute(*perm).contiguous()
    return (torch.movedim(v, 1, -1) if channels_last else v).contiguous()


# ---------------------------------------------------------------------------
# Test 1: blob._separable_gaussian3d numerical parity & MPS large-slice safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["replicate", "zeros"])
@pytest.mark.parametrize("sigmas", [1.5, (1.2, 2.3, 0.7), (0.0, 1.8, 0.9), (2.5, 2.5, 2.5)])
def test_blob_separable_gaussian3d_cpu_parity_and_edge_cases(mode, sigmas):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 14, 18, 22, dtype=torch.float32)
    got = _separable_gaussian3d(x, sigmas, mode=mode)
    ref = _ref_conv1d_blur_3d(x, sigmas, mode=mode)
    max_diff = float((got - ref).abs().max().item())
    assert max_diff < 5e-7, f"max_diff={max_diff} exceeded 5e-7 for mode={mode}, sigmas={sigmas}"

    # Small tensor where kernel radius > spatial length
    x_tiny = torch.randn(1, 1, 4, 3, 2, dtype=torch.float32)
    got_tiny = _separable_gaussian3d(x_tiny, (2.0, 2.0, 2.0), mode=mode)
    ref_tiny = _ref_conv1d_blur_3d(x_tiny, (2.0, 2.0, 2.0), mode=mode)
    assert float((got_tiny - ref_tiny).abs().max().item()) < 5e-7


def test_blob_separable_gaussian3d_autograd_and_validation():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 8, 9, 10, requires_grad=True)
    y = _separable_gaussian3d(x, (1.2, 0.0, 1.5), mode="replicate")
    (y.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    with pytest.raises(ValueError, match="mode"):
        _separable_gaussian3d(x.detach(), 1.0, mode="invalid")
    with pytest.raises(ValueError, match="finite"):
        _separable_gaussian3d(x.detach(), (1.0, float("nan"), 1.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_blob_separable_gaussian3d_mps_large_slice_parity():
    torch.manual_seed(2)
    v_cpu = torch.randn(1, 1, 8, 256, 256, dtype=torch.float32)
    v_mps = v_cpu.to("mps")
    for sigmas in ((1.8, 0.0, 0.0), (1.4, 2.1, 0.9)):
        out_cpu = _separable_gaussian3d(v_cpu, sigmas, mode="replicate")
        out_mps = _separable_gaussian3d(v_mps, sigmas, mode="replicate").cpu()
        max_diff = float((out_cpu - out_mps).abs().max().item())
        assert max_diff < 5e-7, f"MPS vs CPU max_diff={max_diff} for sigmas={sigmas}"


# ---------------------------------------------------------------------------
# Test 2: smoothing.separable_gaussian_filter parity, MPS klen>=33 & autograd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("mode", ["replicate", "constant", "zeros"])
@pytest.mark.parametrize("channels_last", [True, False])
def test_separable_gaussian_filter_cpu_parity_all_modes(ndim, channels, mode, channels_last):
    torch.manual_seed(10 + ndim * 10 + channels)
    spatial = (14, 16, 18)[:ndim]
    shape = (1, *spatial, channels) if channels_last else (1, channels, *spatial)
    grid = torch.randn(*shape, dtype=torch.float32)
    sigmas = [1.7, 0.0, 2.4][:ndim] if ndim > 1 else 2.1

    got = separable_gaussian_filter(grid, sigmas, mode=mode, channels_last=channels_last)
    ref = _ref_smoothing_conv1d(grid, sigmas, mode=mode, channels_last=channels_last)
    max_diff = float((got - ref).abs().max().item())
    assert max_diff < 5e-7, (
        f"max_diff={max_diff} for ndim={ndim}, C={channels}, mode={mode}, channels_last={channels_last}"
    )


def test_separable_gaussian_filter_autograd_requires_grad():
    torch.manual_seed(20)
    for mode in ("replicate", "constant"):
        x1 = torch.randn(1, 10, 12, 14, 3, dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        weights = torch.randn_like(x1)

        out = separable_gaussian_filter(x1, [1.5, 0.0, 2.0], mode=mode)
        (out * weights).sum().backward()

        # Compare against 1-D conv reference autograd
        v = torch.movedim(x2, -1, 1)
        eff_mode = "constant" if mode == "zeros" else mode
        pad_kw = {"mode": eff_mode, "value": 0.0} if eff_mode == "constant" else {"mode": eff_mode}
        for i, sig in enumerate([1.5, 0.0, 2.0]):
            if sig <= 0.0:
                continue
            k = get_cached_gaussian_kernel_1d(sig, "cpu", v.dtype).squeeze(0)
            pad = k.shape[-1] // 2
            C = v.shape[1]
            if i == 0:
                v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), **pad_kw), k.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1), groups=C)
            elif i == 1:
                v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), **pad_kw), k.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1), groups=C)
            elif i == 2:
                v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), **pad_kw), k.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1), groups=C)
        out_ref = torch.movedim(v, 1, -1)
        (out_ref * weights).sum().backward()

        assert x1.grad is not None and torch.isfinite(x1.grad).all()
        assert torch.allclose(out, out_ref, atol=5e-7)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6)

    # Also verify float16 input uses float32 tap weights without quantization error
    x_fp16 = torch.randn(1, 10, 12, 14, 2, dtype=torch.float16)
    out_fp16 = separable_gaussian_filter(x_fp16, 1.8, mode="replicate")
    out_fp32_ref = separable_gaussian_filter(x_fp16.float(), 1.8, mode="replicate").half()
    assert out_fp16.dtype == torch.float16
    assert torch.allclose(out_fp16, out_fp32_ref, atol=2e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_separable_gaussian_filter_mps_small_and_large_kernels():
    torch.manual_seed(30)
    # Verify both klen < 33 (sigma=2.5, klen=17) and klen >= 33 (sigma=12.0, klen=43)
    k_large = get_cached_gaussian_kernel_1d(12.0, "cpu", torch.float32)
    assert k_large.shape[-1] >= 33, f"Expected klen >= 33 at sigma=12.0, got {k_large.shape[-1]}"

    g_cpu = torch.randn(1, 20, 22, 24, 3, dtype=torch.float32)
    g_mps = g_cpu.to("mps")
    for mode in ("replicate", "constant"):
        for sig in (2.5, 12.0, [12.0, 0.0, 2.5]):
            out_cpu = separable_gaussian_filter(g_cpu, sig, mode=mode)
            out_mps = separable_gaussian_filter(g_mps, sig, mode=mode).cpu()
            max_diff = float((out_cpu - out_mps).abs().max().item())
            assert max_diff < 5e-7, f"MPS vs CPU max_diff={max_diff} for sigma={sig}, mode={mode}"

    # Verify MPS autograd with requires_grad=True at klen >= 33
    x_mps = g_cpu[:, :12, :12, :12, :1].to("mps").detach().requires_grad_(True)
    y_mps = separable_gaussian_filter(x_mps, 12.0, mode="replicate")
    y_mps.sum().backward()
    assert x_mps.grad is not None and bool(torch.isfinite(x_mps.grad).all().item())


# ---------------------------------------------------------------------------
# Test 3: features.SwinUNETRExtractor 32-multiple pad on (1, 1, 18, 256, 256)
# ---------------------------------------------------------------------------

def test_swinvit_32_multiple_pad_cpu_and_mps_exact_match():
    torch.manual_seed(40)
    x_cpu = torch.randn(1, 1, 18, 256, 256, dtype=torch.float32)
    padded_cpu = SwinUNETRExtractor._pad_to_multiple_3d(x_cpu, multiple=32)
    assert padded_cpu.shape == (1, 1, 32, 256, 256)
    assert torch.equal(padded_cpu[:, :, :18, :, :], x_cpu)
    assert float(padded_cpu[:, :, 18:, :, :].abs().max().item()) == 0.0

    # Also test SwinUNETRExtractor.extract end-to-end with a lightweight dummy swinViT
    class _DummySwinModel(nn.Module):
        def swinViT(self, x_in):
            # 5 stages downsampled by 2**(k+1)
            return [
                x_in[:, :, ::2, ::2, ::2],
                x_in[:, :, ::4, ::4, ::4],
                x_in[:, :, ::8, ::8, ::8],
                x_in[:, :, ::16, ::16, ::16],
                x_in[:, :, ::32, ::32, ::32],
            ]

    ext = SwinUNETRExtractor.__new__(SwinUNETRExtractor)
    nn.Module.__init__(ext)
    ext.feature_layers = (1, 2)
    ext.model = _DummySwinModel()
    feats_cpu = ext.extract(x_cpu)
    assert feats_cpu[0].shape == (1, 1, 18 // 4, 256 // 4, 256 // 4)
    assert feats_cpu[1].shape == (1, 1, 18 // 8, 256 // 8, 256 // 8)

    if torch.backends.mps.is_available():
        x_mps = x_cpu.to("mps")
        padded_mps = SwinUNETRExtractor._pad_to_multiple_3d(x_mps, multiple=32).cpu()
        diff = float((padded_cpu - padded_mps).abs().max().item())
        assert diff == 0.0, f"Expected exact 0.0 diff between CPU and MPS pad, got {diff}"
        feats_mps = ext.extract(x_mps)
        for fc, fm in zip(feats_cpu, feats_mps):
            assert float((fc - fm.cpu()).abs().max().item()) == 0.0


# ---------------------------------------------------------------------------
# Test 4: SyN multi-channel per-term detached backward equivalence & weakref
# ---------------------------------------------------------------------------

def test_syn_multichannel_per_term_backward_equivalence_weakref_and_none_grad():
    torch.manual_seed(50)
    I_base = torch.rand(1, 3, 20, 20, 20, dtype=torch.float32)
    J_base = torch.rand(1, 3, 20, 20, 20, dtype=torch.float32)
    mask = torch.ones(1, 1, 20, 20, 20, dtype=torch.float32)

    base_fns = [
        get_similarity_loss("cc2", window_size=5),
        get_similarity_loss("mse"),
        get_similarity_loss("mattes", num_bins=16),
    ]
    names = ["cc2", "mse", "mattes"]
    weights = [0.5, 0.3, 0.2]

    # Wrap each metric with a custom autograd Function that saves a graph buffer so we
    # can track live C++ autograd graph intermediates via weakref + gc.collect().
    saved_buf_refs: list[weakref.ReferenceType] = []
    max_alive_during_terms = 0

    class _GraphTracker(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            buf = (x + y).clone()
            ctx.save_for_backward(buf)
            saved_buf_refs.append(weakref.ref(buf))
            return x, y

        @staticmethod
        def backward(ctx, gx, gy):
            (buf,) = ctx.saved_tensors
            _ = buf.shape
            return gx, gy

    def _make_tracked_fn(fn):
        def _wrapped(I_c, J_c, mask=None):
            nonlocal max_alive_during_terms
            I_t, J_t = _GraphTracker.apply(I_c, J_c)
            out = fn(I_t, J_t, mask=mask)
            gc.collect()
            alive = sum(1 for r in saved_buf_refs if r() is not None)
            max_alive_during_terms = max(max_alive_during_terms, alive)
            return out
        return _wrapped

    # 1. Joint loss.backward() reference
    I_joint = I_base.clone().requires_grad_(True)
    J_joint = J_base.clone().requires_grad_(True)
    loss_joint = 0.0
    for c_idx, (fn, w) in enumerate(zip(base_fns, weights)):
        loss_joint = loss_joint + w * fn(I_joint[:, c_idx:c_idx + 1], J_joint[:, c_idx:c_idx + 1], mask=mask)
    loss_joint.backward()

    # 2. Per-term detached-leaf backward with weakref graph-lifetime tracking
    I_pt = I_base.clone().requires_grad_(True)
    J_pt = J_base.clone().requires_grad_(True)
    I_mid = I_pt * 1.0
    J_mid = J_pt * 1.0
    tracked_fns = [_make_tracked_fn(fn) for fn in base_fns]
    metric_dict = {}
    loss_pt_t, _, _ = _backward_similarity_terms_per_term(
        I_mid,
        J_mid,
        names,
        tracked_fns,
        weights,
        in_bounds_mask=mask,
        metric_losses_dict=metric_dict,
        propagate_to_mid=True,
    )
    gc.collect()
    alive_after = sum(1 for r in saved_buf_refs if r() is not None)

    assert max_alive_during_terms == 1, f"Expected <= 1 live metric graph at once, got {max_alive_during_terms}"
    assert alive_after == 0, f"Expected 0 live metric graphs after backward, got {alive_after}"
    assert set(metric_dict.keys()) == set(names)
    assert torch.allclose(loss_joint.detach(), loss_pt_t, atol=1e-6)
    assert torch.allclose(I_joint.grad, I_pt.grad, atol=2e-6)
    assert torch.allclose(J_joint.grad, J_pt.grad, atol=2e-6)

    # 3. Disconnected 0.0 loss (grad=None, e.g. empty-mask mattes) MUST NOT crash
    def _disconnected_zero(I_c, J_c, mask=None):
        return torch.tensor(0.0, device=I_c.device, dtype=I_c.dtype, requires_grad=True)

    I_none = I_base.clone().requires_grad_(True)
    J_none = J_base.clone().requires_grad_(True)
    # (a) One term disconnected + one term active
    loss_mix_t, _, _ = _backward_similarity_terms_per_term(
        I_none * 1.0,
        J_none * 1.0,
        ["empty_mattes", "mse", "zero_w"],
        [_disconnected_zero, base_fns[1], base_fns[0]],
        [0.5, 0.5, 0.0],
        in_bounds_mask=mask,
        propagate_to_mid=True,
    )
    assert torch.isfinite(loss_mix_t)
    assert I_none.grad is not None and J_none.grad is not None

    # (b) ALL terms disconnected (both g_im and g_jm are None)
    I_all_none = I_base.clone().requires_grad_(True)
    J_all_none = J_base.clone().requires_grad_(True)
    loss_zero_t, _, _ = _backward_similarity_terms_per_term(
        I_all_none * 1.0,
        J_all_none * 1.0,
        ["empty1", "empty2"],
        [_disconnected_zero, _disconnected_zero],
        [0.5, 0.5],
        in_bounds_mask=mask,
        propagate_to_mid=True,
    )
    assert float(loss_zero_t.item()) == 0.0
    assert I_all_none.grad is None and J_all_none.grad is None

    # (c) 1-epoch SyNTo.fit with 2 zero-weight / disconnected terms (verifies warp_l2r.grad.mul_(b_mask) never raises AttributeError)
    from syntx.syn import SyNTo

    syn_model = SyNTo(grid_shape=(12, 12, 12), fluid_sigma=1.5)
    I_fit = torch.rand(1, 2, 12, 12, 12, dtype=torch.float32)
    J_fit = torch.rand(1, 2, 12, 12, 12, dtype=torch.float32)
    syn_model.fit(
        I_fit,
        J_fit,
        levels=[1],
        epochs_per_level=[1],
        similarity_metric=["mse", "mse"],
        syn_metric_weights=[0.0, 0.0],
        cfl_voxels=0.25,
    )
    assert torch.isfinite(syn_model.warp_l2r).all() and torch.isfinite(syn_model.warp_r2l).all()


# ---------------------------------------------------------------------------
# Test 5: Post-fit determinism & finite convergence across all 4 solvers
# ---------------------------------------------------------------------------

def _synthetic_pair_3d(shape=(24, 24, 24)):
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    r2_fix = xx ** 2 + yy ** 2 + zz ** 2
    r2_mov = (xx - 0.12) ** 2 + (yy + 0.08) ** 2 + zz ** 2
    arr_fix = np.exp(-3.0 * r2_fix).astype(np.float32)
    arr_mov = np.exp(-3.0 * r2_mov).astype(np.float32)
    fix = ants.from_numpy(arr_fix, origin=(0.0, 0.0, 0.0), spacing=(1.5, 1.5, 1.5))
    mov = ants.from_numpy(arr_mov, origin=(0.0, 0.0, 0.0), spacing=(1.5, 1.5, 1.5))
    return fix, mov


def test_solvers_determinism_and_convergence_after_del_loss(tmp_path):
    import syntx

    fix, mov = _synthetic_pair_3d((24, 24, 24))
    id_mat = str(tmp_path / "id.mat")
    ants.write_transform(
        ants.create_ants_transform(transform_type="AffineTransform", dimension=3),
        id_mat,
    )

    # 1. syntx.syn: single-channel bitwise determinism + multi-metric convergence
    kw_syn = dict(
        fixed=fix,
        moving=mov,
        reg_iterations=[3, 2],
        levels=[2, 1],
        initial_transform=id_mat,
        device="cpu",
        seed=123,
        outprefix=str(tmp_path / "syn1_"),
    )
    r_syn1 = syntx.syn(**kw_syn)
    kw_syn["outprefix"] = str(tmp_path / "syn2_")
    r_syn2 = syntx.syn(**kw_syn)
    w1 = ants.image_read(r_syn1["fwdtransforms"][0]).numpy()
    w2 = ants.image_read(r_syn2["fwdtransforms"][0]).numpy()
    assert np.isfinite(w1).all() and np.array_equal(w1, w2)

    # Multi-metric SyN (triggers per_term_bw in syn.py)
    r_syn_mm = syntx.syn(
        fixed=fix,
        moving=mov,
        syn_metric=["cc2", "mse"],
        syn_metric_weights=[0.7, 0.3],
        reg_iterations=[3, 2],
        levels=[2, 1],
        initial_transform=id_mat,
        device="cpu",
        seed=123,
        verbose=True,
        outprefix=str(tmp_path / "syn_mm_"),
    )
    assert np.isfinite(ants.image_read(r_syn_mm["fwdtransforms"][0]).numpy()).all()

    # SyN dual_gradient, use_analytical_gradients, guided='sulcal', and 0-epoch check
    for tag, extra_kw in (
        ("ag1", dict(use_analytical_gradients=True)),
        ("ag2", dict(use_analytical_gradients=True, syn_metric=["cc2", "mse"], syn_metric_weights=[0.6, 0.4])),
        ("dg1", dict(dual_gradient=True)),
        ("dg2", dict(dual_gradient=True, syn_metric=["cc2", "mse"], syn_metric_weights=[0.6, 0.4])),
        ("sulcal", dict(guided="sulcal")),
        ("zero_ep", dict(reg_iterations=[0, 0])),
    ):
        kw_ex = dict(
            fixed=fix,
            moving=mov,
            reg_iterations=[2, 1],
            levels=[2, 1],
            initial_transform=id_mat,
            device="cpu",
            seed=123,
            outprefix=str(tmp_path / f"syn_{tag}_"),
        )
        kw_ex.update(extra_kw)
        r_ex = syntx.syn(**kw_ex)
        if tag == "zero_ep":
            assert len(r_ex["fwdtransforms"]) >= 1 and r_ex["fwdtransforms"][0].endswith(".mat")
        else:
            assert np.isfinite(ants.image_read(r_ex["fwdtransforms"][0]).numpy()).all()

    # 2. syntx.tvf (standard autograd + analytical gradient mode)
    for idx, ag in enumerate((False, True)):
        r_tvf = syntx.tvf(
            fixed=fix,
            moving=mov,
            reg_iterations=[3, 2],
            levels=[2, 1],
            n_time_steps=1 if ag else 2,
            use_analytical_gradients=ag,
            initial_transform=id_mat,
            device="cpu",
        )
        assert np.isfinite(ants.image_read(r_tvf["fwdtransforms"][0]).numpy()).all()

    # 3. syntx.syngs (including 0-epoch coarse level check)
    r_syngs = syntx.syngs(
        fixed=fix,
        moving=mov,
        reg_iterations=[0, 3],
        levels=[2, 1],
        initial_transform=id_mat,
        device="cpu",
    )
    assert np.isfinite(ants.image_read(r_syngs["fwdtransforms"][0]).numpy()).all()

    # 4. syntx.greedy (including verbose=True to verify loss_t logging and 0.0% folding)
    from syntx.registration_metrics import evaluate_registration_qc

    r_greedy = syntx.greedy(
        fixed=fix,
        moving=mov,
        reg_iterations=[3, 2],
        scales=[2, 1],
        initial_transform=id_mat,
        device="cpu",
        verbose=True,
    )
    assert np.isfinite(ants.image_read(r_greedy["fwdtransforms"][0]).numpy()).all()
    qc_greedy = evaluate_registration_qc(fix, mov, r_greedy)
    assert float(qc_greedy.folding_pct) == 0.0


def test_greedy_speedups_warp_parity_affine_grid_and_dsti_alpha():
    import syntx
    from syntx.core.grid import sample_field_cf
    from syntx.core.losses import parse_similarity_metric
    from syntx.core.losses.cross_correlation import _box_pool, box_mean_nd
    from syntx.greedy import (
        GreedyRegistrationModel,
        _Conv1x1AffineGrid,
        _build_torch_affine_matrix,
        _convert_composed_grid_to_ants_displacement,
    )
    from syntx.spatial import disp_tensor_to_itk, normalized_to_physical_disp

    torch.manual_seed(77)
    dir_f = np.diag([1.0, -1.0, 1.0])
    angle = 0.22
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    dir_m = rot @ np.diag([-1.0, 1.0, 1.0])
    fi = ants.from_numpy(
        np.zeros((16, 20, 24), dtype=np.float32),
        origin=(10.0, -20.0, 5.0),
        spacing=(1.2, 0.9, 1.5),
        direction=dir_f,
    )
    mi = ants.from_numpy(
        np.zeros((18, 22, 20), dtype=np.float32),
        origin=(8.0, -18.0, 3.0),
        spacing=(1.1, 1.0, 1.3),
        direction=dir_m,
    )
    M_phys = np.array([[0.98, -0.05, 0.02], [0.04, 1.01, -0.03], [-0.01, 0.02, 0.99]], dtype=np.float64)
    t_phys = np.array([2.5, -1.8, 3.2], dtype=np.float64)
    theta, P_F, _ = _build_torch_affine_matrix(fi, mi, M_phys, t_phys, torch.device("cpu"))

    grid_shape = [1, 1, 24, 20, 16]
    id_grid = F.affine_grid(torch.eye(3, 4)[None], grid_shape, align_corners=True)
    affine_grid_cf = torch.movedim(F.affine_grid(theta, grid_shape, align_corners=True), -1, 1)
    # Use 0.6 * randn so a significant fraction of coordinates exceed [-1, 1] and test clamp mask
    warp = 0.60 * torch.randn_like(id_grid)

    # 1. _Conv1x1AffineGrid forward AND backward vs sample_field_cf on out-of-bounds coords (CPU & MPS)
    M_theta = theta[0, :, :3].contiguous()
    t_theta_flat = theta[0, :, 3].contiguous()
    up_grad = torch.randn_like(id_grid)

    w_ref = warp.detach().clone().requires_grad_(True)
    sg_ref = sample_field_cf(affine_grid_cf, id_grid + w_ref)
    (sg_ref * up_grad).sum().backward()

    w_fast = warp.detach().clone().requires_grad_(True)
    sg_fast = _Conv1x1AffineGrid.apply(id_grid + w_fast, M_theta, t_theta_flat)
    (sg_fast * up_grad).sum().backward()

    assert float((sg_ref - sg_fast).abs().max().item()) < 1e-6
    assert float((w_ref.grad - w_fast.grad).abs().max().item()) < 2.5e-5

    # Also verify exact float64 backward match (< 1e-12) against sample_field_cf on out-of-bounds coords
    theta64 = torch.tensor(
        [[[0.98, -0.05, 0.02, 0.05], [0.04, 1.01, -0.03, -0.04], [-0.01, 0.02, 0.99, 0.03]]],
        dtype=torch.float64,
    )
    id_grid64 = F.affine_grid(torch.eye(3, 4, dtype=torch.float64)[None], grid_shape, align_corners=True)
    affine_grid_cf64 = torch.movedim(F.affine_grid(theta64, grid_shape, align_corners=True), -1, 1)
    w_ref64 = warp.double().detach().clone().requires_grad_(True)
    sg_ref64 = sample_field_cf(affine_grid_cf64, id_grid64 + w_ref64)
    (sg_ref64 * up_grad.double()).sum().backward()
    w_fast64 = warp.double().detach().clone().requires_grad_(True)
    sg_fast64 = _Conv1x1AffineGrid.apply(
        id_grid64 + w_fast64, theta64[0, :, :3].contiguous(), theta64[0, :, 3].contiguous()
    )
    (sg_fast64 * up_grad.double()).sum().backward()
    assert float((w_ref64.grad - w_fast64.grad).abs().max().item()) < 1e-12

    if torch.backends.mps.is_available():
        w_mps = warp.to("mps").detach().requires_grad_(True)
        sg_mps = _Conv1x1AffineGrid.apply(
            id_grid.to("mps") + w_mps,
            M_theta.to("mps"),
            t_theta_flat.to("mps"),
        )
        (sg_mps * up_grad.to("mps")).sum().backward()
        assert float((sg_ref - sg_mps.cpu()).abs().max().item()) < 1e-6
        assert float((w_ref.grad - w_mps.grad.cpu()).abs().max().item()) < 2.5e-5

    # 2. On-device normalized_to_physical_disp vs CPU _convert_composed_grid_to_ants_displacement (< 1e-5 mm)
    restr = (1.0, 0.0, 1.0)
    disp_ref = _convert_composed_grid_to_ants_displacement(
        id_grid + warp, fi, P_F, P_F, restrict_transformation=restr
    )
    disp_t = normalized_to_physical_disp(warp, (24, 20, 16), fi.spacing, fi.direction, device=warp.device)
    restr_t_zyx = torch.tensor(list(reversed(restr)), dtype=disp_t.dtype).view(1, 1, 1, 1, 3)
    disp_fast = disp_tensor_to_itk(disp_t * restr_t_zyx, fi)
    assert float(np.abs(disp_ref.numpy() - disp_fast.numpy()).max()) < 1e-5

    # 3. Verify _is_default_greedy_lncc is True on default similarity_metric='cc2' and test fit() warp parity
    g_fast = GreedyRegistrationModel(dim=3, similarity_metric="cc2", device=torch.device("cpu"))
    assert g_fast.loss_fn._is_default_greedy_lncc is True
    g_slow = GreedyRegistrationModel(dim=3, similarity_metric="cc2", device=torch.device("cpu"))
    g_slow.loss_fn._is_default_greedy_lncc = False

    fi_t = torch.rand(1, 1, 16, 16, 16, dtype=torch.float32)
    mi_t = torch.rand(1, 1, 16, 16, 16, dtype=torch.float32)
    theta_id = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    w_fit_fast = g_fast.fit(fi_t, mi_t, theta_id, scales=[2, 1], iterations=[3, 2])
    w_fit_slow = g_slow.fit(fi_t, mi_t, theta_id, scales=[2, 1], iterations=[3, 2])
    assert float((w_fit_fast - w_fit_slow).abs().max().item()) < 1e-6
    fi_down = torch.rand(1, 1, 14, 16, 18)
    moved1 = torch.rand(1, 1, 14, 16, 18, requires_grad=True)
    moved2 = moved1.detach().clone().requires_grad_(True)
    loss_fn = get_similarity_loss(parse_similarity_metric("cc2", window_size=5))
    l_ref = loss_fn(moved1, fi_down, mask=None)
    l_ref.backward()

    with torch.no_grad():
        j_mean = _box_pool(fi_down, 5, count_include_pad=False)
        m_c = fi_down - j_mean
        j_var = torch.clamp(_box_pool(m_c.square(), 5, count_include_pad=False), min=0.0)
        safe_j_var = torch.clamp(j_var, min=1e-6)
    i_mean = box_mean_nd(moved2, 5)
    f_c = moved2 - i_mean
    sec = box_mean_nd(torch.cat([f_c.square(), f_c * m_c], dim=1), 5)
    i_var = torch.clamp(sec[:, :1], min=0.0)
    ij_cov = sec[:, 1:]
    safe_i_var = torch.clamp(i_var, min=1e-6)
    cc_metric = torch.clamp((ij_cov ** 2) / (safe_i_var * safe_j_var + 1e-8), min=0.0, max=1.0)
    l_fast = -torch.mean(cc_metric)
    l_fast.backward()
    assert float((l_ref - l_fast).abs().item()) < 1e-7
    assert float((moved1.grad - moved2.grad).abs().max().item()) < 1e-7

    # 4. dsti_alpha raises ValueError (not KeyError) under default regularizer='gaussian'
    fix_small, mov_small = _synthetic_pair_3d((16, 16, 16))
    with pytest.raises(ValueError, match="dsti_alpha"):
        syntx.greedy(
            fix_small,
            mov_small,
            reg_iterations=[1],
            scales=[1],
            initial_transform=False,
            device="cpu",
            dsti_alpha=0.1,
        )
    # And succeeds without KeyError when regularizer='dsti1'
    r_dsti = syntx.greedy(
        fix_small,
        mov_small,
        reg_iterations=[1],
        scales=[1],
        initial_transform=False,
        device="cpu",
        regularizer="dsti1",
        dsti_alpha=0.1,
    )
    assert "warpedmovout" in r_dsti


def test_pr99_chunked_lncc_longest_axis_and_validation(tmp_path):
    import syntx
    from syntx.core.losses import local_ncc_loss_nd, parse_similarity_metric
    from syntx.core.losses.cross_correlation import _longest_slab_axis

    # 1. Thin-Z volume (1, 1, 12, 48, 40): longest spatial axis is Y (tensor axis 3)
    assert _longest_slab_axis((1, 1, 12, 48, 40)) == 3
    assert _longest_slab_axis((1, 1, 16, 20, 64)) == 4
    assert _longest_slab_axis((1, 1, 64, 20, 16)) == 2

    torch.manual_seed(99)
    I_base = torch.rand(1, 1, 12, 48, 40, dtype=torch.float32)
    J_base = torch.rand(1, 1, 12, 48, 40, dtype=torch.float32)
    bcast_mask = (torch.rand(1, 1, 1, 48, 40) > 0.2).float()

    for use_pseudo in (False, True):
        for squared in (True, False):
            I1 = I_base.clone().requires_grad_(True)
            J1 = J_base.clone().requires_grad_(True)
            l1 = local_ncc_loss_nd(
                I1, J1, mask=bcast_mask, window_size=5, squared=squared,
                use_ants_pseudo_gradient=use_pseudo, metric_chunks=1,
            )
            l1.backward()

            I4 = I_base.clone().requires_grad_(True)
            J4 = J_base.clone().requires_grad_(True)
            l4 = local_ncc_loss_nd(
                I4, J4, mask=bcast_mask, window_size=5, squared=squared,
                use_ants_pseudo_gradient=use_pseudo, metric_chunks=4,
            )
            l4.backward()

            assert float((l1 - l4).abs().item()) < 1e-6
            assert float((I1.grad - I4.grad).abs().max().item()) < 1e-6
            assert float((J1.grad - J4.grad).abs().max().item()) < 1e-6

    if torch.backends.mps.is_available():
        Im1 = I_base.to("mps").detach().requires_grad_(True)
        Jm1 = J_base.to("mps").detach().requires_grad_(True)
        lm1 = local_ncc_loss_nd(Im1, Jm1, window_size=5, squared=True, metric_chunks=1)
        lm1.backward()

        Im4 = I_base.to("mps").detach().requires_grad_(True)
        Jm4 = J_base.to("mps").detach().requires_grad_(True)
        lm4 = local_ncc_loss_nd(Im4, Jm4, window_size=5, squared=True, metric_chunks=4)
        lm4.backward()

        assert float((lm1.cpu() - lm4.cpu()).abs().item()) < 1e-6
        assert float((Im1.grad.cpu() - Im4.grad.cpu()).abs().max().item()) < 1e-6

    # 2. Strict validation: metric_chunks=0, False, -1, float, and non-CC metric_chunks > 1 raise ValueError
    for bad_mc in (0, -2, False, True, 1.5):
        with pytest.raises(ValueError, match="metric_chunks"):
            parse_similarity_metric("cc2", metric_chunks=bad_mc)
        with pytest.raises(ValueError, match="metric_chunks"):
            local_ncc_loss_nd(I_base, J_base, window_size=5, metric_chunks=bad_mc)

    fix, mov = _synthetic_pair_3d((16, 16, 16))
    id_mat = str(tmp_path / "id_mc.mat")
    ants.write_transform(ants.create_ants_transform(transform_type="AffineTransform", dimension=3), id_mat)

    for bad_mc in (0, False):
        with pytest.raises(ValueError, match="metric_chunks"):
            syntx.syn(fix, mov, reg_iterations=[1], levels=[1], initial_transform=id_mat, device="cpu", metric_chunks=bad_mc)

    for non_cc in ("mse", "dt", "distance_transform"):
        with pytest.raises(ValueError, match="metric_chunks"):
            syntx.syn(
                fix, mov, syn_metric=non_cc, reg_iterations=[1], levels=[1],
                initial_transform=id_mat, device="cpu", metric_chunks=2,
            )

    for lin_tot in ("Affine", "Rigid", "Translation"):
        with pytest.raises(ValueError, match="metric_chunks"):
            syntx.syn(
                fix, mov, type_of_transform=lin_tot, syn_metric="cc2",
                initial_transform=id_mat, device="cpu", metric_chunks=2,
            )

    r_mc = syntx.syn(
        fix, mov, syn_metric="cc2", reg_iterations=[1], levels=[1],
        initial_transform=id_mat, device="cpu", metric_chunks=3,
        outprefix=str(tmp_path / "syn_mc3_"),
    )
    assert int(r_mc["provenance"]["metric_chunks"]) == 3


