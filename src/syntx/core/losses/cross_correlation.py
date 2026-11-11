"""
Canonical cross-correlation similarity losses (LNCC, CC2, BoxLNCC).

Supports 2-D and 3-D tensors (B, C, *spatial), exact autograd or ANTs/ITK pseudo-derivatives,
deterministic pooling on GPU/MPS, variance floors (>= 1e-6), float64 gradcheck compatibility,
and AMP float16 overflow safety.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..smoothing import separable_1d_filter


_WINDOW_COUNT_CACHE: dict[
    tuple,
    tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
] = {}
_FUSE_MAX_VOXELS = 16_777_216


def _get_window_count_views(
    spatial_shape: tuple[int, ...],
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Return cached ``(inv_counts_1d, counts_1d, pad_ratios_1d)`` broadcast views."""
    key = (tuple(int(s) for s in spatial_shape), int(window_size), str(device), dtype)
    cached = _WINDOW_COUNT_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_WINDOW_COUNT_CACHE) >= 64:
        _WINDOW_COUNT_CACHE.clear()

    dim = len(spatial_shape)
    pad = window_size // 2
    k_f = float(window_size)
    inv_views: list[torch.Tensor] = []
    cnt_views: list[torch.Tensor] = []
    ratio_views: list[torch.Tensor] = []
    for d, n in enumerate(spatial_shape):
        idx = torch.arange(n, device=device, dtype=dtype)
        c_1d = torch.clamp(idx + pad, max=n - 1) - torch.clamp(idx - pad, min=0) + 1.0
        v_shape = [1, 1] + [1] * dim
        v_shape[d + 2] = n
        cnt_views.append(c_1d.reshape(v_shape))
        inv_views.append((1.0 / c_1d).reshape(v_shape))
        ratio_views.append((k_f / c_1d).reshape(v_shape))

    res = (tuple(inv_views), tuple(cnt_views), tuple(ratio_views))
    _WINDOW_COUNT_CACHE[key] = res
    return res


def _scale_by_views_inplace(y: torch.Tensor, views: tuple[torch.Tensor, ...]) -> torch.Tensor:
    for v in views:
        y.mul_(v)
    return y


def _separable_avg_pool_nd(x: torch.Tensor, window_size: int, count_include_pad: bool) -> torch.Tensor:
    """Exact separable 1-D ``avg_pool2d`` / ``avg_pool3d`` passes (`3 * k` or `2 * k` taps instead of `k^d`)."""
    pad = window_size // 2
    if x.dim() == 4:
        y = F.avg_pool2d(
            x, kernel_size=(window_size, 1), stride=1, padding=(pad, 0), count_include_pad=count_include_pad
        )
        return F.avg_pool2d(
            y, kernel_size=(1, window_size), stride=1, padding=(0, pad), count_include_pad=count_include_pad
        )
    if x.dim() == 5:
        y = F.avg_pool3d(
            x, kernel_size=(window_size, 1, 1), stride=1, padding=(pad, 0, 0), count_include_pad=count_include_pad
        )
        y = F.avg_pool3d(
            y, kernel_size=(1, window_size, 1), stride=1, padding=(0, pad, 0), count_include_pad=count_include_pad
        )
        return F.avg_pool3d(
            y, kernel_size=(1, 1, window_size), stride=1, padding=(0, 0, pad), count_include_pad=count_include_pad
        )
    raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D.")


def _box_pool(x: torch.Tensor, window_size: int, count_include_pad: bool) -> torch.Tensor:
    """Stride-1 local box mean with zero padding ``window_size // 2`` (exact separable 1-D for odd windows)."""
    if window_size % 2 == 0:
        pool_fn = F.avg_pool2d if x.dim() == 4 else F.avg_pool3d
        return pool_fn(
            x,
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2,
            count_include_pad=count_include_pad,
        )
    if window_size <= 1:
        return x.clone()
    return _separable_avg_pool_nd(x, window_size, count_include_pad=count_include_pad)


class _DeterministicBoxMean(torch.autograd.Function):
    """Exact separable local box mean (stride 1, zero padding, ``count_include_pad=False``)
    with zero saved tensors and deterministic single-pass backward on CPU, CUDA, and MPS."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, window_size: int) -> torch.Tensor:
        ctx.window_size = window_size
        return _box_pool(x, window_size, count_include_pad=False)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        k = ctx.window_size
        if k <= 1:
            return g.clone(), None
        _, _, pad_ratios = _get_window_count_views(g.shape[2:], k, g.device, g.dtype)
        scaled = _scale_by_views_inplace(g.clone(), pad_ratios)
        return _separable_avg_pool_nd(scaled, k, count_include_pad=True), None


# Devices on which box_mean_nd uses the deterministic backward (all devices use separable _DeterministicBoxMean).
DETERMINISTIC_POOL_DEVICES = ("mps", "cuda", "cpu")


def box_mean_nd(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Local box mean of ``x`` (B, C, *spatial), 2-D or 3-D: border mean is over in-image voxels only."""
    if window_size % 2 == 1 and x.requires_grad and torch.is_grad_enabled():
        return _DeterministicBoxMean.apply(x, window_size)
    return _box_pool(x, window_size, count_include_pad=False)


def _window_counts(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Number of in-image voxels of each box window (zero padding excluded), shape ``(B, 1, *spatial)``."""
    if window_size % 2 == 1:
        _, cnt_views, _ = _get_window_count_views(x.shape[2:], window_size, x.device, x.dtype)
        prod = cnt_views[0]
        for cv in cnt_views[1:]:
            prod = prod * cv
        return prod.expand(x.shape[0], 1, *x.shape[2:]).clone()
    dim = x.dim() - 2
    pool = F.avg_pool2d if dim == 2 else F.avg_pool3d
    ones = torch.ones_like(x[:, :1])
    pad = window_size // 2
    return pool(ones, kernel_size=window_size, stride=1, padding=pad, count_include_pad=True) * (window_size**dim)


def _centered_moments_nd(
    I_w: torch.Tensor,
    J_w: torch.Tensor,
    window_size: int,
    differentiable: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(F_c, M_c, I_var, J_var, IJ_cov)`` using fused ``[2C]`` + ``[3C]`` separable box means
    below ``_FUSE_MAX_VOXELS``, falling back to per-channel passes on ultra-large unchunked volumes."""
    pool = box_mean_nd if differentiable else (lambda t, k: _box_pool(t, k, count_include_pad=False))
    c = I_w.shape[1]
    if I_w.numel() <= _FUSE_MAX_VOXELS:
        ij_mean = pool(torch.cat([I_w, J_w], dim=1), window_size)
        I_mean = ij_mean[:, :c]
        J_mean = ij_mean[:, c:]
        F_c = I_w - I_mean
        M_c = J_w - J_mean
        del ij_mean, I_mean, J_mean
        sec = pool(torch.cat([F_c.square(), M_c.square(), F_c * M_c], dim=1), window_size)
        I_var = torch.clamp(sec[:, :c], min=0.0)
        J_var = torch.clamp(sec[:, c : 2 * c], min=0.0)
        IJ_cov = sec[:, 2 * c :]
        return F_c, M_c, I_var, J_var, IJ_cov

    I_mean = pool(I_w, window_size)
    J_mean = pool(J_w, window_size)
    F_c = I_w - I_mean
    M_c = J_w - J_mean
    del I_mean, J_mean
    I_var = torch.clamp(pool(F_c.square(), window_size), min=0.0)
    J_var = torch.clamp(pool(M_c.square(), window_size), min=0.0)
    IJ_cov = pool(F_c * M_c, window_size)
    return F_c, M_c, I_var, J_var, IJ_cov


def _single_pass_5ch_moments_nd(
    I_w: torch.Tensor,
    J_w: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-window ``E[X^2] - E[X]^2`` and ``E[XY] - E[X]E[Y]`` separable box moments
    after translation-invariant global-mean centering, split as ``[2C] + [3C]`` below
    ``_FUSE_MAX_VOXELS`` to halve peak concatenation traffic on unified-memory GPUs."""
    spatial_dims = tuple(range(2, I_w.dim()))
    I_0 = I_w - I_w.detach().mean(dim=spatial_dims, keepdim=True)
    J_0 = J_w - J_w.detach().mean(dim=spatial_dims, keepdim=True)
    c = I_w.shape[1]
    if I_w.numel() <= _FUSE_MAX_VOXELS:
        m1 = box_mean_nd(torch.cat([I_0, J_0], dim=1), window_size)
        m_I = m1[:, :c]
        m_J = m1[:, c:]
        m2 = box_mean_nd(torch.cat([I_0.square(), J_0.square(), I_0 * J_0], dim=1), window_size)
        m_II = m2[:, :c]
        m_JJ = m2[:, c : 2 * c]
        m_IJ = m2[:, 2 * c :]
    else:
        m_I = box_mean_nd(I_0, window_size)
        m_J = box_mean_nd(J_0, window_size)
        m_II = box_mean_nd(I_0.square(), window_size)
        m_JJ = box_mean_nd(J_0.square(), window_size)
        m_IJ = box_mean_nd(I_0 * J_0, window_size)
    I_var = torch.clamp(m_II - m_I.square(), min=0.0)
    J_var = torch.clamp(m_JJ - m_J.square(), min=0.0)
    IJ_cov = m_IJ - m_I * m_J
    return I_var, J_var, IJ_cov


class AnalyticalLNCC(torch.autograd.Function):
    """Local CC loss (not CC^2) with a hand-written, ANTs-style approximate backward pass."""

    @staticmethod
    def forward(ctx, I: torch.Tensor, J: torch.Tensor, mask: torch.Tensor | None, window_size: int):
        dim = I.dim() - 2
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")

        # Safe execution across CPU and accelerators for Half precision
        needs_upcast = I.dtype in (torch.float16, torch.bfloat16)
        work_dtype = torch.float32 if needs_upcast else I.dtype
        I_w = I.to(work_dtype)
        J_w = J.to(work_dtype)
        mask_w = mask.to(work_dtype) if mask is not None else None

        F_centered, M_centered, I_var, J_var, IJ_cov = _centered_moments_nd(
            I_w, J_w, window_size, differentiable=False
        )

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        denom = torch.sqrt(safe_I_var * safe_J_var) + 1e-6
        cc_raw = IJ_cov / denom
        cc = torch.clamp(cc_raw, min=-1.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, cc, safe_I_var, safe_J_var)
        ctx.window_size = window_size
        ctx.needs_upcast = needs_upcast
        ctx.orig_dtype = I.dtype

        if mask_w is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask_w > 0.5)).to(work_dtype)
            loss = -torch.sum(cc * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc)
            ctx.active = None

        return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        F_centered, M_centered, cc, safe_I_var, safe_J_var = ctx.saved_tensors

        work_dtype = F_centered.dtype
        grad_out_w = grad_output.to(work_dtype)

        inv_denom = 1.0 / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)
        inv_counts, _, _ = _get_window_count_views(
            F_centered.shape[2:], ctx.window_size, F_centered.device, work_dtype
        )
        scale = _scale_by_views_inplace(-inv_denom, inv_counts)

        grad_J = scale * (F_centered - cc * M_centered)
        grad_I = scale * (M_centered - cc * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        g_I = (grad_I * grad_out_w).to(ctx.orig_dtype)
        g_J = (grad_J * grad_out_w).to(ctx.orig_dtype)
        return g_I, g_J, None, None


class ANTsPseudoLNCC(torch.autograd.Function):
    """Local CC^2 loss with the ITK / ANTs pseudo-derivative as its backward pass."""

    @staticmethod
    def forward(ctx, I: torch.Tensor, J: torch.Tensor, mask: torch.Tensor | None, window_size: int):
        dim = I.dim() - 2
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")

        needs_upcast = I.dtype in (torch.float16, torch.bfloat16)
        work_dtype = torch.float32 if needs_upcast else I.dtype
        I_w = I.to(work_dtype)
        J_w = J.to(work_dtype)
        mask_w = mask.to(work_dtype) if mask is not None else None

        F_centered, M_centered, I_var, J_var, IJ_cov = _centered_moments_nd(
            I_w, J_w, window_size, differentiable=False
        )

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        cc2_raw = (IJ_cov**2) / (safe_I_var * safe_J_var + 1e-8)
        cc2 = torch.clamp(cc2_raw, min=0.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var)
        ctx.window_size = window_size
        ctx.needs_upcast = needs_upcast
        ctx.orig_dtype = I.dtype

        if mask_w is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask_w > 0.5)).to(work_dtype)
            loss = -torch.sum(cc2 * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc2)
            ctx.active = None

        return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var = ctx.saved_tensors

        work_dtype = F_centered.dtype
        grad_out_w = grad_output.to(work_dtype)

        s_FM = IJ_cov
        s_FF = safe_I_var
        s_MM = safe_J_var
        sFF_sMM = s_FF * s_MM + 1e-8

        inv_counts, _, _ = _get_window_count_views(
            F_centered.shape[2:], ctx.window_size, F_centered.device, work_dtype
        )
        grad_factor = _scale_by_views_inplace(-2.0 * (s_FM / sFF_sMM), inv_counts)
        grad_J = grad_factor * (F_centered - (s_FM / (s_MM + 1e-8)) * M_centered)
        grad_I = grad_factor * (M_centered - (s_FM / (s_FF + 1e-8)) * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        g_I = (grad_I * grad_out_w).to(ctx.orig_dtype)
        g_J = (grad_J * grad_out_w).to(ctx.orig_dtype)
        return g_I, g_J, None, None


def _slab_bounds(depth: int, n_chunks: int, radius: int) -> list[tuple[int, int, int, int, int, int]]:
    """Partition ``[0, depth)`` into ``n_chunks`` core slabs ``[z0, z1)`` padded by a ``2 * radius``
    halo ``[s0, s1)`` with local core slice ``[c0, c1)``."""
    halo = 2 * radius
    slabs: list[tuple[int, int, int, int, int, int]] = []
    for idx in range(n_chunks):
        z0 = (idx * depth) // n_chunks
        z1 = ((idx + 1) * depth) // n_chunks
        if z1 <= z0:
            continue
        s0 = max(0, z0 - halo)
        s1 = min(depth, z1 + halo)
        slabs.append((z0, z1, s0, s1, z0 - s0, z1 - s0))
    return slabs


def _lncc_chunked_no_grad(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None,
    window_size: int,
    squared: bool,
    floor_val: float,
    n_chunks: int,
    work_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded no-grad evaluation of LNCC numerator and denominator across depth slabs."""
    slabs = _slab_bounds(I.shape[2], n_chunks, window_size // 2)
    total_num = torch.zeros((), device=I.device, dtype=work_dtype)
    total_active = torch.zeros((), device=I.device, dtype=work_dtype) if mask is not None else None

    for z0, z1, s0, s1, c0, c1 in slabs:
        I_s = I[:, :, s0:s1].to(work_dtype)
        J_s = J[:, :, s0:s1].to(work_dtype)
        F_c, M_c, I_var_f, J_var_f, IJ_cov_f = _centered_moments_nd(
            I_s, J_s, window_size, differentiable=False
        )
        del I_s, J_s, F_c, M_c
        I_var_c = I_var_f[:, :, c0:c1]
        J_var_c = J_var_f[:, :, c0:c1]
        IJ_cov_c = IJ_cov_f[:, :, c0:c1]
        del I_var_f, J_var_f, IJ_cov_f

        safe_I = torch.clamp(I_var_c, min=floor_val)
        safe_J = torch.clamp(J_var_c, min=floor_val)
        if squared:
            cc_c = torch.clamp((IJ_cov_c**2) / (safe_I * safe_J + 1e-8), min=0.0, max=1.0)
        else:
            cc_c = torch.clamp(IJ_cov_c / (torch.sqrt(safe_I * safe_J) + 1e-6), min=-1.0, max=1.0)
        del IJ_cov_c, safe_I, safe_J

        if mask is not None:
            m_c = (mask[:, :, z0:z1] if mask.shape[2] > 1 else mask).to(work_dtype)
            active_c = ((I_var_c > floor_val) & (J_var_c > floor_val) & (m_c > 0.5)).to(work_dtype)
            total_num = total_num - torch.sum(cc_c * active_c)
            total_active = total_active + torch.sum(active_c)
            del m_c, active_c
        else:
            total_num = total_num - torch.sum(cc_c)
        del I_var_c, J_var_c, cc_c

    if mask is not None:
        total_den = total_active + 1e-8
    else:
        total_den = torch.tensor(float(I.numel()), device=I.device, dtype=work_dtype)
    return total_num, total_den


class _ChunkedLocalNCC(torch.autograd.Function):
    """Exact slab-halo-chunked LNCC / CC^2 (autograd or ANTs pseudo-gradient) along spatial axis 0.

    Each core slab ``[z0, z1)`` is evaluated with a ``2 * (window_size // 2)`` halo ``[s0, s1)``
    so both nested ``box_mean_nd`` passes and their backward passes see exact interior window
    counts at internal slab boundaries and exact border counts at ``0`` and ``D``.
    """

    @staticmethod
    def forward(
        ctx,
        I: torch.Tensor,
        J: torch.Tensor,
        mask: torch.Tensor | None,
        window_size: int,
        squared: bool,
        floor_val: float,
        n_chunks: int,
        use_pseudo: bool,
    ) -> torch.Tensor:
        needs_upcast = I.dtype in (torch.float16, torch.bfloat16)
        work_dtype = torch.float32 if needs_upcast else I.dtype
        dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

        if mask is not None and mask.dim() < I.dim():
            mask = mask.expand_as(I[:, :1])

        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=False):
            total_num, total_den = _lncc_chunked_no_grad(
                I, J, mask, window_size, squared, floor_val, n_chunks, work_dtype
            )
            loss = total_num / total_den

        if mask is not None:
            ctx.save_for_backward(I, J, mask, total_den)
            ctx.has_mask = True
        else:
            ctx.save_for_backward(I, J, total_den)
            ctx.has_mask = False
        ctx.window_size = window_size
        ctx.squared = squared
        ctx.floor_val = floor_val
        ctx.n_chunks = n_chunks
        ctx.use_pseudo = use_pseudo
        ctx.work_dtype = work_dtype
        ctx.orig_dtype = I.dtype
        ctx.dev_type = dev_type
        return loss if needs_upcast else loss.to(I.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        need_I = ctx.needs_input_grad[0]
        need_J = ctx.needs_input_grad[1]
        if not need_I and not need_J:
            return None, None, None, None, None, None, None, None

        if ctx.has_mask:
            I, J, mask, total_den = ctx.saved_tensors
        else:
            I, J, total_den = ctx.saved_tensors
            mask = None

        work_dtype = ctx.work_dtype
        window_size = ctx.window_size
        squared = ctx.squared
        floor_val = ctx.floor_val
        slabs = _slab_bounds(I.shape[2], ctx.n_chunks, window_size // 2)

        grad_I = torch.zeros(I.shape, device=I.device, dtype=work_dtype) if need_I else None
        grad_J = torch.zeros(J.shape, device=J.device, dtype=work_dtype) if need_J else None

        if ctx.use_pseudo:
            with torch.no_grad(), torch.amp.autocast(device_type=ctx.dev_type, enabled=False):
                for z0, z1, s0, s1, c0, c1 in slabs:
                    I_s = I[:, :, s0:s1].to(work_dtype)
                    J_s = J[:, :, s0:s1].to(work_dtype)
                    F_full, M_full, I_var_f, J_var_f, IJ_cov_f = _centered_moments_nd(
                        I_s, J_s, window_size, differentiable=False
                    )
                    del I_s, J_s
                    I_var_c = I_var_f[:, :, c0:c1]
                    J_var_c = J_var_f[:, :, c0:c1]
                    IJ_cov_c = IJ_cov_f[:, :, c0:c1]
                    del I_var_f, J_var_f, IJ_cov_f
                    n_win_c = _window_counts(F_full, window_size)[:, :, c0:c1]
                    F_c = F_full[:, :, c0:c1]
                    M_c = M_full[:, :, c0:c1]
                    del F_full, M_full

                    safe_I = torch.clamp(I_var_c, min=floor_val)
                    safe_J = torch.clamp(J_var_c, min=floor_val)
                    if squared:
                        sFF_sMM = safe_I * safe_J + 1e-8
                        grad_factor = -2.0 * (1.0 / n_win_c) * (IJ_cov_c / sFF_sMM)
                        gJ_c = grad_factor * (F_c - (IJ_cov_c / (safe_J + 1e-8)) * M_c)
                        gI_c = grad_factor * (M_c - (IJ_cov_c / (safe_I + 1e-8)) * F_c)
                    else:
                        cc_c = torch.clamp(IJ_cov_c / (torch.sqrt(safe_I * safe_J) + 1e-6), min=-1.0, max=1.0)
                        scale = -(1.0 / n_win_c) / (torch.sqrt(safe_I * safe_J) + 1e-6)
                        gJ_c = scale * (F_c - cc_c * M_c)
                        gI_c = scale * (M_c - cc_c * F_c)

                    if mask is not None:
                        m_c = (mask[:, :, z0:z1] if mask.shape[2] > 1 else mask).to(work_dtype)
                        active_c = ((I_var_c > floor_val) & (J_var_c > floor_val) & (m_c > 0.5)).to(work_dtype)
                        gI_c = gI_c * active_c
                        gJ_c = gJ_c * active_c

                    if grad_I is not None:
                        grad_I[:, :, z0:z1].copy_(gI_c)
                    if grad_J is not None:
                        grad_J[:, :, z0:z1].copy_(gJ_c)

            pseudo_den = total_den if mask is not None else (float(I.numel()) / float(I.shape[0]))
            scale_out = grad_output.to(work_dtype) / pseudo_den
        else:
            with torch.enable_grad(), torch.amp.autocast(device_type=ctx.dev_type, enabled=False):
                for z0, z1, s0, s1, c0, c1 in slabs:
                    I_s = I[:, :, s0:s1].detach().to(work_dtype).requires_grad_(need_I)
                    J_s = J[:, :, s0:s1].detach().to(work_dtype).requires_grad_(need_J)

                    F_s, M_s, I_var_f, J_var_f, IJ_cov_f = _centered_moments_nd(
                        I_s, J_s, window_size, differentiable=True
                    )
                    I_var_c = I_var_f[:, :, c0:c1]
                    J_var_c = J_var_f[:, :, c0:c1]
                    IJ_cov_c = IJ_cov_f[:, :, c0:c1]
                    del I_var_f, J_var_f, IJ_cov_f

                    safe_I = torch.clamp(I_var_c, min=floor_val)
                    safe_J = torch.clamp(J_var_c, min=floor_val)
                    if squared:
                        cc_c = torch.clamp((IJ_cov_c**2) / (safe_I * safe_J + 1e-8), min=0.0, max=1.0)
                    else:
                        cc_c = torch.clamp(IJ_cov_c / (torch.sqrt(safe_I * safe_J) + 1e-6), min=-1.0, max=1.0)

                    if mask is not None:
                        m_c = (mask[:, :, z0:z1] if mask.shape[2] > 1 else mask).detach().to(work_dtype)
                        active_c = (
                            (I_var_c.detach() > floor_val) & (J_var_c.detach() > floor_val) & (m_c > 0.5)
                        ).to(work_dtype)
                        slab_num = -torch.sum(cc_c * active_c)
                    else:
                        slab_num = -torch.sum(cc_c)

                    if slab_num.requires_grad:
                        req_inputs: list[torch.Tensor] = []
                        if need_I:
                            req_inputs.append(I_s)
                        if need_J:
                            req_inputs.append(J_s)
                        grads = torch.autograd.grad(slab_num, req_inputs, allow_unused=True)
                        g_idx = 0
                        if need_I:
                            if grads[g_idx] is not None:
                                grad_I[:, :, s0:s1].add_(grads[g_idx])
                            g_idx += 1
                        if need_J:
                            if grads[g_idx] is not None:
                                grad_J[:, :, s0:s1].add_(grads[g_idx])

                    del I_s, J_s, F_s, M_s, I_var_c, J_var_c, IJ_cov_c, safe_I, safe_J, cc_c, slab_num

            scale_out = grad_output.to(work_dtype) / total_den

        if grad_I is not None:
            grad_I = (grad_I * scale_out).to(ctx.orig_dtype)
        if grad_J is not None:
            grad_J = (grad_J * scale_out).to(ctx.orig_dtype)
        return grad_I, grad_J, None, None, None, None, None, None


def local_ncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    window_size: int = 9,
    use_ants_pseudo_gradient: bool = False,
    squared: bool = False,
    var_floor: float = 1e-6,
    use_analytical_gradients: bool = False,
    metric_chunks: int = 1,
    single_pass: bool = False,
) -> torch.Tensor:
    """
    Negative local normalised cross-correlation (LNCC) of 2-D or 3-D images I and J.

    Parameters
    ----------
    I, J : torch.Tensor
        Images (B, C, *spatial), same shape.
    mask : torch.Tensor, optional
        (B, 1, *spatial) or broadcastable. Active voxels: mask > 0.5.
    window_size : int, default 9
        Box width in voxels; must be odd.
    use_ants_pseudo_gradient : bool, default False
        True: use hand-written ANTs pseudo-gradient backward pass.
    squared : bool, default False
        Use CC^2 instead of linear CC.
    var_floor : float, default 1e-6
        Variance floor per GEMINI.md Rule 3.
    use_analytical_gradients : bool, default False
        Alias for ``use_ants_pseudo_gradient``.
    metric_chunks : int, default 1
        Number of slabs along spatial axis 0 (depth in 3-D, height in 2-D). ``> 1`` evaluates
        each slab with an exact ``2 * (window_size // 2)`` halo, cutting transient LNCC
        intermediate memory by ``~metric_chunks`` with identical forward and backward values.
    single_pass : bool, default False
        When True (and ``not use_pseudo`` and ``metric_chunks == 1``), fuse all 5 moments
        ``(I, J, I^2, J^2, I*J)`` into a single ``[B, 5*C, *spatial]`` separable pass
        (ANTs single-window ``E[IJ] - E[I]E[J]`` formulation).

    Returns
    -------
    torch.Tensor
        Scalar similarity loss (lower is better).
    """
    dim = I.dim() - 2
    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"local_ncc_loss_nd: window_size must be a positive odd integer, got {window_size}")
    if int(metric_chunks) < 1:
        raise ValueError(f"local_ncc_loss_nd: metric_chunks must be >= 1, got {metric_chunks}")

    use_pseudo = use_ants_pseudo_gradient or use_analytical_gradients

    # Adapt window size dynamically if input is smaller than kernel
    min_spatial = min(I.shape[2:])
    if window_size > min_spatial:
        window_size = min_spatial
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)

    n_chunks = min(int(metric_chunks), I.shape[2])
    if n_chunks > 1:
        floor_c = 1e-6 if use_pseudo else max(float(var_floor), 1e-6)
        if not torch.is_grad_enabled() or (not I.requires_grad and not J.requires_grad):
            needs_upcast = I.dtype in (torch.float16, torch.bfloat16)
            work_dtype = torch.float32 if needs_upcast else I.dtype
            dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")
            mask_exp = mask.expand_as(I[:, :1]) if (mask is not None and mask.dim() < I.dim()) else mask
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=False):
                total_num, total_den = _lncc_chunked_no_grad(
                    I, J, mask_exp, window_size, squared, floor_c, n_chunks, work_dtype
                )
                loss = total_num / total_den
            return loss if needs_upcast else loss.to(I.dtype)
        return _ChunkedLocalNCC.apply(I, J, mask, window_size, squared, floor_c, n_chunks, use_pseudo)

    if use_pseudo and squared:
        return ANTsPseudoLNCC.apply(I, J, mask, window_size)
    elif use_pseudo and not squared:
        return AnalyticalLNCC.apply(I, J, mask, window_size)

    # Autograd path with AMP safety
    work_dtype = torch.float32 if I.dtype in (torch.float16, torch.bfloat16) else I.dtype
    dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        I_w = I.to(work_dtype)
        J_w = J.to(work_dtype)

        if single_pass and I_w.numel() <= _FUSE_MAX_VOXELS:
            I_var, J_var, IJ_cov = _single_pass_5ch_moments_nd(I_w, J_w, window_size)
        else:
            _, _, I_var, J_var, IJ_cov = _centered_moments_nd(
                I_w, J_w, window_size, differentiable=True
            )

        floor_val = max(float(var_floor), 1e-6)
        safe_I_var = torch.clamp(I_var, min=floor_val)
        safe_J_var = torch.clamp(J_var, min=floor_val)

        if squared:
            cc_metric = (IJ_cov**2) / (safe_I_var * safe_J_var + 1e-8)
            cc_metric = torch.clamp(cc_metric, min=0.0, max=1.0)
        else:
            cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)
            cc_metric = torch.clamp(cc_raw, min=-1.0, max=1.0)

        if mask is not None:
            m = mask.to(work_dtype)
            active_mask_float = ((I_var > floor_val) & (J_var > floor_val) & (m > 0.5)).to(work_dtype)
            loss = -torch.sum(cc_metric * active_mask_float) / (torch.sum(active_mask_float) + 1e-8)
        else:
            loss = -torch.mean(cc_metric)

    return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)


class BoxLNCCLoss(nn.Module):
    """
    Box-window local cross-correlation loss (CC^2 by default, or CC), computed with separable box sums.

    Target statistics are persistently cached per module instance when ``target`` does not require grad.
    Preserves float64 precision (eliminating gradcheck truncation) and upcasts float16/bfloat16 under AMP.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        squared: bool = True,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"BoxLNCCLoss: kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        # Enforce variance floor >= 1e-6 per GEMINI.md Rule 3
        self.smooth_dr = max(float(smooth_dr), 1e-6)
        self.squared = squared
        self._target_cache = {}
        self._target_refs = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute box LNCC loss between pred (moving) and target (fixed).

        Preserves input dtype (float64 preserved for exact gradcheck; float16 upcast to float32).
        """
        work_dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
        dev_type = "cuda" if pred.is_cuda else ("mps" if pred.device.type == "mps" else "cpu")

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            pred_w = pred.to(work_dtype)
            target_w = target.to(work_dtype)

            dim = pred_w.dim() - 2
            kernel_vol = float(self.kernel_size**dim)
            k = torch.ones(self.kernel_size, dtype=work_dtype, device=pred.device)
            kernels = [k] * dim

            target_id = id(target)
            version = getattr(target, "_version", 0)
            if (
                not target.requires_grad
                and target_id in self._target_cache
                and self._target_cache[target_id][0] == version
            ):
                _, t_sum, t_var = self._target_cache[target_id]
            else:
                t_sum = separable_1d_filter(target_w, kernels)
                t2_sum = separable_1d_filter(target_w * target_w, kernels)
                t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=self.smooth_dr)
                if not target.requires_grad:
                    if len(self._target_cache) >= 4:
                        self._target_cache.clear()
                        self._target_refs.clear()
                    self._target_cache[target_id] = (version, t_sum, t_var)
                    self._target_refs[target_id] = target

            p_sum = separable_1d_filter(pred_w, kernels)
            p2_sum = separable_1d_filter(pred_w * pred_w, kernels)
            tp_sum = separable_1d_filter(target_w * pred_w, kernels)

            cross = tp_sum - p_sum * t_sum / kernel_vol
            p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=self.smooth_dr)

            if self.squared:
                ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
            else:
                ncc = cross / (torch.sqrt(t_var * p_var) + self.smooth_dr)

            if mask is not None:
                m = mask.to(work_dtype)
                spatial_dims = tuple(range(2, pred_w.dim()))
                sum_m = torch.sum(m, dim=spatial_dims, keepdim=True)
                weighted_ncc = torch.sum(ncc * m, dim=spatial_dims, keepdim=True) / torch.clamp_min(sum_m, 1e-8)
                loss = -torch.mean(weighted_ncc)
            else:
                loss = -torch.mean(ncc)

        return loss if pred.dtype in (torch.float16, torch.bfloat16) else loss.to(pred.dtype)


def box_lncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    squared: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Functional wrapper for ``BoxLNCCLoss`` (linear CC by default, squared=False).
    """
    loss_fn = BoxLNCCLoss(
        kernel_size=window_size,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        squared=squared,
    )
    return loss_fn(I, J, mask=mask)


def box_cc2_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> torch.Tensor:
    """
    Functional wrapper for ``BoxLNCCLoss`` with ``squared=True`` (CC^2).
    """
    return box_lncc_loss_nd(
        I,
        J,
        window_size=window_size,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        squared=True,
    )


def pearson_correlation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative global Pearson correlation per row: ``x`` (B, S) against ``y`` (S,) shared or (B, S) one reference per
    row -> (B,), in [-1, 1]. For intensity samples of one subject and contrast (close to linearly related)."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    x_c = x - x.mean(dim=1, keepdim=True)
    y_c = y - y.mean(dim=1, keepdim=True)
    cov = (x_c * y_c).sum(dim=1)
    x_std = torch.sqrt((x_c ** 2).sum(dim=1) + 1e-8)
    y_std = torch.sqrt((y_c ** 2).sum(dim=1) + 1e-8)
    return -(cov / (x_std * y_std + 1e-8))
