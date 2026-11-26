"""
Canonical Mattes Mutual Information similarity loss family.

Implements cubic B-spline Parzen windowing, exact 4-tap compact polynomial evaluation,
deterministic chunked accumulation for GPU/MPS/CPU reproducibility, analytical closed-form
pointwise Parzen gradients, boundary padding (pad=2.0) for exact partition of unity,
and support for fixed histogram bounds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def b_spline_3(x: torch.Tensor) -> torch.Tensor:
    """
    Cubic B-spline B3(x), elementwise:
    2/3 - x^2 + |x|^3 / 2 for |x| < 1, (2 - |x|)^3 / 6 for 1 <= |x| < 2, else 0.
    """
    abs_x = torch.abs(x)
    x2 = abs_x * abs_x
    y1 = (2.0 / 3.0) - x2 + 0.5 * x2 * abs_x
    rem = 2.0 - abs_x
    y2 = (1.0 / 6.0) * (rem * rem * rem)
    return torch.where(abs_x < 1.0, y1, torch.where(abs_x < 2.0, y2, torch.zeros_like(x)))


def b_spline_3_deriv(x: torch.Tensor) -> torch.Tensor:
    """
    First derivative of cubic B-spline B3(x), elementwise:
    -2x + 1.5 x |x| for |x| < 1, -0.5 (2 - |x|)^2 sgn(x) for 1 <= |x| < 2, else 0.
    """
    abs_x = torch.abs(x)
    s = torch.sign(x)
    d1 = -2.0 * x + 1.5 * x * abs_x
    rem = 2.0 - abs_x
    d2 = -0.5 * (rem * rem) * s
    return torch.where(abs_x < 1.0, d1, torch.where(abs_x < 2.0, d2, torch.zeros_like(x)))


def get_4tap_splines(
    v: torch.Tensor,
    num_bins: int = 32,
    min_val: float = -1.0,
    max_val: float = 1.0,
    pad: float = 2.0,
):
    """
    Compute 4-tap compact cubic B-spline bin indices, weights, and analytical derivatives.

    Cubic B-splines have compact support radius of 2. For any intensity sample, at most
    4 consecutive bins have non-zero weight. Evaluates non-zero taps in O(1) polynomial operations.
    """
    scale = (float(num_bins - 1) - 2.0 * pad) / (max_val - min_val)
    fin = torch.isfinite(v)
    v_clean = torch.clamp(torch.nan_to_num(v, nan=0.0), min_val, max_val)
    u = pad + (v_clean - min_val) * scale
    k0 = torch.floor(u).long()
    d = (u - k0.to(v.dtype)).unsqueeze(-1)

    b0 = torch.clamp(k0 - 1, 0, num_bins - 1)
    b1 = torch.clamp(k0, 0, num_bins - 1)
    b2 = torch.clamp(k0 + 1, 0, num_bins - 1)
    b3 = torch.clamp(k0 + 2, 0, num_bins - 1)
    bins = torch.stack([b0, b1, b2, b3], dim=-1)

    omd = 1.0 - d
    w0 = (1.0 / 6.0) * (omd * omd * omd)
    w1 = (2.0 / 3.0) - d * d + 0.5 * d * d * d
    w2 = (1.0 / 6.0) + 0.5 * d + 0.5 * d * d - 0.5 * d * d * d
    w3 = (1.0 / 6.0) * (d * d * d)
    weights = torch.cat([w0, w1, w2, w3], dim=-1) * fin.unsqueeze(-1).to(v.dtype)

    clamp_mask = (v >= min_val) & (v <= max_val) & fin
    dw0 = -0.5 * (omd * omd)
    dw1 = -2.0 * d + 1.5 * d * d
    dw2 = 0.5 + d - 1.5 * d * d
    dw3 = 0.5 * (d * d)
    dweights = torch.cat([dw0, dw1, dw2, dw3], dim=-1) * (scale * clamp_mask.unsqueeze(-1).to(v.dtype))

    return bins, weights, dweights


class CanonicalMattesMIFunction(torch.autograd.Function):
    """
    Canonical high-performance Mattes Mutual Information engine.

    Features:
    1. Compact 4-tap evaluation avoiding multi-gigabyte dense [N, B] tensor allocations.
    2. Deterministic chunked matrix accumulation for bitwise reproducibility on MPS/CUDA/CPU.
    3. Exact analytical pointwise Parzen gradient backward pass.
    4. Float32/Float64 accumulation to prevent Float16 AMP overflow while supporting double gradchecks.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        num_bins: int = 32,
        min_val: float = -1.0,
        max_val: float = 1.0,
        pad: float = 2.0,
        chunk_size: int = 32768,
        fixed_weights: torch.Tensor | None = None,
    ):
        scale = (float(num_bins - 1) - 2.0 * pad) / (max_val - min_val)
        ctx.num_bins = num_bins
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.pad = pad
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        N = x.numel()
        ctx.N = N

        ctx.save_for_backward(x, y, fixed_weights)

        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        H = torch.zeros(num_bins, num_bins, device=x.device, dtype=work_dtype)

        dev_type = "cuda" if x.is_cuda else ("mps" if x.device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                C = end - start
                xc = x[start:end]
                bx, wx, _ = get_4tap_splines(xc, num_bins, min_val, max_val, pad=pad)
                mat_x = torch.zeros(C, num_bins, device=x.device, dtype=work_dtype)
                mat_x.scatter_(1, bx, wx.to(work_dtype))

                if fixed_weights is not None:
                    mat_y = fixed_weights[start:end].to(work_dtype)
                else:
                    yc = y[start:end]
                    by, wy, _ = get_4tap_splines(yc, num_bins, min_val, max_val, pad=pad)
                    mat_y = torch.zeros(C, num_bins, device=y.device, dtype=work_dtype)
                    mat_y.scatter_(1, by, wy.to(work_dtype))

                H = H + torch.mm(mat_x.t(), mat_y)

            total_sum = H.sum() + 1e-12
            pxy = H / total_sum
            px = pxy.sum(dim=1, keepdim=True)
            py = pxy.sum(dim=0, keepdim=True)
            ratio = pxy / (px * py + 1e-12)
            loss = -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-12)))

        ctx.ratio = ratio
        ctx.total_sum = total_sum
        return loss if x.dtype in (torch.float16, torch.bfloat16) else loss.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y, fixed_weights = ctx.saved_tensors
        num_bins = ctx.num_bins
        min_val = ctx.min_val
        max_val = ctx.max_val
        pad = ctx.pad
        chunk_size = ctx.chunk_size
        N = ctx.N
        ratio = ctx.ratio
        total_sum = ctx.total_sum

        grad_x = None
        grad_y = None

        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        M = (-(1.0 + torch.log(torch.clamp(ratio, min=1e-12))) / total_sum) * grad_output.to(work_dtype)

        need_x = ctx.needs_input_grad[0]
        need_y = ctx.needs_input_grad[1]

        if need_x:
            grad_x = torch.empty(N, device=x.device, dtype=x.dtype)
        if need_y:
            grad_y = torch.empty(N, device=y.device, dtype=y.dtype)

        dev_type = "cuda" if x.is_cuda else ("mps" if x.device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                C = end - start
                xc = x[start:end]
                bx, wx, dwx = get_4tap_splines(xc, num_bins, min_val, max_val, pad=pad)

                if fixed_weights is not None:
                    mat_y = fixed_weights[start:end].to(work_dtype)
                else:
                    yc = y[start:end]
                    by, wy, dwy = get_4tap_splines(yc, num_bins, min_val, max_val, pad=pad)
                    mat_y = torch.zeros(C, num_bins, device=y.device, dtype=work_dtype)
                    mat_y.scatter_(1, by, wy.to(work_dtype))

                if need_x:
                    dmat_x = torch.zeros(C, num_bins, device=x.device, dtype=work_dtype)
                    dmat_x.scatter_(1, bx, dwx.to(work_dtype))
                    W_y_M = torch.mm(mat_y, M.t())
                    grad_x[start:end] = torch.sum(dmat_x * W_y_M, dim=1).to(x.dtype)

                if need_y:
                    mat_x = torch.zeros(C, num_bins, device=x.device, dtype=work_dtype)
                    mat_x.scatter_(1, bx, wx.to(work_dtype))
                    dmat_y = torch.zeros(C, num_bins, device=y.device, dtype=work_dtype)
                    dmat_y.scatter_(1, by, dwy.to(work_dtype))
                    W_x_M = torch.mm(mat_x, M)
                    grad_y[start:end] = torch.sum(dmat_y * W_x_M, dim=1).to(y.dtype)

        return grad_x, grad_y, None, None, None, None, None, None


class BatchedMattesMIFunction(torch.autograd.Function):
    """Batched 4-tap analytical Mattes Mutual Information for ``M`` moving sample vectors
    ``x_batch`` ``(M, N)`` against one shared ``fixed_weights`` ``(N, num_bins)``, returning ``(M,)``."""

    @staticmethod
    def forward(
        ctx,
        x_batch: torch.Tensor,
        fixed_weights: torch.Tensor,
        num_bins: int = 32,
        min_val: float = -1.0,
        max_val: float = 1.0,
        pad: float = 2.0,
        chunk_size: int = 16384,
    ):
        M_sz, N = int(x_batch.shape[0]), int(x_batch.shape[1])
        ctx.num_bins = num_bins
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.pad = pad
        ctx.chunk_size = chunk_size
        ctx.M_sz = M_sz
        ctx.N = N
        ctx.save_for_backward(x_batch, fixed_weights)

        work_dtype = torch.float64 if x_batch.dtype == torch.float64 else torch.float32
        H = torch.zeros((M_sz, num_bins, num_bins), device=x_batch.device, dtype=work_dtype)

        dev_type = "cuda" if x_batch.is_cuda else ("mps" if x_batch.device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                C = end - start
                xc = x_batch[:, start:end]
                bx, wx, _ = get_4tap_splines(xc, num_bins, min_val, max_val, pad=pad)
                mat_x = torch.zeros((M_sz, C, num_bins), device=x_batch.device, dtype=work_dtype)
                mat_x.scatter_(2, bx, wx.to(work_dtype))
                mat_y = fixed_weights[start:end].to(work_dtype)
                H = H + torch.matmul(mat_x.transpose(1, 2), mat_y)

            total_sum = H.sum(dim=(1, 2), keepdim=True) + 1e-12
            pxy = H / total_sum
            px = pxy.sum(dim=2, keepdim=True)
            py = pxy.sum(dim=1, keepdim=True)
            ratio = pxy / (px * py + 1e-12)
            losses = -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-12)), dim=(1, 2))

        ctx.ratio = ratio
        ctx.total_sum = total_sum
        return losses if x_batch.dtype in (torch.float16, torch.bfloat16) else losses.to(x_batch.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_batch, fixed_weights = ctx.saved_tensors
        num_bins = ctx.num_bins
        min_val = ctx.min_val
        max_val = ctx.max_val
        pad = ctx.pad
        chunk_size = ctx.chunk_size
        M_sz = ctx.M_sz
        N = ctx.N
        ratio = ctx.ratio
        total_sum = ctx.total_sum

        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None

        work_dtype = torch.float64 if x_batch.dtype == torch.float64 else torch.float32
        M_mat = (-(1.0 + torch.log(torch.clamp(ratio, min=1e-12))) / total_sum) * grad_output.to(work_dtype).view(
            M_sz, 1, 1
        )
        grad_x = torch.empty((M_sz, N), device=x_batch.device, dtype=x_batch.dtype)

        dev_type = "cuda" if x_batch.is_cuda else ("mps" if x_batch.device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                C = end - start
                xc = x_batch[:, start:end]
                bx, _, dwx = get_4tap_splines(xc, num_bins, min_val, max_val, pad=pad)
                dmat_x = torch.zeros((M_sz, C, num_bins), device=x_batch.device, dtype=work_dtype)
                dmat_x.scatter_(2, bx, dwx.to(work_dtype))
                mat_y = fixed_weights[start:end].to(work_dtype)
                W_y_M = torch.matmul(M_mat, mat_y.t()).transpose(1, 2)
                grad_x[:, start:end] = torch.sum(dmat_x * W_y_M, dim=2).to(x_batch.dtype)

        return grad_x, None, None, None, None, None, None


def _extract_sparse4_from_fixed_weights(fixed_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and cache the 4 non-zero Parzen bins ``(fi_idx4, fi_w4)`` from ``(..., num_bins)``."""
    cached = getattr(fixed_weights, "_cached_sparse4", None)
    ver = getattr(fixed_weights, "_version", -1)
    if cached is not None and cached[0] == ver:
        return cached[1], cached[2]
    with torch.no_grad():
        fw_f = fixed_weights.to(torch.float32)
        k_sz = int(fw_f.shape[-1])
        if k_sz >= 4:
            w4, idx4_l = torch.topk(fw_f, k=4, dim=-1, sorted=False)
            fi_idx4 = idx4_l.to(torch.int32).contiguous()
            fi_w4 = w4.contiguous()
        else:
            lead = fw_f.shape[:-1]
            fi_idx4 = torch.zeros((*lead, 4), device=fw_f.device, dtype=torch.int32)
            fi_w4 = torch.zeros((*lead, 4), device=fw_f.device, dtype=torch.float32)
            fi_idx4[..., :k_sz] = torch.arange(k_sz, device=fw_f.device, dtype=torch.int32)
            fi_w4[..., :k_sz] = fw_f
    try:
        fixed_weights._cached_sparse4 = (ver, fi_idx4, fi_w4)
    except AttributeError:
        pass
    return fi_idx4, fi_w4


def batched_mattes_mi_loss_shared_fixed(
    I_batch: torch.Tensor,
    fixed_weights: torch.Tensor,
    num_bins: int = 32,
    fixed_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Vectorized negative Mattes MI for ``(M, P)`` moving samples against precomputed
    shared ``fixed_weights`` ``(P, num_bins)``, returning ``(M,)`` losses."""
    if I_batch.dtype in (torch.float16, torch.bfloat16):
        I_batch = I_batch.to(torch.float32)
    min_i, max_i = float(fixed_range[0]), float(fixed_range[1])
    x_scaled = (I_batch - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
    if x_scaled.dtype == torch.float32 and int(num_bins) >= 6:
        from ..kernels import RegisterParzenMattesMIFunction, has_mps_shader_backend

        if has_mps_shader_backend(x_scaled.device):
            fi_idx4, fi_w4 = _extract_sparse4_from_fixed_weights(fixed_weights)
            return RegisterParzenMattesMIFunction.apply(x_scaled, fi_idx4, fi_w4, int(num_bins), -1.0, 1.0, 2.0)
    chunk_sz = max(2048, min(16384, 65536 // max(1, int(I_batch.shape[0]))))
    return BatchedMattesMIFunction.apply(x_scaled, fixed_weights, num_bins, -1.0, 1.0, 2.0, chunk_sz)


def _parzen_joint_histogram(w_x: torch.Tensor, w_y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """
    Joint Parzen histogram H = w_x^T w_y accumulated as a batch of chunk-row blocks (torch.bmm).

    Deterministic across allocations on Apple Silicon MPS and CUDA.
    """
    n, nb = w_x.shape
    pad = (-n) % chunk
    if pad:
        z = torch.zeros(pad, nb, dtype=w_x.dtype, device=w_x.device)
        w_x = torch.cat([w_x, z])
        w_y = torch.cat([w_y, z])
    a = w_x.view(-1, chunk, nb).transpose(1, 2)
    b = w_y.view(-1, chunk, nb)
    return torch.bmm(a, b).sum(dim=0)


def parzen_weights(
    v: torch.Tensor,
    num_bins: int = 32,
    min_val: float = -1.0,
    max_val: float = 1.0,
    pad: float = 2.0,
) -> torch.Tensor:
    """Cubic B-spline Parzen weights of intensity samples, shape [N, num_bins], float32."""
    v = v.float().reshape(-1)
    bins, weights, _ = get_4tap_splines(v, num_bins=num_bins, min_val=min_val, max_val=max_val, pad=pad)
    out = torch.zeros(v.numel(), num_bins, device=v.device, dtype=torch.float32)
    out.scatter_(1, bins, weights)
    return out


_default_parzen_weights = parzen_weights


def mattes_mi_from_weights(w_x: torch.Tensor, w_y: torch.Tensor) -> torch.Tensor:
    """Negative mutual information from two [N, B] Parzen weight matrices."""
    joint_hist = _parzen_joint_histogram(w_x, w_y)
    pxy = joint_hist / (joint_hist.sum() + 1e-8)
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)))


def batched_mattes_mi_from_weights(w_x_batch: torch.Tensor, w_y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Per-frame negative mutual information from Parzen weights: (B, S, nbins) frames against one shared reference
    (S, nbins) or one reference per frame (B, S, nbins) -> (B,). Same estimator as ``mattes_mi_from_weights`` (one
    pair); the joint histograms are accumulated in zero-padded blocks of ``chunk`` samples."""
    B, S, nb = w_x_batch.shape
    if w_y.dim() == 2:
        w_y = w_y.unsqueeze(0)
    pad = (-S) % chunk
    if pad:
        w_x_batch = F.pad(w_x_batch, (0, 0, 0, pad))
        w_y_p = F.pad(w_y, (0, 0, 0, pad))
    else:
        w_y_p = w_y
    a = w_x_batch.view(B, -1, chunk, nb).permute(0, 1, 3, 2)
    b = w_y_p.view(w_y_p.shape[0], -1, chunk, nb).expand(B, -1, -1, -1)
    joint = torch.matmul(a, b).sum(dim=1)
    pxy = joint / (joint.sum(dim=(1, 2), keepdim=True) + 1e-8)
    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)), dim=(1, 2))


_MATTES_SAMPLE_IDX_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def mattes_sample_indices(n: int, sampling_percentage: float, device: torch.device | None = None) -> torch.Tensor:
    """Indices of regular Mattes subsample with exact integer rounding identical across devices."""
    k = max(1, round(float(sampling_percentage) * n))
    tgt_dev = torch.device("cpu") if device is None else torch.device(device)
    if k == 1:
        return torch.zeros(1, dtype=torch.long, device=tgt_dev)
    cache_ok = k <= 2_000_000
    if cache_ok:
        key = (int(n), int(k), str(tgt_dev))
        cached = _MATTES_SAMPLE_IDX_CACHE.get(key)
        if cached is not None:
            return cached
    i = torch.arange(k, dtype=torch.long, device=tgt_dev)
    idx = torch.div(2 * i * (n - 1) + (k - 1), 2 * (k - 1), rounding_mode="floor")
    if cache_ok:
        while len(_MATTES_SAMPLE_IDX_CACHE) >= 32:
            _MATTES_SAMPLE_IDX_CACHE.pop(next(iter(_MATTES_SAMPLE_IDX_CACHE)))
        _MATTES_SAMPLE_IDX_CACHE[key] = idx
    return idx


def mattes_mi_loss_core(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_bins: int = 32,
    min_val: float = -1.0,
    max_val: float = 1.0,
    sampling_percentage: float | None = None,
    fixed_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Negative Mattes mutual information of intensities already scaled to [min_val, max_val].

    Evaluated using ``CanonicalMattesMIFunction`` with exact 4-tap compact splines and analytical gradients.
    """
    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if sampling_percentage is not None and sampling_percentage < 1.0 and x.numel() > 0:
        idx = mattes_sample_indices(x.numel(), sampling_percentage, x.device)
        x = x[idx]
        y = y[idx]

    if x.numel() == 0:
        return I.sum() * 0.0 + J.sum() * 0.0
    if fixed_weights is not None and fixed_weights.shape[0] != x.numel():
        raise ValueError(f"fixed_weights has {fixed_weights.shape[0]} rows for {x.numel()} samples")

    if parzen_weights is not _default_parzen_weights:
        w_x = parzen_weights(x, num_bins, min_val, max_val)
        w_y = fixed_weights if fixed_weights is not None else parzen_weights(y, num_bins, min_val, max_val)
        return mattes_mi_from_weights(w_x, w_y)

    if (
        x.dtype == torch.float32
        and not y.requires_grad
        and int(num_bins) >= 6
        and x.numel() >= 256
    ):
        from ..kernels import RegisterParzenMattesMIFunction, has_mps_shader_backend, prepack_fixed_parzen_sparse4

        if has_mps_shader_backend(x.device):
            if fixed_weights is not None:
                fi_idx4, fi_w4 = _extract_sparse4_from_fixed_weights(fixed_weights)
            else:
                fi_idx4, fi_w4, _ = prepack_fixed_parzen_sparse4(
                    y.detach(), num_bins=int(num_bins), min_val=float(min_val), max_val=float(max_val), pad=2.0
                )
            return RegisterParzenMattesMIFunction.apply(
                x.unsqueeze(0), fi_idx4, fi_w4, int(num_bins), float(min_val), float(max_val), 2.0
            ).squeeze(0)

    return CanonicalMattesMIFunction.apply(x, y, num_bins, min_val, max_val, 2.0, 32768, fixed_weights)


def mattes_mi_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_bins: int = 32,
    sampling_percentage: float | None = None,
    auto_mask: bool = True,
    fixed_range: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] | None = (0.0, 1.0),
    fixed_weights: torch.Tensor | None = None,
    return_per_batch: bool = False,
) -> torch.Tensor:
    """
    Negative Mattes mutual information of images I (moving) and J (fixed), any dimension.

    Parameters
    ----------
    I, J : torch.Tensor
        Images of matching shape.
    mask : torch.Tensor, optional
        Foreground selection mask.
    num_bins : int, default 32
    sampling_percentage : float, optional
    auto_mask : bool, default True
        Filter out background |I| < 1% of max.
    fixed_range : tuple of float, default (0.0, 1.0)
        Histogram bounds. If None, computed dynamically from selected samples.
    fixed_weights : torch.Tensor, optional
        Precomputed fixed-image Parzen weights.

    Returns
    -------
    torch.Tensor
        Scalar similarity loss (nats).
    """
    # Multi-channel stacks [B, C, *spatial]: one MI per channel, mean-reduced
    # (the [B,1,...] mask is shared by all channels, as in the other metrics).
    if I.ndim >= 4 and I.shape[1] > 1:
        if fixed_weights is not None:
            raise ValueError("fixed_weights is not supported for multi-channel (C>1) Mattes MI")
        losses = [
            mattes_mi_loss_nd(
                I[:, c : c + 1],
                J[:, c : c + 1],
                mask=mask,
                num_bins=num_bins,
                sampling_percentage=sampling_percentage,
                auto_mask=auto_mask,
                fixed_range=fixed_range,
            )
            for c in range(I.shape[1])
        ]
        return torch.stack(losses).mean()

    if I.dtype in (torch.float16, torch.bfloat16):
        I = I.to(torch.float32)
        J = J.to(torch.float32)

    # Vectorized multi-subject / multi-candidate evaluation when B > 1 against shared J (B_J == 1)
    # or when explicit per-batch losses are requested.
    if I.ndim >= 4 and I.shape[0] > 1 and (return_per_batch or J.shape[0] == 1):
        from ..kernels import RegisterParzenMattesMIFunction, prepack_fixed_parzen_sparse4

        b_sz = int(I.shape[0])
        x_flat = I.reshape(b_sz, -1)
        y_flat = J.reshape(int(J.shape[0]), -1)
        if mask is not None and int(mask.shape[0]) == 1:
            valid_1d = (mask.reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(-1)
            x_flat = x_flat.index_select(1, valid_1d)
            y_flat = y_flat.index_select(1, valid_1d)
        if sampling_percentage is not None and sampling_percentage < 1.0 and x_flat.shape[1] > 0:
            s_idx = mattes_sample_indices(int(x_flat.shape[1]), sampling_percentage, x_flat.device)
            x_flat = x_flat.index_select(1, s_idx)
            y_flat = y_flat.index_select(1, s_idx)
        if x_flat.shape[1] == 0:
            z_b = I.reshape(b_sz, -1).sum(dim=1) * 0.0
            return z_b if return_per_batch else z_b.sum()
        if fixed_range is None:
            min_i = x_flat.amin(dim=1, keepdim=True).detach()
            max_i = x_flat.amax(dim=1, keepdim=True).detach()
            min_j = y_flat.amin(dim=1, keepdim=True).detach()
            max_j = y_flat.amax(dim=1, keepdim=True).detach()
        else:
            fr_b = fixed_range if not isinstance(fixed_range[0], (int, float)) else (fixed_range, fixed_range)
            min_i, max_i = float(fr_b[0][0]), float(fr_b[0][1])
            min_j, max_j = float(fr_b[1][0]), float(fr_b[1][1])
        x_sc_b = (x_flat - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
        y_sc_b = (y_flat - min_j) / (max_j - min_j + 1e-8) * 2.0 - 1.0
        fi_i4, fi_w4_b, _ = prepack_fixed_parzen_sparse4(
            y_sc_b.squeeze(0) if int(y_sc_b.shape[0]) == 1 else y_sc_b,
            num_bins=int(num_bins),
            min_val=-1.0,
            max_val=1.0,
            pad=2.0,
        )
        losses_b = RegisterParzenMattesMIFunction.apply(x_sc_b, fi_i4, fi_w4_b, int(num_bins), -1.0, 1.0, 2.0, None)
        return losses_b if return_per_batch else losses_b.mean()

    if fixed_weights is not None and (fixed_range is None or auto_mask):
        raise ValueError("fixed_weights needs fixed_range and auto_mask=False (a fixed sample set)")
    if auto_mask:
        fg_mask = (I.abs() > 0.01 * I.detach().abs().max()) | (J.abs() > 0.01 * J.detach().abs().max())
        if mask is not None:
            mask = (mask > 0.5) & fg_mask
        else:
            mask = fg_mask

    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if x.numel() == 0:
        return I.sum() * 0.0 + J.sum() * 0.0

    if fixed_range is None:
        min_i, max_i = x.min().detach(), x.max().detach()
        min_j, max_j = y.min().detach(), y.max().detach()
    else:
        fr = fixed_range
        if isinstance(fr[0], (int, float)):
            fr = (fr, fr)
        min_i, max_i = float(fr[0][0]), float(fr[0][1])
        min_j, max_j = float(fr[1][0]), float(fr[1][1])

    x_scaled = (x - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
    y_scaled = (y - min_j) / (max_j - min_j + 1e-8) * 2.0 - 1.0

    return mattes_mi_loss_core(
        x_scaled,
        y_scaled,
        mask=None,
        num_bins=num_bins,
        min_val=-1.0,
        max_val=1.0,
        sampling_percentage=sampling_percentage,
        fixed_weights=fixed_weights,
    )
