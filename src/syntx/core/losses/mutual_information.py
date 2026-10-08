"""
Canonical Mattes Mutual Information similarity loss family.

Implements cubic B-spline Parzen windowing, exact 4-tap compact polynomial evaluation,
deterministic chunked accumulation for GPU/MPS/CPU reproducibility, analytical closed-form
pointwise Parzen gradients, boundary padding (pad=2.0) for exact partition of unity,
and support for fixed histogram bounds.
"""

from __future__ import annotations

import torch


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


def mattes_sample_indices(n: int, sampling_percentage: float, device: torch.device | None = None) -> torch.Tensor:
    """Indices of regular Mattes subsample with exact integer rounding identical across devices."""
    k = max(1, round(float(sampling_percentage) * n))
    if k == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    i = torch.arange(k, dtype=torch.long)
    return ((2 * i * (n - 1) + (k - 1)) // (2 * (k - 1))).to(device)


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
