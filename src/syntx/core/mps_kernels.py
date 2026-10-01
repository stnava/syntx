"""Deterministic Metal (MPS) kernels.

``grid_sample_backward_mps`` -- the bilinear/trilinear ``grid_sample`` backward
(``align_corners=True``, ``'zeros'`` / ``'border'`` padding) with bit-reproducible output.

The stock MPS backward accumulates the input gradient with float atomics, whose result depends on
thread arrival order (and on MPS the grid gradient varies too), so identical calls differ by
~1e-7 and iterative registration amplifies that into different results. Here each sample point is
one thread: the grid gradient is summed in a fixed order in-thread, and the input-gradient
contributions are added as fixed-point integers. Integer addition is associative, so the sum is
exact and order independent. Metal has no 64-bit atomic add, so each contribution
q = round(v * scale), |q| <= 2^31, is split into limbs q = hi * 2^16 + lo (lo unsigned 16-bit,
hi signed, |hi| <= 2^15) accumulated in two int32 atomics; a voxel can take 65536 contributions
before a limb could overflow. Resolution is 2^-31 * max|grad_output|. ``scale`` is a power of two
computed on device from max|grad_output| (no host sync).
"""

import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// Power-of-two scale so every |q| = |v * scale| < 2^31 (|v| <= vmax = max|grad_output|).
inline float fixed_scale(float vmax) {
    return exp2(floor(log2(2147483392.0f / max(vmax, 1e-20f))));
}

inline void add_fixed(device atomic_uint* lo, device atomic_int* hi, uint idx, float v, float scale) {
    int q = (int)rint(v * scale);
    if (q == 0) return;
    uint a = (uint)(q & 0xFFFF);
    int b = q >> 16;
    if (a) atomic_fetch_add_explicit(lo + idx, a, memory_order_relaxed);
    if (b) atomic_fetch_add_explicit(hi + idx, b, memory_order_relaxed);
}

// out[i] = (hi[i] * 2^16 + lo[i]) / scale, NaN when max|grad_output| was not finite.
kernel void fixed_to_float(device const uint* lo [[buffer(0)]],
                           device const int* hi [[buffer(1)]],
                           device const float* scale_buf [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    const float vmax = scale_buf[0];
    const long t = ((long)hi[i] << 16) + (long)lo[i];
    out[i] = isfinite(vmax) ? (float)t / fixed_scale(vmax) : NAN;
}

inline float unnorm(float g, int size, bool border, thread float& du) {
    float u = (g + 1.0f) * 0.5f * (float)(size - 1);
    du = 0.5f * (float)(size - 1);
    if (border) {
        if (!(u > 0.0f && u < (float)(size - 1))) du = 0.0f;
        u = clamp(u, 0.0f, (float)(size - 1));
    }
    return u;
}

// dims: B, C, D, H, W, Npts, border, need_input, need_grid
kernel void gs3d_bwd(device const float* inp [[buffer(0)]],
                     device const float* grid [[buffer(1)]],
                     device const float* gout [[buffer(2)]],
                     device const float* scale_buf [[buffer(3)]],
                     device float* ggrid [[buffer(4)]],
                     device atomic_uint* lo [[buffer(5)]],
                     device atomic_int* hi [[buffer(6)]],
                     device const int* dims [[buffer(7)]],
                     uint tid [[thread_position_in_grid]]) {
    const int B = dims[0], C = dims[1], D = dims[2], H = dims[3], W = dims[4], N = dims[5];
    const bool border = dims[6] != 0, need_in = dims[7] != 0, need_grid = dims[8] != 0;
    if ((int)tid >= B * N) return;
    const int b = (int)tid / N, p = (int)tid - b * N;
    const float scale = fixed_scale(scale_buf[0]);
    float dux, duy, duz;
    const float ux = unnorm(grid[tid * 3 + 0], W, border, dux);
    const float uy = unnorm(grid[tid * 3 + 1], H, border, duy);
    const float uz = unnorm(grid[tid * 3 + 2], D, border, duz);
    const float fx0 = floor(ux), fy0 = floor(uy), fz0 = floor(uz);
    const int x0 = (int)fx0, y0 = (int)fy0, z0 = (int)fz0;
    const float fx = ux - fx0, fy = uy - fy0, fz = uz - fz0;
    const long HW = (long)H * W, DHW = (long)D * HW;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int c = 0; c < 8; ++c) {
        const int hx = c & 1, hy = (c >> 1) & 1, hz = (c >> 2) & 1;
        const int ix = x0 + hx, iy = y0 + hy, iz = z0 + hz;
        if (ix < 0 || ix >= W || iy < 0 || iy >= H || iz < 0 || iz >= D) continue;
        const float wx = hx ? fx : 1.0f - fx, wy = hy ? fy : 1.0f - fy, wz = hz ? fz : 1.0f - fz;
        const float sx = hx ? 1.0f : -1.0f, sy = hy ? 1.0f : -1.0f, sz = hz ? 1.0f : -1.0f;
        const float w = wx * wy * wz;
        const long vox = (long)iz * HW + (long)iy * W + ix;
        float gv = 0.0f;
        for (int ch = 0; ch < C; ++ch) {
            const long bc = (long)(b * C + ch);
            const float go = gout[bc * N + p];
            if (need_grid) gv += go * inp[bc * DHW + vox];
            if (need_in) add_fixed(lo, hi, (uint)(bc * DHW + vox), go * w, scale);
        }
        if (need_grid) {
            gx += gv * sx * wy * wz;
            gy += gv * wx * sy * wz;
            gz += gv * wx * wy * sz;
        }
    }
    if (need_grid) {
        ggrid[tid * 3 + 0] = gx * dux;
        ggrid[tid * 3 + 1] = gy * duy;
        ggrid[tid * 3 + 2] = gz * duz;
    }
}

// dims: B, C, 1, H, W, Npts, border, need_input, need_grid
kernel void gs2d_bwd(device const float* inp [[buffer(0)]],
                     device const float* grid [[buffer(1)]],
                     device const float* gout [[buffer(2)]],
                     device const float* scale_buf [[buffer(3)]],
                     device float* ggrid [[buffer(4)]],
                     device atomic_uint* lo [[buffer(5)]],
                     device atomic_int* hi [[buffer(6)]],
                     device const int* dims [[buffer(7)]],
                     uint tid [[thread_position_in_grid]]) {
    const int B = dims[0], C = dims[1], H = dims[3], W = dims[4], N = dims[5];
    const bool border = dims[6] != 0, need_in = dims[7] != 0, need_grid = dims[8] != 0;
    if ((int)tid >= B * N) return;
    const int b = (int)tid / N, p = (int)tid - b * N;
    const float scale = fixed_scale(scale_buf[0]);
    float dux, duy;
    const float ux = unnorm(grid[tid * 2 + 0], W, border, dux);
    const float uy = unnorm(grid[tid * 2 + 1], H, border, duy);
    const float fx0 = floor(ux), fy0 = floor(uy);
    const int x0 = (int)fx0, y0 = (int)fy0;
    const float fx = ux - fx0, fy = uy - fy0;
    const long HW = (long)H * W;
    float gx = 0.0f, gy = 0.0f;
    for (int c = 0; c < 4; ++c) {
        const int hx = c & 1, hy = (c >> 1) & 1;
        const int ix = x0 + hx, iy = y0 + hy;
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
        const float wx = hx ? fx : 1.0f - fx, wy = hy ? fy : 1.0f - fy;
        const float sx = hx ? 1.0f : -1.0f, sy = hy ? 1.0f : -1.0f;
        const long vox = (long)iy * W + ix;
        float gv = 0.0f;
        for (int ch = 0; ch < C; ++ch) {
            const long bc = (long)(b * C + ch);
            const float go = gout[bc * N + p];
            if (need_grid) gv += go * inp[bc * HW + vox];
            if (need_in) add_fixed(lo, hi, (uint)(bc * HW + vox), go * wx * wy, scale);
        }
        if (need_grid) {
            gx += gv * sx * wy;
            gy += gv * wx * sy;
        }
    }
    if (need_grid) {
        ggrid[tid * 2 + 0] = gx * dux;
        ggrid[tid * 2 + 1] = gy * duy;
    }
}
"""

_LIB = None
_DIMS = {}


def _dims(device, vals):
    """Cached int32 device tensor of kernel dimensions (avoids a host->device copy per call)."""
    key = (str(device), vals)
    t = _DIMS.get(key)
    if t is None:
        if len(_DIMS) > 256:
            _DIMS.clear()
        t = _DIMS[key] = torch.tensor(vals, dtype=torch.int32, device=device)
    return t


def _lib():
    """The compiled Metal shader library (``torch.mps.compile_shader``), compiled once."""
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SRC)
    return _LIB


def available():
    """True if MPS is available, torch has ``torch.mps.compile_shader`` and the kernels compile
    (compiles them on the first call); False on any exception."""
    try:
        return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader") and _lib() is not None
    except Exception:
        return False


def grid_sample_backward_mps(grad_out, input, grid, padding_mode, need_input=True, need_grid=True):
    """Deterministic (grad_input, grad_grid) of
    ``F.grid_sample(input, grid, 'bilinear', padding_mode, align_corners=True)`` on MPS.

    Parameters
    ----------
    grad_out : Tensor (B, C, *out_spatial)
        Gradient of the loss w.r.t. the grid_sample output.
    input : Tensor (B, C, H, W) or (B, C, D, H, W), on MPS.
    grid : Tensor (B, *out_spatial, 2 or 3)
        Normalised sampling coordinates in [-1, 1], last axis in (x, y[, z]) order as for
        ``F.grid_sample``.
    padding_mode : {'border', 'zeros'}
        'border' clamps coordinates (grid gradient 0 where clamped); 'zeros': out-of-range
        corners contribute nothing. Other values raise ValueError.
    need_input, need_grid : bool, default True
        Which gradients to compute; the other is returned as None.

    Returns
    -------
    (grad_input, grad_grid)
        Shapes of ``input`` and ``grid``, in the dtypes of ``grad_out`` and ``grid``; all
        arithmetic is float32. grad_input is fixed-point accumulated (resolution
        2^-31 * max|grad_out|; NaN everywhere if grad_out has a non-finite value). Accuracy
        requires fewer than 65536 contributions per input voxel (not checked).

    Raises
    ------
    ValueError
        If B * C * voxels >= 2^32, or for an unsupported ``padding_mode``.
    """
    if padding_mode not in ('border', 'zeros'):
        raise ValueError(f"grid_sample_backward_mps supports padding_mode 'border' or 'zeros', "
                         f"got {padding_mode!r}")
    nd = input.dim() - 2
    B, C = input.shape[:2]
    spatial = tuple(input.shape[2:])
    n_pts = grid[..., 0].numel() // B
    n_vox = 1
    for s in spatial:
        n_vox *= s
    if B * C * n_vox >= 2 ** 32:
        raise ValueError("grid_sample_backward_mps: input too large for 32-bit voxel indices")
    inp = input.detach().contiguous().float()
    g = grid.detach().contiguous().float()
    go = grad_out.detach().contiguous().float()
    D, H, W = (spatial if nd == 3 else (1,) + spatial)
    dims = _dims(input.device, (B, C, D, H, W, n_pts, int(padding_mode == 'border'), int(need_input), int(need_grid)))
    # max|grad_out| -> the kernels derive the fixed-point scale from it (and propagate NaN/inf).
    scale_buf = go.abs().amax().reshape(1)
    ggrid = torch.empty(g.shape if need_grid else (1,), dtype=torch.float32, device=input.device)
    n_acc = B * C * n_vox if need_input else 1
    acc = torch.zeros(2 * n_acc, dtype=torch.int32, device=input.device)
    lo, hi = acc[:n_acc], acc[n_acc:]
    kern = _lib().gs3d_bwd if nd == 3 else _lib().gs2d_bwd
    kern(inp, g, go, scale_buf, ggrid, lo, hi, dims, threads=B * n_pts)
    grad_input = grad_grid = None
    if need_input:
        out = torch.empty(n_acc, dtype=torch.float32, device=input.device)
        _lib().fixed_to_float(lo, hi, scale_buf, out, threads=n_acc)
        grad_input = out.view(input.shape).to(grad_out.dtype)
    if need_grid:
        grad_grid = ggrid.view(grid.shape).to(grid.dtype)
    return grad_input, grad_grid
