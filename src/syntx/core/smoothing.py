"""
Smoothing operators for images, gradients and velocity / displacement fields.

Fields are channel-last, (B, *spatial, dim), spatial axes in tensor order (z, y, x); spacing
arguments are in ANTs (x, y, z) order and reversed internally.

- Gaussian: ``separable_gaussian_filter`` (ITK-style discrete Gaussian from modified Bessel
  functions, ``get_cached_gaussian_kernel_1d``; replicate / constant / reflect padding) and
  ``fast_separable_gaussian_filter`` (compact erf kernel, ``gaussian_1d_compact``, zero
  padding); ``separable_1d_filter`` applies any separable 1-D kernels with zero padding.
- Sobolev Green's operator K = (1 + alpha |k|^2)^-s: ``apply_sobolev_green_operator`` (FFT,
  periodic boundary) and its energy ``sobolev_energy``; ``apply_dsti_green_operator`` /
  ``smooth_displacement_field_dst`` (DST-I, zero Dirichlet boundary) with a thread-safe LRU
  cache of the spectral filters.
- ``smooth_displacement_field_bspline``: B-spline least-squares fit via ANTsTorch (optional).
- ``get_boundary_mask``: 0 on a rim of border voxels, 1 inside.
"""

import collections
import math
import threading
from typing import Optional, Sequence, Union, Literal
import torch
import torch.nn.functional as F
import numpy as np

_gaussian_kernel_cache = {}
_tensor_kernel_cache = {}

def get_cached_gaussian_kernel_1d(sig: float, device, dtype):
    """Discrete Gaussian kernel (ITK style) for standard deviation ``sig`` voxels, as a
    (1, 1, 2r + 1) tensor on ``device`` / ``dtype``.

    Taps are exp(-t) I_|n|(t) (``scipy.special.ive``) with t = sig^2, normalised to sum 1; the
    radius r is the first n with tap value <= 0.005 (before normalisation). ``sig`` is rounded
    to 5 decimals. Cached per (sigma, device, dtype) for the life of the process (no eviction).
    """
    sig_key = round(float(sig), 5)
    cache_key = (sig_key, str(device), str(dtype))
    if len(_tensor_kernel_cache) > 512:          # bounded: many distinct sigmas / devices
        _tensor_kernel_cache.clear()
        _gaussian_kernel_cache.clear()
    if cache_key not in _tensor_kernel_cache:
        if sig_key not in _gaussian_kernel_cache:
            from scipy.special import ive
            variance = float(sig_key)**2
            radius = 0
            while ive(radius, variance) > 0.005:
                radius += 1
            offsets = np.arange(-radius, radius + 1)
            k_np = np.array([ive(abs(k), variance) for k in offsets], dtype=np.float32)
            k_np /= k_np.sum()
            _gaussian_kernel_cache[sig_key] = k_np
        k_np = _gaussian_kernel_cache[sig_key]
        _tensor_kernel_cache[cache_key] = torch.from_numpy(k_np).to(device=device, dtype=dtype).view(1, 1, -1)
    return _tensor_kernel_cache[cache_key]


def gaussian_1d_compact(
    sigma: float,
    truncated: float = 2.0,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """1-D Gaussian kernel integrated over unit bins (erf differences), normalised to sum 1.

    Radius = round(max(sigma * truncated, 0.5)) (at least 1 tap each side), so the kernel has
    2 * radius + 1 taps. ``sigma`` in voxels. Returns a 1-D tensor, or None if sigma <= 0.
    """
    if sigma <= 0.0:
        return None
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=dtype, device=device)
    t = 0.70710678 / float(sigma)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf()).clamp(min=0)
    return out / out.sum()


def separable_1d_filter(x: torch.Tensor, kernels: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    """
    Separable 1-D filtering of a channel-first tensor with zero padding (output = input shape).

    ``x`` is (B, C, H, W) with 2 kernels or (B, C, D, H, W) with 3; ``kernels[d]`` filters
    spatial axis d (tensor order). Each kernel is any shape flattened to 1-D, of odd length
    (an even length breaks the reshape), on x's device / dtype. None, or a single tap of value
    1, skips the axis. Applied as cross-correlation (``F.conv1d``), the same as convolution for
    symmetric kernels. ValueError unless there is one kernel per spatial axis (2 or 3).
    """
    spatial_dims = len(kernels)
    if spatial_dims not in (2, 3) or x.dim() != spatial_dims + 2:
        raise ValueError(f"separable_1d_filter: need one kernel per spatial axis of a 4-D / 5-D "
                         f"tensor; got {spatial_dims} kernels for shape {tuple(x.shape)}")
    for d in range(spatial_dims):
        k = kernels[d]
        if k is None:
            continue
        k = k.view(1, 1, -1)
        if k.shape[-1] == 1 and k.squeeze() == 1.0:
            continue
        if spatial_dims == 3:
            if d == 0:   # Depth (dim 2)
                B, C, D, H, W = x.shape
                x = x.permute(0, 1, 3, 4, 2).reshape(B * C * H * W, 1, D)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, H, W, D).permute(0, 1, 4, 2, 3)
            elif d == 1: # Height (dim 3)
                B, C, D, H, W = x.shape
                x = x.permute(0, 1, 2, 4, 3).reshape(B * C * D * W, 1, H)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, D, W, H).permute(0, 1, 2, 4, 3)
            elif d == 2: # Width (dim 4)
                B, C, D, H, W = x.shape
                x = x.reshape(B * C * D * H, 1, W)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, D, H, W)
        elif spatial_dims == 2:
            if d == 0:   # Height (dim 2)
                B, C, H, W = x.shape
                x = x.permute(0, 1, 3, 2).reshape(B * C * W, 1, H)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, W, H).permute(0, 1, 3, 2)
            elif d == 1: # Width (dim 3)
                B, C, H, W = x.shape
                x = x.reshape(B * C * H, 1, W)
                pad = k.shape[-1] // 2
                x = F.conv1d(x, k, padding=pad).view(B, C, H, W)
    return x


def _resolve_sigmas(sigma, spacing, sigma_mode, num_spatial):
    """Per-axis sigmas (voxels, tensor order) for the Gaussian filters; ValueError for an
    unknown ``sigma_mode``, a wrong-length sigma sequence, a sequence with
    ``sigma_mode='physical'``, a non-positive spacing, or a ``spacing`` that would be
    ignored (voxel mode)."""
    if sigma_mode not in ('voxel', 'physical'):
        raise ValueError(f"sigma_mode must be 'voxel' or 'physical', got {sigma_mode!r}")
    if isinstance(sigma, (tuple, list)):
        if sigma_mode == 'physical':
            raise ValueError("a per-axis sigma sequence is in voxels: use sigma_mode='voxel'")
        if len(sigma) != num_spatial:
            raise ValueError(f"need one sigma per spatial axis ({num_spatial}), got {len(sigma)}")
        return [float(x) for x in sigma]
    if sigma_mode == 'physical':
        if spacing is None:
            raise ValueError("sigma_mode='physical' needs spacing")
        if any(not float(sp) > 0 for sp in spacing):
            raise ValueError(f"spacing must be positive, got {list(spacing)}")
        return [float(np.clip(float(sigma) / sp, 0.5, 10.0)) for sp in tuple(reversed(spacing))]
    if spacing is not None:
        raise ValueError("spacing is only used with sigma_mode='physical' (sigma in mm); in voxel "
                         "mode it would be ignored -- drop it or pass sigma_mode='physical'")
    return [float(sigma)] * num_spatial


def fast_separable_gaussian_filter(
    grid: torch.Tensor,
    sigma: Union[float, Sequence[float]],
    spacing: Optional[Sequence[float]] = None,
    sigma_mode: str = 'voxel',
    truncated: float = 2.0,
) -> torch.Tensor:
    """
    Separable Gaussian smoothing of a channel-last field with compact erf kernels
    (``gaussian_1d_compact``) and ZERO padding, so values are pulled toward 0 near the border.

    Parameters
    ----------
    grid : Tensor (B, *spatial, C), 2-D or 3-D spatial, any C.
    sigma : float or sequence
        Voxels. A sequence gives one sigma per spatial axis in tensor order (z, y, x) and is
        always taken in voxels.
    spacing : sequence, optional
        ANTs (x, y, z) order, required by and only allowed with ``sigma_mode='physical'`` and a
        scalar sigma: per-axis sigma = sigma / spacing, clipped to [0.5, 10] voxels. (Invalid
        combinations raise ValueError.)
    sigma_mode : {'voxel', 'physical'}, default 'voxel'
    truncated : float, default 2.0
        Kernel radius in sigmas.

    Returns
    -------
    Tensor of the input shape (contiguous); ``grid`` itself if every sigma <= 0. Axes with
    sigma <= 0 are not filtered.
    """
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)

    sigma_list = _resolve_sigmas(sigma, spacing, sigma_mode, num_spatial)

    if all(s <= 0.0 for s in sigma_list):
        return grid

    v = torch.movedim(grid, -1, 1)
    kernels = [gaussian_1d_compact(s, truncated=truncated, device=device, dtype=dtype) for s in sigma_list]
    v_filtered = separable_1d_filter(v, kernels)
    return torch.movedim(v_filtered, 1, -1).contiguous()


def separable_gaussian_filter(
    grid: torch.Tensor,
    sigma,
    spacing=None,
    sigma_mode='voxel',
    mode: str = 'replicate',
    kernel_type: str = 'bessel',
) -> torch.Tensor:
    """
    Separable Gaussian smoothing of a channel-last field along each spatial axis.

    Parameters
    ----------
    grid : Tensor (B, *spatial, C)
        Any C (vector components or image channels); spatial axes in tensor order (z, y, x).
    sigma : float or sequence
        Standard deviation in voxels. A sequence gives one value per spatial axis (tensor
        order; it must have one entry per axis) and is always taken in voxels.
    spacing : sequence, optional
        ANTs (x, y, z) order, required by and only allowed with ``sigma_mode='physical'`` and a
        scalar sigma: per-axis sigma = sigma / spacing, clipped to [0.5, 10] voxels. (Invalid
        combinations raise ValueError.)
    sigma_mode : {'voxel', 'physical'}, default 'voxel'
    mode : str, default 'replicate'
        ``F.pad`` mode: 'replicate', 'constant' (zeros) or 'reflect'.
        Ignored for the compact kernel (always zero padding).
    kernel_type : str, default 'bessel'
        'bessel': ITK-style discrete Gaussian (``get_cached_gaussian_kernel_1d``).
        'compact' / 'compact_gaussian' / 'erf': ``fast_separable_gaussian_filter``.
        'gaussian' and 'sobolev' are accepted aliases of 'bessel'; other values raise ValueError.

    Returns
    -------
    Tensor of the input shape (contiguous); ``grid`` itself if every sigma <= 0. Axes with
    sigma <= 0 are not filtered. 3-D input off MPS uses depthwise ``conv3d``; MPS and 2-D use
    reshaped ``conv1d`` (same result).
    """
    if kernel_type in ('compact', 'compact_gaussian', 'erf'):
        return fast_separable_gaussian_filter(grid, sigma, spacing=spacing, sigma_mode=sigma_mode)
    if kernel_type not in ('bessel', 'gaussian', 'sobolev'):
        raise ValueError(f"unknown kernel_type {kernel_type!r}: use 'bessel' (aliases 'gaussian', "
                         "'sobolev') or 'compact' / 'compact_gaussian' / 'erf'")
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)
    
    sigma_list = _resolve_sigmas(sigma, spacing, sigma_mode, num_spatial)
        
    if all(s <= 0.0 for s in sigma_list):
        return grid
        
    v = torch.movedim(grid, -1, 1)
    
    pad_kwargs = {'mode': mode}
    if mode == 'constant':
        pad_kwargs['value'] = 0.0

    _is_mps = hasattr(device, 'type') and device.type == 'mps'
    if num_spatial == 3 and not _is_mps:
        # Fast path: F.conv3d with degenerate 1D kernels (broken on MPS for large volumes)
        C = v.shape[1]
        for i, sig in enumerate(sigma_list):
            if sig <= 0.0:
                continue
            kernel_1d = get_cached_gaussian_kernel_1d(sig, device, dtype).squeeze(0)
            pad = kernel_1d.shape[-1] // 2
            
            if i == 0:
                kz = kernel_1d.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), **pad_kwargs), kz, groups=C)
            elif i == 1:
                ky = kernel_1d.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), **pad_kwargs), ky, groups=C)
            elif i == 2:
                kx = kernel_1d.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), **pad_kwargs), kx, groups=C)
        return torch.movedim(v, 1, -1).contiguous()
        
    for i in range(num_spatial):
        sig = sigma_list[i]
        if sig <= 0.0:
            continue
            
        kernel = get_cached_gaussian_kernel_1d(sig, device, dtype)
        kernel_size = kernel.shape[-1]
        pad_size = kernel_size // 2
        
        target_dim = i + 2
        dims = list(range(v.ndim))
        dims[-1], dims[target_dim] = dims[target_dim], dims[-1]
        v_permuted = v.permute(*dims).contiguous()
        
        last_dim_size = v_permuted.shape[-1]
        v_reshaped = v_permuted.view(-1, 1, last_dim_size)
        v_padded = F.pad(v_reshaped, (pad_size, pad_size), **pad_kwargs)
        
        v_conv = F.conv1d(v_padded, kernel)
        v_conv_reshaped = v_conv.view(*v_permuted.shape)
        v_out = v_conv_reshaped.permute(*dims).contiguous()
        v = v_out
        
    return torch.movedim(v, 1, -1).contiguous()

_SOBOLEV_FILTER_CACHE = {}

def _get_sobolev_filter_cached(spatial_shape, alpha_val, s, spacing, device, dtype):
    """(1, 1, *rfft_shape) float32 filter (1 + alpha |k|^2)^-s on the ``rfftn`` grid, k in
    radians per spacing unit (``spacing`` in ANTs order). Cached; cache cleared above 32."""
    sp_tuple = tuple(float(x) for x in spacing) if spacing is not None else None
    cache_key = (tuple(spatial_shape), float(alpha_val), float(s), sp_tuple, str(device), str(dtype))
    if cache_key in _SOBOLEV_FILTER_CACHE:
        return _SOBOLEV_FILTER_CACHE[cache_key]
    
    dim = len(spatial_shape)
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        if d == dim - 1:
            k_d = (torch.fft.rfftfreq(n_d, device=device) * (2.0 * math.pi)) / max(sp_d, 1e-4)
        else:
            k_d = (torch.fft.fftfreq(n_d, device=device) * (2.0 * math.pi)) / max(sp_d, 1e-4)
        k_axes.append(k_d)
        
    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    k_sq = sum(k_j ** 2 for k_j in k_mesh)
    K_fourier = (1.0 / ((1.0 + alpha_val * k_sq) ** s)).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    
    # Maintain reasonable cache size
    if len(_SOBOLEV_FILTER_CACHE) > 32:
        _SOBOLEV_FILTER_CACHE.clear()
    _SOBOLEV_FILTER_CACHE[cache_key] = K_fourier
    return K_fourier


def sobolev_energy(v, alpha, spacing=None, s=2.0):
    """Per-field V-norm energy <v, L v> / N of velocity fields ``v`` (B, *spatial, dim), where
    L = (1 + alpha |k|^2)^s is the inverse of the periodic Sobolev kernel K applied by
    ``apply_sobolev_green_operator`` (same alpha, same k-grid; that function uses s = 2 and,
    without ``alpha``, alpha = fluid_sigma / 2). Summed over components, averaged over the N
    voxels (Parseval); ``spacing`` in ANTs (x, y, z) order, None = unit spacing. Computed in
    float32. Returns a (B,) tensor, differentiable. The LDDMM path energy is the time integral
    of this quantity.
    """
    dim = v.ndim - 2
    spatial_shape = v.shape[1:-1]
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else (1.0,) * dim
    k_axes = [torch.fft.fftfreq(n, device=v.device) * (2.0 * math.pi) / max(float(sp), 1e-4)
              for n, sp in zip(spatial_shape, spacing_zyx)]
    k_sq = sum(k ** 2 for k in torch.meshgrid(*k_axes, indexing="ij"))
    L = (1.0 + float(alpha) * k_sq) ** s
    v_cf = torch.movedim(v, -1, 1).to(torch.float32)
    vf = torch.fft.fftn(v_cf, dim=tuple(range(2, 2 + dim)))
    n = float(math.prod(spatial_shape))
    # Parseval: sum_x |v|^2 = sum_k |v_hat|^2 / N ; divide by N again for a per-voxel mean
    e = (vf.real ** 2 + vf.imag ** 2) * L
    return e.sum(dim=tuple(range(1, e.ndim))) / (n * n)


def apply_sobolev_green_operator(m, fluid_sigma=3.0, alpha=None, spacing=None):
    """
    Smooth a channel-last field with the Sobolev Green's operator K = (1 + alpha |k|^2)^-2,
    applied by FFT (``rfftn``), so the boundary is periodic (opposite faces interact).

    Parameters
    ----------
    m : Tensor (B, *spatial, C)
        Field (e.g. gradient / momentum); spatial axes in tensor order (z, y, x).
    fluid_sigma : float, default 3.0
        <= 0 returns ``m`` unchanged. Otherwise used only to set alpha = fluid_sigma / 2 when
        ``alpha`` is None.
    alpha : float, optional
        Kernel width, in squared spacing units (k is in radians per spacing unit).
    spacing : sequence, optional
        Voxel spacing in ANTs (x, y, z) order; None = 1.

    Returns
    -------
    Tensor of the input shape and dtype (computed in float32).
    """
    if fluid_sigma <= 0:
        return m
    device = m.device
    dtype = m.dtype
    dim = m.ndim - 2  # input is (B, *spatial, channels)
    if alpha is not None:
        alpha_val = float(alpha)
    else:
        alpha_val = float(fluid_sigma) / 2.0
    s = 2.0
    
    spatial_shape = m.shape[1:-1]
    spatial_dims = tuple(range(2, 2 + dim))
    m_cf = torch.movedim(m, -1, 1).to(torch.float32)
    
    K_bc = _get_sobolev_filter_cached(spatial_shape, alpha_val, s, spacing, device, dtype)
    
    m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)
    v_fft = m_fft * K_bc
    v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    
    return torch.movedim(v_cf, 1, -1)


# ==============================================================================
# Thread-safe LRU Cache for Discrete Sine Transform Type-I (DST-I) Green Operator
# ==============================================================================

CacheInfo = collections.namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

_DST_FILTER_CACHE: collections.OrderedDict = collections.OrderedDict()
_DST_CACHE_LOCK = threading.Lock()
_MAX_DST_CACHE_SIZE: int = 64
_DST_CACHE_HITS: int = 0
_DST_CACHE_MISSES: int = 0

# Backward compatibility aliases
_DSTI_FILTER_CACHE = _DST_FILTER_CACHE
_DSTI_CACHE_LOCK = _DST_CACHE_LOCK
_MAX_DSTI_CACHE_SIZE = _MAX_DST_CACHE_SIZE


def clear_dst_cache() -> None:
    """Empty the DST-I filter cache and reset its hit / miss counters (thread-safe)."""
    global _DST_CACHE_HITS, _DST_CACHE_MISSES
    with _DST_CACHE_LOCK:
        _DST_FILTER_CACHE.clear()
        _DST_CACHE_HITS = 0
        _DST_CACHE_MISSES = 0


clear_dsti_filter_cache = clear_dst_cache


def get_dst_cache_info() -> CacheInfo:
    """``CacheInfo(hits, misses, maxsize, currsize)`` of the DST-I filter cache."""
    with _DST_CACHE_LOCK:
        return CacheInfo(
            hits=_DST_CACHE_HITS,
            misses=_DST_CACHE_MISSES,
            maxsize=_MAX_DST_CACHE_SIZE,
            currsize=len(_DST_FILTER_CACHE),
        )


get_dsti_cache_info = get_dst_cache_info


def get_dsti_filter_cache_size() -> int:
    """Number of DST-I filters currently cached (thread-safe)."""
    with _DST_CACHE_LOCK:
        return len(_DST_FILTER_CACHE)


def _get_dsti_filter_cached(
    spatial_shape: tuple[int, ...],
    alpha_val: float,
    s: float = 2.0,
    spacing: tuple[float, ...] | list[float] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
) -> torch.Tensor:
    """
    DST-I spectral filter (1 + alpha * lambda)^-s, shape (1, 1, *spatial_shape), float32.

    lambda = sum_d 4 sin^2(pi k_d / (2 (n_d + 1))) / h_d^2, k_d = 1 .. n_d: the eigenvalues of
    the negative finite-difference Laplacian with zero Dirichlet values one voxel outside the
    grid; h_d = spacing (ANTs order, reversed), and h_d = 1 when ``spacing`` is None.
    LRU-cached (``_MAX_DST_CACHE_SIZE`` = 64 entries) under a lock, keyed by shape, spacing,
    device, dtype, alpha and s.
    """
    global _DST_CACHE_HITS, _DST_CACHE_MISSES

    sp_tuple = tuple(float(x) for x in spacing) if spacing is not None else None
    cache_key = (
        tuple(spatial_shape),
        sp_tuple,
        str(device),
        str(dtype),
        float(alpha_val),
        float(s),
    )

    with _DST_CACHE_LOCK:
        if cache_key in _DST_FILTER_CACHE:
            _DST_FILTER_CACHE.move_to_end(cache_key)
            _DST_CACHE_HITS += 1
            return _DST_FILTER_CACHE[cache_key]
        _DST_CACHE_MISSES += 1

    dim = len(spatial_shape)
    k_axes = []
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        scale = (sp_d ** 2) if spacing is not None else 1.0
        lambda_d = (4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)) / scale
        k_axes.append(lambda_d)

    with torch.no_grad():
        if dim == 1:
            lambda_sq = k_axes[0]
        elif dim == 2:
            lambda_sq = k_axes[0][:, None] + k_axes[1][None, :]
        elif dim == 3:
            lambda_sq = k_axes[0][:, None, None] + k_axes[1][None, :, None] + k_axes[2][None, None, :]
        else:
            shape_view = [1] * dim
            shape_view[0] = -1
            lambda_sq = k_axes[0].view(shape_view)
            for d in range(1, dim):
                shape_view = [1] * dim
                shape_view[d] = -1
                lambda_sq = lambda_sq + k_axes[d].view(shape_view)

        K_dst = (
            (1.0 / ((1.0 + alpha_val * lambda_sq) ** s))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

    with _DST_CACHE_LOCK:
        if cache_key in _DST_FILTER_CACHE:
            _DST_FILTER_CACHE.move_to_end(cache_key)
            return _DST_FILTER_CACHE[cache_key]
        _DST_FILTER_CACHE[cache_key] = K_dst
        if len(_DST_FILTER_CACHE) > _MAX_DST_CACHE_SIZE:
            _DST_FILTER_CACHE.popitem(last=False)
        return K_dst


_get_dst_filter_cached = _get_dsti_filter_cached


def apply_dsti_green_operator(
    m: torch.Tensor,
    fluid_sigma: float = 3.0,
    alpha: float | None = None,
    spacing: tuple[float, ...] | list[float] | None = None,
    s: float = 2.0,
) -> torch.Tensor:
    """
    Smooth a channel-last field with the Sobolev Green's operator (1 + alpha * lambda)^-s in
    the DST-I basis: zero Dirichlet boundary, i.e. the field is treated as 0 one voxel outside
    the grid (the border voxels themselves are not forced to 0, but are strongly damped).

    Each spatial axis is transformed with a DST-I computed from an FFT of the odd extension
    (length 2 (n + 1)), multiplied by the cached filter (``_get_dsti_filter_cached``; lambda are
    finite-difference Laplacian eigenvalues) and transformed back.

    Parameters
    ----------
    m : Tensor (B, *spatial, C)
        Field; spatial axes in tensor order (z, y, x). Any number of spatial axes.
    fluid_sigma : float, default 3.0
        On / off gate: <= 0 returns ``m`` unchanged. Its value only matters when no alpha is
        given (alpha = fluid_sigma / 2).
    alpha : float, optional
        Kernel width (squared spacing units). Falls back to
        fluid_sigma / 2.
    spacing : sequence, optional
        Voxel spacing in ANTs (x, y, z) order; None = 1.
    s : float, default 2.0
        Operator power.

    Returns
    -------
    Tensor of the input shape and dtype (computed in float32).
    """
    # fluid_sigma is used ONLY as an on/off gate for DST-I smoothing.
    # Its VALUE does not affect the kernel — only alpha controls kernel shape.
    # Any positive value enables smoothing; 0 or negative disables it entirely.
    # If you intend to tune regularization strength, use the alpha parameter instead.
    if fluid_sigma <= 0:
        return m

    device = m.device
    dtype = m.dtype
    spatial_shape = m.shape[1:-1]
    dim = len(spatial_shape)

    if alpha is not None:
        alpha_val = float(alpha)
    else:
        alpha_val = float(fluid_sigma) / 2.0
    s_val = float(s)
    spacing_val = spacing

    K_dst = _get_dsti_filter_cached(
        spatial_shape=spatial_shape,
        alpha_val=alpha_val,
        s=s_val,
        spacing=spacing_val,
        device=device,
        dtype=dtype,
    )

    # Channel-first representation: (B, C, *spatial)
    curr = m.movedim(-1, 1).to(torch.float32)

    # Forward separable DST-I across all spatial dimensions
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = -torch.imag(F.narrow(axis, 1, n_d)).contiguous()

    # Multiply by pre-unsqueezed Dirichlet Sobolev Green's kernel
    curr = curr * K_dst

    # Inverse separable DST-I across all spatial dimensions
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = (-torch.imag(F.narrow(axis, 1, n_d)) / (2.0 * (n_d + 1))).contiguous()

    return curr.to(dtype=dtype).movedim(1, -1)


apply_dsti1_green_operator = apply_dsti_green_operator


def smooth_displacement_field_dst(
    m: torch.Tensor | None = None,
    fluid_sigma: float = 3.0,
    alpha: float | None = None,
    spacing: tuple[float, ...] | list[float] | None = None,
    s: float = 2.0,
    alpha_val: float | None = None,
    displacement: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    ``apply_dsti_green_operator`` (DST-I Sobolev smoothing, zero Dirichlet boundary) of a
    displacement / velocity field (B, *spatial, C), accepting the field as ``m`` or
    ``displacement`` (``displacement`` wins) and alpha as ``alpha`` or ``alpha_val``
    (``alpha_val`` wins). Other arguments are passed through. Raises ValueError if no field is
    given. Returns the smoothed field (input shape and dtype).
    """
    tensor = displacement if displacement is not None else m
    if tensor is None:
        raise ValueError("Must provide either 'm' or 'displacement' tensor to smooth_displacement_field_dst.")
    eff_alpha = alpha_val if alpha_val is not None else alpha
    return apply_dsti_green_operator(
        tensor,
        fluid_sigma=fluid_sigma,
        alpha=eff_alpha,
        spacing=spacing,
        s=s,
        **kwargs,
    )


def get_boundary_mask(spatial, device, dtype, rim_size=1):
    """
    Mask of shape (1, *spatial, 1) on ``device`` / ``dtype``: 0 on the outer ``rim_size`` voxels
    of every face, 1 inside (broadcasts against channel-last fields). ``rim_size`` <= 0 gives an
    all-ones mask.
    """
    boundary_mask = torch.ones((1, *spatial, 1), device=device, dtype=dtype)
    if rim_size <= 0:
        return boundary_mask
    for i in range(len(spatial)):
        slices = [slice(None)] * boundary_mask.ndim
        slices[i + 1] = slice(0, rim_size)
        boundary_mask[tuple(slices)] = 0
        slices[i + 1] = slice(-rim_size, None)
        boundary_mask[tuple(slices)] = 0
    return boundary_mask


def has_antstorch() -> bool:
    """True if ``antstorch.bspline_flows`` imports (False only on ImportError; other import
    errors propagate). Imports antstorch as a side effect; if matplotlib was already imported
    with a backend other than 'Agg', that backend is re-selected afterwards."""
    try:
        import sys
        orig_backend = None
        if 'matplotlib' in sys.modules:
            import matplotlib
            orig_backend = matplotlib.get_backend()
        from antstorch.bspline_flows import fit_bspline_displacement_field, ImageDomain
        if orig_backend and orig_backend != 'Agg':
            try:
                import matplotlib
                matplotlib.use(orig_backend)
            except Exception:
                pass
        return True
    except ImportError:
        return False


def _require_antstorch():
    """Raise ImportError unless ``has_antstorch()``."""
    if not has_antstorch():
        raise ImportError(
            "ANTsTorch B-spline facilities require 'antstorch'. "
            "Please install it via 'pip install antstorch'."
        )


def smooth_displacement_field_bspline(
    field: torch.Tensor,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    mesh_size: Optional[Union[int, Sequence[int]]] = None,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    fluid_sigma: Optional[float] = None,
    enforce_stationary_boundary: bool = False,
    order: int = 3,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
) -> torch.Tensor:
    """Smooth a displacement / velocity field by a B-spline least-squares fit (ANTsTorch).

    Fits a B-spline control lattice to the dense field
    (``antstorch.bspline_flows.fit_bspline_displacement_field``, one fitting level) and
    evaluates it back on the grid -- the BSplineSyN-style regulariser. Each batch item is fitted
    separately. Raises ImportError if antstorch is missing.

    Parameters
    ----------
    field : torch.Tensor
        (B, *spatial, dim) or (*spatial, dim); spatial axes always in tensor order (z, y, x).
    spacing, origin : sequence of float, optional
        Grid geometry, in the order given by ``coord_convention``. Default 1 and 0 per axis.
    mesh_size : int or sequence of int, optional
        B-spline spans per axis (``coord_convention`` order); an int applies to all axes.
    spline_distance : float or sequence of float, optional
        Knot spacing in spacing units; mesh = ceil(physical extent / distance) per axis
        (``mesh_size_for_spline_distance``). A sequence follows ``coord_convention`` like
        ``spacing`` / ``mesh_size`` (reversed internally for 'zyx').
    fluid_sigma : float, optional
        Used only if neither of the above is given and > 0: spline distance =
        max(4 * fluid_sigma, 12).
    enforce_stationary_boundary : bool, default False
        Passed to the fit: pin the field to zero at the domain boundary.
    order : int, default 3
        Spline order.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Order of the vector components and of ``spacing`` / ``origin`` / sequence
        ``mesh_size``: 'xyz' = ANTs (x, y, z), 'zyx' = tensor order (reversed internally).

    Priority: ``spline_distance``, then ``mesh_size``, then ``fluid_sigma``, else 6 spans per
    axis; every axis gets at least 2 spans.

    Returns
    -------
    torch.Tensor
        Smoothed field of the input shape (dtype / device as returned by antstorch).
    """
    _require_antstorch()
    from antstorch.bspline_flows import (
        fit_bspline_displacement_field,
        ImageDomain,
        mesh_size_for_spline_distance,
    )

    d = field.shape[-1]
    if field.dim() == d + 1:
        unbatched = True
        v = field.unsqueeze(0)
    elif field.dim() == d + 2:
        unbatched = False
        v = field
    else:
        raise ValueError(
            f"Expected field of dimension {d + 1} (*spatial, {d}) or "
            f"{d + 2} (B, *spatial, {d}), got {field.dim()}"
        )

    B = v.shape[0]
    spatial_shape = v.shape[1:-1]
    size_itk = tuple(reversed(spatial_shape))

    if spacing is not None:
        if coord_convention == 'zyx':
            spacing_itk = tuple(reversed(spacing))
        else:
            spacing_itk = tuple(float(s) for s in spacing)
    else:
        spacing_itk = (1.0,) * d

    if origin is not None:
        if coord_convention == 'zyx':
            origin_itk = tuple(reversed(origin))
        else:
            origin_itk = tuple(float(o) for o in origin)
    else:
        origin_itk = (0.0,) * d

    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        if coord_convention == 'zyx' and not isinstance(spline_distance, (int, float)):
            spline_distance = tuple(reversed(tuple(spline_distance)))
        mesh_size_itk = mesh_size_for_spline_distance(domain, spline_distance)
    elif mesh_size is not None:
        if isinstance(mesh_size, int):
            mesh_size_itk = (int(mesh_size),) * d
        else:
            mesh_size_itk = tuple(reversed(mesh_size)) if coord_convention == 'zyx' else tuple(mesh_size)
    elif fluid_sigma is not None and float(fluid_sigma) > 0:
        eff_dist = max(4.0 * float(fluid_sigma), 12.0)
        mesh_size_itk = mesh_size_for_spline_distance(domain, eff_dist)
    else:
        mesh_size_itk = (6,) * d

    # Ensure at least 2 intervals per axis for cubic splines
    mesh_size_itk = tuple(max(2, int(m)) for m in mesh_size_itk)

    outs = []
    for b in range(B):
        vb = v[b : b + 1]  # (1, *spatial, d)
        # Convert to channel-first (1, d, *spatial)
        if coord_convention == 'zyx':
            vb_itk = vb.movedim(-1, 1).flip(1)
        else:
            vb_itk = vb.movedim(-1, 1)

        sm = fit_bspline_displacement_field(
            displacement_field=vb_itk,
            domain=domain,
            number_of_fitting_levels=1,
            mesh_size=mesh_size_itk,
            spline_order=order,
            enforce_stationary_boundary=enforce_stationary_boundary,
        )

        if coord_convention == 'zyx':
            sm_syntx = sm.flip(1).movedim(1, -1)
        else:
            sm_syntx = sm.movedim(1, -1)

        outs.append(sm_syntx)

    res = torch.cat(outs, dim=0) if B > 1 else outs[0]
    return res.squeeze(0) if unbatched else res


apply_bspline_fluid_operator = smooth_displacement_field_bspline


__all__ = [
    "separable_gaussian_filter",
    "get_cached_gaussian_kernel_1d",
    "apply_sobolev_green_operator",
    "apply_dsti_green_operator",
    "apply_dsti1_green_operator",
    "smooth_displacement_field_dst",
    "smooth_displacement_field_bspline",
    "apply_bspline_fluid_operator",
    "has_antstorch",
    "clear_dst_cache",
    "clear_dsti_filter_cache",
    "get_dst_cache_info",
    "get_dsti_cache_info",
    "get_dsti_filter_cache_size",
    "get_boundary_mask",
    "CacheInfo",
]

