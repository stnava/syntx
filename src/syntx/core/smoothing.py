import collections
import math
import threading
import torch
import torch.nn.functional as F
import numpy as np

_gaussian_kernel_cache = {}
_tensor_kernel_cache = {}

def get_cached_gaussian_kernel_1d(sig: float, device, dtype):
    sig_key = round(float(sig), 5)
    cache_key = (sig_key, str(device), str(dtype))
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

def separable_gaussian_filter(grid: torch.Tensor, sigma, spacing=None, sigma_mode='voxel', mode: str = 'replicate') -> torch.Tensor:
    """
    Applies separable Gaussian filtering along each spatial dimension.
    Input format: (B, *spatial, dim) - channel-last representation of coordinates.
    sigma: float or tuple of floats per spatial dimension.
    sigma_mode: 'voxel' (default) or 'physical' (scales voxel sigma per axis by spacing).
    mode: padding mode ('replicate' by default, 'constant' for Dirichlet zero-padding, 'reflect').
    """
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)
    
    if isinstance(sigma, (tuple, list)):
        sigma_list = [float(s) for s in sigma]
    elif sigma_mode == 'physical' and spacing is not None:
        spacing_rev = tuple(reversed(spacing))
        sigma_list = [float(np.clip(float(sigma) / sp, 0.5, 10.0)) for sp in spacing_rev]
    elif isinstance(sigma, (int, float)):
        sigma_list = [float(sigma)] * num_spatial
    else:
        sigma_list = [float(sigma)] * num_spatial
        
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


def apply_sobolev_green_operator(m, fluid_sigma=3.0, alpha=None, border_width=0, spacing=None, pad_to_fast=False, **kwargs):
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
    m_cf = m.permute(0, 3, 1, 2) if dim == 2 else m.permute(0, 4, 1, 2, 3)
    
    K_bc = _get_sobolev_filter_cached(spatial_shape, alpha_val, s, spacing, device, dtype)
    
    m_fft = torch.fft.rfftn(m_cf.to(torch.float32), dim=spatial_dims)
    v_fft = m_fft * K_bc
    v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    
    if dim == 2:
        return v_cf.permute(0, 2, 3, 1)
    else:
        return v_cf.permute(0, 2, 3, 4, 1)


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
    """Thread-safe clearance of cached DST-I Green operator eigenvalue filters."""
    global _DST_CACHE_HITS, _DST_CACHE_MISSES
    with _DST_CACHE_LOCK:
        _DST_FILTER_CACHE.clear()
        _DST_CACHE_HITS = 0
        _DST_CACHE_MISSES = 0


clear_dsti_filter_cache = clear_dst_cache


def get_dst_cache_info() -> CacheInfo:
    """Returns a namedtuple (hits, misses, maxsize, currsize) for the DST filter cache."""
    with _DST_CACHE_LOCK:
        return CacheInfo(
            hits=_DST_CACHE_HITS,
            misses=_DST_CACHE_MISSES,
            maxsize=_MAX_DST_CACHE_SIZE,
            currsize=len(_DST_FILTER_CACHE),
        )


get_dsti_cache_info = get_dst_cache_info


def get_dsti_filter_cache_size() -> int:
    """Thread-safe query of the current number of cached DST-I eigenvalue filters."""
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
    Retrieves or computes cached DST-I Green operator eigenvalues.
    Uses double-checked locking with thread-safe LRU eviction.
    Keyed by (tuple(spatial_shape), tuple(float(x) for x in spacing) if spacing is not None else None, str(device), str(dtype), float(alpha_val), float(s)).
    Eliminates dynamic torch.meshgrid allocations on cache miss via broadcasted eigenvalues.
    Cached tensor is pre-shaped to (1, 1, *spatial_shape) to eliminate per-call unsqueeze overhead.
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
    **kwargs,
) -> torch.Tensor:
    """
    Applies Sobolev Green's operator in Discrete Sine Transform Type-I (DST-I) space.
    Analytically enforces exact homogeneous Dirichlet boundary conditions (v = 0 at boundaries)
    using memory-efficient separable 1D DST-I transforms and cached eigenvalues.
    """
    if fluid_sigma <= 0:
        return m

    device = m.device
    dtype = m.dtype
    spatial_shape = m.shape[1:-1]
    dim = len(spatial_shape)

    if alpha is not None:
        alpha_val = float(alpha)
    elif "alpha_val" in kwargs:
        alpha_val = float(kwargs["alpha_val"])
    else:
        alpha_val = float(fluid_sigma) / 2.0
    s_val = float(kwargs.get("s", s))
    spacing_val = kwargs.get("spacing", spacing)

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
    Smooths a displacement or velocity field using Sobolev Green's operator in Discrete
    Sine Transform Type-I (DST-I) space with exact homogeneous Dirichlet boundary conditions.

    Public alias for apply_dsti_green_operator supporting both positional and keyword argument
    conventions ('m' or 'displacement', 'fluid_sigma', 'alpha' or 'alpha_val', 'spacing', 's').
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
    Constructs a boundary mask where boundary voxels are 0 and interior voxels are 1.
    """
    boundary_mask = torch.ones((1, *spatial, 1), device=device, dtype=dtype)
    for i in range(len(spatial)):
        slices = [slice(None)] * boundary_mask.ndim
        slices[i + 1] = slice(0, rim_size)
        boundary_mask[tuple(slices)] = 0
        slices[i + 1] = slice(-rim_size, None)
        boundary_mask[tuple(slices)] = 0
    return boundary_mask


__all__ = [
    "separable_gaussian_filter",
    "get_cached_gaussian_kernel_1d",
    "apply_sobolev_green_operator",
    "apply_dsti_green_operator",
    "apply_dsti1_green_operator",
    "smooth_displacement_field_dst",
    "clear_dst_cache",
    "clear_dsti_filter_cache",
    "get_dst_cache_info",
    "get_dsti_cache_info",
    "get_dsti_filter_cache_size",
    "get_boundary_mask",
    "CacheInfo",
]

