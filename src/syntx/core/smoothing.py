import math
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

def separable_gaussian_filter(grid: torch.Tensor, sigma, spacing=None, sigma_mode='voxel') -> torch.Tensor:
    """
    Applies separable Gaussian filtering along each spatial dimension.
    Input format: (B, *spatial, dim) - channel-last representation of coordinates.
    sigma: float or tuple of floats per spatial dimension.
    sigma_mode: 'voxel' (default) or 'physical' (scales voxel sigma per axis by spacing).
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
                v = F.conv3d(F.pad(v, (0, 0, 0, 0, pad, pad), mode='replicate'), kz, groups=C)
            elif i == 1:
                ky = kernel_1d.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (0, 0, pad, pad, 0, 0), mode='replicate'), ky, groups=C)
            elif i == 2:
                kx = kernel_1d.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
                v = F.conv3d(F.pad(v, (pad, pad, 0, 0, 0, 0), mode='replicate'), kx, groups=C)
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
        v_padded = F.pad(v_reshaped, (pad_size, pad_size), mode='replicate')
        
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
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing[d]) if (spacing is not None and d < len(spacing)) else 1.0
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


def apply_dsti_green_operator(m, fluid_sigma=3.0, alpha=None):
    """
    Applies Sobolev Green's operator in Discrete Sine Transform Type-I (DST-I) space.
    Analytically enforces exact homogeneous Dirichlet boundary conditions (v = 0 at boundaries)
    using memory-efficient separable 1D DST-I transforms.
    """
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
    s = 2.0

    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
        k_axes.append(lambda_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_j for k_j in k_mesh)
    K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

    # Channel first representation: (B, C, *spatial)
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

    # Multiply by Dirichlet Sobolev Green's kernel
    curr = curr * K_dst.unsqueeze(0).unsqueeze(0)

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


def apply_dsti1_green_operator(m, fluid_sigma=3.0, alpha=None):
    """
    Alias for apply_dsti_green_operator (separable 1D DST-I transforms).
    """
    return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)


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
