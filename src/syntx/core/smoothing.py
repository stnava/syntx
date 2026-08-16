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

def apply_sobolev_green_operator(m, fluid_sigma=3.0, alpha=None, border_width=0, **kwargs):
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
    pad = 8  # reflection padding to prevent Gibbs ringing
    pad_shape = tuple(sz + 2 * pad for sz in spatial_shape)
    
    k_axes = []
    for d in range(dim):
        n_d = pad_shape[d]
        if d == dim - 1:
            k_d = torch.fft.rfftfreq(n_d, device=device) * (2.0 * math.pi)
        else:
            k_d = torch.fft.fftfreq(n_d, device=device) * (2.0 * math.pi)
        k_axes.append(k_d)
        
    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    k_sq = sum(k_j ** 2 for k_j in k_mesh)
    K_fourier = 1.0 / ((1.0 + alpha_val * k_sq) ** s)
    
    spatial_dims = tuple(range(2, 2 + dim))
    m_cf = m.permute(0, 3, 1, 2) if dim == 2 else m.permute(0, 4, 1, 2, 3)
    
    pad_tuple = (pad, pad) * dim
    m_padded = torch.nn.functional.pad(m_cf, pad_tuple, mode='reflect')
    
    m_fft = torch.fft.rfftn(m_padded.to(torch.float32), dim=spatial_dims)
    K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
    v_fft = m_fft * K_bc
    v_padded = torch.fft.irfftn(v_fft, s=pad_shape, dim=spatial_dims).to(dtype=dtype)
    
    if dim == 2:
        v_cf = v_padded[..., pad:-pad, pad:-pad]
        return v_cf.permute(0, 2, 3, 1)
    else:
        v_cf = v_padded[..., pad:-pad, pad:-pad, pad:-pad]
        return v_cf.permute(0, 2, 3, 4, 1)

def apply_dsti_green_operator(m, fluid_sigma=3.0, alpha=None):
    """
    Applies Sobolev Green's operator in Discrete Sine Transform Type-I (DST-I) space.
    Analytically enforces exact homogeneous Dirichlet boundary conditions (v = 0 at boundaries).
    """
    if fluid_sigma <= 0:
        return m

    device = m.device
    dtype = m.dtype
    dim = m.ndim - 2

    if alpha is not None:
        alpha_val = float(alpha)
    else:
        alpha_val = float(fluid_sigma) / 2.0
    s = 2.0

    spatial_shape = m.shape[1:-1]
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
        k_axes.append(lambda_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_j for k_j in k_mesh)
    K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

    m_cf = m.movedim(-1, 1).to(torch.float32)

    padded = m_cf
    for d in range(dim):
        axis = 2 + d
        z_shape = list(padded.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(padded, dims=[axis])
        padded = torch.cat([z, padded, z, rev], dim=axis)

    spatial_axes = tuple(range(2, 2 + dim))
    fft_padded = torch.fft.fftn(padded, dim=spatial_axes)

    slices = [slice(None), slice(None)]
    for n_d in spatial_shape:
        slices.append(slice(1, n_d + 1))

    if dim % 2 == 1:
        sign = -1.0 if (dim % 4 == 1) else 1.0
        dst_coeff = sign * (0.5 ** dim) * torch.imag(fft_padded[tuple(slices)])
    else:
        sign = -1.0 if (dim % 4 == 2) else 1.0
        dst_coeff = sign * (0.5 ** dim) * torch.real(fft_padded[tuple(slices)])

    K_bc = K_dst.unsqueeze(0).unsqueeze(0)
    dst_filtered = dst_coeff * K_bc

    padded_c = dst_filtered
    norm_factor = 1.0
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        norm_factor *= 4.0 / (n_d + 1)
        z_shape = list(padded_c.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev_c = -torch.flip(padded_c, dims=[axis])
        padded_c = torch.cat([z, padded_c, z, rev_c], dim=axis)

    fft_padded_c = torch.fft.fftn(padded_c, dim=spatial_axes)

    if dim % 2 == 1:
        sign = -1.0 if (dim % 4 == 1) else 1.0
        idst_out = sign * (0.5 ** dim) * torch.imag(fft_padded_c[tuple(slices)]) * norm_factor
    else:
        sign = -1.0 if (dim % 4 == 2) else 1.0
        idst_out = sign * (0.5 ** dim) * torch.real(fft_padded_c[tuple(slices)]) * norm_factor

    if str(device) == 'mps':
        torch.mps.empty_cache()

    return idst_out.to(dtype=dtype).movedim(1, -1)

def apply_dsti1_green_operator(m, fluid_sigma=3.0, alpha=None):
    """
    Applies Sobolev Green's operator using separable 1D DST-I transforms.
    """
    if fluid_sigma <= 0:
        return m

    device = m.device
    dtype = m.dtype
    dim = m.ndim - 2

    if alpha is not None:
        alpha_val = float(alpha)
    else:
        alpha_val = float(fluid_sigma) / 2.0
    s = 2.0

    spatial_shape = m.shape[1:-1]

    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
        k_axes.append(lambda_d)

    k_mesh = torch.meshgrid(*k_axes, indexing='ij')
    lambda_sq = sum(k_j for k_j in k_mesh)
    K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

    m_cf = m.movedim(-1, 1).to(torch.float32)

    def _dst1_1d(arr, axis):
        n_d = arr.shape[axis]
        z_shape = list(arr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(arr, dims=[axis])
        padded = torch.cat([z, arr, z, rev], dim=axis)
        fft_1d = torch.fft.rfft(padded, dim=axis)
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(1, n_d + 1)
        out = -0.5 * torch.imag(fft_1d[tuple(sl)]).clone()
        
        del z, rev, padded, fft_1d
        if str(device) == 'mps':
            torch.mps.empty_cache()
        return out

    curr = m_cf
    for d in range(dim):
        axis = 2 + d
        curr = _dst1_1d(curr, axis)

    K_bc = K_dst.unsqueeze(0).unsqueeze(0)
    v_dst = curr * K_bc

    curr_inv = v_dst
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        curr_inv = _dst1_1d(curr_inv, axis) * (4.0 / float(n_d + 1))

    if str(device) == 'mps':
        torch.mps.empty_cache()

    return curr_inv.to(dtype=dtype).movedim(1, -1)

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
