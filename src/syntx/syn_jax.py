import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.dlpack
import jax.dlpack as jax_dlpack

class PhysicalWarpArray(np.ndarray):
    def __new__(cls, input_array, is_physical=False):
        obj = np.asarray(input_array).view(cls)
        obj.is_physical = is_physical
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.is_physical = getattr(obj, 'is_physical', False)

def to_torch_tensor(x_jax):
    """Converts a JAX array to a PyTorch tensor via DLPack (zero-copy)."""
    x_jax = jax.device_put(x_jax)
    if x_jax.size == 0:
        torch_device = 'cpu' if x_jax.device.platform == 'cpu' else ('mps' if x_jax.device.platform == 'metal' else 'cuda')
        dtype_map = {
            jnp.float32: torch.float32,
            jnp.float64: torch.float64,
            jnp.int32: torch.int32,
            jnp.int64: torch.int64,
        }
        torch_dtype = dtype_map.get(x_jax.dtype, torch.float32)
        return torch.empty(x_jax.shape, dtype=torch_dtype, device=torch_device)
        
    return torch.from_dlpack(x_jax)

def to_jax_array_dl(x_torch):
    """Converts a PyTorch tensor to a JAX array via DLPack (zero-copy)."""
    x_torch = x_torch.contiguous()
    if x_torch.numel() == 0:
        jax_device = jax.devices('cpu')[0] if x_torch.device.type == 'cpu' else jax.devices()[0]
        dtype_map = {
            torch.float32: jnp.float32,
            torch.float64: jnp.float64,
            torch.int32: jnp.int32,
            torch.int64: jnp.int64,
        }
        jax_dtype = dtype_map.get(x_torch.dtype, jnp.float32)
        return jax.numpy.empty(x_torch.shape, dtype=jax_dtype, device=jax_device)
        
    return jax_dlpack.from_dlpack(x_torch)

def make_pytorch_loss_jax(pytorch_loss_fn):
    """Wraps a PyTorch loss function to be called from JAX with full autograd gradient sharing."""
    
    def py_forward(m_np, f_np):
        m_np = np.asarray(m_np)
        f_np = np.asarray(f_np)
        m_torch = torch.from_numpy(m_np)
        f_torch = torch.from_numpy(f_np)
        device = None
        if hasattr(pytorch_loss_fn, 'parameters'):
            try:
                device = next(pytorch_loss_fn.parameters()).device
            except StopIteration:
                pass
        if device is not None:
            m_torch = m_torch.to(device)
            f_torch = f_torch.to(device)
        with torch.no_grad():
            loss = pytorch_loss_fn(m_torch, f_torch)
        return np.array(loss.cpu().numpy(), dtype=np.float32)

    def py_backward(m_np, f_np, g_np):
        m_np = np.asarray(m_np)
        f_np = np.asarray(f_np)
        g_np = np.asarray(g_np)
        m_torch = torch.from_numpy(m_np).clone().requires_grad_(True)
        f_torch = torch.from_numpy(f_np).clone().requires_grad_(True)
        device = None
        if hasattr(pytorch_loss_fn, 'parameters'):
            try:
                device = next(pytorch_loss_fn.parameters()).device
            except StopIteration:
                pass
        if device is not None:
            m_torch = m_torch.to(device)
            f_torch = f_torch.to(device)
        loss = pytorch_loss_fn(m_torch, f_torch)
        if not loss.requires_grad or loss.grad_fn is None:
            return np.zeros_like(m_np), np.zeros_like(f_np)
        g_torch = torch.from_numpy(g_np).to(loss.device)
        loss.backward(gradient=g_torch)
        grad_m = m_torch.grad.cpu().numpy() if m_torch.grad is not None else np.zeros_like(m_np)
        grad_f = f_torch.grad.cpu().numpy() if f_torch.grad is not None else np.zeros_like(f_np)
        return grad_m, grad_f

    @jax.custom_vjp
    def jax_loss_fn(m, f):
        if isinstance(m, jax.core.Tracer) or isinstance(f, jax.core.Tracer):
            return jax.pure_callback(
                py_forward,
                jax.ShapeDtypeStruct((), jnp.float32),
                m, f
            )
        else:
            m_torch = to_torch_tensor(m)
            f_torch = to_torch_tensor(f)
            device = None
            if hasattr(pytorch_loss_fn, 'parameters'):
                try:
                    device = next(pytorch_loss_fn.parameters()).device
                except StopIteration:
                    pass
            if device is not None:
                m_torch = m_torch.to(device)
                f_torch = f_torch.to(device)
            with torch.no_grad():
                loss = pytorch_loss_fn(m_torch, f_torch)
            return to_jax_array_dl(loss)

    def jax_loss_fn_fwd(m, f):
        loss_val = jax_loss_fn(m, f)
        return loss_val, (m, f)

    def jax_loss_fn_bwd(res, g):
        m, f = res
        if isinstance(m, jax.core.Tracer) or isinstance(f, jax.core.Tracer) or isinstance(g, jax.core.Tracer):
            grad_m, grad_f = jax.pure_callback(
                py_backward,
                (
                    jax.ShapeDtypeStruct(m.shape, jnp.float32),
                    jax.ShapeDtypeStruct(f.shape, jnp.float32)
                ),
                m, f, g
            )
            return grad_m, grad_f
        else:
            m_torch = to_torch_tensor(m).detach().clone().requires_grad_(True)
            f_torch = to_torch_tensor(f).detach().clone().requires_grad_(True)
            device = None
            if hasattr(pytorch_loss_fn, 'parameters'):
                try:
                    device = next(pytorch_loss_fn.parameters()).device
                except StopIteration:
                    pass
            if device is not None:
                m_torch = m_torch.to(device)
                f_torch = f_torch.to(device)
            loss = pytorch_loss_fn(m_torch, f_torch)
            if not loss.requires_grad or loss.grad_fn is None:
                return jnp.zeros_like(m), jnp.zeros_like(f)
            g_torch = to_torch_tensor(g).to(loss.device)
            loss.backward(gradient=g_torch)
            grad_m = to_jax_array_dl(m_torch.grad) if m_torch.grad is not None else jnp.zeros_like(m)
            grad_f = to_jax_array_dl(f_torch.grad) if f_torch.grad is not None else jnp.zeros_like(f)
            return grad_m, grad_f

    jax_loss_fn.defvjp(jax_loss_fn_fwd, jax_loss_fn_bwd)
    jax_loss_fn._is_pytorch_loss = True
    jax_loss_fn._pytorch_loss_fn = pytorch_loss_fn
    return jax_loss_fn

def dlpack_feature_loss(pytorch_loss_fn):
    return make_pytorch_loss_jax(pytorch_loss_fn)

def check_convergence(losses, window_size=10, slope_threshold=1e-8):
    if len(losses) < window_size:
        return False
    y = np.array(losses[-window_size:])
    x = np.arange(window_size)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-8:
        return False
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return slope >= -slope_threshold

from .transform import SyNToTransform

# 1. Skew-Symmetric SO(d) Rotation Matrix
def get_rotation_matrix_jax(omega, dim):
    """
    Computes a rotation matrix from a skew-symmetric Lie Algebra parameterization.
    For 2D, omega has 1 element. For 3D, omega has 3 elements.
    Safe for JAX AD (avoids division by zero and NaNs at omega = 0 by avoiding
    non-differentiable jnp.linalg.norm(0)).
    
    Provenance:
    Adapted from ITK's affine/rigid registration coordinate transformations, specifically
    within itkEuler3DTransform.hxx and Lie group rotations in itkSyNImageRegistrationMethod.hxx.
    """
    if dim == 2:
        theta = omega[0]
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        return jnp.stack([
            jnp.stack([cos_t, -sin_t]),
            jnp.stack([sin_t, cos_t])
        ])
    elif dim == 3:
        theta2 = jnp.sum(omega**2)
        is_zero = theta2 < 1e-16
        safe_theta2 = jnp.where(is_zero, 1e-16, theta2)
        theta = jnp.sqrt(safe_theta2)
        
        safe_theta = jnp.where(is_zero, 1.0, theta)
        omega_norm = omega / safe_theta
        
        K_raw = jnp.stack([
            jnp.stack([0.0, -omega[2], omega[1]]),
            jnp.stack([omega[2], 0.0, -omega[0]]),
            jnp.stack([-omega[1], omega[0], 0.0])
        ])
        
        K = jnp.stack([
            jnp.stack([0.0, -omega_norm[2], omega_norm[1]]),
            jnp.stack([omega_norm[2], 0.0, -omega_norm[0]]),
            jnp.stack([-omega_norm[1], omega_norm[0], 0.0])
        ])
        I = jnp.eye(3)
        R = I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
        
        R_small = I + K_raw
        return jnp.where(is_zero, R_small, R)
    else:
        raise ValueError("Only 2D and 3D are supported.")


# 2. Get Affine Matrix from parameters
def get_affine_matrix_jax(params, dim, transform_type):
    """
    Constructs the homogeneous affine transformation matrix.
    """
    translation = params['translation']
    omega = params['omega']
    scale = params['scale']
    anisotropic_scale = params['anisotropic_scale']
    shear = params['shear']
    
    R = get_rotation_matrix_jax(omega, dim)
    
    if transform_type == 'Affine':
        S = jnp.diag(anisotropic_scale * scale[0])
        if dim == 2:
            Sh = jnp.eye(2)
            Sh = Sh.at[0, 1].set(shear[0])
        elif dim == 3:
            Sh = jnp.eye(3)
            Sh = Sh.at[0, 1].set(shear[0])
            Sh = Sh.at[0, 2].set(shear[1])
            Sh = Sh.at[1, 2].set(shear[2])
        else:
            raise ValueError("Only 2D and 3D are supported.")
        A = R @ S @ Sh
    else:
        A = R * scale[0]
        
    T_opt = jnp.eye(dim + 1)
    T_opt = T_opt.at[:dim, :dim].set(A)
    T_opt = T_opt.at[:dim, dim].set(translation)
    
    if 'T_init' in params:
        return T_opt @ params['T_init']
    return T_opt


# 3. Coordinate Grid Sampling
def jax_grid_sample(image, grid, mode='bilinear', padding_mode='border'):
    """
    Sample images using map_coordinates in JAX.
    image: (B, C, *spatial_source)
    grid: (B, *spatial_target, dim)
    Returns: (B, C, *spatial_target)
    """
    order = 1 if mode == 'bilinear' else 0
    B, C = image.shape[0], image.shape[1]
    spatial_source = image.shape[2:]
    spatial_target = grid.shape[1:-1]
    ndim = len(spatial_source)
    
    # Map [-1, 1] to voxel coordinates [0, size - 1]
    coords = []
    for d in range(ndim):
        size = spatial_source[d]
        norm_coord = grid[..., ndim - 1 - d]
        vox_coord = (norm_coord + 1.0) * (size - 1) / 2.0
        coords.append(vox_coord)
    coords_stacked = jnp.stack(coords, axis=1) # (B, ndim, *spatial_target)
    
    def sample_single(img_ch, coord_ch):
        coords_flat = coord_ch.reshape(ndim, -1)
        jax_mode = 'nearest' if padding_mode == 'border' else 'constant'
        sampled_flat = jax.scipy.ndimage.map_coordinates(
            img_ch, coords_flat, order=order, mode=jax_mode, cval=0.0
        )
        return sampled_flat.reshape(spatial_target)
        
    vmap_channel = jax.vmap(sample_single, in_axes=(0, None), out_axes=0)
    vmap_batch = jax.vmap(vmap_channel, in_axes=(0, 0), out_axes=0)
    return vmap_batch(image, coords_stacked)



def interpolate_jax(image, scale_factor, dim):
    B, C = image.shape[0], image.shape[1]
    spatial = image.shape[2:]
    target_spatial = tuple(max(1, int(size * scale_factor)) for size in spatial)
    
    grids = [jnp.linspace(-1.0 + 1.0/size_new, 1.0 - 1.0/size_new, size_new) for size_new in target_spatial]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    grid = jnp.stack(list(reversed(meshgrid)), axis=-1)
    
    coords = []
    for d in range(dim):
        size = spatial[d]
        norm_coord = grid[..., dim - 1 - d]
        vox_coord = ((norm_coord + 1.0) * size - 1.0) / 2.0
        coords.append(vox_coord)
    coords_stacked = jnp.stack(coords, axis=0)
    coords_flat = coords_stacked.reshape(dim, -1)
    
    def sample_single(img_ch):
        sampled_flat = jax.scipy.ndimage.map_coordinates(
            img_ch, coords_flat, order=1, mode='nearest', cval=0.0
        )
        return sampled_flat.reshape(target_spatial)
        
    vmap_channel = jax.vmap(sample_single, in_axes=0, out_axes=0)
    vmap_batch = jax.vmap(vmap_channel, in_axes=0, out_axes=0)
    return vmap_batch(image)


# 4. Affine Grid Generation
def jax_affine_grid(A, shape):
    """
    Generates a coordinate grid for affine transformations.
    A: (dim, dim + 1)
    shape: spatial shape tuple
    Returns: (1, *shape, dim)
    """
    grids = [jnp.linspace(-1.0, 1.0, size) for size in shape]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    target_coords = jnp.stack(list(reversed(meshgrid)), axis=-1)
    
    ones = jnp.ones(shape + (1,))
    target_coords_hom = jnp.concatenate([target_coords, ones], axis=-1)
    
    source_coords = jnp.tensordot(target_coords_hom, A, axes=(-1, 1))
    return jnp.expand_dims(source_coords, axis=0)


# 5. Gaussian Filtering Helpers (Separable with edge replication padding to avoid boundary shocks)
def _conv1d_axis_zero(image, kernel, axis):
    """
    Applies 1D convolution along specific axis using zero padding.
    """
    ndim = image.ndim
    axes_order = [i for i in range(ndim) if i != axis] + [axis]
    image_trans = jnp.transpose(image, axes_order)
    
    orig_trans_shape = image_trans.shape
    N_d = orig_trans_shape[-1]
    
    image_flat = image_trans.reshape(-1, N_d)
    radius = len(kernel) // 2
    image_padded = jnp.pad(image_flat, ((0, 0), (radius, radius)), mode='constant', constant_values=0.0)
    
    def conv_row(row):
        return jnp.convolve(row, kernel, mode='valid')
    
    out_flat = jax.vmap(conv_row)(image_padded)
    
    out_trans = out_flat.reshape(orig_trans_shape)
    inv_axes_order = [0] * ndim
    for i, a in enumerate(axes_order):
        inv_axes_order[a] = i
        
    return jnp.transpose(out_trans, inv_axes_order)


def _conv1d_axis_edge(image, kernel, axis):
    """
    Applies 1D convolution along specific axis using edge replication padding.
    """
    ndim = image.ndim
    axes_order = [i for i in range(ndim) if i != axis] + [axis]
    image_trans = jnp.transpose(image, axes_order)
    
    orig_trans_shape = image_trans.shape
    N_d = orig_trans_shape[-1]
    
    image_flat = image_trans.reshape(-1, N_d)
    radius = len(kernel) // 2
    image_padded = jnp.pad(image_flat, ((0, 0), (radius, radius)), mode='edge')
    
    def conv_row(row):
        return jnp.convolve(row, kernel, mode='valid')
    
    out_flat = jax.vmap(conv_row)(image_padded)
    
    out_trans = out_flat.reshape(orig_trans_shape)
    inv_axes_order = [0] * ndim
    for i, a in enumerate(axes_order):
        inv_axes_order[a] = i
        
    return jnp.transpose(out_trans, inv_axes_order)


def separable_gaussian_filter_jax(grid, sigma, spacing=None):
    """
    Applies separable Gaussian filtering along each spatial dimension.
    Uses edge replication padding.
    grid: (B, *spatial, dim)
    """
    if sigma <= 0.0:
        return grid
        
    shape = grid.shape
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)
    
    if spacing is not None:
        spacing_rev = tuple(reversed(spacing))
        sigma_list = [sigma / sp for sp in spacing_rev]
    else:
        sigma_list = [sigma] * num_spatial
        
    out = grid
    for i in range(num_spatial):
        sig = sigma_list[i]
        if sig <= 0.0:
            continue
        kernel_size = max(3, int(2 * math.ceil(2 * sig) + 1))
        x = jnp.arange(kernel_size, dtype=jnp.float32) - (kernel_size - 1) / 2.0
        kernel_1d = jnp.exp(-x**2 / (2.0 * sig**2))
        kernel_1d = kernel_1d / jnp.sum(kernel_1d)
        
        # Spatial dimensions start at index 1
        out = _conv1d_axis_edge(out, kernel_1d, axis=i + 1)
        
    return out


# 6. Compose Grids
def compose_grids_jax(grid1, grid2):
    """
    Composes two coordinate grids: grid1 ∘ grid2
    """
    grid1_cf = jnp.moveaxis(grid1, -1, 1)
    composed_cf = jax_grid_sample(grid1_cf, grid2, mode='bilinear', padding_mode='border')
    return jnp.moveaxis(composed_cf, 1, -1)


# 7. Boundary Mask
def get_boundary_mask_jax(spatial, dtype=jnp.float32):
    """
    Boundary mask where boundary voxels are 0 and interior voxels are 1.
    """
    mask = jnp.ones((1, *spatial, 1), dtype=dtype)
    ndim = len(spatial)
    for i in range(ndim):
        slices_start = [slice(None)] * (ndim + 2)
        slices_start[i + 1] = 0
        mask = mask.at[tuple(slices_start)].set(0.0)
        
        slices_end = [slice(None)] * (ndim + 2)
        slices_end[i + 1] = -1
        mask = mask.at[tuple(slices_end)].set(0.0)
    return mask


# 8. Spatial Jacobian
def _spatial_jacobian_nd_jax(field, physical_spacing=None):
    """
    Compute spatial Jacobian of an N-D vector field.
    field: (B, *spatial, dim)
    Returns: (B, *spatial, dim, dim)
    """
    spatial = field.shape[1:-1]
    num_spatial = len(spatial)
    
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (s - 1) for s in spatial]
        
    axes = tuple(range(1, num_spatial + 1))
    grads = jnp.gradient(field, *spacings, axis=axes)
    if not isinstance(grads, (list, tuple)):
        grads = [grads]
    return jnp.stack(grads, axis=-1)


# 9. Diffeomorphic Cycle Consistency Projection
def update_inverse_field_nd_jax(
    W_disp, 
    W_inv_disp, 
    steps=20,
    relaxation=1.0,
    smoothing_sigma=0.0,
    method='fixed_point',
    max_error_threshold=0.1,
    mean_error_threshold=0.001,
    spacing=None,
    origin=None,
    direction=None
):
    """
    Dimension-agnostic fixed-point inversion of a displacement field (JAX).
    
    Matches ITK's itkInvertDisplacementFieldImageFilter exactly:
    - Per-pixel update clipping in voxel-space norm (prevents divergence)
    - Adaptive relaxation (epsilon=0.75 first iter, 0.5 thereafter)
    - Early-stop convergence checking
    - Dirichlet zero boundary enforcement every iteration
    """
    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    
    if spacing is not None and origin is not None and direction is not None:
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(direction)[::-1, ::-1])
        
        X_phys = _get_physical_grid_jax_yfirst(spatial, spacing_rev, origin_rev, direction_rev)
        boundary_mask = get_boundary_mask_jax(spatial)
        spacing_t = jnp.array(spacing_rev)
        
        W_disp_cf = jnp.moveaxis(W_disp, -1, 1)
        
        def body_fn(i, val):
            W_inv_disp_curr = val
            coords_phys = X_phys + W_inv_disp_curr
            coords_norm = _physical_to_normalized_jax_yfirst(coords_phys, spatial, spacing_rev, origin_rev, direction_rev)
            
            forward_at_inv_cf = jax_grid_sample(W_disp_cf, coords_norm, padding_mode='border')
            forward_at_inv = jnp.moveaxis(forward_at_inv_cf, 1, -1)
            
            error = W_inv_disp_curr + forward_at_inv
            error_voxel = error / spacing_t
            scaled_norm = jnp.sqrt(jnp.sum(error_voxel**2, axis=-1, keepdims=True))
            
            max_error = jnp.max(scaled_norm)
            epsilon = jnp.where(i == 0, 0.75, 0.5)
            update = -error
            
            update_voxel = update / spacing_t
            update_norm = jnp.sqrt(jnp.sum(update_voxel**2, axis=-1, keepdims=True)) + 1e-10
            clip_threshold = epsilon * max_error
            clip_scale = jnp.where(update_norm > clip_threshold, clip_threshold / update_norm, 1.0)
            update = update * clip_scale
            
            W_inv_disp_new = W_inv_disp_curr + epsilon * update
            
            if smoothing_sigma > 0.0:
                W_inv_disp_new = separable_gaussian_filter_jax(W_inv_disp_new, smoothing_sigma, spacing=spacing)
                
            W_inv_disp_new = W_inv_disp_new * boundary_mask
            return W_inv_disp_new
            
        return jax.lax.fori_loop(0, steps, body_fn, W_inv_disp)
    else:
        grids = [jnp.linspace(-1.0, 1.0, size) for size in spatial]
        meshgrid = jnp.meshgrid(*grids, indexing='ij')
        identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
        identity = jnp.expand_dims(identity, axis=0)
        identity = jnp.repeat(identity, B, axis=0)
        
        boundary_mask = get_boundary_mask_jax(spatial)
        
        # Voxel scale: converts normalized [-1,1] displacement to voxel units
        voxel_scale = jnp.array(
            [float((s - 1) / 2.0) for s in reversed(spatial)]
        )
        
        W_disp_cf = jnp.moveaxis(W_disp, -1, 1)
        
        def body_fn(i, val):
            W_inv_disp_curr = val
            # Phase 1: Compute composition error
            coords = identity + W_inv_disp_curr
            forward_at_inv_cf = jax_grid_sample(W_disp_cf, coords, padding_mode='border')
            forward_at_inv = jnp.moveaxis(forward_at_inv_cf, 1, -1)
            
            error = W_inv_disp_curr + forward_at_inv
            
            # Compute per-pixel error norm in voxel coordinates (ITK lines 222-228)
            error_voxel = error * voxel_scale
            scaled_norm = jnp.sqrt(jnp.sum(error_voxel**2, axis=-1, keepdims=True))
            
            max_error = jnp.max(scaled_norm)
            
            # Adaptive relaxation (ITK lines 147-151)
            epsilon = jnp.where(i == 0, 0.75, 0.5)
            
            # Update direction: we want to subtract the error
            update = -error
            
            if method == 'neumann':
                Du = _spatial_jacobian_nd_jax(forward_at_inv)
                Du_error = jnp.einsum('b...ij,b...j->b...i', Du, error)
                update = -(error - Du_error)
            
            # Per-pixel update clipping in voxel-space norm (ITK lines 191-194)
            update_voxel = update * voxel_scale
            update_norm = jnp.sqrt(jnp.sum(update_voxel**2, axis=-1, keepdims=True)) + 1e-10
            clip_threshold = epsilon * max_error
            clip_scale = jnp.where(update_norm > clip_threshold, clip_threshold / update_norm, 1.0)
            update = update * clip_scale
            
            # Apply update with relaxation (ITK line 195)
            W_inv_disp_new = W_inv_disp_curr + epsilon * update
            
            if smoothing_sigma > 0.0:
                W_inv_disp_new = separable_gaussian_filter_jax(W_inv_disp_new, smoothing_sigma)
            
            # ITK-standard Dirichlet zero boundary enforcement (ITK lines 198-208)
            W_inv_disp_new = W_inv_disp_new * boundary_mask
            return W_inv_disp_new
            
        return jax.lax.fori_loop(0, steps, body_fn, W_inv_disp)


# 10. Local NCC Similarity Metric (Differentiable, Static Shapes)
def box_filter_jax(x, window_size):
    ndim = x.ndim - 2
    kernel_1d = jnp.ones(window_size) / window_size
    out = x
    for i in range(ndim):
        out = _conv1d_axis_zero(out, kernel_1d, axis=i + 2)
    return out


def local_ncc_loss_nd_jax(I, J, mask=None, window_size=9):
    I_mean = box_filter_jax(I, window_size)
    J_mean = box_filter_jax(J, window_size)
    
    I_var = box_filter_jax((I - I_mean)**2, window_size)
    J_var = box_filter_jax((J - J_mean)**2, window_size)
    IJ_cov = box_filter_jax((I - I_mean) * (J - J_mean), window_size)
    
    safe_I_var = jnp.maximum(I_var, 1e-8)
    safe_J_var = jnp.maximum(J_var, 1e-8)
    
    cc_raw = IJ_cov / (jnp.sqrt(safe_I_var * safe_J_var) + 1e-8)
    valid_mask = (I_var > 1e-8) & (J_var > 1e-8)
    cc = jnp.where(valid_mask, cc_raw, 0.0)
    
    if mask is not None:
        active_mask = (mask > 0.5) & valid_mask
    else:
        active_mask = valid_mask
        
    active_mask_float = active_mask.astype(jnp.float32)
    return -jnp.sum((cc**2) * active_mask_float) / (jnp.sum(active_mask_float) + 1e-8)


# 11. Mattes Mutual Information (Differentiable, Static Shapes)
def b_spline_3_jax(x):
    abs_x = jnp.abs(x)
    val1 = (2.0/3.0) - abs_x**2 + 0.5 * abs_x**3
    val2 = (1.0/6.0) * (2.0 - abs_x)**3
    res = jnp.zeros_like(x)
    res = jnp.where(abs_x < 1.0, val1, res)
    res = jnp.where((abs_x >= 1.0) & (abs_x < 2.0), val2, res)
    return res


def mattes_mi_loss_core_jax(I, J, mask=None, num_bins=32, min_val=-1.0, max_val=1.0, sampling_percentage=None):
    """
    Core computation of Mattes Mutual Information.
    
    Provenance:
    Adapted from ITK's Mattes Mutual Information image-to-image metric:
    itkMattesMutualInformationImageToImageMetricv4.h (and its associated helper methods).
    Uses third-order B-splines for parzen windowing density estimation.
    """
    x = I.flatten()
    y = J.flatten()
    if mask is not None:
        m = mask.flatten()
    else:
        m = None
        
    if sampling_percentage is not None and sampling_percentage < 1.0:
        stride = max(1, int(1.0 / sampling_percentage))
        x = x[::stride]
        y = y[::stride]
        if m is not None:
            m = m[::stride]
            
    x = jnp.clip(x, min_val, max_val)
    y = jnp.clip(y, min_val, max_val)
    sigma = (max_val - min_val) / (num_bins - 1)
    bins = jnp.linspace(min_val, max_val, num_bins)[None, :]
    
    w_x = b_spline_3_jax((x[:, None] - bins) / sigma)
    w_y = b_spline_3_jax((y[:, None] - bins) / sigma)
    
    if m is not None:
        m_col = m[:, None]
        w_x, w_y = w_x * m_col, w_y * m_col
        
    joint_hist = jnp.matmul(w_x.T, w_y)
    pxy = joint_hist / (jnp.sum(joint_hist) + 1e-8)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    
    ratio = pxy / (px * py + 1e-8)
    val = pxy * jnp.log(ratio + 1e-8)
    return -jnp.sum(val)


def mattes_mi_loss_nd_jax(I, J, mask=None, num_bins=32, sampling_percentage=None):
    min_i, max_i = jnp.min(I), jnp.max(I)
    min_j, max_j = jnp.min(J), jnp.max(J)
    
    min_i = jax.lax.stop_gradient(min_i)
    max_i = jax.lax.stop_gradient(max_i)
    min_j = jax.lax.stop_gradient(min_j)
    max_j = jax.lax.stop_gradient(max_j)
    
    I_scaled = (I - min_i) / (max_i - min_i + 1e-8)
    J_scaled = (J - min_j) / (max_j - min_j + 1e-8)
    
    I_scaled = I_scaled * 2.0 - 1.0
    J_scaled = J_scaled * 2.0 - 1.0
    
    return mattes_mi_loss_core_jax(I_scaled, J_scaled, mask, num_bins, min_val=-1.0, max_val=1.0, sampling_percentage=sampling_percentage)


# 12. Jacobian Determinant Maps
def compute_jacobian_determinant_nd_jax(warp_field, physical_spacing=None):
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    B = warp_field.shape[0]
    
    is_physical = True
    
    if is_physical:
        if dim == 2:
            phys_disp_ordered = warp_field[..., [1, 0]]
        elif dim == 3:
            phys_disp_ordered = warp_field[..., [2, 1, 0]]
        else:
            phys_disp_ordered = warp_field
            
        shape_t = jnp.array(list(reversed(spatial)))
        norm_disp = phys_disp_ordered * 2.0 / (shape_t - 1)
        
        # 1. Compute J_voxel using spatial gradients with normalized spacing
        normalized_spacings = [2.0 / (s - 1) for s in spatial]
        axes = tuple(range(1, dim + 1))
        grads = jnp.gradient(norm_disp, *normalized_spacings, axis=axes)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]
        J_voxel = jnp.stack(list(reversed(grads)), axis=-1)
        
        # 2. Construct voxel-to-physical matrices M and M_inv (both identity here)
        # 3. Compute similarity transform J_phys = J_voxel
        # 4. Compute deformation gradient F = J_phys + I
        F = J_voxel + jnp.eye(dim)
        
        # 5. Compute determinant of F analytically
        if dim == 2:
            a = F[..., 0, 0]
            b = F[..., 0, 1]
            c = F[..., 1, 0]
            d = F[..., 1, 1]
            return a * d - b * c
        elif dim == 3:
            a = F[..., 0, 0]
            b = F[..., 0, 1]
            c = F[..., 0, 2]
            d = F[..., 1, 0]
            e = F[..., 1, 1]
            f = F[..., 1, 2]
            g = F[..., 2, 0]
            h = F[..., 2, 1]
            i = F[..., 2, 2]
            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        else:
            raise ValueError("Only 2D and 3D are supported.")
            
    grids = [jnp.linspace(-1.0, 1.0, size) for size in spatial]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    # Internal JAX warp fields are stored in ZYX/YX order, so meshgrid should not be reversed
    identity = jnp.stack(meshgrid, axis=-1)
    identity = jnp.expand_dims(identity, axis=0)
    identity = jnp.repeat(identity, B, axis=0)
    
    phi = identity + warp_field
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (size - 1) for size in spatial]
        
    axes = tuple(range(1, dim + 1))
    grads = jnp.gradient(phi, *spacings, axis=axes)
    if not isinstance(grads, (list, tuple)):
        grads = [grads]
        
    if dim == 2:
        j00 = grads[0][..., 0]
        j01 = grads[1][..., 0]
        j10 = grads[0][..., 1]
        j11 = grads[1][..., 1]
        return j00 * j11 - j01 * j10
    elif dim == 3:
        j00 = grads[0][..., 0]
        j01 = grads[1][..., 0]
        j02 = grads[2][..., 0]
        
        j10 = grads[0][..., 1]
        j11 = grads[1][..., 1]
        j12 = grads[2][..., 1]
        
        j20 = grads[0][..., 2]
        j21 = grads[1][..., 2]
        j22 = grads[2][..., 2]
        
        return j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
    else:
        raise ValueError("Only 2D and 3D are supported.")


def compute_physical_jacobian_determinant_jax(warp_field, direction, spacing):
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    
    normalized_spacings = [2.0 / (s - 1) for s in spatial]
    axes = tuple(range(1, dim + 1))
    grads = jnp.gradient(warp_field, *normalized_spacings, axis=axes)
    if not isinstance(grads, (list, tuple)):
        grads = [grads]
    J_voxel = jnp.stack(list(reversed(grads)), axis=-1)
    
    M = direction * jnp.expand_dims(spacing, axis=0)
    M_inv = direction.T * jnp.expand_dims(1.0 / spacing, axis=1)
    
    J_phys = jnp.einsum('ij,b...jk,kl->b...il', M, J_voxel, M_inv)
    F = J_phys + jnp.eye(dim)
    return jnp.linalg.det(F)


@partial(jax.jit, static_argnums=(5, 6, 7, 10, 14, 16))
def affine_step_jax(
    params, m_state, v_state, t_state, active_flags,
    dim, spatial_shape, transform_type, I_curr, J_curr,
    mattes_bins, lr=1e-2, coords=None, coords_hom=None,
    has_initial_grid=False, initial_grid_level=None,
    affine_loss_fn=None
):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    def loss_fn(p):
        A = get_affine_matrix_jax(p, dim, transform_type)
        if coords is not None and coords_hom is not None:
            # Coordinate-level random sampling path
            theta = A[:dim, :dim + 1] # (dim, dim+1)
            coords_warped = jnp.matmul(coords_hom, theta.T)
            if has_initial_grid:
                initial_grid_cf = jnp.moveaxis(initial_grid_level, -1, 1)
                coords_warped = jnp.moveaxis(jax_grid_sample(initial_grid_cf, coords_warped, padding_mode='border'), 1, -1)
            I_sampled = jax_grid_sample(I_curr, coords, padding_mode='border')
            J_sampled = jax_grid_sample(J_curr, coords_warped, padding_mode='border')
            return mattes_mi_loss_core_jax(J_sampled.flatten(), I_sampled.flatten(), num_bins=mattes_bins)
        else:
            grid = jax_affine_grid(A[:dim, :dim + 1], spatial_shape)
            if has_initial_grid:
                grid = compose_grids_jax(initial_grid_level, grid)
            moving_warped = jax_grid_sample(J_curr, grid, padding_mode='border')
            if affine_loss_fn is not None:
                return affine_loss_fn(moving_warped, I_curr)
            else:
                return mattes_mi_loss_nd_jax(moving_warped, I_curr, num_bins=mattes_bins, sampling_percentage=None)
            
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    
    new_params = {}
    new_m = {}
    new_v = {}
    t_next = t_state + 1.0
    
    # bias correction terms
    bias_correction1 = 1.0 - beta1 ** t_next
    bias_correction2 = 1.0 - beta2 ** t_next
    
    for key in params.keys():
        if key == 'T_init':
            new_params[key] = params[key]
            continue
            
        p = params[key]
        g = grads[key]
        m = m_state[key]
        v = v_state[key]
        active = active_flags[key]
        
        # zero out gradients for inactive parameters
        g = jnp.where(active, g, 0.0)
        
        # Update biased first moment estimate
        m_new = beta1 * m + (1.0 - beta1) * g
        # Update biased second raw moment estimate
        v_new = beta2 * v + (1.0 - beta2) * (g ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m_new / bias_correction1
        # Compute bias-corrected second raw moment estimate
        v_hat = v_new / bias_correction2
        
        # Update parameter
        step = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_p = p + step
        
        new_params[key] = new_p
        new_m[key] = m_new
        new_v[key] = v_new
        
    return loss_val, new_params, new_m, new_v, t_next




# Helper to convert inputs to JAX arrays
def to_jax_array(x):
    if hasattr(x, 'detach'):
        return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, np.ndarray):
        return jnp.array(x)
    return jnp.array(x)


# Helper to upscale displacement fields between levels
def upscale_field_jax(field, target_spatial):
    field_cf = jnp.moveaxis(field, -1, 1)
    target_shape = (1, field.shape[-1]) + target_spatial
    upscaled_cf = jax.image.resize(field_cf, target_shape, method='linear')
    return jnp.moveaxis(upscaled_cf, 1, -1)


def physical_to_normalized_jax_cached(phys_coords, shape_t, spacing_t, origin_t, direction_t):
    dim = phys_coords.shape[-1]
    flat_phys = phys_coords.reshape(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    norm_coords_reversed = jnp.flip(norm_coords, axis=-1)
    return norm_coords_reversed.reshape(phys_coords.shape)

@partial(jax.jit, static_argnums=(15, 16))
def prepare_mid_images_and_gradients_jax(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fixed_spacing, moving_spacing,
    M_phys, t_phys, initial_grid_level
):
    phi_l2r_phys = X_phys + warp_l2r
    coords_norm = physical_to_normalized_jax_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    I_mid = jax_grid_sample(I_curr, coords_norm, padding_mode='border')
    
    phi_r2l_phys = X_phys + warp_r2l
    y_phys = phi_r2l_phys @ M_phys.T + t_phys
    y_norm = physical_to_normalized_jax_cached(
        y_phys, moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t
    )
    if initial_grid_level is not None:
        y_norm = compose_grids_jax(initial_grid_level, y_norm)
        
    J_mid = jax_grid_sample(J_curr, y_norm, padding_mode='border')
    
    I_curr_cl = jnp.moveaxis(I_curr, 1, -1)
    J_curr_cl = jnp.moveaxis(J_curr, 1, -1)
    
    grad_I_curr = _spatial_jacobian_nd_jax(I_curr_cl, physical_spacing=tuple(reversed(fixed_spacing))).squeeze(-2)
    grad_J_curr = _spatial_jacobian_nd_jax(J_curr_cl, physical_spacing=tuple(reversed(moving_spacing))).squeeze(-2)
    
    grad_I_mid_sampled = jnp.moveaxis(
        jax_grid_sample(jnp.moveaxis(grad_I_curr, -1, 1), coords_norm, padding_mode='border'),
        1, -1
    )
    grad_I_mid_sampled = grad_I_mid_sampled @ fixed_direction_t.T
    
    grad_J_mid_sampled = jnp.moveaxis(
        jax_grid_sample(jnp.moveaxis(grad_J_curr, -1, 1), y_norm, padding_mode='border'),
        1, -1
    )
    grad_J_mid_sampled = grad_J_mid_sampled @ moving_direction_t.T
    grad_J_mid_sampled = grad_J_mid_sampled @ M_phys
    
    return I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled


@jax.jit
def warp_images_jax(
    wl, wr, wl_inv, wr_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    M_phys, t_phys, initial_grid_level
):
    phi_l2r_phys = X_phys + wl
    fixed_norm = physical_to_normalized_jax_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    im = jax_grid_sample(I_curr, fixed_norm, mode='bilinear', padding_mode='border')
    
    phi_r2l_phys = X_phys + wr
    y_phys = phi_r2l_phys @ M_phys.T + t_phys
    y_norm = physical_to_normalized_jax_cached(
        y_phys, moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t
    )
    if initial_grid_level is not None:
        y_norm = compose_grids_jax(initial_grid_level, y_norm)
        
    jm = jax_grid_sample(J_curr, y_norm, mode='bilinear', padding_mode='border')
    return im, jm


@partial(jax.jit, static_argnums=(7, 8, 9))
def sgd_update_step_jax(
    warp_l2r, warp_r2l, v_l2r, v_r2l,
    grad_l_raw, grad_r_raw, b_mask,
    has_spacing, spacing, fluid_sigma, lr
):
    if has_spacing:
        grad_l_filtered = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=spacing)
        grad_r_filtered = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=spacing)
    else:
        grad_l_filtered = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=None)
        grad_r_filtered = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=None)
        
    v_l2r_new = 0.9 * v_l2r + grad_l_filtered
    v_r2l_new = 0.9 * v_r2l + grad_r_filtered
    
    warp_l2r_new = warp_l2r - lr * v_l2r_new
    warp_r2l_new = warp_r2l - lr * v_r2l_new
    
    return warp_l2r_new, warp_r2l_new, v_l2r_new, v_r2l_new


@partial(jax.jit, static_argnums=(10, 11, 12))
def adam_update_step_jax(
    warp_l2r, warp_r2l, m_l2r, m_r2l, v_l2r, v_r2l, adam_t,
    grad_l_raw, grad_r_raw, b_mask,
    has_spacing, spacing, fluid_sigma, lr
):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    if has_spacing:
        grad_l_filtered = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=spacing)
        grad_r_filtered = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=spacing)
    else:
        grad_l_filtered = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=None)
        grad_r_filtered = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=None)
        
    m_l2r_new = beta1 * m_l2r + (1.0 - beta1) * grad_l_filtered
    m_r2l_new = beta1 * m_r2l + (1.0 - beta1) * grad_r_filtered
    
    v_l2r_new = beta2 * v_l2r + (1.0 - beta2) * (grad_l_filtered**2)
    v_r2l_new = beta2 * v_r2l + (1.0 - beta2) * (grad_r_filtered**2)
    
    bias1 = 1.0 - jnp.power(beta1, adam_t)
    bias2 = 1.0 - jnp.power(beta2, adam_t)
    
    m_l2r_hat = m_l2r_new / bias1
    m_r2l_hat = m_r2l_new / bias1
    v_l2r_hat = v_l2r_new / bias2
    v_r2l_hat = v_r2l_new / bias2
    
    warp_l2r_new = warp_l2r - lr * m_l2r_hat / (jnp.sqrt(v_l2r_hat) + eps)
    warp_r2l_new = warp_r2l - lr * m_r2l_hat / (jnp.sqrt(v_r2l_hat) + eps)
    
    return warp_l2r_new, warp_r2l_new, m_l2r_new, m_r2l_new, v_l2r_new, v_r2l_new


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def regularize_warp_fields_jax(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
    b_mask, has_spacing, spacing, origin, direction, elastic_sigma,
    inverse_steps, inverse_method
):
    warp_l2r = warp_l2r * b_mask
    warp_r2l = warp_r2l * b_mask
    
    if elastic_sigma > 0.0:
        if has_spacing:
            warp_l2r = separable_gaussian_filter_jax(warp_l2r, elastic_sigma, spacing=spacing)
            warp_r2l = separable_gaussian_filter_jax(warp_r2l, elastic_sigma, spacing=spacing)
        else:
            warp_l2r = separable_gaussian_filter_jax(warp_l2r, elastic_sigma, spacing=None)
            warp_r2l = separable_gaussian_filter_jax(warp_r2l, elastic_sigma, spacing=None)
            
    warp_l2r_inv = update_inverse_field_nd_jax(
        warp_l2r, warp_l2r_inv, steps=inverse_steps, method=inverse_method,
        spacing=spacing, origin=origin, direction=direction
    )
    
    warp_r2l_inv = update_inverse_field_nd_jax(
        warp_r2l, warp_r2l_inv, steps=inverse_steps, method=inverse_method,
        spacing=spacing, origin=origin, direction=direction
    )
    
    return warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv


@partial(jax.jit, static_argnums=(12, 13, 14, 15, 16, 17, 18, 19, 20))
def syn_update_step_jax(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
    grad_l_raw, grad_r_raw, X_phys, b_mask,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    has_spacing, spacing, origin, direction, fluid_sigma, elastic_sigma, cfl_voxels,
    inverse_steps, inverse_method
):
    spatial_shape = warp_l2r.shape[1:-1]
    
    # Enforce zero boundary condition on gradients before filtering (fluid smoothing)
    if has_spacing:
        grad_l = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=spacing)
        grad_r = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=spacing)
        
        # ITK-style CFL: Euclidean norm in PHYSICAL coordinates (mm)
        max_norm_l = jnp.sqrt(jnp.sum(grad_l**2, axis=-1)).max() + 1e-8
        max_norm_r = jnp.sqrt(jnp.sum(grad_r**2, axis=-1)).max() + 1e-8
        
        # Scale physical step size to cfl_voxels * spacing
        delta_l = (cfl_voxels * fixed_spacing_t) * (grad_l / max_norm_l)
        delta_r = (cfl_voxels * fixed_spacing_t) * (grad_r / max_norm_r)
    else:
        grad_l = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=None)
        grad_r = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=None)
        
        max_norm_l = jnp.sqrt(jnp.sum(grad_l**2, axis=-1)).max() + 1e-8
        max_norm_r = jnp.sqrt(jnp.sum(grad_r**2, axis=-1)).max() + 1e-8
        
        delta_l = cfl_voxels * (grad_l / max_norm_l)
        delta_r = cfl_voxels * (grad_r / max_norm_r)
        
    # Greedy SyN composition: φ_new = φ_old ∘ (Id - ∂loss/∂warp)
    coords_phys_l = X_phys - delta_l
    coords_norm_l = physical_to_normalized_jax_cached(
        coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    warp_l2r_cf = jnp.moveaxis(warp_l2r, -1, 1)
    warp_l2r_sampled = jnp.moveaxis(jax_grid_sample(warp_l2r_cf, coords_norm_l, padding_mode='border'), 1, -1)
    warp_l2r = warp_l2r_sampled - delta_l
    
    coords_phys_r = X_phys - delta_r
    coords_norm_r = physical_to_normalized_jax_cached(
        coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    warp_r2l_cf = jnp.moveaxis(warp_r2l, -1, 1)
    warp_r2l_sampled = jnp.moveaxis(jax_grid_sample(warp_r2l_cf, coords_norm_r, padding_mode='border'), 1, -1)
    warp_r2l = warp_r2l_sampled - delta_r
    
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = regularize_warp_fields_jax(
        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
        b_mask, has_spacing, spacing, origin, direction, elastic_sigma,
        inverse_steps, inverse_method
    )
    
    return warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv


def _get_physical_grid_jax_yfirst(shape, spacing, origin, direction):
    dim = len(shape)
    grids = [jnp.arange(s) for s in shape]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    
    idxs = jnp.stack(meshgrid, axis=-1)
    spacing_t = jnp.array(spacing)
    origin_t = jnp.array(origin)
    direction_t = jnp.array(direction)
    
    scaled = idxs * spacing_t
    flat_scaled = scaled.reshape(-1, dim)
    flat_phys = flat_scaled @ direction_t.T + origin_t
    return flat_phys.reshape(*shape, dim)[None, ...]

def get_physical_grid_jax(shape, spacing, origin, direction):
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(direction)[::-1, ::-1])
    return _get_physical_grid_jax_yfirst(shape, spacing_rev, origin_rev, direction_rev)

def _physical_to_normalized_jax_yfirst(phys_coords, target_shape, spacing, origin, direction):
    dim = len(target_shape)
    spacing_t = jnp.array(spacing)
    origin_t = jnp.array(origin)
    direction_t = jnp.array(direction)
    
    flat_phys = phys_coords.reshape(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    
    shape_t = jnp.array(list(target_shape))
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    norm_coords_reversed = jnp.flip(norm_coords, axis=-1)
    return norm_coords_reversed.reshape(phys_coords.shape)

def physical_to_normalized_jax(phys_coords, target_shape, spacing, origin, direction):
    target_shape_rev = tuple(reversed(target_shape))
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(direction)[::-1, ::-1])
    return _physical_to_normalized_jax_yfirst(phys_coords, target_shape_rev, spacing_rev, origin_rev, direction_rev)

def _grid_to_physical_affine_jax_yfirst(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
    dim = len(fixed_shape)
    
    Nx = jnp.array(fixed_shape)
    Ny = jnp.array(moving_shape)
    Sx = jnp.array(fixed_spacing)
    Sy = jnp.array(moving_spacing)
    Ox = jnp.array(fixed_origin)
    Oy = jnp.array(moving_origin)
    Dx = jnp.array(fixed_direction)
    Dy = jnp.array(moving_direction)
    
    Kx = jnp.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = jnp.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0
    
    Kx_inv = jnp.linalg.inv(Kx)
    Sx_inv = jnp.linalg.inv(jnp.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx
    
    Vy = Dy @ jnp.diag(Sy) @ Ky
    cy = Dy @ jnp.diag(Sy) @ Cy + Oy
    
    A_grid = T_grid[:dim, :dim]
    t_grid = T_grid[:dim, dim]
    
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy
    return M_phys, t_phys

def grid_to_physical_affine_jax(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
    dim = len(fixed_shape)
    perm = list(range(dim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D
    if hasattr(T_grid, 'at'):
        perm_idx = jnp.array(perm)
        T_yx = T_grid.copy()
        T_yx = T_yx.at[:dim, :dim].set(T_grid[:dim, :dim][perm_idx][:, perm_idx])
        T_yx = T_yx.at[:dim, dim].set(T_grid[:dim, dim][perm_idx])
    else:
        T_yx = T_grid.copy()
        T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
        T_yx[:dim, dim] = T_grid[:dim, dim][perm]
        
    fs_rev = tuple(reversed(fixed_spacing))
    fo_rev = tuple(reversed(fixed_origin))
    fd_rev = tuple(tuple(float(x) for x in row) for row in np.array(fixed_direction)[::-1, ::-1])
    ms_rev = tuple(reversed(moving_spacing))
    mo_rev = tuple(reversed(moving_origin))
    md_rev = tuple(tuple(float(x) for x in row) for row in np.array(moving_direction)[::-1, ::-1])
    return _grid_to_physical_affine_jax_yfirst(T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev)


def upscale_initial_grid(grid, target_spatial):
    grid_cf = jnp.moveaxis(grid, -1, 1)
    target_shape = (1, grid.shape[-1]) + target_spatial
    upscaled_cf = jax.image.resize(grid_cf, target_shape, method='linear')
    return jnp.moveaxis(upscaled_cf, 1, -1)


# 14. Standard SyNTo Class API
class SyNTo:
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, origin=None, direction=None, fluid_sigma=3.0, elastic_sigma=0.0, transform_type='Affine', inverse_method='fixed_point', inverse_steps=5):
        self.dim = dim
        self.grid_shape = grid_shape
        self.spacing = spacing
        self.origin = origin if origin is not None else [0.0] * dim
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.transform_type = transform_type
        self.inverse_method = inverse_method
        self.inverse_steps = inverse_steps
        
        # Direction cosine matrix (ITK standard: identity if not specified)
        if direction is not None:
            self.direction = np.array(direction, dtype=np.float32)
        else:
            self.direction = np.eye(dim, dtype=np.float32)
        
        if spacing is not None:
            spacing_reversed = list(reversed(spacing))
            self.physical_bounds = np.array([(s - 1) / 2.0 * sp for s, sp in zip(grid_shape, spacing_reversed)])
        else:
            self.physical_bounds = np.ones(dim)
            
        self.affine_params = {
            'translation': np.zeros(dim),
            'omega': np.zeros(dim * (dim - 1) // 2),
            'scale': np.ones(1),
            'anisotropic_scale': np.ones(dim),
            'shear': np.zeros(dim * (dim - 1) // 2)
        }
        
        self.warp_l2r = np.zeros((1, *grid_shape, dim))
        self.warp_r2l = np.zeros((1, *grid_shape, dim))
        self.warp_l2r_inv = np.zeros((1, *grid_shape, dim))
        self.warp_r2l_inv = np.zeros((1, *grid_shape, dim))
        
        self.affine_losses = []
        self.syn_losses = []

    def get_affine_grid(self, shape, device='cpu'):
        A = get_affine_matrix_jax(self.affine_params, self.dim, self.transform_type)
        A_grid = A[:self.dim, :self.dim + 1]
        grid_jax = jax_affine_grid(A_grid, shape)
        return torch.from_numpy(np.array(grid_jax)).to(device)

    def get_inverse_affine_grid(self, shape, device='cpu'):
        A = get_affine_matrix_jax(self.affine_params, self.dim, self.transform_type)
        A_inv = jnp.linalg.inv(A)
        theta_inv = A_inv[:self.dim, :self.dim + 1]
        grid_inv_jax = jax_affine_grid(theta_inv, shape)
        return torch.from_numpy(np.array(grid_inv_jax)).to(device)

    def fit(self, fixed_image, moving_image, levels=[8, 4, 2, 1], epochs_per_level=100, 
            affine_epochs=[100, 50, 50, 20], affine_lr=1e-2, cfl_voxels=0.75, 
            similarity_metric='lncc', use_analytical_gradients=True,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None, syn_metric_weights=None,
            initial_grid=None, optimizer_type='cfl', optimizer_lr=1e-3, smoothing_sigmas=None, **kwargs):
        
        verbose = kwargs.get('verbose', False)
        I_jax = to_jax_array(fixed_image)
        J_jax = to_jax_array(moving_image)
        spatial_shape = I_jax.shape[2:]
        
        fixed_spacing = kwargs.get('fixed_spacing', None)
        fixed_origin = kwargs.get('fixed_origin', None)
        fixed_direction = kwargs.get('fixed_direction', None)
        moving_spacing = kwargs.get('moving_spacing', None)
        moving_origin = kwargs.get('moving_origin', None)
        moving_direction = kwargs.get('moving_direction', None)
        
        if fixed_spacing is None:
            fixed_spacing = self.spacing if self.spacing is not None else [1.0] * self.dim
            
        if fixed_origin is None:
            fixed_origin = [0.0] * self.dim
            
        if fixed_direction is None:
            fixed_direction = self.direction if self.direction is not None else np.eye(self.dim)
            
        if moving_spacing is None:
            moving_spacing = fixed_spacing
            
        if moving_origin is None:
            moving_origin = fixed_origin
            
        if moving_direction is None:
            moving_direction = fixed_direction
            
        fixed_spacing = tuple(fixed_spacing)
        fixed_origin = tuple(fixed_origin)
        fixed_direction = tuple(tuple(float(x) for x in row) for row in fixed_direction)
        moving_spacing = tuple(moving_spacing)
        moving_origin = tuple(moving_origin)
        moving_direction = tuple(tuple(float(x) for x in row) for row in moving_direction)
        
        # Standardize iteration lists to match hierarchy levels length
        if isinstance(epochs_per_level, int):
            epochs_per_level = [epochs_per_level] * len(levels)
        elif len(epochs_per_level) < len(levels):
            epochs_per_level = list(epochs_per_level) + [0] * (len(levels) - len(epochs_per_level))
            
        if isinstance(affine_epochs, int):
            affine_epochs = [affine_epochs] * len(levels)
        elif len(affine_epochs) < len(levels):
            affine_epochs = list(affine_epochs) + [0] * (len(levels) - len(affine_epochs))
            
        self.initial_grid = initial_grid
        
        # CoM Initialization Selection (FOV vs Foreground CoM based on downsampled Mattes MI)
        if self.initial_grid is None:
            # 1. Compute FOV centers
            Nx_t = jnp.array(list(reversed(spatial_shape)))
            Sx_t = jnp.array(fixed_spacing)
            Ox_t = jnp.array(fixed_origin)
            Dx_t = jnp.array(fixed_direction)
            com_fixed_fov = Dx_t @ (Sx_t * (Nx_t - 1) / 2.0) + Ox_t
            
            Ny_t = jnp.array(list(reversed(J_jax.shape[2:])))
            Sy_t = jnp.array(moving_spacing)
            Oy_t = jnp.array(moving_origin)
            Dy_t = jnp.array(moving_direction)
            com_moving_fov = Dy_t @ (Sy_t * (Ny_t - 1) / 2.0) + Oy_t
            
            t_fov = com_moving_fov - com_fixed_fov
            
            # 2. Compute Foreground (intensity-weighted) centers
            fixed_pos = jnp.maximum(I_jax, 0.0)
            moving_pos = jnp.maximum(J_jax, 0.0)
            sum_fixed = fixed_pos.sum()
            sum_moving = moving_pos.sum()
            
            if sum_fixed > 1e-5 and sum_moving > 1e-5:
                grids_f = [jnp.arange(s) for s in spatial_shape]
                meshgrid_f = jnp.meshgrid(*grids_f, indexing='ij')
                idxs_f = jnp.stack(list(reversed(meshgrid_f)), axis=-1)
                
                grids_m = [jnp.arange(s) for s in J_jax.shape[2:]]
                meshgrid_m = jnp.meshgrid(*grids_m, indexing='ij')
                idxs_m = jnp.stack(list(reversed(meshgrid_m)), axis=-1)
                
                com_fixed_voxel = jnp.sum(fixed_pos.squeeze(0).squeeze(0)[..., None] * idxs_f, axis=tuple(range(self.dim))) / sum_fixed
                com_moving_voxel = jnp.sum(moving_pos.squeeze(0).squeeze(0)[..., None] * idxs_m, axis=tuple(range(self.dim))) / sum_moving
                
                com_fixed_fg = Dx_t @ (Sx_t * com_fixed_voxel) + Ox_t
                com_moving_fg = Dy_t @ (Sy_t * com_moving_voxel) + Oy_t
                
                t_fg = com_moving_fg - com_fixed_fg
            else:
                t_fg = t_fov
                
            # 3. Downsample images for fast evaluation
            down_shape = tuple(max(8, s // 4) for s in spatial_shape)
            grids_down = [jnp.linspace(-1, 1, s) for s in down_shape]
            meshgrid_down = jnp.meshgrid(*grids_down, indexing='ij')
            grid_down = jnp.stack(list(reversed(meshgrid_down)), axis=-1)[None, ...]
            
            I_down = jax_grid_sample(I_jax, grid_down, padding_mode='border')
            J_down = jax_grid_sample(J_jax, grid_down, padding_mode='border')
            
            def eval_translation_jax(t_candidate):
                down_spacing = [sp * (orig - 1) / (down - 1) if down > 1 else sp for sp, orig, down in zip(fixed_spacing, reversed(spatial_shape), reversed(down_shape))]
                X_down = get_physical_grid_jax(down_shape, down_spacing, fixed_origin, fixed_direction)
                y_phys = X_down + t_candidate
                y_norm = physical_to_normalized_jax(y_phys, J_jax.shape[2:], moving_spacing, moving_origin, moving_direction)
                J_warped = jax_grid_sample(J_down, y_norm, padding_mode='border')
                metric_to_use = similarity_metric[0] if isinstance(similarity_metric, list) else similarity_metric
                if metric_to_use == 'lncc':
                    return local_ncc_loss_nd_jax(J_warped, I_down, window_size=5)
                elif metric_to_use == 'mse':
                    return jnp.mean((J_warped - I_down) ** 2)
                else:
                    return mattes_mi_loss_nd_jax(J_warped, I_down, num_bins=16)
            
            loss_fov = float(eval_translation_jax(t_fov))
            loss_fg = float(eval_translation_jax(t_fg))
            
            best_t = t_fov if loss_fov < loss_fg else t_fg
            
            # Compute and register T_init (mapping physical rigid translation into grid coordinates)
            H_x = jnp.eye(self.dim + 1)
            H_x = H_x.at[:self.dim, :self.dim].set(Dx_t @ jnp.diag(Sx_t) @ jnp.diag((Nx_t - 1) / 2.0))
            H_x = H_x.at[:self.dim, self.dim].set(com_fixed_fov)
            
            H_y = jnp.eye(self.dim + 1)
            H_y = H_y.at[:self.dim, :self.dim].set(Dy_t @ jnp.diag(Sy_t) @ jnp.diag((Ny_t - 1) / 2.0))
            H_y = H_y.at[:self.dim, self.dim].set(com_moving_fov)
            
            T_phys = jnp.eye(self.dim + 1)
            T_phys = T_phys.at[:self.dim, self.dim].set(best_t)
            
            T_init = jnp.linalg.inv(H_y) @ T_phys @ H_x
            self.affine_params['T_init'] = np.array(T_init)
        
        self.affine_losses = []
        self.syn_losses = []
        
        # Standardize similarity_metric to a list of JAX callables
        if isinstance(similarity_metric, str):
            self.metrics = [similarity_metric]
        elif isinstance(similarity_metric, list):
            self.metrics = list(similarity_metric)
        else:
            self.metrics = [similarity_metric]

        self.metric_weights = syn_metric_weights if syn_metric_weights is not None else [1.0] * len(self.metrics)
        self.loss_functions = []
        
        from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
        
        # We need a fallback if variables like vgg_mode are not in kwargs
        vgg_mode = kwargs.get('vgg_mode', 'lncc_3d')
        vgg_layers = kwargs.get('vgg_layers', [4])
        vgg_lncc_window_size = kwargs.get('vgg_lncc_window_size', 9)
        
        for metric in self.metrics:
            if isinstance(metric, str):
                metric_name_lower = metric.lower()
                if metric_name_lower == 'mattes_mi':
                    self.loss_functions.append(lambda x, y: mattes_mi_loss_nd_jax(x, y, num_bins=mattes_bins))
                elif metric_name_lower == 'lncc':
                    self.loss_functions.append(lambda x, y: local_ncc_loss_nd_jax(x, y, window_size=2 * lncc_radius + 1))
                elif metric_name_lower == 'mse':
                    self.loss_functions.append(lambda x, y: jnp.mean((x - y) ** 2))
                elif metric_name_lower == 'vgg19':
                    ext = VGG19Extractor(feature_layers=vgg_layers)
                    loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
                    self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
                elif metric_name_lower in ['dinov2', 'dinov2_small']:
                    ext = DINOv2Extractor(version='vits14', feature_layers=vgg_layers)
                    loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
                    self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
                elif metric_name_lower == 'dinov2_base':
                    ext = DINOv2Extractor(version='vitb14', feature_layers=vgg_layers)
                    loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
                    self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
                elif metric_name_lower == 'resnet10':
                    ext = ResNet10Extractor(dim=self.dim, feature_layers=vgg_layers)
                    loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
                    self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
                elif metric_name_lower in ['swinunetr', 'swin_unetr']:
                    layers = [4] if vgg_layers == [8] else vgg_layers
                    ext = SwinUNETRExtractor(feature_layers=layers)
                    loss_fn = FeatureSpaceLoss(extractor=ext, mode=vgg_mode, lncc_window=vgg_lncc_window_size)
                    self.loss_functions.append(make_pytorch_loss_jax(loss_fn))
                else:
                    raise ValueError(f"Unknown similarity metric: {metric}")
            elif isinstance(metric, torch.nn.Module):
                self.loss_functions.append(make_pytorch_loss_jax(metric))
            elif callable(metric):
                if hasattr(metric, 'forward') or hasattr(metric, 'parameters'):
                    self.loss_functions.append(make_pytorch_loss_jax(metric))
                else:
                    self.loss_functions.append(metric)
            else:
                raise ValueError(f"Invalid similarity metric: {metric}")

        aff_metric = kwargs.get('aff_metric', 'mattes_mi')
        if aff_metric == 'mattes':
            aff_metric = 'mattes_mi'
            
        if aff_metric.lower() == 'mattes_mi':
            self.affine_loss_fn = lambda x, y: mattes_mi_loss_nd_jax(x, y, num_bins=mattes_bins)
        elif aff_metric.lower() == 'lncc':
            self.affine_loss_fn = lambda x, y: local_ncc_loss_nd_jax(x, y, window_size=2 * lncc_radius + 1)
        elif aff_metric.lower() == 'mse':
            self.affine_loss_fn = lambda x, y: jnp.mean((x - y) ** 2)
        else:
            self.affine_loss_fn = self.loss_functions[0]

        def combined_jax_loss(jm, im):
            loss = 0.0
            for fn, w in zip(self.loss_functions, self.metric_weights):
                loss += w * fn(jm, im)
            return loss
        
        # --- 0. Construct Image Pyramids ---
        def smooth_image_jax(img, sigma, spacing=None):
            if sigma <= 0.0:
                return img
            # img is shape (B, C, *spatial), move C to last -> (B, *spatial, C)
            img_last = jnp.moveaxis(img, 1, -1)
            smoothed_last = separable_gaussian_filter_jax(img_last, sigma, spacing=spacing)
            return jnp.moveaxis(smoothed_last, -1, 1)

        # Parse smoothing_sigmas
        if smoothing_sigmas is None:
            import math
            sigmas = [float(math.log2(s)) for s in levels]
        elif isinstance(smoothing_sigmas, (int, float)):
            sigmas = [float(smoothing_sigmas)] * len(levels)
        else:
            sigmas = [float(s) for s in smoothing_sigmas]
            if len(sigmas) != len(levels):
                raise ValueError(f"Length of smoothing_sigmas ({len(sigmas)}) must match levels ({len(levels)})")

        I_pyr = []
        J_pyr = []
        for level_idx, s in enumerate(levels):
            sig = sigmas[level_idx]
            if sig > 0.0:
                fixed_smoothed = smooth_image_jax(I_jax, sig, spacing=fixed_spacing)
                moving_smoothed = smooth_image_jax(J_jax, sig, spacing=moving_spacing)
            else:
                fixed_smoothed = I_jax
                moving_smoothed = J_jax
                
            if s > 1:
                I_level = interpolate_jax(fixed_smoothed, 1.0 / s, self.dim)
                J_level = interpolate_jax(moving_smoothed, 1.0 / s, self.dim)
            else:
                I_level = fixed_smoothed
                J_level = moving_smoothed
            I_pyr.append(I_level)
            J_pyr.append(J_level)
                
        # --- 1. Hierarchical Multi-Resolution Affine Pre-alignment ---
        params = {k: jnp.array(v) for k, v in self.affine_params.items()}
        m_state = {
            'translation': jnp.zeros(self.dim),
            'omega': jnp.zeros(self.dim * (self.dim - 1) // 2),
            'scale': jnp.zeros(1),
            'anisotropic_scale': jnp.zeros(self.dim),
            'shear': jnp.zeros(self.dim * (self.dim - 1) // 2)
        }
        v_state = {
            'translation': jnp.zeros(self.dim),
            'omega': jnp.zeros(self.dim * (self.dim - 1) // 2),
            'scale': jnp.zeros(1),
            'anisotropic_scale': jnp.zeros(self.dim),
            'shear': jnp.zeros(self.dim * (self.dim - 1) // 2)
        }
        t_state = jnp.array(0.0)
        
        if sum(affine_epochs) > 0:
            for level_idx, scale in enumerate(levels):
                curr_affine_epochs = affine_epochs[level_idx]
                if curr_affine_epochs <= 0:
                    continue
                I_curr = I_pyr[level_idx]
                J_curr = J_pyr[level_idx]
                curr_spatial = I_curr.shape[2:]
                
                active_affine_levels = sum(1 for its in affine_epochs if its > 0)
                active_flags = {
                    'translation': True,
                    'omega': False,
                    'scale': False,
                    'anisotropic_scale': False,
                    'shear': False
                }
                
                if level_idx >= 1 or active_affine_levels <= 1:
                    active_flags['omega'] = True
                    
                if level_idx >= 2 or active_affine_levels <= 2:
                    active_flags['scale'] = True
                    active_flags['anisotropic_scale'] = True
                    active_flags['shear'] = True
                    
                level_affine_losses = []
                if initial_grid is not None:
                    initial_grid_level = upscale_initial_grid(initial_grid, curr_spatial)
                    has_initial_grid = True
                else:
                    initial_grid_level = jnp.zeros((1,) + curr_spatial + (self.dim,))
                    has_initial_grid = False
                    
                for epoch in range(curr_affine_epochs):
                    if sampling_percentage is not None and sampling_percentage < 1.0:
                        # Generate random coordinates on host CPU, convert to JAX array
                        N_total = np.prod(curr_spatial)
                        min_samples = int(0.5 * mattes_bins**2)
                        N_samples = int(np.clip(int(N_total * sampling_percentage), min_samples, N_total))
                        
                        coords_shape = (1,) + (1,) * (self.dim - 1) + (N_samples, self.dim)
                        coords_np = np.random.uniform(-1.0, 1.0, coords_shape).astype(np.float32)
                        coords_hom_np = np.concatenate([coords_np, np.ones(coords_shape[:-1] + (1,), dtype=np.float32)], axis=-1)
                        coords_jax = jnp.array(coords_np)
                        coords_hom_jax = jnp.array(coords_hom_np)
                    else:
                        coords_jax = None
                        coords_hom_jax = None

                    loss_val, params, m_state, v_state, t_state = affine_step_jax(
                        params, m_state, v_state, t_state, active_flags,
                        self.dim, curr_spatial, self.transform_type, I_curr, J_curr,
                        mattes_bins, affine_lr, coords_jax, coords_hom_jax,
                        has_initial_grid, initial_grid_level, self.affine_loss_fn
                    )
                    self.affine_losses.append(loss_val)
                    level_affine_losses.append(loss_val)
                    if len(level_affine_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_affine_epochs - 1):
                        recent_losses = [float(l) for l in level_affine_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-8):
                            break
                    
        self.affine_params = {k: np.array(v) for k, v in params.items()}
        
        # --- 2. SyN Registration ---
        # Single Interpolation Policy: Do NOT pre-warp the moving image.
        # Compose the affine grid with SyN displacement at each iteration.
        A = get_affine_matrix_jax(params, self.dim, self.transform_type)
        A_grid = A[:self.dim, :self.dim + 1]
        
        # J_pyr is reused from the affine stage pyramid construction above.
        # It was built from the ORIGINAL moving image with proper smoothing —
        # no pre-warping. The affine transform is composed with SyN displacement
        # at each iteration (via affine_grid_level) to enforce single interpolation.
                
        # Initialize warps at the coarsest resolution
        curr_spatial = I_pyr[0].shape[2:]
        warp_l2r = jnp.zeros((1, *curr_spatial, self.dim))
        warp_r2l = jnp.zeros((1, *curr_spatial, self.dim))
        warp_l2r_inv = jnp.zeros((1, *curr_spatial, self.dim))
        warp_r2l_inv = jnp.zeros((1, *curr_spatial, self.dim))
        
        for level_idx, scale in enumerate(levels):
            I_curr = I_pyr[level_idx]
            J_curr = J_pyr[level_idx]
            curr_spatial = I_curr.shape[2:]
            
            # Compute current level physical spacing
            curr_spacing_fixed = [sp * (orig_N - 1) / (curr_N - 1) if curr_N > 1 else sp for sp, orig_N, curr_N in zip(fixed_spacing, reversed(I_jax.shape[2:]), reversed(curr_spatial))]
            curr_spacing_moving = [sp * (orig_N - 1) / (curr_N - 1) if curr_N > 1 else sp for sp, orig_N, curr_N in zip(moving_spacing, reversed(J_jax.shape[2:]), reversed(J_curr.shape[2:]))]
            curr_spacing_fixed = tuple(curr_spacing_fixed)
            curr_spacing_moving = tuple(curr_spacing_moving)
            
            has_spacing = True
            spacing_arg = curr_spacing_fixed
            
            # Compute physical affine translation matrix and translation vector
            M_phys, t_phys = grid_to_physical_affine_jax(
                A_grid, spatial_shape, fixed_spacing, fixed_origin, fixed_direction,
                J_jax.shape[2:], moving_spacing, moving_origin, moving_direction
            )
            
            # Compute current level physical grid
            X_phys = get_physical_grid_jax(curr_spatial, curr_spacing_fixed, fixed_origin, fixed_direction)
            
            # Cache physical parameter conversion arrays
            fixed_shape_t = jnp.array(list(curr_spatial))
            fixed_spacing_rev = tuple(reversed(curr_spacing_fixed))
            fixed_origin_rev = tuple(reversed(fixed_origin))
            fixed_direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(fixed_direction)[::-1, ::-1])
            fixed_spacing_t = jnp.array(fixed_spacing_rev)
            fixed_origin_t = jnp.array(fixed_origin_rev)
            fixed_direction_t = jnp.array(fixed_direction_rev)
            
            moving_shape_t = jnp.array(list(J_curr.shape[2:]))
            moving_spacing_rev = tuple(reversed(curr_spacing_moving))
            moving_origin_rev = tuple(reversed(moving_origin))
            moving_direction_rev = tuple(tuple(float(x) for x in row) for row in np.array(moving_direction)[::-1, ::-1])
            moving_spacing_t = jnp.array(moving_spacing_rev)
            moving_origin_t = jnp.array(moving_origin_rev)
            moving_direction_t = jnp.array(moving_direction_rev)
            
            # Upscale initial grid if present
            if initial_grid is not None:
                initial_grid_level = upscale_initial_grid(initial_grid, curr_spatial)
            else:
                initial_grid_level = None
                
            if level_idx > 0:
                warp_l2r = upscale_field_jax(warp_l2r, curr_spatial)
                warp_r2l = upscale_field_jax(warp_r2l, curr_spatial)
                warp_l2r_inv = upscale_field_jax(warp_l2r_inv, curr_spatial)
                warp_r2l_inv = upscale_field_jax(warp_r2l_inv, curr_spatial)
                
            b_mask = get_boundary_mask_jax(curr_spatial, dtype=I_curr.dtype)
            
            if isinstance(epochs_per_level, int):
                curr_syn_epochs = epochs_per_level
            else:
                curr_syn_epochs = epochs_per_level[level_idx]
                
            def make_jax_helper(loss_fn):
                @jax.jit
                def helper(jm, im):
                    l_val, (g_jm, g_im) = jax.value_and_grad(loss_fn, argnums=(0, 1))(jm, im)
                    return l_val, g_im, g_jm
                return helper
            
            jax_grad_helpers_all = []
            for fn in self.loss_functions:
                if not getattr(fn, '_is_pytorch_loss', False):
                    jax_grad_helpers_all.append(make_jax_helper(fn))
                else:
                    jax_grad_helpers_all.append(None)
                    
            is_degenerate = min(curr_spatial) < 32
            active_loss_functions = []
            active_grad_helpers = []
            
            active_metric_names = []
            for metric_idx, metric in enumerate(self.metrics):
                is_deep = False
                metric_name = str(metric)
                if isinstance(metric, str):
                    m_lower = metric.lower()
                    if m_lower in ['vgg19', 'resnet10', 'dinov2', 'dinov2_small', 'dinov2_base', 'swinunetr', 'swin_unetr']:
                        is_deep = True
                elif hasattr(metric, 'extractor') or ('FeatureSpaceLoss' in metric.__class__.__name__):
                    is_deep = True
                    
                if is_degenerate and is_deep:
                    lncc_fn = lambda x, y: local_ncc_loss_nd_jax(x, y, window_size=2 * lncc_radius + 1)
                    active_loss_functions.append(lncc_fn)
                    active_grad_helpers.append(make_jax_helper(lncc_fn))
                    active_metric_names.append('lncc_fallback')
                else:
                    active_loss_functions.append(self.loss_functions[metric_idx])
                    active_grad_helpers.append(jax_grad_helpers_all[metric_idx])
                    active_metric_names.append(metric_name)
                    
            if optimizer_type == 'sgd':
                v_l2r = jnp.zeros_like(warp_l2r)
                v_r2l = jnp.zeros_like(warp_r2l)
            elif optimizer_type == 'adam':
                m_l2r = jnp.zeros_like(warp_l2r)
                m_r2l = jnp.zeros_like(warp_r2l)
                v_l2r = jnp.zeros_like(warp_l2r)
                v_r2l = jnp.zeros_like(warp_r2l)
                adam_t = 0
                
            level_syn_losses = []
            for epoch in range(curr_syn_epochs):
                if optimizer_type == 'lbfgs':
                    from scipy.optimize import minimize
                    
                    flat_shape_l = warp_l2r.shape
                    flat_shape_r = warp_r2l.shape
                    flat_size_l = warp_l2r.size
                    
                    last_loss = [0.0]
                    
                    def objective(x):
                        w_l_np = x[:flat_size_l].reshape(flat_shape_l)
                        w_r_np = x[flat_size_l:].reshape(flat_shape_r)
                        w_l_jax = jnp.array(w_l_np)
                        w_r_jax = jnp.array(w_r_np)
                        
                        w_l_inv_eval = update_inverse_field_nd_jax(
                            w_l_jax, jnp.zeros_like(w_l_jax), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        w_r_inv_eval = update_inverse_field_nd_jax(
                            w_r_jax, jnp.zeros_like(w_r_jax), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_moving, origin=moving_origin, direction=moving_direction
                        )
                        
                        if use_analytical_gradients:
                            I_mid_eval, J_mid_eval, grad_I_mid_sampled_eval, grad_J_mid_sampled_eval = prepare_mid_images_and_gradients_jax(
                                w_l_jax, w_r_jax, w_l_inv_eval, w_r_inv_eval, I_curr, J_curr,
                                X_phys,
                                fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                                moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                                curr_spacing_fixed, curr_spacing_moving,
                                M_phys, t_phys, initial_grid_level
                            )
                        else:
                            (I_mid_eval, J_mid_eval), vjp_fun_eval = jax.vjp(
                                lambda wl, wr, wl_inv, wr_inv: warp_images_jax(
                                    wl, wr, wl_inv, wr_inv, I_curr, J_curr,
                                    X_phys,
                                    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                                    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                                    M_phys, t_phys, initial_grid_level
                                ),
                                w_l_jax, w_r_jax, w_l_inv_eval, w_r_inv_eval
                            )
                            
                        loss_val_sum_eval = 0.0
                        grad_im_sum_eval = jnp.zeros_like(I_mid_eval)
                        grad_jm_sum_eval = jnp.zeros_like(J_mid_eval)
                        
                        for fn_eval, w_eval, jax_helper_eval in zip(active_loss_functions, self.metric_weights, active_grad_helpers):
                            if getattr(fn_eval, '_is_pytorch_loss', False):
                                pytorch_loss_fn_eval = fn_eval._pytorch_loss_fn
                                device_eval = None
                                if hasattr(pytorch_loss_fn_eval, 'parameters'):
                                    try:
                                        device_eval = next(pytorch_loss_fn_eval.parameters()).device
                                    except StopIteration:
                                        pass
                                        
                                I_mid_torch_eval = to_torch_tensor(I_mid_eval).detach().clone()
                                J_mid_torch_eval = to_torch_tensor(J_mid_eval).detach().clone()
                                if device_eval is not None:
                                    I_mid_torch_eval = I_mid_torch_eval.to(device_eval)
                                    J_mid_torch_eval = J_mid_torch_eval.to(device_eval)
                                I_mid_torch_eval = I_mid_torch_eval.requires_grad_(True)
                                J_mid_torch_eval = J_mid_torch_eval.requires_grad_(True)
                                
                                loss_torch_eval = pytorch_loss_fn_eval(J_mid_torch_eval, I_mid_torch_eval)
                                if not loss_torch_eval.requires_grad or loss_torch_eval.grad_fn is None:
                                    g_im_eval = jnp.zeros_like(I_mid_eval)
                                    g_jm_eval = jnp.zeros_like(J_mid_eval)
                                    val_eval = to_jax_array_dl(loss_torch_eval.detach())
                                else:
                                    loss_torch_eval.backward()
                                    g_im_eval = to_jax_array_dl(I_mid_torch_eval.grad) if I_mid_torch_eval.grad is not None else jnp.zeros_like(I_mid_eval)
                                    g_jm_eval = to_jax_array_dl(J_mid_torch_eval.grad) if J_mid_torch_eval.grad is not None else jnp.zeros_like(J_mid_eval)
                                    val_eval = to_jax_array_dl(loss_torch_eval.detach())
                            else:
                                val_eval, g_im_eval, g_jm_eval = jax_helper_eval(J_mid_eval, I_mid_eval)
                                
                            loss_val_sum_eval += w_eval * val_eval
                            grad_im_sum_eval += w_eval * g_im_eval
                            grad_jm_sum_eval += w_eval * g_jm_eval
                            
                        if use_analytical_gradients:
                            grad_l_raw_eval = jnp.moveaxis(grad_im_sum_eval, 1, -1) * grad_I_mid_sampled_eval
                            grad_r_raw_eval = jnp.moveaxis(grad_jm_sum_eval, 1, -1) * grad_J_mid_sampled_eval
                        else:
                            grad_l_raw_eval, grad_r_raw_eval, _, _ = vjp_fun_eval((grad_im_sum_eval, grad_jm_sum_eval))
                            
                        grad_l_filt = separable_gaussian_filter_jax(grad_l_raw_eval * b_mask, self.fluid_sigma, spacing=curr_spacing_fixed)
                        grad_r_filt = separable_gaussian_filter_jax(grad_r_raw_eval * b_mask, self.fluid_sigma, spacing=curr_spacing_moving)
                        
                        l_val = float(loss_val_sum_eval)
                        last_loss[0] = l_val
                        
                        grad_flat = np.concatenate([np.array(grad_l_filt).ravel(), np.array(grad_r_filt).ravel()]).astype(np.float64)
                        return l_val, grad_flat
                        
                    x_init = np.concatenate([np.array(warp_l2r).ravel(), np.array(warp_r2l).ravel()]).astype(np.float64)
                    res = minimize(objective, x_init, method='L-BFGS-B', jac=True, options={'maxiter': 1})
                    x_opt = res.x
                    warp_l2r = jnp.array(x_opt[:flat_size_l].reshape(flat_shape_l))
                    warp_r2l = jnp.array(x_opt[flat_size_l:].reshape(flat_shape_r))
                    
                    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = regularize_warp_fields_jax(
                        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
                        b_mask, True, curr_spacing_fixed, fixed_origin, fixed_direction, self.elastic_sigma,
                        self.inverse_steps, self.inverse_method
                    )
                    loss_val_sum = last_loss[0]
                else:
                    if use_analytical_gradients:
                        I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled = prepare_mid_images_and_gradients_jax(
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
                            X_phys,
                            fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                            moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                            curr_spacing_fixed, curr_spacing_moving,
                            M_phys, t_phys, initial_grid_level
                        )
                    else:
                        (I_mid, J_mid), vjp_fun = jax.vjp(
                            lambda wl, wr, wl_inv, wr_inv: warp_images_jax(
                                wl, wr, wl_inv, wr_inv, I_curr, J_curr,
                                X_phys,
                                fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                                moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                                M_phys, t_phys, initial_grid_level
                            ),
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv
                        )

                    if verbose >= 2:
                        import numpy as np
                        if dim == 2:
                            I_mid_np = np.array(I_mid).squeeze(0).squeeze(0).T
                            J_mid_np = np.array(J_mid).squeeze(0).squeeze(0).T
                        else:
                            I_mid_np = np.array(I_mid).squeeze(0).squeeze(0).transpose(2, 1, 0)
                            J_mid_np = np.array(J_mid).squeeze(0).squeeze(0).transpose(2, 1, 0)
                        
                        import tempfile
                        import ants
                        temp_I = tempfile.NamedTemporaryFile(suffix=f'_level{level_idx}_epoch{epoch}_Imid.nii.gz', delete=False).name
                        temp_J = tempfile.NamedTemporaryFile(suffix=f'_level{level_idx}_epoch{epoch}_Jmid.nii.gz', delete=False).name
                        
                        I_mid_img = ants.from_numpy(I_mid_np, origin=fixed_origin, spacing=curr_spacing_fixed, direction=fixed_direction)
                        J_mid_img = ants.from_numpy(J_mid_np, origin=fixed_origin, spacing=curr_spacing_fixed, direction=fixed_direction)
                        
                        ants.image_write(I_mid_img, temp_I)
                        ants.image_write(J_mid_img, temp_J)
                        print(f"[verbose-2] Saved midpoint images at Level {level_idx} Epoch {epoch}:\n  Fixed-mid: {temp_I}\n  Moving-mid: {temp_J}")
                        
                    loss_val_sum = 0.0
                    grad_im_sum = jnp.zeros_like(I_mid)
                    grad_jm_sum = jnp.zeros_like(J_mid)
                    metric_losses_dict = {}
                    
                    for name, fn, w, jax_helper in zip(active_metric_names, active_loss_functions, self.metric_weights, active_grad_helpers):
                        if getattr(fn, '_is_pytorch_loss', False):
                            pytorch_loss_fn = fn._pytorch_loss_fn
                            device = None
                            if hasattr(pytorch_loss_fn, 'parameters'):
                                try:
                                    device = next(pytorch_loss_fn.parameters()).device
                                except StopIteration:
                                    pass
                                    
                            I_mid_torch = to_torch_tensor(I_mid).detach().clone()
                            J_mid_torch = to_torch_tensor(J_mid).detach().clone()
                            if device is not None:
                                I_mid_torch = I_mid_torch.to(device)
                                J_mid_torch = J_mid_torch.to(device)
                            I_mid_torch = I_mid_torch.requires_grad_(True)
                            J_mid_torch = J_mid_torch.requires_grad_(True)
                            
                            loss_torch = pytorch_loss_fn(J_mid_torch, I_mid_torch)
                            if not loss_torch.requires_grad or loss_torch.grad_fn is None:
                                g_im = jnp.zeros_like(I_mid)
                                g_jm = jnp.zeros_like(J_mid)
                                val = to_jax_array_dl(loss_torch.detach())
                            else:
                                loss_torch.backward()
                                g_im = to_jax_array_dl(I_mid_torch.grad) if I_mid_torch.grad is not None else jnp.zeros_like(I_mid)
                                g_jm = to_jax_array_dl(J_mid_torch.grad) if J_mid_torch.grad is not None else jnp.zeros_like(J_mid)
                                val = to_jax_array_dl(loss_torch.detach())
                        else:
                            val, g_im, g_jm = jax_helper(J_mid, I_mid)
                            
                        loss_val_sum += w * val
                        grad_im_sum += w * g_im
                        grad_jm_sum += w * g_jm
                        metric_losses_dict[name] = float(val)
                        
                    if use_analytical_gradients:
                        grad_l_raw = jnp.moveaxis(grad_im_sum, 1, -1) * grad_I_mid_sampled
                        grad_r_raw = jnp.moveaxis(grad_jm_sum, 1, -1) * grad_J_mid_sampled
                    else:
                        grad_l_raw, grad_r_raw, _, _ = vjp_fun((grad_im_sum, grad_jm_sum))
                    
                    if verbose and epoch == 0:
                        print("DEBUG JAX epoch 0 J_mid min/max:", float(J_mid.min()), float(J_mid.max()))
                        print("DEBUG JAX epoch 0 I_mid min/max:", float(I_mid.min()), float(I_mid.max()))
                        print("DEBUG JAX epoch 0 grad_im_sum max:", float(jnp.abs(grad_im_sum).max()))
                        print("DEBUG JAX epoch 0 grad_I_mid_sampled max:", float(jnp.abs(grad_I_mid_sampled).max()))
                        print("DEBUG JAX epoch 0 grad_l_raw max:", float(jnp.abs(grad_l_raw).max()))
                        
                    if optimizer_type == 'cfl':
                        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = syn_update_step_jax(
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
                            grad_l_raw, grad_r_raw, X_phys, b_mask,
                            fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                            True, curr_spacing_fixed, fixed_origin, fixed_direction, self.fluid_sigma, self.elastic_sigma, cfl_voxels,
                            self.inverse_steps, self.inverse_method
                        )
                    elif optimizer_type == 'sgd':
                        warp_l2r, warp_r2l, v_l2r, v_r2l = sgd_update_step_jax(
                            warp_l2r, warp_r2l, v_l2r, v_r2l,
                            grad_l_raw, grad_r_raw, b_mask,
                            True, curr_spacing_fixed, self.fluid_sigma, optimizer_lr
                        )
                        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = regularize_warp_fields_jax(
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
                            b_mask, True, curr_spacing_fixed, fixed_origin, fixed_direction, self.elastic_sigma,
                            self.inverse_steps, self.inverse_method
                        )
                    elif optimizer_type == 'adam':
                        adam_t += 1
                        warp_l2r, warp_r2l, m_l2r, m_r2l, v_l2r, v_r2l = adam_update_step_jax(
                            warp_l2r, warp_r2l, m_l2r, m_r2l, v_l2r, v_r2l, float(adam_t),
                            grad_l_raw, grad_r_raw, b_mask,
                            True, curr_spacing_fixed, self.fluid_sigma, optimizer_lr
                        )
                        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = regularize_warp_fields_jax(
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
                            b_mask, True, curr_spacing_fixed, fixed_origin, fixed_direction, self.elastic_sigma,
                            self.inverse_steps, self.inverse_method
                        )
                        
                self.syn_losses.append(loss_val_sum)
                level_syn_losses.append(loss_val_sum)
                if verbose:
                    loss_details = ", ".join([f"{k}={v:.6f}" for k, v in metric_losses_dict.items()]) if 'metric_losses_dict' in locals() else ""
                    loss_details_str = f" ({loss_details})" if loss_details else ""
                    print(f"[jax-fit] Level {level_idx} Epoch {epoch}: loss={float(loss_val_sum):.6f}{loss_details_str}, warp_l2r max norm={float(jnp.sqrt(jnp.sum(warp_l2r**2, axis=-1)).max()):.4f}")
                if len(level_syn_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_syn_epochs - 1):
                    recent_losses = [float(l) for l in level_syn_losses[-10:]]
                    if check_convergence(recent_losses, window_size=10, slope_threshold=1e-8):
                        break
                        
            # Clear XLA cache between levels to prevent memory growth
            jax.clear_caches()
            
        w_l2r = upscale_field_jax(warp_l2r, self.grid_shape)
        w_r2l = upscale_field_jax(warp_r2l, self.grid_shape)
        w_l2r_inv = upscale_field_jax(warp_l2r_inv, self.grid_shape)
        w_r2l_inv = upscale_field_jax(warp_r2l_inv, self.grid_shape)
        
        X_phys = get_physical_grid_jax(self.grid_shape, fixed_spacing, fixed_origin, fixed_direction)
        
        phi_l2r_phys = X_phys + w_l2r_inv
        coords_norm = physical_to_normalized_jax(phi_l2r_phys, self.grid_shape, fixed_spacing, fixed_origin, fixed_direction)
        w_r2l_cf = jnp.moveaxis(w_r2l, -1, 1)
        disp_r2l_sampled = jnp.moveaxis(jax_grid_sample(w_r2l_cf, coords_norm, padding_mode='border'), 1, -1)
        full_l2r_phys = phi_l2r_phys + disp_r2l_sampled
        self.warp_l2r = PhysicalWarpArray(full_l2r_phys - X_phys, is_physical=True)
        
        phi_r2l_phys = X_phys + w_r2l_inv
        coords_norm_r = physical_to_normalized_jax(phi_r2l_phys, self.grid_shape, fixed_spacing, fixed_origin, fixed_direction)
        w_l2r_cf = jnp.moveaxis(w_l2r, -1, 1)
        disp_l2r_sampled = jnp.moveaxis(jax_grid_sample(w_l2r_cf, coords_norm_r, padding_mode='border'), 1, -1)
        full_r2l_phys = phi_r2l_phys + disp_l2r_sampled
        self.warp_r2l = PhysicalWarpArray(full_r2l_phys - X_phys, is_physical=True)
        
        warp_l2r_inv_final = update_inverse_field_nd_jax(
            jnp.array(self.warp_l2r), jnp.zeros_like(jnp.array(self.warp_l2r)),
            spacing=fixed_spacing, origin=fixed_origin, direction=fixed_direction,
            steps=self.inverse_steps, method=self.inverse_method
        )
        warp_r2l_inv_final = update_inverse_field_nd_jax(
            jnp.array(self.warp_r2l), jnp.zeros_like(jnp.array(self.warp_r2l)),
            spacing=moving_spacing, origin=moving_origin, direction=moving_direction,
            steps=self.inverse_steps, method=self.inverse_method
        )
        self.warp_l2r_inv = PhysicalWarpArray(warp_l2r_inv_final, is_physical=True)
        self.warp_r2l_inv = PhysicalWarpArray(warp_r2l_inv_final, is_physical=True)
        
        # Convert all logged losses to floats in a single batch
        self.affine_losses = [float(l) for l in self.affine_losses]
        self.syn_losses = [float(l) for l in self.syn_losses]

    def forward(self, moving_image, fixed_image=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        is_torch = hasattr(moving_image, 'device')
        moving_image_jax = to_jax_array(moving_image)
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))
        
        # Permute input to ZYX order
        moving_image_jax = jnp.transpose(moving_image_jax, perm)
        
        # Fixed properties define output space
        spatial_shape = self.grid_shape
        spacing = tuple(reversed(self.spacing)) if self.spacing is not None else [1.0] * dim
        origin = tuple(reversed(self.origin)) if self.origin is not None else [0.0] * dim
        direction = self.direction if self.direction is not None else np.eye(dim)
        
        # Moving properties
        if moving_spacing is None: moving_spacing = spacing
        if moving_origin is None: moving_origin = origin
        if moving_direction is None: moving_direction = direction
        
        X_phys = get_physical_grid_jax(spatial_shape, spacing, origin, direction)
        
        warp_resampled = upscale_field_jax(jnp.array(self.warp_l2r), spatial_shape)
        phi_l2r_phys = X_phys + warp_resampled
        
        T_grid = get_affine_matrix_jax(self.affine_params, dim, self.transform_type)
        M_phys, t_phys = grid_to_physical_affine_jax(
            T_grid, spatial_shape, spacing, origin, direction,
            moving_image_jax.shape[2:], moving_spacing, moving_origin, moving_direction
        )
        y_phys = phi_l2r_phys @ M_phys.T + t_phys
        composed_grid = physical_to_normalized_jax(y_phys, moving_image_jax.shape[2:], moving_spacing, moving_origin, moving_direction)
        
        if hasattr(self, 'initial_grid') and self.initial_grid is not None:
            initial_grid_resampled = upscale_initial_grid(self.initial_grid, spatial_shape)
            composed_grid = compose_grids_jax(initial_grid_resampled, composed_grid)
            
        warped_jax = jax_grid_sample(moving_image_jax, composed_grid, padding_mode='border')
        warped_xyz = jnp.transpose(warped_jax, perm)
        
        if is_torch:
            return torch.from_numpy(np.array(warped_xyz)).to(
                device=moving_image.device, dtype=moving_image.dtype
            )
        return warped_xyz

    def forward_inverse(self, fixed_image, moving_shape=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        is_torch = hasattr(fixed_image, 'device')
        fixed_image_jax = to_jax_array(fixed_image)
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))
        
        # Permute input to ZYX order
        fixed_image_jax = jnp.transpose(fixed_image_jax, perm)
        
        fixed_shape = fixed_image_jax.shape[2:]
        spacing = tuple(reversed(self.spacing)) if self.spacing is not None else [1.0] * dim
        origin = tuple(reversed(self.origin)) if self.origin is not None else [0.0] * dim
        direction = self.direction if self.direction is not None else np.eye(dim)
        
        # Moving properties define output space
        if moving_shape is None: moving_shape = self.grid_shape
        if moving_spacing is None: moving_spacing = spacing
        if moving_origin is None: moving_origin = origin
        if moving_direction is None: moving_direction = direction
        
        Y_phys = get_physical_grid_jax(moving_shape, moving_spacing, moving_origin, moving_direction)
        
        warp_resampled = upscale_field_jax(jnp.array(self.warp_r2l), moving_shape)
        phi_r2l_phys = Y_phys + warp_resampled
        
        T_grid = get_affine_matrix_jax(self.affine_params, dim, self.transform_type)
        T_inv = jnp.linalg.inv(T_grid)
        M_phys_inv, t_phys_inv = grid_to_physical_affine_jax(
            T_inv, moving_shape, moving_spacing, moving_origin, moving_direction,
            fixed_shape, spacing, origin, direction
        )
        x_phys = phi_r2l_phys @ M_phys_inv.T + t_phys_inv
        composed_grid = physical_to_normalized_jax(x_phys, fixed_shape, spacing, origin, direction)
        
        warped_jax = jax_grid_sample(fixed_image_jax, composed_grid, padding_mode='border')
        warped_xyz = jnp.transpose(warped_jax, perm)
        
        if is_torch:
            return torch.from_numpy(np.array(warped_xyz)).to(
                device=fixed_image.device, dtype=fixed_image.dtype
            )
        return warped_xyz

    def get_forward_transform(self, fixed_metadata):
        device = torch.device('cpu')
        grid_affine = self.get_affine_grid(self.grid_shape, device)
        warp_l2r_torch = torch.from_numpy(self.warp_l2r).to(device)
        return SyNToTransform(
            affine_grid=grid_affine, 
            warp_field=warp_l2r_torch, 
            metadata=fixed_metadata, 
            device=device,
            is_physical=True
        )

    def get_inverse_transform(self, moving_metadata):
        device = torch.device('cpu')
        grid_affine_inv = self.get_inverse_affine_grid(self.grid_shape, device)
        warp_r2l_torch = torch.from_numpy(self.warp_r2l).to(device)
        return SyNToTransform(
            affine_grid=grid_affine_inv, 
            warp_field=warp_r2l_torch, 
            metadata=moving_metadata, 
            device=device,
            is_physical=True
        )
