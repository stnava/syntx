import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import torch

def check_convergence(losses, window_size=10, slope_threshold=1e-5):
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
        
        K = jnp.stack([
            jnp.stack([0.0, -omega_norm[2], omega_norm[1]]),
            jnp.stack([omega_norm[2], 0.0, -omega_norm[0]]),
            jnp.stack([-omega_norm[1], omega_norm[0], 0.0])
        ])
        I = jnp.eye(3)
        R = I + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
        return jnp.where(is_zero, I, R)
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
        
    T = jnp.eye(dim + 1)
    T = T.at[:dim, :dim].set(A)
    T = T.at[:dim, dim].set(translation)
    return T


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
        sigma_list = [sigma / sp for sp in spacing]
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
    return jnp.stack(list(reversed(grads)), axis=-1)


# 9. Diffeomorphic Cycle Consistency Projection
def update_inverse_field_nd_jax(
    W_disp, 
    W_inv_disp, 
    steps=20,
    relaxation=1.0,
    smoothing_sigma=0.0,
    method='fixed_point',
    max_error_threshold=0.1,
    mean_error_threshold=0.001
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
        out = _conv1d_axis_edge(out, kernel_1d, axis=i + 2)
    return out


def local_ncc_loss_nd_jax(I, J, mask=None, window_size=9):
    I_mean = box_filter_jax(I, window_size)
    J_mean = box_filter_jax(J, window_size)
    
    I_var = box_filter_jax(I**2, window_size) - I_mean**2
    J_var = box_filter_jax(J**2, window_size) - J_mean**2
    IJ_cov = box_filter_jax(I*J, window_size) - I_mean * J_mean
    
    safe_I_var = jnp.maximum(I_var, 1e-5)
    safe_J_var = jnp.maximum(J_var, 1e-5)
    
    cc_raw = IJ_cov / (jnp.sqrt(safe_I_var * safe_J_var) + 1e-5)
    valid_mask = (I_var > 1e-5) & (J_var > 1e-5)
    cc = jnp.where(valid_mask, cc_raw, 0.0)
    
    if mask is not None:
        active_mask = (mask > 0.5) & valid_mask
    else:
        active_mask = valid_mask
        
    active_mask_float = active_mask.astype(jnp.float32)
    return -jnp.sum(cc * active_mask_float) / (jnp.sum(active_mask_float) + 1e-8)


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
    
    grids = [jnp.linspace(-1.0, 1.0, size) for size in spatial]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
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
        j00 = grads[1][..., 0]
        j01 = grads[0][..., 0]
        j10 = grads[1][..., 1]
        j11 = grads[0][..., 1]
        return j00 * j11 - j01 * j10
    elif dim == 3:
        j00 = grads[2][..., 0]
        j01 = grads[1][..., 0]
        j02 = grads[0][..., 0]
        
        j10 = grads[2][..., 1]
        j11 = grads[1][..., 1]
        j12 = grads[0][..., 1]
        
        j20 = grads[2][..., 2]
        j21 = grads[1][..., 2]
        j22 = grads[0][..., 2]
        
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


# 13. Functional JIT step updates for Rprop and SyN
@partial(jax.jit, static_argnums=(5, 6, 7, 10))
def affine_step_jax(
    params, m_state, v_state, t_state, active_flags,
    dim, spatial_shape, transform_type, I_curr, J_curr,
    mattes_bins, lr=1e-2, coords=None, coords_hom=None
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
            I_sampled = jax_grid_sample(I_curr, coords, padding_mode='border')
            J_sampled = jax_grid_sample(J_curr, coords_warped, padding_mode='border')
            return mattes_mi_loss_core_jax(J_sampled.flatten(), I_sampled.flatten(), num_bins=mattes_bins)
        else:
            grid = jax_affine_grid(A[:dim, :dim + 1], spatial_shape)
            moving_warped = jax_grid_sample(J_curr, grid, padding_mode='border')
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


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 14, 15, 16, 17, 18, 19))
def syn_step_jax(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
    I_curr, J_curr, identity, b_mask,
    has_spacing, spacing, fluid_sigma, elastic_sigma, cfl_voxels, physical_bounds,
    inverse_steps, inverse_method, similarity_metric, use_analytical_gradients,
    lncc_radius, mattes_bins
):
    """
    Performs a single step of the symmetric diffeomorphic (SyN) registration.
    
    Provenance:
    Adapted from ITK's SyN registration method:
    itkSyNImageRegistrationMethod.hxx (and associated classes).
    - Computes symmetric updates in the middle space (I_mid and J_mid).
    - Enforces cycle consistency via projection in each iteration.
    - Resolves displacement field boundary constraints.
    """
    window_size = 2 * lncc_radius + 1
    phi_l2r = identity + warp_l2r
    phi_r2l = identity + warp_r2l
    
    I_mid = jax_grid_sample(I_curr, phi_l2r, padding_mode='border')
    J_mid = jax_grid_sample(J_curr, phi_r2l, padding_mode='border')
    
    spatial_shape = warp_l2r.shape[1:-1]
    
    if use_analytical_gradients:
        I_curr_cl = jnp.moveaxis(I_curr, 1, -1)
        J_curr_cl = jnp.moveaxis(J_curr, 1, -1)
        
        default_spacings = jnp.array([2.0 / (s - 1) for s in spatial_shape])
        if has_spacing:
            spacing_for_jacobian = jnp.array(spacing)
        else:
            spacing_for_jacobian = default_spacings
        
        grad_I_curr = _spatial_jacobian_nd_jax(I_curr_cl, physical_spacing=spacing_for_jacobian).squeeze(-2)
        grad_J_curr = _spatial_jacobian_nd_jax(J_curr_cl, physical_spacing=spacing_for_jacobian).squeeze(-2)
        
        grad_I_mid_sampled = jnp.moveaxis(
            jax_grid_sample(jnp.moveaxis(grad_I_curr, -1, 1), phi_l2r, padding_mode='border'),
            1, -1
        )
        grad_J_mid_sampled = jnp.moveaxis(
            jax_grid_sample(jnp.moveaxis(grad_J_curr, -1, 1), phi_r2l, padding_mode='border'),
            1, -1
        )
        
        def loss_mid_fn(im, jm):
            if similarity_metric == 'mattes_mi':
                return mattes_mi_loss_nd_jax(jm, im, num_bins=mattes_bins)
            else:
                return local_ncc_loss_nd_jax(jm, im, window_size=window_size)
                
        loss_val, (grad_im_loss, grad_jm_loss) = jax.value_and_grad(loss_mid_fn, argnums=(0, 1))(I_mid, J_mid)
        
        grad_l_raw = jnp.moveaxis(grad_im_loss, 1, -1) * grad_I_mid_sampled
        grad_r_raw = jnp.moveaxis(grad_jm_loss, 1, -1) * grad_J_mid_sampled
    else:
        def loss_warp_fn(wl, wr):
            phi_l = identity + wl
            phi_r = identity + wr
            im = jax_grid_sample(I_curr, phi_l, mode='bilinear', padding_mode='border')
            jm = jax_grid_sample(J_curr, phi_r, mode='bilinear', padding_mode='border')
            if similarity_metric == 'mattes_mi':
                return mattes_mi_loss_nd_jax(jm, im, num_bins=mattes_bins)
            else:
                return local_ncc_loss_nd_jax(jm, im, window_size=window_size)
                
        loss_val, (grad_l_raw, grad_r_raw) = jax.value_and_grad(loss_warp_fn, argnums=(0, 1))(warp_l2r, warp_r2l)
        
    # Enforce zero boundary condition on gradients before filtering (fluid smoothing)
    if has_spacing:
        grad_l = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=spacing)
        grad_r = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=spacing)
    else:
        grad_l = separable_gaussian_filter_jax(grad_l_raw * b_mask, fluid_sigma, spacing=None)
        grad_r = separable_gaussian_filter_jax(grad_r_raw * b_mask, fluid_sigma, spacing=None)
    
    grad_norm_l = jnp.sqrt(jnp.sum(grad_l**2, axis=-1))
    grad_norm_r = jnp.sqrt(jnp.sum(grad_r**2, axis=-1))
    
    max_grad_l = jnp.max(grad_norm_l) + 1e-8
    max_grad_r = jnp.max(grad_norm_r) + 1e-8
    
    # ITK-style CFL: Euclidean norm in VOXEL coordinates
    # Convert normalized-space gradient to voxel coords, compute L2 norm,
    # then apply uniform scalar to bound max voxel displacement to cfl_voxels.
    # Matches itkSyNImageRegistrationMethod::ScaleUpdateField exactly.
    voxel_scale = jnp.array(
        [float((s - 1) / 2.0) for s in reversed(spatial_shape)]
    )
    grad_l_voxel = grad_l * voxel_scale
    grad_r_voxel = grad_r * voxel_scale
    max_norm_l = jnp.sqrt(jnp.sum(grad_l_voxel**2, axis=-1)).max() + 1e-8
    max_norm_r = jnp.sqrt(jnp.sum(grad_r_voxel**2, axis=-1)).max() + 1e-8
    
    # Uniform scalar learning rate: max voxel displacement = cfl_voxels
    lr_l = cfl_voxels / max_norm_l
    lr_r = cfl_voxels / max_norm_r
    
    # Scale the normalized-space gradient uniformly (preserves direction)
    delta_l = lr_l * grad_l
    delta_r = lr_r * grad_r
        
    # Greedy SyN composition: φ_new = φ_old ∘ (Id - ∂loss/∂warp)
    # Since loss = -NCC, delta = -∂CC/∂warp, so (Id - delta) = (Id + ∂CC/∂warp)
    # This correctly moves toward better alignment, matching ITK's convention.
    warp_l2r = compose_grids_jax(identity + warp_l2r, identity - delta_l) - identity
    warp_r2l = compose_grids_jax(identity + warp_r2l, identity - delta_r) - identity
    
    # ITK-standard Dirichlet zero boundary enforcement after composition.
    # Matches itkSyNImageRegistrationMethod::GaussianSmoothDisplacementField lines 770-805.
    warp_l2r = warp_l2r * b_mask
    warp_r2l = warp_r2l * b_mask
    
    if elastic_sigma > 0.0:
        if has_spacing:
            warp_l2r = separable_gaussian_filter_jax(warp_l2r, elastic_sigma, spacing=spacing)
            warp_r2l = separable_gaussian_filter_jax(warp_r2l, elastic_sigma, spacing=spacing)
        else:
            warp_l2r = separable_gaussian_filter_jax(warp_l2r, elastic_sigma, spacing=None)
            warp_r2l = separable_gaussian_filter_jax(warp_r2l, elastic_sigma, spacing=None)
        
    # ITK-style diffeomorphic projection: double-inversion.
    # Projects the composed field back to the space of invertible transforms.
    # Matches itkSyNImageRegistrationMethod.hxx lines 227-244.
    warp_l2r_inv = update_inverse_field_nd_jax(
        warp_l2r, warp_l2r_inv, steps=inverse_steps, method=inverse_method
    )
    warp_l2r = update_inverse_field_nd_jax(
        warp_l2r_inv, warp_l2r, steps=inverse_steps, method=inverse_method
    )
    
    warp_r2l_inv = update_inverse_field_nd_jax(
        warp_r2l, warp_r2l_inv, steps=inverse_steps, method=inverse_method
    )
    warp_r2l = update_inverse_field_nd_jax(
        warp_r2l_inv, warp_r2l, steps=inverse_steps, method=inverse_method
    )
    
    return loss_val, warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv


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


# 14. Standard SyNTo Class API
class SyNTo:
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, direction=None, fluid_sigma=1.732, elastic_sigma=1.0, transform_type='Affine', inverse_method='neumann', inverse_steps=20):
        self.dim = dim
        self.grid_shape = grid_shape
        self.spacing = spacing
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
            lncc_radius=4, mattes_bins=32, sampling_percentage=None):
        
        I_jax = to_jax_array(fixed_image)
        J_jax = to_jax_array(moving_image)
        spatial_shape = I_jax.shape[2:]
        
        self.affine_losses = []
        self.syn_losses = []
        
        # --- 0. Construct Image Pyramids ---
        I_pyr = []
        J_pyr = []
        for s in levels:
            if s > 1:
                curr_spatial = tuple(max(1, int(size / s)) for size in spatial_shape)
                I_down = interpolate_jax(I_jax, 1.0 / s, self.dim)
                J_down = interpolate_jax(J_jax, 1.0 / s, self.dim)
                I_pyr.append(I_down)
                J_pyr.append(J_down)
            else:
                I_pyr.append(I_jax)
                J_pyr.append(J_jax)
                
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
        
        # Pad iteration lists to match hierarchy levels length
        if isinstance(epochs_per_level, int):
            epochs_per_level = [epochs_per_level] * len(levels)
        elif len(epochs_per_level) < len(levels):
            epochs_per_level = list(epochs_per_level) + [0] * (len(levels) - len(epochs_per_level))
            
        if isinstance(affine_epochs, int):
            affine_epochs = [affine_epochs] * len(levels)
        elif len(affine_epochs) < len(levels):
            affine_epochs = list(affine_epochs) + [0] * (len(levels) - len(affine_epochs))
            
        if sum(affine_epochs) > 0:
            for level_idx, scale in enumerate(levels):
                curr_affine_epochs = affine_epochs[level_idx]
                if curr_affine_epochs <= 0:
                    continue
                I_curr = I_pyr[level_idx]
                J_curr = J_pyr[level_idx]
                curr_spatial = I_curr.shape[2:]
                
                active_flags = {
                    'translation': True,
                    'omega': False,
                    'scale': False,
                    'anisotropic_scale': False,
                    'shear': False
                }
                
                if level_idx >= 1 or len(levels) == 1:
                    active_flags['omega'] = True
                    
                if level_idx >= 2 or len(levels) <= 2:
                    active_flags['scale'] = True
                    active_flags['anisotropic_scale'] = True
                    active_flags['shear'] = True
                    
                level_affine_losses = []
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
                        mattes_bins, affine_lr, coords_jax, coords_hom_jax
                    )
                    self.affine_losses.append(loss_val)
                    level_affine_losses.append(loss_val)
                    if len(level_affine_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_affine_epochs - 1):
                        recent_losses = [float(l) for l in level_affine_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-5):
                            break
                    
        self.affine_params = {k: np.array(v) for k, v in params.items()}
        
        # --- 2. SyN Registration ---
        A = get_affine_matrix_jax(params, self.dim, self.transform_type)
        A_grid = A[:self.dim, :self.dim + 1]
        grid = jax_affine_grid(A_grid, spatial_shape)
        moving_affine = jax_grid_sample(J_jax, grid, padding_mode='border')
        
        J_pyr = []
        for s in levels:
            if s > 1:
                J_down = interpolate_jax(moving_affine, 1.0 / s, self.dim)
                J_pyr.append(J_down)
            else:
                J_pyr.append(moving_affine)
                
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
            
            if level_idx > 0:
                warp_l2r = upscale_field_jax(warp_l2r, curr_spatial)
                warp_r2l = upscale_field_jax(warp_r2l, curr_spatial)
                warp_l2r_inv = upscale_field_jax(warp_l2r_inv, curr_spatial)
                warp_r2l_inv = upscale_field_jax(warp_r2l_inv, curr_spatial)
                
            if self.spacing is not None:
                curr_physical_spacing = tuple(float(2.0 * b / (s - 1)) for b, s in zip(self.physical_bounds, curr_spatial))
                has_spacing = True
                spacing_arg = curr_physical_spacing
            else:
                has_spacing = False
                spacing_arg = tuple([1.0] * self.dim)
                
            physical_bounds_arg = jnp.array(self.physical_bounds)
                
            grids = [jnp.linspace(-1.0, 1.0, size) for size in curr_spatial]
            meshgrid = jnp.meshgrid(*grids, indexing='ij')
            identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
            identity = jnp.expand_dims(identity, axis=0)
            
            b_mask = get_boundary_mask_jax(curr_spatial, dtype=I_curr.dtype)
            
            if isinstance(epochs_per_level, int):
                curr_syn_epochs = epochs_per_level
            else:
                curr_syn_epochs = epochs_per_level[level_idx]
                
            level_syn_losses = []
            for epoch in range(curr_syn_epochs):
                loss_val, warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv = syn_step_jax(
                    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
                    I_curr, J_curr, identity, b_mask,
                    has_spacing, spacing_arg, self.fluid_sigma, self.elastic_sigma, cfl_voxels, physical_bounds_arg,
                    self.inverse_steps, self.inverse_method, similarity_metric, use_analytical_gradients,
                    lncc_radius, mattes_bins
                )
                self.syn_losses.append(loss_val)
                level_syn_losses.append(loss_val)
                if len(level_syn_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_syn_epochs - 1):
                    recent_losses = [float(l) for l in level_syn_losses[-10:]]
                    if check_convergence(recent_losses, window_size=10, slope_threshold=1e-5):
                        break
                
            # Clear XLA cache between levels to prevent memory growth
            jax.clear_caches()
            
        w_l2r = upscale_field_jax(warp_l2r, self.grid_shape)
        w_r2l = upscale_field_jax(warp_r2l, self.grid_shape)
        w_l2r_inv = upscale_field_jax(warp_l2r_inv, self.grid_shape)
        w_r2l_inv = upscale_field_jax(warp_r2l_inv, self.grid_shape)
        
        grids_full = [jnp.linspace(-1.0, 1.0, size) for size in self.grid_shape]
        meshgrid_full = jnp.meshgrid(*grids_full, indexing='ij')
        identity_full = jnp.stack(list(reversed(meshgrid_full)), axis=-1)
        identity_full = jnp.expand_dims(identity_full, axis=0)
        
        full_l2r = compose_grids_jax(identity_full + w_r2l, identity_full + w_l2r_inv)
        full_r2l = compose_grids_jax(identity_full + w_l2r, identity_full + w_r2l_inv)
        
        self.warp_l2r = np.array(full_l2r - identity_full)
        self.warp_r2l = np.array(full_r2l - identity_full)
        
        warp_l2r_inv_final = update_inverse_field_nd_jax(
            jnp.array(self.warp_l2r), jnp.zeros_like(jnp.array(self.warp_l2r)),
            steps=self.inverse_steps, method=self.inverse_method
        )
        warp_r2l_inv_final = update_inverse_field_nd_jax(
            jnp.array(self.warp_r2l), jnp.zeros_like(jnp.array(self.warp_r2l)),
            steps=self.inverse_steps, method=self.inverse_method
        )
        self.warp_l2r_inv = np.array(warp_l2r_inv_final)
        self.warp_r2l_inv = np.array(warp_r2l_inv_final)
        
        # Convert all logged losses to floats in a single batch
        self.affine_losses = [float(l) for l in self.affine_losses]
        self.syn_losses = [float(l) for l in self.syn_losses]

    def forward(self, moving_image, fixed_image=None):
        is_torch = hasattr(moving_image, 'device')
        moving_image_jax = to_jax_array(moving_image)
        spatial_shape = moving_image_jax.shape[2:]
        
        A = get_affine_matrix_jax(self.affine_params, self.dim, self.transform_type)
        A_grid = A[:self.dim, :self.dim + 1]
        grid_affine = jax_affine_grid(A_grid, spatial_shape)
        
        warp_resampled = upscale_field_jax(jnp.array(self.warp_l2r), spatial_shape)
        
        grids = [jnp.linspace(-1.0, 1.0, size) for size in spatial_shape]
        meshgrid = jnp.meshgrid(*grids, indexing='ij')
        identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
        identity = jnp.expand_dims(identity, axis=0)
        
        phi_l2r = identity + warp_resampled
        composed_grid = compose_grids_jax(grid_affine, phi_l2r)
        
        warped_jax = jax_grid_sample(moving_image_jax, composed_grid, padding_mode='border')
        
        if is_torch:
            return torch.from_numpy(np.array(warped_jax)).to(
                device=moving_image.device, dtype=moving_image.dtype
            )
        return warped_jax

    def forward_inverse(self, fixed_image):
        is_torch = hasattr(fixed_image, 'device')
        fixed_image_jax = to_jax_array(fixed_image)
        spatial_shape = fixed_image_jax.shape[2:]
        
        A = get_affine_matrix_jax(self.affine_params, self.dim, self.transform_type)
        A_inv = jnp.linalg.inv(A)
        theta_inv = A_inv[:self.dim, :self.dim + 1]
        grid_affine_inv = jax_affine_grid(theta_inv, spatial_shape)
        
        warp_resampled = upscale_field_jax(jnp.array(self.warp_r2l), spatial_shape)
        
        grids = [jnp.linspace(-1.0, 1.0, size) for size in spatial_shape]
        meshgrid = jnp.meshgrid(*grids, indexing='ij')
        identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
        identity = jnp.expand_dims(identity, axis=0)
        
        phi_r2l = identity + warp_resampled
        composed_grid = compose_grids_jax(grid_affine_inv, phi_r2l)
        
        warped_jax = jax_grid_sample(fixed_image_jax, composed_grid, padding_mode='border')
        
        if is_torch:
            return torch.from_numpy(np.array(warped_jax)).to(
                device=fixed_image.device, dtype=fixed_image.dtype
            )
        return warped_jax

    def get_forward_transform(self, fixed_metadata):
        device = torch.device('cpu')
        grid_affine = self.get_affine_grid(self.grid_shape, device)
        warp_l2r_torch = torch.from_numpy(self.warp_l2r).to(device)
        return SyNToTransform(
            affine_grid=grid_affine, 
            warp_field=warp_l2r_torch, 
            metadata=fixed_metadata, 
            device=device
        )

    def get_inverse_transform(self, moving_metadata):
        device = torch.device('cpu')
        grid_affine_inv = self.get_inverse_affine_grid(self.grid_shape, device)
        warp_r2l_torch = torch.from_numpy(self.warp_r2l).to(device)
        return SyNToTransform(
            affine_grid=grid_affine_inv, 
            warp_field=warp_r2l_torch, 
            metadata=moving_metadata, 
            device=device
        )
