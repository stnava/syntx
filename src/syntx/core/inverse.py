"""
Inverting displacement fields, measuring inverse consistency, and integrating velocities.

Conventions shared by every function here:

- Fields are displacements in tensor order, shape (B, *spatial, dim) with spatial (z, y, x)
  and components (x, y, z). Channels-first input (B, dim, *spatial) is detected for the
  inverse solvers and returned in the same layout.
- ``spacing`` / ``origin`` / ``direction`` are ANTs (x-first) metadata. With all three
  given, displacements are in physical units (mm); without them, in normalised [-1, 1]
  grid units.
- Convergence thresholds and reported solver errors are in voxels.

Inversion: ``update_inverse_field_nd`` dispatches to the ITK-style fixed-point iteration,
``update_inverse_field_nd_anderson`` (default) or ``update_inverse_field_nd_hybrid_lm``.
Inverse error: ``calculate_inverse_identity_error`` (summary dict; exported as
``syntx.calculate_inverse_identity_error``) and ``compute_inverse_identity_error_nd``
(error map).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .smoothing import separable_gaussian_filter, get_boundary_mask
from .grid import (
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    grid_sample_nd,
    compose_grids,
    sample_field_cf,
)


def _batch_reduce_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item max over all spatial and channel axes.

    Uses a 2-stage 1-axis reduction on 5-D+ tensors so PyTorch MPS never dispatches a
    4-axis reduction kernel on large 3-D volumes (which corrupts outputs above ~16M elements).
    """
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).amax(dim=2).amax(dim=1)
    return x.reshape(b, -1).amax(dim=1)


def _batch_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item mean over all spatial and channel axes (2-stage safe on 5-D+ MPS)."""
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).mean(dim=2).mean(dim=1)
    return x.reshape(b, -1).mean(dim=1)


def _batch_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item sum over all spatial and channel axes (2-stage safe on 5-D+ MPS)."""
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).sum(dim=2).sum(dim=1)
    return x.reshape(b, -1).sum(dim=1)


def _scaled_sq_sum(error: torch.Tensor, scale: torch.Tensor, divide: bool) -> torch.Tensor:
    """Compute ``sum_c ((error[..., c] / scale[c]) ** 2)`` (or ``* scale[c]``) as ``(B, *spatial, 1)``
    without allocating a full multi-channel ``(B, *spatial, dim)`` squared tensor."""
    dim = error.shape[-1]
    if divide:
        sq = (error[..., 0:1] / scale[0]).square_()
        for c in range(1, dim):
            sq.add_((error[..., c:c + 1] / scale[c]).square_())
    else:
        sq = (error[..., 0:1] * scale[0]).square_()
        for c in range(1, dim):
            sq.add_((error[..., c:c + 1] * scale[c]).square_())
    return sq


def _dot_1d(a: torch.Tensor, b: torch.Tensor, block_size: int = 4_194_304) -> torch.Tensor:
    """1-D inner product preserving exact ``torch.dot`` for ``n <= block_size`` while blocking
    very large vectors (>4M elements) to avoid MPS large-inner-dim kernel bugs and float32
    mantissa swamping."""
    n = a.numel()
    if n <= block_size:
        return torch.dot(a, b)
    acc = torch.zeros((), device=a.device, dtype=a.dtype)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        acc = acc + torch.dot(a[start:end], b[start:end])
    return acc


def update_inverse_field_nd_hybrid_lm(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor, 
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    damping_factor: float = 1.0,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None
) -> torch.Tensor:
    """
    Inverse of a displacement field by damped Newton (Levenberg-Marquardt) steps.

    Each iteration solves [I + grad u(x + v) + lambda I] dv = -(v + u(x + v)) per voxel for
    the update of the inverse v of u. The damping lambda = 10 * damping_factor *
    clamp(0.2 - det(I + grad u), 0, 1) is zero where u is regular (plain Newton) and grows
    where u nearly folds, which makes the step more like a fixed-point one. Steps are clipped
    and scaled by ``relaxation`` as in ``update_inverse_field_nd``; the boundary rim is set to
    zero.

    Requires ``spacing`` (physical mode). Without it, falls back to
    ``update_inverse_field_nd`` (fixed point, normalised units, same thresholds). Other parameters and the return value: see ``update_inverse_field_nd``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    boundary_mask = get_boundary_mask(spatial, device, dtype)
    W_disp_cf = torch.movedim(W_disp, -1, 1)
    
    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    if X_phys is not None or (spacing is not None and origin is not None and direction is not None):
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
        
        max_error_norm = float('inf')
        mean_error_norm = float('inf')
        
        for iteration in range(steps):
            if max_error_norm <= max_error_threshold and mean_error_norm <= mean_error_threshold:
                break
            
            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
            del coords_phys
            forward_at_inv = sample_field_cf(W_disp_cf, coords_norm)
            del coords_norm
            error = W_inv_disp + forward_at_inv

            scaled_norm = _scaled_sq_sum(error, spacing_t, divide=True).sqrt_()
            max_error_norm = float(_batch_reduce_amax(scaled_norm).max().item())
            mean_error_norm = float(_batch_reduce_mean(scaled_norm).mean().item())
            
            # Vectorized central finite differences for spatial gradient grad_u (B, *spatial, d, d)
            grad_dims = []
            for s_idx in range(dim):
                spatial_dim = s_idx + 1
                g_s = (torch.roll(forward_at_inv, -1, dims=spatial_dim) - torch.roll(forward_at_inv, 1, dims=spatial_dim)) / (2.0 * spacing_t[s_idx])
                grad_dims.append(g_s)
            grad_u = torch.stack(grad_dims, dim=-1)  # (B, *spatial, d, d)
            
            # M = I + grad(u)
            I_mat = torch.eye(dim, device=device, dtype=dtype).reshape(*([1] * (dim + 1)), dim, dim)
            J_mat = I_mat + grad_u
            
            # Determinant of J_mat (for 3D: det(J))
            if dim == 3:
                det_J = (
                    J_mat[..., 0, 0] * (J_mat[..., 1, 1] * J_mat[..., 2, 2] - J_mat[..., 1, 2] * J_mat[..., 2, 1]) -
                    J_mat[..., 0, 1] * (J_mat[..., 1, 0] * J_mat[..., 2, 2] - J_mat[..., 1, 2] * J_mat[..., 2, 0]) +
                    J_mat[..., 0, 2] * (J_mat[..., 1, 0] * J_mat[..., 2, 1] - J_mat[..., 1, 1] * J_mat[..., 2, 0])
                )
            else:
                det_J = J_mat[..., 0, 0] * J_mat[..., 1, 1] - J_mat[..., 0, 1] * J_mat[..., 1, 0]
            
            # Dynamic spatial damping lambda(y)
            lambda_spatial = torch.clamp(0.2 - det_J.unsqueeze(-1), min=0.0, max=1.0) * damping_factor * 10.0
            
            # Damped system matrix M_lambda = J_mat + lambda * I
            M_lambda = J_mat + lambda_spatial.unsqueeze(-1) * I_mat
            
            # Closed-form Cramer's Rule for M_lambda * delta_v = -error
            b_vec = -error
            a00 = M_lambda[..., 0, 0]
            a01 = M_lambda[..., 0, 1]
            a02 = M_lambda[..., 0, 2] if dim == 3 else torch.zeros_like(a00)
            a10 = M_lambda[..., 1, 0]
            a11 = M_lambda[..., 1, 1]
            a12 = M_lambda[..., 1, 2] if dim == 3 else torch.zeros_like(a00)
            a20 = M_lambda[..., 2, 0] if dim == 3 else torch.zeros_like(a00)
            a21 = M_lambda[..., 2, 1] if dim == 3 else torch.zeros_like(a00)
            a22 = M_lambda[..., 2, 2] if dim == 3 else torch.ones_like(a00)
            
            b0 = b_vec[..., 0]
            b1 = b_vec[..., 1]
            b2 = b_vec[..., 2] if dim == 3 else torch.zeros_like(b0)
            
            det_M = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
            safe_det = torch.where(det_M.abs() < 1e-6, torch.sign(det_M + 1e-6) * 1e-6, det_M)
            
            x0 = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / safe_det
            x1 = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / safe_det
            if dim == 3:
                x2 = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / safe_det
                delta_v = torch.stack([x0, x1, x2], dim=-1)
            else:
                delta_v = torch.stack([x0, x1], dim=-1)
            
            epsilon = 0.75 if iteration == 0 else 0.5
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = delta_v * clip_scale
            W_inv_disp = W_inv_disp + update * relaxation * epsilon
            
            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma)
            W_inv_disp = W_inv_disp * boundary_mask
            
        return torch.movedim(W_inv_disp, -1, 1) if channels_first else W_inv_disp
    else:
        res = update_inverse_field_nd(W_disp, W_inv_disp, steps=steps, relaxation=relaxation, smoothing_sigma=smoothing_sigma,
                                      method='fixed_point', max_error_threshold=max_error_threshold,
                                      mean_error_threshold=mean_error_threshold)
        return torch.movedim(res, -1, 1) if channels_first else res


def integrate_time_varying_velocity_field(
    velocity_fields,
    dt: float = None,
    mode: str = 'forward',
    solver: str = 'rk4',
    spacing=None,
    origin=None,
    direction=None
):
    """
    Displacement of the flow of a time-varying velocity field sampled at T time points.

    Starting from phi = 0, one step is taken per time point k with the velocity v_k held
    fixed during the step: phi <- phi + dt * v_k(x + phi) (or its RK4 / midpoint version).

    Parameters
    ----------
    velocity_fields : list of Tensor, or Tensor (T, B, *spatial, dim)
        v_0 ... v_{T-1}, displacements per unit time.
    dt : float, optional
        Step size; default 1 / T (integrates over [0, 1]).
    mode : {'forward', 'backward'}
        'forward': v_0 first, steps of +dt; 'backward': v_{T-1} first, steps of -dt (others raise).
    solver : {'rk4', 'midpoint', 'euler'}
        Other values raise ValueError.
    spacing, origin, direction : optional
        All three: physical mode (mm). Otherwise normalised [-1, 1] units.

    Returns
    -------
    Tensor, same shape as one v_k: the displacement phi(x) - x.
    """
    if solver not in ('rk4', 'midpoint', 'euler'):
        raise ValueError(f"solver must be 'rk4', 'midpoint' or 'euler', got {solver!r}")
    if mode not in ('forward', 'backward'):
        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")
    if isinstance(velocity_fields, torch.Tensor) and velocity_fields.ndim == 5:
        T = velocity_fields.shape[0]
        vel_list = [velocity_fields[i] for i in range(T)]
    else:
        vel_list = list(velocity_fields)
        T = len(vel_list)
    if dt is None:
        dt = 1.0 / T
        
    B = vel_list[0].shape[0]
    dim = vel_list[0].shape[-1]
    spatial = vel_list[0].shape[1:-1]
    device = vel_list[0].device
    dtype = vel_list[0].dtype
    
    # Initialize composite deformation field at identity
    if spacing is not None and origin is not None and direction is not None:
        X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        direction_rev = np.asarray(direction)[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
        
        # Continuous displacement field mapping
        phi = torch.zeros_like(vel_list[0])
        
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                coords_phys = X_phys + curr_phi
                coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
                return sample_field_cf(v_k_cf, coords_norm)
            
            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == 'midpoint':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                phi = phi + (sign * dt) * k2
            else:  # euler
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi
    else:
        # Standard normalized space branch
        grids = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in spatial]
        identity = torch.stack(                    # (x, y, z) components on the tensor grid
            torch.meshgrid(*grids, indexing='ij')[::-1], dim=-1
        ).unsqueeze(0).expand(B, *spatial, dim)
        
        phi = torch.zeros_like(vel_list[0])
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                sample_coords = identity + curr_phi
                return sample_field_cf(v_k_cf, sample_coords)
            
            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == 'midpoint':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                phi = phi + (sign * dt) * k2
            else:
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi


def _solve_small_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve ``A x = b`` for a batch of small symmetric positive-definite systems, A (B, n, n),
    b (B, n, 1), returning x as (B, 1, n). Jacobi-equilibrated Gauss-Jordan elimination
    without pivoting using only elementwise kernels: valid for SPD matrices (all pivots
    positive) and several times cheaper than ``torch.linalg.solve`` for n <= 8 on MPS and
    CUDA, where the library solve pays a fixed per-call cost far above the arithmetic.
    """
    n = A.shape[-1]
    # Jacobi equilibration D A D y = D b, x = D y with D = diag(A)^-1/2: unit diagonal keeps
    # the unpivoted elimination accurate on the ill-conditioned Gram matrices Anderson
    # produces (measured 3.8e-7 against torch.linalg.solve on a 32x32 ripple field, 3.3e-6
    # without equilibration).
    d_inv = torch.rsqrt(torch.diagonal(A, dim1=1, dim2=2))          # (B, n)
    A = A * d_inv.unsqueeze(1) * d_inv.unsqueeze(2)
    b = b * d_inv.unsqueeze(2)
    M = torch.cat([A, b], dim=-1)                       # (B, n, n+1)
    for i in range(n):
        row = M[:, i:i + 1, :] / M[:, i:i + 1, i:i + 1]  # normalised pivot row (B, 1, n+1)
        fac = M[:, :, i:i + 1]                           # column i (B, n, 1)
        M = M - fac * row
        M = torch.cat([M[:, :i], row, M[:, i + 1:]], dim=1)
    return (M[:, :, -1] * d_inv).unsqueeze(1)


def update_inverse_field_nd_anderson(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    m: int = 5,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
    check_interval: int = 1,
    check_after: int | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Inverse of a displacement field: the fixed-point iteration of ``update_inverse_field_nd``
    with Anderson acceleration over the last ``m`` iterates.

    Each accelerated iterate is a least-squares combination of the recent fixed-point
    iterates. It is kept only if its residual is below that of the plain fixed-point step;
    otherwise the plain step is used. The run stops when both the max and the mean error are
    at or below their thresholds.

    Every batch element is solved independently: the clip threshold, the Anderson
    least-squares system and the stopping test are per element, and an element that has met
    the thresholds is frozen (its value no longer changes) while the others continue. A call on
    a batch of fields therefore returns exactly what the same calls on each field alone return.

    Host-device synchronisation happens only at the stopping tests. ``check_interval`` is how
    often (in iterations) the test runs; ``check_after`` moves the first test to that
    iteration (clamped to ``[1, steps]``), so a caller that knows from the previous call how
    many iterations were needed can run them unchecked and pay one synchronisation. Freezing
    makes the result independent of when the test runs. When ``stats`` is a dict, it receives
    ``'steps'`` (the largest number of iterations any element needed to converge, or the
    number run when not converged) and ``'converged'``.
    Other parameters and the return value: see ``update_inverse_field_nd``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()

    use_physical = (X_phys is not None or
                    (spacing is not None and origin is not None and direction is not None))

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
        W_disp_cf = torch.movedim(W_disp, -1, 1)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    bshape = (B,) + (1,) * (W_disp.dim() - 1)
    ones_b = torch.ones(bshape, device=device, dtype=dtype)

    def itk_fixed_point_step(v_curr, iteration, compute_mean: bool = True):
        if use_physical:
            coords_phys = X_phys + v_curr
            coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
            del coords_phys
            forward_at_inv = sample_field_cf(W_disp_cf, coords_norm)
            del coords_norm
            error = forward_at_inv.add_(v_curr)
            sq_sum = _scaled_sq_sum(error, spacing_t, divide=True)
        else:
            coords = identity + v_curr
            forward_at_inv = sample_field_cf(W_disp_cf, coords)
            del coords
            error = forward_at_inv.add_(v_curr)
            sq_sum = _scaled_sq_sum(error, voxel_scale, divide=False)

        # Compute squared residual sum (B,) before turning sq_sum into scaled_norm in-place,
        # so the 3-channel error field does not need to stay alive until the Anderson check.
        residual_fp = _batch_reduce_sum(sq_sum)
        scaled_norm = sq_sum.sqrt_()

        # Per-element reductions (B,): each field in the batch is independent.
        max_err = _batch_reduce_amax(scaled_norm)
        mean_err = _batch_reduce_mean(scaled_norm) if compute_mean else None

        epsilon = 0.75 if iteration == 0 else 0.5
        clip_threshold = (epsilon * max_err).reshape(bshape)
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            ones_b,
        )
        del scaled_norm
        error.neg_()
        error.mul_(clip_scale)
        del clip_scale
        error.mul_(epsilon)
        v_new = error.add_(v_curr)
        del error

        if smoothing_sigma > 0.0:
            v_new = separable_gaussian_filter(v_new, smoothing_sigma)     # sigma in voxels

        v_new.mul_(boundary_mask)
        return v_new, max_err, mean_err, residual_fp

    v_k = W_inv_disp.clone()
    R_history: list[torch.Tensor] = []
    G_history: list[torch.Tensor] = []
    done = torch.zeros(B, dtype=torch.bool, device=device)         # element has met the thresholds
    conv_step = torch.full((B,), steps, dtype=torch.int64, device=device)   # 0-based iteration it converged at
    eye_reg = 1e-10 * torch.eye(max(m, 1), device=device, dtype=dtype)
    first_check = int(min(max(check_after, 1), steps)) if check_after is not None else None
    steps_taken = steps
    converged = False
    mps_large_field = (device.type == 'mps' and v_k.numel() >= 16_777_216)

    for iteration in range(steps):
        checks_on = (check_interval is not None and check_interval > 0)
        if first_check is None:
            should_check = checks_on and ((iteration + 1) % check_interval == 0 or iteration == steps - 1)
        else:
            should_check = (iteration + 1 == first_check
                            or (checks_on and iteration + 1 > first_check
                                and (iteration + 1 - first_check) % check_interval == 0)
                            or iteration == steps - 1)
        g_k, max_err, mean_err, residual_fp = itk_fixed_point_step(v_k, iteration, compute_mean=True)
        conv_now = (max_err <= max_error_threshold) & (mean_err <= mean_error_threshold)   # (B,)
        conv_step = torch.where(conv_now & ~done, torch.full_like(conv_step, iteration), conv_step)
        done_next = done | conv_now

        # Check convergence before allocating history/Anderson buffers: when every element in
        # the batch is either already frozen or converging now, v_next is unused.
        if should_check:
            flag = torch.stack([done_next.all().to(torch.int64), conv_step.max()]).tolist()   # one sync
            if flag[0]:
                v_k = torch.where(done.reshape(bshape), v_k, g_k)
                del g_k
                converged = True
                steps_taken = int(flag[1]) + 1
                break

        g_k_flat = g_k.reshape(B, -1)
        if len(R_history) == m + 1:
            r_k = R_history.pop(0)
            G_history.pop(0)
            torch.sub(g_k_flat, v_k.reshape(B, -1), out=r_k)
        else:
            r_k = g_k_flat - v_k.reshape(B, -1)
        R_history.append(r_k)
        G_history.append(g_k_flat)

        m_k = len(R_history)

        if m_k < 2:
            v_next = g_k
        else:
            n_cols = m_k - 1
            r_newest = R_history[-1]
            g_newest = G_history[-1]
            # Compute residual differences dR_j = r_newest - R_j on the fly into reusable
            # scratch buffers instead of materialising all m difference fields at once.
            diff_i = torch.empty_like(r_newest)
            diff_j = torch.empty_like(r_newest) if n_cols > 1 else None
            gram = torch.zeros(B, n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(B, n_cols, 1, device=device, dtype=dtype)
            for i_col in range(n_cols):
                torch.sub(r_newest, R_history[i_col], out=diff_i)
                for b_idx in range(B):
                    rhs[b_idx, i_col, 0] = _dot_1d(diff_i[b_idx], r_newest[b_idx])
                    gram[b_idx, i_col, i_col] = _dot_1d(diff_i[b_idx], diff_i[b_idx])
                for j_col in range(i_col + 1, n_cols):
                    torch.sub(r_newest, R_history[j_col], out=diff_j)
                    for b_idx in range(B):
                        val = _dot_1d(diff_i[b_idx], diff_j[b_idx])
                        gram[b_idx, i_col, j_col] = val
                        gram[b_idx, j_col, i_col] = val
            del diff_j
            gram = gram + eye_reg[:n_cols, :n_cols]
            gamma = _solve_small_spd(gram, rhs).squeeze(1)                    # (B, n)
            v_new_flat = g_newest.clone()
            tmp_flat = torch.empty_like(r_newest)
            for j in range(n_cols):
                torch.sub(r_newest, R_history[j], out=diff_i)
                torch.sub(g_newest, G_history[j], out=tmp_flat)
                tmp_flat.add_(diff_i)
                tmp_flat.mul_(gamma[:, j:j + 1])
                v_new_flat.sub_(tmp_flat)
            del diff_i, tmp_flat

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate.mul_(boundary_mask)

            if use_physical:
                coords_phys_c = X_phys + v_candidate
                coords_norm_c = physical_to_normalized_fast(coords_phys_c, M_norm, b_norm)
                del coords_phys_c
                fwd_at_c = sample_field_cf(W_disp_cf, coords_norm_c)
                del coords_norm_c
                error_c = fwd_at_c.add_(v_candidate)
                sq_c = _scaled_sq_sum(error_c, spacing_t, divide=True)
                del error_c
                residual_aa = _batch_reduce_sum(sq_c)
                del sq_c
            else:
                coords_c = identity + v_candidate
                fwd_at_c = sample_field_cf(W_disp_cf, coords_c)
                del coords_c
                error_c = fwd_at_c.add_(v_candidate)
                sq_c = _scaled_sq_sum(error_c, voxel_scale, divide=False)
                del error_c
                residual_aa = _batch_reduce_sum(sq_c)
                del sq_c

            accept = (residual_aa <= residual_fp * 1.21).reshape(bshape)      # squared norms: 1.1**2
            v_next = torch.where(accept, v_candidate, g_k)
            del v_candidate

        # Freeze: an element already converged keeps its value; one converging now takes the
        # plain step g_k (as the unbatched early exit did); the rest take v_next.
        v_k = torch.where(done.reshape(bshape), v_k,
                          torch.where(conv_now.reshape(bshape), g_k, v_next))
        del v_next, g_k
        done = done_next

        if not should_check and mps_large_field and (iteration + 1) % 4 == 0:
            torch.mps.synchronize()

    del R_history, G_history
    if stats is not None:
        stats['steps'] = steps_taken
        stats['converged'] = converged

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


def update_inverse_field_nd(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor = None, 
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    method: str = 'anderson',
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing = None,
    origin = None,
    direction = None,
    X_phys = None,
    check_interval: int = 1,
    max_iters: int | None = None,
    check_after: int | None = None,
    stats: dict | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Inverse of a displacement field u: v with v(x) + u(x + v(x)) = 0, so that x -> x + v(x)
    undoes x -> x + u(x).

    The fixed-point iteration (``method='fixed_point'``; other names raise ValueError) follows
    ITK's InvertDisplacementFieldImageFilter:
    v <- v - eps * clip(v + u(x + v)), with eps = 0.75 on the first iteration and 0.5 after.
    Each voxel's update is clipped to eps * (the current max error). After each step, v is
    optionally Gaussian-smoothed and set to zero on the one-voxel boundary rim.

    Parameters
    ----------
    W_disp : Tensor (B, *spatial, dim) or (B, dim, *spatial)
        The field u to invert.
    W_inv_disp : Tensor, optional
        Starting guess (warm start). Default -u.
    steps : int, default 30
        Maximum iterations (``max_iters`` overrides it).
    relaxation : float, default 1.0
        Extra step scale ('fixed_point' and 'hybrid_lm'; a value other than 1 with 'anderson'
        raises ValueError).
    smoothing_sigma : float, default 0
        Gaussian sigma (voxels) applied to v after each step (0 = none).
    method : {'anderson', 'hybrid_lm', 'fixed_point'}, default 'anderson'
        See ``update_inverse_field_nd_anderson`` / ``update_inverse_field_nd_hybrid_lm``.
    max_error_threshold, mean_error_threshold : float, default 0.1, 0.001
        Stopping thresholds on the max / mean residual |v + u(x + v)|, in voxels; the
        solvers stop when both are met.
    spacing, origin, direction : optional
        ANTs metadata. With all three, u and v are in mm. Otherwise they are in normalised
        [-1, 1] units.
    X_phys : Tensor, optional
        Precomputed physical grid (1, *spatial, dim) of that metadata, to save rebuilding it
        (the metadata is still required: ValueError without it).
    check_interval : int, default 1
        'anderson' and 'fixed_point': how often the stopping test runs.
    check_after : int, optional
        'anderson' and 'fixed_point': iteration of the first stopping test (see
        ``update_inverse_field_nd_anderson``).
    stats : dict, optional
        'anderson' and 'fixed_point': receives ``'steps'`` and ``'converged'``.

    Returns
    -------
    Tensor: v, in the layout of ``W_disp``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    if max_iters is not None:
        steps = max_iters
    if method not in ('anderson', 'hybrid_lm', 'fixed_point'):
        raise ValueError(f"unknown inverse method {method!r}: use 'anderson', 'hybrid_lm' or 'fixed_point'")
    if method == 'anderson' and relaxation != 1.0:
        raise ValueError("relaxation is not used by method='anderson' (use 'fixed_point' or 'hybrid_lm')")
    if method == 'hybrid_lm':
        res = update_inverse_field_nd_hybrid_lm(
            W_disp, W_inv_disp, steps=steps, relaxation=relaxation,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    effective_check_interval = kwargs.get('check_interval', check_interval)
    if method == 'anderson':
        res = update_inverse_field_nd_anderson(
            W_disp, W_inv_disp, steps=steps,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys,
            check_interval=effective_check_interval,
            check_after=check_after, stats=stats,
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys needs the spacing / origin / direction it was built from")

    use_physical = (X_phys is not None or
                    (spacing is not None and origin is not None and direction is not None))

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )

    W_disp_cf = torch.movedim(W_disp, -1, 1)
    v_k = -W_disp.clone() if W_inv_disp is None else W_inv_disp.clone()
    bshape = (B,) + (1,) * (W_disp.dim() - 1)
    ones_b = torch.ones(bshape, device=device, dtype=dtype)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    conv_step = torch.full((B,), steps, dtype=torch.int64, device=device)
    first_check = int(min(max(check_after, 1), steps)) if check_after is not None else None
    steps_taken = steps
    converged = False
    mps_large_field = (device.type == 'mps' and v_k.numel() >= 16_777_216)

    for iteration in range(steps):
        checks_on = (effective_check_interval is not None and effective_check_interval > 0)
        if first_check is None:
            should_check = checks_on and (
                (iteration + 1) % effective_check_interval == 0 or iteration == steps - 1
            )
        else:
            should_check = (
                iteration + 1 == first_check
                or (checks_on and iteration + 1 > first_check
                    and (iteration + 1 - first_check) % effective_check_interval == 0)
                or iteration == steps - 1
            )

        if use_physical:
            coords_phys = X_phys + v_k
            coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
            del coords_phys
            forward_at_inv = sample_field_cf(W_disp_cf, coords_norm)
            del coords_norm
            error = forward_at_inv.add_(v_k)
            scaled_norm = _scaled_sq_sum(error, spacing_t, divide=True).sqrt_()
        else:
            coords = identity + v_k
            forward_at_inv = sample_field_cf(W_disp_cf, coords)
            del coords
            error = forward_at_inv.add_(v_k)
            scaled_norm = _scaled_sq_sum(error, voxel_scale, divide=False).sqrt_()

        max_err = _batch_reduce_amax(scaled_norm)
        mean_err = _batch_reduce_mean(scaled_norm)
        conv_now = (max_err <= max_error_threshold) & (mean_err <= mean_error_threshold)
        conv_step = torch.where(conv_now & ~done, torch.full_like(conv_step, iteration), conv_step)
        done_next = done | conv_now

        epsilon = 0.75 if iteration == 0 else 0.5
        clip_threshold = (epsilon * max_err).reshape(bshape)
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            ones_b,
        )
        del scaled_norm
        error.neg_()
        error.mul_(clip_scale)
        del clip_scale
        error.mul_(relaxation * epsilon)
        g_k = error.add_(v_k)
        del error

        if smoothing_sigma > 0.0:
            g_k = separable_gaussian_filter(g_k, smoothing_sigma)
        g_k.mul_(boundary_mask)

        v_k = torch.where(done.reshape(bshape), v_k, g_k)
        del g_k
        done = done_next

        if should_check:
            flag = torch.stack([done.all().to(torch.int64), conv_step.max()]).tolist()
            if flag[0]:
                converged = True
                steps_taken = int(flag[1]) + 1
                break
        elif mps_large_field and (iteration + 1) % 4 == 0:
            torch.mps.synchronize()

    if stats is not None:
        stats['steps'] = steps_taken
        stats['converged'] = converged

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


def compute_inverse_identity_error_nd(
    warp_fwd: torch.Tensor,
    warp_inv: torch.Tensor,
    spacing=None,
    origin=None,
    direction=None,
    is_displacement: bool = True,
    fwd_is_disp: bool = None,
    inv_is_disp: bool = None
) -> torch.Tensor:
    """
    Map of the inverse-consistency error |u_fwd(x) + u_inv(x + u_fwd(x))| in mm: how far
    x -> forward -> inverse lands from x.

    Parameters
    ----------
    warp_fwd, warp_inv : Tensor (B, *spatial, dim) or (*spatial, dim)
        Physical displacements (or absolute positions, see below), tensor order.
    spacing, origin, direction : optional
        ANTs metadata. Default unit spacing, zero origin, identity direction.
    is_displacement : bool, default True
        False: both inputs are absolute positions x + u(x).
    fwd_is_disp, inv_is_disp : bool, optional
        Per-input override of ``is_displacement``.

    Returns
    -------
    Tensor (B, *spatial). No boundary or out-of-domain masking: points that leave the grid
    read u_inv from the border.
    """
    dim = warp_fwd.shape[-1]
    if warp_fwd.dim() == dim + 1:
        warp_fwd = warp_fwd.unsqueeze(0)
    if warp_inv.dim() == dim + 1:
        warp_inv = warp_inv.unsqueeze(0)

    spatial = warp_fwd.shape[1:-1]
    device = warp_fwd.device
    dtype = warp_fwd.dtype

    if spacing is None:
        spacing = [1.0] * dim
    if origin is None:
        origin = [0.0] * dim
    if direction is None:
        direction = np.eye(dim)
    else:
        direction = np.asarray(direction)[:dim, :dim]

    spatial_tuple = tuple(spatial)
    spacing_tuple = tuple(float(s) for s in spacing)
    origin_tuple = tuple(float(o) for o in origin)

    X_phys = get_physical_grid_torch(spatial_tuple, spacing_tuple, origin_tuple, direction, device=device, dtype=dtype)

    use_fwd_disp = is_displacement if fwd_is_disp is None else fwd_is_disp
    use_inv_disp = is_displacement if inv_is_disp is None else inv_is_disp

    disp_fwd = warp_fwd if use_fwd_disp else (warp_fwd - X_phys)
    disp_inv = warp_inv if use_inv_disp else (warp_inv - X_phys)

    y_norm = physical_to_normalized_torch(X_phys + disp_fwd, spatial_tuple, spacing_tuple, origin_tuple, direction)
    disp_inv_sampled = compose_grids(disp_inv, y_norm)

    return torch.norm(disp_fwd + disp_inv_sampled, dim=-1)


def calculate_inverse_identity_error(W_disp: torch.Tensor, W_inv_disp: torch.Tensor, spacing, origin, direction) -> dict:
    """
    Inverse-consistency error of a displacement field and its inverse, in mm:
    |u_inv(x) + u(x + u_inv(x))|, i.e. how far x -> inverse -> forward lands from x. The
    order is the reverse of ``compute_inverse_identity_error_nd``.

    Parameters
    ----------
    W_disp, W_inv_disp : Tensor (1, *spatial, dim) or (*spatial, dim)
        Physical displacements u and u_inv, tensor order (spatial z, y, x).
    spacing, origin, direction : ANTs metadata of the grid (all required).

    Returns
    -------
    dict
        - ``'max_error'``: max over interior voxels.
        - ``'mean_error'``: sum over interior voxels / number of interior voxels.
        - ``'error_map'``: Tensor (*spatial).

        Interior voxels exclude the one-voxel boundary rim and points whose x + u_inv(x)
        leaves the grid; they are set to 0 in the map and excluded from the mean.
    """
    dim = len(spacing)
    if W_disp.ndim == dim + 1:
        W_disp = W_disp.unsqueeze(0)
    if W_inv_disp.ndim == dim + 1:
        W_inv_disp = W_inv_disp.unsqueeze(0)
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
    coords_phys = X_phys + W_inv_disp
    del X_phys
    
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    
    shape_t = torch.tensor(spatial, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    
    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
    del coords_phys
    
    forward_at_inv = compose_grids(W_disp, coords_norm)
    valid_mask = ((coords_norm >= -1) & (coords_norm <= 1)).all(dim=-1)
    del coords_norm
    for ax in range(len(spatial)):
        sl_lo = [slice(None)] * valid_mask.ndim
        sl_lo[ax + 1] = slice(0, 1)
        valid_mask[tuple(sl_lo)] = False
        sl_hi = [slice(None)] * valid_mask.ndim
        sl_hi[ax + 1] = slice(-1, None)
        valid_mask[tuple(sl_hi)] = False

    forward_at_inv.add_(W_inv_disp)
    error_norm = torch.linalg.vector_norm(forward_at_inv, dim=-1)
    del forward_at_inv
    error_norm.masked_fill_(~valid_mask, 0.0)
    if valid_mask.device.type == 'mps':
        n_valid = max(float(_batch_reduce_sum(valid_mask.to(dtype)).item()), 1.0)
        err_sum = float(_batch_reduce_sum(error_norm).item())
        err_max = float(_batch_reduce_amax(error_norm).item())
    else:
        n_valid = max(float(valid_mask.sum(dtype=torch.int64).item()), 1.0)
        err_sum = float(error_norm.sum(dtype=torch.float64).item())
        err_max = float(error_norm.max().item())
    del valid_mask
    return {
        'max_error': err_max,
        'mean_error': err_sum / n_valid,
        'error_map': error_norm.squeeze(0),
    }
