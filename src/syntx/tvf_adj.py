"""
Deprecated prototype of time-varying velocity field (TVF) registration with a hand-written
adjoint (backward-in-time) gradient instead of autograd.

.. deprecated:: 0.5.0
    Superseded by ``syntx.tvf`` (``syntx.tvf.TVFModel``), which uses autograd through the ODE
    integration. Importing this module emits a ``DeprecationWarning``. It is kept for the
    tests in ``tests/test_tvf_adj.py``; do not use it for real registrations.

Known limitation: the "adjoint" is approximate -- the image gradient is multiplied by the
similarity gradient at every keyframe and propagated back by warping with ``identity + v * dt``;
it is not the exact adjoint of ``integrate_forward``. ``tvf_registration_adjoint`` registers a
moving image already resampled through ``initial_transform`` (``warpedmovout`` itself is
resampled once, from the original moving image); like every syntx registration it returns its
warps as files owned by the caller.

Conventions: velocities / displacements are in voxels with channels (x, y[, z]); normalised
coordinates use ``grid_sample(align_corners=True)``, so one voxel is 2 / (n - 1).
"""

import os
import math
import warnings
import torch
import torch.nn.functional as F
from .syn import separable_gaussian_filter
import numpy as np
import ants
import time
from .transform import SyNToTransform

warnings.warn(
    "syntx.tvf_adj is deprecated and contains known defects. "
    "Use syntx.tvf() for production registration.",
    DeprecationWarning,
    stacklevel=2,
)

def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    """
    Physical coordinates of every voxel of a 2-D or 3-D grid (not used inside this module).

    Parameters
    ----------
    shape : tuple of int
        Grid shape, tensor order (z, y, x) / (y, x).
    spacing, origin : sequence of float
        ANTs (x, y[, z]) order.
    direction : array-like
        dim x dim direction matrix (or its flattened form), applied to (x, y[, z]) vectors.
    device, dtype : torch device / dtype, default 'cpu', float32

    Returns
    -------
    torch.Tensor
        Shape (*shape, dim); the last axis holds (x, y[, z]) physical coordinates
        ``direction @ (index * spacing) + origin``. ``physical_to_normalized_torch`` is its
        inverse (up to the normalisation).
    """
    dim = len(shape)
    if dim not in (2, 3):
        raise ValueError(f"get_physical_grid_torch: 2-D or 3-D shape expected, got {shape}")
    axes = [torch.arange(n, device=device, dtype=dtype) for n in shape]          # tensor order
    idx = torch.stack(torch.meshgrid(*axes, indexing='ij')[::-1], dim=-1)         # (x, y[, z])
    sp = torch.as_tensor(np.asarray(spacing, dtype=float), device=device, dtype=dtype)
    org = torch.as_tensor(np.asarray(origin, dtype=float), device=device, dtype=dtype)
    D = torch.as_tensor(np.asarray(direction, dtype=float).reshape(dim, dim), device=device, dtype=dtype)
    return (idx * sp) @ D.t() + org


def physical_to_normalized_torch(phys_grid, shape, spacing, origin, direction):
    """
    Map physical points (last axis (x, y[, z]), as returned by ``get_physical_grid_torch``) to
    the normalised [-1, 1] coordinates used by ``grid_sample(align_corners=True)``, last axis
    (x, y[, z]). ``shape`` is in tensor order; ``spacing`` / ``origin`` in ANTs (x, y[, z])
    order; ``direction`` dim x dim (or flattened). Not used inside this module.

    Returns
    -------
    torch.Tensor
        Same shape as ``phys_grid``.
    """
    device, dtype = phys_grid.device, phys_grid.dtype
    dim = phys_grid.shape[-1]
    sp = torch.as_tensor(np.asarray(spacing, dtype=float), device=device, dtype=dtype)
    org = torch.as_tensor(np.asarray(origin, dtype=float), device=device, dtype=dtype)
    D = torch.as_tensor(np.asarray(direction, dtype=float).reshape(dim, dim), device=device, dtype=dtype)
    idx = ((phys_grid - org) @ torch.linalg.inv(D).t()) / sp                      # (x, y[, z]) voxels
    size_xyz = torch.as_tensor(list(shape)[::-1], device=device, dtype=dtype)
    return idx / (size_xyz - 1) * 2.0 - 1.0


def image_gradient(I):
    """
    Central-difference spatial gradient of a single-channel image, in voxel units, with
    periodic boundaries (``torch.roll``).

    Parameters
    ----------
    I : torch.Tensor
        Shape (B, 1, H, W) or (B, 1, D, H, W).

    Returns
    -------
    torch.Tensor
        Shape (B, dim, *spatial), channels ordered (d/dx, d/dy[, d/dz]) (x = last tensor axis).
        None for other dimensionalities.
    """
    dim = I.dim() - 2
    if dim == 2:
        grad_x = (torch.roll(I, shifts=-1, dims=-1) - torch.roll(I, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(I, shifts=-1, dims=-2) - torch.roll(I, shifts=1, dims=-2)) / 2.0
        return torch.cat([grad_x, grad_y], dim=1) # [B, 2, H, W]
    elif dim == 3:
        grad_x = (torch.roll(I, shifts=-1, dims=-1) - torch.roll(I, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(I, shifts=-1, dims=-2) - torch.roll(I, shifts=1, dims=-2)) / 2.0
        grad_z = (torch.roll(I, shifts=-1, dims=-3) - torch.roll(I, shifts=1, dims=-3)) / 2.0
        return torch.cat([grad_x, grad_y, grad_z], dim=1) # [B, 3, D, H, W]



def fluid_smooth(v, sigma, dim):
    """
    Gaussian smoothing of a channels-first vector field with ``syn.separable_gaussian_filter``
    (not used inside this module). Not in place: returns a new tensor.

    Parameters
    ----------
    v : torch.Tensor
        Shape (B, dim, *spatial).
    sigma : float
        Gaussian sigma passed to ``separable_gaussian_filter`` with its default ``sigma_mode``;
        ``sigma <= 0`` returns ``v`` unchanged.
    dim : int
        Unused.
    """
    if sigma <= 0:
        return v
    v_smooth = v.movedim(1, -1)
    v_smooth = separable_gaussian_filter(v_smooth, sigma=sigma)
    return v_smooth.movedim(-1, 1)

def _identity_grid(spatial_shape, device, dtype=torch.float32):
    """Normalised identity sampling grid (1, *spatial, dim), last axis (x, y[, z])."""
    axes = [torch.linspace(-1, 1, n, device=device, dtype=dtype) for n in spatial_shape]
    return torch.stack(torch.meshgrid(*axes, indexing='ij')[::-1], dim=-1).unsqueeze(0)


def _voxel_to_normalised(spatial_shape, device, dtype=torch.float32):
    """Per-channel factor (x, y[, z]) from voxels to align_corners=True normalised units."""
    return torch.tensor([2.0 / (n - 1) for n in list(spatial_shape)[::-1]], device=device, dtype=dtype)


def integrate_svf(v, n_steps=5):
    """
    Scaling-and-squaring exponential of a stationary velocity field.

    ``v`` is (1, dim, *spatial) in voxels, channels (x, y[, z]). The field is scaled by
    ``2**-n_steps`` and composed with itself ``n_steps`` times
    (u <- u + u(x + u), bilinear, border padding). Returns the displacement of exp(v) in
    voxels, same shape and channel order as ``v``.
    """
    dim = v.shape[1]
    spatial = v.shape[2:]
    grid = _identity_grid(spatial, v.device, v.dtype)
    scale = _voxel_to_normalised(spatial, v.device, v.dtype)
    u = v / (2 ** n_steps)
    for _ in range(n_steps):
        u_cl = u.movedim(1, -1)                                   # (1, *spatial, dim) voxels
        u_at = F.grid_sample(u, grid + u_cl * scale, mode='bilinear', padding_mode='border',
                             align_corners=True)
        u = u + u_at
    return u


def integrate_forward(v_list, spatial_shape):
    """
    Euler integration of piecewise-constant velocity fields on a normalised [-1, 1] grid.

    Starting from the identity grid, each step samples the velocity at the current positions
    (bilinear, border padding, align_corners=True), converts it from voxels to normalised
    units (x 2 / (size - 1)), and moves the points by ``phi <- phi - v * dt`` with
    ``dt = 1 / len(v_list)`` (note the minus sign).

    Parameters
    ----------
    v_list : list of torch.Tensor
        One velocity per time step, each (1, dim, *spatial), channels (vx, vy[, vz]) in voxels.
    spatial_shape : tuple of int
        (H, W) or (D, H, W).

    Returns
    -------
    list of torch.Tensor
        ``len(v_list) + 1`` sampling grids (1, *spatial, dim), (x, y[, z]) order: the identity
        followed by the grid after each step. The last one is the final map.
    """
    device = v_list[0].device
    phi = _identity_grid(spatial_shape, device)
    scale = _voxel_to_normalised(spatial_shape, device)
    dt = 1.0 / len(v_list)

    phi_history = [phi.clone()]
    for v in v_list:
        v_sampled = F.grid_sample(v, phi, mode='bilinear', padding_mode='border', align_corners=True)
        phi = phi - v_sampled.movedim(1, -1) * scale * dt
        phi_history.append(phi.clone())

    return phi_history

class TVFRegistrationAdjoint:
    """
    Deprecated adjoint-gradient TVF optimiser (multi-resolution, 3 velocity keyframes).

    Images are converted to tensor (z, y, x) order and z-scored. The velocity ``self.v`` has
    shape (3, 1, dim, *level_shape) (voxel units of the current level, channels (x, y[, z]))
    and starts at zeros on the coarsest grid ``max(8, s // levels[0])``.

    Parameters
    ----------
    fixed_image, moving_image : ants.ANTsImage
        2-D or 3-D. Only the fixed image's spacing is used.
    flow_sigma : float, default 2.0
        Gaussian sigma (physical units, divided by the level spacing) for smoothing the
        adjoint gradient.
    total_sigma : float, default 0.5
        Gaussian sigma (physical units) for smoothing the velocity after each update.
    lr : float, default 0.5
        Maximum voxel step: the update is scaled so its largest vector has norm ``lr``.
    lncc_radius : int, default 2
        Local NCC window is ``2 * lncc_radius + 1``.
    device : str, default 'cpu'
        Torch device.
    levels : list of int, default [4, 2, 1]
        Shrink factors; level shape is ``max(8, s // level)``.
    reg_iterations : list of int, default [100, 100, 20]
        Iterations per level; 0 skips the level.

    Fixed internals: ``n_time_steps = 3``, ``cfl_momentum = 0.95``. Voxel-unit velocities are
    rescaled per axis by (n_new - 1) / (n_old - 1) when moved to a finer level.
    """
    def __init__(self, fixed_image, moving_image, flow_sigma=2.0, total_sigma=0.5, lr=0.5, lncc_radius=2, device='cpu', levels=[4, 2, 1], reg_iterations=[100, 100, 20]):
        self.fixed = fixed_image
        self.moving = moving_image
        self.dim = fixed_image.dimension
        self.device = torch.device(device)
        self.flow_sigma = flow_sigma
        self.total_sigma = total_sigma
        self.lr = lr
        self.lncc_window = 2 * lncc_radius + 1
        self.levels = levels
        self.reg_iterations = reg_iterations
        self.cfl_momentum = 0.95
        self.n_time_steps = 3
        
        # Load images into tensors. ANTs is XYZ, PyTorch needs ZYX.
        f_np = self.fixed.numpy()
        m_np = self.moving.numpy()
        if self.dim == 2:
            f_np = f_np.T
            m_np = m_np.T
        elif self.dim == 3:
            f_np = f_np.transpose(2, 1, 0)
            m_np = m_np.transpose(2, 1, 0)
            
        # ANTs spacing is XYZ, we need ZYX for PyTorch grids
        self.spacing = list(self.fixed.spacing)[::-1]
        self.shape = f_np.shape
        self.f_tensor = torch.tensor(f_np, device=self.device).unsqueeze(0).unsqueeze(0)
        self.m_tensor = torch.tensor(m_np, device=self.device).unsqueeze(0).unsqueeze(0)
        
        # Initialize TVF velocity field [T, 1, dim, *spatial]
        self.v = torch.zeros((self.n_time_steps, 1, self.dim, *self.shape), device=self.device)# Normalize
        self.f_tensor = (self.f_tensor - self.f_tensor.mean()) / (self.f_tensor.std() + 1e-8)
        self.m_tensor = (self.m_tensor - self.m_tensor.mean()) / (self.m_tensor.std() + 1e-8)
        
        self.shape = self.f_tensor.shape[2:]
        # Initialize v at the coarsest level
        coarsest_shape = [max(8, s // self.levels[0]) for s in self.shape]
        self.v = torch.zeros((self.n_time_steps, 1, self.dim, *coarsest_shape), device=self.device)
        
    def fit(self):
        """
        Run the multi-resolution adjoint optimisation and return the velocity.

        Per iteration: integrate (``integrate_forward``), warp the moving image, compute the
        local-NCC gradient with respect to the warped image, form per-keyframe gradients
        ``adjoint * grad(M_t)`` while warping the adjoint back with ``identity + v * dt``,
        Gaussian-smooth them (``flow_sigma``), take a step of maximum voxel length ``lr`` with
        bias-corrected momentum (0.95), then smooth the velocity (``total_sigma``). Prints
        progress unconditionally.

        Returns
        -------
        torch.Tensor
            ``self.v``, shape (3, 1, dim, *self.shape) when the last executed level is at full
            resolution. If the final level was skipped (0 iterations) the final upsampling
            ``view`` is applied to a coarse tensor and raises.
        """
        print(f"Starting Multi-Res Adjoint TVF Optimization on {self.device}...")
        
        for level_idx, (level, iters) in enumerate(zip(self.levels, self.reg_iterations)):
            if iters == 0:
                continue
                
            curr_shape = [max(8, s // level) for s in self.shape]
            
            # Interpolate TVF to current level
            if list(self.v.shape[3:]) != list(curr_shape):
                self.v = self._resample_velocity(curr_shape)
            
            # Interpolate images to current level
            curr_f = F.interpolate(self.f_tensor, size=curr_shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
            curr_m = F.interpolate(self.m_tensor, size=curr_shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
            
            print(f"--- Level {level} (Shape: {curr_shape}) | Iterations: {iters} ---")
            momentum_buffer = None
            curr_spacing = [sp * (img_dim / curr_dim) for sp, img_dim, curr_dim in zip(self.spacing, self.shape, curr_shape)]
            
            norm_scale = 1.0 / _voxel_to_normalised(curr_shape, self.device)   # voxels per normalised unit
            
            for epoch in range(iters):
                v_list = [self.v[t] for t in range(self.n_time_steps)]
                phi_history = integrate_forward(v_list, curr_shape)
                
                phi_final = phi_history[-1]
                moved_final = F.grid_sample(curr_m, phi_final, align_corners=True, mode='bilinear', padding_mode='zeros')
                
                from .syn import local_ncc_loss_nd
                
                with torch.enable_grad():
                    moved_detached = moved_final.detach().clone()
                    moved_detached.requires_grad_(True)
                    loss = local_ncc_loss_nd(curr_f, moved_detached, window_size=self.lncc_window)
                    loss_mean = loss.mean()
                    loss_mean.backward()
                    grad_J = moved_detached.grad.clone()
                    
                # Adjoint Backward Pass
                adjoint = grad_J
                dt = 1.0 / self.n_time_steps
                
                adj_grads = []
                for t in reversed(range(self.n_time_steps)):
                    # Compute spatial gradient of M(t)
                    phi_t = phi_history[t]
                    M_t = F.grid_sample(curr_m, phi_t, align_corners=True, mode='bilinear', padding_mode='zeros')
                    grad_M_t = image_gradient(M_t.detach())
                    
                    # Gradient w.r.t v(t) is adjoint * grad_M(t)
                    adj_grad_t = adjoint * grad_M_t
                    adj_grads.append(adj_grad_t)
                    
                    # Propagate adjoint backward: A(t-1) = A(t) warped by -v(t)
                    if t > 0:
                        v_norm = v_list[t].detach().squeeze(0).movedim(0, -1) / norm_scale
                        # We use -v to warp backward
                        inv_phi = phi_history[0] + v_norm * dt
                        adjoint = F.grid_sample(adjoint, inv_phi, align_corners=True, mode='bilinear', padding_mode='zeros')
                
                adj_grads = adj_grads[::-1] # Reverse to match time steps 0, 1, 2
                adj_grad_tensor = torch.stack(adj_grads, dim=0) # [T, 1, dim, *spatial]
                
                # Fluid Gaussian Smoothing
                from .syn import separable_gaussian_filter
                sigma_list = [self.flow_sigma / sp for sp in curr_spacing]
                adj_grad_cl = adj_grad_tensor.squeeze(1).movedim(1, -1)
                adj_grad_cl_smooth = separable_gaussian_filter(adj_grad_cl, sigma=sigma_list, sigma_mode='voxel')
                adj_grad_tensor = adj_grad_cl_smooth.movedim(-1, 1).unsqueeze(1)
                
                # CFL Gradient Normalization (ITK-style) per-time-step or global
                max_g_voxel = torch.sqrt(torch.sum(adj_grad_tensor**2, dim=2)).max()
                
                if max_g_voxel > 1e-8:
                    update = (self.lr / max_g_voxel) * adj_grad_tensor
                    
                    if self.cfl_momentum > 0:
                        if momentum_buffer is None:
                            momentum_buffer = torch.zeros_like(self.v)
                        momentum_buffer.mul_(self.cfl_momentum).add_(update)
                        bias_corr = 1.0 - (self.cfl_momentum ** (epoch + 1))
                        corrected_buf = momentum_buffer / max(bias_corr, 1e-8)
                        self.v = self.v - corrected_buf
                    else:
                        self.v = self.v - update
                
                # Elastic Total Field Smoothing
                if self.total_sigma > 0:
                    elastic_sigma_list = [self.total_sigma / sp for sp in curr_spacing]
                    v_cl = self.v.squeeze(1).movedim(1, -1)
                    v_cl_smooth = separable_gaussian_filter(v_cl, sigma=elastic_sigma_list, sigma_mode='voxel')
                    self.v = v_cl_smooth.movedim(-1, 1).unsqueeze(1)
                
                if (epoch+1) % max(1, iters // 5) == 0:
                    print(f"  Epoch {epoch+1:03d} | Max Adj Grad: {max_g_voxel.item():.6f}")
                    
        # Final upsample to native resolution if needed
        if list(self.v.shape[3:]) != list(self.shape):
            self.v = self._resample_velocity(list(self.shape))
        return self.v

    def _resample_velocity(self, new_shape):
        """``self.v`` interpolated to ``new_shape`` with its voxel-unit components rescaled
        per axis by (n_new - 1) / (n_old - 1) (channels (x, y[, z]))."""
        old_shape = list(self.v.shape[3:])
        v = self.v.reshape(self.n_time_steps, self.dim, *old_shape)
        v = F.interpolate(v, size=list(new_shape), mode='bilinear' if self.dim == 2 else 'trilinear',
                          align_corners=True)
        ratio = [(n - 1) / max(o - 1, 1) for n, o in zip(new_shape, old_shape)][::-1]   # (x, y[, z])
        v = v * torch.tensor(ratio, device=v.device, dtype=v.dtype).view(1, self.dim, *([1] * self.dim))
        return v.reshape(self.n_time_steps, 1, self.dim, *new_shape)
        
def tvf_registration_adjoint(fixed, moving, initial_transform=None, flow_sigma=2.0, total_sigma=0.5, lr=0.5, levels=[4, 2, 1], reg_iterations=[100, 100, 20], device='cpu'):
    """
    Deprecated adjoint TVF registration wrapper (use ``syntx.tvf``).

    If ``initial_transform`` is given the moving image is first resampled into the fixed
    space with it (linear), then ``TVFRegistrationAdjoint`` is fitted. The forward map
    integrates the keyframes; the inverse integrates the negated keyframes in reverse order.
    Displacements are converted from normalised units to voxels ((size - 1) / 2) and then to
    physical vectors ``direction @ (voxels * spacing)``, and written as ANTs warp files in the
    system temp directory (returned to the caller, like every syntx registration).

    Parameters
    ----------
    fixed, moving : ants.ANTsImage
    initial_transform : str or list of str, optional
        ANTs transform file(s) applied to the moving image before registration and appended
        to the returned transform lists.
    flow_sigma, total_sigma, levels, reg_iterations
        Passed to ``TVFRegistrationAdjoint``.
    lr : float, default 0.5
        Maximum voxel step per iteration (the class default; was 50).
    device : str, default 'cpu'
        Torch device for the optimisation.

    Returns
    -------
    dict
        ``'warpedmovout'`` (moving image warped by ``fwdtransforms``), ``'fwdtransforms'``
        ([warp] + initial transforms), ``'invtransforms'`` (initial transforms + [inverse
        warp]), ``'whichtoinvert_inv'`` (True for each initial transform, False for the warp),
        ``'model'`` (the fitted ``TVFRegistrationAdjoint``).
    """
    from .syn import parse_ants_affine
    import tempfile
    from .transform import export_ants_displacement_field, export_ants_affine_transform
    
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        moving_affine = ants.apply_transforms(fixed, moving, init_tx_list, interpolator='linear')
    else:
        moving_affine = moving
        
    adj = TVFRegistrationAdjoint(fixed, moving_affine, flow_sigma=flow_sigma, total_sigma=total_sigma, lr=lr, device=device, levels=levels, reg_iterations=reg_iterations)
    v_final = adj.fit()
    v_list_fwd = [v_final[t] for t in range(adj.n_time_steps)]
    v_list_inv = [-v_final[t] for t in reversed(range(adj.n_time_steps))]
    
    # Forward and Inverse integration (Normalized grids)
    phi_history_fwd = integrate_forward(v_list_fwd, adj.shape)
    phi_history_inv = integrate_forward(v_list_inv, adj.shape)
    
    phi_fwd = phi_history_fwd[-1]
    phi_inv = phi_history_inv[-1]
    
    # normalised -> voxels -> physical (direction @ (voxels * spacing)), last axis (x, y[, z])
    grid = _identity_grid(adj.shape, phi_fwd.device)
    vox = 1.0 / _voxel_to_normalised(adj.shape, phi_fwd.device)
    M = torch.tensor(np.asarray(fixed.direction, dtype=float) @ np.diag(fixed.spacing),
                     device=phi_fwd.device, dtype=phi_fwd.dtype)
    disp_fwd = ((phi_fwd - grid) * vox) @ M.t()
    disp_inv = ((phi_inv - grid) * vox) @ M.t()

    # export_ants_displacement_field takes tensor-order components (dz, dy, dx)
    disp_fwd_np = disp_fwd.squeeze(0).flip(-1).cpu().numpy()
    disp_inv_np = disp_inv.squeeze(0).flip(-1).cpu().numpy()
    
    # Export physical displacement fields
    fwd_img = export_ants_displacement_field(disp_fwd_np, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)
    inv_img = export_ants_displacement_field(disp_inv_np, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)
    
    fwd_file = tempfile.NamedTemporaryFile(suffix='_tvf_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_tvf_inv_Warp.nii.gz', delete=False).name
    ants.image_write(fwd_img, fwd_file)
    ants.image_write(inv_img, inv_file)
    
    # Generate return dictionary
    fwd_transforms = [fwd_file]
    inv_transforms = [inv_file]
    if initial_transform is not None:
        fwd_transforms.extend(init_tx_list)
        inv_transforms = init_tx_list + inv_transforms
        whichtoinvert_inv = [True] * len(init_tx_list) + [False]
    else:
        whichtoinvert_inv = [False]
        
    warpedmovout = ants.apply_transforms(fixed, moving, fwd_transforms)
    
    ret_dict = {
        'warpedmovout': warpedmovout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'model': adj
    }
    
    return ret_dict
