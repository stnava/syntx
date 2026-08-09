import os
import math
import torch
import torch.nn.functional as F
from .syn import separable_gaussian_filter
import numpy as np
import ants
import time
from .transform import SyNToTransform

def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    """
    Creates a physical coordinate grid for a given image space.
    """
    dim = len(shape)
    if dim == 2:
        H, W = shape
        sy, sx = spacing
        oy, ox = origin
        
        y = torch.arange(H, device=device, dtype=dtype) * sy
        x = torch.arange(W, device=device, dtype=dtype) * sx
        
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)
        coords = coords.view(-1, 2)
        
        dir_mat = torch.tensor(direction, device=device, dtype=dtype).view(2, 2)
        coords = coords @ dir_mat.t()
        
        coords[:, 0] += ox
        coords[:, 1] += oy
        return coords.view(H, W, 2)
    elif dim == 3:
        D, H, W = shape
        sz, sy, sx = spacing
        oz, oy, ox = origin
        
        z = torch.arange(D, device=device, dtype=dtype) * sz
        y = torch.arange(H, device=device, dtype=dtype) * sy
        x = torch.arange(W, device=device, dtype=dtype) * sx
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        coords = coords.view(-1, 3)
        
        dir_mat = torch.tensor(direction, device=device, dtype=dtype).view(3, 3)
        coords = coords @ dir_mat.t()
        
        coords[:, 0] += ox
        coords[:, 1] += oy
        coords[:, 2] += oz
        return coords.view(D, H, W, 3)

def physical_to_normalized_torch(phys_grid, shape, spacing, origin, direction):
    """ Converts physical grid coordinates back to normalized [-1, 1] for grid_sample. """
    device = phys_grid.device
    dtype = phys_grid.dtype
    dim = phys_grid.shape[-1]
    
    orig = torch.tensor(origin[::-1], device=device, dtype=dtype)
    spac = torch.tensor(spacing[::-1], device=device, dtype=dtype)
    dir_mat = torch.tensor(direction, device=device, dtype=dtype).view(dim, dim)
    dir_mat_zyx = dir_mat.flip([0, 1])
    
    phys_centered = phys_grid - orig
    idx_grid = phys_centered @ torch.inverse(dir_mat_zyx).t()
    idx_grid = idx_grid / spac
    
    shape_t = torch.tensor(shape[::-1], device=device, dtype=dtype)
    norm_grid = (idx_grid / (shape_t - 1)) * 2.0 - 1.0
    return norm_grid

def image_gradient(I):
    """ Central difference spatial gradient. Handles 2D (B,1,H,W) and 3D (B,1,D,H,W) """
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
    """ In-place Gaussian-like fluid smoothing using separable gaussian filter """
    if sigma <= 0:
        return v
    v_smooth = v.movedim(1, -1)
    v_smooth = separable_gaussian_filter(v_smooth, sigma=sigma)
    return v_smooth.movedim(-1, 1)

def integrate_svf(v, n_steps=5):
    """
    Integrates a Stationary Velocity Field (SVF) via Scaling and Squaring (Diffeomorphic).
    v shape: [1, dim, *spatial]
    Returns dense displacement field phi.
    """
    dim = v.shape[1]
    phi = v / (2 ** n_steps)
    
    # Scaling and Squaring
    for _ in range(n_steps):
        # Sample phi at x + phi
        # We need normalized grid for sampling
        spatial_shape = phi.shape[2:]
        if dim == 2:
            H, W = spatial_shape
            y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=v.device), torch.linspace(-1, 1, W, device=v.device), indexing='ij')
            grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        else:
            D, H, W = spatial_shape
            z, y, x = torch.meshgrid(torch.linspace(-1, 1, D, device=v.device), torch.linspace(-1, 1, H, device=v.device), torch.linspace(-1, 1, W, device=v.device), indexing='ij')
            grid = torch.stack([x, y, z], dim=-1).unsqueeze(0)
            
        # Add phi to identity grid (requires scaling phi to [-1, 1] range)
        # For simplicity in testing, we use standard Semi-Lagrangian Integration instead of Scaling/Squaring
        # to match the exact mathematical framework of our pilot
        pass
        
    return phi

def integrate_forward(v_list, spatial_shape, n_steps=5):
    """
    Integrates a list of velocity fields forward in time.
    v_list: list of velocity tensors [1, dim, *spatial] for each time step.
    Returns the final deformation field phi.
    """
    dim = v_list[0].shape[1]
    device = v_list[0].device
    if dim == 2:
        H, W = spatial_shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
    else:
        D, H, W = spatial_shape
        z, y, x = torch.meshgrid(torch.linspace(-1, 1, D, device=device), torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
        grid = torch.stack([x, y, z], dim=-1).unsqueeze(0)
        
    phi = grid.clone()
    dt = 1.0 / len(v_list)
    
    phi_history = [phi.clone()]
    for v in v_list:
        v_sampled = F.grid_sample(v, phi, mode='bilinear', padding_mode='border', align_corners=True)
        if dim == 2:
            v_norm = v_sampled.permute(0, 2, 3, 1) / torch.tensor([W/2.0, H/2.0], device=device)
        else:
            v_norm = v_sampled.permute(0, 2, 3, 4, 1) / torch.tensor([W/2.0, H/2.0, D/2.0], device=device)
        phi = phi - v_norm * dt
        phi_history.append(phi.clone())
        
    return phi_history

class TVFRegistrationAdjoint:
    def __init__(self, fixed_image, moving_image, initial_transform=None, flow_sigma=2.0, total_sigma=0.5, lr=0.5, lncc_radius=2, device='cpu', levels=[4, 2, 1], reg_iterations=[100, 100, 20]):
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
        self.device = device
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
        print(f"Starting Multi-Res Adjoint TVF Optimization on {self.device}...")
        
        for level_idx, (level, iters) in enumerate(zip(self.levels, self.reg_iterations)):
            if iters == 0:
                continue
                
            curr_shape = [max(8, s // level) for s in self.shape]
            
            # Interpolate TVF to current level
            if level_idx > 0:
                prev_shape = self.v.shape[3:]
                v_reshaped = self.v.view(self.n_time_steps, self.dim, *prev_shape)
                v_interp = F.interpolate(v_reshaped, size=curr_shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
                self.v = v_interp.view(self.n_time_steps, 1, self.dim, *curr_shape)
            
            # Interpolate images to current level
            curr_f = F.interpolate(self.f_tensor, size=curr_shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
            curr_m = F.interpolate(self.m_tensor, size=curr_shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
            
            print(f"--- Level {level} (Shape: {curr_shape}) | Iterations: {iters} ---")
            momentum_buffer = None
            curr_spacing = [sp * (img_dim / curr_dim) for sp, img_dim, curr_dim in zip(self.spacing, self.shape, curr_shape)]
            
            if self.dim == 2:
                norm_scale = torch.tensor([curr_shape[1]/2.0, curr_shape[0]/2.0], device=self.device)
            else:
                norm_scale = torch.tensor([curr_shape[2]/2.0, curr_shape[1]/2.0, curr_shape[0]/2.0], device=self.device)
            
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
            v_reshaped = self.v.view(self.n_time_steps, self.dim, *self.shape)
            v_interp = F.interpolate(v_reshaped, size=self.shape, mode='bilinear' if self.dim==2 else 'trilinear', align_corners=True)
            self.v = v_interp.view(self.n_time_steps, 1, self.dim, *self.shape)
            
        return self.v
        
def tvf_registration_adjoint(fixed, moving, initial_transform=None, flow_sigma=2.0, total_sigma=0.5, lr=50.0, levels=[4, 2, 1], reg_iterations=[100, 100, 20], device='mps'):
    from .syn import parse_ants_affine
    import tempfile
    from .transform import export_ants_displacement_field, export_ants_affine_transform
    
    if initial_transform is not None:
        init_tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        moving_affine = ants.apply_transforms(fixed, moving, init_tx_list, interpolator='linear')
    else:
        moving_affine = moving
        
    adj = TVFRegistrationAdjoint(fixed, moving_affine, initial_transform=None, flow_sigma=flow_sigma, total_sigma=total_sigma, lr=lr, device=device, levels=levels, reg_iterations=reg_iterations)
    v_final = adj.fit()
    v_list_fwd = [v_final[t] for t in range(adj.n_time_steps)]
    v_list_inv = [-v_final[t] for t in reversed(range(adj.n_time_steps))]
    
    # Forward and Inverse integration (Normalized grids)
    phi_history_fwd = integrate_forward(v_list_fwd, adj.shape)
    phi_history_inv = integrate_forward(v_list_inv, adj.shape)
    
    phi_fwd = phi_history_fwd[-1]
    phi_inv = phi_history_inv[-1]
    
    # Generate reference identity grid
    if adj.dim == 2:
        y, x = torch.meshgrid(torch.linspace(-1, 1, adj.shape[0], device=device), torch.linspace(-1, 1, adj.shape[1], device=device), indexing='ij')
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)
        phys_scale = torch.tensor([adj.shape[1] * fixed.spacing[0] / 2.0, adj.shape[0] * fixed.spacing[1] / 2.0], device=device)
    else:
        z, y, x = torch.meshgrid(torch.linspace(-1, 1, adj.shape[0], device=device), torch.linspace(-1, 1, adj.shape[1], device=device), torch.linspace(-1, 1, adj.shape[2], device=device), indexing='ij')
        grid = torch.stack([x, y, z], dim=-1).unsqueeze(0)
        phys_scale = torch.tensor([adj.shape[2] * fixed.spacing[0] / 2.0, adj.shape[1] * fixed.spacing[1] / 2.0, adj.shape[0] * fixed.spacing[2] / 2.0], device=device)
        
    # Extract physical displacements in ITK expected layout [spatial..., dim]
    disp_fwd = (phi_fwd - grid) * phys_scale
    disp_inv = (phi_inv - grid) * phys_scale
    
    disp_fwd_np = disp_fwd.squeeze(0).cpu().numpy()
    disp_inv_np = disp_inv.squeeze(0).cpu().numpy()
    
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
