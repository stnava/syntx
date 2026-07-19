import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .transform import SyNToTransform

def get_rotation_matrix(omega, dim):
    """
    Computes a rotation matrix from a skew-symmetric Lie Algebra parameterization.
    For 2D, omega has 1 element. For 3D, omega has 3 elements.
    Safe for AD (avoids division by zero and NaNs at omega = 0 by avoiding
    non-differentiable norm(0)).
    """
    device = omega.device
    dtype = omega.dtype
    if dim == 2:
        theta = omega[0]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t])
        ])
    elif dim == 3:
        theta2 = torch.sum(omega**2)
        is_zero = theta2 < 1e-16
        safe_theta2 = torch.where(is_zero, torch.tensor(1e-16, device=device, dtype=dtype), theta2)
        theta = torch.sqrt(safe_theta2)
        
        safe_theta = torch.where(is_zero, torch.tensor(1.0, device=device, dtype=dtype), theta)
        omega_norm = omega / safe_theta
        
        K_raw = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega[2], omega[1]]),
            torch.stack([omega[2], torch.tensor(0.0, device=device, dtype=dtype), -omega[0]]),
            torch.stack([-omega[1], omega[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[2], omega_norm[1]]),
            torch.stack([omega_norm[2], torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[0]]),
            torch.stack([-omega_norm[1], omega_norm[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)
        R_small = I + K_raw
        return torch.where(is_zero, R_small, R)
    else:
        raise ValueError("Only 2D and 3D are supported.")

class TriPlanarVGG3DLoss(nn.Module):
    def __init__(self, dim=3, feature_layers=[4], num_slices=4, patch_size=32, num_patches=8, mode='lncc_3d', vgg_lncc_window_size=9):
        """
        Computes 3D Perceptual Loss supporting multiple local patch/metric configurations:
        - mode='patch_walk': Random-walk cluster of overlapping patches.
        - mode='patch_grid': Dense regular grid-based patch sampling.
        - mode='lncc': Feature-space Local Normalized Cross-Correlation (LNCC) on global slice VGG feature maps.
        - mode='lncc_3d': 3D Feature-Space LNCC (5x5x5 window) on reconstructed deep feature volumes.
        """
        super().__init__()
        import torchvision.models as models
        self.dim = dim
        self.num_slices = num_slices
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mode = mode
        self.vgg_lncc_window_size = vgg_lncc_window_size
        
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(feature_layers) + 1)])
        
        for m in self.vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
                
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.feature_layers = feature_layers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_nd, target_nd):
        B = input_nd.shape[0]
        device = input_nd.device
        dtype = input_nd.dtype
        
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.std = self.std.to(device=device, dtype=dtype)
        
        if self.dim == 3 and self.mode == 'lncc_3d':
            D, H, W = input_nd.shape[2:]
            
            # Helper to reconstruct 3D feature volume along all three axes
            def reconstruct_3d_features(x):
                # 1. Axial
                slices_ax = []
                for z in range(1, D - 1):
                    slices_ax.append(x[:, 0, z-1:z+2])
                batch_ax = (torch.cat(slices_ax, dim=0) - self.mean) / self.std
                
                # 2. Coronal
                slices_co = []
                for y in range(1, H - 1):
                    slices_co.append(x[:, 0, :, y-1:y+2, :].movedim(2, 1))
                batch_co = (torch.cat(slices_co, dim=0) - self.mean) / self.std
                
                # 3. Sagittal
                slices_sa = []
                for xi in range(1, W - 1):
                    slices_sa.append(x[:, 0, :, :, xi-1:xi+2].movedim(3, 1))
                batch_sa = (torch.cat(slices_sa, dim=0) - self.mean) / self.std
                
                # Run through VGG
                feat_ax = self.vgg(batch_ax)
                feat_co = self.vgg(batch_co)
                feat_sa = self.vgg(batch_sa)
                
                # Permute back to standard (B, C, Depth, Height, Width) ordering
                vol_ax = feat_ax.view(D-2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4)
                vol_co = feat_co.view(H-2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4)
                vol_sa = feat_sa.view(W-2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0)
                
                return vol_ax, vol_co, vol_sa
                
            vol_in_ax, vol_in_co, vol_in_sa = reconstruct_3d_features(input_nd)
            vol_tg_ax, vol_tg_co, vol_tg_sa = reconstruct_3d_features(target_nd)
            
            # Sum the LNCC losses across the three orthogonal 3D feature spaces
            loss_ax = local_ncc_loss_nd(vol_in_ax, vol_tg_ax, window_size=5)
            loss_co = local_ncc_loss_nd(vol_in_co, vol_tg_co, window_size=5)
            loss_sa = local_ncc_loss_nd(vol_in_sa, vol_tg_sa, window_size=5)
            
            return loss_ax + loss_co + loss_sa
            
        elif self.dim == 3:
            D, H, W = input_nd.shape[2:]
            
            if self.mode == 'lncc':
                # Option 2: Feature-Space LNCC
                # Extract global slices across Axial, Coronal, and Sagittal directions
                z_indices = torch.linspace(D // 4, 3 * D // 4, self.num_slices, dtype=torch.long, device=device)
                y_indices = torch.linspace(H // 4, 3 * H // 4, self.num_slices, dtype=torch.long, device=device)
                x_indices = torch.linspace(W // 4, 3 * W // 4, self.num_slices, dtype=torch.long, device=device)
                
                target_size = max(D, H, W)
                slices_in = []
                slices_tg = []
                
                # Axial
                for z in z_indices:
                    slices_in.append(F.interpolate(input_nd[:, 0, z-1:z+2], size=(target_size, target_size), mode='bilinear', align_corners=True))
                    slices_tg.append(F.interpolate(target_nd[:, 0, z-1:z+2], size=(target_size, target_size), mode='bilinear', align_corners=True))
                # Coronal
                for y in y_indices:
                    slices_in.append(F.interpolate(input_nd[:, 0, :, y-1:y+2, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                    slices_tg.append(F.interpolate(target_nd[:, 0, :, y-1:y+2, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                # Sagittal
                for xi in x_indices:
                    slices_in.append(F.interpolate(input_nd[:, 0, :, :, xi-1:xi+2].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                    slices_tg.append(F.interpolate(target_nd[:, 0, :, :, xi-1:xi+2].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                    
                input_rgb = (torch.cat(slices_in, dim=0) - self.mean) / self.std
                target_rgb = (torch.cat(slices_tg, dim=0) - self.mean) / self.std
                
            else:
                # Option 1 or existing: Patch extraction
                # Compute effective patch size for each dimension to handle coarse scales
                P_z = min(self.patch_size, D)
                P_y = min(self.patch_size, H)
                P_x = min(self.patch_size, W)
                P_target = max(P_z, P_y, P_x)
                
                S_z = P_z // 2
                S_y = P_y // 2
                S_x = P_x // 2
                
                if self.mode == 'patch_grid':
                    # Option 1: Dense grid-based patch sampling
                    z_grid = torch.arange(P_z // 2, max(P_z // 2 + 1, D - P_z // 2), max(1, S_z), device=device)
                    y_grid = torch.arange(P_y // 2, max(P_y // 2 + 1, H - P_y // 2), max(1, S_y), device=device)
                    x_grid = torch.arange(P_x // 2, max(P_x // 2 + 1, W - P_x // 2), max(1, S_x), device=device)
                    
                    grid_centers = torch.stack(torch.meshgrid(z_grid, y_grid, x_grid, indexing='ij'), dim=-1).reshape(-1, 3)
                    
                    if grid_centers.shape[0] > self.num_patches:
                        indices = torch.randperm(grid_centers.shape[0], device=device)[:self.num_patches]
                        centers = grid_centers[indices]
                    else:
                        centers = grid_centers
                        
                    z_centers = centers[:, 0]
                    y_centers = centers[:, 1]
                    x_centers = centers[:, 2]
                    num_sampled_patches = centers.shape[0]
                else:
                    # mode='patch_walk'
                    zc = torch.randint(P_z // 2, max(P_z // 2 + 1, D - P_z // 2), (1,), device=device)
                    yc = torch.randint(P_y // 2, max(P_y // 2 + 1, H - P_y // 2), (1,), device=device)
                    xc = torch.randint(P_x // 2, max(P_x // 2 + 1, W - P_x // 2), (1,), device=device)
                    
                    z_centers = [zc]
                    y_centers = [yc]
                    x_centers = [xc]
                    
                    for k in range(self.num_patches - 1):
                        dz = torch.randint(-S_z, S_z + 1, (1,), device=device) if S_z > 0 else torch.zeros(1, dtype=torch.long, device=device)
                        dy = torch.randint(-S_y, S_y + 1, (1,), device=device) if S_y > 0 else torch.zeros(1, dtype=torch.long, device=device)
                        dx = torch.randint(-S_x, S_x + 1, (1,), device=device) if S_x > 0 else torch.zeros(1, dtype=torch.long, device=device)
                        
                        zc_new = torch.clamp(z_centers[-1] + dz, P_z // 2, max(P_z // 2 + 1, D - P_z // 2))
                        yc_new = torch.clamp(y_centers[-1] + dy, P_y // 2, max(P_y // 2 + 1, H - P_y // 2))
                        xc_new = torch.clamp(x_centers[-1] + dx, P_x // 2, max(P_x // 2 + 1, W - P_x // 2))
                        
                        z_centers.append(zc_new)
                        y_centers.append(yc_new)
                        x_centers.append(xc_new)
                        
                    z_centers = torch.cat(z_centers)
                    y_centers = torch.cat(y_centers)
                    x_centers = torch.cat(x_centers)
                    num_sampled_patches = self.num_patches
                
                # Helper to extract slices from specific centers
                def extract_slices(x):
                    slices = []
                    for k in range(num_sampled_patches):
                        zc, yc, xc = z_centers[k], y_centers[k], x_centers[k]
                        # Extract 3D patch: (B, 1, P_z, P_y, P_x)
                        patch = x[:, :, zc - P_z//2 : zc + P_z//2, yc - P_y//2 : yc + P_y//2, xc - P_x//2 : xc + P_x//2]
                        
                        z_indices = torch.linspace(P_z // 4, 3 * P_z // 4, self.num_slices, dtype=torch.long, device=device)
                        y_indices = torch.linspace(P_y // 4, 3 * P_y // 4, self.num_slices, dtype=torch.long, device=device)
                        x_indices = torch.linspace(P_x // 4, 3 * P_x // 4, self.num_slices, dtype=torch.long, device=device)
                        
                        # Axial
                        for z in z_indices:
                            triplet = patch[:, 0, z-1:z+2]
                            triplet_res = F.interpolate(triplet, size=(P_target, P_target), mode='bilinear', align_corners=True)
                            slices.append(triplet_res)
                        # Coronal
                        for y in y_indices:
                            triplet = patch[:, 0, :, y-1:y+2, :].movedim(2, 1)
                            triplet_res = F.interpolate(triplet, size=(P_target, P_target), mode='bilinear', align_corners=True)
                            slices.append(triplet_res)
                        # Sagittal
                        for xi in x_indices:
                            triplet = patch[:, 0, :, :, xi-1:xi+2].movedim(3, 1)
                            triplet_res = F.interpolate(triplet, size=(P_target, P_target), mode='bilinear', align_corners=True)
                            slices.append(triplet_res)
                    rgb = torch.cat(slices, dim=0)
                    return (rgb - self.mean) / self.std
                    
                input_rgb = extract_slices(input_nd)
                target_rgb = extract_slices(target_nd)
        else:
            # 2D case: repeat channels and normalize
            input_rgb = (input_nd.repeat(1, 3, 1, 1) - self.mean) / self.std
            target_rgb = (target_nd.repeat(1, 3, 1, 1) - self.mean) / self.std
            
        loss = 0.0
        x_in = input_rgb
        x_tg = target_rgb
        
        for i, layer in enumerate(self.vgg):
            x_in = layer(x_in)
            x_tg = layer(x_tg)
            if i in self.feature_layers:
                if self.mode == 'lncc':
                    loss += local_ncc_loss_nd(x_in, x_tg, window_size=self.vgg_lncc_window_size)
                elif self.mode == 'mse':
                    loss += F.mse_loss(x_in, x_tg)
                else:
                    loss += F.l1_loss(x_in, x_tg)
                    
        return loss

class HierarchicalAffine(nn.Module):
    def __init__(self, dim=3, transform_type='Affine'):
        """
        Supports 'Translation', 'Rigid', 'Similarity', and 'Affine' transforms.
        Rotations are parameterized on the Lie Algebra SO(d) to prevent gimbal lock.
        """
        super().__init__()
        self.dim = dim
        self.type = transform_type
        
        # Translation
        self.translation = nn.Parameter(torch.zeros(dim))
        
        # Rotation (Lie Algebra SO(d))
        num_rot = dim * (dim - 1) // 2
        self.omega = nn.Parameter(torch.zeros(num_rot))
        
        # Scale (Similarity)
        if transform_type in ['Similarity', 'Affine']:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
            
        # Shear/Anisotropic Scale
        if transform_type == 'Affine':
            self.anisotropic_scale = nn.Parameter(torch.ones(dim))
            self.shear = nn.Parameter(torch.zeros(num_rot))
        else:
            self.register_buffer('anisotropic_scale', torch.ones(dim))
            self.register_buffer('shear', torch.zeros(num_rot))
            
        self.register_buffer('T_init', None)

    def get_matrix(self):
        R = get_rotation_matrix(self.omega, self.dim)
        
        if self.type == 'Affine':
            S = torch.diag(self.anisotropic_scale * self.scale)
            Sh = torch.eye(self.dim, device=self.shear.device, dtype=self.shear.dtype)
            triu_indices = torch.triu_indices(self.dim, self.dim, offset=1)
            Sh[triu_indices[0], triu_indices[1]] = self.shear
            A = R @ S @ Sh
        else:
            A = R * self.scale
            
        T = torch.eye(self.dim + 1, device=self.translation.device, dtype=self.translation.dtype)
        T[:self.dim, :self.dim] = A
        T[:self.dim, self.dim] = self.translation
        
        if hasattr(self, 'T_init') and self.T_init is not None:
            return T @ self.T_init
        return T

    def get_affine_grid_matrix(self):
        T = self.get_matrix()
        return T[:self.dim, :self.dim + 1]


def separable_gaussian_filter(grid: torch.Tensor, sigma: float, spacing=None) -> torch.Tensor:
    """
    Applies separable Gaussian filtering along each spatial dimension.
    Input format: (B, *spatial, dim) - channel-last representation of coordinates.
    """
    if sigma <= 0.0:
        return grid
        
    device = grid.device
    dtype = grid.dtype
    shape = grid.shape
    B = shape[0]
    dim = shape[-1]
    spatial_shape = shape[1:-1]
    num_spatial = len(spatial_shape)
    
    if spacing is not None:
        spacing_rev = tuple(reversed(spacing))
        sigma_list = [sigma / sp for sp in spacing_rev]
    else:
        sigma_list = [sigma] * num_spatial
        
    v = torch.movedim(grid, -1, 1)
    
    for i in range(num_spatial):
        sig = sigma_list[i]
        if sig <= 0.0:
            continue
            
        kernel_size = max(3, int(2 * math.ceil(2 * sig) + 1))
        x = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x**2 / (2.0 * sig**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Shape for F.conv1d: [out_channels=1, in_channels=1, kernel_size]
        kernel = kernel_1d.view(1, 1, kernel_size).clone()
        pad_size = kernel_size // 2
        
        # We want to convolve along spatial dimension `i`.
        # In `v` (shape [B, C, spatial...]), spatial dimension `i` is at index `i + 2`.
        target_dim = i + 2
        
        # 1. Permute target_dim to the last position
        # List of dimensions: [0, 1, ..., ndim-1]
        dims = list(range(v.ndim))
        dims[-1], dims[target_dim] = dims[target_dim], dims[-1]
        v_permuted = v.permute(*dims).contiguous()
        
        # 2. Reshape to [B * C * other_spatial..., 1, target_spatial_size]
        last_dim_size = v_permuted.shape[-1]
        v_reshaped = v_permuted.view(-1, 1, last_dim_size)
        
        # 3. Apply F.conv1d with replicate padding (Neumann boundary conditions) to avoid boundary shocks
        v_padded = F.pad(v_reshaped, (pad_size, pad_size), mode='replicate')
        v_convolved = F.conv1d(v_padded, kernel)
        
        # 4. Reshape back and permute back
        v_convolved = v_convolved.view(v_permuted.shape)
        v = v_convolved.permute(*dims).contiguous()
        
    return torch.movedim(v, 1, -1).contiguous()


def compose_grids(grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
    """
    Composes two coordinate grids: grid1 ∘ grid2
    grid1: (B, *spatial, dim)
    grid2: (B, *spatial, dim)
    """
    grid1_cf = torch.movedim(grid1, -1, 1)
    composed_cf = F.grid_sample(grid1_cf, grid2, mode='bilinear', padding_mode='border', align_corners=True)
    return torch.movedim(composed_cf, 1, -1)


def get_boundary_mask(spatial, device, dtype):
    """
    Constructs a boundary mask where boundary voxels are 0 and interior voxels are 1.
    """
    boundary_mask = torch.ones((1, *spatial, 1), device=device, dtype=dtype)
    for i in range(len(spatial)):
        slices = [slice(None)] * boundary_mask.ndim
        slices[i + 1] = 0
        boundary_mask[tuple(slices)] = 0
        slices[i + 1] = -1
        boundary_mask[tuple(slices)] = 0
    return boundary_mask
def _get_physical_grid_torch_yfirst(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    # 'ij' indexing yields (dim0, dim1, ...) = (y, x) or (z, y, x) matching
    # the reversed spacing/origin/direction this function receives
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    idxs = torch.stack(meshgrid, dim=-1)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)

def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _get_physical_grid_torch_yfirst(shape, spacing_rev, origin_rev, direction_rev, device, dtype)

def _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing, origin, direction):
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)
    
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)
    
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    inv_direction_t = torch.inverse(direction_t.t())
    rotated = diff @ inv_direction_t
    voxel_coords = rotated / spacing_t
    
    shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    # Flip from internal YX order to grid_sample's expected XY order
    norm_coords = torch.flip(norm_coords, dims=[-1])
    return norm_coords.view(phys_coords.shape)

def physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction):
    # target_shape is in tensor order (Z, Y, X). _yfirst expects all params in Z-first order.
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing_rev, origin_rev, direction_rev)

def _grid_to_physical_affine_torch_yfirst(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
    dim = len(fixed_shape)
    device = T_grid.device
    dtype = T_grid.dtype
    
    Nx = torch.tensor(fixed_shape, device=device, dtype=dtype)
    Ny = torch.tensor(moving_shape, device=device, dtype=dtype)
    Sx = torch.tensor(fixed_spacing, device=device, dtype=dtype)
    Sy = torch.tensor(moving_spacing, device=device, dtype=dtype)
    Ox = torch.tensor(fixed_origin, device=device, dtype=dtype)
    Oy = torch.tensor(moving_origin, device=device, dtype=dtype)
    Dx = torch.tensor(fixed_direction, device=device, dtype=dtype)
    Dy = torch.tensor(moving_direction, device=device, dtype=dtype)
    
    Kx = torch.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = torch.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0
    
    Kx_inv = torch.inverse(Kx)
    Sx_inv = torch.inverse(torch.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.t()
    bx = - Kx_inv @ Sx_inv @ Dx.t() @ Ox - Kx_inv @ Cx
    
    Vy = Dy @ torch.diag(Sy) @ Ky
    cy = Dy @ torch.diag(Sy) @ Cy + Oy
    
    A_grid = T_grid[:dim, :dim]
    t_grid = T_grid[:dim, dim]
    
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy
    return M_phys, t_phys

def grid_to_physical_affine_torch(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
    dim = len(fixed_shape)
    # T_grid operates in grid_sample's XY order; permute to YX for _yfirst
    perm = list(range(dim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D
    T_yx = T_grid.clone()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]
    fs_rev = tuple(reversed(fixed_spacing))
    fo_rev = tuple(reversed(fixed_origin))
    fd_rev = np.asarray(fixed_direction)[::-1, ::-1].copy()
    ms_rev = tuple(reversed(moving_spacing))
    mo_rev = tuple(reversed(moving_origin))
    md_rev = np.asarray(moving_direction)[::-1, ::-1].copy()
    return _grid_to_physical_affine_torch_yfirst(T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev)

def physical_to_grid_affine(M_phys, t_phys, fixed_img, moving_img):
    import numpy as np
    dim = fixed_img.dimension
    Nx = np.array(fixed_img.shape)
    Ny = np.array(moving_img.shape)
    Sx = np.array(fixed_img.spacing)
    Sy = np.array(moving_img.spacing)
    Ox = np.array(fixed_img.origin)
    Oy = np.array(moving_img.origin)
    Dx = np.array(fixed_img.direction)
    Dy = np.array(moving_img.direction)
    
    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0
    
    Wx_inv = Dx @ np.diag(Sx) @ Kx
    bx = - np.linalg.inv(Kx) @ np.linalg.inv(np.diag(Sx)) @ Dx.T @ Ox - np.linalg.inv(Kx) @ Cx
    
    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
    Vy_inv = np.linalg.inv(Vy)
    
    A_grid = Vy_inv @ M_phys @ Wx_inv
    t_grid = Vy_inv @ (t_phys - cy) - A_grid @ bx
    
    T_grid = np.eye(dim + 1, dtype=np.float32)
    T_grid[:dim, :dim] = A_grid
    T_grid[:dim, dim] = t_grid
    
    perm = list(range(dim - 1, -1, -1))
    T_xyz = T_grid.copy()
    T_xyz[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_xyz[:dim, dim] = T_grid[:dim, dim][perm]
    return T_xyz

def physical_to_normalized_torch_cached(phys_coords, shape_t, spacing_t, origin_t, direction_t):
    dim = phys_coords.shape[-1]
    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    rotated = diff @ direction_t
    voxel_coords = rotated / spacing_t
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    norm_coords_reversed = torch.flip(norm_coords, dims=[-1])
    return norm_coords_reversed.view(phys_coords.shape)

def prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fixed_spacing, moving_spacing,
    M_phys, t_phys, initial_grid_level
):
    phi_l2r_phys = X_phys + warp_l2r
    coords_norm = physical_to_normalized_torch_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    I_mid = F.grid_sample(I_curr, coords_norm, padding_mode='border', align_corners=True)
    
    phi_r2l_phys = X_phys + warp_r2l
    y_phys = phi_r2l_phys @ M_phys.t() + t_phys
    if initial_grid_level is not None:
        y_norm_fixed = physical_to_normalized_torch_cached(
            y_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
        )
        y_norm = compose_grids(initial_grid_level, y_norm_fixed)
    else:
        y_norm = physical_to_normalized_torch_cached(
            y_phys, moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t
        )
        
    J_mid = F.grid_sample(J_curr, y_norm, padding_mode='border', align_corners=True)
    
    grad_I_curr = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=tuple(reversed(fixed_spacing))).squeeze(-2)
    grad_J_curr = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=tuple(reversed(moving_spacing))).squeeze(-2)
    
    grad_I_mid_sampled = F.grid_sample(grad_I_curr.movedim(-1, 1), coords_norm, padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
    grad_I_mid_sampled = torch.matmul(grad_I_mid_sampled, fixed_direction_t.t())
    
    grad_J_mid_sampled = F.grid_sample(grad_J_curr.movedim(-1, 1), y_norm, padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
    grad_J_mid_sampled = torch.matmul(grad_J_mid_sampled, moving_direction_t.t())
    grad_J_mid_sampled = torch.matmul(grad_J_mid_sampled, M_phys)
    
    dim = coords_norm.shape[-1]
    mask_I = (coords_norm[..., 0] >= -1.0) & (coords_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_I = mask_I & (coords_norm[..., d] >= -1.0) & (coords_norm[..., d] <= 1.0)
        
    mask_J = (y_norm[..., 0] >= -1.0) & (y_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_J = mask_J & (y_norm[..., d] >= -1.0) & (y_norm[..., d] <= 1.0)
        
    in_bounds_mask = (mask_I & mask_J).unsqueeze(1).to(dtype=I_mid.dtype)
    
    return I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask

def _spatial_jacobian_nd(field: torch.Tensor, physical_spacing=None) -> torch.Tensor:
    """Compute the spatial Jacobian of an N-D vector field via central differences.
    
    field: (B, *spatial, d) vector field
    Returns: (B, *spatial, d, d) Jacobian tensor J[..., i, j] = ∂field_i / ∂x_j
    """
    dim = field.shape[-1]
    spatial = field.shape[1:-1]
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (s - 1) for s in spatial]
    
    # torch.gradient returns a list of gradients, one per spatial dimension (ij order)
    grads = torch.gradient(field, spacing=spacings, dim=list(range(1, len(spatial) + 1)))
    
    # Keep in internal (y, x) or (z, y, x) ordering convention
    return torch.stack(grads, dim=-1)  # (B, *spatial, d, d)


def update_inverse_field_nd(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor, 
    steps: int = 20,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    method: str = 'fixed_point',
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing = None,
    origin = None,
    direction = None
) -> torch.Tensor:
    """
    Dimension-agnostic fixed-point inversion of a displacement field.
    
    W_disp: (B, *spatial, d) — forward displacement field
    W_inv_disp: (B, *spatial, d) — current inverse estimate
    steps: max iterations (ITK default: 20)
    method: 'fixed_point' (ITK standard)
    """
    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    if spacing is not None and origin is not None and direction is not None:
        X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        direction_rev = np.asarray(direction)[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        
        for iteration in range(steps):
            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            
            forward_at_inv_cf = F.grid_sample(torch.movedim(W_disp, -1, 1), coords_norm, padding_mode='border', align_corners=True)
            forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)
            
            error = W_inv_disp + forward_at_inv
            error_voxel = error / spacing_t
            scaled_norm = torch.sqrt(torch.sum(error_voxel**2, dim=-1, keepdim=True))
            
            max_error = scaled_norm.max()
            if max_error < mean_error_threshold:
                break
                
            epsilon = 0.75 if iteration == 0 else 0.5
            update = -error
            
            update_voxel = update / spacing_t
            update_norm = torch.sqrt(torch.sum(update_voxel**2, dim=-1, keepdim=True)) + 1e-10
            clip_threshold = max_error_threshold  # ITK clips at max_displacement to stabilize iterations
            clip_mask = update_norm > clip_threshold
            clip_scale = torch.where(clip_mask, clip_threshold / update_norm, torch.ones_like(update_norm))
            update = update * clip_scale
            
            W_inv_disp = W_inv_disp + epsilon * update
            
            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma, spacing=spacing)
            
            W_inv_disp = W_inv_disp * boundary_mask
            
        return W_inv_disp
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
        for iteration in range(steps):
            coords = identity + W_inv_disp
            forward_at_inv_cf = F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True)
            forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)
            error = W_inv_disp + forward_at_inv
            error_voxel = error * voxel_scale
            scaled_norm = torch.sqrt(torch.sum(error_voxel**2, dim=-1, keepdim=True))
            max_error = scaled_norm.max()
            if max_error < mean_error_threshold:
                break
            epsilon = 0.75 if iteration == 0 else 0.5
            update = -error
            update_voxel = update * voxel_scale
            update_norm = torch.sqrt(torch.sum(update_voxel**2, dim=-1, keepdim=True)) + 1e-10
            clip_threshold = max_error  # ITK clips at max_displacement, not epsilon*max_displacement
            clip_mask = update_norm > clip_threshold
            clip_scale = torch.where(clip_mask, clip_threshold / update_norm, torch.ones_like(update_norm))
            update = update * clip_scale
            W_inv_disp = W_inv_disp + epsilon * update
            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma)
        return W_inv_disp


def local_ncc_loss_nd(I, J, mask=None, window_size=9):
    """
    Computes Local Normalized Cross Correlation (LNCC) loss between N-D images I and J.
    I, J: (B, C, *spatial) where C=1
    """
    device = I.device
    dim = I.dim() - 2
    
    # Adapt window size dynamically if input is smaller than kernel
    min_spatial = min(I.shape[2:])
    if window_size > min_spatial:
        window_size = min_spatial
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)
            
    pad = window_size // 2
    
    if dim == 2:
        pool_fn = F.avg_pool2d
    elif dim == 3:
        pool_fn = F.avg_pool3d
    else:
        raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")
        
    def box_filter(x):
        return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)
        
    I_mean = box_filter(I)
    J_mean = box_filter(J)
    
    I_var = box_filter((I - I_mean)**2)
    J_var = box_filter((J - J_mean)**2)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))
    
    valid_mask = (I_var > 1e-8) & (J_var > 1e-8)
    
    safe_I_var = torch.clamp(I_var, min=1e-8)
    safe_J_var = torch.clamp(J_var, min=1e-8)
    
    cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-8)
    cc = torch.where(valid_mask, cc_raw, torch.zeros_like(cc_raw))
    
    if mask is not None:
        active_mask = (mask > 0.5) & valid_mask
    else:
        active_mask = valid_mask
        
    active_mask_float = active_mask.to(dtype=I.dtype)
    return -torch.sum(cc * active_mask_float) / (torch.sum(active_mask_float) + 1e-8)


def b_spline_3(x):
    """3rd-order B-spline kernel for Parzen windowing."""
    abs_x = torch.abs(x)
    y1 = (2.0/3.0) - abs_x**2 + 0.5 * abs_x**3
    y2 = (1.0/6.0) * (2.0 - abs_x)**3
    return torch.where(abs_x < 1.0, y1, torch.where(abs_x < 2.0, y2, 0.0))


def mattes_mi_loss_core(I, J, mask=None, num_bins=32, min_val=-1.0, max_val=1.0, sampling_percentage=None):
    """
    Differentiable Mattes Mutual Information (Parzen window using 3rd-order B-spline).
    Returns Negative Mutual Information (for minimization).
    """
    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()
        
    if sampling_percentage is not None and sampling_percentage < 1.0:
        stride = max(1, int(1.0 / sampling_percentage))
        x = x[::stride]
        y = y[::stride]
        
    if x.numel() == 0:
        return torch.tensor(0.0, device=I.device, requires_grad=True)
        
    x = torch.clamp(x, min_val, max_val)
    y = torch.clamp(y, min_val, max_val)
    
    sigma = (max_val - min_val) / (num_bins - 1)
    bins = torch.linspace(min_val, max_val, num_bins, device=I.device).unsqueeze(0)
    
    u_x = (x.view(-1, 1) - bins) / sigma
    u_y = (y.view(-1, 1) - bins) / sigma
    
    w_x = b_spline_3(u_x)
    w_y = b_spline_3(u_y)
    
    joint_hist = torch.matmul(w_x.t(), w_y)
    
    pxy = joint_hist / (joint_hist.sum() + 1e-8)
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    
    ratio = pxy / (px * py + 1e-8)
    mi = torch.sum(pxy * torch.log(ratio + 1e-8))
    
    return -mi


def mattes_mi_loss_nd(I, J, mask=None, num_bins=32, sampling_percentage=None):
    """
    N-dimensional Mattes Mutual Information loss wrapper.
    Scale images to [-1, 1] internally.
    """
    min_i, max_i = I.min().detach(), I.max().detach()
    min_j, max_j = J.min().detach(), J.max().detach()
    
    I_scaled = (I - min_i) / (max_i - min_i + 1e-8)
    J_scaled = (J - min_j) / (max_j - min_j + 1e-8)
    
    I_scaled = I_scaled * 2.0 - 1.0
    J_scaled = J_scaled * 2.0 - 1.0
    
    return mattes_mi_loss_core(I_scaled, J_scaled, mask, num_bins, min_val=-1.0, max_val=1.0, sampling_percentage=sampling_percentage)


def compute_jacobian_determinant_nd(warp_field: torch.Tensor, physical_spacing=None) -> torch.Tensor:
    """
    Computes the Jacobian determinant of a warp field (displacement or deformation).
    warp_field: (B, *spatial, dim) - displacement field (normalized or physical coordinates)
    Returns: (B, *spatial) - Jacobian determinant values
    """
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype
    
    is_physical = getattr(warp_field, 'is_physical', False)
    
    if is_physical:
        if physical_spacing is not None:
            spacings = list(physical_spacing)
        else:
            spacings = [1.0] * dim
        grads = torch.gradient(warp_field, spacing=spacings, dim=list(range(1, dim + 1)))
        
        if dim == 2:
            j00 = 1.0 + grads[1][..., 0]
            j01 = grads[0][..., 0]
            j10 = grads[1][..., 1]
            j11 = 1.0 + grads[0][..., 1]
            return j00 * j11 - j01 * j10
        elif dim == 3:
            j00 = 1.0 + grads[2][..., 0]
            j01 = grads[1][..., 0]
            j02 = grads[0][..., 0]
            
            j10 = grads[2][..., 1]
            j11 = 1.0 + grads[1][..., 1]
            j12 = grads[0][..., 1]
            
            j20 = grads[2][..., 2]
            j21 = grads[1][..., 2]
            j22 = 1.0 + grads[0][..., 2]
            
            return j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
        else:
            raise ValueError("Only 2D and 3D are supported.")
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(warp_field.shape[0], *([-1] * (dim + 1)))
        
        phi = identity + warp_field
        if physical_spacing is not None:
            spacings = list(physical_spacing)
        else:
            spacings = [2.0 / (size - 1) for size in spatial]
        grads = torch.gradient(phi, spacing=spacings, dim=list(range(1, dim + 1)))
        
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


def compute_physical_jacobian_determinant(
    warp_field: torch.Tensor, 
    direction: torch.Tensor, 
    spacing: torch.Tensor
) -> torch.Tensor:
    """
    Computes the physical Jacobian determinant of a warp field.
    
    Parameters:
    - warp_field: (B, *spatial, dim) displacement field (normalized or physical)
    - direction: (dim, dim) physical direction matrix D
    - spacing: (dim,) voxel spacing S (in mm)
    
    Returns:
    - jac_det_phys: (B, *spatial) physical Jacobian determinant map
    """
    is_physical = getattr(warp_field, 'is_physical', False)
    if is_physical:
        return compute_jacobian_determinant_nd(warp_field, physical_spacing=spacing)
        
    device = warp_field.device
    dtype = warp_field.dtype
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    
    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction, device=device, dtype=dtype)
    else:
        direction = direction.to(device=device, dtype=dtype)
        
    if not isinstance(spacing, torch.Tensor):
        spacing = torch.tensor(spacing, device=device, dtype=dtype)
    else:
        spacing = spacing.to(device=device, dtype=dtype)
        
    # 1. Compute J_voxel using spatial gradients with normalized spacing
    normalized_spacings = [2.0 / (s - 1) for s in spatial]
    grads = torch.gradient(warp_field, spacing=normalized_spacings, dim=list(range(1, dim + 1)))
    # Reverse gradient list to align with (x, y, [z]) component convention
    J_voxel = torch.stack(list(reversed(grads)), dim=-1)  # (B, *spatial, dim, dim)
    
    # 2. Construct voxel-to-physical matrices M and M_inv
    # M = D @ diag(S) -> column-wise scaling
    M = direction * spacing.unsqueeze(0)  # (dim, dim)
    # M_inv = diag(1/S) @ D^T -> row-wise scaling
    M_inv = direction.t() * (1.0 / spacing).unsqueeze(1)  # (dim, dim)
    
    # 3. Compute similarity transform J_phys = M @ J_voxel @ M_inv
    J_phys = torch.einsum('ij,b...jk,kl->b...il', M, J_voxel, M_inv)
    
    # 4. Compute deformation gradient F = J_phys + I
    F = J_phys + torch.eye(dim, device=device, dtype=dtype)
    
    # 5. Compute determinant of F analytically to avoid MPS batch LU decomposition deadlocks
    if dim == 2:
        a = F[..., 0, 0]
        b = F[..., 0, 1]
        c = F[..., 1, 0]
        d = F[..., 1, 1]
        jac_det_phys = a * d - b * c
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
        jac_det_phys = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    else:
        jac_det_phys = torch.linalg.det(F)
        
    return jac_det_phys


class SyNTo(nn.Module):
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, origin=None, direction=None, fluid_sigma=3.0, elastic_sigma=0.0, transform_type='Affine', inverse_method='fixed_point', inverse_steps=20):
        """
        Generalized Symmetric Normalization (SyN) in PyTorch.
        Includes hierarchical affine pre-alignment and dense symmetric velocity/displacement fields.
        
        Parameters
        ----------
        spacing : tuple or None
            Physical voxel spacing (in ITK/ANTs axis order).
        origin : tuple or None
            Physical origin.
        direction : array-like or None
            Direction cosine matrix (dim x dim). Defaults to identity if None.
            Follows ITK convention: maps voxel axes to physical axes.
        """
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape
        self.spacing = spacing
        self.origin = origin if origin is not None else [0.0] * dim
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.inverse_method = inverse_method
        self.inverse_steps = inverse_steps
        
        # Direction cosine matrix (ITK standard: identity if not specified)
        if direction is not None:
            self.direction = torch.tensor(direction, dtype=torch.float32)
        else:
            self.direction = torch.eye(dim)
        
        # Physical bounds for mapping between normalized [-1, 1] and physical space
        if spacing is not None:
            spacing_reversed = list(reversed(spacing))
            self.physical_bounds = torch.tensor([(s - 1) / 2.0 * sp for s, sp in zip(grid_shape, spacing_reversed)])
        else:
            self.physical_bounds = torch.ones(dim)
            
        # Low-dimensional pre-alignment
        self.affine = HierarchicalAffine(dim=dim, transform_type=transform_type)
        
        # Dense Symmetric Displacement Fields stored as parameters/buffers
        self.warp_l2r = nn.Parameter(torch.zeros(1, *grid_shape, dim))
        self.warp_r2l = nn.Parameter(torch.zeros(1, *grid_shape, dim))
        self.warp_l2r_inv = nn.Parameter(torch.zeros(1, *grid_shape, dim))
        self.warp_r2l_inv = nn.Parameter(torch.zeros(1, *grid_shape, dim))
        
        # Loss convergence tracking
        self.affine_losses = []
        self.syn_losses = []

    def get_affine_grid(self, shape, device):
        theta = self.affine.get_affine_grid_matrix().unsqueeze(0)
        grid = F.affine_grid(theta, size=[1, 1] + list(shape), align_corners=True)
        return grid

    def get_inverse_affine_grid(self, shape, device):
        T = self.affine.get_matrix()
        T_inv = torch.inverse(T)
        theta_inv = T_inv[:self.dim, :self.dim + 1].unsqueeze(0)
        grid_inv = F.affine_grid(theta_inv, size=[1, 1] + list(shape), align_corners=True)
        return grid_inv

    def fit(self, fixed_image, moving_image, levels=[8, 4, 2, 1], epochs_per_level=100, 
            affine_epochs=[100, 50, 50, 20], affine_lr=1e-2, cfl_voxels=0.15, 
            similarity_metric='lncc', use_analytical_gradients=True,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None,
            vgg_layers=[4], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='lncc_3d',
            vgg_lncc_window_size=9, syn_metric_weights=None, initial_grid=None, **kwargs):
        """
        Runs the full native pre-alignment and SyN multi-resolution optimization loop.
        fixed_image: (1, 1, *spatial)
        moving_image: (1, 1, *spatial)
        """
        verbose = kwargs.get('verbose', False)
        optimizer_type = kwargs.get('optimizer_type', 'cfl')
        optimizer_lr = kwargs.get('optimizer_lr', 1e-3)
        lncc_window_size = 2 * lncc_radius + 1
        device = fixed_image.device
        dtype = fixed_image.dtype
        dim = self.dim
        spatial_shape = fixed_image.shape[2:]
        
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
            fixed_direction = np.eye(self.dim)
            
        if moving_spacing is None:
            moving_spacing = [1.0] * self.dim
            
        if sampling_percentage is None:
            sampling_percentage = 0.2
            
        if moving_origin is None:
            moving_origin = [0.0] * self.dim
            
        if moving_direction is None:
            moving_direction = np.eye(self.dim)
        
        # Standardize iteration lists to match hierarchy levels length
        if isinstance(epochs_per_level, int):
            epochs_per_level = [epochs_per_level] * len(levels)
        elif len(epochs_per_level) < len(levels):
            epochs_per_level = list(epochs_per_level) + [0] * (len(levels) - len(epochs_per_level))
            
        if isinstance(affine_epochs, int):
            affine_epochs = [affine_epochs] * len(levels)
        elif len(affine_epochs) < len(levels):
            affine_epochs = list(affine_epochs) + [0] * (len(levels) - len(affine_epochs))
            
        self.affine_losses = []
        self.syn_losses = []
        self.initial_grid = initial_grid
        
        
        init_M_phys = kwargs.get('init_M_phys', None)
        init_t_phys = kwargs.get('init_t_phys', None)
        # CoM Initialization Selection (FOV vs Foreground CoM based on downsampled Mattes MI)
        if self.initial_grid is None and init_M_phys is None:
            with torch.no_grad():
                # 1. Compute FOV centers
                Nx_t = torch.tensor(list(reversed(fixed_image.shape[2:])), device=device, dtype=dtype)
                Sx_t = torch.tensor(list(fixed_spacing), device=device, dtype=dtype)
                Ox_t = torch.tensor(list(fixed_origin), device=device, dtype=dtype)
                Dx_t = torch.tensor(np.asarray(fixed_direction), device=device, dtype=dtype)
                com_fixed_fov = Dx_t @ (Sx_t * (Nx_t - 1) / 2.0) + Ox_t
                
                Ny_t = torch.tensor(list(reversed(moving_image.shape[2:])), device=device, dtype=dtype)
                Sy_t = torch.tensor(list(moving_spacing), device=device, dtype=dtype)
                Oy_t = torch.tensor(list(moving_origin), device=device, dtype=dtype)
                Dy_t = torch.tensor(np.asarray(moving_direction), device=device, dtype=dtype)
                com_moving_fov = Dy_t @ (Sy_t * (Ny_t - 1) / 2.0) + Oy_t
                
                t_fov = com_moving_fov - com_fixed_fov
                
                # 2. Compute Foreground (intensity-weighted) centers
                fixed_pos = torch.clamp(fixed_image, min=0.0)
                moving_pos = torch.clamp(moving_image, min=0.0)
                sum_fixed = fixed_pos.sum()
                sum_moving = moving_pos.sum()
                
                if sum_fixed > 1e-5 and sum_moving > 1e-5:
                    grids_f = [torch.arange(s, device=device, dtype=dtype) for s in fixed_image.shape[2:]]
                    meshgrid_f = torch.meshgrid(*grids_f, indexing='ij')
                    idxs_f = torch.stack(list(reversed(meshgrid_f)), dim=-1)
                    
                    grids_m = [torch.arange(s, device=device, dtype=dtype) for s in moving_image.shape[2:]]
                    meshgrid_m = torch.meshgrid(*grids_m, indexing='ij')
                    idxs_m = torch.stack(list(reversed(meshgrid_m)), dim=-1)
                    
                    com_fixed_voxel = torch.sum(fixed_pos.squeeze(0).squeeze(0).unsqueeze(-1) * idxs_f, dim=list(range(dim))) / sum_fixed
                    com_moving_voxel = torch.sum(moving_pos.squeeze(0).squeeze(0).unsqueeze(-1) * idxs_m, dim=list(range(dim))) / sum_moving
                    
                    com_fixed_fg = Dx_t @ (Sx_t * com_fixed_voxel) + Ox_t
                    com_moving_fg = Dy_t @ (Sy_t * com_moving_voxel) + Oy_t
                    
                    t_fg = com_moving_fg - com_fixed_fg
                else:
                    t_fg = t_fov
                
                def eval_translation(t_candidate):
                    X_phys = get_physical_grid_torch(fixed_image.shape[2:], fixed_spacing, fixed_origin, fixed_direction, device=device, dtype=dtype)
                    y_phys = X_phys + t_candidate
                    y_norm = physical_to_normalized_torch(y_phys, moving_image.shape[2:], moving_spacing, moving_origin, moving_direction)
                    J_warped = F.grid_sample(moving_image, y_norm, padding_mode='border', align_corners=True)
                    
                    metric_to_use = similarity_metric[0] if isinstance(similarity_metric, list) else similarity_metric
                    if metric_to_use == 'lncc':
                        return local_ncc_loss_nd(J_warped, fixed_image, window_size=5).item()
                    elif metric_to_use == 'mse':
                        return torch.mean((J_warped - fixed_image) ** 2).item()
                    else:
                        return mattes_mi_loss_nd(J_warped, fixed_image, num_bins=32).item()
                
                loss_fov = eval_translation(t_fov)
                loss_fg = eval_translation(t_fg)
                if verbose:
                    print(f"[CoM Init] t_fov: {t_fov.data.cpu().numpy()}, loss_fov: {loss_fov:.4f}")
                    print(f"[CoM Init] t_fg: {t_fg.data.cpu().numpy()}, loss_fg: {loss_fg:.4f}")
                
                best_t = t_fov if loss_fov < loss_fg else t_fg
                
                # Compute and register T_init (mapping physical rigid translation into grid coordinates)
                H_x = torch.eye(dim + 1, device=device, dtype=dtype)
                H_x[:dim, :dim] = Dx_t @ torch.diag(Sx_t) @ torch.diag((Nx_t - 1) / 2.0)
                H_x[:dim, dim] = com_fixed_fov
                
                H_y = torch.eye(dim + 1, device=device, dtype=dtype)
                H_y[:dim, :dim] = Dy_t @ torch.diag(Sy_t) @ torch.diag((Ny_t - 1) / 2.0)
                H_y[:dim, dim] = com_moving_fov
                
                T_phys = torch.eye(dim + 1, device=device, dtype=dtype)
                T_phys[:dim, dim] = best_t
                
                T_init = torch.inverse(H_y) @ T_phys @ H_x
                self.affine.T_init = T_init
        
        # Standardize similarity_metric to a list of metrics
        if isinstance(similarity_metric, str):
            self.metrics = [similarity_metric]
        elif isinstance(similarity_metric, list):
            self.metrics = list(similarity_metric)
        else:
            self.metrics = [similarity_metric]

        self.metric_weights = syn_metric_weights if syn_metric_weights is not None else [1.0] * len(self.metrics)
        self.loss_functions = []
        
        # Determine if analytical gradients are viable
        use_analytical_gradients = kwargs.get('use_analytical_gradients', True)
        if use_analytical_gradients:
            for metric in self.metrics:
                if isinstance(metric, str):
                    m_str = metric.lower()
                    if 'dinov2' in m_str:
                        print(f"Warning: Metric '{metric}' does not support analytical gradients well. Falling back to autograd.")
                        use_analytical_gradients = False
                        break
        kwargs['use_analytical_gradients'] = use_analytical_gradients
        
        from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
        
        for metric in self.metrics:
            if isinstance(metric, str):
                metric_name_lower = metric.lower()
                if metric_name_lower == 'mattes_mi':
                    self.loss_functions.append(lambda x, y, mask=None: mattes_mi_loss_nd(x, y, mask=mask, num_bins=mattes_bins))
                elif metric_name_lower == 'lncc':
                    self.loss_functions.append(lambda x, y, mask=None: local_ncc_loss_nd(x, y, mask=mask, window_size=lncc_window_size))
                elif metric_name_lower == 'mse':
                    self.loss_functions.append(lambda x, y, mask=None: torch.mean((x - y) ** 2) if mask is None else torch.sum(((x - y) ** 2) * mask) / (mask.sum() + 1e-8))
                elif metric_name_lower == 'vgg19':
                    extractor = VGG19Extractor(feature_layers=vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower in ['dinov2', 'dinov2_small']:
                    extractor = DINOv2Extractor(version='vits14', feature_layers=vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower == 'dinov2_base':
                    extractor = DINOv2Extractor(version='vitb14', feature_layers=vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower == 'resnet10':
                    extractor = ResNet10Extractor(dim=dim, feature_layers=vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower in ['swinunetr', 'swin_unetr']:
                    layers = [4] if vgg_layers == [8] else vgg_layers
                    extractor = SwinUNETRExtractor(feature_layers=layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                else:
                    raise ValueError(f"Unknown similarity metric: {metric}")
            elif isinstance(metric, torch.nn.Module) or callable(metric):
                self.loss_functions.append(metric)
            else:
                raise ValueError(f"Invalid similarity metric: {metric}")
        
        aff_metric = kwargs.get('aff_metric', 'mattes_mi')
        if aff_metric == 'mattes':
            aff_metric = 'mattes_mi'
            
        if aff_metric.lower() == 'mattes_mi':
            self.affine_loss_fn = lambda x, y: mattes_mi_loss_nd(x, y, num_bins=mattes_bins)
        elif aff_metric.lower() == 'lncc':
            self.affine_loss_fn = lambda x, y: local_ncc_loss_nd(x, y, window_size=lncc_window_size)
        elif aff_metric.lower() == 'mse':
            self.affine_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
        else:
            self.affine_loss_fn = self.loss_functions[0]
        
        # Parse smoothing_sigmas
        smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
        if smoothing_sigmas is None:
            import math
            sigmas = [float(math.sqrt(s)) if s > 1 else 0.0 for s in levels]
        elif isinstance(smoothing_sigmas, (int, float)):
            sigmas = [float(smoothing_sigmas)] * len(levels)
        else:
            sigmas = [float(s) for s in smoothing_sigmas]
            if len(sigmas) != len(levels):
                raise ValueError(f"Length of smoothing_sigmas ({len(sigmas)}) must match levels ({len(levels)})")
                
        # --- 0. Construct Image Pyramids ---
        I_pyr = []
        J_pyr = []
        for level_idx, s in enumerate(levels):
            sig = sigmas[level_idx]
            if sig > 0.0:
                fixed_smoothed = separable_gaussian_filter(fixed_image.movedim(1, -1), sig, spacing=fixed_spacing).movedim(-1, 1)
                moving_smoothed = separable_gaussian_filter(moving_image.movedim(1, -1), sig, spacing=moving_spacing).movedim(-1, 1)
            else:
                fixed_smoothed = fixed_image
                moving_smoothed = moving_image
                
            if s > 1:
                I_level = F.interpolate(fixed_smoothed, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                J_level = F.interpolate(moving_smoothed, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
            else:
                I_level = fixed_smoothed
                J_level = moving_smoothed
            I_pyr.append(I_level)
            J_pyr.append(J_level)
        
        if sum(affine_epochs) > 0:
            for level_idx, scale in enumerate(levels):
                curr_affine_epochs = affine_epochs[level_idx]
                if curr_affine_epochs <= 0:
                    continue
                I_curr = I_pyr[level_idx]
                J_curr = J_pyr[level_idx]
                curr_spatial = I_curr.shape[2:]
                
                if initial_grid is not None:
                    initial_grid_level = F.interpolate(
                        torch.movedim(initial_grid, -1, 1),
                        size=curr_spatial,
                        mode='bilinear' if dim == 2 else 'trilinear',
                        align_corners=True
                    )
                    initial_grid_level = torch.movedim(initial_grid_level, 1, -1)
                else:
                    initial_grid_level = None
                
                # Hierarchical Parameter Unlocking
                # Count only active affine levels (those with iterations > 0)
                active_affine_levels = sum(1 for its in affine_epochs if its > 0)
                active_params = [self.affine.translation]
                
                # Rigid unlocking: at 2nd active level, or if only 1 active level
                if level_idx >= 1 or active_affine_levels <= 1:
                    if hasattr(self.affine, 'omega') and isinstance(self.affine.omega, nn.Parameter):
                        active_params.append(self.affine.omega)
                        
                # Affine unlocking: at 3rd active level, or if ≤2 active levels
                if level_idx >= 2 or active_affine_levels <= 2:
                    if hasattr(self.affine, 'scale') and isinstance(self.affine.scale, nn.Parameter):
                        active_params.append(self.affine.scale)
                    if hasattr(self.affine, 'anisotropic_scale') and isinstance(self.affine.anisotropic_scale, nn.Parameter):
                        active_params.append(self.affine.anisotropic_scale)
                    if hasattr(self.affine, 'shear') and isinstance(self.affine.shear, nn.Parameter):
                        active_params.append(self.affine.shear)
                        
                if level_idx == 0:
                    optimizer = torch.optim.Adam(active_params, lr=affine_lr)
                else:
                    existing_params = set()
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            existing_params.add(p)
                    new_params = [p for p in active_params if p not in existing_params]
                    if new_params:
                        optimizer.add_param_group({'params': new_params})
                
                level_affine_losses = []
                for epoch in range(curr_affine_epochs):
                    optimizer.zero_grad()
                    is_pure_mattes_sampled = (
                        len(self.metrics) == 1 and 
                        self.metrics[0].lower() == 'mattes_mi' and 
                        sampling_percentage is not None and 
                        sampling_percentage < 1.0 and
                        initial_grid is None
                    )
                    
                    if is_pure_mattes_sampled:
                        # Coordinate-level random sampling for Mattes MI
                        N_total = np.prod(curr_spatial)
                        min_samples = int(0.5 * mattes_bins**2)
                        N_samples = int(np.clip(int(N_total * sampling_percentage), min_samples, N_total))
                        
                        coords_shape = (1,) + (1,) * (dim - 1) + (N_samples, dim)
                        coords = torch.rand(coords_shape, device=device, dtype=dtype) * 2.0 - 1.0
                        coords_hom = torch.cat([coords, torch.ones(coords_shape[:-1] + (1,), device=device, dtype=dtype)], dim=-1)
                        
                        theta = self.affine.get_affine_grid_matrix().unsqueeze(0)
                        coords_warped = torch.matmul(coords_hom, theta.transpose(-1, -2))
                        
                        I_sampled = F.grid_sample(I_curr, coords, padding_mode='border', align_corners=True)
                        moving_warped = F.grid_sample(J_curr, coords_warped, padding_mode='border', align_corners=True)
                        loss = mattes_mi_loss_core(moving_warped.flatten(), I_sampled.flatten(), num_bins=mattes_bins)
                    else:
                        grid = self.get_affine_grid(curr_spatial, device)
                        if initial_grid_level is not None:
                            grid = compose_grids(initial_grid_level, grid)
                        moving_warped = F.grid_sample(J_curr, grid, padding_mode='border', align_corners=True)
                        loss = self.affine_loss_fn(moving_warped, I_curr)
                    
                    loss.backward()
                    optimizer.step()
                    self.affine_losses.append(loss)
                    level_affine_losses.append(loss)
                    if verbose:
                        print(f"[pytorch-fit] Affine Level {level_idx} Epoch {epoch}: loss={loss.item():.6f}")
                    if len(level_affine_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_affine_epochs - 1):
                        recent_losses = [l.item() for l in level_affine_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-8):
                            break
                
        # --- 2. SyN Registration ---
        # Initialize warps at the coarsest level resolution
        curr_spatial = I_pyr[0].shape[2:]
        
        warp_l2r = torch.zeros(1, *curr_spatial, dim, device=device, dtype=dtype)
        warp_r2l = torch.zeros(1, *curr_spatial, dim, device=device, dtype=dtype)
        warp_l2r_inv = torch.zeros_like(warp_l2r)
        warp_r2l_inv = torch.zeros_like(warp_r2l)
        
        for level_idx, scale in enumerate(levels):
            I_curr = I_pyr[level_idx]
            J_curr = J_pyr[level_idx]
            curr_spatial = I_curr.shape[2:]
            
            if level_idx > 0:
                warp_l2r = F.interpolate(torch.movedim(warp_l2r, -1, 1), size=curr_spatial, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                warp_l2r = torch.movedim(warp_l2r, 1, -1)
                
                warp_r2l = F.interpolate(torch.movedim(warp_r2l, -1, 1), size=curr_spatial, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                warp_r2l = torch.movedim(warp_r2l, 1, -1)
                
                warp_l2r_inv = F.interpolate(torch.movedim(warp_l2r_inv, -1, 1), size=curr_spatial, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                warp_l2r_inv = torch.movedim(warp_l2r_inv, 1, -1)
                
                warp_r2l_inv = F.interpolate(torch.movedim(warp_r2l_inv, -1, 1), size=curr_spatial, mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                warp_r2l_inv = torch.movedim(warp_r2l_inv, 1, -1)
                
            warp_l2r.requires_grad_(True)
            warp_r2l.requires_grad_(True)
            
            # Compute current level physical spacing
            curr_spacing_fixed = [sp * (orig_N - 1) / (curr_N - 1) if curr_N > 1 else sp for sp, orig_N, curr_N in zip(fixed_spacing, reversed(spatial_shape), reversed(curr_spatial))]
            curr_spacing_moving = [sp * (orig_N - 1) / (curr_N - 1) if curr_N > 1 else sp for sp, orig_N, curr_N in zip(moving_spacing, reversed(moving_image.shape[2:]), reversed(J_curr.shape[2:]))]
            curr_spacing_fixed = tuple(curr_spacing_fixed)
            curr_spacing_moving = tuple(curr_spacing_moving)
            
            with torch.no_grad():
                if hasattr(self, 'affine'):
                    if init_M_phys is not None and init_t_phys is not None:
                        M_phys = init_M_phys.to(device)
                        t_phys = init_t_phys.to(device)
                        self.init_M_phys = M_phys
                        self.init_t_phys = t_phys
                    else:
                        T_grid = self.affine.get_matrix()
                        M_phys, t_phys = grid_to_physical_affine_torch(
                            T_grid,
                            spatial_shape, fixed_spacing, fixed_origin, fixed_direction,
                            moving_image.shape[2:], moving_spacing, moving_origin, moving_direction
                        )
                        self.init_M_phys = None
                        self.init_t_phys = None
                X_phys = get_physical_grid_torch(curr_spatial, curr_spacing_fixed, fixed_origin, fixed_direction, device=device, dtype=dtype)
                b_mask = get_boundary_mask(curr_spatial, device, dtype)
                
                # Cache physical parameter conversion tensors
                fixed_shape_t = torch.tensor(list(curr_spatial), device=device, dtype=dtype)
                fixed_spacing_rev = tuple(reversed(curr_spacing_fixed))
                fixed_origin_rev = tuple(reversed(fixed_origin))
                fixed_direction_rev = np.asarray(fixed_direction)[::-1, ::-1].copy()
                fixed_spacing_t = torch.tensor(fixed_spacing_rev, device=device, dtype=dtype)
                fixed_origin_t = torch.tensor(fixed_origin_rev, device=device, dtype=dtype)
                fixed_direction_t = torch.tensor(fixed_direction_rev, device=device, dtype=dtype)
                
                moving_shape_t = torch.tensor(list(J_curr.shape[2:]), device=device, dtype=dtype)
                moving_spacing_rev = tuple(reversed(curr_spacing_moving))
                moving_origin_rev = tuple(reversed(moving_origin))
                moving_direction_rev = np.asarray(moving_direction)[::-1, ::-1].copy()
                moving_spacing_t = torch.tensor(moving_spacing_rev, device=device, dtype=dtype)
                moving_origin_t = torch.tensor(moving_origin_rev, device=device, dtype=dtype)
                moving_direction_t = torch.tensor(moving_direction_rev, device=device, dtype=dtype)
                
                curr_spacing_fixed_t = torch.tensor(list(reversed(curr_spacing_fixed)), device=device, dtype=dtype)
                
                if self.initial_grid is not None:
                    initial_grid_level = F.interpolate(
                        torch.movedim(self.initial_grid.to(device=device, dtype=dtype), -1, 1),
                        size=curr_spatial,
                        mode='bilinear' if dim == 2 else 'trilinear',
                        align_corners=True
                    ).movedim(1, -1)
                else:
                    initial_grid_level = None
            
            # Deep feature degeneracy check: fall back to LNCC if min(curr_spatial) < 32
            is_degenerate = min(curr_spatial) < 32
            active_loss_functions = []
            active_metric_names = []
            for metric in self.metrics:
                is_deep = False
                metric_name = str(metric)
                if isinstance(metric, str):
                    m_lower = metric.lower()
                    if m_lower in ['vgg19', 'resnet10', 'dinov2', 'dinov2_small', 'dinov2_base', 'swinunetr', 'swin_unetr']:
                        is_deep = True
                elif hasattr(metric, 'extractor') or ('FeatureSpaceLoss' in metric.__class__.__name__):
                    is_deep = True
                    
                if is_degenerate and is_deep:
                    active_loss_functions.append(lambda x, y: local_ncc_loss_nd(x, y, window_size=lncc_window_size))
                    active_metric_names.append('lncc_fallback')
                else:
                    metric_idx = self.metrics.index(metric)
                    active_loss_functions.append(self.loss_functions[metric_idx])
                    active_metric_names.append(metric_name)
            
            if isinstance(epochs_per_level, int):
                curr_syn_epochs = epochs_per_level
            else:
                curr_syn_epochs = epochs_per_level[level_idx]
                
            if optimizer_type == 'rprop':
                self._rprop_step_l = torch.ones_like(warp_l2r) * optimizer_lr
                self._rprop_step_r = torch.ones_like(warp_r2l) * optimizer_lr
                self._rprop_prev_grad_l = torch.zeros_like(warp_l2r)
                self._rprop_prev_grad_r = torch.zeros_like(warp_r2l)
            elif optimizer_type == 'adam':
                self._adam_m_l = torch.zeros_like(warp_l2r)
                self._adam_m_r = torch.zeros_like(warp_r2l)
                self._adam_v_l = torch.zeros_like(warp_l2r)
                self._adam_v_r = torch.zeros_like(warp_r2l)
                self._adam_t = 0
                
            level_syn_losses = []
            for epoch in range(curr_syn_epochs):
                if warp_l2r.grad is not None: warp_l2r.grad.zero_()
                if warp_r2l.grad is not None: warp_r2l.grad.zero_()
                
                # Real SyN: Pull both images to the midpoint domain
                I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask = prepare_mid_images_and_gradients_torch(
                    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
                    X_phys,
                    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                    curr_spacing_fixed, curr_spacing_moving,
                    M_phys, t_phys, initial_grid_level
                )

                if verbose >= 2:
                    if dim == 2:
                        I_mid_np = I_mid.detach().squeeze(0).squeeze(0).cpu().numpy().T
                        J_mid_np = J_mid.detach().squeeze(0).squeeze(0).cpu().numpy().T
                    else:
                        I_mid_np = I_mid.detach().squeeze(0).squeeze(0).cpu().numpy().transpose(2, 1, 0)
                        J_mid_np = J_mid.detach().squeeze(0).squeeze(0).cpu().numpy().transpose(2, 1, 0)
                    
                    import tempfile
                    import ants
                    temp_I = tempfile.NamedTemporaryFile(suffix=f'_level{level_idx}_epoch{epoch}_Imid.nii.gz', delete=False).name
                    temp_J = tempfile.NamedTemporaryFile(suffix=f'_level{level_idx}_epoch{epoch}_Jmid.nii.gz', delete=False).name
                    
                    I_mid_img = ants.from_numpy(I_mid_np, origin=fixed_origin, spacing=curr_spacing_fixed, direction=fixed_direction)
                    J_mid_img = ants.from_numpy(J_mid_np, origin=fixed_origin, spacing=curr_spacing_fixed, direction=fixed_direction)
                    
                    self.fixed_mid_img = I_mid_img
                    self.moving_mid_img = J_mid_img
                    
                    ants.image_write(I_mid_img, temp_I)
                    ants.image_write(J_mid_img, temp_J)
                    print(f"[verbose-2] Saved midpoint images at Level {level_idx} Epoch {epoch}:\n  Fixed-mid: {temp_I}\n  Moving-mid: {temp_J}")

                if use_analytical_gradients:
                    I_mid_det = I_mid.detach().requires_grad_(True)
                    J_mid_det = J_mid.detach().requires_grad_(True)
                    
                    loss = 0.0
                    metric_losses_dict = {}
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, self.metric_weights):
                        try:
                            val_loss = fn(J_mid_det, I_mid_det, mask=in_bounds_mask)
                        except TypeError:
                            val_loss = fn(J_mid_det, I_mid_det)
                        loss += weight * val_loss
                        metric_losses_dict[name] = val_loss.item()

                    loss.backward()
                    loss_val = loss.item()
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)
                    
                    with torch.no_grad():
                        g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                        g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                        
                        grad_l_raw = (g_im.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                        warp_l2r.grad = grad_l_raw

                        grad_r_raw = (g_jm.movedim(1, -1) * grad_J_mid_sampled).contiguous()
                        warp_r2l.grad = grad_r_raw

                else:
                    loss = 0.0
                    metric_losses_dict = {}
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, self.metric_weights):
                        try:
                            val_loss = fn(J_mid, I_mid, mask=in_bounds_mask)
                        except TypeError:
                            val_loss = fn(J_mid, I_mid)
                        loss += weight * val_loss
                        metric_losses_dict[name] = val_loss.item()
                        
                    loss.backward()
                    loss_val = loss.item()
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)

                
                with torch.no_grad():
                    grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, self.fluid_sigma, spacing=curr_spacing_fixed)
                    grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, self.fluid_sigma, spacing=curr_spacing_fixed)
                    
                    # ITK-style CFL: Euclidean norm in PHYSICAL coordinates (mm)
                    max_norm_l = torch.sqrt(torch.sum(grad_l**2, dim=-1)).max()
                    max_norm_r = torch.sqrt(torch.sum(grad_r**2, dim=-1)).max()
                    
                    if verbose >= 2:
                        print(f"DEBUG PyTorch L{level_idx} E{epoch} max_norm_l: {float(max_norm_l)}, max_norm_r: {float(max_norm_r)}")
                    
                    if optimizer_type == 'cfl':
                        # Scale physical step size to cfl_voxels * spacing, avoiding scaling noise
                        if max_norm_l > 1e-12:
                            delta_l = (cfl_voxels * curr_spacing_fixed_t) * (grad_l / max_norm_l)
                        else:
                            delta_l = torch.zeros_like(grad_l)
                            
                        if max_norm_r > 1e-12:
                            delta_r = (cfl_voxels * curr_spacing_fixed_t) * (grad_r / max_norm_r)
                        else:
                            delta_r = torch.zeros_like(grad_r)
                        
                        # Greedy SyN composition: φ_new = φ_old ∘ (Id - ∂loss/∂warp)
                        coords_phys_l = X_phys - delta_l
                        coords_norm_l = physical_to_normalized_torch_cached(
                            coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                        )
                        warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
                        warp_l2r.copy_(warp_l2r_sampled - delta_l)
                        
                        coords_phys_r = X_phys - delta_r
                        coords_norm_r = physical_to_normalized_torch_cached(
                            coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                        )
                        warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
                        warp_r2l.copy_(warp_r2l_sampled - delta_r)
                        
                        # ITK-standard Dirichlet zero boundary enforcement after composition.
                        warp_l2r.mul_(b_mask)
                        warp_r2l.mul_(b_mask)
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_spacing_fixed))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_spacing_fixed))
                            
                        # ITK-style diffeomorphic projection: compute inverse fields
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                    
                    elif optimizer_type == 'rprop':
                        def rprop_update(grad, prev_grad, step):
                            sign_change = grad * prev_grad
                            step_inc = torch.clamp(step * 1.2, max=50.0)
                            step_dec = torch.clamp(step * 0.5, min=1e-6)
                            
                            step_new = torch.where(sign_change > 0, step_inc, torch.where(sign_change < 0, step_dec, step))
                            update = torch.where(sign_change >= 0, -torch.sign(grad) * step_new, torch.zeros_like(grad))
                            grad_new = torch.where(sign_change < 0, torch.zeros_like(grad), grad)
                            return update, step_new, grad_new
                            
                        update_l, self._rprop_step_l, self._rprop_prev_grad_l = rprop_update(grad_l, self._rprop_prev_grad_l, self._rprop_step_l)
                        update_r, self._rprop_step_r, self._rprop_prev_grad_r = rprop_update(grad_r, self._rprop_prev_grad_r, self._rprop_step_r)
                        
                        warp_l2r.copy_(warp_l2r + update_l)
                        warp_r2l.copy_(warp_r2l + update_r)
                        
                        warp_l2r.mul_(b_mask)
                        warp_r2l.mul_(b_mask)
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_spacing_fixed))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_spacing_fixed))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        
                    elif optimizer_type == 'adam':
                        self._adam_t += 1
                        beta1, beta2 = 0.9, 0.999
                        eps = 1e-8
                        
                        self._adam_m_l = beta1 * self._adam_m_l + (1 - beta1) * grad_l
                        self._adam_v_l = beta2 * self._adam_v_l + (1 - beta2) * (grad_l ** 2)
                        m_hat_l = self._adam_m_l / (1 - beta1 ** self._adam_t)
                        v_hat_l = self._adam_v_l / (1 - beta2 ** self._adam_t)
                        update_l = -optimizer_lr * m_hat_l / (torch.sqrt(v_hat_l) + eps)
                        
                        self._adam_m_r = beta1 * self._adam_m_r + (1 - beta1) * grad_r
                        self._adam_v_r = beta2 * self._adam_v_r + (1 - beta2) * (grad_r ** 2)
                        m_hat_r = self._adam_m_r / (1 - beta1 ** self._adam_t)
                        v_hat_r = self._adam_v_r / (1 - beta2 ** self._adam_t)
                        update_r = -optimizer_lr * m_hat_r / (torch.sqrt(v_hat_r) + eps)
                        
                        warp_l2r.copy_(warp_l2r + update_l)
                        warp_r2l.copy_(warp_r2l + update_r)
                        
                        warp_l2r.mul_(b_mask)
                        warp_r2l.mul_(b_mask)
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_spacing_fixed))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_spacing_fixed))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        
                    elif optimizer_type == 'sgd':
                        update_l = -optimizer_lr * grad_l
                        update_r = -optimizer_lr * grad_r
                        
                        warp_l2r.copy_(warp_l2r + update_l)
                        warp_r2l.copy_(warp_r2l + update_r)
                        
                        warp_l2r.mul_(b_mask)
                        warp_r2l.mul_(b_mask)
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_spacing_fixed))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_spacing_fixed))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                    
                    if False:
                        warp_r2l.copy_(update_inverse_field_nd(
                            warp_r2l_inv.detach(), warp_r2l.detach(), steps=self.inverse_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        ))
                    
                    if verbose:
                        loss_details = ", ".join([f"{k}={v:.6f}" for k, v in metric_losses_dict.items()])
                        print(f"[pytorch-fit] SyN Level {level_idx} Epoch {epoch}: loss={loss_val:.6f} ({loss_details}), warp_l2r max norm={float(torch.sqrt(torch.sum(warp_l2r**2, dim=-1)).max()):.4f}")
                    if len(level_syn_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_syn_epochs - 1):
                        recent_losses = [l.item() if hasattr(l, 'item') else l for l in level_syn_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-8):
                            break
                    
            warp_l2r.requires_grad_(False)
            warp_r2l.requires_grad_(False)
            
        with torch.no_grad():
            # Interpolate midpoint fields to target grid resolution
            w_l2r = F.interpolate(torch.movedim(warp_l2r, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            w_r2l = F.interpolate(torch.movedim(warp_r2l, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            
            # Recompute midpoint inverses at full resolution for accurate composition
            # Using interpolated in-loop inverses as warm-start initial guesses
            w_l2r_inv_interp = F.interpolate(torch.movedim(warp_l2r_inv, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            w_r2l_inv_interp = F.interpolate(torch.movedim(warp_r2l_inv, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            midpoint_inv_steps = self.inverse_steps  # Warm-started from in-loop inverse; fewer steps needed
            w_l2r_inv = update_inverse_field_nd(
                w_l2r, w_l2r_inv_interp.detach(), steps=midpoint_inv_steps, method=self.inverse_method,
                spacing=fixed_spacing, origin=fixed_origin, direction=fixed_direction
            )
            w_r2l_inv = update_inverse_field_nd(
                w_r2l, w_r2l_inv_interp.detach(), steps=midpoint_inv_steps, method=self.inverse_method,
                spacing=fixed_spacing, origin=fixed_origin, direction=fixed_direction
            )
            
            X_phys = get_physical_grid_torch(self.grid_shape, fixed_spacing, fixed_origin, fixed_direction, device=device, dtype=dtype)
            
            # Pre-compute normalization tensors for composition
            comp_shape_t = torch.tensor(list(self.grid_shape), device=device, dtype=dtype)
            comp_spacing_t = torch.tensor(list(reversed(fixed_spacing)), device=device, dtype=dtype)
            comp_origin_t = torch.tensor(list(reversed(fixed_origin)), device=device, dtype=dtype)
            comp_direction_t = torch.tensor(np.asarray(fixed_direction)[::-1, ::-1].copy(), device=device, dtype=dtype)
            
            # Compose midpoint fields in physical space
            phi_l2r_phys = X_phys + w_l2r_inv
            coords_norm = physical_to_normalized_torch_cached(phi_l2r_phys, comp_shape_t, comp_spacing_t, comp_origin_t, comp_direction_t)
            disp_r2l_sampled = F.grid_sample(torch.movedim(w_r2l, -1, 1), coords_norm, padding_mode='border', align_corners=True).movedim(1, -1)
            full_l2r_phys = phi_l2r_phys + disp_r2l_sampled
            self.warp_l2r = nn.Parameter(full_l2r_phys - X_phys)
            self.warp_l2r.is_physical = True
            
            phi_r2l_phys = X_phys + w_r2l_inv
            coords_norm_r = physical_to_normalized_torch_cached(phi_r2l_phys, comp_shape_t, comp_spacing_t, comp_origin_t, comp_direction_t)
            disp_l2r_sampled = F.grid_sample(torch.movedim(w_l2r, -1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
            full_r2l_phys = phi_r2l_phys + disp_l2r_sampled
            self.warp_r2l = nn.Parameter(full_r2l_phys - X_phys)
            self.warp_r2l.is_physical = True
            
            # The mathematical inverse of F -> M (phi_2 o phi_1^-1) 
            # is M -> F (phi_1 o phi_2^-1). We compute it algebraically for a near-perfect guess.
            phi_r2l_phys = X_phys + w_r2l_inv
            coords_norm_r = physical_to_normalized_torch_cached(phi_r2l_phys, comp_shape_t, comp_spacing_t, comp_origin_t, comp_direction_t)
            disp_l2r_sampled = F.grid_sample(torch.movedim(w_l2r, -1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
            algebraic_inv = (phi_r2l_phys + disp_l2r_sampled) - X_phys
            
            # Refine the inverse using the fixed-point solver to drive max_error < 1.0 and fix boundary artifacts
            final_inv_steps = max(self.inverse_steps, 50)
            refined_inv = update_inverse_field_nd(
                self.warp_l2r.data, algebraic_inv,
                spacing=fixed_spacing, origin=fixed_origin, direction=fixed_direction,
                steps=final_inv_steps, method=self.inverse_method
            )
            self.warp_l2r_inv = nn.Parameter(refined_inv)
            self.warp_l2r_inv.is_physical = True
            
            # To strictly satisfy both phi_1 and phi_2 inverse checks without modifying the highly-accurate forward mapping
            # (which would drop Dice scores), we polish the reverse mapping's inverse independently.
            self.warp_r2l = nn.Parameter(algebraic_inv.clone())
            refined_r2l_inv = update_inverse_field_nd(
                self.warp_r2l.data, self.warp_l2r.data.clone(),
                spacing=fixed_spacing, origin=fixed_origin, direction=fixed_direction,
                steps=final_inv_steps, method=self.inverse_method
            )
            self.warp_r2l_inv = nn.Parameter(refined_r2l_inv)
            
            self.warp_r2l.is_physical = True
            self.warp_r2l_inv.is_physical = True
            
            # Convert all logged losses to floats in a single batch
            self.affine_losses = [l.item() if hasattr(l, 'item') else l for l in self.affine_losses]
            self.syn_losses = [l.item() if hasattr(l, 'item') else l for l in self.syn_losses]

    def forward(self, moving_image, fixed_image=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        """
        Warps the moving image using the affine pre-alignment and dense forward field.
        """
        device = moving_image.device
        dtype = moving_image.dtype
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))
        
        # Permute input to ZYX order
        moving_image_zyx = moving_image.permute(perm)
        
        # Fixed properties define output space
        spatial_shape = self.grid_shape
        spacing = self.spacing if self.spacing is not None else [1.0] * dim
        origin = self.origin if self.origin is not None else [0.0] * dim
        direction = self.direction if self.direction is not None else torch.eye(dim, device=device, dtype=dtype)
        
        # Moving properties
        if moving_spacing is None: moving_spacing = spacing
        if moving_origin is None: moving_origin = origin
        if moving_direction is None: moving_direction = direction
        
        X_phys = get_physical_grid_torch(spatial_shape, spacing, origin, direction, device=device, dtype=dtype)
        
        warp_resampled = F.interpolate(
            torch.movedim(self.warp_l2r, -1, 1), 
            size=spatial_shape, 
            mode='bilinear' if dim == 2 else 'trilinear', 
            align_corners=True
        )
        warp_resampled = torch.movedim(warp_resampled, 1, -1)
        
        phi_l2r_phys = X_phys + warp_resampled
        
        if hasattr(self, 'init_M_phys') and self.init_M_phys is not None and hasattr(self, 'init_t_phys') and self.init_t_phys is not None:
            M_phys = self.init_M_phys.to(device)
            t_phys = self.init_t_phys.to(device)
        else:
            T_grid = self.affine.get_matrix()
            M_phys, t_phys = grid_to_physical_affine_torch(
                T_grid, spatial_shape, spacing, origin, direction,
                moving_image_zyx.shape[2:], moving_spacing, moving_origin, moving_direction
            )
        
        y_phys = phi_l2r_phys @ M_phys.t() + t_phys
        composed_grid = physical_to_normalized_torch(y_phys, moving_image_zyx.shape[2:], moving_spacing, moving_origin, moving_direction)
        
        if hasattr(self, 'initial_grid') and self.initial_grid is not None:
            initial_grid_resampled = F.interpolate(
                torch.movedim(self.initial_grid.to(device=device, dtype=dtype), -1, 1),
                size=spatial_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            )
            initial_grid_resampled = torch.movedim(initial_grid_resampled, 1, -1)
            composed_grid = compose_grids(initial_grid_resampled, composed_grid)
            
        warped_zyx = F.grid_sample(moving_image_zyx, composed_grid, padding_mode='border', align_corners=True)
        return warped_zyx.permute(perm)

    def forward_inverse(self, fixed_image, moving_shape=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        """
        Warps the fixed image using the inverse dense warp and inverse affine transform.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))
        
        # Permute input to ZYX order
        fixed_image_zyx = fixed_image.permute(perm)
        
        fixed_shape = fixed_image_zyx.shape[2:]
        spacing = self.spacing if self.spacing is not None else [1.0] * dim
        origin = self.origin if self.origin is not None else [0.0] * dim
        direction = self.direction if self.direction is not None else torch.eye(dim, device=device, dtype=dtype)
        
        # Moving properties define output space
        if moving_shape is None: moving_shape = self.grid_shape
        if moving_spacing is None: moving_spacing = spacing
        if moving_origin is None: moving_origin = origin
        if moving_direction is None: moving_direction = direction
        
        Y_phys = get_physical_grid_torch(moving_shape, moving_spacing, moving_origin, moving_direction, device=device, dtype=dtype)
        
        warp_resampled = F.interpolate(
            torch.movedim(self.warp_r2l, -1, 1), 
            size=moving_shape, 
            mode='bilinear' if dim == 2 else 'trilinear', 
            align_corners=True
        )
        warp_resampled = torch.movedim(warp_resampled, 1, -1)
        
        phi_r2l_phys = Y_phys + warp_resampled
        
        if hasattr(self, 'init_M_phys') and self.init_M_phys is not None and hasattr(self, 'init_t_phys') and self.init_t_phys is not None:
            M_phys = self.init_M_phys.to(device)
            t_phys = self.init_t_phys.to(device)
            M_phys_inv = torch.linalg.inv(M_phys)
            t_phys_inv = - (M_phys_inv @ t_phys)
        else:
            T_grid = self.affine.get_matrix()
            T_inv = torch.linalg.inv(T_grid)
            M_phys_inv, t_phys_inv = grid_to_physical_affine_torch(
                T_inv, moving_shape, moving_spacing, moving_origin, moving_direction,
                fixed_shape, spacing, origin, direction
            )
        
        x_phys = phi_r2l_phys @ M_phys_inv.t() + t_phys_inv
        composed_grid = physical_to_normalized_torch(x_phys, fixed_shape, spacing, origin, direction)
        
        warped_zyx = F.grid_sample(fixed_image_zyx, composed_grid, padding_mode='border', align_corners=True)
        return warped_zyx.permute(perm)

    def get_forward_transform(self, fixed_metadata):
        """Returns the fully interoperable SyNToTransform object for the forward (moving->fixed) mapping."""
        device = self.warp_l2r.device
        grid_affine = self.get_affine_grid(self.grid_shape, device)
        return SyNToTransform(
            affine_grid=grid_affine, 
            warp_field=self.warp_l2r, 
            metadata=fixed_metadata, 
            device=device,
            is_physical=True
        )

    def get_inverse_transform(self, moving_metadata):
        """Returns the fully interoperable SyNToTransform object for the inverse (fixed->moving) mapping."""
        device = self.warp_r2l.device
        grid_affine_inv = self.get_inverse_affine_grid(self.grid_shape, device)
        return SyNToTransform(
            affine_grid=grid_affine_inv, 
            warp_field=self.warp_r2l, 
            metadata=moving_metadata, 
            device=device,
            is_physical=True
        )

def grid_to_physical_affine(T_grid, fixed, moving):
    dim = len(fixed.shape)
    Nx = np.array(list(reversed(fixed.shape)), dtype=np.float32)
    Ny = np.array(list(reversed(moving.shape)), dtype=np.float32)
    
    # Reverse spacing, origin, and direction to match PyTorch/JAX (z, y, x) order
    Sx = np.array(fixed.spacing)[::-1]
    Sy = np.array(moving.spacing)[::-1]
    Ox = np.array(fixed.origin)[::-1]
    Oy = np.array(moving.origin)[::-1]
    Dx = np.array(fixed.direction)[::-1, ::-1]
    Dy = np.array(moving.direction)[::-1, ::-1]
    
    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    
    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0
    
    Kx_inv = np.linalg.inv(Kx)
    Sx_inv = np.linalg.inv(np.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx
    
    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
    
    perm = list(range(dim - 1, -1, -1))
    T_yx = T_grid.copy()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]
    
    A_grid = T_yx[:dim, :dim]
    t_grid = T_yx[:dim, dim]
    
    # Compute in (z, y, x) space
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy
    
    # Permute from (z, y, x) to (x, y, z) for ITK physical space
    P = np.eye(dim)[::-1]
    M_phys_xyz = P @ M_phys @ P
    t_phys_xyz = P @ t_phys
    
    return M_phys_xyz, t_phys_xyz


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


def parse_ants_affine(tx_list, dim):
    """
    Parses a single ANTs .mat affine transform into M_phys and t_phys tensors.
    Takes into account the center of rotation C as per rule:
    t_new = t + C - M @ C
    Returns None, None if the transform list is not a single .mat file.
    """
    import ants
    import numpy as np
    import torch
    
    if len(tx_list) != 1:
        return None, None
        
    tx_path = tx_list[0]
    if not tx_path.endswith('.mat'):
        return None, None
        
    try:
        tx = ants.read_transform(tx_path)
    except Exception:
        return None, None
        
    params = tx.parameters
    fixed_params = tx.fixed_parameters
    
    if len(params) == 12 and dim == 3:
        M = np.array(params[:9]).reshape(3, 3)
        t = np.array(params[9:])
        C = np.array(fixed_params)
    elif len(params) == 6 and dim == 2:
        M = np.array(params[:4]).reshape(2, 2)
        t = np.array(params[4:])
        C = np.array(fixed_params)
    else:
        return None, None
        
    t_new = t + C - M @ C
    
    # Permute from (x, y, z) to (z, y, x) for PyTorch physical space
    P = np.eye(dim)[::-1]
    M_zyx = P @ M @ P
    t_zyx = P @ t_new
    
    M_phys = torch.from_numpy(M_zyx).to(torch.float32)
    t_phys = torch.from_numpy(t_zyx).to(torch.float32)
    return M_phys, t_phys


def compute_initial_grid(fixed, moving, tx_list):
    """
    Computes an initial_grid (representing the mapping from fixed space to moving space
    under the initial transform) using coordinate warping.
    """
    import numpy as np
    import ants
    dim = moving.dimension
    
    # 1. Get moving physical coordinates via numpy meshgrid
    shape = moving.shape
    grids = [np.arange(s) for s in shape]
    meshgrid_idxs = np.meshgrid(*grids, indexing='ij')
    idxs = np.stack(meshgrid_idxs, axis=-1)
    
    direction = np.array(moving.direction)
    spacing = np.array(moving.spacing)
    origin = np.array(moving.origin)
    
    idxs_flat = idxs.reshape(-1, dim)
    scaled_idxs = idxs_flat * spacing
    phys_flat = (direction @ scaled_idxs.T).T + origin
    coord_np = phys_flat.reshape(shape + (dim,)).astype(np.float32)
    
    # 2. Warp each coordinate component image to the fixed space
    warped_coords = []
    for d in range(dim):
        c_img = ants.from_numpy(coord_np[..., d], origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
        w_c_img = ants.apply_transforms(fixed=fixed, moving=c_img, transformlist=tx_list)
        warped_coords.append(w_c_img.numpy())
        
    moving_phys_at_fixed = np.stack(warped_coords, axis=-1)
    
    # 3. Map physical coordinates to voxel indices in moving space
    shape = moving_phys_at_fixed.shape
    phys_flat = moving_phys_at_fixed.reshape(-1, dim)
    
    direction_inv = np.linalg.inv(direction)
    diff = phys_flat - origin
    sp_idx = diff @ direction_inv.T
    voxel_idx = sp_idx / spacing
    
    # 4. Normalize voxel indices to [-1, 1] and reverse component order to align with grid_sample (x, y, [z]) convention
    normalized_coords = []
    for d in range(dim):
        N = moving.shape[d]
        norm_d = (voxel_idx[:, d] / (N - 1)) * 2.0 - 1.0
        normalized_coords.append(norm_d)
        
    normalized_grid_flat = np.stack(normalized_coords, axis=-1)
    
    initial_grid = normalized_grid_flat.reshape((1,) + fixed.shape + (dim,))
    return initial_grid.astype(np.float32)


def calculate_inverse_identity_error(W_disp: torch.Tensor, W_inv_disp: torch.Tensor, spacing, origin, direction) -> dict:
    """
    Computes the maximum and mean inverse identity error (in physical units)
    between a displacement field and its inverse.
    Error = || W_inv_disp(x) + W_disp( x + W_inv_disp(x) ) ||_2
    """
    import torch
    import torch.nn.functional as F
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
    
    coords_phys = X_phys + W_inv_disp
    
    # physical_to_normalized_torch_cached expects reversed spacing, origin, direction
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    
    shape_t = torch.tensor(spatial, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    
    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
    
    forward_at_inv_cf = F.grid_sample(torch.movedim(W_disp, -1, 1), coords_norm, padding_mode='zeros', align_corners=True)
    forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)
    
    error = W_inv_disp + forward_at_inv
    
    # Mask out points that evaluate outside the grid (padding artifacts)
    inside_mask = (coords_norm >= -1) & (coords_norm <= 1)
    inside_mask = inside_mask.all(dim=-1, keepdim=True)
    
    boundary_mask = get_boundary_mask(spatial, device, dtype).squeeze(0).squeeze(-1)
    error = error * boundary_mask.unsqueeze(0).unsqueeze(-1) * inside_mask
    
    error_norm = torch.sqrt(torch.sum(error**2, dim=-1))
    return {'max_error': float(error_norm.max().item()), 'mean_error': float(error_norm.sum().item() / (boundary_mask.sum().item() + 1e-8))}


def registration(
    fixed,
    moving,
    type_of_transform='SyNTo',
    aff_metric='mattes',
    aff_sampling=32,
    syn_metric='lncc',
    syn_sampling=4,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.75,
    flow_sigma=3.0,
    total_sigma=0.0,
    verbose=False,
    backend='pytorch',
    initial_transform=None,
    levels=None,
    sampling_percentage=None,
    vgg_layers=[4],
    vgg_mode='lncc_3d',
    vgg_patch_size=32,
    vgg_num_patches=8,
    vgg_lncc_window_size=9,
    optimizer='cfl',
    optimizer_lr=1e-3,
    **kwargs
):
    """
    High-level, image-first registration function matching ants.registration interface.
    
    Parameters
    ----------
    fixed : ANTsImage
        Fixed target image.
    moving : ANTsImage
        Moving source image.
    type_of_transform : str
        Ignored (default 'SyNTo'). Included to match ants.registration signature.
    aff_metric : str
        Metric for affine registration. Supported values:
            - 'mattes_mi' or 'mattes' (default)
            - 'lncc'
            - 'mse'
    aff_sampling : int
        Number of bins for Mattes MI (used when aff_metric is 'mattes' or 'mattes_mi').
    syn_metric : str or list of str or callable
        Metric for SyN registration. Defaults to 'lncc'.
        Supported values:
            - 'lncc': Local Normalized Cross-Correlation.
            - 'mattes_mi' or 'mattes': Mattes Mutual Information.
            - 'mse': Mean Squared Error.
            - 'vgg19': Extractor using VGG19 feature layers.
            - 'dinov2' or 'dinov2_small': DINOv2 ViT-S/14 extractor.
            - 'dinov2_base': DINOv2 ViT-B/14 extractor.
            - 'resnet10': 2D/3D ResNet10 extractor.
            - 'swinunetr' or 'swin_unetr': SwinUNETR extractor.
            - list of metrics: Mix multiple metrics (e.g., ['lncc', 'vgg19']).
            - Custom callable or torch.nn.Module: Custom similarity loss.
    syn_sampling : int
        LNCC radius (window_size = 2 * syn_sampling + 1).
    reg_iterations : list of int or None
        Number of iterations per level for SyN stage.
    affine_iterations : list of int or None
        Number of iterations per level for Affine stage.
    grad_step : float
        CFL voxel bound step size (default 0.75).
    flow_sigma : float
        Standard deviation of Gaussian fluid regularizer (default 3.0).
    total_sigma : float
        Standard deviation of Gaussian elastic regularizer (default 0.0).
    verbose : bool
        If True, prints progress details.
    backend : str
        'pytorch' or 'jax' computational backend.
    initial_transform : str or list of str or ANTsTransform or None
        Optional initial transform(s) to apply to moving image before registration.
    vgg_layers : list of int
        Feature layers to extract from deep extractors (default: [4]).
    vgg_mode : str
        Feature loss computation mode (default: 'lncc_3d'; choices: 'lncc_3d', 'lncc').
    vgg_patch_size : int
        Patch size used for local patch metrics (default: 32).
    vgg_num_patches : int
        Number of patches to sample (default: 8).
    vgg_lncc_window_size : int
        LNCC window size for feature space metrics (default: 9).
    **kwargs : dict
        Additional parameters, including:
            - similarity_metric: alias for syn_metric
            - num_slices: number of slices to project for 2D networks (default: 4)
            - smoothing_sigmas: list of sigmas for pyramid smoothing
    """
    import tempfile
    import ants
    import numpy as np
    if 'similarity_metric' in kwargs:
        syn_metric = kwargs.pop('similarity_metric')

    # 1. Extract physical properties
    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    direction = fixed.direction
    
    # Apply initial transform if provided
    tx_list = []
    initial_grid = kwargs.pop('initial_grid', None)
    
    init_M_phys, init_t_phys = None, None
    
    if initial_grid is not None:
        if dim == 2:
            initial_grid = initial_grid.transpose(0, 2, 1, 3)
        elif dim == 3:
            initial_grid = initial_grid.transpose(0, 3, 2, 1, 4)
    elif initial_transform is not None:
        tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        init_M_phys, init_t_phys = parse_ants_affine(tx_list, dim)
        if init_M_phys is None:
            initial_grid = compute_initial_grid(fixed, moving, tx_list)
            if dim == 2:
                initial_grid = initial_grid.transpose(0, 2, 1, 3)
            elif dim == 3:
                initial_grid = initial_grid.transpose(0, 3, 2, 1, 4)
        if affine_iterations is None:
            affine_iterations = [0]
    moving_reg = moving
    
    # 2. Extract and Normalize numpy arrays
    fi_np = fixed.numpy()
    mi_np = moving_reg.numpy()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    
    # Keep spacing in native X-first order (reversal handled internally by helper functions)
    sp_ordered = spacing
    
    # Parse type_of_transform
    transform_type = 'Affine'
    is_linear_only = False
    
    tot_lower = type_of_transform.lower()
    if tot_lower == 'rigid':
        transform_type = 'Rigid'
        is_linear_only = True
    elif tot_lower == 'translation':
        transform_type = 'Translation'
        is_linear_only = True
    elif tot_lower == 'affine':
        transform_type = 'Affine'
        is_linear_only = True
    elif tot_lower in ['syn', 'synto']:
        transform_type = 'Affine'
        is_linear_only = False
        
    levels_len = len(levels) if levels is not None else (4 if dim == 2 else 3)
    if is_linear_only:
        reg_iterations = [0] * levels_len
    elif reg_iterations is None:
        reg_iterations = [100, 100, 100, 50] if dim == 2 else [100, 100, 50]
        
    inverse_steps = kwargs.get('inverse_steps', 20)
    inverse_method = kwargs.get('inverse_method', 'fixed_point')
    vgg_layers = kwargs.get('vgg_layers', vgg_layers)
    vgg_patch_size = kwargs.get('vgg_patch_size', vgg_patch_size)
    vgg_num_patches = kwargs.get('vgg_num_patches', vgg_num_patches)
    vgg_mode = kwargs.get('vgg_mode', vgg_mode)
    vgg_lncc_window_size = kwargs.get('vgg_lncc_window_size', vgg_lncc_window_size)
        
    # 3. Initialize and fit the model
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    grid_shape_zyx = tuple(reversed(grid_shape))
    if backend == 'pytorch':
        from .syn import SyNTo as SyNToPy
        import torch
        device = 'cpu'
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        
        model = SyNToPy(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed.origin, direction=direction,
            fluid_sigma=flow_sigma, elastic_sigma=total_sigma, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps
        ).to(device)
    elif backend == 'jax':
        from .syn_jax import SyNTo as SyNToJax
        import jax.numpy as jnp
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fixed.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *moving.shape).transpose(perm)
        
        model = SyNToJax(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed.origin, direction=direction,
            fluid_sigma=flow_sigma, elastic_sigma=total_sigma, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    affine_lr_param = kwargs.get('affine_lr', 1e-2)
    smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
    if smoothing_sigmas is None:
        levels_to_use = levels if levels is not None else ([8, 4, 2, 1] if dim == 2 else [4, 2, 1])
        import math
        smoothing_sigmas = [float(math.sqrt(s)) if s > 1 else 0.0 for s in levels_to_use]
        
    if backend == 'pytorch':
        initial_grid_tensor = torch.tensor(initial_grid, dtype=torch.float32, device=device) if initial_grid is not None else None
        model.fit(
            I_tensor, J_tensor,
            levels=levels if levels is not None else ([8, 4, 2, 1] if dim == 2 else [4, 2, 1]),
            epochs_per_level=reg_iterations if reg_iterations is not None else [100, 100, 100, 50],
            affine_epochs=affine_iterations if affine_iterations is not None else [100, 50, 50, 20],
            affine_lr=affine_lr_param,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            lncc_radius=syn_sampling,
            mattes_bins=aff_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size,
            initial_grid=initial_grid_tensor,
            fixed_spacing=fixed.spacing,
            fixed_origin=fixed.origin,
            fixed_direction=fixed.direction,
            moving_spacing=moving.spacing,
            moving_origin=moving.origin,
            moving_direction=moving.direction,
            aff_metric=aff_metric,
            smoothing_sigmas=smoothing_sigmas,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=kwargs.get('use_analytical_gradients', True),
            init_M_phys=init_M_phys,
            init_t_phys=init_t_phys
        )
    else:
        import jax.numpy as jnp
        initial_grid_tensor = jnp.array(initial_grid) if initial_grid is not None else None
        model.fit(
            I_tensor, J_tensor,
            levels=levels if levels is not None else ([8, 4, 2, 1] if dim == 2 else [4, 2, 1]),
            epochs_per_level=reg_iterations if reg_iterations is not None else [100, 100, 100, 50],
            affine_epochs=affine_iterations if affine_iterations is not None else [100, 50, 50, 20],
            affine_lr=affine_lr_param,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            lncc_radius=syn_sampling,
            mattes_bins=aff_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size,
            initial_grid=initial_grid_tensor,
            fixed_spacing=fixed.spacing,
            fixed_origin=fixed.origin,
            fixed_direction=fixed.direction,
            moving_spacing=moving.spacing,
            moving_origin=moving.origin,
            moving_direction=moving.direction,
            aff_metric=aff_metric,
            smoothing_sigmas=smoothing_sigmas,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=kwargs.get('use_analytical_gradients', True),
            init_M_phys=init_M_phys.cpu().numpy() if init_M_phys is not None else None,
            init_t_phys=init_t_phys.cpu().numpy() if init_t_phys is not None else None
        )
    
    # 4. Save displacement fields to temp files to match ANTs file-based transforms
    fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_inv.nii.gz', delete=False).name
    
    affine_file = None
    affine_inv_file = None
    
    if backend == 'pytorch':
        with torch.no_grad():
            if sum(reg_iterations) > 0:
                fixed_shape = fixed.shape
            if hasattr(model, 'warp_l2r'):
                # model.warp_l2r is already the total forward deformable displacement
                total_fwd_deformable = model.warp_l2r.data
                
                # model.warp_r2l is already the total inverse deformable displacement (from moving to fixed space)
                total_inv_deformable = model.warp_r2l.data
                
                if total_fwd_deformable.device.type == 'cuda' or total_fwd_deformable.device.type == 'mps':
                    total_fwd_deformable = total_fwd_deformable.cpu()
                if total_inv_deformable.device.type == 'cuda' or total_inv_deformable.device.type == 'mps':
                    total_inv_deformable = total_inv_deformable.cpu()
                
                warp_l2r_np = total_fwd_deformable.numpy()
                warp_r2l_np = total_inv_deformable.numpy()
                if dim == 2:
                    warp_l2r_np = warp_l2r_np.transpose(0, 2, 1, 3)
                    warp_r2l_np = warp_r2l_np.transpose(0, 2, 1, 3)
                elif dim == 3:
                    warp_l2r_np = warp_l2r_np.transpose(0, 3, 2, 1, 4)
                    warp_r2l_np = warp_r2l_np.transpose(0, 3, 2, 1, 4)
            else:
                warp_l2r_np = np.zeros((1, *fixed.shape, dim), dtype=np.float32)
                warp_r2l_np = np.zeros((1, *fixed.shape, dim), dtype=np.float32)

            
            if hasattr(model, 'affine'):
                # Convert internal grid affine to physical ITK AffineTransform
                T_grid = model.affine.get_matrix().cpu().numpy()
                if verbose:
                    print(f"[pytorch] T_grid:\n", T_grid)
                moving_target = fixed if initial_transform is not None else moving_reg
                M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving_target)
                
                # Save physical forward affine transform to file
                affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
                tx_fwd = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
                tx_fwd.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
                tx_fwd.set_fixed_parameters(np.zeros(dim))
                ants.write_transform(tx_fwd, affine_file)
                
                # Invert physical affine transform and save to file
                affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
                M_phys_inv = np.linalg.inv(M_phys)
                t_phys_inv = - M_phys_inv @ t_phys
                tx_inv = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
                tx_inv.set_parameters(np.concatenate([M_phys_inv.ravel(), t_phys_inv]))
                tx_inv.set_fixed_parameters(np.zeros(dim))
                ants.write_transform(tx_inv, affine_inv_file)
    else:
        # For JAX:
        import jax
        import jax.numpy as jnp
        from .syn_jax import get_affine_matrix_jax, get_physical_grid_jax, physical_to_normalized_jax, jax_grid_sample
        
        if hasattr(model, 'warp_l2r'):
            warp_l2r_np = np.array(model.warp_l2r)
            warp_r2l_np = np.array(model.warp_r2l)
            if dim == 2:
                warp_l2r_np = warp_l2r_np.transpose(0, 2, 1, 3)
                warp_r2l_np = warp_r2l_np.transpose(0, 2, 1, 3)
            elif dim == 3:
                warp_l2r_np = warp_l2r_np.transpose(0, 3, 2, 1, 4)
                warp_r2l_np = warp_r2l_np.transpose(0, 3, 2, 1, 4)
        else:
            warp_l2r_np = np.zeros((1, *fixed.shape, dim), dtype=np.float32)
            warp_r2l_np = np.zeros((1, *fixed.shape, dim), dtype=np.float32)
        
        if hasattr(model, 'affine_params'):
            T_grid = get_affine_matrix_jax(model.affine_params, dim, model.transform_type)
            T_grid = np.array(T_grid)
            if verbose:
                print(f"[jax] T_grid:\n", T_grid)
            moving_target = fixed if initial_transform is not None else moving_reg
            M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving_target)
            
            # Save physical forward affine transform to file
            affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
            tx_fwd = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
            tx_fwd.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
            tx_fwd.set_fixed_parameters(np.zeros(dim))
            ants.write_transform(tx_fwd, affine_file)
            
            # Invert physical affine transform and save to file
            affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
            M_phys_inv = np.linalg.inv(M_phys)
            t_phys_inv = - M_phys_inv @ t_phys
            tx_inv = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
            tx_inv.set_parameters(np.concatenate([M_phys_inv.ravel(), t_phys_inv]))
            tx_inv.set_fixed_parameters(np.zeros(dim))
            ants.write_transform(tx_inv, affine_inv_file)
        
    if sum(reg_iterations) > 0:
        disp_l2r = warp_l2r_np[0].astype(np.float32)
        disp_r2l = warp_r2l_np[0].astype(np.float32)
        
        if dim == 2:
            # Reverse vector components (Y, X) -> (X, Y)
            disp_l2r_t = disp_l2r[..., ::-1].copy()
            disp_r2l_t = disp_r2l[..., ::-1].copy()
        elif dim == 3:
            # Reverse vector components (Z, Y, X) -> (X, Y, Z)
            disp_l2r_t = disp_l2r[..., ::-1].copy()
            disp_r2l_t = disp_r2l[..., ::-1].copy()

        fwd_img = ants.from_numpy(disp_l2r_t, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True)
        inv_img = ants.from_numpy(disp_r2l_t, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True)
        
        ants.image_write(fwd_img, fwd_file)
        ants.image_write(inv_img, inv_file)
        
        if initial_transform is not None:
            if affine_file is not None:
                fwd_transforms = [fwd_file, affine_file] + tx_list
                inv_transforms = tx_list + [affine_file, inv_file]
                whichtoinvert_inv = [True] * len(tx_list) + [True, False]
            else:
                fwd_transforms = [fwd_file] + tx_list
                inv_transforms = tx_list + [inv_file]
                whichtoinvert_inv = [True] * len(tx_list) + [False]
        elif affine_file is not None:
            fwd_transforms = [fwd_file, affine_file]
            inv_transforms = [affine_file, inv_file]
            whichtoinvert_inv = [True, False]
        else:
            fwd_transforms = [fwd_file]
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]
    else:
        if initial_transform is not None:
            if affine_file is not None:
                fwd_transforms = [affine_file] + tx_list
                inv_transforms = tx_list + [affine_file]
                whichtoinvert_inv = [True] * len(tx_list) + [True]
            else:
                fwd_transforms = tx_list
                inv_transforms = tx_list
                whichtoinvert_inv = [True] * len(tx_list)
        elif affine_file is not None:
            fwd_transforms = [affine_file]
            inv_transforms = [affine_file]
            whichtoinvert_inv = [True]
        else:
            fwd_transforms = []
            inv_transforms = []
            whichtoinvert_inv = []
    
    inverse_identity_errors = {}
    if sum(reg_iterations) > 0 and hasattr(model, 'warp_l2r') and hasattr(model, 'warp_l2r_inv'):
        import torch
        if backend == 'pytorch':
            w_l2r = model.warp_l2r.data.cpu()
            w_l2r_inv = model.warp_l2r_inv.data.cpu()
            w_r2l = model.warp_r2l.data.cpu()
            w_r2l_inv = model.warp_r2l_inv.data.cpu()
        else:
            w_l2r = torch.from_numpy(np.array(model.warp_l2r))
            w_l2r_inv = torch.from_numpy(np.array(model.warp_l2r_inv))
            w_r2l = torch.from_numpy(np.array(model.warp_r2l))
            w_r2l_inv = torch.from_numpy(np.array(model.warp_r2l_inv))
            
        inverse_identity_errors['phi_1'] = calculate_inverse_identity_error(w_l2r, w_l2r_inv, fixed.spacing, fixed.origin, fixed.direction)
        inverse_identity_errors['phi_2'] = calculate_inverse_identity_error(w_r2l, w_r2l_inv, fixed.spacing, fixed.origin, fixed.direction)
        
    # 6. Apply transforms to generate warped output images
    warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms)
    warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms, whichtoinvert=whichtoinvert_inv)
    
    ret_dict = {'model': model,
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'syn_losses': list(model.syn_losses) if hasattr(model, 'syn_losses') else [],
        'affine_losses': list(model.affine_losses) if hasattr(model, 'affine_losses') else [],
        'inverse_identity_errors': inverse_identity_errors
    }
    
    if verbose >= 2 and hasattr(model, 'fixed_mid_img'):
        ret_dict['fixed_mid'] = model.fixed_mid_img
        ret_dict['moving_mid'] = model.moving_mid_img
        
    return ret_dict
