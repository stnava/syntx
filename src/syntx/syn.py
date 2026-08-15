"""
syn.py — Symmetric Normalization (SyNTo) & Diffeomorphic Registration Core
============================================================================

This module implements Symmetric Normalization (SyN) registration in PyTorch, featuring:
- Lie Algebra SO(d) parameterization for rigid/affine initial alignment.
- Local Normalized Cross-Correlation (LNCC) with variance floors and Cauchy-Schwarz clamping.
- Deep Feature LNCC incorporating VGG 3D Layer 4 perceptual features.
- Symmetric diffeomorphic warp composition and fixed-point inverse field updates.
- Jacobian determinant regularity checks and topological inverse identity error tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc

from .transform import SyNToTransform


def get_rotation_matrix(omega: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes a 2D or 3D rotation matrix from a Lie Algebra parameterization ($so(2)$ or $so(3)$).

    Uses a first-order Taylor expansion near $\\omega = 0$ to prevent zero-angle gradient locking
    and division-by-zero singularities during automatic differentiation (GEMINI.md Rule 6).

    Parameters
    ----------
    omega : torch.Tensor
        Lie algebra rotation vector (1 element for 2D angle; 3 elements `[w0, w1, w2]` for 3D axis-angle).
    dim : int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    torch.Tensor
        Rotation matrix $R \\in SO(d)$ of shape `(2, 2)` or `(3, 3)`.

    Raises
    ------
    ValueError
        If `dim` is not 2 or 3.
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
        safe_theta2 = torch.where(is_zero, 1e-16, theta2)
        theta = torch.sqrt(safe_theta2)
        
        safe_theta = torch.where(is_zero, 1.0, theta)
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
    """
    Hierarchical Differentiable Linear Transformation Module in PyTorch.

    Parameterizes physical linear transformations using Lie Algebra $SO(d)$ rotation representation
    to eliminate gimbal lock and maintain continuous gradient flow at identity initialization.

    Supported Transformation Hierarchy (`transform_type`):
    - `'Translation'`: $d$-dimensional physical shift vector.
    - `'Rigid'`: Translation + $SO(d)$ Lie algebra rotation.
    - `'Similarity'`: Rigid + isotropic scaling factor $s$.
    - `'Affine'`: Similarity + anisotropic scaling $S$ + upper-triangular shear matrix $Sh$.

    Parameters
    ----------
    dim : int, default=3
        Spatial dimensionality (2 or 3).
    transform_type : str, default='Affine'
        Linear transformation model ('Translation', 'Rigid', 'Similarity', 'Affine').

    Attributes
    ----------
    translation : nn.Parameter
        Translation parameter vector of shape `(dim,)`.
    omega : nn.Parameter
        Lie algebra rotation vector of shape `(dim*(dim-1)//2,)`.
    scale : nn.Parameter or torch.Tensor
        Isotropic scaling factor.
    anisotropic_scale : nn.Parameter or torch.Tensor
        Per-axis scaling factor vector of shape `(dim,)`.
    shear : nn.Parameter or torch.Tensor
        Upper-triangular shear parameter vector.
    """

    def __init__(self, dim: int = 3, transform_type: str = 'Affine'):
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

    def clamp_parameters(self):
        with torch.no_grad():
            if isinstance(self.scale, nn.Parameter):
                self.scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.anisotropic_scale, nn.Parameter):
                self.anisotropic_scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.shear, nn.Parameter):
                self.shear.clamp_(min=-5.0, max=5.0)
            if isinstance(self.omega, nn.Parameter):
                self.omega.clamp_(min=-3.14159265, max=3.14159265)

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


def grid_sample_bspline_torch(
    image: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = 'border',
    align_corners: bool = True
) -> torch.Tensor:
    """
    C1-continuous 3D/2D cubic B-spline grid sampling for PyTorch tensors.
    image: (B, C, H, W) or (B, C, D, H, W)
    grid: (B, H_out, W_out, 2) or (B, D_out, H_out, W_out, 3) in [-1, 1]
    """
    ndim = image.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(f"Only 2D and 3D grid sampling supported, got ndim={ndim}")

    B, C = image.shape[:2]
    device = image.device
    dtype = image.dtype
    spatial_target = grid.shape[1:-1]

    def get_w(u):
        u2 = u * u
        u3 = u2 * u
        w0 = (1.0 - u)**3 / 6.0
        w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
        w2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
        w3 = u3 / 6.0
        return [w0, w1, w2, w3]

    if ndim == 2:
        H, W = image.shape[2:]
        gx, gy = grid[..., 0], grid[..., 1]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        ix, iy = torch.floor(vx), torch.floor(vy)
        ux, uy = vx - ix, vy - iy
        wx, wy = get_w(ux), get_w(uy)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, H * W)

        for ky in range(4):
            jy_raw = (iy + ky - 1).long()
            jy = jy_raw.clamp(0, H - 1)
            valid_y = (jy_raw >= 0) & (jy_raw < H)
            w_y = wy[ky].unsqueeze(1)
            for kx in range(4):
                jx_raw = (ix + kx - 1).long()
                jx = jx_raw.clamp(0, W - 1)
                valid_x = (jx_raw >= 0) & (jx_raw < W)
                w_yx = w_y * wx[kx].unsqueeze(1)
                idx = (jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                sampled_flat = torch.gather(img_flat, 2, idx)
                sampled = sampled_flat.view(B, C, *spatial_target)
                if padding_mode == 'zeros':
                    valid_mask = (valid_y & valid_x).view(B, 1, *spatial_target)
                    sampled = sampled * valid_mask
                out = out + w_yx * sampled
        return out
    else:
        D, H, W = image.shape[2:]
        gx, gy, gz = grid[..., 0], grid[..., 1], grid[..., 2]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        vz = (gz + 1.0) * (D - 1) / 2.0 if align_corners else (gz + 1.0) * D / 2.0 - 0.5
        ix, iy, iz = torch.floor(vx), torch.floor(vy), torch.floor(vz)
        ux, uy, uz = vx - ix, vy - iy, vz - iz
        wx, wy, wz = get_w(ux), get_w(uy), get_w(uz)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, D * H * W)

        for kz in range(4):
            jz_raw = (iz + kz - 1).long()
            jz = jz_raw.clamp(0, D - 1)
            valid_z = (jz_raw >= 0) & (jz_raw < D)
            w_z = wz[kz].unsqueeze(1)
            for ky in range(4):
                jy_raw = (iy + ky - 1).long()
                jy = jy_raw.clamp(0, H - 1)
                valid_y = (jy_raw >= 0) & (jy_raw < H)
                w_zy = w_z * wy[ky].unsqueeze(1)
                for kx in range(4):
                    jx_raw = (ix + kx - 1).long()
                    jx = jx_raw.clamp(0, W - 1)
                    valid_x = (jx_raw >= 0) & (jx_raw < W)
                    w_zyx = w_zy * wx[kx].unsqueeze(1)
                    idx = (jz * (H * W) + jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                    sampled_flat = torch.gather(img_flat, 2, idx)
                    sampled = sampled_flat.view(B, C, *spatial_target)
                    if padding_mode == 'zeros':
                        valid_mask = (valid_z & valid_y & valid_x).view(B, 1, *spatial_target)
                        sampled = sampled * valid_mask
                    out = out + w_zyx * sampled
        return out


def _image_spatial_gradient(image):
    dim = image.dim() - 2
    if dim == 2:
        grad_x = (torch.roll(image, shifts=-1, dims=-1) - torch.roll(image, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(image, shifts=-1, dims=-2) - torch.roll(image, shifts=1, dims=-2)) / 2.0
        return torch.stack([grad_x, grad_y], dim=2)
    elif dim == 3:
        grad_x = (torch.roll(image, shifts=-1, dims=-1) - torch.roll(image, shifts=1, dims=-1)) / 2.0
        grad_y = (torch.roll(image, shifts=-1, dims=-2) - torch.roll(image, shifts=1, dims=-2)) / 2.0
        grad_z = (torch.roll(image, shifts=-1, dims=-3) - torch.roll(image, shifts=1, dims=-3)) / 2.0
        return torch.stack([grad_x, grad_y, grad_z], dim=2)
    return None

class AnalyticalGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='border', align_corners=True):
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.save_for_backward(input, grid)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        
        dim = input.dim() - 2
        spatial_shape = input.shape[2:]
        B, C = input.shape[:2]
        
        # 1. Compute spatial gradients on source input image: dI/dx, dI/dy, dI/dz
        grad_I = _image_spatial_gradient(input) # (B, C, dim, *spatial_shape)
        
        # 2. Sample source gradients at grid lookup coordinates G
        grad_I_flat = grad_I.view(B, C * dim, *spatial_shape)
        grad_I_sampled = F.grid_sample(grad_I_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        grad_I_sampled = grad_I_sampled.view(B, C, dim, *grid.shape[1:-1]) # (B, C, dim, *spatial_grid)
        
        # 3. Inner product with incoming loss gradient grad_output (B, C, *spatial_grid)
        grad_grid = torch.sum(grad_output.unsqueeze(2) * grad_I_sampled, dim=1).movedim(1, -1) # (B, *spatial_grid, dim)
        
        # 4. Apply voxel-to-normalized grid coordinate scaling
        scales = []
        for d in range(dim):
            size = spatial_shape[dim - 1 - d] # X is dim - 1, Y is dim - 2, Z is dim - 3
            s = (size - 1) / 2.0 if align_corners else size / 2.0
            scales.append(s)
        scale_t = torch.tensor(scales, dtype=grad_grid.dtype, device=grad_grid.device)
        grad_grid = grad_grid * scale_t
        
        return None, grad_grid, None, None, None

def grid_sample_nd(input, grid, mode='bilinear', padding_mode='border', align_corners=True, interpolator='linear', use_analytical_gradients=True):
    if interpolator in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor') or mode in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor'):
        mode = 'nearest'
    if interpolator == 'bspline' or mode == 'bspline':
        return grid_sample_bspline_torch(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    if use_analytical_gradients and grid.requires_grad and not input.requires_grad:
        return AnalyticalGridSample.apply(input, grid, mode, padding_mode, align_corners)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)




def compose_grids(grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
    """
    Composes two coordinate grids: grid1 ∘ grid2
    grid1: (B, *spatial, dim)
    grid2: (B, *spatial, dim)
    """
    grid1_cf = torch.movedim(grid1, -1, 1)
    composed_cf = F.grid_sample(grid1_cf, grid2, mode='bilinear', padding_mode='border', align_corners=True)
    return torch.movedim(composed_cf, 1, -1)


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
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
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
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(target_shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
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
    M_phys_zyx, t_phys_zyx = _grid_to_physical_affine_torch_yfirst(T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev)
    
    # Return ZYX physical affine matrices directly to match PyTorch tensor coordinate ordering (Z, Y, X)
    return M_phys_zyx, t_phys_zyx


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
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = - (origin_t @ M) - 1.0
    M_rev = torch.flip(M, dims=[1])
    b_rev = torch.flip(b, dims=[0])
    norm_coords_reversed = flat_phys @ M_rev + b_rev
    return norm_coords_reversed.view(phys_coords.shape)







def prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fixed_spacing, moving_spacing,
    M_phys, t_phys, initial_grid_level,
    interpolator='linear',
    grad_I_curr=None, grad_J_curr=None,
    use_analytical_gradients=True
):
    phi_l2r_phys = X_phys + warp_l2r
    coords_norm = physical_to_normalized_torch_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    I_mid = grid_sample_nd(I_curr, coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)
    
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
        
    J_mid = grid_sample_nd(J_curr, y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)
    
    if grad_I_curr is None:
        grad_I_curr = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=tuple(reversed(fixed_spacing))).squeeze(-2)
    if grad_J_curr is None:
        grad_J_curr = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=tuple(reversed(moving_spacing))).squeeze(-2)
    print("DEBUG PyTorch grad_I_curr max:", float(grad_I_curr.abs().max()))
    
    grad_I_mid_sampled = grid_sample_nd(grad_I_curr.movedim(-1, 1), coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1).contiguous()
    print("DEBUG PT PRE-MATMUL max:", float(grad_I_mid_sampled.abs().max()))
    grad_I_mid_sampled = torch.matmul(grad_I_mid_sampled, fixed_direction_t.t())
    print("DEBUG PT POST-MATMUL max:", float(grad_I_mid_sampled.abs().max()))
    print("DEBUG PT coords_norm max:", float(coords_norm.abs().max()))
    print("DEBUG PyTorch grad_I_curr PRE SAMPLE max:", float(grad_I_curr.abs().max()))
    
    grad_J_mid_sampled = grid_sample_nd(grad_J_curr.movedim(-1, 1), y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1).contiguous()
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

def _spatial_jacobian_nd(field: torch.Tensor, physical_spacing=None, method='central') -> torch.Tensor:
    """Compute the spatial Jacobian of an N-D vector field via central differences or Cubic B-Spline derivatives.
    
    field: (B, *spatial, d) vector field
    Returns: (B, *spatial, d, d) Jacobian tensor J[..., i, j] = ∂field_i / ∂x_j
    """
    dim = field.shape[-1]
    spatial = field.shape[1:-1]
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (s - 1) for s in spatial]
    
    if method == 'bspline':
        # 1D Cubic B-Spline derivative filter [-1/12, -8/12, 0, 8/12, 1/12] (4th-order accurate B-spline derivative)
        grads = []
        for i, sp in enumerate(spacings):
            k_np = np.array([-1/12, -8/12, 0.0, 8/12, 1/12], dtype=np.float32) / sp
            k_t = torch.from_numpy(k_np).to(device=field.device, dtype=field.dtype)
            
            # Conv along spatial dimension i
            pad = [0, 0] + [0, 0] * (len(spatial) - 1 - i) + [2, 2] + [0, 0] * i
            padded = F.pad(field, pad, mode='replicate')
            
            # Transpose to put target dim i at end for 1D conv
            perm = [0] + [j + 1 for j in range(len(spatial)) if j != i] + [i + 1, len(spatial) + 1]
            perm_inv = [0] + [0] * len(spatial) + [len(spatial) + 1]
            for orig_pos, p_val in enumerate(perm[1:-1], start=1):
                perm_inv[p_val] = orig_pos
                
            field_perm = padded.permute(perm)
            orig_shape = field_perm.shape
            flat_in = field_perm.reshape(-1, 1, orig_shape[-2])
            k_view = k_t.view(1, 1, 5)
            conv_out = F.conv1d(flat_in, k_view)
            conv_restored = conv_out.view(orig_shape[0], *orig_shape[1:-2], conv_out.shape[-1], orig_shape[-1])
            g_i = conv_restored.permute(perm_inv)
            grads.append(g_i)
        return torch.stack(grads, dim=-1)
    
    # torch.gradient returns a list of gradients, one per spatial dimension (ij order)
    grads = torch.gradient(field, spacing=spacings, dim=list(range(1, len(spatial) + 1)))
    
    # Keep in internal (y, x) or (z, y, x) ordering convention
    return torch.stack(grads, dim=-1)  # (B, *spatial, d, d)


def update_inverse_field_nd_hybrid_lm(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor, 
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    damping_factor: float = 1.0,
    spacing = None,
    origin = None,
    direction = None,
    X_phys = None
) -> torch.Tensor:
    """
    Damped Levenberg-Marquardt (LM) Hybrid Inverse Solver.
    Bridges 1st-order Fixed-Point iteration and 2nd-order Newton solvers.
    
    Solves [ I + grad(u) + lambda * I ] * delta_v = - ( v + u(y + v) )
    where local spatial damping lambda(y) dynamically ramps up when det(I + grad(u)) < 0.2,
    achieving quadratic Newton convergence in regular regions while maintaining guaranteed
    Fixed-Point stability under large non-linear deformations.
    """
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
        
        max_error_norm = float('inf')
        mean_error_norm = float('inf')
        
        for iteration in range(steps):
            if max_error_norm <= max_error_threshold and mean_error_norm <= mean_error_threshold:
                break
            
            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t)**2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())
            
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
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma, spacing=spacing)
            W_inv_disp = W_inv_disp * boundary_mask
            
        return W_inv_disp
    else:
        return update_inverse_field_nd(W_disp, W_inv_disp, steps=steps, relaxation=relaxation, smoothing_sigma=smoothing_sigma)


def integrate_time_varying_velocity_field(
    velocity_fields,
    dt: float = 0.25,
    mode: str = 'forward',
    solver: str = 'rk4',
    spacing = None,
    origin = None,
    direction = None
):
    """
    Integrates a discretized time-varying velocity field sequence v(x, t) forward or backward in time.
    
    velocity_fields: List[torch.Tensor] or Tensor of shape (T, B, *spatial, d)
    dt: time step size
    mode: 'forward' (t: 0 -> 1) or 'backward' (t: 1 -> 0)
    solver: 'rk4', 'midpoint', or 'euler'
    """
    if isinstance(velocity_fields, torch.Tensor) and velocity_fields.ndim == 5:
        T = velocity_fields.shape[0]
        vel_list = [velocity_fields[i] for i in range(T)]
    else:
        vel_list = list(velocity_fields)
        T = len(vel_list)
        
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
        
        # Continuous displacement field mapping
        phi = torch.zeros_like(vel_list[0])
        
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                coords_phys = X_phys + curr_phi
                coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
                return torch.movedim(
                    F.grid_sample(v_k_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
                )
            
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
            else: # euler
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi
    else:
        # Standard normalized space branch
        # Create identity grid in [-1, 1] for grid_sample coordinate lookup
        grids = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in spatial]
        identity = torch.stack(
            torch.meshgrid(*reversed(grids), indexing='ij')[::-1], dim=-1
        ).unsqueeze(0).expand(B, *spatial, dim)
        
        phi = torch.zeros_like(vel_list[0])
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                # Sample velocity at current deformed position (identity + displacement)
                sample_coords = identity + curr_phi
                return torch.movedim(
                    F.grid_sample(v_k_cf, sample_coords, padding_mode='border', align_corners=True), 1, -1
                )
            
            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi


def update_inverse_field_nd_anderson(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor,
    steps: int = 30,
    m: int = 5,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None
) -> torch.Tensor:
    """
    Anderson-accelerated fixed-point inversion of a displacement field.

    Wraps the standard ITK fixed-point iteration g(v) with Anderson Acceleration
    (Type-I, window size m). At each step, a small (m_k+1)-dimensional constrained
    least-squares problem is solved over the sliding window of recent residuals to
    find an optimal extrapolated iterate, achieving superlinear convergence.

    Parameters
    ----------
    W_disp : torch.Tensor
        Forward displacement field, shape (B, *spatial, dim).
    W_inv_disp : torch.Tensor
        Initial guess for inverse displacement field (same shape).
    steps : int
        Maximum number of iterations.
    m : int
        Anderson window size. Larger m uses more memory but may converge faster.
        Typical values: 3–7.
    smoothing_sigma : float
        Optional Gaussian smoothing sigma (in voxel space) applied to the
        displacement field after each iteration.
    max_error_threshold, mean_error_threshold : float
        Convergence thresholds (ITK parity: while max > thresh || mean > thresh).
    spacing, origin, direction : array-like or None
        Physical space parameters. Required for 3D registration.
    X_phys : torch.Tensor or None
        Pre-computed physical coordinate grid.

    Returns
    -------
    torch.Tensor
        Inverse displacement field, same shape as W_inv_disp.
    """
    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    # Determine physical vs normalized mode
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

    def itk_fixed_point_step(v_curr, iteration):
        """One ITK fixed-point step: g(v) = v - eps * clip(v + u(x + v))"""
        if use_physical:
            coords_phys = X_phys + v_curr
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t)**2, dim=-1, keepdim=True))
        else:
            coords = identity + v_curr
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = v_curr + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale)**2, dim=-1, keepdim=True))

        max_error_norm = float(scaled_norm.max())
        mean_error_norm = float(scaled_norm.mean())

        epsilon = 0.75 if iteration == 0 else 0.5
        update = -error
        clip_threshold = epsilon * max_error_norm
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            torch.ones_like(scaled_norm)
        )
        update = update * clip_scale

        v_new = v_curr + update * epsilon

        if smoothing_sigma > 0.0:
            if use_physical:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma, spacing=spacing)
            else:
                v_new = separable_gaussian_filter(v_new, smoothing_sigma)

        v_new = v_new * boundary_mask

        return v_new, max_error_norm, mean_error_norm

    # --- Anderson Acceleration main loop ---
    # History buffers: store flattened iterates and residuals
    N = W_inv_disp.numel()
    v_k = W_inv_disp.clone()

    # Lists for sliding window (max size m)
    # Store flattened residuals R_j = g(v_j) - v_j and flattened iterates G_j = g(v_j)
    R_history = []  # list of 1D tensors
    G_history = []  # list of 1D tensors

    for iteration in range(steps):
        # Apply one ITK fixed-point step
        g_k, max_err, mean_err = itk_fixed_point_step(v_k, iteration)

        # Convergence check (ITK parity: logical_or)
        if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
            v_k = g_k
            break

        # Residual: r_k = g(v_k) - v_k
        r_k = (g_k - v_k).reshape(-1)

        # Store in sliding window
        R_history.append(r_k)
        G_history.append(g_k.reshape(-1))

        # Trim window to size m+1
        if len(R_history) > m + 1:
            R_history.pop(0)
            G_history.pop(0)

        m_k = len(R_history)

        if m_k < 2:
            # Not enough history to accelerate — use plain fixed-point
            v_k = g_k
        else:
            # Build the difference matrix Delta_R = [r_k - r_{k-1}, r_k - r_{k-2}, ...]
            # and solve the constrained least-squares: min ||sum_j alpha_j r_j||^2  s.t. sum_j alpha_j = 1
            # Equivalent: form R_mat = [r_0, r_1, ..., r_{m_k-1}] and solve
            # (R^T R + reg*I) alpha = R^T 0  s.t. 1^T alpha = 1
            # Use the unconstrained-to-constrained reduction via residual differences.

            # Form the (m_k-1) x N matrix of residual differences
            n_cols = m_k - 1
            # dR[:, j] = R_history[-1] - R_history[j]  for j = 0..m_k-2
            r_newest = R_history[-1]
            dR_cols = []
            for j in range(n_cols):
                dR_cols.append(r_newest - R_history[j])

            # Gram matrix: dR^T dR, shape (n_cols, n_cols)
            # Use dot products for memory efficiency (don't form full N x n_cols matrix)
            gram = torch.zeros(n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(n_cols, device=device, dtype=dtype)
            for i_col in range(n_cols):
                rhs[i_col] = torch.dot(dR_cols[i_col], r_newest)
                for j_col in range(i_col, n_cols):
                    val = torch.dot(dR_cols[i_col], dR_cols[j_col])
                    gram[i_col, j_col] = val
                    gram[j_col, i_col] = val

            # Tikhonov regularization for numerical stability
            gram += 1e-10 * torch.eye(n_cols, device=device, dtype=dtype)

            # Solve for gamma: gram @ gamma = rhs
            # gamma_j are the mixing weights for the residual differences
            try:
                gamma = torch.linalg.solve(gram, rhs)
            except torch.linalg.LinAlgError:
                # Fallback to plain fixed-point if solve fails
                v_k = g_k
                continue

            # Reconstruct the accelerated iterate:
            # v_{k+1} = g_newest - sum_j gamma_j * (dG_j + dR_j)
            # This is the standard Type-I Anderson formulation.
            g_newest = G_history[-1]
            v_new_flat = g_newest.clone()
            for j in range(n_cols):
                dG_j = g_newest - G_history[j]
                v_new_flat = v_new_flat - gamma[j] * (dG_j + dR_cols[j])

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate = v_candidate * boundary_mask

            # --- Safeguard: verify the Anderson iterate actually reduces the residual ---
            # Evaluate composition residual for the candidate
            if use_physical:
                coords_phys_c = X_phys + v_candidate
                coords_norm_c = physical_to_normalized_torch_cached(coords_phys_c, shape_t, spacing_t, origin_t, direction_t)
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_norm_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c / spacing_t)**2).sqrt())
            else:
                coords_c = identity + v_candidate
                fwd_at_c = torch.movedim(
                    F.grid_sample(W_disp_cf, coords_c, padding_mode='border', align_corners=True), 1, -1
                )
                error_c = v_candidate + fwd_at_c
                residual_aa = float(torch.sum((error_c * voxel_scale)**2).sqrt())

            # Compare with the plain fixed-point residual
            residual_fp = float(torch.dot(r_k, r_k).sqrt())

            if residual_aa <= residual_fp * 1.1:
                # Anderson iterate is at least comparable — accept it
                v_k = v_candidate
            else:
                # Anderson overshooting — fall back to plain fixed-point
                v_k = g_k
    else:
        # Loop completed without break — v_k already holds the last iterate
        pass

    return v_k


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
    X_phys = None
) -> torch.Tensor:
    """
    Dimension-agnostic fixed-point inversion of a displacement field.
    Exactly matches ITK's itkInvertDisplacementFieldImageFilter.hxx.
    """
    if method == 'hybrid_lm':
        return update_inverse_field_nd_hybrid_lm(
            W_disp, W_inv_disp, steps=steps, relaxation=relaxation,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )

    if method == 'anderson':
        return update_inverse_field_nd_anderson(
            W_disp, W_inv_disp, steps=steps,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys
        )

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    if X_phys is not None or (spacing is not None and origin is not None and direction is not None):
        # Physical-space branch (used for 3D registration)
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
        W_disp_cf = torch.movedim(W_disp, -1, 1)
        
        # ITK: m_MaxErrorNorm = NumericTraits<RealType>::max()
        # ITK: m_MeanErrorNorm = NumericTraits<RealType>::max()
        max_error_norm = float('inf')
        mean_error_norm = float('inf')
        
        for iteration in range(steps):
            # ITK while-loop: check PREVIOUS iteration's error at loop entry
            if max_error_norm <= max_error_threshold and mean_error_norm <= mean_error_threshold:
                break
            
            # Phase 1: Compose and compute error norms
            # ITK: ComposeDisplacementFieldsImageFilter computes v(y) + u(y + v(y))
            coords_phys = X_phys + W_inv_disp
            coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords_norm, padding_mode='border', align_corners=True), 1, -1
            )
            
            # error = v(y) + u(y + v(y)) — the composition residual
            error = W_inv_disp + forward_at_inv
            
            # Compute scaled norm in voxel space (ITK: displacement * inverseSpacing)
            scaled_norm = torch.sqrt(torch.sum((error / spacing_t)**2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())
            
            # Negate error to get update direction (ITK Phase 1: ItE.Set(-displacement))
            update = -error
            
            # ITK: m_Epsilon = 0.75 if iteration==0 else 0.5
            epsilon = 0.75 if iteration == 0 else 0.5
            
            # Phase 2: Clip and apply update
            # ITK: if (scaledNorm > epsilon * maxErrorNorm) update *= (epsilon * maxErrorNorm / scaledNorm)
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = update * clip_scale
            
            # ITK: update = ItI.Get() + update * epsilon
            W_inv_disp = W_inv_disp + update * epsilon
            
            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma, spacing=spacing)
            
            # ITK: EnforceBoundaryCondition — zero at boundaries
            W_inv_disp = W_inv_disp * boundary_mask
            
        return W_inv_disp
    else:
        # Normalized-space branch (used for 2D tests without physical coordinates)
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)
        
        max_error_norm = float('inf')
        mean_error_norm = float('inf')
        
        for iteration in range(steps):
            if max_error_norm <= max_error_threshold or mean_error_norm <= mean_error_threshold:
                break
            
            coords = identity + W_inv_disp
            forward_at_inv = torch.movedim(
                F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True), 1, -1
            )
            error = W_inv_disp + forward_at_inv
            scaled_norm = torch.sqrt(torch.sum((error * voxel_scale)**2, dim=-1, keepdim=True))
            max_error_norm = float(scaled_norm.max())
            mean_error_norm = float(scaled_norm.mean())
            
            epsilon = 0.75 if iteration == 0 else 0.5
            update = -error
            
            # Clip at epsilon * max_error (matching ITK exactly)
            clip_threshold = epsilon * max_error_norm
            clip_scale = torch.where(
                scaled_norm > clip_threshold,
                clip_threshold / scaled_norm.clamp(min=1e-10),
                torch.ones_like(scaled_norm)
            )
            update = update * clip_scale
            
            W_inv_disp = W_inv_disp + update * epsilon
            
            if smoothing_sigma > 0.0:
                W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma)
            
            W_inv_disp = W_inv_disp * boundary_mask
            
        return W_inv_disp


class AnalyticalLNCC(torch.autograd.Function):
    """Analytically-differentiated Local NCC (CC, not CC²).

    Computes forward CC = cov(I,J) / sqrt(var(I)*var(J)) identical to
    the autograd path in ``local_ncc_loss_nd(..., squared=False)``, but
    manually implements backward() so that PyTorch never builds a memory-
    heavy autograd graph through ``F.avg_pool3d``.  This makes it as fast
    as ``ANTsPseudoLNCC`` on Apple MPS while optimising the true CC loss
    landscape instead of the CC² pseudo-derivative.

    Analytical gradient of -mean(CC) w.r.t. center pixel J_c (analogous
    for I_c by symmetry):

        dCC/dJ_c = (1/N) * 1/sqrt(var_F * var_M) * (F_c - CC * M_c)

    where F_c, M_c are mean-subtracted center-pixel intensities and N is
    the window volume.
    """

    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim

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

        F_centered = I - I_mean
        M_centered = J - J_mean

        I_var = torch.clamp(box_filter(F_centered ** 2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered ** 2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        denom = torch.sqrt(safe_I_var * safe_J_var) + 1e-6
        cc_raw = IJ_cov / denom
        cc = torch.clamp(cc_raw, min=-1.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, cc, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window

        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc)
            ctx.active = None

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, cc, safe_I_var, safe_J_var, mask = ctx.saved_tensors

        inv_denom = 1.0 / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)

        # Analytical derivative of CC w.r.t. center pixel:
        #   dCC/dJ_c = (1/N) / sqrt(sFF * sMM) * (F_c - CC * M_c)
        #   dCC/dI_c = (1/N) / sqrt(sFF * sMM) * (M_c - CC * F_c)
        # Loss is -CC, so negate:
        scale = -(1.0 / ctx.N_window) * inv_denom

        grad_J = scale * (F_centered - cc * M_centered)
        grad_I = scale * (M_centered - cc * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        return grad_I * grad_output, grad_J * grad_output, None, None


class ANTsPseudoLNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim
        
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
        
        F_centered = I - I_mean
        M_centered = J - J_mean
        
        I_var = torch.clamp(box_filter(F_centered**2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered**2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)
        
        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)
        
        # ITK uses CC^2: localCC = sFixedMoving * sFixedMoving / (sFixedFixed * sMovingMoving)
        cc2_raw = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc2 = torch.clamp(cc2_raw, min=0.0, max=1.0)
        
        ctx.save_for_backward(F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window
        
        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc2 * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc2)
            ctx.active = None
            
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask = ctx.saved_tensors
        
        s_FM = IJ_cov
        s_FF = safe_I_var
        s_MM = safe_J_var
        
        sFF_sMM = s_FF * s_MM + 1e-8
        
        # ITK's pseudo-derivative for +CC^2 wrt moving center pixel M_c is:
        # 2/N * cov / (var_F * var_M) * (F_c - cov / var_M * M_c)
        # By symmetry, wrt fixed center pixel F_c is:
        # 2/N * cov / (var_F * var_M) * (M_c - cov / var_F * F_c)
        
        # Since our loss is -CC^2, the gradient of the loss is the negative of this.
        grad_factor = -2.0 * (1.0 / ctx.N_window) * (s_FM / sFF_sMM)
        
        grad_J = grad_factor * (F_centered - (s_FM / (s_MM + 1e-8)) * M_centered)
        grad_I = grad_factor * (M_centered - (s_FM / (s_FF + 1e-8)) * F_centered)
        
        # Scale by the spatial reduction (mean over all pixels)
        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial
            
        return grad_I * grad_output, grad_J * grad_output, None, None


def local_ncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 9,
    use_ants_pseudo_gradient: bool = False,
    squared: bool = False
) -> torch.Tensor:
    r"""
    Computes Local Normalized Cross-Correlation (LNCC) Loss between N-D images $I$ and $J$.

    Formulation & Rule Guardrails (GEMINI.md Rule 2):
    - Sliding Box Filter: Evaluates local mean $\mu_I, \mu_J$, local variance $\text{Var}(I), \text{Var}(J)$,
      and covariance $\text{Cov}(I, J)$ over a window of size `window_size`.
    - Variance Floor (Singularity Prevention): Enforces a variance floor $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$
      to prevent $\frac{1}{\text{Var}(I)}$ analytical autograd derivative spikes in flat intensity or zero-padded background regions.
    - Cauchy-Schwarz $[-1.0, 1.0]$ Clamping: Enforces strictly bounded correlation coefficient
      $\text{CC} = \text{clamp}\left(\frac{\text{Cov}(I, J)}{\sqrt{\text{Var}_{\text{safe}}(I) \text{Var}_{\text{safe}}(J)}}, -1.0, 1.0\right)$
      to eliminate 32-bit floating-point roundoff overflow near sharp boundary edges.

    Parameters
    ----------
    I : torch.Tensor
        First image tensor of shape `(B, 1, *spatial)`.
    J : torch.Tensor
        Second image tensor of shape `(B, 1, *spatial)`.
    mask : torch.Tensor, optional
        Binary mask tensor of shape `(B, 1, *spatial)` identifying active evaluation voxels.
    window_size : int, default=9
        Sliding box-filter window size in voxels.
    use_ants_pseudo_gradient : bool, default=False
        If True, uses ANTs C++ style analytical pseudo-gradient autograd function (`ANTsPseudoLNCC`). Implicitly optimizes CC^2.
    squared : bool, default=False
        If True, optimizes squared LNCC (CC^2) instead of CC. This acts as a multi-modal metric.

    Returns
    -------
    torch.Tensor
        Scalar negative LNCC loss tensor (range `[-1.0, 0.0]`, where `-1.0` indicates perfect alignment).
    """
    device = I.device
    dim = I.dim() - 2
    
    # Adapt window size dynamically if input is smaller than kernel
    min_spatial = min(I.shape[2:])
    if window_size > min_spatial:
        window_size = min_spatial
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)
            
    if use_ants_pseudo_gradient and squared:
        # ANTsPseudoLNCC optimizes CC^2 using the ITK pseudo-derivative formula.
        return ANTsPseudoLNCC.apply(I, J, mask, window_size)
    elif use_ants_pseudo_gradient and not squared:
        # AnalyticalLNCC optimizes true CC with exact analytical gradients.
        # Same speed as ANTsPseudoLNCC (no autograd graph through avg_pool3d).
        return AnalyticalLNCC.apply(I, J, mask, window_size)
            
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
    
    # 1. Non-negative variance enforcement
    I_var = torch.clamp(box_filter((I - I_mean)**2), min=0.0)
    J_var = torch.clamp(box_filter((J - J_mean)**2), min=0.0)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))
    
    # 2. Variance floor to prevent 1/var derivative explosion
    var_floor = 1e-6
    safe_I_var = torch.clamp(I_var, min=var_floor)
    safe_J_var = torch.clamp(J_var, min=var_floor)
    
    if squared:
        cc_metric = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc_metric = torch.clamp(cc_metric, min=0.0, max=1.0)
    else:
        cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)
        cc_metric = torch.clamp(cc_raw, min=-1.0, max=1.0)
    
    if mask is not None:
        active_mask_float = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(dtype=I.dtype)
        return -torch.sum(cc_metric * active_mask_float) / (torch.sum(active_mask_float) + 1e-8)
    else:
        return -torch.mean(cc_metric)


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
        
    x = torch.nan_to_num(torch.clamp(x, min_val, max_val), nan=0.0)
    y = torch.nan_to_num(torch.clamp(y, min_val, max_val), nan=0.0)
    
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
    safe_ratio = torch.clamp(ratio, min=1e-8)
    mi = torch.sum(pxy * torch.log(safe_ratio))
    
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
    
    if warp_field.dim() == dim:
        warp_field = warp_field.unsqueeze(0)

    is_physical = getattr(warp_field, 'is_physical', physical_spacing is not None)
    
    if is_physical:
        if physical_spacing is not None:
            spacings = tuple(float(s) for s in physical_spacing)
        else:
            spacings = tuple(1.0 for _ in range(dim))
            
        grads = torch.gradient(warp_field, spacing=spacings, dim=tuple(range(1, dim + 1)))
        
        if dim == 2:
            # grads[0] is d/dy (spatial axis 1), grads[1] is d/dx (spatial axis 2)
            # warp[..., 0] is u_y, warp[..., 1] is u_x
            du_y_dy = grads[0][..., 0]
            du_y_dx = grads[1][..., 0]
            du_x_dy = grads[0][..., 1]
            du_x_dx = grads[1][..., 1]

            j00 = 1.0 + du_x_dx
            j11 = 1.0 + du_y_dy
            j01 = du_x_dy
            j10 = du_y_dx
            return j00 * j11 - j01 * j10
        elif dim == 3:
            # grads[0]=d/dz, grads[1]=d/dy, grads[2]=d/dx
            # warp[..., 0]=u_z, warp[..., 1]=u_y, warp[..., 2]=u_x
            du_z_dz = grads[0][..., 0]
            du_z_dy = grads[1][..., 0]
            du_z_dx = grads[2][..., 0]

            du_y_dz = grads[0][..., 1]
            du_y_dy = grads[1][..., 1]
            du_y_dx = grads[2][..., 1]

            du_x_dz = grads[0][..., 2]
            du_x_dy = grads[1][..., 2]
            du_x_dx = grads[2][..., 2]

            j00 = 1.0 + du_x_dx
            j01 = du_x_dy
            j02 = du_x_dz

            j10 = du_y_dx
            j11 = 1.0 + du_y_dy
            j12 = du_y_dz

            j20 = du_z_dx
            j21 = du_z_dy
            j22 = 1.0 + du_z_dz

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
    Computes the physical spatial Jacobian determinant map $\\det(J_{\\text{phys}}(x))$ from a displacement field.

    Mathematical Formulation:
    1. Evaluates spatial gradients $\\nabla \\mathbf{u}(x)$ using physical spacing $S$ and direction matrix $D$.
    2. Constructs total spatial deformation gradient matrix $F(x) = I + \\nabla \\mathbf{u}(x)$.
    3. Computes point-wise determinant $\\det(F(x))$. Negative or zero determinants ($\\det(J) \\le 0$)
       indicate topological grid folding and loss of diffeomorphic invertibility.

    Parameters
    ----------
    warp_field : torch.Tensor
        Displacement field tensor of shape `(B, *spatial, dim)` in normalized or physical mm coordinates.
    direction : torch.Tensor or list
        Physical direction cosine matrix of shape `(dim, dim)`.
    spacing : torch.Tensor or list
        Physical voxel spacing vector in mm of shape `(dim,)`.

    Returns
    -------
    torch.Tensor
        Physical Jacobian determinant map of shape `(B, *spatial)`.
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
    Computes the true composed physical inverse identity error map (in mm):
      Error(x) = || disp_fwd(x) + disp_inv(x + disp_fwd(x)) ||

    Parameters:
    - warp_fwd: Forward warp tensor (displacement or total physical coordinate grid)
    - warp_inv: Inverse warp tensor (displacement or total physical coordinate grid)
    - is_displacement: Default True. If True, inputs are treated directly as displacement fields.
                       No heuristic guessing is performed.
    - fwd_is_disp: Explicit override for warp_fwd (defaults to is_displacement if None).
    - inv_is_disp: Explicit override for warp_inv (defaults to is_displacement if None).
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
    disp_inv_cf = torch.movedim(disp_inv, -1, 1)
    disp_inv_sampled_cf = grid_sample_nd(disp_inv_cf, y_norm, mode='bilinear', padding_mode='border')
    disp_inv_sampled = torch.movedim(disp_inv_sampled_cf, 1, -1)

    return torch.norm(disp_fwd + disp_inv_sampled, dim=-1)


class SyNTo(nn.Module):
    """
    Generalized Symmetric Normalization (SyNTo) Registration Model in PyTorch.

    Parameterizes symmetric diffeomorphic deformations via forward and reverse
    velocity/displacement fields, maintaining topology preservation.

    Parameters
    ----------
    dim : int, optional
        Spatial dimensionality (2 or 3). Default 3.
    grid_shape : tuple of int, optional
        Image grid shape in ZYX order. Default (64, 64, 64).
    spacing : list of float, optional
        Voxel spacing in XYZ order. Default 1.0 per dimension.
    origin : list of float, optional
        Image origin in XYZ order. Default 0.0 per dimension.
    direction : Tensor or list, optional
        Direction matrix. Default identity.
    fluid_sigma : float, optional
        Fluid regularization standard deviation. Default 3.0.
    elastic_sigma : float, optional
        Elastic regularization standard deviation. Default 0.0.
    transform_type : str, optional
        Affine transform type ('Affine', 'Rigid', 'Translation'). Default 'Affine'.
    inverse_method : str, optional
        Fixed-point inverse solver ('anderson' or 'fixed_point'). Default 'anderson'.
    inverse_steps : int, optional
        Number of fixed-point inverse solver iterations. Default 30.
    project_inverse : bool, optional
        Whether to enforce symmetric inverse identity projection. Default True.
    projection_frequency : int, optional
        Frequency of inverse projection. Default 5.
    interpolator : str, optional
        Image interpolation method ('linear' or 'nearestNeighbor'). Default 'linear'.
    boundary_suppression_thresh : float or None, optional
        Threshold for boundary gradient suppression. Default None.
    image_grad_clip : float, optional
        Maximum magnitude for image gradient clipping. Default 6.0.
    antisymmetric : bool, optional
        Whether to enforce antisymmetry. Default True.
    use_ants_pseudo_gradient : bool, optional
        Whether to use ANTs-style pseudo-gradient for similarity. Default False.
    """
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, origin=None, direction=None, fluid_sigma=3.0, elastic_sigma=0.0, transform_type='Affine', inverse_method='anderson', inverse_steps=30, in_loop_inv_steps=6, project_inverse=True, projection_frequency=1, interpolator='linear', boundary_suppression_thresh=None, image_grad_clip=0.0, antisymmetric=True, use_ants_pseudo_gradient=False, inv_tolerance=None):
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape
        self.spacing = spacing
        self.origin = origin if origin is not None else [0.0] * dim
        
        if inv_tolerance is None:
            self.inv_tolerance = 0.1 * min(spacing) if spacing is not None else 0.1
        else:
            self.inv_tolerance = inv_tolerance
            
        self.fluid_sigma = fluid_sigma
        self.elastic_sigma = elastic_sigma
        self.transform_type = transform_type
        self.inverse_method = inverse_method
        self.inverse_steps = inverse_steps

        self.in_loop_inv_steps = in_loop_inv_steps
        self.project_inverse = project_inverse
        self.projection_frequency = max(1, projection_frequency)
        self.interpolator = interpolator
        self.boundary_suppression_thresh = boundary_suppression_thresh
        self.image_grad_clip = image_grad_clip
        self.antisymmetric = antisymmetric
        self.use_ants_pseudo_gradient = use_ants_pseudo_gradient
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

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None, border_width=0, **kwargs):
        if fluid_sigma <= 0:
            return m
        device = m.device
        dtype = m.dtype
        dim = self.dim
        # Do not auto-scale alpha by dimension
        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s = 2.0
        
        spatial_shape = m.shape[1:-1]
        pad = 8  # Use reflection padding to prevent Gibbs ringing without zeroing out boundary cortex
        pad_shape = tuple(s + 2 * pad for s in spatial_shape)
        
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
        
        # Apply reflection padding
        pad_tuple = (pad, pad) * dim
        m_padded = torch.nn.functional.pad(m_cf, pad_tuple, mode='reflect')
        
        m_fft = torch.fft.rfftn(m_padded.to(torch.float32), dim=spatial_dims)
        K_bc = K_fourier.unsqueeze(0).unsqueeze(0).to(torch.float32)
        v_fft = m_fft * K_bc
        v_padded = torch.fft.irfftn(v_fft, s=pad_shape, dim=spatial_dims).to(dtype=dtype)
        
        # Crop back to original shape
        if dim == 2:
            v_cf = v_padded[..., pad:-pad, pad:-pad]
            return v_cf.permute(0, 2, 3, 1)
        else:
            v_cf = v_padded[..., pad:-pad, pad:-pad, pad:-pad]
            return v_cf.permute(0, 2, 3, 4, 1)

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """
        Applies Sobolev Green's operator in Discrete Sine Transform Type-I (DST-I) space.
        Analytically enforces exact homogeneous Dirichlet boundary conditions (v = 0 at boundaries).
        """
        if fluid_sigma <= 0:
            return m

        device = m.device
        dtype = m.dtype
        dim = self.dim

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

        # nD DST-I via odd-symmetric FFT extension
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

        # Inverse nD DST-I
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

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """
        Applies Sobolev Green's operator using separable 1D DST-I transforms.

        Instead of building a single massive nD odd-symmetric tensor and calling
        torch.fft.fftn (which expands memory by 2^dim and requires all axes to have
        FFT-friendly prime factorizations), this implementation applies 1D DST-I
        independently along each spatial axis using torch.fft.rfft.

        Mathematically equivalent to _apply_dsti_green_operator but:
        - 8x less peak memory for 3D volumes (no simultaneous nD padding)
        - Each 1D rfft operates on size 2N+2 along one axis only
        - More robust on MPS (Apple Silicon) backend
        """
        if fluid_sigma <= 0:
            return m

        device = m.device
        dtype = m.dtype
        dim = self.dim

        if alpha is not None:
            alpha_val = float(alpha)
        else:
            alpha_val = float(fluid_sigma) / 2.0
        s = 2.0

        spatial_shape = m.shape[1:-1]

        # Build separable eigenvalue filter K_dst
        k_axes = []
        for d in range(dim):
            n_d = spatial_shape[d]
            k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
            lambda_d = 4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)
            k_axes.append(lambda_d)

        k_mesh = torch.meshgrid(*k_axes, indexing='ij')
        lambda_sq = sum(k_j for k_j in k_mesh)
        K_dst = 1.0 / ((1.0 + alpha_val * lambda_sq) ** s)

        # Reshape to channels-first: (B, *spatial, dim) -> (B, dim, *spatial)
        m_cf = m.movedim(-1, 1).to(torch.float32)

        def _dst1_1d(arr, axis):
            """Compute 1D DST-I along a single axis via rfft on odd-symmetric extension."""
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
            
            # Aggressive cleanup to prevent MPS OOM
            del z, rev, padded, fft_1d
            if str(device) == 'mps':
                torch.mps.empty_cache()
            return out

        # Forward separable DST-I: apply 1D DST-I along each spatial axis sequentially
        curr = m_cf
        for d in range(dim):
            axis = 2 + d
            curr = _dst1_1d(curr, axis)

        # Multiply by Green's function filter in DST-I spectral domain
        K_bc = K_dst.unsqueeze(0).unsqueeze(0)
        v_dst = curr * K_bc

        # Inverse separable DST-I: DST-I is its own inverse (up to normalization)
        curr_inv = v_dst
        for d in range(dim):
            axis = 2 + d
            n_d = spatial_shape[d]
            curr_inv = _dst1_1d(curr_inv, axis) * (4.0 / float(n_d + 1))

        if str(device) == 'mps':
            torch.mps.empty_cache()

        return curr_inv.to(dtype=dtype).movedim(1, -1)

    def fit(self, fixed_image, moving_image, levels=[4, 2, 1], epochs_per_level=[100, 100, 50], 
            affine_epochs=[100, 50, 20], affine_lr=1e-2, cfl_voxels=0.15, 
            similarity_metric='lncc', use_analytical_gradients=True,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None,
            vgg_layers=[4], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='lncc_3d',
            vgg_lncc_window_size=9, syn_metric_weights=None, initial_grid=None, interpolator=None, **kwargs):
        """
        Runs the full native pre-alignment and SyN multi-resolution optimization loop.
        fixed_image: (1, 1, *spatial)
        moving_image: (1, 1, *spatial)
        """
        import math
        self.elastic_sigma = float(kwargs.get('elastic_sigma', getattr(self, 'elastic_sigma', 0.0)))
        verbose = kwargs.get('verbose', False)
        optimizer_type = kwargs.get('optimizer_type', 'cfl')
        optimizer_lr = kwargs.get('optimizer_lr', 1e-3)
        lncc_window_size = 2 * lncc_radius + 1
        i_min, i_max = torch.min(fixed_image), torch.max(fixed_image)
        j_min, j_max = torch.min(moving_image), torch.max(moving_image)
        fixed_image = (fixed_image - i_min) / (i_max - i_min + 1e-8)
        moving_image = (moving_image - j_min) / (j_max - j_min + 1e-8)
        
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

        self.moving_shape = moving_image.shape[2:]
        self.moving_spacing = moving_spacing
        self.moving_origin = moving_origin
        self.moving_direction = moving_direction

        
        # Standardize iteration lists to match hierarchy levels length
        if isinstance(epochs_per_level, int):
            epochs_per_level = [epochs_per_level] * len(levels)
        elif len(epochs_per_level) < len(levels):
            epochs_per_level = [0] * (len(levels) - len(epochs_per_level)) + list(epochs_per_level)
        elif len(epochs_per_level) > len(levels):
            epochs_per_level = list(epochs_per_level)[-len(levels):]
            
        if isinstance(affine_epochs, int):
            affine_epochs = [affine_epochs] * len(levels)
        elif len(affine_epochs) < len(levels):
            affine_epochs = [0] * (len(levels) - len(affine_epochs)) + list(affine_epochs)
        elif len(affine_epochs) > len(levels):
            affine_epochs = list(affine_epochs)[-len(levels):]
            
        self.affine_losses = []
        self.syn_losses = []
        self.initial_grid = initial_grid
        
        
        init_M_phys = kwargs.get('init_M_phys', None)
        init_t_phys = kwargs.get('init_t_phys', None)
        # CoM Initialization Selection (FOV vs Foreground CoM based on downsampled Mattes MI)
        if self.initial_grid is None:
            with torch.no_grad():
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
                
                down_shape = tuple(max(8, s // 4) for s in fixed_image.shape[2:])
                I_down = F.interpolate(fixed_image, size=down_shape, mode='trilinear' if dim == 3 else 'bilinear', align_corners=True)
                J_down = F.interpolate(moving_image, size=down_shape, mode='trilinear' if dim == 3 else 'bilinear', align_corners=True)
                down_spacing = [sp * (orig - 1) / (down - 1) if down > 1 else sp for sp, orig, down in zip(fixed_spacing, reversed(fixed_image.shape[2:]), reversed(down_shape))]
                X_down = get_physical_grid_torch(down_shape, down_spacing, fixed_origin, fixed_direction, device=device, dtype=dtype)
                
                def eval_translation(t_candidate):
                    t_candidate_zyx = torch.flip(t_candidate, dims=[-1])
                    y_phys = X_down + t_candidate_zyx
                    y_norm = physical_to_normalized_torch(y_phys, moving_image.shape[2:], moving_spacing, moving_origin, moving_direction)
                    J_warped = grid_sample_nd(J_down, y_norm, padding_mode='zeros', align_corners=True, interpolator='linear')
                    
                    return mattes_mi_loss_nd(J_warped, I_down, num_bins=16).item()
                
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
                if init_M_phys is None:
                    T_phys[:dim, dim] = best_t
                else:
                    # In ITK: y_phys = M_phys @ x_phys + t_phys
                    # Since grid mapping transforms fixed grid coordinates X_norm to moving grid coordinates Y_norm,
                    # T_phys maps fixed physical coords to moving physical coords.
                    T_phys[:dim, :dim] = init_M_phys.to(device=device, dtype=dtype)
                    T_phys[:dim, dim] = init_t_phys.to(device=device, dtype=dtype)
                
                T_init = torch.inverse(H_y) @ T_phys @ H_x
                self.affine.T_init = T_init
        
        # Standardize similarity_metric to a list of metrics
        if isinstance(similarity_metric, str):
            self.metrics = [similarity_metric]
        elif isinstance(similarity_metric, list):
            self.metrics = list(similarity_metric)
        else:
            self.metrics = [similarity_metric]

        self.syn_metric_weights = syn_metric_weights
        self.metric_weights = syn_metric_weights if syn_metric_weights is not None else [1.0] * len(self.metrics)
        self.loss_functions = []
        
        # Determine if analytical gradients are viable
        if True:
            for metric in self.metrics:
                if isinstance(metric, str):
                    m_str = metric.lower()
                    if 'dinov2' in m_str or 'dino' in m_str:
                        print(f"Warning: Metric '{metric}' does not support analytical gradients well. Falling back to autograd.")
                        use_analytical_gradients = False
                        break
        kwargs['use_analytical_gradients'] = use_analytical_gradients
        
        from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
        
        for metric in self.metrics:
            if isinstance(metric, str):
                metric_name_lower = metric.lower()
                if metric_name_lower in ['mattes_mi', 'mattes']:
                    self.loss_functions.append(lambda x, y, mask=None: mattes_mi_loss_nd(x, y, mask=mask, num_bins=mattes_bins))
                elif metric_name_lower in ['lncc', 'cc']:
                    self.loss_functions.append(lambda x, y, mask=None, uag=use_analytical_gradients: local_ncc_loss_nd(x, y, mask=mask, window_size=lncc_window_size, use_ants_pseudo_gradient=uag, squared=False))
                elif metric_name_lower in ['lncc2', 'cc2']:
                    self.loss_functions.append(lambda x, y, mask=None, uag=use_analytical_gradients: local_ncc_loss_nd(x, y, mask=mask, window_size=lncc_window_size, use_ants_pseudo_gradient=uag, squared=True))
                elif metric_name_lower == 'mse':
                    self.loss_functions.append(lambda x, y, mask=None: torch.mean((x - y) ** 2) if mask is None else torch.sum(((x - y) ** 2) * mask) / (mask.sum() + 1e-8))
                elif metric_name_lower in ['vgg19', 'vgg_4_lncc'] or metric_name_lower.startswith('vgg_'):
                    cur_vgg_layers = vgg_layers
                    cur_vgg_mode = vgg_mode
                    if metric_name_lower == 'vgg_4_lncc':
                        cur_vgg_layers = [4]
                        if dim == 3:
                            cur_vgg_mode = 'lncc_3d'
                    elif metric_name_lower.startswith('vgg_'):
                        parts = metric_name_lower.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]
                            if parts[2] == 'lncc' and dim == 3:
                                cur_vgg_mode = 'lncc_3d'
                            elif parts[2] == 'lncc':
                                cur_vgg_mode = 'lncc'
                    extractor = VGG19Extractor(feature_layers=cur_vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=cur_vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower in ['dinov2', 'dinov2_small', 'dino_2_lncc'] or metric_name_lower.startswith('dino_'):
                    cur_vgg_layers = vgg_layers
                    cur_vgg_mode = vgg_mode
                    if metric_name_lower == 'dino_2_lncc':
                        cur_vgg_layers = [2]
                        if dim == 3:
                            cur_vgg_mode = 'lncc_3d'
                    elif metric_name_lower.startswith('dino_'):
                        parts = metric_name_lower.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]
                            if parts[2] == 'lncc' and dim == 3:
                                cur_vgg_mode = 'lncc_3d'
                            elif parts[2] == 'lncc':
                                cur_vgg_mode = 'lncc'
                    extractor = DINOv2Extractor(version='vits14', feature_layers=cur_vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=cur_vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower == 'dinov2_base':
                    extractor = DINOv2Extractor(version='vitb14', feature_layers=vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower in ['resnet10', 'resnet_2_lncc'] or metric_name_lower.startswith('resnet_'):
                    cur_vgg_layers = vgg_layers
                    cur_vgg_mode = vgg_mode
                    if metric_name_lower.startswith('resnet_'):
                        parts = metric_name_lower.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]
                            if parts[2] == 'lncc' and dim == 3:
                                cur_vgg_mode = 'lncc_3d'
                            elif parts[2] == 'lncc':
                                cur_vgg_mode = 'lncc'
                    extractor = ResNet10Extractor(dim=dim, feature_layers=cur_vgg_layers).to(device=device)
                    self.loss_functions.append(FeatureSpaceLoss(
                        extractor=extractor, mode=cur_vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                    ).to(device=device))
                elif metric_name_lower in ['swinunetr', 'swin_unetr', 'swin_2_lncc'] or metric_name_lower.startswith('swin_'):
                    layers = [4] if vgg_layers == [8] else vgg_layers
                    if metric_name_lower.startswith('swin_'):
                        parts = metric_name_lower.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            layers = [int(parts[1])]
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
            self.affine_loss_fn = lambda x, y: mattes_mi_loss_nd(x, y, num_bins=mattes_bins, sampling_percentage=sampling_percentage)
        elif aff_metric.lower() in ['lncc', 'cc']:
            self.affine_loss_fn = lambda x, y, uag=use_analytical_gradients: local_ncc_loss_nd(x, y, window_size=lncc_window_size, use_ants_pseudo_gradient=uag, squared=False)
        elif aff_metric.lower() in ['lncc2', 'cc2']:
            self.affine_loss_fn = lambda x, y, uag=use_analytical_gradients: local_ncc_loss_nd(x, y, window_size=lncc_window_size, use_ants_pseudo_gradient=uag, squared=True)
        elif aff_metric.lower() == 'mse':
            self.affine_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
        else:
            self.affine_loss_fn = self.loss_functions[0]
        
        # Parse smoothing_sigmas
        smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
        if smoothing_sigmas is None:
            import math
            sigmas = [float(math.log2(s)) if s > 1 else 0.0 for s in levels]
        elif isinstance(smoothing_sigmas, (int, float)):
            sigmas = [float(smoothing_sigmas)] * len(levels)
        else:
            sigmas = [float(s) for s in smoothing_sigmas]
            if len(sigmas) != len(levels):
                raise ValueError(f"Length of smoothing_sigmas ({len(sigmas)}) must match levels ({len(levels)})")
                
        # --- 0. Construct Image Pyramids ---
        from .pyramid import build_image_pyramid
        I_pyr = build_image_pyramid(fixed_image, spacing=fixed_spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        J_pyr = build_image_pyramid(moving_image, spacing=moving_spacing, levels=levels, smoothing_sigmas=smoothing_sigmas, sigma_mode='voxel')
        
        if sum(affine_epochs) > 0:
            optimizer = None
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
                        
                if optimizer is None:
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
                        
                        I_sampled = grid_sample_nd(I_curr, coords, padding_mode='zeros', align_corners=True, interpolator=self.interpolator)
                        moving_warped = grid_sample_nd(J_curr, coords_warped, padding_mode='zeros', align_corners=True, interpolator=self.interpolator)
                        min_i, max_i = I_sampled.detach().min(), I_sampled.detach().max()
                        min_j, max_j = moving_warped.detach().min(), moving_warped.detach().max()
                        I_scaled = ((I_sampled - min_i) / (max_i - min_i + 1e-8)) * 2.0 - 1.0
                        moving_scaled = ((moving_warped - min_j) / (max_j - min_j + 1e-8)) * 2.0 - 1.0
                        loss = mattes_mi_loss_core(moving_scaled.flatten(), I_scaled.flatten(), num_bins=mattes_bins)
                    else:
                        grid = self.get_affine_grid(curr_spatial, device)
                        if initial_grid_level is not None:
                            grid = compose_grids(initial_grid_level, grid)
                        moving_warped = grid_sample_nd(J_curr, grid, padding_mode='zeros', align_corners=True, interpolator=self.interpolator)
                        loss = self.affine_loss_fn(moving_warped, I_curr)
                    
                    loss.backward()
                    optimizer.step()
                    self.affine.clamp_parameters()
                    self.affine_losses.append(loss.detach())
                    level_affine_losses.append(loss.detach())
                    if verbose:
                        print(f"[pytorch-fit] Affine Level {level_idx} Epoch {epoch}: loss={loss.item():.6f}")
                    if len(level_affine_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_affine_epochs - 1):
                        recent_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in level_affine_losses[-10:]]
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
                    if m_lower in ['vgg19', 'resnet10', 'dinov2', 'dinov2_small', 'dinov2_base', 'swinunetr', 'swin_unetr'] or any(p in m_lower for p in ['vgg', 'dino', 'resnet', 'swin']):
                        is_deep = True
                elif hasattr(metric, 'extractor') or ('FeatureSpaceLoss' in metric.__class__.__name__):
                    is_deep = True
                    
                if is_degenerate and is_deep:
                    active_loss_functions.append(lambda x, y, uag=use_analytical_gradients: local_ncc_loss_nd(x, y, window_size=lncc_window_size, use_ants_pseudo_gradient=uag))
                    active_metric_names.append('lncc_fallback')
                else:
                    metric_idx = self.metrics.index(metric)
                    active_loss_functions.append(self.loss_functions[metric_idx])
                    active_metric_names.append(metric_name)
            
            # Level-dependent scale-space metric weight schedule
            raw_weights = self.syn_metric_weights if self.syn_metric_weights is not None else getattr(self, 'metric_weights', None)
            if raw_weights is not None and len(raw_weights) > 0 and isinstance(raw_weights[0], (list, tuple, np.ndarray)):
                if level_idx < len(raw_weights):
                    curr_metric_weights = list(raw_weights[level_idx])
                else:
                    curr_metric_weights = list(raw_weights[-1])
            elif raw_weights is not None:
                curr_metric_weights = list(raw_weights)
            else:
                curr_metric_weights = [1.0 / len(self.metrics)] * len(self.metrics)

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
            with torch.no_grad():
                grad_I_curr_level = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=tuple(reversed(curr_spacing_fixed))).squeeze(-2)
                grad_J_curr_level = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=tuple(reversed(curr_spacing_moving))).squeeze(-2)
            
            # Checkpoint warp state at level start for divergence retry
            max_syn_retries = 2
            syn_retry_count = 0
            # Multi-resolution CFL step scaling.
            # The raw shrink_ratio = curr_res / full_res (e.g. 0.25 at 4× downsampling)
            # enforces constant physical step but is very conservative at coarse levels.
            # Using sqrt(shrink_ratio) as a geometric mean heuristic: at 4× downsampling
            # this gives 0.5× (vs 0.25× raw or 4.0× old-buggy-inverted), providing
            # aggressive coarse convergence while preventing fine-resolution grid tearing.
            original_spatial = I_pyr[-1].shape[2:]
            shrink_ratio = float(curr_spatial[0]) / float(original_spatial[0])
            import math
            level_cfl_voxels = float(cfl_voxels) * math.sqrt(shrink_ratio)
            warp_l2r_checkpoint = warp_l2r.detach().clone()
            warp_r2l_checkpoint = warp_r2l.detach().clone()
            warp_l2r_inv_checkpoint = warp_l2r_inv.detach().clone()
            warp_r2l_inv_checkpoint = warp_r2l_inv.detach().clone()
            best_level_loss = float('inf')
            
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
                    M_phys, t_phys, initial_grid_level,
                    interpolator=self.interpolator,
                    grad_I_curr=grad_I_curr_level, grad_J_curr=grad_J_curr_level,
                    use_analytical_gradients=use_analytical_gradients
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
                    print(f"DEBUG PyTorch epoch {epoch} I_mid min/max: {I_mid_det.min().item()} {I_mid_det.max().item()} mean: {I_mid_det.mean().item()} var: {I_mid_det.var().item()}")
                    print(f"DEBUG PyTorch epoch {epoch} J_mid min/max: {J_mid_det.min().item()} {J_mid_det.max().item()} mean: {J_mid_det.mean().item()} var: {J_mid_det.var().item()}")
                    if epoch == 0:
                        torch.save(I_mid_det, "/tmp/pt_imid.pt")
                        torch.save(J_mid_det, "/tmp/pt_jmid.pt")
                        torch.save(in_bounds_mask, "/tmp/pt_mask.pt")
                    print(f"DEBUG PyTorch epoch {epoch} mask sum: {in_bounds_mask.sum().item() if in_bounds_mask is not None else 'None'}")
                    print(f"DEBUG PyTorch weights: {curr_metric_weights}")
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                        try:
                            val_loss = fn(I_mid_det, J_mid_det, mask=in_bounds_mask)
                        except TypeError:
                            val_loss = fn(I_mid_det, J_mid_det)

                        loss += weight * val_loss
                        metric_losses_dict[name] = val_loss.item()
                    
                    loss.backward()
                    loss_val = loss.item()
                    g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                    g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                    print(f"DEBUG PyTorch L{level_idx} E{epoch} g_im max: {g_im.abs().max().item()}, g_jm max: {g_jm.abs().max().item()}")
                        
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)
                    
                    with torch.no_grad():
                        if self.image_grad_clip is not None and self.image_grad_clip > 0:
                            mult = float(self.image_grad_clip)
                            norm_I = torch.sqrt(torch.sum(grad_I_mid_sampled**2, dim=-1, keepdim=True) + 1e-16)
                            norm_J = torch.sqrt(torch.sum(grad_J_mid_sampled**2, dim=-1, keepdim=True) + 1e-16)
                            max_I = mult * norm_I.mean()
                            max_J = mult * norm_J.mean()
                            print(f"DEBUG PyTorch max_I: {max_I.item()}, max_J: {max_J.item()}")
                            grad_I_mid_sampled = torch.where(norm_I > max_I, grad_I_mid_sampled * max_I / norm_I, grad_I_mid_sampled)
                            grad_J_mid_sampled = torch.where(norm_J > max_J, grad_J_mid_sampled * max_J / norm_J, grad_J_mid_sampled)

                        grad_l_raw = (g_im.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                        warp_l2r.grad = grad_l_raw
                        print(f"DEBUG PyTorch L{level_idx} E{epoch} grad_l_raw max: {grad_l_raw.abs().max().item()}")
                        print(f"DEBUG PyTorch L{level_idx} E{epoch} grad_l_raw L2 norm max: {torch.sqrt(torch.sum((grad_l_raw / curr_spacing_fixed_t)**2, dim=-1)).max().item()}")

                        grad_r_raw = (g_jm.movedim(1, -1) * grad_J_mid_sampled).contiguous()
                        warp_r2l.grad = grad_r_raw

                else:
                    loss = 0.0
                    metric_losses_dict = {}
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                        try:
                            val_loss = fn(I_mid, J_mid, mask=in_bounds_mask)
                        except TypeError:
                            val_loss = fn(I_mid, J_mid)

                        loss += weight * val_loss
                        metric_losses_dict[name] = val_loss.item()
                        
                    loss.backward()
                    loss_val = loss.item()
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)

                if isinstance(self.fluid_sigma, (list, tuple)):
                    curr_fluid_var = self.fluid_sigma[min(level_idx, len(self.fluid_sigma) - 1)]
                else:
                    curr_fluid_var = self.fluid_sigma
                curr_fluid_sig = float(curr_fluid_var)
                    
                regularizer = kwargs.get('regularizer', 'gaussian')
                with torch.no_grad():
                    raw_alpha = kwargs.get('sobolev_alpha')
                    if raw_alpha is None:
                        raw_alpha = kwargs.get('alpha')
                    if raw_alpha is None:
                        raw_alpha = curr_fluid_sig / 2.0
                    alpha_sobolev = float(raw_alpha)

                    _fs_raw = kwargs.get('fast_smooth', False)
                    fast_smooth = bool(_fs_raw) if _fs_raw is not None else False

                    if regularizer == 'sobolev':
                        if fast_smooth:
                            # FFT Sobolev Green's operator only (standard mode)
                            grad_l = self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # FFT Sobolev Green's operator + spatial Gaussian post-filter (conservative mode)
                            grad_l = separable_gaussian_filter(self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                            grad_r = separable_gaussian_filter(self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                    elif regularizer in ['dsti', 'dst1', 'dst_i']:
                        if fast_smooth:
                            # FFT DST-I Green's operator only (standard mode)
                            grad_l = self._apply_dsti_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_dsti_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # FFT DST-I Green's operator + spatial Gaussian post-filter (conservative mode)
                            grad_l = separable_gaussian_filter(self._apply_dsti_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                            grad_r = separable_gaussian_filter(self._apply_dsti_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                    elif regularizer == 'dsti1':
                        if fast_smooth:
                            # Separable 1D DST-I Green's operator only (MPS-safe mode)
                            grad_l = self._apply_dsti1_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_dsti1_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # Separable 1D DST-I + spatial Gaussian post-filter
                            grad_l = separable_gaussian_filter(self._apply_dsti1_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                            grad_r = separable_gaussian_filter(self._apply_dsti1_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5)
                    else:
                        if fast_smooth:
                            # Spectral Gaussian: Sobolev Green's with soft alpha (FFT-based)
                            grad_l = self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=curr_fluid_sig / 2.0)
                            grad_r = self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=curr_fluid_sig / 2.0)
                        else:
                            # Spatial Gaussian: separable convolution filter
                            grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, curr_fluid_sig)
                            grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, curr_fluid_sig)

                    # Deformed-space smoothing: warp gradient to deformed config,
                    # smooth there, warp back. This bounds ∇_y δ (gradient in deformed
                    # space) matching ANTs C++ behavior, preventing Eulerian grid folding.
                    smooth_deformed = getattr(self, 'smooth_in_deformed_space', False)
                    if smooth_deformed and getattr(self, 'formulation', 'lagrangian') != 'lagrangian':
                        # Forward warp: reference → deformed via warp_l2r
                        def _deformed_smooth(grad_field, warp_fwd, warp_inv):
                            coords_def = X_phys + warp_fwd
                            coords_def_norm = physical_to_normalized_torch_cached(
                                coords_def, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            # Map gradient to deformed space
                            grad_def = F.grid_sample(
                                grad_field.movedim(-1, 1).contiguous(),
                                coords_def_norm.contiguous(),
                                padding_mode='border', align_corners=True
                            ).movedim(1, -1).contiguous()
                            # Smooth in deformed space (bounds ∇_y δ)
                            grad_def_smooth = separable_gaussian_filter(grad_def, curr_fluid_sig * 0.5)
                            # Map back to reference via inverse warp
                            coords_ref = X_phys + warp_inv
                            coords_ref_norm = physical_to_normalized_torch_cached(
                                coords_ref, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            return F.grid_sample(
                                grad_def_smooth.movedim(-1, 1).contiguous(),
                                coords_ref_norm.contiguous(),
                                padding_mode='border', align_corners=True
                            ).movedim(1, -1).contiguous()
                        grad_l = _deformed_smooth(grad_l, warp_l2r, warp_l2r_inv)
                        grad_r = _deformed_smooth(grad_r, warp_r2l, warp_r2l_inv)



                    grad_l_voxel = grad_l / curr_spacing_fixed_t  # convert to voxel units
                    grad_r_voxel = grad_r / curr_spacing_fixed_t
                    max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max()
                    max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max()
                    
                    if verbose >= 2:
                        print(f"DEBUG PyTorch L{level_idx} E{epoch} max_norm_l: {float(max_norm_l)}, max_norm_r: {float(max_norm_r)}")
                    
                    # Track best loss for divergence detection
                    best_level_loss = min(best_level_loss, float(loss_val))
                    
                    in_loop_inv_steps = self.in_loop_inv_steps if self.inverse_steps > 0 else 0
                    if optimizer_type == 'cfl':
                        # ITK: scaledUpdate = (learningRate / maxNorm) * gradient
                        # gradient is in mm, maxNorm is in voxels, so result is in mm
                        effective_cfl = float(level_cfl_voxels)
                        max_norm_l_safe = max_norm_l if use_analytical_gradients else torch.clamp(max_norm_l, min=1e-4)
                        max_norm_r_safe = max_norm_r if use_analytical_gradients else torch.clamp(max_norm_r, min=1e-4)
                        if max_norm_l > 1e-12:
                            delta_l = (effective_cfl / max_norm_l_safe) * grad_l
                        else:
                            delta_l = torch.zeros_like(grad_l)
                            
                        if max_norm_r > 1e-12:
                            delta_r = (effective_cfl / max_norm_r_safe) * grad_r
                        else:
                            delta_r = torch.zeros_like(grad_r)


                        
                        # Antisymmetric velocity projection: remove common-mode drift
                        # to anchor the geodesic midpoint at the Fréchet mean.
                        # Decomposes (δ_l, δ_r) into antisymmetric (geodesic) and
                        # symmetric (drift) components, then discards the drift.
                        if getattr(self, 'antisymmetric', True):
                            e0 = delta_l + delta_r
                            delta_l = delta_l - 0.5 * e0
                            delta_r = delta_r - 0.5 * e0

                        # syntx default is now lagrangian due to fundamental PyTorch
                        # coordinate-frame smoothing limitations in the Eulerian formulation
                        # that prevent guaranteed diffeomorphic 0.0% grid folding.
                        if getattr(self, 'formulation', 'lagrangian') == 'lagrangian':
                            # Lagrangian Pullback (GEMINI.md): φ_new = φ_old - u ∘ (Id + φ_old)
                            # Uses SUBTRACTION to enforce correct velocity field pullback
                            # direction for gradient descent (delta points uphill).
                            coords_phys_l = X_phys + warp_l2r
                            coords_norm_l = physical_to_normalized_torch_cached(
                                coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            # Pull back the velocity field u (delta_l) to the current configuration
                            delta_l_pb = F.grid_sample(delta_l.movedim(-1, 1).contiguous(), coords_norm_l.contiguous(), padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                            
                            coords_phys_r = X_phys + warp_r2l
                            coords_norm_r = physical_to_normalized_torch_cached(
                                coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            delta_r_pb = F.grid_sample(delta_r.movedim(-1, 1).contiguous(), coords_norm_r.contiguous(), padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                            
                            with torch.no_grad():
                                warp_l2r.sub_(delta_l_pb)
                                warp_r2l.sub_(delta_r_pb)
                        else:
                            # SyN composition (Eulerian right-composition): φ_new = φ_old ∘ (Id - δ) - δ
                            # Note: PyTorch fixed-space smoothing bounds ∇_x δ, not ∇_y δ,
                            # causing 0.01-0.07% grid folding. Use Lagrangian instead.
                            coords_phys_l = X_phys - delta_l
                            coords_norm_l = physical_to_normalized_torch_cached(
                                coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1).contiguous(), coords_norm_l.contiguous(), padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                            warp_l2r.copy_(warp_l2r_sampled - delta_l)
                            
                            coords_phys_r = X_phys - delta_r
                            coords_norm_r = physical_to_normalized_torch_cached(
                                coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1).contiguous(), coords_norm_r.contiguous(), padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                            warp_r2l.copy_(warp_r2l_sampled - delta_r)

                        
                        # Removed ITK-standard Dirichlet zero boundary enforcement after composition
                        # to prevent massive gradient-exploding discontinuities at the boundary.
                        
                        if self.elastic_sigma > 0.0:
                            elastic_sig_val = float(self.elastic_sigma)
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, elastic_sig_val))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, elastic_sig_val))
                            
                        # ITK-style diffeomorphic projection: compute inverse fields
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                        )
                        
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
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
                        
                        
                        
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        if self.project_inverse:
                            warp_l2r.copy_(update_inverse_field_nd(
                                warp_l2r_inv, warp_l2r.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                            warp_r2l.copy_(update_inverse_field_nd(
                                warp_r2l_inv, warp_r2l.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                        
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
                        
                        
                        
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        if self.project_inverse:
                            warp_l2r.copy_(update_inverse_field_nd(
                                warp_l2r_inv, warp_l2r.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                            warp_r2l.copy_(update_inverse_field_nd(
                                warp_r2l_inv, warp_r2l.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                        
                    elif optimizer_type == 'sgd':
                        update_l = -optimizer_lr * grad_l
                        update_r = -optimizer_lr * grad_r
                        
                        warp_l2r.copy_(warp_l2r + update_l)
                        warp_r2l.copy_(warp_r2l + update_r)
                        
                        
                        
                        
                        if self.elastic_sigma > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction
                        )
                        if self.project_inverse:
                            warp_l2r.copy_(update_inverse_field_nd(
                                warp_l2r_inv, warp_l2r.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                            warp_r2l.copy_(update_inverse_field_nd(
                                warp_r2l_inv, warp_r2l.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                                spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                            ))
                    
                    # Enforce exact zero Dirichlet boundary condition after all smoothing and projections
                    
                    
                    
                    if verbose:
                        loss_details = ", ".join([f"{k}={v:.6f}" for k, v in metric_losses_dict.items()])
                        print(f"[pytorch-fit] SyN Level {level_idx} Epoch {epoch}: loss={loss_val:.6f} ({loss_details}), warp_l2r max norm={float(torch.sqrt(torch.sum(warp_l2r**2, dim=-1)).max()):.4f}")
                    if len(level_syn_losses) >= 10:
                        recent_losses = [l.item() if hasattr(l, 'item') else l for l in level_syn_losses[-10:]]
                        # For LNCC, metric values can be noisy. A less strict threshold helps stop tearing.
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-6):
                            if verbose:
                                print(f"[pytorch-fit] SyN Level {level_idx} converged at Epoch {epoch}.")
                            break
            # Post-level divergence detection: if loss diverged beyond 2× running min,
            # restore warp checkpoint and retry the level with halved CFL step (up to 2 retries).
            if len(level_syn_losses) > 5 and curr_syn_epochs > 0:
                best_level_loss = min(float(l) for l in level_syn_losses)
                final_level_loss = float(level_syn_losses[-1])
                # Divergence = loss worsened (increased) by more than |best_loss|.
                # This handles negative losses (e.g. LNCC) correctly.
                loss_worsened = final_level_loss - best_level_loss
                if (loss_worsened > abs(best_level_loss)
                        and syn_retry_count < max_syn_retries):
                    syn_retry_count += 1
                    level_cfl_voxels *= 0.5
                    if verbose:
                        print(f"[pytorch-fit] SyN Level {level_idx} diverged (final={final_level_loss:.6f}, best={best_level_loss:.6f}, worsened_by={loss_worsened:.6f}). Retry {syn_retry_count}/{max_syn_retries} with CFL={level_cfl_voxels:.4f}")
                    # Restore warp checkpoint and re-run the level
                    with torch.no_grad():
                        warp_l2r.data.copy_(warp_l2r_checkpoint)
                        warp_r2l.data.copy_(warp_r2l_checkpoint)
                        warp_l2r_inv = warp_l2r_inv_checkpoint.clone()
                        warp_r2l_inv = warp_r2l_inv_checkpoint.clone()
                    level_syn_losses = []
                    # Re-run the epoch loop with reduced CFL
                    for epoch in range(curr_syn_epochs):
                        if warp_l2r.grad is not None: warp_l2r.grad.zero_()
                        if warp_r2l.grad is not None: warp_r2l.grad.zero_()
                        I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask = prepare_mid_images_and_gradients_torch(
                            warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, I_curr, J_curr,
                            X_phys,
                            fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                            moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                            curr_spacing_fixed, curr_spacing_moving,
                            M_phys, t_phys, initial_grid_level,
                            interpolator=self.interpolator,
                            grad_I_curr=grad_I_curr_level, grad_J_curr=grad_J_curr_level
                        )
                        if True:
                            I_mid_det = I_mid.detach().requires_grad_(True)
                            J_mid_det = J_mid.detach().requires_grad_(True)
                            loss = 0.0
                            for name, fn, weight in zip(active_metric_names, active_loss_functions, self.metric_weights):
                                try:
                                    val_loss = fn(J_mid_det, I_mid_det, mask=in_bounds_mask)
                                except TypeError:
                                    val_loss = fn(J_mid_det, I_mid_det)
                                loss += weight * val_loss
                            loss.backward()
                            loss_val = loss.item()
                            g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                            g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                            warp_l2r.grad = (g_im.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                            warp_r2l.grad = (g_jm.movedim(1, -1) * grad_J_mid_sampled).contiguous()
                        else:
                            loss = 0.0
                            for name, fn, weight in zip(active_metric_names, active_loss_functions, self.metric_weights):
                                try:
                                    val_loss = fn(J_mid, I_mid, mask=in_bounds_mask)
                                except TypeError:
                                    val_loss = fn(J_mid, I_mid)
                                loss += weight * val_loss
                            loss.backward()
                            loss_val = loss.item()
                            level_syn_losses.append(loss_val)
                        with torch.no_grad():
                            grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, self.fluid_sigma)
                            grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, self.fluid_sigma)
                            # Gradient outlier clamping (same as main loop)
                            grad_l_norm = torch.sqrt(torch.sum(grad_l**2, dim=-1, keepdim=True) + 1e-16)
                            grad_r_norm = torch.sqrt(torch.sum(grad_r**2, dim=-1, keepdim=True) + 1e-16)
                            grad_l_ref = grad_l_norm.mean()
                            grad_r_ref = grad_r_norm.mean()
                            max_allowed_l = 8.0 * grad_l_ref
                            max_allowed_r = 8.0 * grad_r_ref
                            grad_l = torch.where(grad_l_norm > max_allowed_l, grad_l * max_allowed_l / grad_l_norm, grad_l)
                            grad_r = torch.where(grad_r_norm > max_allowed_r, grad_r * max_allowed_r / grad_r_norm, grad_r)
                            grad_l_voxel = grad_l / curr_spacing_fixed_t
                            grad_r_voxel = grad_r / curr_spacing_fixed_t
                            max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max()
                            max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max()
                            in_loop_inv_steps = self.in_loop_inv_steps if self.inverse_steps > 0 else 0
                            effective_cfl = float(level_cfl_voxels)
                            
                            # Analytical gradients are naturally tiny (1e-8) due to 1/N scaling. 
                            # We must not clamp them to 1e-4 or the CFL step will be suppressed.
                            max_norm_l_safe = max_norm_l
                            max_norm_r_safe = max_norm_r
                            
                            delta_l = (effective_cfl / max_norm_l_safe) * grad_l if max_norm_l > 1e-12 else torch.zeros_like(grad_l)
                            delta_r = (effective_cfl / max_norm_r_safe) * grad_r if max_norm_r > 1e-12 else torch.zeros_like(grad_r)
                            
                            print(f"DEBUG delta_l max: {delta_l.abs().max().item():.6f}, max_norm_l: {max_norm_l.item():.2e}")
                            
                            e0 = delta_l + delta_r
                            delta_l = delta_l - 0.5 * e0
                            delta_r = delta_r - 0.5 * e0
                            coords_phys_l = X_phys - delta_l
                            coords_norm_l = physical_to_normalized_torch_cached(coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t)
                            warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
                            warp_l2r.copy_(warp_l2r_sampled - delta_l)
                            coords_phys_r = X_phys - delta_r
                            coords_norm_r = physical_to_normalized_torch_cached(coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t)
                            warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
                            warp_r2l.copy_(warp_r2l_sampled - delta_r)
                            
                            
                            if self.elastic_sigma > 0.0:
                                warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma))
                                warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma))
                            warp_l2r_inv = update_inverse_field_nd(warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method, spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01)
                            warp_r2l_inv = update_inverse_field_nd(warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method, spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01)
                            if self.project_inverse:
                                warp_l2r.copy_(update_inverse_field_nd(warp_l2r_inv, warp_l2r.detach(), steps=in_loop_inv_steps, method=self.inverse_method, spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01))
                                warp_r2l.copy_(update_inverse_field_nd(warp_r2l_inv, warp_r2l.detach(), steps=in_loop_inv_steps, method=self.inverse_method, spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01))
                        
                        # Removed exact zero Dirichlet boundary enforcement after all smoothing and projections
                        # because multiplying a smoothed displacement field by a binary mask creates a massive
                        # discontinuity at the boundary (e.g., from 8.0 to 0.0 in one voxel), which explodes
                        # the spatial gradient and forces the Jacobian determinant heavily negative.
                        
                        if len(level_syn_losses) >= 10:
                            recent_losses = level_syn_losses[-10:]
                            if check_convergence(recent_losses, window_size=10, slope_threshold=0.0):
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
            w_l2r_inv = w_l2r_inv_interp
            w_r2l_inv = w_r2l_inv_interp
            
            X_phys = get_physical_grid_torch(self.grid_shape, fixed_spacing, fixed_origin, fixed_direction, device=device, dtype=dtype)
            
            # Pre-compute normalization tensors for composition
            comp_shape_t = torch.tensor(list(self.grid_shape), device=device, dtype=dtype)
            comp_spacing_t = torch.tensor(list(reversed(fixed_spacing)), device=device, dtype=dtype)
            comp_origin_t = torch.tensor(list(reversed(fixed_origin)), device=device, dtype=dtype)
            comp_direction_t = torch.tensor(np.asarray(fixed_direction)[::-1, ::-1].copy(), device=device, dtype=dtype)
            
            # Preserve uncomposed half-warp fields for midpoint image export.
            # w_l2r maps midpoint→fixed, w_r2l maps midpoint→(affine)moving.
            # These are destroyed by the full composition below.
            self.midpoint_warp_l2r = nn.Parameter(w_l2r.clone(), requires_grad=False)
            self.midpoint_warp_l2r.is_physical = True
            self.midpoint_warp_r2l = nn.Parameter(w_r2l.clone(), requires_grad=False)
            self.midpoint_warp_r2l.is_physical = True
            
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
            
            self.warp_l2r_inv = nn.Parameter(algebraic_inv.clone())
            self.warp_l2r_inv.is_physical = True
            
            self.warp_r2l = nn.Parameter(algebraic_inv.clone())
            self.warp_r2l.is_physical = True
            
            self.warp_r2l_inv = nn.Parameter(self.warp_l2r.data.clone())
            self.warp_r2l_inv.is_physical = True
            
            # Convert all logged losses to floats in a single batch
            self.affine_losses = [l.item() if hasattr(l, 'item') else float(l) for l in self.affine_losses]
            self.syn_losses = [l.item() if hasattr(l, 'item') else float(l) for l in self.syn_losses]

            # Free temporary pyramid & optimizer buffers
            if 'I_pyr' in locals(): del I_pyr
            if 'J_pyr' in locals(): del J_pyr
            dev_str = str(getattr(device, 'type', str(device))).lower()
            gc.collect()
            if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()


    def forward(self, moving_image, fixed_image=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        """
        Warps the moving image using the affine pre-alignment and dense forward field.
        Accepts either an ants.ANTsImage or a torch.Tensor.
        """
        import ants
        from .spatial import image_to_tensor
        is_ants = isinstance(moving_image, ants.ANTsImage)
        device = self.warp_l2r.device
        dtype = self.warp_l2r.dtype
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))

        if is_ants:
            ref_ants = moving_image
            moving_tensor = image_to_tensor(moving_image, device=device)
            if moving_spacing is None: moving_spacing = moving_image.spacing
            if moving_origin is None: moving_origin = moving_image.origin
            if moving_direction is None: moving_direction = moving_image.direction
        else:
            ref_ants = None
            moving_tensor = moving_image
        
        # Permute input to ZYX order
        moving_image_zyx = moving_tensor.permute(perm)
        
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
        
        T_grid = self.affine.get_matrix()
        moving_shape_xyz = tuple(reversed(moving_image_zyx.shape[2:]))
        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, spatial_shape, spacing, origin, direction,
            moving_shape_xyz, moving_spacing, moving_origin, moving_direction
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
            
        warped_zyx = grid_sample_nd(moving_image_zyx, composed_grid, padding_mode='zeros', align_corners=True, interpolator=self.interpolator)
        warped_xyz = warped_zyx.permute(perm)
        if is_ants:
            arr_np = warped_xyz.squeeze(0).squeeze(0).detach().cpu().numpy()
            dir_np = direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else np.asarray(direction)
            return ants.from_numpy(arr_np, origin=origin, spacing=spacing, direction=dir_np)
        return warped_xyz

    def forward_inverse(self, fixed_image, moving_shape=None, moving_spacing=None, moving_origin=None, moving_direction=None):
        """
        Warps the fixed image into moving space using the inverse mapping.
        Accepts either an ants.ANTsImage or a torch.Tensor.
        """
        import ants
        from .spatial import image_to_tensor
        is_ants = isinstance(fixed_image, ants.ANTsImage)
        device = self.warp_r2l.device
        dtype = self.warp_r2l.dtype
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))

        if is_ants:
            fixed_tensor = image_to_tensor(fixed_image, device=device)
            fixed_spacing = fixed_image.spacing
            fixed_origin = fixed_image.origin
            fixed_direction = fixed_image.direction
        else:
            fixed_tensor = fixed_image
            fixed_spacing = self.spacing
            fixed_origin = self.origin
            fixed_direction = self.direction
        
        # Permute input to ZYX order
        fixed_image_zyx = fixed_tensor.permute(perm)
        
        fixed_shape = fixed_image_zyx.shape[2:]
        spacing = fixed_spacing if fixed_spacing is not None else [1.0] * dim
        origin = fixed_origin if fixed_origin is not None else [0.0] * dim
        direction = fixed_direction if fixed_direction is not None else torch.eye(dim, device=device, dtype=dtype)
        
        # Moving properties define output space
        if moving_shape is None: moving_shape = getattr(self, 'moving_shape', self.grid_shape)
        if moving_spacing is None: moving_spacing = getattr(self, 'moving_spacing', spacing)
        if moving_origin is None: moving_origin = getattr(self, 'moving_origin', origin)
        if moving_direction is None: moving_direction = getattr(self, 'moving_direction', direction)

        Y_phys = get_physical_grid_torch(moving_shape, moving_spacing, moving_origin, moving_direction, device=device, dtype=dtype)
        
        warp_resampled = F.interpolate(
            torch.movedim(self.warp_r2l, -1, 1), 
            size=Y_phys.shape[1:-1], 
            mode='bilinear' if dim == 2 else 'trilinear', 
            align_corners=True
        )
        warp_resampled = torch.movedim(warp_resampled, 1, -1)
        
        phi_r2l_phys = Y_phys + warp_resampled

        T_grid = self.affine.get_matrix()
        T_inv = torch.linalg.inv(T_grid)
        fixed_shape_xyz = tuple(reversed(fixed_shape))
        M_phys_inv, t_phys_inv = grid_to_physical_affine_torch(
            T_inv, moving_shape, moving_spacing, moving_origin, moving_direction,
            fixed_shape_xyz, spacing, origin, direction
        )
        
        x_phys = phi_r2l_phys @ M_phys_inv.t() + t_phys_inv
        composed_grid = physical_to_normalized_torch(x_phys, fixed_shape, spacing, origin, direction)
        
        warped_zyx = grid_sample_nd(fixed_image_zyx, composed_grid, padding_mode='zeros', align_corners=True, interpolator=self.interpolator)
        warped_xyz = warped_zyx.permute(perm)

        if is_ants:
            arr_np = warped_zyx.squeeze(0).squeeze(0).detach().cpu().numpy()
            dir_np = moving_direction.detach().cpu().numpy() if isinstance(moving_direction, torch.Tensor) else np.asarray(moving_direction)
            return ants.from_numpy(arr_np, origin=moving_origin, spacing=moving_spacing, direction=dir_np)
        return warped_xyz



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
    Parses a single ANTs affine transform (path string or ANTsTransform) into M_phys and t_phys tensors.
    Takes into account the center of rotation C as per rule:
    t_new = t + C - M @ C
    """
    import ants
    import numpy as np
    import torch
    
    if not isinstance(tx_list, (list, tuple)):
        tx_list = [tx_list]
    if len(tx_list) == 0:
        return None, None
    M_composed = np.eye(dim, dtype=np.float32)
    t_composed = np.zeros(dim, dtype=np.float32)
    parsed_any = False

    for tx_item in tx_list:
        tx = None
        try:
            if hasattr(tx_item, 'parameters') and hasattr(tx_item, 'fixed_parameters'):
                tx = tx_item
            elif isinstance(tx_item, str):
                try:
                    tx = ants.read_transform(tx_item)
                except Exception:
                    continue
        except Exception:
            continue

        if tx is None:
            continue

        params = tx.parameters
        fixed_params = tx.fixed_parameters

        if len(params) == 12 and dim == 3:
            M = np.array(params[:9], dtype=np.float32).reshape(3, 3)
            t = np.array(params[9:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 3 else np.zeros(3, dtype=np.float32)
        elif len(params) == 6 and dim == 2:
            M = np.array(params[:4], dtype=np.float32).reshape(2, 2)
            t = np.array(params[4:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 2 else np.zeros(2, dtype=np.float32)
        elif len(params) == dim:  # TranslationTransform (2D: 2, 3D: 3)
            M = np.eye(dim, dtype=np.float32)
            t = np.array(params, dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == dim else np.zeros(dim, dtype=np.float32)
        else:
            continue

        t_new = t + C - M @ C
        t_composed = M @ t_composed + t_new
        M_composed = M @ M_composed
        parsed_any = True

    if not parsed_any:
        return None, None

    M_phys = torch.from_numpy(M_composed).to(torch.float32)
    t_phys = torch.from_numpy(t_composed).to(torch.float32)
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
        
    normalized_grid_flat = np.stack(normalized_coords[::-1], axis=-1)
    
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
    
    # physical_to_normalized_torch_cached expects reversed spacing, origin, direction
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    
    shape_t = torch.tensor(spatial, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    
    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
    
    forward_at_inv_cf = F.grid_sample(torch.movedim(W_disp, -1, 1), coords_norm, padding_mode='border', align_corners=True)
    forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)
    
    error = W_inv_disp + forward_at_inv
    
    # Mask out points that evaluate outside the grid (padding artifacts)
    inside_mask = (coords_norm >= -1) & (coords_norm <= 1)
    inside_mask = inside_mask.all(dim=-1, keepdim=True)
    
    boundary_mask = get_boundary_mask(spatial, device, dtype).squeeze(0).squeeze(-1)
    error = error * boundary_mask.unsqueeze(0).unsqueeze(-1) * inside_mask
    
    error_norm = torch.sqrt(torch.sum(error**2, dim=-1))
    return {
        'max_error': float(error_norm.max().item()),
        'mean_error': float(error_norm.sum().item() / (boundary_mask.sum().item() + 1e-8)),
        'error_map': error_norm.squeeze(0)
    }


def registration(
    fixed,
    moving,
    type_of_transform='SyNTo',
    aff_metric='mattes',
    aff_sampling=32,
    syn_metric='lncc',
    syn_sampling=2,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.50,
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
    project_inverse=True,
    projection_frequency=1,
    interpolator='linear',
    inverse_method='anderson',
    inverse_steps=30,
    inv_tolerance=None,
    cfl_momentum=None,
    multipoint_loss=None,
    fast_smooth=None,
    n_time_steps=None,
    n_steps=None,
    antisymmetric=True,
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
    type_of_transform : str, optional
        Transform descriptor (default 'SyNTo'). Included to match ants.registration signature.
    aff_metric : str, optional
        Metric for affine registration ('mattes', 'mattes_mi', 'lncc', 'mse'). Default 'mattes'.
    aff_sampling : int, optional
        Number of bins for Mattes MI when aff_metric is 'mattes'. Default 32.
    syn_metric : str or list of str or callable, optional
        Similarity metric ('lncc', 'mattes_mi', 'vgg19', etc.). Default 'lncc'.
    syn_sampling : int, optional
        LNCC radius (window_size = 2 * syn_sampling + 1). Default 2.
    reg_iterations : list of int or None, optional
        Number of iterations per level for SyN stage. Default [150, 150, 0].
    affine_iterations : list of int or None, optional
        Number of iterations per level for Affine stage. Default [100, 50, 20].
    grad_step : float, optional
        CFL voxel bound step size. Default 0.25.
    flow_sigma : float, optional
        Standard deviation of Gaussian fluid regularizer. Default 3.0.
    total_sigma : float, optional
        Standard deviation of Gaussian elastic regularizer. Default 0.0.
    verbose : bool, optional
        If True, prints progress details. Default False.
    backend : str, optional
        Computational backend ('pytorch' or 'jax'). Default 'pytorch'.
    initial_transform : str or list of str or ANTsTransform or None, optional
        Optional initial transform(s) to apply before registration. Default None.
    levels : list of int or None, optional
        Multi-resolution pyramid downsampling factors. Default [4, 2, 1].
    sampling_percentage : float or None, optional
        Sampling percentage for Mattes MI affine evaluation. Default None.
    vgg_layers : list of int, optional
        Feature layers to extract for deep metrics. Default [4].
    vgg_mode : str, optional
        Deep feature loss mode ('lncc_3d' or 'lncc'). Default 'lncc_3d'.
    vgg_patch_size : int, optional
        Patch size for local feature metrics. Default 32.
    vgg_num_patches : int, optional
        Number of patches to sample. Default 8.
    vgg_lncc_window_size : int, optional
        LNCC window size for feature metrics. Default 9.
    optimizer : str, optional
        Deformable optimizer ('cfl' or 'adam'). Default 'cfl'.
    optimizer_lr : float, optional
        Learning rate for Adam optimizer if used. Default 1e-3.
    project_inverse : bool, optional
        Whether to project inverse displacement field. Default True.
    projection_frequency : int, optional
        Frequency of inverse projection. Default 5.
    interpolator : str, optional
        Image interpolator ('linear' or 'nearestNeighbor'). Default 'linear'.
    inverse_method : str, optional
        Inverse fixed-point solver method ('anderson' or 'fixed_point'). Default 'anderson'.
    inverse_steps : int, optional
        Number of fixed-point inverse solver steps. Default 30.
    cfl_momentum : float or None, optional
        Present for API consistency with syntx.tvf() / syntx.syngs(). Not natively used by SyNTo.
    multipoint_loss : list of float or None, optional
        Present for API consistency with syntx.tvf() / syntx.syngs(). Not natively used by SyNTo.
    fast_smooth : bool or None, optional
        Present for API consistency with syntx.tvf() / syntx.syngs(). Not natively used by SyNTo.
    n_time_steps : int or None, optional
        Present for API consistency with syntx.tvf(). Not natively used by SyNTo.
    n_steps : int or None, optional
        Present for API consistency with syntx.syngs(). Not natively used by SyNTo.
    **kwargs : dict
        Additional parameters, including:
            - similarity_metric: alias for syn_metric
            - num_slices: number of slices to project for 2D networks (default: 4)
            - smoothing_sigmas: list of sigmas for pyramid smoothing

    Returns
    -------
    dict
        Same format as ants.registration:
            - 'warpedmovout': ANTsImage (moving warped to fixed space)
            - 'warpedfixout': ANTsImage (fixed warped to moving space)
            - 'fwdtransforms': list of str (file paths to forward transforms)
            - 'invtransforms': list of str (file paths to inverse transforms)
            - 'whichtoinvert_inv': list of bool
            - 'model': SyNTo model object
            - 'provenance': dict

    Examples
    --------
    >>> import syntx
    >>> reg = syntx.syn(fixed=fi, moving=mi)
    >>> warped = reg['warpedmovout']
    >>> transforms = reg['fwdtransforms']
    """
    import tempfile
    import ants
    import numpy as np
    if 'similarity_metric' in kwargs:
        syn_metric = kwargs.pop('similarity_metric')
    syn_metric_weights = kwargs.pop('syn_metric_weights', None)

    # 1. Extract physical properties
    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    direction = fixed.direction
    
    if inv_tolerance is None:
        inv_tolerance = 0.1 * min(spacing)
    
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
    moving_reg = moving
    
    # 2. Winsorize and Normalize numpy arrays
    fi_np = fixed.numpy()
    mi_np = moving_reg.numpy()
    
    # Winsorize intensity outliers (matches ANTs winsorize_image_intensities)
    winsorize_quantiles = kwargs.get('winsorize_quantiles', None)
    if winsorize_quantiles is not None:
        lo_f, hi_f = np.quantile(fi_np[fi_np > 0], winsorize_quantiles) if (fi_np > 0).any() else (fi_np.min(), fi_np.max())
        fi_np = np.clip(fi_np, lo_f, hi_f)
        lo_m, hi_m = np.quantile(mi_np[mi_np > 0], winsorize_quantiles) if (mi_np > 0).any() else (mi_np.min(), mi_np.max())
        mi_np = np.clip(mi_np, lo_m, hi_m)
    
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
        
    if isinstance(affine_iterations, int):
        affine_iterations = [affine_iterations]
    if isinstance(reg_iterations, int):
        reg_iterations = [reg_iterations]

    if levels is None:
        if reg_iterations is not None or affine_iterations is not None:
            num_levels = max(len(reg_iterations) if reg_iterations else 0, len(affine_iterations) if affine_iterations else 0)
            levels_to_use = [2**i for i in range(num_levels)][::-1] if num_levels > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
        else:
            levels_to_use = [4, 2, 1] if dim == 3 else [8, 4, 2, 1]
    else:
        levels_to_use = levels

    levels_len = len(levels_to_use)
    if is_linear_only:
        reg_iterations = [0] * levels_len
    elif reg_iterations is None:
        reg_iterations = [100, 100, 50] if dim == 3 else [100, 100, 100, 50]
        
    if affine_iterations is None:
        affine_iterations = [100, 50, 20] if dim == 3 else [100, 100, 50, 20]
        
    inverse_steps = kwargs.get('inverse_steps', inverse_steps)
    inverse_method = kwargs.get('inverse_method', inverse_method)
    vgg_layers = kwargs.get('vgg_layers', vgg_layers)
    vgg_patch_size = kwargs.get('vgg_patch_size', vgg_patch_size)
    vgg_num_patches = kwargs.get('vgg_num_patches', vgg_num_patches)
    vgg_mode = kwargs.get('vgg_mode', vgg_mode)
    vgg_lncc_window_size = kwargs.get('vgg_lncc_window_size', vgg_lncc_window_size)
        
    boundary_suppression_thresh = kwargs.get('boundary_suppression_thresh', None)
    image_grad_clip = kwargs.get('image_grad_clip', 6.0)
        
    # Convert flow_sigma/total_sigma from ITK variance convention to actual sigma.
    # ANTs/ITK uses SetVariance(v) where v = σ², so σ = √v.
    # Our separable_gaussian_filter takes σ directly.
    import math
    if isinstance(flow_sigma, (list, tuple)):
        fluid_sigma_actual = [math.sqrt(s) if s > 0 else 0.0 for s in flow_sigma]
    else:
        fluid_sigma_actual = math.sqrt(flow_sigma) if flow_sigma > 0 else 0.0
    elastic_sigma_actual = math.sqrt(total_sigma) if total_sigma > 0 else 0.0
    
    # 3. Initialize and fit the model
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    grid_shape_zyx = tuple(reversed(grid_shape))
    if backend == 'pytorch':
        from .syn import SyNTo as SyNToPy
        import torch
        device = kwargs.get('device', None)
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        
        use_analytical = kwargs.get('use_analytical_gradients', kwargs.get('use_ants_pseudo_gradient', False))
        model = SyNToPy(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed.origin, direction=direction,
            fluid_sigma=fluid_sigma_actual, elastic_sigma=elastic_sigma_actual, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps, in_loop_inv_steps=kwargs.get('in_loop_inv_steps', 6), project_inverse=project_inverse,
            use_ants_pseudo_gradient=use_analytical,
            projection_frequency=projection_frequency, interpolator=interpolator,
            boundary_suppression_thresh=boundary_suppression_thresh,
            image_grad_clip=image_grad_clip,
            antisymmetric=antisymmetric,
            inv_tolerance=inv_tolerance
        ).to(device)
        model.formulation = kwargs.get('formulation', 'eulerian')
        model.smooth_in_deformed_space = kwargs.get('smooth_in_deformed_space', False)
        model.kernel_type = kwargs.get('kernel_type', 'bessel')
    elif backend == 'jax':
        from .syn_jax import SyNTo as SyNToJax
        import jax.numpy as jnp
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fixed.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *moving.shape).transpose(perm)
        
        model = SyNToJax(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed.origin, direction=direction,
            fluid_sigma=fluid_sigma_actual, elastic_sigma=elastic_sigma_actual, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps, project_inverse=project_inverse,
            projection_frequency=projection_frequency, interpolator=interpolator,
            boundary_suppression_thresh=boundary_suppression_thresh,
            image_grad_clip=image_grad_clip,
            antisymmetric=antisymmetric
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    affine_lr_param = kwargs.get('affine_lr', 1e-2)
    # levels_to_use is defined above
        
    smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
    if smoothing_sigmas is None:
        import math
        smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels_to_use]
        
    if backend == 'pytorch':
        initial_grid_tensor = torch.tensor(initial_grid, dtype=torch.float32, device=device) if initial_grid is not None else None
        model.fit(
            I_tensor, J_tensor,
            levels=levels_to_use,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            affine_lr=affine_lr_param,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            syn_metric_weights=syn_metric_weights,
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
            regularizer=kwargs.get('regularizer', 'gaussian'),
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=kwargs.get('use_analytical_gradients', True),
            init_M_phys=init_M_phys,
            init_t_phys=init_t_phys,
            interpolator=interpolator
        )
    else:
        import jax.numpy as jnp
        initial_grid_tensor = jnp.array(initial_grid) if initial_grid is not None else None
        model.fit(
            I_tensor, J_tensor,
            levels=levels_to_use,
            epochs_per_level=reg_iterations,
            affine_epochs=affine_iterations,
            affine_lr=affine_lr_param,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            syn_metric_weights=syn_metric_weights,
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
            regularizer=kwargs.get('regularizer', 'gaussian'),
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=kwargs.get('use_analytical_gradients', True),
            init_M_phys=init_M_phys.cpu().numpy() if init_M_phys is not None else None,
            init_t_phys=init_t_phys.cpu().numpy() if init_t_phys is not None else None,
            interpolator=interpolator
        )
    
    # 4. Save displacement fields to temp files to match ANTs file-based transforms
    fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd_Warp.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_inv_Warp.nii.gz', delete=False).name
    
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
                T_grid = model.affine.get_matrix().detach().cpu().numpy()
                if verbose:
                    print(f"[pytorch] T_grid:\n", T_grid)
                moving_target = fixed if initial_grid is not None else moving_reg
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
            moving_target = fixed if initial_grid is not None else moving_reg
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
        
        if affine_file is not None:
            fwd_transforms = [fwd_file, affine_file]
            inv_transforms = [affine_file, inv_file]
            whichtoinvert_inv = [True, False]
        else:
            fwd_transforms = [fwd_file]
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]
    else:
        if affine_file is not None:
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
    
    fwd_midpoint_warp = None
    inv_midpoint_warp = None
    midpoint_fixed = None
    midpoint_moving = None

    if sum(reg_iterations) > 0 and hasattr(model, 'midpoint_warp_l2r') and hasattr(model, 'midpoint_warp_r2l'):
        fwd_mid_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
        inv_mid_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name

        if backend == 'pytorch':
            w_l2r_np = model.midpoint_warp_l2r.detach().cpu().numpy()
            w_r2l_np = model.midpoint_warp_r2l.detach().cpu().numpy()
        else:
            w_l2r_np = np.array(model.midpoint_warp_l2r)
            w_r2l_np = np.array(model.midpoint_warp_r2l)
        if dim == 2:
            w_l2r_np = w_l2r_np.transpose(0, 2, 1, 3)[0]
            w_r2l_np = w_r2l_np.transpose(0, 2, 1, 3)[0]
        elif dim == 3:
            w_l2r_np = w_l2r_np.transpose(0, 3, 2, 1, 4)[0]
            w_r2l_np = w_r2l_np.transpose(0, 3, 2, 1, 4)[0]

        disp_l2r_t = w_l2r_np[..., ::-1].copy()
        disp_r2l_t = w_r2l_np[..., ::-1].copy()

        fwd_mid_img = ants.from_numpy(disp_l2r_t, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True)
        inv_mid_img = ants.from_numpy(disp_r2l_t, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True)

        ants.image_write(fwd_mid_img, fwd_mid_file)
        ants.image_write(inv_mid_img, inv_mid_file)

        fwd_midpoint_warp = fwd_mid_file
        inv_midpoint_warp = inv_mid_file

        midpoint_fixed = ants.apply_transforms(fixed=fixed, moving=fixed, transformlist=[fwd_midpoint_warp])
        if affine_file is not None:
            midpoint_moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[inv_midpoint_warp, affine_file])
        else:
            midpoint_moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[inv_midpoint_warp])

    ret_dict = {'model': model,
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'midpoint_fixed': midpoint_fixed,
        'midpoint_moving': midpoint_moving,
        'fwd_midpoint_warp': fwd_midpoint_warp,
        'inv_midpoint_warp': inv_midpoint_warp,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'whichtoinvert_inv': whichtoinvert_inv,
        'syn_losses': list(model.syn_losses) if hasattr(model, 'syn_losses') else [],
        'affine_losses': list(model.affine_losses) if hasattr(model, 'affine_losses') else [],
        'inverse_identity_errors': inverse_identity_errors
    }

    try:
        from .reporting import build_engine_provenance
        fit_time_val = (time.time() - t_start) if 't_start' in locals() else None
        provenance = build_engine_provenance(
            algorithm="syntx.syn",
            backend=backend,
            device=str(device) if 'device' in locals() and device is not None else "cpu",
            fit_time=fit_time_val,
            reg_iterations=reg_iterations,
            affine_iterations=affine_iterations,
            solver="SyN",
            fluid_sigma=flow_sigma,
            elastic_sigma=total_sigma,
            learning_rate=grad_step,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            similarity_metric=syn_metric,
            syn_sampling=syn_sampling,
            aff_metric=aff_metric,
            aff_sampling=aff_sampling,
            levels=levels,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_mode=vgg_mode,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_lncc_window_size=vgg_lncc_window_size,
            project_inverse=project_inverse,
            projection_frequency=projection_frequency,
            interpolator=interpolator,
            inverse_method=inverse_method,
            inverse_steps=inverse_steps,
            fixed_shape=tuple(fixed.shape) if isinstance(fixed, ants.ANTsImage) else None,
            fixed_spacing=tuple(fixed.spacing) if isinstance(fixed, ants.ANTsImage) else None,
            fixed_orientation=str(fixed.orientation) if isinstance(fixed, ants.ANTsImage) else None,
            moving_shape=tuple(moving.shape) if isinstance(moving, ants.ANTsImage) else None,
            moving_spacing=tuple(moving.spacing) if isinstance(moving, ants.ANTsImage) else None,
            moving_orientation=str(moving.orientation) if isinstance(moving, ants.ANTsImage) else None,
            cfl_momentum=cfl_momentum,
            multipoint_loss=multipoint_loss,
            fast_smooth=fast_smooth,
            n_time_steps=n_time_steps,
            n_steps=n_steps,
            antisymmetric=antisymmetric

        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass
    
    if backend == 'pytorch':
        dev_str = str(device).lower() if 'device' in locals() and device is not None else ''
        gc.collect()
        if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()

    return ret_dict


syn = registration


def auto_reg(fixed, moving, verbose=False, **kwargs):
    """
    Performs general-purpose 2D/3D image registration using zero-effort "best defaults".

    Defaults (automatically configured unless overridden in kwargs):
    ---------------------------------------------------------------
    - backend: Auto-detected ('jax' if available, else 'pytorch')
    - device: Auto-detected ('cuda' -> 'mps' -> 'cpu')
    - type_of_transform: 'SyNTo'
    - levels: [4, 2, 1] (3-level multi-resolution pyramid)
    - affine_iterations: [100, 50, 20] (with FOV/Foreground CoM initialization selection)
    - reg_iterations: [100, 100, 20]
    - grad_step: 0.50 (Bounded CFL step multiplier)
    - flow_sigma: 3.0 (ITK Discrete Gaussian Bessel Kernel, σ² = 3.0)
    - syn_metric: 'lncc' (Local Normalized Cross-Correlation, window_size=5)
    - syn_sampling: 2
    - interpolator: 'linear' (Hardware-accelerated grid sampling)
    - inverse_steps: 30 (Symmetric diffeomorphic inversion)
    - inverse_method: 'anderson'

    Parameters:
    -----------
    fixed : ANTsImage, PyTorch Tensor, JAX Array, or NumPy array
        Target/Fixed image to register to.
    moving : ANTsImage, PyTorch Tensor, JAX Array, or NumPy array
        Moving image to be deformed into fixed space.
    verbose : bool, default=False
        If True, prints progress and iteration metrics during registration.
    **kwargs : dict
        Optional parameter overrides for underlying registration options.

    Returns:
    --------
    dict containing:
        - 'warpedmovout': Warped moving image in fixed space
        - 'warpedfixout': Warped fixed image in moving space
        - 'fwdtransforms': List of forward transform file paths (Warp + Affine)
        - 'invtransforms': List of inverse transform file paths (Affine + Inverse Warp)
        - 'metrics': Dictionary containing standard evaluation metrics:
            * 'jac_mean': Mean Jacobian determinant
            * 'jac_min': Minimum Jacobian determinant
            * 'jac_max': Maximum Jacobian determinant
            * 'jac_std': Standard deviation of Jacobian determinant
            * 'folding_pct': Percentage of folding voxels (J <= 0)
            * 'smooth_1st': 1st derivative grid smoothness ||∇u||
            * 'smooth_2nd': 2nd derivative grid smoothness ||∇²u||
            * 'lncc_score': Local NCC similarity score
            * 'mse_score': Mean Squared Error
            * 'mattes_mi_score': Mattes Mutual Information score
            * 'inverse_identity_mean_error': Mean topological inverse identity error
            * 'inverse_identity_max_error': Max topological inverse identity error
            * 'execution_time_seconds': Total registration runtime in seconds
            * 'device_used': Auto-detected hardware device ('cuda', 'mps', or 'cpu')
            * 'backend_used': Auto-detected compute engine ('jax' or 'pytorch')
    """
    import time
    import ants
    t0 = time.time()
    
    # 1. Hardware & backend auto-detection
    target_backend = kwargs.pop('backend', None)
    if target_backend is None:
        try:
            import jax
            target_backend = 'jax'
        except ImportError:
            target_backend = 'pytorch'
            
    target_device = kwargs.pop('device', None)
    if target_device is None:
        import torch
        if torch.cuda.is_available():
            target_device = 'cuda'
        elif torch.backends.mps.is_available():
            target_device = 'mps'
        else:
            target_device = 'cpu'
            
    # 2. Optimal "Best Defaults" (Adaptive for Anisotropic / Special Scans)
    sigma_mode = 'voxel'
    if hasattr(fixed, 'spacing'):
        sp = fixed.spacing
        if len(sp) > 1 and (max(sp) / max(min(sp), 1e-5)) >= 1.5:
            sigma_mode = 'physical'

    reg_params = {
        'backend': target_backend,
        'device': target_device,
        'type_of_transform': 'SyNTo',
        'levels': [4, 2, 1],
        'affine_iterations': [100, 50, 20],
        'reg_iterations': [100, 100, 20],
        'grad_step': 0.50,
        'flow_sigma': 3.0,
        'sigma_mode': sigma_mode,
        'syn_metric': 'lncc',
        'syn_sampling': 2,
        'interpolator': 'linear',
        'inverse_steps': 30,
        'inverse_method': 'anderson',
        'boundary_suppression_thresh': None,
        'image_grad_clip': 6.0,
        'verbose': verbose
    }
    reg_params.update(kwargs)
    
    # 3. Execute Registration
    res = registration(fixed=fixed, moving=moving, **reg_params)
    t_elapsed = time.time() - t0
    
    # 4. Compute Standard Metrics
    warpedmovout = res['warpedmovout']
    fwd_tx = res['fwdtransforms']
    
    metrics = {
        'execution_time_seconds': float(t_elapsed),
        'device_used': str(target_device),
        'backend_used': str(target_backend)
    }
    
    # Jacobian determinant & folding % if forward warp exists
    warp_file = next((tx for tx in fwd_tx if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
    if warp_file is not None:
        try:
            disp_img = ants.image_read(warp_file)
            disp_np = disp_img.numpy()
            if disp_np.ndim == 4 and disp_np.shape[0] == 3:
                disp_np = np.moveaxis(disp_np, 0, -1)
            elif disp_np.ndim == 3 and disp_np.shape[0] == 2:
                disp_np = np.moveaxis(disp_np, 0, -1)
                
            sp = disp_img.spacing
            sp_x = sp[0]
            sp_y = sp[1] if len(sp) > 1 else 1.0
            sp_z = sp[2] if len(sp) > 2 else 1.0
            
            if disp_np.ndim == 4:  # 3D image
                try:
                    jac_img = ants.create_jacobian_determinant_image(fixed, warp_file)
                    jac_np = jac_img.numpy()
                except Exception:
                    du_dx = (disp_np[1:, :-1, :-1] - disp_np[:-1, :-1, :-1]) / sp_x
                    du_dy = (disp_np[:-1, 1:, :-1] - disp_np[:-1, :-1, :-1]) / sp_y
                    du_dz = (disp_np[:-1, :-1, 1:] - disp_np[:-1, :-1, :-1]) / sp_z
                    j11 = 1.0 + du_dx[..., 0]
                    j22 = 1.0 + du_dy[..., 1]
                    j33 = 1.0 + du_dz[..., 2]
                    jac_np = j11 * j22 * j33
                    
                mask_np = ants.get_mask(fixed).numpy() > 0 if hasattr(fixed, 'numpy') else np.ones_like(jac_np, dtype=bool)
                metrics['jac_mean'] = float(np.mean(jac_np))
                metrics['jac_min'] = float(np.min(jac_np))
                metrics['jac_max'] = float(np.max(jac_np))
                metrics['jac_std'] = float(np.std(jac_np))
                metrics['folding_pct'] = float(np.mean(jac_np[mask_np] <= 0) * 100.0) if np.sum(mask_np) > 0 else 0.0
                
                du_dx = (disp_np[1:, :-1, :-1] - disp_np[:-1, :-1, :-1]) / sp_x
                du_dy = (disp_np[:-1, 1:, :-1] - disp_np[:-1, :-1, :-1]) / sp_y
                du_dz = (disp_np[:-1, :-1, 1:] - disp_np[:-1, :-1, :-1]) / sp_z
                metrics['smooth_1st'] = float(np.mean(np.sqrt(du_dx**2 + du_dy**2 + du_dz**2)))
                
                d2u_dx2 = (du_dx[1:, :-1, :-1] - du_dx[:-1, :-1, :-1]) / sp_x
                d2u_dy2 = (du_dy[:-1, 1:, :-1] - du_dy[:-1, :-1, :-1]) / sp_y
                d2u_dz2 = (du_dz[:-1, :-1, 1:] - du_dz[:-1, :-1, :-1]) / sp_z
                metrics['smooth_2nd'] = float(np.mean(np.sqrt(d2u_dx2**2 + d2u_dy2**2 + d2u_dz2**2)))
            elif disp_np.ndim == 3:  # 2D image
                du_dx = (disp_np[1:, :-1] - disp_np[:-1, :-1]) / sp_x
                du_dy = (disp_np[:-1, 1:] - disp_np[:-1, :-1]) / sp_y
                
                j11 = 1.0 + du_dx[..., 0]
                j12 = du_dy[..., 0]
                j21 = du_dx[..., 1]
                j22 = 1.0 + du_dy[..., 1]
                jac_np = j11 * j22 - j12 * j21
                
                mask_np = ants.get_mask(fixed).numpy() > 0 if hasattr(fixed, 'numpy') else np.ones_like(jac_np, dtype=bool)
                if mask_np.shape != jac_np.shape:
                    slices = tuple(slice(0, s) for s in jac_np.shape)
                    mask_np = mask_np[slices]
                metrics['jac_mean'] = float(np.mean(jac_np))
                metrics['jac_min'] = float(np.min(jac_np))
                metrics['jac_max'] = float(np.max(jac_np))
                metrics['jac_std'] = float(np.std(jac_np))
                metrics['folding_pct'] = float(np.mean(jac_np[mask_np] <= 0) * 100.0) if np.sum(mask_np) > 0 else 0.0
                
                metrics['smooth_1st'] = float(np.mean(np.sqrt(du_dx**2 + du_dy**2)))
                d2u_dx2 = (du_dx[1:, :-1] - du_dx[:-1, :-1]) / sp_x
                d2u_dy2 = (du_dy[:-1, 1:] - du_dy[:-1, :-1]) / sp_y
                metrics['smooth_2nd'] = float(np.mean(np.sqrt(d2u_dx2**2 + d2u_dy2**2)))
        except Exception as e:
            if verbose:
                print(f"[auto_reg] Jacobian calculation skipped: {e}")
    else:
        # Affine-only registration fallback metrics
        metrics['jac_mean'] = 1.0
        metrics['jac_min'] = 1.0
        metrics['jac_max'] = 1.0
        metrics['jac_std'] = 0.0
        metrics['folding_pct'] = 0.0
        metrics['smooth_1st'] = 0.0
        metrics['smooth_2nd'] = 0.0
                
    # Image similarity scores via image_compare
    try:
        import ants
        from .image_compare import image_compare
        metrics['lncc_score'] = float(image_compare(fixed, warpedmovout, metricname='lncc'))
        metrics['mse_score'] = float(image_compare(fixed, warpedmovout, metricname='mse'))
        metrics['mattes_mi_score'] = float(image_compare(fixed, warpedmovout, metricname='mattes_mi'))
    except Exception as e:
        if verbose:
            print(f"[auto_reg] Image similarity calculation skipped: {e}")
            
    # Inverse identity topology errors
    inv_errs = res.get('inverse_identity_errors', {})
    if inv_errs:
        err_vals_mean = [v['mean_error'] for v in inv_errs.values() if isinstance(v, dict) and 'mean_error' in v]
        err_vals_max = [v['max_error'] for v in inv_errs.values() if isinstance(v, dict) and 'max_error' in v]
        if err_vals_mean:
            metrics['inverse_identity_mean_error'] = float(np.mean(err_vals_mean))
        if err_vals_max:
            metrics['inverse_identity_max_error'] = float(np.max(err_vals_max))
            
    res['metrics'] = metrics
    return res


def normalize_tensor(
    tensor: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8,
    p_min: float = 1.0,
    p_max: float = 99.0,
    dim=None,
    keepdim: bool = True
) -> torch.Tensor:
    """
    Normalizes input PyTorch tensor using specified strategy.

    Args:
        tensor: Input PyTorch tensor (any spatial dimension).
        method: Normalization strategy:
            - 'minmax': Rescales values linearly to [0, 1].
            - 'zscore': Subtracts mean and divides by standard deviation (zero-mean, unit-variance).
            - 'robust' / 'percentile': Rescales between p_min and p_max percentiles and clamps to [0, 1].
            - 'l2' / 'unit_norm': Scales tensor by its L2 norm.
            - 'l1' / 'unit_sum': Scales tensor by its L1 norm.
            - 'sigmoid': Applies logistic sigmoid transformation.
        eps: Numerical stability floor to prevent division by zero (default: 1e-8).
        p_min: Lower percentile threshold for 'robust' scaling (default: 1.0).
        p_max: Upper percentile threshold for 'robust' scaling (default: 99.0).
        dim: Dimension(s) over which to compute statistics. If None, computes globally over all elements.
        keepdim: Retain reduced dimensions when dim is specified.

    Returns:
        Normalized PyTorch tensor with same shape and dtype.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    method = method.lower().strip()

    if method in ('minmax', '01'):
        if dim is None:
            t_min = tensor.min()
            t_max = tensor.max()
        else:
            t_min = tensor.amin(dim=dim, keepdim=keepdim)
            t_max = tensor.amax(dim=dim, keepdim=keepdim)
        return (tensor - t_min) / (t_max - t_min + eps)

    elif method in ('zscore', 'standard'):
        if dim is None:
            t_mean = tensor.mean()
            t_std = tensor.std(unbiased=False)
        else:
            t_mean = tensor.mean(dim=dim, keepdim=keepdim)
            t_std = tensor.std(dim=dim, keepdim=keepdim, unbiased=False)
        return (tensor - t_mean) / (t_std + eps)

    elif method in ('robust', 'percentile'):
        if dim is None:
            q_min = torch.quantile(tensor.float(), p_min / 100.0).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0).to(tensor.dtype)
        else:
            q_min = torch.quantile(tensor.float(), p_min / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
        res = (tensor - q_min) / (q_max - q_min + eps)
        return torch.clamp(res, 0.0, 1.0)

    elif method in ('l2', 'unit_norm'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=2)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('l1', 'unit_sum'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=1)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=1, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('sigmoid', 'logistic'):
        return torch.sigmoid(tensor)

    else:
        raise ValueError(f"Unknown normalization method '{method}'. Options: 'minmax', 'zscore', 'robust', 'l2', 'l1', 'sigmoid'.")


def plot_deformation_grid(
    warp,
    fixed=None,
    slice_axis: int = 2,
    slice_idx=None,
    grid_spacing: int = 8,
    color: str = '#38bdf8',
    linewidth: float = 1.2,
    background_cmap: str = 'gray',
    figsize=(8, 8),
    ax=None,
    title: str = "Deformation Grid",
    show: bool = False,
    filename=None
):
    """
    Plots a 2D or 3D spatial deformation coordinate grid matching ANTs image orientation conventions.

    Args:
        warp: Displacement field as ANTsImage, PyTorch Tensor, or NumPy array.
        fixed: Optional background image as ANTsImage, PyTorch Tensor, or NumPy array.
        slice_axis: Axis along which to slice 3D volumes (0: Sagittal, 1: Coronal, 2: Axial). Default: 2.
        slice_idx: Slice index along slice_axis. Defaults to midpoint.
        grid_spacing: Subsampling interval for grid lines (default: 8 voxels).
        color: Grid line color (default: '#38bdf8').
        linewidth: Grid line width (default: 1.2).
        background_cmap: Colormap for background image (default: 'gray').
        figsize: Figure size tuple (default: (8, 8)).
        ax: Optional existing Matplotlib Axes object.
        title: Optional plot title.
        show: If True, calls plt.show().
        filename: Optional path to save figure file.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Parse Warp & Fixed Image Data and Metadata
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    try:
        import ants
        if isinstance(warp, ants.ANTsImage):
            warp_img = warp.reorient_image2('LAI') if warp.dimension == 3 else warp
            spacing = warp_img.spacing
            origin = warp_img.origin
            w_np = warp_img.numpy()
        elif hasattr(warp, 'cpu'):
            w_np = warp.squeeze().cpu().numpy()
        else:
            w_np = np.asarray(warp)

        if fixed is not None:
            if isinstance(fixed, ants.ANTsImage):
                fixed_img = fixed.reorient_image2('LAI') if fixed.dimension == 3 else fixed
                spacing = fixed_img.spacing
                origin = fixed_img.origin
                bg_np = fixed_img.numpy()
            elif hasattr(fixed, 'cpu'):
                bg_np = fixed.squeeze().cpu().numpy()
            else:
                bg_np = np.asarray(fixed)
        else:
            bg_np = None
    except Exception:
        if hasattr(warp, 'cpu'):
            w_np = warp.squeeze().cpu().numpy()
        else:
            w_np = np.asarray(warp)
        if fixed is not None:
            if hasattr(fixed, 'cpu'):
                bg_np = fixed.squeeze().cpu().numpy()
            else:
                bg_np = np.asarray(fixed)
        else:
            bg_np = None

    # Ensure w_np components are last dimension
    if w_np.ndim == 4 and w_np.shape[0] in (2, 3):
        w_np = np.moveaxis(w_np, 0, -1)
    elif w_np.ndim == 3 and w_np.shape[0] == 2:
        w_np = np.moveaxis(w_np, 0, -1)

    is_3d = (w_np.ndim == 4)

    # 2. Extract 2D Slice and Spatial Coordinates matching ANTs plotting conventions
    if is_3d:
        shape = w_np.shape[:3]
        if slice_idx is None:
            slice_idx = shape[slice_axis] // 2
        slice_idx = max(0, min(shape[slice_axis] - 1, slice_idx))

        if slice_axis == 2:  # Axial: Horizontal = X (0), Vertical = Y (1)
            w_2d = w_np[:, :, slice_idx, :]
            bg_slice = bg_np[:, :, slice_idx] if bg_np is not None else None
            r_bg = bg_slice.T if bg_slice is not None else None
            sp_h, sp_v = spacing[0], spacing[1]
            orig_h, orig_v = origin[0], origin[1]
            h_dim, v_dim = 0, 1
            extent = [orig_h, orig_h + shape[0] * sp_h, orig_v + shape[1] * sp_v, orig_v]
            origin_mode = 'upper'
        elif slice_axis == 1:  # Coronal: Horizontal = X (0), Vertical = Z (2)
            w_2d = w_np[:, slice_idx, :, :]
            bg_slice = bg_np[:, slice_idx, :] if bg_np is not None else None
            r_bg = bg_slice.T[::-1, :] if bg_slice is not None else None
            sp_h, sp_v = spacing[0], spacing[2]
            orig_h, orig_v = origin[0], origin[2]
            h_dim, v_dim = 0, 2
            extent = [orig_h, orig_h + shape[0] * sp_h, orig_v, orig_v + shape[2] * sp_v]
            origin_mode = 'lower'
        else:  # Sagittal: Horizontal = Y (1), Vertical = Z (2)
            w_2d = w_np[slice_idx, :, :, :]
            bg_slice = bg_np[slice_idx, :, :] if bg_np is not None else None
            r_bg = bg_slice.T[::-1, :] if bg_slice is not None else None
            sp_h, sp_v = spacing[1], spacing[2]
            orig_h, orig_v = origin[1], origin[2]
            h_dim, v_dim = 1, 2
            extent = [orig_h, orig_h + shape[1] * sp_h, orig_v, orig_v + shape[2] * sp_v]
            origin_mode = 'lower'
    else:
        shape = w_np.shape[:2]
        w_2d = w_np
        bg_slice = bg_np
        r_bg = bg_slice.T if bg_slice is not None else None
        sp_h, sp_v = spacing[0], spacing[1]
        orig_h, orig_v = origin[0], origin[1]
        h_dim, v_dim = 0, 1
        extent = [orig_h, orig_h + shape[0] * sp_h, orig_v + shape[1] * sp_v, orig_v]
        origin_mode = 'upper'

    nh, nv = w_2d.shape[:2]
    step = max(1, grid_spacing)

    grid_h = np.arange(0, nh, step)
    grid_v = np.arange(0, nv, step)

    hh, vv = np.meshgrid(np.arange(nh), np.arange(nv), indexing='ij')

    phys_h = orig_h + hh * sp_h + w_2d[..., h_dim]
    phys_v = orig_v + vv * sp_v + w_2d[..., v_dim]

    # 3. Create Matplotlib Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0b0f17')
    else:
        fig = ax.figure

    ax.set_facecolor('#0b0f17')

    # Background Image Overlay
    if r_bg is not None:
        ax.imshow(r_bg, cmap=background_cmap, origin=origin_mode, extent=extent, alpha=0.6)

    # Plot Deformed Grid Lines in Physical Space
    for idx_h in grid_h:
        ax.plot(phys_h[idx_h, :], phys_v[idx_h, :], color=color, lw=linewidth)

    for idx_v in grid_v:
        ax.plot(phys_h[:, idx_v], phys_v[:, idx_v], color=color, lw=linewidth)

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, color='#f8fafc', fontsize=13, fontweight='bold', pad=12)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0b0f17')

    if show:
        plt.show()

    return fig


def extract_2d_slice(img, slice_axis: int = 2, slice_idx=None, ref_image=None):
    """
    Extracts a 2D scalar NumPy slice (H, W) or 2D vector slice (H, W, 2)
    from 2D or 3D images, tensors, ANTsImages, file paths, or vector fields in standard ANTs LAI anatomical orientation.
    """
    import os
    import numpy as np
    import ants

    if isinstance(img, str) and os.path.exists(img):
        img = ants.image_read(img)

    if isinstance(img, ants.ANTsImage):
        if img.dimension == 3:
            img_lai = img.reorient_image2('LAI')
            arr = img_lai.numpy()
        else:
            arr = img.numpy()
    else:
        if hasattr(img, 'cpu'):
            arr = img.detach().cpu().numpy()
        elif hasattr(img, 'numpy'):
            arr = img.numpy()
        else:
            arr = np.asarray(img)
        arr = np.squeeze(arr)

        if ref_image is not None and isinstance(ref_image, ants.ANTsImage) and arr.ndim in (3, 4):
            if arr.ndim == 3:
                img_ants = ants.from_numpy(arr, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)
                arr = img_ants.reorient_image2('LAI').numpy()
            elif arr.ndim == 4:
                if arr.shape[0] in (2, 3) and arr.shape[1] > 4:
                    arr = np.moveaxis(arr, 0, -1)
                comps = []
                for c in range(arr.shape[-1]):
                    img_c = ants.from_numpy(arr[..., c], origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)
                    comps.append(img_c.reorient_image2('LAI').numpy())
                arr = np.stack(comps, axis=-1)
        elif arr.ndim >= 3:
            if arr.ndim == 3:
                arr = arr[::-1, ::-1, :]
            elif arr.ndim == 4:
                if arr.shape[0] in (2, 3) and arr.shape[1] > 4:
                    arr = np.moveaxis(arr, 0, -1)
                arr = arr[::-1, ::-1, :, :]

    if arr.ndim < 2:
        if ref_image is not None:
            ref_slice = extract_2d_slice(ref_image, slice_axis=slice_axis, slice_idx=slice_idx)
            return np.full(ref_slice.shape, float(arr), dtype=np.float32)
        return np.array([[float(arr)]], dtype=np.float32)

    # 2D Scalar image (H, W)
    if arr.ndim == 2:
        return arr.T

    # 2D Vector displacement field: (2, H, W) or (H, W, 2)
    if arr.ndim == 3 and arr.shape[0] in (2, 3) and arr.shape[1] > 4 and arr.shape[2] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return np.transpose(arr, (1, 0, 2))

    # 3D Scalar volume (H, W, D)
    if arr.ndim == 3:
        depth = arr.shape[slice_axis]
        s_idx = depth // 2 if slice_idx is None else max(0, min(depth - 1, slice_idx))
        if slice_axis == 2:
            return arr[:, :, s_idx].T
        elif slice_axis == 1:
            return arr[:, s_idx, :].T[::-1, :]
        else:
            return arr[s_idx, :, :].T[::-1, :]

    # 3D Vector field (3, H, W, D) or (H, W, D, 3)
    if arr.ndim == 4:
        if arr.shape[0] in (2, 3) and arr.shape[1] > 4:
            arr = np.moveaxis(arr, 0, -1)
        depth = arr.shape[slice_axis]
        s_idx = depth // 2 if slice_idx is None else max(0, min(depth - 1, slice_idx))
from .viz import (
    extract_2d_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
    render_input_pair_figure
)



