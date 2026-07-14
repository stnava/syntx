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
        
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[2], omega_norm[1]]),
            torch.stack([omega_norm[2], torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[0]]),
            torch.stack([-omega_norm[1], omega_norm[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)
        return torch.where(is_zero, I, R)
    else:
        raise ValueError("Only 2D and 3D are supported.")

class TriPlanarVGG3DLoss(nn.Module):
    def __init__(self, dim=3, feature_layers=[8], num_slices=4, patch_size=32, num_patches=8, mode='patch_walk', vgg_lncc_window_size=9):
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
        sigma_list = [sigma / sp for sp in spacing]
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
    
    # grads[k] shape: (B, *spatial, d) = derivative of all field components w.r.t. spatial dim k
    # Reverse to match our (x,y,[z]) component ordering convention
    return torch.stack(list(reversed(grads)), dim=-1)  # (B, *spatial, d, d)


def update_inverse_field_nd(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor, 
    steps: int = 20,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    method: str = 'fixed_point',
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001
) -> torch.Tensor:
    """
    Dimension-agnostic fixed-point inversion of a displacement field.
    
    Matches ITK's itkInvertDisplacementFieldImageFilter exactly:
    - Per-pixel update clipping in voxel-space norm (prevents divergence)
    - Adaptive relaxation (epsilon=0.75 first iter, 0.5 thereafter)
    - Early-stop convergence checking (max/mean error thresholds)
    - Dirichlet zero boundary enforcement every iteration
    
    W_disp: (B, *spatial, d) — forward displacement field
    W_inv_disp: (B, *spatial, d) — current inverse estimate (updated in-place conceptually)
    steps: max iterations (ITK default: 20)
    method: 'fixed_point' (ITK standard) or 'neumann' (Neumann series preconditioner)
    """
    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
    
    boundary_mask = get_boundary_mask(spatial, device, dtype)
    
    # Voxel scale: converts normalized [-1,1] displacement to voxel units
    # In ITK, scaledNorm = ||disp / spacing||_2; in normalized coords, 1 voxel = 2/(N-1)
    voxel_scale = torch.tensor(
        [float((s - 1) / 2.0) for s in reversed(spatial)],
        device=device, dtype=dtype
    )
    
    W_disp_cf = torch.movedim(W_disp, -1, 1)
    
    for iteration in range(steps):
        # Phase 1: Compute composition error = W_inv + W(x + W_inv(x))
        # This should be zero if W_inv is the perfect inverse
        coords = identity + W_inv_disp
        forward_at_inv_cf = F.grid_sample(W_disp_cf, coords, padding_mode='border', align_corners=True)
        forward_at_inv = torch.movedim(forward_at_inv_cf, 1, -1)
        
        error = W_inv_disp + forward_at_inv  # composition error
        
        # Compute per-pixel error norm in voxel coordinates (ITK lines 222-228)
        error_voxel = error * voxel_scale
        scaled_norm = torch.sqrt(torch.sum(error_voxel**2, dim=-1, keepdim=True))  # (B, *spatial, 1)
        
        max_error = scaled_norm.max()
        
        # Adaptive relaxation (ITK lines 147-151)
        epsilon = 0.75 if iteration == 0 else 0.5
        
        # Update direction: we want to subtract the error
        update = -error
        
        if method == 'neumann':
            # Neumann preconditioner: (I - ∇u) · (-error) for quasi-Newton convergence
            Du = _spatial_jacobian_nd(forward_at_inv)
            Du_error = torch.einsum('b...ij,b...j->b...i', Du, error)
            update = -(error - Du_error)
        
        # Per-pixel update clipping in voxel-space norm (ITK lines 191-194)
        # Prevents divergence at high-deformation pixels
        update_voxel = update * voxel_scale
        update_norm = torch.sqrt(torch.sum(update_voxel**2, dim=-1, keepdim=True)) + 1e-10
        clip_threshold = epsilon * max_error
        clip_mask = update_norm > clip_threshold
        clip_scale = torch.where(clip_mask, clip_threshold / update_norm, torch.ones_like(update_norm))
        update = update * clip_scale
        
        # Apply update with relaxation (ITK line 195)
        W_inv_disp = W_inv_disp + epsilon * update
        
        # Optional Gaussian smoothing relaxation
        if smoothing_sigma > 0.0:
            W_inv_disp = separable_gaussian_filter(W_inv_disp, smoothing_sigma)
            
        # ITK-standard Dirichlet zero boundary enforcement (ITK lines 198-208)
        W_inv_disp = W_inv_disp * boundary_mask
        
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
        return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=True)
        
    I_mean = box_filter(I)
    J_mean = box_filter(J)
    
    I_var = box_filter(I**2) - I_mean**2
    J_var = box_filter(J**2) - J_mean**2
    IJ_cov = box_filter(I*J) - I_mean * J_mean
    
    valid_mask = (I_var > 1e-5) & (J_var > 1e-5)
    
    safe_I_var = torch.clamp(I_var, min=1e-5)
    safe_J_var = torch.clamp(J_var, min=1e-5)
    
    cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-5)
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
    warp_field: (B, *spatial, dim) - displacement field (normalized coordinates)
    Returns: (B, *spatial) - Jacobian determinant values
    """
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype
    
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
    - warp_field: (B, *spatial, dim) normalized displacement field in [-1, 1]
    - direction: (dim, dim) physical direction matrix D
    - spacing: (dim,) voxel spacing S (in mm)
    
    Returns:
    - jac_det_phys: (B, *spatial) physical Jacobian determinant map
    """
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
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, direction=None, fluid_sigma=1.732, elastic_sigma=1.0, transform_type='Affine', inverse_method='fixed_point', inverse_steps=5):
        """
        Generalized Symmetric Normalization (SyN) in PyTorch.
        Includes hierarchical affine pre-alignment and dense symmetric velocity/displacement fields.
        
        Parameters
        ----------
        spacing : tuple or None
            Physical voxel spacing (in ITK/ANTs axis order).
        direction : array-like or None
            Direction cosine matrix (dim x dim). Defaults to identity if None.
            Follows ITK convention: maps voxel axes to physical axes.
        """
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape
        self.spacing = spacing
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
            affine_epochs=[100, 50, 50, 20], affine_lr=1e-2, cfl_voxels=0.75, 
            similarity_metric='lncc', use_analytical_gradients=True,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None,
            vgg_layers=[8], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='patch_walk',
            vgg_lncc_window_size=9, syn_metric_weights=None, **kwargs):
        """
        Runs the full native pre-alignment and SyN multi-resolution optimization loop.
        fixed_image: (1, 1, *spatial)
        moving_image: (1, 1, *spatial)
        """
        lncc_window_size = 2 * lncc_radius + 1
        device = fixed_image.device
        dtype = fixed_image.dtype
        dim = self.dim
        spatial_shape = fixed_image.shape[2:]
        
        self.affine_losses = []
        self.syn_losses = []
        
        # Standardize similarity_metric to a list of metrics
        if isinstance(similarity_metric, str):
            self.metrics = [similarity_metric]
        else:
            self.metrics = list(similarity_metric)

        self.metric_weights = syn_metric_weights if syn_metric_weights is not None else [1.0] * len(self.metrics)
        self.loss_functions = []
        
        from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor
        
        for metric_name in self.metrics:
            metric_name_lower = metric_name.lower()
            if metric_name_lower == 'mattes_mi':
                self.loss_functions.append(lambda x, y: mattes_mi_loss_nd(x, y, num_bins=mattes_bins))
            elif metric_name_lower == 'lncc':
                self.loss_functions.append(lambda x, y: local_ncc_loss_nd(x, y, window_size=lncc_radius))
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
                raise ValueError(f"Unknown similarity metric: {metric_name}")
        
        # --- 0. Construct Image Pyramids ---
        I_pyr = [F.interpolate(fixed_image, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=False) if s > 1 else fixed_image for s in levels]
        J_pyr = [F.interpolate(moving_image, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=False) if s > 1 else moving_image for s in levels]
        
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
                
                # Hierarchical Parameter Unlocking
                active_params = [self.affine.translation]
                
                # Rigid unlocking
                if level_idx >= 1 or len(levels) == 1:
                    if hasattr(self.affine, 'omega') and isinstance(self.affine.omega, nn.Parameter):
                        active_params.append(self.affine.omega)
                        
                # Affine unlocking
                if level_idx >= 2 or len(levels) <= 2:
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
                        sampling_percentage < 1.0
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
                        moving_warped = F.grid_sample(J_curr, grid, padding_mode='border', align_corners=True)
                        loss = 0.0
                        for fn, weight in zip(self.loss_functions, self.metric_weights):
                            loss += weight * fn(moving_warped, I_curr)
                    
                    loss.backward()
                    optimizer.step()
                    self.affine_losses.append(loss)
                    level_affine_losses.append(loss)
                    if len(level_affine_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_affine_epochs - 1):
                        recent_losses = [l.item() for l in level_affine_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-5):
                            break
                
        # --- 2. SyN Registration ---
        with torch.no_grad():
            grid = self.get_affine_grid(spatial_shape, device)
            moving_affine = F.grid_sample(moving_image, grid, padding_mode='border', align_corners=True)
            
        J_pyr = [F.interpolate(moving_affine, scale_factor=1.0/s, mode='bilinear' if dim==2 else 'trilinear', align_corners=False) if s > 1 else moving_affine for s in levels]
        
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
            
            if self.spacing is not None:
                curr_physical_spacing = [float(2.0 * b / (s - 1)) for b, s in zip(self.physical_bounds, curr_spatial)]
            else:
                curr_physical_spacing = None
                
            grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in curr_spatial]
            meshgrid = torch.meshgrid(*grids, indexing='ij')
            identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
            
            if isinstance(epochs_per_level, int):
                curr_syn_epochs = epochs_per_level
            else:
                curr_syn_epochs = epochs_per_level[level_idx]
                
            level_syn_losses = []
            for epoch in range(curr_syn_epochs):
                if warp_l2r.grad is not None: warp_l2r.grad.zero_()
                if warp_r2l.grad is not None: warp_r2l.grad.zero_()
                
                phi_l2r = identity + warp_l2r
                phi_r2l = identity + warp_r2l
                
                # Real SyN: Pull both images to the midpoint domain
                I_mid = F.grid_sample(I_curr, phi_l2r, padding_mode='border', align_corners=True)
                J_mid = F.grid_sample(J_curr, phi_r2l, padding_mode='border', align_corners=True)
                
                if use_analytical_gradients:
                    with torch.no_grad():
                        grad_I_curr = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=curr_physical_spacing).squeeze(-2)
                        grad_J_curr = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=curr_physical_spacing).squeeze(-2)
                        
                        grad_I_mid_sampled = F.grid_sample(grad_I_curr.movedim(-1, 1), phi_l2r, padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                        grad_J_mid_sampled = F.grid_sample(grad_J_curr.movedim(-1, 1), phi_r2l, padding_mode='border', align_corners=True).movedim(1, -1).contiguous()
                        
                    I_mid.retain_grad()
                    J_mid.retain_grad()
                    
                    loss = 0.0
                    for fn, weight in zip(self.loss_functions, self.metric_weights):
                        loss += weight * fn(J_mid, I_mid)
                        
                    loss.backward()
                    loss_val = loss.item()
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)
                    
                    with torch.no_grad():
                        if I_mid.grad is not None:
                            grad_l_raw = (I_mid.grad.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                            warp_l2r.grad = grad_l_raw
                        if J_mid.grad is not None:
                            grad_r_raw = (J_mid.grad.movedim(1, -1) * grad_J_mid_sampled).contiguous()
                            warp_r2l.grad = grad_r_raw
                else:
                    loss = 0.0
                    for fn, weight in zip(self.loss_functions, self.metric_weights):
                        loss += weight * fn(J_mid, I_mid)
                        
                    loss.backward()
                    self.syn_losses.append(loss)
                    level_syn_losses.append(loss)
                
                with torch.no_grad():
                    b_mask = get_boundary_mask(curr_spatial, device, dtype)
                    
                    grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, self.fluid_sigma, spacing=curr_physical_spacing)
                    grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, self.fluid_sigma, spacing=curr_physical_spacing)
                    
                    grad_norm_l = torch.sqrt(torch.sum(grad_l**2, dim=-1))
                    grad_norm_r = torch.sqrt(torch.sum(grad_r**2, dim=-1))
                    
                    max_grad_l = torch.max(grad_norm_l) + 1e-8
                    max_grad_r = torch.max(grad_norm_r) + 1e-8
                    
                    # ITK-style CFL: Euclidean norm in VOXEL coordinates
                    # Convert normalized-space gradient to voxel coords, compute L2 norm,
                    # then apply uniform scalar to bound max voxel displacement to cfl_voxels.
                    # Matches itkSyNImageRegistrationMethod::ScaleUpdateField exactly.
                    # In normalized [-1,1] space, 1 voxel = 2/(N-1) per axis,
                    # so grad_voxel[d] = grad_norm[d] * (N_d - 1) / 2
                    voxel_scale = torch.tensor(
                        [float((s - 1) / 2.0) for s in reversed(curr_spatial)],
                        device=device, dtype=dtype
                    )
                    grad_l_voxel = grad_l * voxel_scale
                    grad_r_voxel = grad_r * voxel_scale
                    max_norm_l = torch.sqrt(torch.sum(grad_l_voxel**2, dim=-1)).max() + 1e-8
                    max_norm_r = torch.sqrt(torch.sum(grad_r_voxel**2, dim=-1)).max() + 1e-8
                    
                    # Uniform scalar learning rate: max voxel displacement = cfl_voxels
                    lr_l = cfl_voxels / max_norm_l
                    lr_r = cfl_voxels / max_norm_r
                    
                    # Scale the normalized-space gradient uniformly (preserves direction)
                    delta_l = lr_l * grad_l
                    delta_r = lr_r * grad_r
                    
                    # Greedy SyN composition: φ_new = φ_old ∘ (Id - ∂loss/∂warp)
                    # Since loss = -NCC, delta = -∂CC/∂warp, so (Id - delta) = (Id + ∂CC/∂warp)
                    # This correctly moves toward better alignment, matching ITK's convention.
                    coords_l = identity - delta_l
                    coords_r = identity - delta_r
                    
                    warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_l, padding_mode='border', align_corners=True).movedim(1, -1)
                    warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_r, padding_mode='border', align_corners=True).movedim(1, -1)
                    
                    warp_l2r.copy_(warp_l2r_sampled - delta_l)
                    warp_r2l.copy_(warp_r2l_sampled - delta_r)
                    
                    # ITK-standard Dirichlet zero boundary enforcement after composition.
                    # Matches itkSyNImageRegistrationMethod::GaussianSmoothDisplacementField lines 770-805.
                    warp_l2r.mul_(b_mask)
                    warp_r2l.mul_(b_mask)
                    
                    if self.elastic_sigma > 0.0:
                        warp_l2r.copy_(separable_gaussian_filter(warp_l2r, self.elastic_sigma, spacing=curr_physical_spacing))
                        warp_r2l.copy_(separable_gaussian_filter(warp_r2l, self.elastic_sigma, spacing=curr_physical_spacing))
                        
                    # ITK-style diffeomorphic projection: double-inversion.
                    # Projects the composed field back to the space of invertible transforms.
                    # Matches itkSyNImageRegistrationMethod.hxx lines 227-244.
                    # Uses self.inverse_steps iterations (ITK standard default: 20).
                    warp_l2r_inv = update_inverse_field_nd(warp_l2r, warp_l2r_inv.detach(), steps=self.inverse_steps, method=self.inverse_method)
                    warp_l2r.copy_(update_inverse_field_nd(warp_l2r_inv, warp_l2r.detach(), steps=self.inverse_steps, method=self.inverse_method))
                    
                    warp_r2l_inv = update_inverse_field_nd(warp_r2l, warp_r2l_inv.detach(), steps=self.inverse_steps, method=self.inverse_method)
                    warp_r2l.copy_(update_inverse_field_nd(warp_r2l_inv, warp_r2l.detach(), steps=self.inverse_steps, method=self.inverse_method))
                    
                    if len(level_syn_losses) >= 10 and (epoch % 5 == 4 or epoch == curr_syn_epochs - 1):
                        recent_losses = [l.item() if hasattr(l, 'item') else l for l in level_syn_losses[-10:]]
                        if check_convergence(recent_losses, window_size=10, slope_threshold=1e-5):
                            break
                    
            warp_l2r.requires_grad_(False)
            warp_r2l.requires_grad_(False)
            
        with torch.no_grad():
            # Interpolate midpoint fields to target grid resolution
            w_l2r = F.interpolate(torch.movedim(warp_l2r, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            w_r2l = F.interpolate(torch.movedim(warp_r2l, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            w_l2r_inv = F.interpolate(torch.movedim(warp_l2r_inv, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            w_r2l_inv = F.interpolate(torch.movedim(warp_r2l_inv, -1, 1), size=self.grid_shape, mode='bilinear' if dim==2 else 'trilinear', align_corners=True).movedim(1, -1)
            
            grids_full = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in self.grid_shape]
            identity_full = torch.stack(list(reversed(torch.meshgrid(*grids_full, indexing='ij'))), dim=-1).unsqueeze(0)
            
            # Compose midpoint fields into full endpoint-to-endpoint fields
            # Full L2R (Moving -> Fixed) = phi_r2l(phi_l2r_inv(x))
            full_l2r = compose_grids(identity_full + w_r2l, identity_full + w_l2r_inv)
            # Full R2L (Fixed -> Moving) = phi_l2r(phi_r2l_inv(x))
            full_r2l = compose_grids(identity_full + w_l2r, identity_full + w_r2l_inv)
            
            self.warp_l2r = nn.Parameter(full_l2r - identity_full)
            self.warp_r2l = nn.Parameter(full_r2l - identity_full)
            
            # Compute exact inverses for the full composed fields
            self.warp_l2r_inv = nn.Parameter(update_inverse_field_nd(self.warp_l2r.data, torch.zeros_like(self.warp_l2r.data), steps=self.inverse_steps, method=self.inverse_method))
            self.warp_r2l_inv = nn.Parameter(update_inverse_field_nd(self.warp_r2l.data, torch.zeros_like(self.warp_r2l.data), steps=self.inverse_steps, method=self.inverse_method))
            
            # Convert all logged losses to floats in a single batch
            self.affine_losses = [l.item() if hasattr(l, 'item') else l for l in self.affine_losses]
            self.syn_losses = [l.item() if hasattr(l, 'item') else l for l in self.syn_losses]

    def forward(self, moving_image, fixed_image=None):
        """
        Warps the moving image using the affine pre-alignment and dense forward field.
        """
        device = moving_image.device
        dtype = moving_image.dtype
        spatial_shape = moving_image.shape[2:]
        
        grid_affine = self.get_affine_grid(spatial_shape, device)
        
        warp_resampled = F.interpolate(
            torch.movedim(self.warp_l2r, -1, 1), 
            size=spatial_shape, 
            mode='bilinear' if self.dim == 2 else 'trilinear', 
            align_corners=True
        )
        warp_resampled = torch.movedim(warp_resampled, 1, -1)
        
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        phi_l2r = identity + warp_resampled
        composed_grid = compose_grids(grid_affine, phi_l2r)
        
        return F.grid_sample(moving_image, composed_grid, padding_mode='border', align_corners=True)

    def forward_inverse(self, fixed_image):
        """
        Warps the fixed image using the inverse dense warp and inverse affine transform.
        """
        device = fixed_image.device
        dtype = fixed_image.dtype
        spatial_shape = fixed_image.shape[2:]
        
        warp_resampled = F.interpolate(
            torch.movedim(self.warp_r2l, -1, 1), 
            size=spatial_shape, 
            mode='bilinear' if self.dim == 2 else 'trilinear', 
            align_corners=True
        )
        warp_resampled = torch.movedim(warp_resampled, 1, -1)
        
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        phi_r2l = identity + warp_resampled
        grid_affine_inv = self.get_inverse_affine_grid(spatial_shape, device)
        composed_grid = compose_grids(grid_affine_inv, phi_r2l)
        
        return F.grid_sample(fixed_image, composed_grid, padding_mode='border', align_corners=True)

    def get_forward_transform(self, fixed_metadata):
        """Returns the fully interoperable SyNToTransform object for the forward (moving->fixed) mapping."""
        device = self.warp_l2r.device
        grid_affine = self.get_affine_grid(self.grid_shape, device)
        return SyNToTransform(
            affine_grid=grid_affine, 
            warp_field=self.warp_l2r, 
            metadata=fixed_metadata, 
            device=device
        )

    def get_inverse_transform(self, moving_metadata):
        """Returns the fully interoperable SyNToTransform object for the inverse (fixed->moving) mapping."""
        device = self.warp_r2l.device
        grid_affine_inv = self.get_inverse_affine_grid(self.grid_shape, device)
        return SyNToTransform(
            affine_grid=grid_affine_inv, 
            warp_field=self.warp_r2l, 
            metadata=moving_metadata, 
            device=device
        )

def grid_to_physical_affine(T_grid, fixed, moving):
    dim = len(fixed.shape)
    Nx = np.array(list(reversed(fixed.shape)), dtype=np.float32)
    Ny = np.array(list(reversed(moving.shape)), dtype=np.float32)
    
    Sx = np.array(fixed.spacing)
    Sy = np.array(moving.spacing)
    Ox = np.array(fixed.origin)
    Oy = np.array(moving.origin)
    Dx = np.array(fixed.direction)
    Dy = np.array(moving.direction)
    
    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    
    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0
    
    # Calculate grid-to-physical linear conversion mapping
    # Wx transforms physical x to grid u_x: u_x = Wx @ x_phys + bx
    Kx_inv = np.linalg.inv(Kx)
    Sx_inv = np.linalg.inv(np.diag(Sx))
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx
    
    # Vy transforms grid u_y to physical y: y_phys = Vy @ u_y + cy
    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
    
    A_grid = T_grid[:dim, :dim]
    t_grid = T_grid[:dim, dim]
    
    # Permute from PyTorch/JAX axis ordering (z, y, x) to ITK physical ordering (x, y, z)
    P = np.eye(dim)[::-1]
    A_grid = P @ A_grid @ P
    t_grid = P @ t_grid
    
    # Compute physical transform parameters
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy
    return M_phys, t_phys


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


def registration(
    fixed,
    moving,
    type_of_transform='SyNTo',
    aff_metric='mattes_mi',
    aff_sampling=32,
    syn_metric='lncc',
    syn_sampling=4,
    reg_iterations=None,
    affine_iterations=None,
    grad_step=0.6,
    flow_sigma=1.732,
    total_sigma=0.0,
    verbose=False,
    backend='pytorch',
    initial_transform=None,
    levels=None,
    sampling_percentage=None,
    vgg_layers=[8],
    vgg_mode='patch_walk',
    vgg_patch_size=32,
    vgg_num_patches=8,
    vgg_lncc_window_size=9,
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
        Metric for affine registration.
    aff_sampling : int
        Number of bins for Mattes MI.
    syn_metric : str
        Metric for SyN registration ('lncc' or 'mattes_mi').
    syn_sampling : int
        LNCC radius (window_size = 2 * syn_sampling + 1).
    reg_iterations : list of int or None
        Number of iterations per level for SyN stage.
    affine_iterations : list of int or None
        Number of iterations per level for Affine stage.
    grad_step : float
        CFL voxel bound step size (default 0.2).
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
    """
    import tempfile
    import ants
    import numpy as np
    
    # 1. Extract physical properties
    dim = fixed.dimension
    grid_shape = fixed.shape
    spacing = fixed.spacing
    direction = fixed.direction
    
    # Apply initial transform if provided
    tx_list = []
    if initial_transform is not None:
        tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        moving_reg = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_list)
        if affine_iterations is None:
            affine_iterations = [0]
    else:
        moving_reg = moving
    
    # 2. Extract and Normalize numpy arrays
    fi_np = fixed.numpy()
    mi_np = moving_reg.numpy()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    
    # Convert spacing order (X, Y) -> (Y, X) for registration grid alignment
    sp_ordered = tuple(reversed(spacing))
    
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
        
    inverse_steps = kwargs.get('inverse_steps', 5)
    inverse_method = kwargs.get('inverse_method', 'fixed_point')
    vgg_layers = kwargs.get('vgg_layers', vgg_layers)
    vgg_patch_size = kwargs.get('vgg_patch_size', vgg_patch_size)
    vgg_num_patches = kwargs.get('vgg_num_patches', vgg_num_patches)
    vgg_mode = kwargs.get('vgg_mode', vgg_mode)
    vgg_lncc_window_size = kwargs.get('vgg_lncc_window_size', vgg_lncc_window_size)
        
    # 3. Initialize and fit the model
    if backend == 'pytorch':
        from .syn import SyNTo as SyNToPy
        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        model = SyNToPy(
            dim=dim, grid_shape=grid_shape, spacing=sp_ordered, direction=direction,
            fluid_sigma=flow_sigma, elastic_sigma=total_sigma, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps
        ).to(device)
    elif backend == 'jax':
        from .syn_jax import SyNTo as SyNToJax
        import jax.numpy as jnp
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *grid_shape)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *grid_shape)
        
        model = SyNToJax(
            dim=dim, grid_shape=grid_shape, spacing=sp_ordered, direction=direction,
            fluid_sigma=flow_sigma, elastic_sigma=total_sigma, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    if backend == 'pytorch':
        model.fit(
            I_tensor, J_tensor,
            levels=levels if levels is not None else ([8, 4, 2, 1] if dim == 2 else [4, 2, 1]),
            epochs_per_level=reg_iterations if reg_iterations is not None else [100, 100, 100, 50],
            affine_epochs=affine_iterations if affine_iterations is not None else [100, 50, 50, 20],
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            lncc_radius=syn_sampling,
            mattes_bins=aff_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size
        )
    else:
        model.fit(
            I_tensor, J_tensor,
            levels=levels if levels is not None else ([8, 4, 2, 1] if dim == 2 else [4, 2, 1]),
            epochs_per_level=reg_iterations if reg_iterations is not None else [100, 100, 100, 50],
            affine_epochs=affine_iterations if affine_iterations is not None else [100, 50, 50, 20],
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            lncc_radius=syn_sampling,
            mattes_bins=aff_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size
        )
    
    # 4. Save displacement fields to temp files to match ANTs file-based transforms
    fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
    inv_file = tempfile.NamedTemporaryFile(suffix='_inv.nii.gz', delete=False).name
    
    affine_file = None
    affine_inv_file = None
    
    if backend == 'pytorch':
        with torch.no_grad():
            warp_l2r = model.warp_l2r.cpu().numpy()
            warp_r2l = model.warp_r2l.cpu().numpy()
            
            if sum(affine_iterations) > 0:
                # Convert internal grid affine to physical ITK AffineTransform
                T_grid = model.affine.get_matrix().cpu().numpy()
                print(f"[pytorch] T_grid:\n", T_grid)
                M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving_reg)
                
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
        from .syn_jax import get_affine_matrix_jax
        
        warp_l2r = np.array(model.warp_l2r)
        warp_r2l = np.array(model.warp_r2l)
        
        if sum(affine_iterations) > 0:
            T_grid = get_affine_matrix_jax(model.affine_params, dim, model.transform_type)
            T_grid = np.array(T_grid)
            print(f"[jax] T_grid:\n", T_grid)
            M_phys, t_phys = grid_to_physical_affine(T_grid, fixed, moving_reg)
            
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
        
    # Scale from [-1, 1] coordinate range to physical mm displacements in physical component order
    disp_l2r_components = []
    disp_r2l_components = []
    for k in range(dim):
        c_idx = dim - 1 - k
        axis_idx = dim - 1 - k
        N = grid_shape[axis_idx]
        sp = spacing[k]
        
        disp_l2r_c = warp_l2r[0, ..., c_idx] * ((N - 1) / 2.0) * sp
        disp_r2l_c = warp_r2l[0, ..., c_idx] * ((N - 1) / 2.0) * sp
        
        disp_l2r_components.append(disp_l2r_c)
        disp_r2l_components.append(disp_r2l_c)
        
    if sum(reg_iterations) > 0:
        disp_l2r = np.stack(disp_l2r_components, axis=-1).astype(np.float32)
        disp_r2l = np.stack(disp_r2l_components, axis=-1).astype(np.float32)
        
        fwd_img = ants.from_numpy(disp_l2r, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction, has_components=True)
        inv_img = ants.from_numpy(disp_r2l, origin=moving.origin, spacing=moving.spacing, direction=moving.direction, has_components=True)
        
        ants.image_write(fwd_img, fwd_file)
        ants.image_write(inv_img, inv_file)
        
        if initial_transform is not None:
            if affine_file is not None:
                fwd_transforms = [fwd_file, affine_file] + tx_list
                inv_transforms = tx_list + [affine_inv_file, inv_file]
                whichtoinvert_inv = [True] * len(tx_list) + [False, False]
            else:
                fwd_transforms = [fwd_file] + tx_list
                inv_transforms = tx_list + [inv_file]
                whichtoinvert_inv = [True] * len(tx_list) + [False]
        elif affine_file is not None:
            fwd_transforms = [fwd_file, affine_file]
            inv_transforms = [affine_inv_file, inv_file]
            whichtoinvert_inv = [False, False]
        else:
            fwd_transforms = [fwd_file]
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]
    else:
        if initial_transform is not None:
            if affine_file is not None:
                fwd_transforms = [affine_file] + tx_list
                inv_transforms = tx_list + [affine_inv_file]
                whichtoinvert_inv = [True] * len(tx_list) + [False]
            else:
                fwd_transforms = tx_list
                inv_transforms = tx_list
                whichtoinvert_inv = [True] * len(tx_list)
        elif affine_file is not None:
            fwd_transforms = [affine_file]
            inv_transforms = [affine_inv_file]
            whichtoinvert_inv = [False]
        else:
            fwd_transforms = []
            inv_transforms = []
            whichtoinvert_inv = []
    
    # 6. Apply transforms to generate warped output images
    warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms)
    warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms, whichtoinvert=whichtoinvert_inv)
    
    return {
        'warpedmovout': warpedmovout,
        'warpedfixout': warpedfixout,
        'fwdtransforms': fwd_transforms,
        'invtransforms': inv_transforms,
        'syn_losses': list(model.syn_losses) if hasattr(model, 'syn_losses') else [],
        'affine_losses': list(model.affine_losses) if hasattr(model, 'affine_losses') else []
    }
