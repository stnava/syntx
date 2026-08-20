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
from .core.affine import (
    get_rotation_matrix,
    HierarchicalAffine,
    _grid_to_physical_affine_torch_yfirst,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    grid_to_physical_affine,
    parse_ants_affine,
    compute_initial_grid,
)
from .core.grid import (
    grid_sample_bspline_torch,
    _image_spatial_gradient,
    AnalyticalGridSample,
    grid_sample_nd,
    compose_grids,
    _get_physical_grid_torch_yfirst,
    get_physical_grid_torch,
    _physical_to_normalized_torch_yfirst,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    prepare_mid_images_and_gradients_torch,
)
from .core.smoothing import (
    separable_gaussian_filter,
    get_cached_gaussian_kernel_1d,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)
from .core.losses import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    local_ncc_loss_nd,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
)
from .core.jacobian import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_physical_jacobian_determinant,
)
from .core.inverse import (
    update_inverse_field_nd_hybrid_lm,
    integrate_time_varying_velocity_field,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
)
from .core.optimizers import (
    LARS,
    get_cfl_max_norm,
    compute_cfl_step,
    check_convergence,
)
from .core.pipeline import (
    auto_detect_device,
    normalize_and_tensorize,
    cleanup_gpu,
)
from .core.utils import (
    normalize_tensor,
)

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
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, origin=None, direction=None, fluid_sigma=3.0, elastic_sigma=0.0, transform_type='Affine', inverse_method='anderson', inverse_steps=30, in_loop_inv_steps=6, project_inverse=True, projection_frequency=1, interpolator='linear', boundary_suppression_thresh=None, image_grad_clip=0.0, antisymmetric=True, use_ants_pseudo_gradient=False, inv_tolerance=None, dual_gradient=False, dual_gradient_weight=0.5):
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
        self.dual_gradient = dual_gradient
        self.dual_gradient_weight = dual_gradient_weight
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
        from .core.smoothing import apply_sobolev_green_operator
        return apply_sobolev_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha, border_width=border_width, **kwargs)

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        from .core.smoothing import apply_dsti_green_operator
        return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        from .core.smoothing import apply_dsti1_green_operator
        return apply_dsti1_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)


    def fit(self, fixed_image, moving_image, levels=[4, 2, 1], epochs_per_level=[100, 100, 50], 
            affine_epochs=[100, 50, 20], affine_lr=1e-2, cfl_voxels=0.15, 
            similarity_metric='lncc', use_analytical_gradients=False,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None,
            vgg_layers=[4], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='lncc_3d',
            vgg_lncc_window_size=9, syn_metric_weights=None, initial_grid=None, interpolator=None, **kwargs):
        """
        Runs the full native pre-alignment and SyN multi-resolution optimization loop.
        fixed_image: (1, 1, *spatial)
        moving_image: (1, 1, *spatial)
        """
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
                if verbose >= 2:
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
                    if verbose and (epoch % 10 == 0 or epoch == curr_affine_epochs - 1 or verbose >= 2):
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


                
                curr_spacing_fixed_zyx = torch.tensor(list(reversed(curr_spacing_fixed)), device=device, dtype=dtype)
                curr_spacing_fixed_xyz = torch.tensor(curr_spacing_fixed, device=device, dtype=dtype)
                
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

                dual_gradient = kwargs.get('dual_gradient', getattr(self, 'dual_gradient', False))
                dual_w = float(kwargs.get('dual_gradient_weight', getattr(self, 'dual_gradient_weight', 0.5)))

                if dual_gradient:
                    # 1. Analytical Pseudo-Gradient Branch
                    I_mid_det = I_mid.detach().requires_grad_(True)
                    J_mid_det = J_mid.detach().requires_grad_(True)
                    
                    loss_a = 0.0
                    metric_losses_dict = {}
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                        try:
                            val_loss_a = fn(I_mid_det, J_mid_det, mask=in_bounds_mask, uag=True)
                        except TypeError:
                            try:
                                val_loss_a = fn(I_mid_det, J_mid_det, mask=in_bounds_mask)
                            except TypeError:
                                val_loss_a = fn(I_mid_det, J_mid_det)
                        loss_a += weight * val_loss_a
                        metric_losses_dict[name] = val_loss_a.item()
                        
                    loss_a.backward()
                    g_im = I_mid_det.grad if I_mid_det.grad is not None else torch.zeros_like(I_mid_det)
                    g_jm = J_mid_det.grad if J_mid_det.grad is not None else torch.zeros_like(J_mid_det)
                    
                    with torch.no_grad():
                        if self.image_grad_clip is not None and self.image_grad_clip > 0:
                            mult = float(self.image_grad_clip)
                            norm_I = torch.sqrt(torch.sum(grad_I_mid_sampled**2, dim=-1, keepdim=True) + 1e-16)
                            norm_J = torch.sqrt(torch.sum(grad_J_mid_sampled**2, dim=-1, keepdim=True) + 1e-16)
                            max_I = mult * norm_I.mean()
                            max_J = mult * norm_J.mean()
                            grad_I_mid_sampled = torch.where(norm_I > max_I, grad_I_mid_sampled * max_I / norm_I, grad_I_mid_sampled)
                            grad_J_mid_sampled = torch.where(norm_J > max_J, grad_J_mid_sampled * max_J / norm_J, grad_J_mid_sampled)

                        grad_l_analytic = (g_im.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                        grad_r_analytic = (g_jm.movedim(1, -1) * grad_J_mid_sampled).contiguous()

                    # 2. End-to-End Autograd Branch
                    if warp_l2r.grad is not None:
                        warp_l2r.grad = None
                    if warp_r2l.grad is not None:
                        warp_r2l.grad = None

                    loss_auto = 0.0
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                        try:
                            val_loss_auto = fn(I_mid, J_mid, mask=in_bounds_mask, uag=False)
                        except TypeError:
                            try:
                                val_loss_auto = fn(I_mid, J_mid, mask=in_bounds_mask)
                            except TypeError:
                                val_loss_auto = fn(I_mid, J_mid)
                        loss_auto += weight * val_loss_auto

                    loss_auto.backward()
                    loss_val = loss_auto.item()

                    autograd_scale_fixed = torch.flip((fixed_shape_t - 1.0) * fixed_spacing_t / 2.0, dims=[0])
                    autograd_scale_moving = torch.flip((moving_shape_t - 1.0) * moving_spacing_t / 2.0, dims=[0])
                    grad_l_autograd = warp_l2r.grad * autograd_scale_fixed
                    grad_r_autograd = warp_r2l.grad * autograd_scale_moving

                    # 3. Dual-Gradient Convex Combination (Averaging)
                    with torch.no_grad():
                        warp_l2r.grad = (1.0 - dual_w) * grad_l_analytic + dual_w * grad_l_autograd
                        warp_r2l.grad = (1.0 - dual_w) * grad_r_analytic + dual_w * grad_r_autograd

                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)

                elif use_analytical_gradients:
                    I_mid_det = I_mid.detach().requires_grad_(True)
                    J_mid_det = J_mid.detach().requires_grad_(True)
                    
                    loss = 0.0
                    metric_losses_dict = {}
                    if verbose >= 2:
                        print(f"DEBUG PyTorch epoch {epoch} I_mid min/max: {I_mid_det.min().item()} {I_mid_det.max().item()} mean: {I_mid_det.mean().item()} var: {I_mid_det.var().item()}")
                        print(f"DEBUG PyTorch epoch {epoch} J_mid min/max: {J_mid_det.min().item()} {J_mid_det.max().item()} mean: {J_mid_det.mean().item()} var: {J_mid_det.var().item()}")
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
                    if verbose >= 2:
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
                            if verbose >= 2:
                                print(f"DEBUG PyTorch max_I: {max_I.item()}, max_J: {max_J.item()}")
                            grad_I_mid_sampled = torch.where(norm_I > max_I, grad_I_mid_sampled * max_I / norm_I, grad_I_mid_sampled)
                            grad_J_mid_sampled = torch.where(norm_J > max_J, grad_J_mid_sampled * max_J / norm_J, grad_J_mid_sampled)

                        grad_l_raw = (g_im.movedim(1, -1) * grad_I_mid_sampled).contiguous()
                        warp_l2r.grad = grad_l_raw
                        if verbose >= 2:
                            print(f"DEBUG PyTorch L{level_idx} E{epoch} grad_l_raw max: {grad_l_raw.abs().max().item()}")
                            print(f"DEBUG PyTorch L{level_idx} E{epoch} grad_l_raw L2 norm max: {torch.sqrt(torch.sum((grad_l_raw / curr_spacing_fixed_xyz)**2, dim=-1)).max().item()}")

                        grad_r_raw = (g_jm.movedim(1, -1) * grad_J_mid_sampled).contiguous()
                        warp_r2l.grad = grad_r_raw

                else:
                    loss = 0.0
                    metric_losses_dict = {}
                    dev_type = 'cuda' if 'cuda' in str(device) else ('mps' if 'mps' in str(device) else 'cpu')
                    use_amp = bool(kwargs.get('amp', True)) and (dev_type in ('cuda', 'mps'))
                    amp_dtype = torch.float16

                    with torch.amp.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                        for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                            try:
                                val_loss = fn(I_mid, J_mid, mask=in_bounds_mask)
                            except TypeError:
                                val_loss = fn(I_mid, J_mid)

                            loss += weight * val_loss
                            metric_losses_dict[name] = val_loss.item()
                        
                    loss.backward()
                    loss_val = loss.item()
                    
                    # Rescale autograd gradients from normalized grid space [-1, 1] to physical mm
                    # coords_norm = (x_phys - origin) * 2 / (spacing * (shape - 1)) - 1
                    # dLoss/dx_phys = dLoss/dcoords_norm * 2 / (spacing * (shape - 1))
                    # Rescaling by (shape - 1) * spacing / 2 converts back to consistent physical displacement gradient:
                    autograd_scale_fixed = torch.flip((fixed_shape_t - 1.0) * fixed_spacing_t / 2.0, dims=[0])
                    autograd_scale_moving = torch.flip((moving_shape_t - 1.0) * moving_spacing_t / 2.0, dims=[0])
                    if warp_l2r.grad is not None:
                        warp_l2r.grad = warp_l2r.grad * autograd_scale_fixed
                    if warp_r2l.grad is not None:
                        warp_r2l.grad = warp_r2l.grad * autograd_scale_moving
                        
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)

                if isinstance(self.fluid_sigma, (list, tuple)):
                    curr_fluid_var = self.fluid_sigma[min(level_idx, len(self.fluid_sigma) - 1)]
                else:
                    curr_fluid_var = self.fluid_sigma
                curr_fluid_sig = float(curr_fluid_var)
                    
                regularizer = kwargs.get('regularizer', kwargs.get('kernel_type', 'gaussian'))
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



                    grad_l_voxel = grad_l / curr_spacing_fixed_xyz  # convert to voxel units
                    grad_r_voxel = grad_r / curr_spacing_fixed_xyz
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
                    
                    
                    
                    if verbose and (epoch % 10 == 0 or epoch == curr_syn_epochs - 1 or verbose >= 2):
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
                            grad_l_voxel = grad_l / curr_spacing_fixed_xyz
                            grad_r_voxel = grad_r / curr_spacing_fixed_xyz
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
                            
                            if verbose >= 2:
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
    
    from .core.pipeline import normalize_and_tensorize, auto_detect_device, cleanup_gpu
    
    # 2. Winsorize and Normalize numpy arrays
    I_tensor_unused, J_tensor_unused = normalize_and_tensorize(
        fixed, moving_reg, winsorize_quantiles=kwargs.get('winsorize_quantiles', None), backend=backend
    )
    # Re-fetch normalized arrays since SyNTo setup might still rely on numpy logic initially
    fi_np = fixed.numpy()
    mi_np = moving_reg.numpy()
    if kwargs.get('winsorize_quantiles', None) is not None:
        wq = kwargs.get('winsorize_quantiles')
        lo_f, hi_f = np.quantile(fi_np[fi_np > 0], wq) if (fi_np > 0).any() else (fi_np.min(), fi_np.max())
        fi_np = np.clip(fi_np, lo_f, hi_f)
        lo_m, hi_m = np.quantile(mi_np[mi_np > 0], wq) if (mi_np > 0).any() else (mi_np.min(), mi_np.max())
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
        if initial_transform is not None or initial_grid is not None:
            affine_iterations = [0] * levels_len
        else:
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
    if isinstance(flow_sigma, (list, tuple)):
        fluid_sigma_actual = [math.sqrt(s) if s > 0 else 0.0 for s in flow_sigma]
    else:
        fluid_sigma_actual = math.sqrt(flow_sigma) if flow_sigma > 0 else 0.0
    elastic_sigma_actual = math.sqrt(total_sigma) if total_sigma > 0 else 0.0
    
    # 3. Initialize and fit the model
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    grid_shape_zyx = tuple(reversed(grid_shape))
    use_analytical = kwargs.get('use_analytical_gradients', kwargs.get('use_ants_pseudo_gradient', False))
    if backend == 'pytorch':
        from .syn import SyNTo as SyNToPy
        import torch
        device = auto_detect_device(backend='pytorch', requested_device=kwargs.get('device', None))
        
        I_tensor, J_tensor = normalize_and_tensorize(
            fixed, moving_reg, winsorize_quantiles=kwargs.get('winsorize_quantiles', None),
            backend='pytorch', device=device
        )
        
        model = SyNToPy(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed.origin, direction=direction,
            fluid_sigma=fluid_sigma_actual, elastic_sigma=elastic_sigma_actual, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps, in_loop_inv_steps=kwargs.get('in_loop_inv_steps', 6), project_inverse=project_inverse,
            use_ants_pseudo_gradient=use_analytical,
            projection_frequency=projection_frequency, interpolator=interpolator,
            boundary_suppression_thresh=boundary_suppression_thresh,
            image_grad_clip=image_grad_clip,
            antisymmetric=antisymmetric,
            inv_tolerance=inv_tolerance,
            dual_gradient=kwargs.get('dual_gradient', False),
            dual_gradient_weight=kwargs.get('dual_gradient_weight', 0.5)
        ).to(device)
        model.formulation = kwargs.get('formulation', 'eulerian')
        model.smooth_in_deformed_space = kwargs.get('smooth_in_deformed_space', False)
        model.kernel_type = kwargs.get('kernel_type', 'bessel')
    elif backend == 'jax':
        from .syn_jax import SyNTo as SyNToJax
        import jax.numpy as jnp
        I_tensor, J_tensor = normalize_and_tensorize(
            fixed, moving_reg, winsorize_quantiles=kwargs.get('winsorize_quantiles', None),
            backend='jax'
        )
        
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
            regularizer=kwargs.get('regularizer', kwargs.get('kernel_type', 'gaussian')),
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=use_analytical,
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
            regularizer=kwargs.get('regularizer', kwargs.get('kernel_type', 'gaussian')),
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=use_analytical,
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
                if verbose >= 2:
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
            if verbose >= 2:
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
            disp_l2r_t = disp_l2r[..., ::-1].copy()
            disp_r2l_t = disp_r2l[..., ::-1].copy()
        elif dim == 3:
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
            levels=levels_to_use,
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
            antisymmetric=antisymmetric,
            use_analytical_gradients=use_analytical
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass
    
    cleanup_gpu(device=device if 'device' in locals() else None, backend=backend)

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
        if err_vals_max:
            metrics['inverse_identity_max_error'] = float(np.max(err_vals_max))
            
    res['metrics'] = metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()
    return res


from .core.utils import normalize_tensor

from .viz import (
    extract_2d_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
    render_input_pair_figure
)



