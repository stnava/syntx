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

import os
import time
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc

from .spatial import (
    reverse_components,
    reverse_metadata,
    itk_shape_to_tensor_shape,
    compute_autograd_physical_scale,
    disp_tensor_to_itk,
    export_ants_affine_transform,
    grid_to_physical_affine,
    tensor_to_image,
    image_to_tensor,
)
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
    get_similarity_loss,
    SimilarityLossConfig,
)
from .core.jacobian import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_jacobian_hinge_penalty,
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
from .core.utils import check_loss_collapse
from .core.regularizers import get_regularizer

_CONTINUUM_REGS = frozenset({
    'solenoidal', 'leray', 'incompressible', 'solenoidal_sobolev',
    'div_curl', 'helmholtz',
    'navier', 'stokes', 'elastic',
    'masked_incompressible',
    'hyperelastic', 'simo_pister', 'log_jacobian',
    'beltrami', 'quasiconformal', 'conformal',
    'poroelastic', 'darcy_stokes', 'biot',
})

class TriPlanarVGG3DLoss(nn.Module):
    """VGG19 perceptual similarity for 2-D / 3-D images (3-D: features of axial, coronal and
    sagittal slices). Used by ``syntx.syn`` for ``syn_metric='vgg19'``; see ``__init__`` for
    the modes. ``forward(input, target)`` returns a scalar loss (lower is more similar)."""
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
        """Scalar feature-space loss between ``input_nd`` and ``target_nd`` (B, C, *spatial); lower = more similar."""
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



_ADAM_FAMILY = ('adam', 'reg_adam', 'regadam', 'sobolev_adam', 'gaussian_adam', 'dsti_adam')
REGADAM_DEFAULT_LR = 0.5  # max per-step displacement = optimizer_lr * grad_step
# optimizers with an update step in SyNTo.fit (PyTorch) and syntx.syn_jax.SyNTo.fit (JAX)
SYN_OPTIMIZERS_PYTORCH = frozenset({'cfl', 'rprop', 'sgd'} | set(_ADAM_FAMILY))
SYN_OPTIMIZERS_JAX = frozenset({'cfl', 'rprop', 'sgd', 'adam', 'lbfgs'})


def resolve_optimizer_lr(optimizer, optimizer_lr=None):
    """Default learning rate per optimizer: 0.5 for the Adam family (RegAdam), 1e-3 otherwise."""
    if optimizer_lr is not None:
        return optimizer_lr
    return REGADAM_DEFAULT_LR if str(optimizer).lower() in _ADAM_FAMILY else 1e-3


class SyNTo(nn.Module):
    """
    The PyTorch model behind ``syntx.syn``: an affine plus two half displacement fields
    (fixed side ``warp_l2r``, moving side ``warp_r2l``) meeting at a midpoint space, with
    their inverses. Most users call ``syntx.syn``; use the class directly only for custom
    pipelines (construct, then ``fit``).

    Fields are stored as ``(1, *grid_shape, dim)`` physical displacements (mm) on the fixed
    grid, tensor (z, y, x) order. After ``fit``: ``warp_l2r`` / ``warp_l2r_inv`` hold the
    composed forward map and its inverse; ``midpoint_warp_l2r``, ``midpoint_warp_r2l``,
    ``midpoint_warp_l2r_inv`` keep the half fields (used by ``syntx.liouville_determinant``).

    Parameters
    ----------
    dim : {2, 3}, default 3
    grid_shape : tuple of int, default (64, 64, 64)
        Fixed-image grid, tensor (z, y, x) order.
    spacing, origin : list of float, optional
        Fixed-image geometry, ITK (x, y, z) order. Defaults: unit spacing (None), zero origin.
    direction : array-like, optional
        Direction cosine matrix (default identity).
    fluid_sigma, elastic_sigma : float, default 3.0, 0.0
        Gaussian sigmas (mm) of the update smoothing and of the displacement smoothing.
        (``syntx.syn`` converts its variance-convention flow_sigma / total_sigma to these.)
    transform_type : {'Affine', 'Rigid', 'Translation'}, default 'Affine'
        The linear part.
    inverse_method : {'anderson', 'fixed_point'}, default 'anderson'
    inverse_steps : int, default 30
        Iterations of the final inverse solve.
    in_loop_inv_steps : int, default 6
        Iterations refreshing each half inverse after every update (``syntx.syn`` passes 10).
    inv_tolerance : float, optional
        Inverse tolerance (mm); default 0.1 x the smallest spacing.
    project_inverse : bool, default True
    projection_frequency : int, default 1
    interpolator : {'linear', 'nearestNeighbor'}, default 'linear'
    boundary_suppression_thresh : float or None
        Threshold for suppressing image gradients near the boundary. Default None (off).
    stationary_boundary : bool, default True
        Keep the displacement exactly zero on every image face (ITK
        ``EnforceStationaryBoundary``), so exported transforms round-trip at the edges.
    image_grad_clip : float, default 0.0
        Clip image-gradient magnitudes to this value (0 = off).
    antisymmetric : bool, default True
        Remove the common part of the two half updates.
    use_ants_pseudo_gradient : bool, default False
        ANTs-style pseudo-gradient of the correlation metric.
    dual_gradient, dual_gradient_weight : bool, float, default False, 0.5
        Blend in the gradient of the opposite image.
    restrict_transformation : sequence of float, optional
        Per-axis deformation weights in [0, 1], physical (x, y, z) order (see ``syntx.syn``).
    seed : int, default 42
    """
    def __init__(self, dim=3, grid_shape=(64, 64, 64), spacing=None, origin=None, direction=None, fluid_sigma=3.0, elastic_sigma=0.0, transform_type='Affine', inverse_method='anderson', inverse_steps=30, in_loop_inv_steps=6, project_inverse=True, projection_frequency=1, interpolator='linear', boundary_suppression_thresh=None, image_grad_clip=0.0, antisymmetric=True, use_ants_pseudo_gradient=False, inv_tolerance=None, dual_gradient=False, dual_gradient_weight=0.5, restrict_transformation=None, seed=42, stationary_boundary=True):
        super().__init__()
        self.dim = dim
        self.grid_shape = grid_shape

        # Per-axis deformation restriction weights, matching ants.registration's
        # `restrict_transformation`: a length-`dim` sequence of weights in [0, 1], in
        # physical XYZ order (same order as `spacing`/`origin`/`direction`). A weight of
        # 1.0 leaves that physical axis free; 0.0 fully suppresses deformation along it.
        # None (default) means no restriction -- behaviour is unchanged from before this
        # option existed. Validated eagerly so a malformed value fails at construction,
        # not silently inside the optimization loop.
        if restrict_transformation is not None:
            rt = list(restrict_transformation)
            if len(rt) != dim:
                raise ValueError(
                    f"restrict_transformation must have length {dim} (one weight per physical "
                    f"axis, XYZ order), got length {len(rt)}: {restrict_transformation!r}"
                )
            rt = [float(w) for w in rt]
            if any(w < 0.0 or w > 1.0 for w in rt):
                raise ValueError(
                    f"restrict_transformation weights must be in [0, 1], got {rt!r}"
                )
            self.restrict_transformation = tuple(rt)
        else:
            self.restrict_transformation = None
        self._restrict_mask_cache = None  # (device, dtype) -> broadcastable mask tensor
        self.seed = int(seed) if seed is not None else None
        self._rng = None
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
        self.stationary_boundary = bool(stationary_boundary)
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
        """``F.affine_grid`` sampling grid (normalised coordinates) of the affine for ``shape``."""
        theta = self.affine.get_affine_grid_matrix().unsqueeze(0)
        grid = F.affine_grid(theta, size=[1, 1] + list(shape), align_corners=True)
        return grid

    def get_inverse_affine_grid(self, shape, device):
        """Sampling grid of the inverse affine for ``shape``."""
        T = self.affine.get_matrix()
        T_inv = torch.inverse(T)
        theta_inv = T_inv[:self.dim, :self.dim + 1].unsqueeze(0)
        grid_inv = F.affine_grid(theta_inv, size=[1, 1] + list(shape), align_corners=True)
        return grid_inv

    def _get_restrict_mask(self, device, dtype):
        """Broadcastable (1,1,...,1,dim) mask of per-axis restriction weights, or None.

        Applied to the fully-regularized update field immediately before it is used to
        step the displacement field, so it takes effect regardless of which similarity
        metric, regularizer (sobolev/dsti/bspline/gaussian), or optimizer branch produced
        that update -- see the three call sites in `fit()`.
        """
        if self.restrict_transformation is None:
            return None
        cached = self._restrict_mask_cache
        if cached is not None and cached[0] == device and cached[1] == dtype:
            return cached[2]
        shape = [1] * (self.dim + 1) + [self.dim]
        # restrict_transformation is documented and passed in physical (X, Y, Z) order.
        # Tensor displacement components are stored in reverse (dz, dy, dx) order.
        mask = torch.tensor(self.restrict_transformation[::-1], device=device, dtype=dtype).view(*shape)
        self._restrict_mask_cache = (device, dtype, mask)
        return mask

    def _apply_sobolev_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """Sobolev smoothing of field ``m`` (see ``core.smoothing.apply_sobolev_green_operator``)."""
        from .core.smoothing import apply_sobolev_green_operator
        return apply_sobolev_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_dsti_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST-based (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti_green_operator``)."""
        from .core.smoothing import apply_dsti_green_operator
        return apply_dsti_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_dsti1_green_operator(self, m, fluid_sigma=3.0, alpha=None):
        """DST-I (Dirichlet) smoothing of ``m`` (``core.smoothing.apply_dsti1_green_operator``)."""
        from .core.smoothing import apply_dsti1_green_operator
        return apply_dsti1_green_operator(m, fluid_sigma=fluid_sigma, alpha=alpha)

    def _apply_bspline_operator(self, m, spacing=None, origin=None, fluid_sigma=None, **kwargs):
        """B-spline smoothing of ``m`` (``core.smoothing.smooth_displacement_field_bspline``; mesh_size / spline_distance / spline_order / enforce_stationary_boundary from kwargs)."""
        from .core.smoothing import smooth_displacement_field_bspline
        b_mesh = kwargs.get('mesh_size', None)
        b_dist = kwargs.get('spline_distance', None)
        b_fsig = fluid_sigma if fluid_sigma is not None else kwargs.get('fluid_sigma', None)
        b_bound = kwargs.get('enforce_stationary_boundary', False)
        b_order = kwargs.get('spline_order', 3)
        return smooth_displacement_field_bspline(
            m,
            spacing=spacing,
            origin=origin,
            mesh_size=b_mesh,
            spline_distance=b_dist,
            fluid_sigma=b_fsig,
            enforce_stationary_boundary=b_bound,
            order=b_order,
            coord_convention='xyz',
        )


    def fit(self, fixed_image, moving_image, levels=[4, 2, 1], epochs_per_level=[100, 100, 50],
            cfl_voxels=0.15,
            similarity_metric='cc2', use_analytical_gradients=False,
            lncc_radius=4, mattes_bins=32, sampling_percentage=None,
            vgg_layers=[4], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='lncc_3d',
            vgg_lncc_window_size=9, syn_metric_weights=None, initial_grid=None, interpolator=None, **kwargs):
        """
        Run the multi-resolution SyN optimisation (the affine is not optimised here: it is
        set beforehand, e.g. from ``syntx.robust_affine`` via ``syntx.syn``).

        Parameters
        ----------
        fixed_image, moving_image : ANTsImage or Tensor
            Tensors are ``(1, 1, *spatial)``; geometry then comes from the ``fixed_*`` /
            ``moving_*`` keywords (spacing, origin, direction), otherwise from the images.
        levels : list of int, default [4, 2, 1]
            Pyramid shrink factors.
        epochs_per_level : list of int, default [100, 100, 50]
            Iterations per level.
        cfl_voxels : float, default 0.15
            CFL step (voxels) -- ``syntx.syn`` passes ``grad_step``.
        similarity_metric : str or list, default 'cc2'
        lncc_radius : int, default 4
            Local-correlation radius (``syntx.syn`` passes ``syn_sampling``).
        mattes_bins, sampling_percentage : Mattes-MI settings (32 bins, all voxels).
        use_analytical_gradients : bool, default False
        vgg_* : deep-feature metric settings.
        syn_metric_weights : list of float, optional
            One weight per channel / metric.
        initial_grid : Tensor, optional
            Non-affine initial transform as a sampling grid.
        interpolator : str, optional
        **kwargs
            ``optimizer_type`` (default 'cfl'), ``optimizer_lr``, ``regularizer``,
            ``sobolev_alpha``, ``fast_smooth``, ``elastic_sigma``, ``device``, ``verbose``,
            ``fixed_spacing`` / ``_origin`` / ``_direction``, ``moving_*``, and the other
            advanced options of ``syntx.syn``.

        Results are stored on the model (see the class docstring); ``syn_losses`` holds the
        loss history.
        """
        _unknown = sorted(set(kwargs) - SYN_ADVANCED_OPTIONS - _SYN_INTERNAL_FIT_KEYS)
        if _unknown:
            raise TypeError(f"SyNTo.fit() got unexpected keyword(s) {_unknown}")
        self.elastic_sigma = float(kwargs.get('elastic_sigma', getattr(self, 'elastic_sigma', 0.0)))
        verbose = kwargs.get('verbose', False)
        optimizer_type = kwargs.get('optimizer_type', 'cfl')
        if optimizer_type not in SYN_OPTIMIZERS_PYTORCH:
            raise ValueError(f"unknown optimizer {optimizer_type!r} for the PyTorch SyN; use one of "
                             f"{sorted(SYN_OPTIMIZERS_PYTORCH)}")
        optimizer_lr = resolve_optimizer_lr(optimizer_type, kwargs.get('optimizer_lr'))
        fixed_spacing = kwargs.get('fixed_spacing', None)
        fixed_origin = kwargs.get('fixed_origin', None)
        fixed_direction = kwargs.get('fixed_direction', None)
        moving_spacing = kwargs.get('moving_spacing', None)
        moving_origin = kwargs.get('moving_origin', None)
        moving_direction = kwargs.get('moving_direction', None)

        if hasattr(fixed_image, 'spacing') and fixed_spacing is None:
            fixed_spacing = fixed_image.spacing
        if hasattr(fixed_image, 'origin') and fixed_origin is None:
            fixed_origin = fixed_image.origin
        if hasattr(fixed_image, 'direction') and fixed_direction is None:
            fixed_direction = fixed_image.direction

        if hasattr(moving_image, 'spacing') and moving_spacing is None:
            moving_spacing = moving_image.spacing
        if hasattr(moving_image, 'origin') and moving_origin is None:
            moving_origin = moving_image.origin
        if hasattr(moving_image, 'direction') and moving_direction is None:
            moving_direction = moving_image.direction

        fit_device = kwargs.get('device', None)
        if fit_device is None:
            if isinstance(fixed_image, torch.Tensor):
                fit_device = fixed_image.device
            elif isinstance(moving_image, torch.Tensor):
                fit_device = moving_image.device
            else:
                try:
                    fit_device = next(self.parameters()).device
                except (StopIteration, AttributeError):
                    fit_device = getattr(self, 'device', 'cpu')

        if not isinstance(fixed_image, torch.Tensor):
            if hasattr(fixed_image, 'numpy'):
                fixed_image = torch.from_numpy(fixed_image.numpy().astype(np.float32)).to(fit_device)
            else:
                fixed_image = torch.as_tensor(fixed_image, dtype=torch.float32, device=fit_device)
        else:
            fixed_image = fixed_image.to(fit_device)

        if not isinstance(moving_image, torch.Tensor):
            if hasattr(moving_image, 'numpy'):
                moving_image = torch.from_numpy(moving_image.numpy().astype(np.float32)).to(fit_device)
            else:
                moving_image = torch.as_tensor(moving_image, dtype=torch.float32, device=fit_device)
        else:
            moving_image = moving_image.to(fit_device)

        while fixed_image.ndim < self.dim + 2:
            fixed_image = fixed_image.unsqueeze(0)
        while moving_image.ndim < self.dim + 2:
            moving_image = moving_image.unsqueeze(0)

        lncc_window_size = 2 * lncc_radius + 1
        for c in range(fixed_image.shape[1]):
            i_min, i_max = torch.min(fixed_image[:, c]), torch.max(fixed_image[:, c])
            if i_max > i_min + 1e-6:
                fixed_image[:, c] = (fixed_image[:, c] - i_min) / (i_max - i_min)
            j_min, j_max = torch.min(moving_image[:, c]), torch.max(moving_image[:, c])
            if j_max > j_min + 1e-6:
                moving_image[:, c] = (moving_image[:, c] - j_min) / (j_max - j_min)
        
        device = fixed_image.device
        dtype = fixed_image.dtype

        if 'seed' in kwargs:
            seed_arg = kwargs.pop('seed')
            self.seed = int(seed_arg) if seed_arg is not None else None
        if self.seed is not None:
            self._rng = torch.Generator(device=device).manual_seed(self.seed)
        else:
            self._rng = None
        dim = self.dim
        spatial_shape = fixed_image.shape[2:]
        
        if fixed_spacing is None:
            fixed_spacing = self.spacing if self.spacing is not None else [1.0] * self.dim
            
        if fixed_origin is None:
            fixed_origin = [0.0] * self.dim
            
        if fixed_direction is None:
            fixed_direction = np.eye(self.dim)
            
        if moving_spacing is None:
            moving_spacing = [1.0] * self.dim
            

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
                
                if init_M_phys is None:
                    # 2. Compute Foreground (intensity-weighted) centers on channel 0
                    fixed_pos = torch.clamp(fixed_image[:, 0], min=0.0)
                    moving_pos = torch.clamp(moving_image[:, 0], min=0.0)
                    sum_fixed = fixed_pos.sum()
                    sum_moving = moving_pos.sum()
                    
                    if sum_fixed > 1e-5 and sum_moving > 1e-5:
                        grids_f = [torch.arange(s, device=device, dtype=dtype) for s in fixed_image.shape[2:]]
                        meshgrid_f = torch.meshgrid(*grids_f, indexing='ij')
                        idxs_f = torch.stack(list(reversed(meshgrid_f)), dim=-1)
                        
                        grids_m = [torch.arange(s, device=device, dtype=dtype) for s in moving_image.shape[2:]]
                        meshgrid_m = torch.meshgrid(*grids_m, indexing='ij')
                        idxs_m = torch.stack(list(reversed(meshgrid_m)), dim=-1)
                        
                        com_fixed_voxel = torch.sum(fixed_pos.squeeze(0).unsqueeze(-1) * idxs_f, dim=list(range(dim))) / sum_fixed
                        com_moving_voxel = torch.sum(moving_pos.squeeze(0).unsqueeze(-1) * idxs_m, dim=list(range(dim))) / sum_moving
                        
                        com_fixed_fg = Dx_t @ (Sx_t * com_fixed_voxel) + Ox_t
                        com_moving_fg = Dy_t @ (Sy_t * com_moving_voxel) + Oy_t
                        
                        t_fg = com_moving_fg - com_fixed_fg
                    else:
                        t_fg = t_fov
                    
                    down_shape = tuple(max(8, s // 4) for s in fixed_image.shape[2:])
                    I_down = F.interpolate(fixed_image[:, 0:1], size=down_shape, mode='trilinear' if dim == 3 else 'bilinear', align_corners=True)
                    J_down = F.interpolate(moving_image[:, 0:1], size=down_shape, mode='trilinear' if dim == 3 else 'bilinear', align_corners=True)
                    down_spacing = [sp * (orig - 1) / (down - 1) if down > 1 else sp for sp, orig, down in zip(fixed_spacing, reversed(fixed_image.shape[2:]), reversed(down_shape))]
                    X_down = get_physical_grid_torch(down_shape, down_spacing, fixed_origin, fixed_direction, device=device, dtype=dtype)
                    
                    com_loss_fn = get_similarity_loss('lncc', window_size=5).to(device=device)

                    def eval_translation(t_candidate):
                        t_candidate_zyx = reverse_components(t_candidate)
                        y_phys = X_down + t_candidate_zyx
                        y_norm = physical_to_normalized_torch(y_phys, moving_image.shape[2:], moving_spacing, moving_origin, moving_direction)
                        J_warped = grid_sample_nd(J_down, y_norm, padding_mode='zeros', align_corners=True, interpolator='linear')
                        
                        return com_loss_fn(I_down, J_warped).item()
                    
                    loss_fov = eval_translation(t_fov)
                    loss_fg = eval_translation(t_fg)
                    if verbose >= 2:
                        print(f"[CoM Init] t_fov: {t_fov.data.cpu().numpy()}, loss_fov: {loss_fov:.4f}")
                        print(f"[CoM Init] t_fg: {t_fg.data.cpu().numpy()}, loss_fg: {loss_fg:.4f}")
                    
                    best_t = t_fov if loss_fov < loss_fg else t_fg
                else:
                    best_t = None
                
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

        if syn_metric_weights is None and 'metric_weights' in kwargs:
            syn_metric_weights = kwargs.get('metric_weights')
        self.syn_metric_weights = syn_metric_weights
        self.metric_weights = syn_metric_weights if syn_metric_weights is not None else [1.0] * len(self.metrics)
        self.loss_functions = []
        
        # Determine if analytical gradients are viable
        if True:
            for metric in self.metrics:
                if isinstance(metric, str):
                    m_str = metric.lower()
                    if m_str in ['mattes_mi', 'mattes', 'mi', 'mmi'] or m_str.startswith(('mattes', 'mi_', 'mmi_')):
                        use_analytical_gradients = False
                    elif 'dinov2' in m_str or 'dino' in m_str:
                        print(f"Warning: Metric '{metric}' does not support analytical gradients well. Falling back to autograd.")
                        use_analytical_gradients = False
                        break
        kwargs['use_analytical_gradients'] = use_analytical_gradients
        
        for metric in self.metrics:
            if isinstance(metric, (str, SimilarityLossConfig)):
                cur_kwargs = dict(kwargs)
                cur_kwargs['window_size'] = lncc_window_size
                cur_kwargs['sampling_percentage'] = sampling_percentage
                cur_kwargs['use_analytical_gradients'] = use_analytical_gradients
                cur_kwargs['fixed_range'] = (0.0, 1.0)
                cur_kwargs['spacing'] = fixed_spacing
                cur_kwargs['device'] = device

                cur_vgg_mode = kwargs.get('vgg_mode', vgg_mode)
                cur_vgg_layers = kwargs.get('vgg_layers', vgg_layers)
                cur_vgg_window = kwargs.get('vgg_lncc_window_size', vgg_lncc_window_size)

                m_str = str(metric.metric if isinstance(metric, SimilarityLossConfig) else metric).lower()
                is_mattes = m_str in ('mattes', 'mattes_mi', 'mi', 'mmi') or m_str.startswith(('mattes_', 'mi_', 'mmi_'))
                if is_mattes and '_' in m_str:
                    parts = m_str.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        cur_kwargs['num_bins'] = int(parts[-1])
                    elif mattes_bins is not None:
                        cur_kwargs['mattes_bins'] = mattes_bins
                elif mattes_bins is not None:
                    cur_kwargs['mattes_bins'] = mattes_bins
                if m_str.startswith(('vgg', 'dino', 'resnet', 'swin')):
                    if m_str == 'vgg_4_lncc':
                        cur_vgg_layers = [4]
                        if dim == 3:
                            cur_vgg_mode = 'lncc_3d'
                    elif m_str.startswith('vgg_'):
                        parts = m_str.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]
                            if parts[2] == 'lncc':
                                cur_vgg_mode = 'lncc_3d' if dim == 3 else 'lncc'
                    elif m_str == 'dino_2_lncc':
                        cur_vgg_layers = [2]
                        if dim == 3:
                            cur_vgg_mode = 'lncc_3d'
                    elif m_str.startswith(('dino_', 'resnet_')):
                        parts = m_str.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]
                            if parts[2] == 'lncc':
                                cur_vgg_mode = 'lncc_3d' if dim == 3 else 'lncc'
                    elif m_str.startswith('swin_'):
                        parts = m_str.split('_')
                        if len(parts) >= 3 and parts[1].isdigit():
                            cur_vgg_layers = [int(parts[1])]

                    cur_kwargs['mode'] = cur_vgg_mode
                    cur_kwargs['feature_layers'] = cur_vgg_layers
                    cur_kwargs['window_size'] = cur_vgg_window

                loss_fn = get_similarity_loss(metric, **cur_kwargs)
                if isinstance(loss_fn, torch.nn.Module) and device is not None:
                    loss_fn = loss_fn.to(device=device)
                self.loss_functions.append(loss_fn)
            elif isinstance(metric, torch.nn.Module) or callable(metric):
                def _wrap_callable(fn):
                    def _wrapped(x, y, mask=None):
                        try:
                            return fn(x, y, mask=mask)
                        except TypeError:
                            return fn(x, y)
                    return _wrapped
                self.loss_functions.append(_wrap_callable(metric))
            else:
                raise TypeError(f"Invalid similarity metric: {metric}")
        
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
                fixed_spacing_rev, fixed_origin_rev, fixed_direction_rev = reverse_metadata(
                    curr_spacing_fixed, fixed_origin, fixed_direction
                )
                fixed_spacing_t = torch.tensor(fixed_spacing_rev, device=device, dtype=dtype)
                fixed_origin_t = torch.tensor(fixed_origin_rev, device=device, dtype=dtype)
                fixed_direction_t = torch.tensor(fixed_direction_rev, device=device, dtype=dtype)
                
                moving_shape_t = torch.tensor(list(J_curr.shape[2:]), device=device, dtype=dtype)
                moving_spacing_rev, moving_origin_rev, moving_direction_rev = reverse_metadata(
                    curr_spacing_moving, moving_origin, moving_direction
                )
                moving_spacing_t = torch.tensor(moving_spacing_rev, device=device, dtype=dtype)
                moving_origin_t = torch.tensor(moving_origin_rev, device=device, dtype=dtype)
                moving_direction_t = torch.tensor(moving_direction_rev, device=device, dtype=dtype)

                curr_spacing_fixed_zyx = fixed_spacing_t
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
                metric_name = metric.metric if isinstance(metric, SimilarityLossConfig) else str(metric)
                m_lower = metric_name.lower()
                if m_lower in ['vgg19', 'resnet10', 'dinov2', 'dinov2_small', 'dinov2_base', 'swinunetr', 'swin_unetr'] or any(p in m_lower for p in ['vgg', 'dino', 'resnet', 'swin']):
                    is_deep = True
                elif hasattr(metric, 'extractor') or ('FeatureSpaceLoss' in metric.__class__.__name__):
                    is_deep = True
                    
                if is_degenerate and is_deep:
                    fallback_fn = get_similarity_loss(
                        'lncc',
                        window_size=lncc_window_size,
                        use_analytical_gradients=use_analytical_gradients
                    ).to(device=device)
                    active_loss_functions.append(fallback_fn)
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
            elif optimizer_type in ['adam', 'reg_adam', 'regadam', 'sobolev_adam', 'gaussian_adam', 'dsti_adam']:
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
            best_warp_l2r = None
            best_warp_r2l = None
            best_warp_l2r_inv = None
            best_warp_r2l_inv = None
            
            for epoch in range(curr_syn_epochs):
                if warp_l2r.grad is not None: warp_l2r.grad.zero_()
                if warp_r2l.grad is not None: warp_r2l.grad.zero_()
                
                # Real SyN: Pull both images to the midpoint domain
                I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask = prepare_mid_images_and_gradients_torch(
                    warp_l2r, warp_r2l, I_curr, J_curr,
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
                        val_loss_a = fn(I_mid_det, J_mid_det, mask=in_bounds_mask)
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
                        val_loss_auto = fn(I_mid, J_mid, mask=in_bounds_mask)
                        loss_auto += weight * val_loss_auto

                    loss_auto.backward()
                    loss_val = loss_auto.item()

                    autograd_scale_fixed = compute_autograd_physical_scale(fixed_shape_t, fixed_spacing_t, device=device, dtype=dtype)
                    autograd_scale_moving = compute_autograd_physical_scale(moving_shape_t, moving_spacing_t, device=device, dtype=dtype)
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
                        for c_idx, (name, fn, weight) in enumerate(zip(active_metric_names, active_loss_functions, curr_metric_weights)):
                            if I_mid.shape[1] > 1 and len(active_loss_functions) == I_mid.shape[1]:
                                I_c = I_mid[:, c_idx:c_idx+1]
                                J_c = J_mid[:, c_idx:c_idx+1]
                            else:
                                I_c = I_mid
                                J_c = J_mid
                            val_loss = fn(I_c, J_c, mask=in_bounds_mask)

                            loss += weight * val_loss
                            metric_losses_dict[name] = val_loss.item()

                        jac_penalty_w = float(kwargs.get('jacobian_penalty_weight', getattr(self, 'jacobian_penalty_weight', 0.0)))
                        if jac_penalty_w > 0.0:
                            jac_eps = float(kwargs.get('jacobian_epsilon', getattr(self, 'jacobian_epsilon', 0.05)))
                            # both half warps live on the current fixed grid (physical, components
                            # z, y, x); the penalty takes that grid's ITK-order spacing
                            l_jac_l = compute_jacobian_hinge_penalty(warp_l2r, physical_spacing=tuple(curr_spacing_fixed), epsilon=jac_eps)
                            l_jac_r = compute_jacobian_hinge_penalty(warp_r2l, physical_spacing=tuple(curr_spacing_fixed), epsilon=jac_eps)
                            scale_norm = float(torch.mean((fixed_shape_t - 1.0) * fixed_spacing_t / 2.0).item())
                            loss = loss + (jac_penalty_w / (scale_norm + 1e-6)) * (l_jac_l + l_jac_r)

                        boot_m = kwargs.get('bootstrap_mode', getattr(self, 'bootstrap_mode', None))
                        if boot_m in ['jitter', 'antithetic', 'strided']:
                            orig_w = float(kwargs.get('bootstrap_orig_weight', getattr(self, 'bootstrap_orig_weight', 0.50)))
                            jitter_amp = float(kwargs.get('bootstrap_jitter_scale', getattr(self, 'bootstrap_jitter_scale', 0.30)))
                            n_boot_samples = int(kwargs.get('bootstrap_samples', getattr(self, 'bootstrap_samples', 2 if boot_m == 'antithetic' else 1)))
                            
                            j_view_shape = [1] * (dim + 1) + [dim]
                            spacing_tensor = torch.tensor(curr_spacing_fixed, device=device, dtype=X_phys.dtype).view(*j_view_shape)
                            
                            if boot_m == 'antithetic' or n_boot_samples >= 2:
                                # Antithetic variance reduction: evaluate +delta and -delta for unbiased coordinate centering
                                if self.seed is not None and (self._rng is None or self._rng.device != device):
                                    self._rng = torch.Generator(device=device).manual_seed(self.seed)

                                if self._rng is not None:
                                    rand_dir = (2.0 * torch.rand(*j_view_shape, generator=self._rng, device=device, dtype=X_phys.dtype) - 1.0) * jitter_amp * spacing_tensor
                                else:
                                    rand_dir = (2.0 * torch.rand(*j_view_shape, device=device, dtype=X_phys.dtype) - 1.0) * jitter_amp * spacing_tensor
                                offsets = [rand_dir, -rand_dir]
                                boot_weight = (1.0 - orig_w) / len(offsets)
                                
                                total_boot_loss = 0.0
                                for offset in offsets:
                                    I_mid_b, J_mid_b, _, _, mask_b = prepare_mid_images_and_gradients_torch(
                                        warp_l2r, warp_r2l, I_curr, J_curr,
                                        X_phys + offset,
                                        fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                                        moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                                        curr_spacing_fixed, curr_spacing_moving,
                                        M_phys, t_phys, initial_grid_level,
                                        interpolator=self.interpolator,
                                        use_analytical_gradients=False
                                    )
                                    loss_b = 0.0
                                    for c_idx, (name, fn, weight) in enumerate(zip(active_metric_names, active_loss_functions, curr_metric_weights)):
                                        if I_mid_b.shape[1] > 1 and len(active_loss_functions) == I_mid_b.shape[1]:
                                            I_bc = I_mid_b[:, c_idx:c_idx+1]
                                            J_bc = J_mid_b[:, c_idx:c_idx+1]
                                        else:
                                            I_bc = I_mid_b
                                            J_bc = J_mid_b
                                        val_loss_b = fn(I_bc, J_bc, mask=mask_b)
                                        loss_b += weight * val_loss_b
                                    total_boot_loss += boot_weight * loss_b
                                
                                loss = orig_w * loss + total_boot_loss
                            else:
                                # Single complementary jitter sample
                                if self.seed is not None and (self._rng is None or self._rng.device != device):
                                    self._rng = torch.Generator(device=device).manual_seed(self.seed)

                                if self._rng is not None:
                                    j_shift = (2.0 * torch.rand(*j_view_shape, generator=self._rng, device=device, dtype=X_phys.dtype) - 1.0) * jitter_amp * spacing_tensor
                                else:
                                    j_shift = (2.0 * torch.rand(*j_view_shape, device=device, dtype=X_phys.dtype) - 1.0) * jitter_amp * spacing_tensor
                                I_mid_j, J_mid_j, _, _, mask_j = prepare_mid_images_and_gradients_torch(
                                    warp_l2r, warp_r2l, I_curr, J_curr,
                                    X_phys + j_shift,
                                    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                                    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                                    curr_spacing_fixed, curr_spacing_moving,
                                    M_phys, t_phys, initial_grid_level,
                                    interpolator=self.interpolator,
                                    use_analytical_gradients=False
                                )
                                loss_j = 0.0
                                for c_idx, (name, fn, weight) in enumerate(zip(active_metric_names, active_loss_functions, curr_metric_weights)):
                                    if I_mid_j.shape[1] > 1 and len(active_loss_functions) == I_mid_j.shape[1]:
                                        I_jc = I_mid_j[:, c_idx:c_idx+1]
                                        J_jc = J_mid_j[:, c_idx:c_idx+1]
                                    else:
                                        I_jc = I_mid_j
                                        J_jc = J_mid_j
                                    val_loss_j = fn(I_jc, J_jc, mask=mask_j)
                                    loss_j += weight * val_loss_j
                                loss = orig_w * loss + (1.0 - orig_w) * loss_j
                        
                    loss.backward()
                    loss_val = loss.item()
                    check_loss_collapse(loss_val, best_level_loss, f"syntx.syn level {level_idx}")
                    
                    # Rescale autograd gradients from normalized grid space [-1, 1] to physical mm
                    # coords_norm = (x_phys - origin) * 2 / (spacing * (shape - 1)) - 1
                    # dLoss/dx_phys = dLoss/dcoords_norm * 2 / (spacing * (shape - 1))
                    # Rescaling by (shape - 1) * spacing / 2 converts back to consistent physical displacement gradient:
                    autograd_scale_fixed = compute_autograd_physical_scale(fixed_shape_t, fixed_spacing_t, device=device, dtype=dtype)
                    autograd_scale_moving = compute_autograd_physical_scale(moving_shape_t, moving_spacing_t, device=device, dtype=dtype)
                    if warp_l2r.grad is not None:
                        warp_l2r.grad = warp_l2r.grad * autograd_scale_fixed
                    if warp_r2l.grad is not None:
                        warp_r2l.grad = warp_r2l.grad * autograd_scale_moving
                        
                    self.syn_losses.append(loss_val)
                    level_syn_losses.append(loss_val)
                    if loss_val < best_level_loss:
                        best_level_loss = loss_val
                        best_warp_l2r = warp_l2r.detach().clone()
                        best_warp_r2l = warp_r2l.detach().clone()
                        best_warp_l2r_inv = warp_l2r_inv.detach().clone()
                        best_warp_r2l_inv = warp_r2l_inv.detach().clone()

                if isinstance(self.fluid_sigma, (list, tuple)):
                    curr_fluid_var = self.fluid_sigma[min(level_idx, len(self.fluid_sigma) - 1)]
                else:
                    curr_fluid_var = self.fluid_sigma
                curr_fluid_sig = float(curr_fluid_var)
                    
                kernel_type = kwargs.get('kernel_type', getattr(self, 'kernel_type', 'bessel'))
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

                    if regularizer in ('compact', 'compact_gaussian', 'erf') or kernel_type in ('compact', 'compact_gaussian', 'erf'):
                        # Compact erf-based Gaussian kernel: always uses compact spatial erf filter (bypassing FFT spectral smoothing)
                        grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, curr_fluid_sig, kernel_type='compact')
                        grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, curr_fluid_sig, kernel_type='compact')
                    elif regularizer == 'sobolev':
                        if fast_smooth:
                            # FFT Sobolev Green's operator only (standard mode)
                            grad_l = self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # FFT Sobolev Green's operator + spatial Gaussian post-filter (conservative mode)
                            grad_l = separable_gaussian_filter(self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                            grad_r = separable_gaussian_filter(self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                    elif regularizer in ['dsti', 'dst1', 'dst_i']:
                        if fast_smooth:
                            # FFT DST-I Green's operator only (standard mode)
                            grad_l = self._apply_dsti_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_dsti_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # FFT DST-I Green's operator + spatial Gaussian post-filter (conservative mode)
                            grad_l = separable_gaussian_filter(self._apply_dsti_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                            grad_r = separable_gaussian_filter(self._apply_dsti_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                    elif regularizer == 'dsti1':
                        if fast_smooth:
                            # Separable 1D DST-I Green's operator only (MPS-safe mode)
                            grad_l = self._apply_dsti1_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            grad_r = self._apply_dsti1_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                        else:
                            # Separable 1D DST-I + spatial Gaussian post-filter
                            grad_l = separable_gaussian_filter(self._apply_dsti1_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                            grad_r = separable_gaussian_filter(self._apply_dsti1_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev), curr_fluid_sig * 0.5, kernel_type=kernel_type)
                    elif regularizer in ['bspline', 'bsplinesyn']:
                        grad_l = self._apply_bspline_operator(
                            warp_l2r.grad * b_mask,
                            spacing=curr_spacing_fixed,
                            origin=fixed_origin,
                            fluid_sigma=curr_fluid_sig,
                            **kwargs,
                        )
                        grad_r = self._apply_bspline_operator(
                            warp_r2l.grad * b_mask,
                            spacing=curr_spacing_fixed,
                            origin=fixed_origin,
                            fluid_sigma=curr_fluid_sig,
                            **kwargs,
                        )
                    elif regularizer in _CONTINUUM_REGS:
                        f_mask = kwargs.get('fixed_mask', kwargs.get('mask', None))
                        if f_mask is not None and not isinstance(f_mask, torch.Tensor):
                            if hasattr(f_mask, 'numpy'):
                                f_mask = torch.from_numpy(f_mask.numpy().astype(np.float32)).to(device)
                            else:
                                f_mask = torch.as_tensor(f_mask, dtype=dtype, device=device)
                        extra_kwargs = {k: v for k, v in kwargs.items() if k not in ('fixed_mask', 'mask', 'regularizer', 'alpha', 'sobolev_alpha', 'spacing', 'fluid_sigma')}
                        reg_fn = get_regularizer(
                            regularizer,
                            alpha=alpha_sobolev,
                            sobolev_alpha=alpha_sobolev,
                            spacing=curr_spacing_fixed,
                            fluid_sigma=curr_fluid_sig,
                            fixed_mask=f_mask,
                            mask=f_mask,
                            **extra_kwargs,
                        )
                        grad_l = reg_fn(warp_l2r.grad * b_mask)
                        grad_r = reg_fn(warp_r2l.grad * b_mask)
                    else:
                        if fast_smooth:
                            # Spectral Gaussian: Sobolev Green's with soft alpha (FFT-based)
                            grad_l = self._apply_sobolev_green_operator(warp_l2r.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=curr_fluid_sig / 2.0)
                            grad_r = self._apply_sobolev_green_operator(warp_r2l.grad * b_mask, fluid_sigma=curr_fluid_sig, alpha=curr_fluid_sig / 2.0)
                        else:
                            # Spatial Gaussian: separable convolution filter
                            grad_l = separable_gaussian_filter(warp_l2r.grad * b_mask, curr_fluid_sig, kernel_type=kernel_type)
                            grad_r = separable_gaussian_filter(warp_r2l.grad * b_mask, curr_fluid_sig, kernel_type=kernel_type)

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

                    restrict_mask = self._get_restrict_mask(grad_l.device, grad_l.dtype)
                    if restrict_mask is not None:
                        grad_l = grad_l * restrict_mask
                        grad_r = grad_r * restrict_mask

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
                        max_norm_l_safe = max_norm_l if use_analytical_gradients else torch.clamp(max_norm_l, min=1e-8)
                        max_norm_r_safe = max_norm_r if use_analytical_gradients else torch.clamp(max_norm_r, min=1e-8)
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
                            delta_l_pb = F.grid_sample(delta_l.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
                            
                            coords_phys_r = X_phys + warp_r2l
                            coords_norm_r = physical_to_normalized_torch_cached(
                                coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            delta_r_pb = F.grid_sample(delta_r.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
                            
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
                            warp_l2r_sampled = F.grid_sample(warp_l2r.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
                            warp_l2r.copy_(warp_l2r_sampled - delta_l)
                            
                            coords_phys_r = X_phys - delta_r
                            coords_norm_r = physical_to_normalized_torch_cached(
                                coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            warp_r2l_sampled = F.grid_sample(warp_r2l.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
                            warp_r2l.copy_(warp_r2l_sampled - delta_r)

                        
                        # Removed ITK-standard Dirichlet zero boundary enforcement after composition
                        # to prevent massive gradient-exploding discontinuities at the boundary.
                        
                        if self.elastic_sigma > 0.0:
                            elastic_sig_val = float(self.elastic_sigma)
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, elastic_sig_val, kernel_type=kernel_type))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, elastic_sig_val, kernel_type=kernel_type))
                            
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
                            elastic_sig_val = float(self.elastic_sigma)
                            if regularizer in ['bspline', 'bsplinesyn']:
                                warp_l2r.copy_(self._apply_bspline_operator(warp_l2r, spacing=curr_spacing_fixed, origin=fixed_origin, spline_distance=kwargs.get('elastic_spline_distance', kwargs.get('spline_distance')), mesh_size=kwargs.get('elastic_mesh_size', kwargs.get('mesh_size')), fluid_sigma=elastic_sig_val, **kwargs))
                                warp_r2l.copy_(self._apply_bspline_operator(warp_r2l, spacing=curr_spacing_fixed, origin=fixed_origin, spline_distance=kwargs.get('elastic_spline_distance', kwargs.get('spline_distance')), mesh_size=kwargs.get('elastic_mesh_size', kwargs.get('mesh_size')), fluid_sigma=elastic_sig_val, **kwargs))
                            else:
                                warp_l2r.copy_(separable_gaussian_filter(warp_l2r, elastic_sig_val, kernel_type=kernel_type))
                                warp_r2l.copy_(separable_gaussian_filter(warp_r2l, elastic_sig_val, kernel_type=kernel_type))
                            
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
                        
                    elif optimizer_type in ['adam', 'reg_adam', 'regadam', 'sobolev_adam', 'gaussian_adam', 'dsti_adam']:
                        self._adam_t += 1
                        beta1, beta2 = 0.9, 0.999
                        eps = 1e-8
                        
                        b1 = 1.0 - beta1 ** self._adam_t
                        b2 = 1.0 - beta2 ** self._adam_t
                        b2_sqrt = math.sqrt(b2)
                        step_size = b2_sqrt / b1
                        eps_scaled = eps * b2_sqrt
                        
                        self._adam_m_l.mul_(beta1).add_(grad_l, alpha=1.0 - beta1)
                        self._adam_v_l.mul_(beta2).addcmul_(grad_l, grad_l, value=1.0 - beta2)
                        u_raw_l = (self._adam_m_l * step_size).div_(self._adam_v_l.sqrt().add_(eps_scaled))
                        
                        self._adam_m_r.mul_(beta1).add_(grad_r, alpha=1.0 - beta1)
                        self._adam_v_r.mul_(beta2).addcmul_(grad_r, grad_r, value=1.0 - beta2)
                        u_raw_r = (self._adam_m_r * step_size).div_(self._adam_v_r.sqrt().add_(eps_scaled))
                        
                        # Spatial pre-smoothing of Adam quotient for RegAdam
                        if optimizer_type in ['reg_adam', 'regadam', 'sobolev_adam', 'gaussian_adam', 'dsti_adam']:
                            if regularizer == 'sobolev':
                                u_reg_l = self._apply_sobolev_green_operator(u_raw_l, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                                u_reg_r = self._apply_sobolev_green_operator(u_raw_r, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            elif regularizer in ['dsti', 'dst1', 'dsti1']:
                                u_reg_l = self._apply_dsti1_green_operator(u_raw_l, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                                u_reg_r = self._apply_dsti1_green_operator(u_raw_r, fluid_sigma=curr_fluid_sig, alpha=alpha_sobolev)
                            elif regularizer in ['bspline', 'bsplinesyn']:
                                u_reg_l = self._apply_bspline_operator(
                                    u_raw_l,
                                    spacing=curr_spacing_fixed,
                                    origin=fixed_origin,
                                    fluid_sigma=curr_fluid_sig,
                                    **kwargs,
                                )
                                u_reg_r = self._apply_bspline_operator(
                                    u_raw_r,
                                    spacing=curr_spacing_fixed,
                                    origin=fixed_origin,
                                    fluid_sigma=curr_fluid_sig,
                                    **kwargs,
                                )
                            elif regularizer in _CONTINUUM_REGS:
                                f_mask = kwargs.get('fixed_mask', kwargs.get('mask', None))
                                if f_mask is not None and not isinstance(f_mask, torch.Tensor):
                                    if hasattr(f_mask, 'numpy'):
                                        f_mask = torch.from_numpy(f_mask.numpy().astype(np.float32)).to(device)
                                    else:
                                        f_mask = torch.as_tensor(f_mask, dtype=dtype, device=device)
                                extra_kwargs = {k: v for k, v in kwargs.items() if k not in ('fixed_mask', 'mask', 'regularizer', 'alpha', 'sobolev_alpha', 'spacing', 'fluid_sigma')}
                                reg_fn = get_regularizer(
                                    regularizer,
                                    alpha=alpha_sobolev,
                                    sobolev_alpha=alpha_sobolev,
                                    spacing=curr_spacing_fixed,
                                    fluid_sigma=curr_fluid_sig,
                                    fixed_mask=f_mask,
                                    mask=f_mask,
                                    **extra_kwargs,
                                )
                                u_reg_l = reg_fn(u_raw_l)
                                u_reg_r = reg_fn(u_raw_r)
                            else:
                                g_sig = kwargs.get('gaussian_sigma', 1.5)
                                u_reg_l = separable_gaussian_filter(u_raw_l, g_sig)
                                u_reg_r = separable_gaussian_filter(u_raw_r, g_sig)
                        else:
                            u_reg_l = u_raw_l
                            u_reg_r = u_raw_r

                        restrict_mask = self._get_restrict_mask(u_reg_l.device, u_reg_l.dtype)
                        if restrict_mask is not None:
                            u_reg_l = u_reg_l * restrict_mask
                            u_reg_r = u_reg_r * restrict_mask

                        # Adaptive CFL step scaling (100% on-device asynchronous execution)
                        effective_cfl = float(level_cfl_voxels)
                        lr_effective = float(optimizer_lr)
                        spatial_dims = tuple(range(1, u_reg_l.dim() - 1))
                        norm_l = torch.norm(u_reg_l, dim=-1)
                        norm_r = torch.norm(u_reg_r, dim=-1)
                        max_l = torch.amax(norm_l, dim=spatial_dims)
                        max_r = torch.amax(norm_r, dim=spatial_dims)
                        max_u = torch.maximum(max_l, max_r).clamp_min(1e-4)
                        cfl_scale = (effective_cfl / max_u).clamp_max(1.0)
                        cfl_scale = cfl_scale.view(-1, *([1] * (u_reg_l.dim() - 1)))
                        delta_l = (cfl_scale * lr_effective) * u_reg_l
                        delta_r = (cfl_scale * lr_effective) * u_reg_r
                        
                        # Antisymmetric velocity projection
                        if getattr(self, 'antisymmetric', True):
                            e0 = delta_l + delta_r
                            delta_l = delta_l - 0.5 * e0
                            delta_r = delta_r - 0.5 * e0
                            
                        # Eulerian or Lagrangian pullback composition
                        if getattr(self, 'formulation', 'lagrangian') == 'lagrangian':
                            coords_phys_l = X_phys + warp_l2r
                            coords_norm_l = physical_to_normalized_torch_cached(
                                coords_phys_l, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            delta_l_pb = F.grid_sample(delta_l.movedim(-1, 1), coords_norm_l, padding_mode='border', align_corners=True).movedim(1, -1)
                            
                            coords_phys_r = X_phys + warp_r2l
                            coords_norm_r = physical_to_normalized_torch_cached(
                                coords_phys_r, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
                            )
                            delta_r_pb = F.grid_sample(delta_r.movedim(-1, 1), coords_norm_r, padding_mode='border', align_corners=True).movedim(1, -1)
                            
                            with torch.no_grad():
                                warp_l2r.sub_(delta_l_pb)
                                warp_r2l.sub_(delta_r_pb)
                        else:
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
                            
                        if self.elastic_sigma > 0.0:
                            elastic_sig_val = float(self.elastic_sigma)
                            if regularizer in ['bspline', 'bsplinesyn']:
                                warp_l2r.copy_(self._apply_bspline_operator(warp_l2r, spacing=curr_spacing_fixed, origin=fixed_origin, spline_distance=kwargs.get('elastic_spline_distance', kwargs.get('spline_distance')), mesh_size=kwargs.get('elastic_mesh_size', kwargs.get('mesh_size')), fluid_sigma=elastic_sig_val, **kwargs))
                                warp_r2l.copy_(self._apply_bspline_operator(warp_r2l, spacing=curr_spacing_fixed, origin=fixed_origin, spline_distance=kwargs.get('elastic_spline_distance', kwargs.get('spline_distance')), mesh_size=kwargs.get('elastic_mesh_size', kwargs.get('mesh_size')), fluid_sigma=elastic_sig_val, **kwargs))
                            else:
                                warp_l2r.copy_(separable_gaussian_filter(warp_l2r, elastic_sig_val))
                                warp_r2l.copy_(separable_gaussian_filter(warp_r2l, elastic_sig_val))
                            
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r, warp_l2r_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l, warp_r2l_inv.detach(), steps=in_loop_inv_steps, method=self.inverse_method,
                            spacing=curr_spacing_fixed, origin=fixed_origin, direction=fixed_direction, X_phys=X_phys, max_error_threshold=self.inv_tolerance, mean_error_threshold=self.inv_tolerance*0.01
                        )
                        
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
                    
                    # ITK/ANTs EnforceStationaryBoundary: zero displacement on every face.
                    if self.stationary_boundary:
                        with torch.no_grad():
                            for _w in (warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv):
                                _w.mul_(b_mask)

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
            # Post-level divergence detection warning (no in-loop mutation or silent retries)
            if len(level_syn_losses) > 5 and curr_syn_epochs > 0:
                best_level_loss = min(float(l) for l in level_syn_losses)
                final_level_loss = float(level_syn_losses[-1])
                loss_worsened = final_level_loss - best_level_loss
                if loss_worsened > abs(best_level_loss) and verbose:
                    warnings.warn(
                        f"syntx.syn Level {level_idx} loss worsened from {best_level_loss:.6f} to "
                        f"{final_level_loss:.6f}. Consider increasing flow_sigma or evaluating with syntx.qc."
                    )
                    
            # Evaluate final state after the last optimization step
            def _eval_syn():
                with torch.no_grad():
                    I_m, J_m, _, _, mask_m = prepare_mid_images_and_gradients_torch(
                        warp_l2r, warp_r2l, I_curr, J_curr,
                        X_phys,
                        fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
                        moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
                        curr_spacing_fixed, curr_spacing_moving,
                        M_phys, t_phys, initial_grid_level,
                        interpolator=self.interpolator,
                        use_analytical_gradients=False
                    )
                    l_eval = 0.0
                    for name, fn, weight in zip(active_metric_names, active_loss_functions, curr_metric_weights):
                        val_l = fn(I_m, J_m, mask=mask_m)
                        l_eval += weight * val_l
                    return float(l_eval.item())

            if curr_syn_epochs > 0:
                final_loss = _eval_syn()
                if final_loss < best_level_loss:
                    best_level_loss = final_loss
                elif best_warp_l2r is not None:
                    with torch.no_grad():
                        warp_l2r.data.copy_(best_warp_l2r)
                        warp_r2l.data.copy_(best_warp_r2l)
                        warp_l2r_inv = best_warp_l2r_inv.clone()
                        warp_r2l_inv = best_warp_r2l_inv.clone()

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
            comp_spacing_rev, comp_origin_rev, comp_direction_rev = reverse_metadata(
                fixed_spacing, fixed_origin, fixed_direction
            )
            comp_shape_t = torch.tensor(list(self.grid_shape), device=device, dtype=dtype)
            comp_spacing_t = torch.tensor(comp_spacing_rev, device=device, dtype=dtype)
            comp_origin_t = torch.tensor(comp_origin_rev, device=device, dtype=dtype)
            comp_direction_t = torch.tensor(comp_direction_rev, device=device, dtype=dtype)
            
            # Preserve uncomposed half-warp fields for midpoint image export.
            # w_l2r maps midpoint→fixed, w_r2l maps midpoint→(affine)moving.
            # These are destroyed by the full composition below.
            self.midpoint_warp_l2r = nn.Parameter(w_l2r.clone(), requires_grad=False)
            self.midpoint_warp_l2r.is_physical = True
            self.midpoint_warp_r2l = nn.Parameter(w_r2l.clone(), requires_grad=False)
            self.midpoint_warp_r2l.is_physical = True
            # inverse of the fixed-side half, exactly as used in the composition below
            # (syntx.liouville_determinant needs it: det phi(x) = det Dphi_r2l(y) / det Dphi_l2r(y),
            # y = phi_l2r^-1(x))
            self.midpoint_warp_l2r_inv = nn.Parameter(w_l2r_inv.clone(), requires_grad=False)
            self.midpoint_warp_l2r_inv.is_physical = True
            
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

            if self.stationary_boundary:
                full_b_mask = get_boundary_mask(self.grid_shape, device, dtype)
                for _w in (self.warp_l2r, self.warp_r2l, self.warp_l2r_inv, self.warp_r2l_inv,
                           self.midpoint_warp_l2r, self.midpoint_warp_r2l):
                    _w.data.mul_(full_b_mask)


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
        # Shapes are tensor (Z, Y, X) order, matching fit().
        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, spatial_shape, spacing, origin, direction,
            tuple(moving_image_zyx.shape[2:]), moving_spacing, moving_origin, moving_direction
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
            
        warped_zyx = grid_sample_nd(moving_image_zyx, composed_grid, padding_mode='itk', align_corners=True, interpolator=self.interpolator)
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

        Evaluates exactly the exported ``invtransforms`` chain ``[affine^-1, warp_inv]``:
        a moving-space point y is first mapped through the inverse affine into the
        (fixed-grid) frame where ``warp_r2l`` lives, the inverse displacement is sampled
        there (zero outside its domain, as in ITK), and the fixed image is sampled at the
        result. ``moving_shape`` is in tensor (Z, Y, X) order.
        """
        import ants
        from .spatial import image_to_tensor
        if getattr(self, 'initial_grid', None) is not None:
            raise NotImplementedError(
                "forward_inverse cannot invert a non-affine initial_transform natively; "
                "use ants.apply_transforms with the registration's invtransforms.")
        is_ants = isinstance(fixed_image, ants.ANTsImage)
        device = self.warp_r2l.device
        dtype = self.warp_r2l.dtype
        dim = self.dim
        perm = [0, 1] + list(range(dim + 1, 1, -1))

        if is_ants:
            fixed_tensor = image_to_tensor(fixed_image, device=device)
            img_spacing, img_origin, img_direction = fixed_image.spacing, fixed_image.origin, fixed_image.direction
        else:
            fixed_tensor = fixed_image
            img_spacing, img_origin, img_direction = self.spacing, self.origin, self.direction

        fixed_image_zyx = fixed_tensor.permute(perm)
        img_shape = tuple(fixed_image_zyx.shape[2:])

        # Domain of the displacement fields: the model's fixed grid.
        grid_shape = tuple(self.grid_shape)
        spacing = self.spacing if self.spacing is not None else [1.0] * dim
        origin = self.origin if self.origin is not None else [0.0] * dim
        direction = self.direction if self.direction is not None else np.eye(dim)
        if img_spacing is None: img_spacing = spacing
        if img_origin is None: img_origin = origin
        if img_direction is None: img_direction = direction

        # Moving properties define output space
        if moving_shape is None: moving_shape = getattr(self, 'moving_shape', grid_shape)
        if moving_spacing is None: moving_spacing = getattr(self, 'moving_spacing', None) or spacing
        if moving_origin is None: moving_origin = getattr(self, 'moving_origin', None) or origin
        if moving_direction is None: moving_direction = getattr(self, 'moving_direction', None)
        if moving_direction is None: moving_direction = direction
        moving_shape = tuple(moving_shape)

        Y_phys = get_physical_grid_torch(moving_shape, moving_spacing, moving_origin, moving_direction, device=device, dtype=dtype)

        # y -> z = A^-1(y), into the frame of warp_r2l
        T_grid = self.affine.get_matrix()
        M_phys, t_phys = grid_to_physical_affine_torch(
            T_grid, grid_shape, spacing, origin, direction,
            moving_shape, moving_spacing, moving_origin, moving_direction
        )
        M_phys = M_phys.to(device=device, dtype=dtype)
        t_phys = t_phys.to(device=device, dtype=dtype)
        z_phys = (Y_phys - t_phys) @ torch.linalg.inv(M_phys).t()

        # z -> x = z + warp_r2l(z); zero displacement outside the field domain (ITK)
        z_norm = physical_to_normalized_torch(z_phys, grid_shape, spacing, origin, direction)
        warp_inv = self.warp_r2l.to(device=device, dtype=dtype)
        if tuple(warp_inv.shape[1:-1]) != grid_shape:
            warp_inv = F.interpolate(torch.movedim(warp_inv, -1, 1), size=grid_shape,
                                     mode='bilinear' if dim == 2 else 'trilinear',
                                     align_corners=True).movedim(1, -1)
        disp = F.grid_sample(torch.movedim(warp_inv, -1, 1), z_norm, mode='bilinear',
                             padding_mode='zeros', align_corners=True).movedim(1, -1)
        x_phys = z_phys + disp

        composed_grid = physical_to_normalized_torch(x_phys, img_shape, img_spacing, img_origin, img_direction)
        warped_zyx = grid_sample_nd(fixed_image_zyx, composed_grid, padding_mode='itk', align_corners=True, interpolator=self.interpolator)
        warped_xyz = warped_zyx.permute(perm)

        if is_ants:
            arr_np = warped_xyz.squeeze(0).squeeze(0).detach().cpu().numpy()
            dir_np = moving_direction.detach().cpu().numpy() if isinstance(moving_direction, torch.Tensor) else np.asarray(moving_direction)
            return ants.from_numpy(arr_np, origin=tuple(moving_origin), spacing=tuple(moving_spacing), direction=dir_np)
        return warped_xyz



    def get_forward_transform(self, fixed_metadata):
        """``SyNToTransform`` of the forward transform (affine + ``warp_l2r``): maps
        fixed-space points into moving space, i.e. warps the moving image into fixed space
        (the ANTs ``fwdtransforms`` direction). ``fixed_metadata``: spacing / origin /
        direction / shape of the fixed image."""
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
        """``SyNToTransform`` of the inverse transform (inverse affine + ``warp_r2l``): warps
        the fixed image into moving space. ``moving_metadata``: geometry of the moving image."""
        device = self.warp_r2l.device
        grid_affine_inv = self.get_inverse_affine_grid(self.grid_shape, device)
        return SyNToTransform(
            affine_grid=grid_affine_inv, 
            warp_field=self.warp_r2l, 
            metadata=moving_metadata, 
            device=device,
            is_physical=True
        )

    def to_transform(self, fixed=None, moving=None, metadata=None):
        """``SyNToTransform`` of the forward transform with geometry from ``metadata``, else
        from ``fixed`` (an ANTsImage), else from the model."""
        device = self.warp_l2r.device
        grid_affine = self.get_affine_grid(self.grid_shape, device)
        if metadata is None:
            metadata = {
                'spacing': self.spacing if self.spacing is not None else (1.0,) * self.dim,
                'origin': getattr(self, 'origin', (0.0,) * self.dim),
                'direction': self.direction.cpu().numpy() if isinstance(self.direction, torch.Tensor) else self.direction,
                'shape': self.grid_shape
            }
            if fixed is not None and hasattr(fixed, 'spacing'):
                metadata['spacing'] = fixed.spacing
                metadata['origin'] = fixed.origin
                metadata['direction'] = fixed.direction
                metadata['shape'] = fixed.shape
            if moving is not None and hasattr(moving, 'spacing'):
                metadata['moving_spacing'] = moving.spacing
                metadata['moving_origin'] = moving.origin
                metadata['moving_direction'] = moving.direction
                metadata['moving_shape'] = moving.shape
                
        affine_mat = None
        if hasattr(self, 'affine') and hasattr(self.affine, 'get_matrix'):
            T_grid = self.affine.get_matrix().detach().cpu().numpy()
            if fixed is not None and moving is not None:
                from .core.affine import grid_to_physical_affine
                affine_mat = grid_to_physical_affine(T_grid, fixed, moving)
            else:
                affine_mat = T_grid
                
        return SyNToTransform(
            affine_grid=grid_affine,
            warp_field=self.warp_l2r.data,
            metadata=metadata,
            warp_inv_field=self.warp_r2l.data,
            affine_matrix=affine_mat,
            device=device,
            is_physical=True
        )

    def export(self, outprefix=None, fixed=None, moving=None, metadata=None):
        """Export the forward transform: with ``outprefix``, write ITK files (warp, inverse
        warp, affine) with that prefix; without, return them in memory (``'fwd_warp'``,
        ``'inv_warp'``, ``'affine_matrix'``). See ``SyNToTransform.export``."""
        tx = self.to_transform(fixed=fixed, moving=moving, metadata=metadata)
        return tx.export(outprefix=outprefix)


# Keyword options ``syntx.syn`` / ``SyNTo.fit`` read from ``**kwargs`` (everything else raises
# TypeError): registration-level, fit-level and B-spline-operator options.
SYN_ADVANCED_OPTIONS = frozenset({
    # registration()
    'alpha', 'boundary_suppression_thresh', 'cohort_type', 'device', 'dof', 'dual_gradient',
    'dual_gradient_weight', 'fixed_mask', 'formulation', 'gaussian_sigma', 'guided', 'guided_weight',
    'image_grad_clip', 'in_memory', 'initial_grid', 'inverse_method', 'inverse_steps',
    'kernel_type', 'learning_rate', 'metric_weights', 'mind_offsets', 'mind_patch_size',
    'optimizer_lr', 'outprefix', 'regularizer',
    'scales', 'similarity_metric', 'smooth_in_deformed_space', 'smoothing_sigmas',
    'sobolev_alpha', 'stationary_boundary', 'syn_metric_weights', 'use_analytical_gradients',
    'use_ants_pseudo_gradient', 'vgg_layers', 'vgg_lncc_window_size', 'vgg_mode',
    'vgg_num_patches', 'vgg_patch_size', 'winsorize_quantiles',
    # SyNTo.fit()
    'amp', 'bootstrap_jitter_scale', 'bootstrap_mode', 'bootstrap_orig_weight',
    'bootstrap_samples', 'elastic_mesh_size', 'elastic_sigma', 'elastic_spline_distance',
    'fast_smooth', 'jacobian_epsilon', 'jacobian_penalty_weight', 'mesh_size', 'num_slices',
    'optimizer_type', 'seed', 'spline_distance', 'verbose',
    # SyNTo._apply_bspline_operator()
    'enforce_stationary_boundary', 'spline_order',
    # Continuum mechanics regularizers
    'beta', 'gamma', 'poisson_ratio', 'darcy_permeability', 'bulk_modulus', 'num_iters', 'dilatation_weight', 'mask',
    's', 'h3_envelope', 'sobolev_envelope', 'envelope_power',
})
# set internally by registration() for fit()
_SYN_INTERNAL_FIT_KEYS = frozenset({'init_M_phys', 'init_t_phys', 'fixed_spacing', 'fixed_origin',
                                    'fixed_direction', 'moving_spacing', 'moving_origin',
                                    'moving_direction'})


def registration(
    fixed,
    moving,
    type_of_transform='SyN',
    syn_metric='cc2',
    syn_sampling=2,
    reg_iterations=None,
    iterations=None,
    grad_step=0.4,
    flow_sigma=2.4,
    total_sigma=0.0,
    sobolev_alpha=2.25,
    verbose=False,
    backend='pytorch',
    initial_transform=None,
    affine_dof=None,
    affine_mode='pytorch',
    affine_seed=None,
    levels=None,
    sampling_percentage=None,
    vgg_layers=[4],
    vgg_mode='lncc_3d',
    vgg_patch_size=32,
    vgg_num_patches=8,
    vgg_lncc_window_size=9,
    optimizer='cfl',
    optimizer_lr=None,
    project_inverse=True,
    projection_frequency=1,
    interpolator='linear',
    inverse_method='anderson',
    inverse_steps=30,
    in_loop_inv_steps=10,
    inv_tolerance=None,
    fast_smooth=None,
    antisymmetric=True,
    restrict_transformation=None,
    seed=42,
    **kwargs
):
    """
    Symmetric diffeomorphic (SyN) registration of ``moving`` to ``fixed`` -- ``syntx.syn``.

    Same calling convention and result as ``ants.registration``::

        reg = syntx.syn(fixed, moving)                     # robust_affine, then SyN
        reg['warpedmovout'], reg['fwdtransforms'], reg['invtransforms']
        ants.apply_transforms(fixed, moving_labels, reg['fwdtransforms'], interpolator='genericLabel')

    Two half transforms are optimised toward a midpoint space -- the fixed side and the moving
    side each carry half the deformation -- and composed at the end, so forward and inverse
    are equally accurate. Each iteration takes a step along the smoothed similarity gradient
    (``grad_step`` voxels at most, with ``optimizer='cfl'``) and composes it into the half
    fields; the inverse of each half is refreshed every iteration. The initial affine comes
    from ``syntx.robust_affine`` unless ``initial_transform`` is given. Defaults
    (grad_step / flow_sigma / sobolev_alpha / in_loop_inv_steps) are the canonical
    benchmark parameters, tuned on Mindboggle pairs 77 / 44 / 0
    (docs/provenance/best_parameters.json, "syntx.syn/canonical_2026_09_29").

    Images
    ------
    fixed, moving : ANTsImage, or list of ANTsImage (multi-channel)
        2-D or 3-D. With lists, the first image of each defines the geometry and
        ``syn_metric`` / ``syn_metric_weights`` give one metric and weight per channel.

    Initial alignment
    -----------------
    initial_transform : str, list of str, ANTsTransform, 'identity' or None, default None
        None: computed by ``syntx.robust_affine`` (``affine_dof``, ``affine_mode``,
        ``affine_seed``), except for ``type_of_transform='SyNOnly'`` (no affine). 'identity':
        start from the scanner-space identity (no centre-of-mass alignment; use for
        opposite-contrast pairs such as T1 -> EPI). Otherwise ANTs transform file(s); an
        affine is absorbed into the model, anything else becomes an initial grid.
    affine_dof : {'affine', 'rigid'} or None, default None
        For the automatic alignment. None: 'rigid' for ``type_of_transform`` 'Rigid' /
        'Translation' or ``dof='rigid'``, else 'affine'.
    affine_mode : str, default 'pytorch'
        ``syntx.robust_affine`` mode for the automatic alignment.
    affine_seed : int or None, default None
        ``syntx.robust_affine`` seed.

    Transform type
    --------------
    type_of_transform : str, default 'SyN'
        'SyN' (alias 'SyNTo'): affine + symmetric deformable. 'SyNOnly': deformable only
        (images already aligned). 'BSplineSyN': SyN with the B-spline regulariser. 'Affine',
        'Rigid', 'Translation': linear only (no deformable iterations): the affine comes from
        ``syntx.robust_affine``, or -- by design -- is ``initial_transform`` returned unchanged
        when one is given (affine and deformable optimisation are separate interfaces; there
        is no affine refinement inside ``syntx.syn``). 'greedy' (or ``formulation='greedy'``):
        delegates to ``syntx.greedy`` (the result is greedy's). Case-insensitive; any other
        string raises ValueError.

    Similarity and schedule
    -----------------------
    syn_metric : str, list of str, or callable, default 'cc2'
        'cc2' (squared local normalised cross-correlation), 'lncc', 'mattes' / 'mattes_mi',
        'mse', deep-feature metrics ('vgg19', ...; see the ``vgg_*`` options), or one per
        channel for list inputs. ``similarity_metric=`` is an alias.
    syn_sampling : int, default 2
        Local-correlation radius: window 2 * syn_sampling + 1 voxels.
    reg_iterations, iterations : int or list of int, default None
        Iterations per pyramid level. None: [100, 100, 20] in 3-D, [100, 100, 100, 50] in 2-D.
        Both names are accepted interchangeably.
    levels : list of int, default None
        Pyramid shrink factors. None: [2**(L-1), ..., 2, 1] for L = len(reg_iterations).
    sampling_percentage : float or None
        Point-sampling fraction for a Mattes-MI metric. Default None (all voxels).

    Regularisation
    --------------
    flow_sigma : float, default 2.4
        Fluid smoothing of each update, as a *variance* (ITK convention): the Gaussian sigma
        is sqrt(flow_sigma) **voxels of the current pyramid level** (spacing is not used; the
        JAX backend uses the same convention). With the default spectral regulariser and
        ``fast_smooth=False`` a Gaussian of half that sigma post-filters the spectral
        operator's output, so its value matters; with ``fast_smooth=True`` it only switches the
        smoothing on (> 0) or off (0) and a non-default value warns.
    total_sigma : float, default 0.0
        Elastic smoothing of the displacement itself after each update, also a variance;
        0 = off.
    sobolev_alpha : float or None, default 2.25
        Strength of the spectral regulariser ('sobolev', 'dsti', 'dsti1'); larger is
        smoother. ``alpha=`` is an alias. None: sqrt(flow_sigma) / 2 (legacy).
    fast_smooth : bool or None, default None
        Spectral regularisers: True = spectral operator only; False / None = spectral
        operator followed by the Gaussian post-filter (the canonical setting).
    restrict_transformation : sequence of float, optional
        Per-axis deformation weights in [0, 1], physical (x, y, z) order like
        ``ants.registration``: 1 free, 0 suppressed, e.g. (0, 1, 0) for EPI distortion along
        y. Prefer ``syntx.spatial.restriction_from_orientation(image, anatomical_axis='AP')``
        (or ``json_sidecar=``), which handles oblique images. PyTorch backend only (JAX
        raises).

    Optimiser
    ---------
    grad_step : float, default 0.4
        Largest displacement per iteration (voxels) -- the CFL step of ``optimizer='cfl'``.
    optimizer : str, default 'cfl'
        'cfl' (update scaled so its largest displacement is grad_step); Adam family
        ('adam', 'reg_adam', 'regadam', 'sobolev_adam', 'gaussian_adam', 'dsti_adam'); 'rprop';
        'sgd'.
    optimizer_lr : float or None, default None
        Adam family: largest step = optimizer_lr * grad_step (None: 0.5). 'rprop' / 'sgd':
        learning rate (None: 1e-3). Unused by 'cfl'.

    Inverse
    -------
    inverse_method : {'anderson', 'fixed_point'}, default 'anderson'
        Solver for the inverse of each half field.
    inverse_steps : int, default 30
        Iterations of the final inverse solve.
    in_loop_inv_steps : int, default 10
        Iterations refreshing each half inverse after every update.
    inv_tolerance : float or None
        Inverse-solver tolerance (mm). None: 0.1 x the smallest spacing.
    project_inverse : bool, default True
        Project the in-loop inverse (keeps the pair consistent).
    projection_frequency : int, default 1
        Project every this many iterations.

    Other
    -----
    antisymmetric : bool, default True
        Remove the common component of the two half updates, keeping the halves
        symmetric.
    interpolator : {'linear', 'nearestNeighbor'}, default 'linear'
        For the warped output images.
    backend : {'pytorch', 'jax'}, default 'pytorch'
        Note: the default regulariser is 'sobolev' with PyTorch but 'gaussian' with JAX.
    seed : int, default 42
    verbose : bool, default False
    vgg_layers, vgg_mode, vgg_patch_size, vgg_num_patches, vgg_lncc_window_size
        Deep-feature metric settings (layers [4], 'lncc_3d', 32, 8, 9).

    **kwargs
        Advanced: ``regularizer`` ('sobolev' default, 'dsti', 'dsti1', 'gaussian',
        'bspline'), ``kernel_type``, ``formulation`` ('eulerian' default, 'lagrangian',
        'greedy'), ``stationary_boundary`` (True: zero displacement at the image border),
        ``dof`` ('rigid' / 'affine'), ``device``, ``smoothing_sigmas`` (pyramid),
        ``winsorize_quantiles``, ``syn_metric_weights`` (multi-channel), ``guided`` (True /
        'sulcal': adds a sulcal-probability channel with a Dice metric), ``guided_weight``,
        ``cohort_type``, ``initial_grid``, ``use_analytical_gradients``, ``dual_gradient``,
        ``image_grad_clip``, ``in_memory`` / ``outprefix`` (export), and options forwarded
        to ``SyNTo.fit``. Removed: ``affine_iterations``, ``aff_metric``, ``aff_sampling``
        (raise; use the affine_* options).

    Returns
    -------
    dict
        ``'fwdtransforms'`` : [warp, affine] files mapping fixed-space points to moving space
            (``ants.apply_transforms(fixed, moving, fwdtransforms)``).
        ``'invtransforms'``, ``'whichtoinvert_inv'`` : the inverse direction.
        ``'warpedmovout'`` / ``'warpedfixout'`` : moving in fixed space / fixed in moving space.
        ``'midpoint_fixed'``, ``'midpoint_moving'``, ``'fwd_midpoint_warp'``,
        ``'inv_midpoint_warp'`` : both images in the midpoint space and the half warps.
        ``'syn_losses'``, ``'affine_losses'`` : loss history.
        ``'inverse_identity_errors'`` : forward-inverse consistency (mm).
        ``'model'`` : the fitted ``SyNTo`` (its half fields feed
            ``syntx.liouville_determinant``).
        ``'provenance'`` : parameters and environment.
        With ``in_memory=True``: also ``'fwd_warp'``, ``'inv_warp'``, ``'affine_matrix'``,
        ``'transform'``.
    """
    import tempfile
    import ants
    import numpy as np
    t_start = time.time()
    if reg_iterations is None:
        reg_iterations = iterations
    if reg_iterations is None and 'iterations' in kwargs:
        reg_iterations = kwargs.pop('iterations')
    elif 'iterations' in kwargs:
        kwargs.pop('iterations')
    _not_syn = sorted({'cfl_momentum', 'multipoint_loss', 'n_time_steps', 'n_steps'} & set(kwargs))
    if _not_syn:
        raise TypeError(f"syntx.syn has no {_not_syn} (they belong to syntx.tvf / syntx.syngs)")
    _greedy = (str(type_of_transform).lower() in ('greedy', 'greedy_compositive')
               or str(kwargs.get('formulation', '')).lower() == 'greedy')
    _unknown = sorted(set(kwargs) - SYN_ADVANCED_OPTIONS - {'affine_iterations', 'aff_metric', 'aff_sampling'})
    if _unknown and not _greedy:        # the greedy delegation validates its own keywords
        raise TypeError(f"syntx.syn() got unexpected keyword(s) {_unknown}; see the docstring and "
                        f"syntx.syn.SYN_ADVANCED_OPTIONS")
    if not _greedy:
        if 'optimizer_type' in kwargs:
            raise TypeError("syntx.syn: pass the optimizer as optimizer=... (optimizer_type is the "
                            "SyNTo.fit name and would be overridden by optimizer)")
        _valid_opt = SYN_OPTIMIZERS_JAX if str(backend).lower() == 'jax' else SYN_OPTIMIZERS_PYTORCH
        if optimizer not in _valid_opt:
            raise ValueError(f"unknown optimizer {optimizer!r} for backend={backend!r}; use one of "
                             f"{sorted(_valid_opt)}")
    _removed_affine_params = {'affine_iterations', 'aff_metric', 'aff_sampling'} & set(kwargs)
    if _removed_affine_params:
        raise TypeError(
            f"registration() no longer accepts {sorted(_removed_affine_params)}: the inline "
            f"affine optimizer they configured has been removed in favor of always using "
            f"syntx.robust_affine for initial alignment. Use affine_dof/affine_mode/affine_seed "
            f"instead, or pass initial_transform explicitly to bypass alignment entirely."
        )
    guided = kwargs.pop('guided', None)
    cohort_type = kwargs.pop('cohort_type', 'auto')
    guided_weight = kwargs.pop('guided_weight', None)

    if 'similarity_metric' in kwargs:
        syn_metric = kwargs.pop('similarity_metric')
    syn_metric_weights = kwargs.pop('syn_metric_weights', kwargs.pop('metric_weights', None))

    # Turnkey Sulcal Guidance: automatically extract probability maps and configure weights
    if guided in (True, 'sulcal'):
        from .surface import extract_sulcal_probability_map
        if not isinstance(fixed, (list, tuple)):
            if verbose:
                print("Extracting sharp sulcal probability map for fixed image...")
            fixed_sulc = extract_sulcal_probability_map(fixed)
            fixed = [fixed, fixed_sulc]
        if not isinstance(moving, (list, tuple)):
            if verbose:
                print("Extracting sharp sulcal probability map for moving image...")
            moving_sulc = extract_sulcal_probability_map(moving)
            moving = [moving, moving_sulc]
        if syn_metric is None or syn_metric in ('cc2', 'lncc'):
            syn_metric = ['cc2', 'dice']
        if syn_metric_weights is None:
            if guided_weight is not None:
                w_s = float(guided_weight)
                syn_metric_weights = [1.0 - w_s, w_s]
            elif cohort_type == 'intra':
                syn_metric_weights = [0.80, 0.20]
            else:
                syn_metric_weights = [0.30, 0.70]
    elif guided == 'mind':
        from .landmarks.mind import compute_mind
        mind_offsets = kwargs.pop('mind_offsets', 4)
        if not isinstance(fixed, (list, tuple)):
            if verbose:
                print("Extracting physical LPS MIND descriptors for fixed image...")
            fixed_mind_t = compute_mind(fixed, n_offsets=mind_offsets)
            fixed_channels = [fixed]
            for c in range(fixed_mind_t.shape[1]):
                arr_c = fixed_mind_t[0, c].cpu().numpy()
                img_c = ants.from_numpy(arr_c, origin=fixed.origin, spacing=fixed.spacing, direction=fixed.direction)
                fixed_channels.append(img_c)
            fixed = fixed_channels

        if not isinstance(moving, (list, tuple)):
            if verbose:
                print("Extracting physical LPS MIND descriptors for moving image...")
            moving_mind_t = compute_mind(moving, n_offsets=mind_offsets)
            moving_channels = [moving]
            for c in range(moving_mind_t.shape[1]):
                arr_c = moving_mind_t[0, c].cpu().numpy()
                img_c = ants.from_numpy(arr_c, origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
                moving_channels.append(img_c)
            moving = moving_channels

        n_extra = len(fixed) - 1
        if syn_metric is None or syn_metric in ('cc2', 'lncc'):
            syn_metric = ['cc2'] + ['cc2'] * n_extra
        elif isinstance(syn_metric, str):
            syn_metric = [syn_metric] + ['cc2'] * n_extra

        if syn_metric_weights is None:
            w_mind = float(guided_weight) if guided_weight is not None else 0.20
            w_primary = 1.0 - w_mind
            syn_metric_weights = [w_primary] + [w_mind / max(n_extra, 1)] * n_extra

    # 1. Extract physical properties
    fixed_primary = fixed[0] if isinstance(fixed, (list, tuple)) else fixed
    moving_primary = moving[0] if isinstance(moving, (list, tuple)) else moving
    dim = fixed_primary.dimension
    grid_shape = fixed_primary.shape
    spacing = fixed_primary.spacing
    direction = fixed_primary.direction
    
    if inv_tolerance is None:
        inv_tolerance = 0.1 * min(spacing)
    
    # Apply initial transform if provided
    tx_list = []
    initial_grid = kwargs.pop('initial_grid', None)
    
    init_M_phys, init_t_phys = None, None
    
    if initial_grid is not None:
        perm_grid = (0, 2, 1, 3) if dim == 2 else (0, 3, 2, 1, 4)
        initial_grid = initial_grid.transpose(perm_grid)
    elif isinstance(initial_transform, str) and initial_transform.lower() == 'identity':
        # ANTs keyword: start from the identity transform in physical (scanner header) space,
        # bypassing the automatic center-of-mass initialization below -- mandatory for
        # opposite-contrast pairs (e.g. T1 structural -> BOLD/EPI) where intensity-centroid
        # alignment fails. `tx_list` stays empty: there is no real transform file to compose,
        # and passing the literal string "Identity" to ants.read_transform/apply_transforms
        # raises ("Transform Identity does not exist").
        import torch
        init_M_phys = torch.eye(dim, dtype=torch.float32)
        init_t_phys = torch.zeros(dim, dtype=torch.float32)
    elif initial_transform is not None:
        tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
        # a list containing a warp is not one linear map: use the dense initial grid instead
        init_M_phys, init_t_phys = parse_ants_affine(tx_list, dim, allow_nonlinear=True)
        if init_M_phys is None:
            initial_grid = compute_initial_grid(fixed_primary, moving_primary, tx_list)
            perm_grid = (0, 2, 1, 3) if dim == 2 else (0, 3, 2, 1, 4)
            initial_grid = initial_grid.transpose(perm_grid)
    else:
        # No caller-supplied initial transform/grid, and not the identity keyword: resolve the
        # initial affine/rigid alignment automatically via the single, validated robust_affine
        # solver -- unless the transform type explicitly means "skip affine alignment entirely"
        # (SyNOnly / greedy dispatch, mirrored from the tot_lower checks below).
        _tot_lower_early = type_of_transform.lower()
        _skip_align_early = (
            _tot_lower_early in ('synonly', 'syn_only') or
            _tot_lower_early in ('greedy', 'greedy_compositive') or
            str(kwargs.get('formulation', '')).lower() == 'greedy'
        )
        if not _skip_align_early:
            from .robust_affine import robust_affine
            _affine_dof_resolved = affine_dof
            if _affine_dof_resolved is None:
                _dof_req_early = kwargs.get('dof', None)
                if _dof_req_early == 'rigid' or _tot_lower_early in ('rigid', 'translation'):
                    _affine_dof_resolved = 'rigid'
                else:
                    _affine_dof_resolved = 'affine'
            _aff_kwargs = {}
            if _tot_lower_early in ('affine', 'rigid', 'translation'):
                if levels is not None:
                    _aff_kwargs['levels'] = levels
                if reg_iterations is not None:
                    _aff_kwargs['iterations'] = reg_iterations
            aff_res = robust_affine(
                fixed_primary, moving_primary,
                dof=_affine_dof_resolved, mode=affine_mode, seed=affine_seed, verbose=verbose,
                **_aff_kwargs
            )
            tx_list = aff_res['fwdtransforms']
            init_M_phys, init_t_phys = parse_ants_affine(tx_list, dim)
    moving_reg = moving
    
    from .core.pipeline import normalize_and_tensorize, auto_detect_device, cleanup_gpu
    
    # 2. Winsorize and Normalize numpy arrays
    I_tensor_unused, J_tensor_unused = normalize_and_tensorize(
        fixed, moving_reg, winsorize_quantiles=kwargs.get('winsorize_quantiles', None), backend=backend
    )
    # Re-fetch normalized arrays since SyNTo setup might still rely on numpy logic initially
    fi_np = fixed_primary.numpy()
    mi_np = moving_primary.numpy()
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
    if tot_lower in ['greedy', 'greedy_compositive'] or str(kwargs.get('formulation', '')).lower() == 'greedy':
        from .greedy import greedy_registration
        scales_to_use = levels if levels is not None else kwargs.pop('scales', None)
        return greedy_registration(
            fixed=fixed,
            moving=moving,
            reg_iterations=reg_iterations,
            scales=scales_to_use,
            learning_rate=kwargs.pop('learning_rate', kwargs.pop('optimizer_lr', 0.4)),
            flow_sigma=flow_sigma,
            total_sigma=total_sigma,
            similarity_metric=syn_metric,
            lncc_radius=syn_sampling,
            initial_transform=initial_transform,
            verbose=verbose,
            device=kwargs.pop('device', None),
            restrict_transformation=restrict_transformation,
            **kwargs
        )
    dof_req = kwargs.get('dof', None)
    if tot_lower == 'rigid':
        transform_type = 'Rigid'
        is_linear_only = True
    elif tot_lower == 'translation':
        transform_type = 'Translation'
        is_linear_only = True
    elif tot_lower == 'affine':
        transform_type = 'Rigid' if dof_req == 'rigid' else 'Affine'
        is_linear_only = True
    elif tot_lower in ['syn', 'synto']:
        transform_type = 'Rigid' if dof_req == 'rigid' else 'Affine'
        is_linear_only = False
    elif tot_lower in ['synonly', 'syn_only']:
        # ANTs semantics: deformable-only -- caller expects images already affinely aligned
        # (typically via `initial_transform`) and only wants the non-linear stage estimated.
        transform_type = 'Rigid' if dof_req == 'rigid' else 'Affine'
        is_linear_only = False
    elif tot_lower in ['bsplinesyn', 'bspline_syn', 'bspline']:
        transform_type = 'Affine'
        is_linear_only = False
        kwargs.setdefault('regularizer', 'bspline')
    else:
        raise ValueError(
            f"syntx.syn: unknown type_of_transform {type_of_transform!r}; expected 'SyN' / 'SyNTo', "
            "'SyNOnly', 'BSplineSyN', 'Affine', 'Rigid', 'Translation' or 'greedy' "
            "(use syntx.tvf / syntx.syngs for those methods)")

    if isinstance(reg_iterations, int):
        reg_iterations = [reg_iterations]

    if levels is None:
        if reg_iterations is not None:
            num_levels = len(reg_iterations)
            levels_to_use = [2**i for i in range(num_levels)][::-1] if num_levels > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
        else:
            levels_to_use = [4, 2, 1] if dim == 3 else [8, 4, 2, 1]
    else:
        levels_to_use = levels

    levels_len = len(levels_to_use)
    if is_linear_only:
        reg_iterations = [0] * levels_len
    elif reg_iterations is None:
        # 3D default aligned to the winning sobolev-regularizer config in
        # docs/provenance/best_parameters.json (90pair_population_benchmark_sobolev_mps):
        # reg_iterations=[100, 100, 20]. 2D has no benchmarked default, left unchanged.
        reg_iterations = [100, 100, 20] if dim == 3 else [100, 100, 100, 50]

    inverse_steps = kwargs.get('inverse_steps', inverse_steps)
    inverse_method = kwargs.get('inverse_method', inverse_method)
    vgg_layers = kwargs.get('vgg_layers', vgg_layers)
    vgg_patch_size = kwargs.get('vgg_patch_size', vgg_patch_size)
    vgg_num_patches = kwargs.get('vgg_num_patches', vgg_num_patches)
    vgg_mode = kwargs.get('vgg_mode', vgg_mode)
    vgg_lncc_window_size = kwargs.get('vgg_lncc_window_size', vgg_lncc_window_size)
        
    boundary_suppression_thresh = kwargs.get('boundary_suppression_thresh', None)
    image_grad_clip = kwargs.get('image_grad_clip', 6.0)


    # --- Parameter relevance validation ---
    reg_mode = str(kwargs.get('regularizer', kwargs.get('kernel_type', 'sobolev'))).lower()
    if kwargs.get('alpha') is not None:  # legacy alias of sobolev_alpha
        sobolev_alpha = kwargs.pop('alpha')
    kwargs.pop('alpha', None)
    _SPECTRAL_REGS = {'sobolev', 'dsti', 'dsti1'} | _CONTINUUM_REGS
    if reg_mode in _SPECTRAL_REGS and sobolev_alpha is not None:
        kwargs['sobolev_alpha'] = sobolev_alpha  # None -> fit()'s legacy sqrt(flow_sigma)/2
    # RegAdam / Adam family default 0.5 (was a 1e-3 sentinel meaning grad_step, i.e. a max
    # step of grad_step**2 -- badly under-registering); rprop / sgd keep 1e-3.
    optimizer_lr = resolve_optimizer_lr(optimizer, optimizer_lr)
    if reg_mode in _SPECTRAL_REGS:
        _default_flow_sigma = 2.4
        if (bool(fast_smooth) and isinstance(flow_sigma, (int, float))
                and flow_sigma != _default_flow_sigma and flow_sigma > 0):
            import warnings
            warnings.warn(
                f"flow_sigma={flow_sigma!r} has no effect on the kernel with regularizer="
                f"'{reg_mode}' and fast_smooth=True: the spectral operator's strength is "
                f"sobolev_alpha, and flow_sigma only switches the smoothing on (> 0) or off (0). "
                f"(With fast_smooth=False, flow_sigma sets the Gaussian post-filter.)",
                UserWarning, stacklevel=2,
            )
        if kwargs.get('gaussian_sigma') is not None:
            raise ValueError(
                f"gaussian_sigma is only valid with regularizer='gaussian'. "
                f"With regularizer='{reg_mode}', smoothing strength is controlled by "
                f"alpha (sobolev_alpha / dsti_alpha). Got gaussian_sigma={kwargs['gaussian_sigma']!r}. "
                f"Pass gaussian_sigma=None or omit it."
            )
    elif reg_mode == 'gaussian':
        for _p in ('alpha', 'sobolev_alpha', 'dsti_alpha'):
            if _p in kwargs and kwargs[_p] is not None:
                raise ValueError(
                    f"{_p} is only valid with spectral regularizers (sobolev, dsti, dsti1). "
                    f"With regularizer='gaussian', smoothing strength is controlled by flow_sigma. "
                    f"Got {_p}={kwargs[_p]!r}. Pass {_p}=None or omit it."
                )

        
    # Convert flow_sigma/total_sigma from ITK variance convention to actual sigma (std dev in mm).
    # ANTs/ITK uses SetVariance(v) where v = σ², so σ = √v.
    # Our separable_gaussian_filter takes σ directly.
    if isinstance(flow_sigma, (list, tuple)):
        fluid_sigma_actual = [math.sqrt(s) if s > 0 else 0.0 for s in flow_sigma]
    else:
        fluid_sigma_actual = math.sqrt(flow_sigma) if flow_sigma > 0 else 0.0

    if isinstance(total_sigma, (list, tuple)):
        elastic_sigma_actual = [math.sqrt(s) if s > 0 else 0.0 for s in total_sigma]
    else:
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
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed_primary.origin, direction=direction,
            fluid_sigma=fluid_sigma_actual, elastic_sigma=elastic_sigma_actual, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps, in_loop_inv_steps=in_loop_inv_steps, project_inverse=project_inverse,
            use_ants_pseudo_gradient=use_analytical,
            projection_frequency=projection_frequency, interpolator=interpolator,
            boundary_suppression_thresh=boundary_suppression_thresh,
            stationary_boundary=kwargs.get('stationary_boundary', True),
            image_grad_clip=image_grad_clip,
            antisymmetric=antisymmetric,
            inv_tolerance=inv_tolerance,
            dual_gradient=kwargs.get('dual_gradient', False),
            dual_gradient_weight=kwargs.get('dual_gradient_weight', 0.5),
            restrict_transformation=restrict_transformation,
            seed=seed
        ).to(device)
        model.formulation = kwargs.get('formulation', 'eulerian')
        model.smooth_in_deformed_space = kwargs.get('smooth_in_deformed_space', False)
        model.kernel_type = kwargs.get('kernel_type', 'bessel')
    elif backend == 'jax':
        if restrict_transformation is not None:
            raise NotImplementedError(
                "restrict_transformation is only implemented for backend='pytorch'. "
                "Passing it with backend='jax' would silently be ignored, which is worse "
                "than failing loudly -- use backend='pytorch' if you need axis-restricted "
                "deformation."
            )
        from .syn_jax import SyNTo as SyNToJax
        import jax.numpy as jnp
        I_tensor, J_tensor = normalize_and_tensorize(
            fixed, moving_reg, winsorize_quantiles=kwargs.get('winsorize_quantiles', None),
            backend='jax'
        )
        
        model = SyNToJax(
            dim=dim, grid_shape=grid_shape_zyx, spacing=sp_ordered, origin=fixed_primary.origin, direction=direction,
            fluid_sigma=fluid_sigma_actual, elastic_sigma=elastic_sigma_actual, transform_type=transform_type,
            inverse_method=inverse_method, inverse_steps=inverse_steps, project_inverse=project_inverse,
            projection_frequency=projection_frequency, interpolator=interpolator,
            boundary_suppression_thresh=boundary_suppression_thresh,
            image_grad_clip=image_grad_clip,
            antisymmetric=antisymmetric
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    # levels_to_use is defined above
        
    smoothing_sigmas = kwargs.get('smoothing_sigmas', None)
    if smoothing_sigmas is None:
                smoothing_sigmas = [float(np.log2(s)) if s > 1 else 0.0 for s in levels_to_use]
        
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in (
        'use_analytical_gradients', 'similarity_metric', 'reg_iterations',
        'reg_epochs', 'epochs_per_level', 'levels',
        'cfl_voxels', 'syn_metric_weights', 'lncc_radius', 'mattes_bins', 'sampling_percentage',
        'vgg_layers', 'vgg_patch_size', 'vgg_num_patches', 'vgg_mode', 'vgg_lncc_window_size',
        'initial_grid',
        'grad_step', 'regularizer', 'sobolev_alpha', 'fast_smooth', 'verbose', 'optimizer',
        'optimizer_type', 'optimizer_lr', 'interpolator', 'initial_transform', 'fixed_spacing',
        'fixed_origin', 'fixed_direction', 'moving_spacing', 'moving_origin', 'moving_direction',
        'smoothing_sigmas'
    )}
    if backend == 'pytorch':
        initial_grid_tensor = torch.tensor(initial_grid, dtype=torch.float32, device=device) if initial_grid is not None else None
        model.fit(
            I_tensor, J_tensor,
            levels=levels_to_use,
            epochs_per_level=reg_iterations,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            syn_metric_weights=syn_metric_weights,
            lncc_radius=syn_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size,
            initial_grid=initial_grid_tensor,
            fixed_spacing=fixed_primary.spacing,
            fixed_origin=fixed_primary.origin,
            fixed_direction=fixed_primary.direction,
            moving_spacing=moving_primary.spacing,
            moving_origin=moving_primary.origin,
            moving_direction=moving_primary.direction,
            smoothing_sigmas=smoothing_sigmas,
            regularizer=kwargs.get('regularizer', kwargs.get('kernel_type', 'sobolev')),
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=use_analytical,
            init_M_phys=init_M_phys,
            init_t_phys=init_t_phys,
            interpolator=interpolator,
            seed=seed,
            **fit_kwargs
        )
    else:
        import jax.numpy as jnp
        initial_grid_tensor = jnp.array(initial_grid) if initial_grid is not None else None
        # like the PyTorch backend, the JAX fit never optimises the affine: it is robust_affine's /
        # initial_transform's, or the identity for SyNOnly (SyNTo.fit would otherwise run its
        # own [100, 50, 20] Mattes-MI affine stage)
        fit_kwargs['affine_epochs'] = 0
        model.fit(
            I_tensor, J_tensor,
            levels=levels_to_use,
            epochs_per_level=reg_iterations,
            cfl_voxels=grad_step,
            similarity_metric=syn_metric,
            syn_metric_weights=syn_metric_weights,
            lncc_radius=syn_sampling,
            sampling_percentage=sampling_percentage,
            vgg_layers=vgg_layers,
            vgg_patch_size=vgg_patch_size,
            vgg_num_patches=vgg_num_patches,
            vgg_mode=vgg_mode,
            vgg_lncc_window_size=vgg_lncc_window_size,
            initial_grid=initial_grid_tensor,
            fixed_spacing=fixed_primary.spacing,
            fixed_origin=fixed_primary.origin,
            fixed_direction=fixed_primary.direction,
            moving_spacing=moving_primary.spacing,
            moving_origin=moving_primary.origin,
            moving_direction=moving_primary.direction,
            smoothing_sigmas=smoothing_sigmas,
            regularizer=reg_mode,          # same default ('sobolev') as the PyTorch backend
            sobolev_alpha=kwargs.get('sobolev_alpha', kwargs.get('alpha', None)),
            fast_smooth=fast_smooth,
            verbose=verbose,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            use_analytical_gradients=use_analytical,
            init_M_phys=init_M_phys.cpu().numpy() if init_M_phys is not None else None,
            init_t_phys=init_t_phys.cpu().numpy() if init_t_phys is not None else None,
            interpolator=interpolator,
            **fit_kwargs
        )
    
    # 4. Save displacement fields to temp files to match ANTs file-based transforms
    outprefix = kwargs.get('outprefix', None)
    if outprefix is not None:
        os.makedirs(os.path.dirname(outprefix) or '.', exist_ok=True)
        fwd_file = f"{outprefix}1Warp.nii.gz"
        inv_file = f"{outprefix}1InverseWarp.nii.gz"
    else:
        fwd_file = tempfile.NamedTemporaryFile(suffix='_fwd.nii.gz', delete=False).name
        inv_file = tempfile.NamedTemporaryFile(suffix='_inv.nii.gz', delete=False).name
    
    affine_file = None
    affine_inv_file = None
    
    total_fwd_deformable = None
    total_inv_deformable = None

    if backend == 'pytorch':
        with torch.no_grad():
            if hasattr(model, 'warp_l2r') and hasattr(model, 'warp_r2l'):
                total_fwd_deformable = model.warp_l2r.data
                total_inv_deformable = model.warp_r2l.data

            if hasattr(model, 'affine'):
                # Convert internal grid affine to physical ITK AffineTransform
                T_grid = model.affine.get_matrix().detach().cpu().numpy()
                if verbose >= 2:
                    print(f"[pytorch] T_grid:\n", T_grid)
                moving_target = fixed_primary if initial_grid is not None else moving_primary
                M_phys, t_phys = grid_to_physical_affine(T_grid, fixed_primary, moving_target)
                
                # Save physical forward affine transform to file
                if outprefix is not None:
                    affine_file = f"{outprefix}0GenericAffine.mat"
                    affine_inv_file = f"{outprefix}0GenericAffine_inv.mat"
                else:
                    affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
                    affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
                tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)
                ants.write_transform(tx_inv, affine_inv_file)
    else:
        # For JAX:
        import jax
        import jax.numpy as jnp
        from .syn_jax import get_affine_matrix_jax, get_physical_grid_jax, physical_to_normalized_jax, jax_grid_sample
        
        if hasattr(model, 'warp_l2r') and hasattr(model, 'warp_r2l'):
            total_fwd_deformable = model.warp_l2r
            total_inv_deformable = model.warp_r2l
        
        if hasattr(model, 'affine_params'):
            T_grid = get_affine_matrix_jax(model.affine_params, dim, model.transform_type)
            T_grid = np.array(T_grid)
            if verbose >= 2:
                print(f"[jax] T_grid:\n", T_grid)
            moving_target = fixed_primary if initial_grid is not None else moving_primary
            M_phys, t_phys = grid_to_physical_affine(T_grid, fixed_primary, moving_target)
            
            # Save physical forward affine transform to file
            if outprefix is not None:
                affine_file = f"{outprefix}0GenericAffine.mat"
                affine_inv_file = f"{outprefix}0GenericAffine_inv.mat"
            else:
                affine_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
                affine_inv_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
            tx_fwd, tx_inv = export_ants_affine_transform(M_phys, t_phys, dim=dim, filename=affine_file)
            ants.write_transform(tx_inv, affine_inv_file)
        
    if sum(reg_iterations) > 0:
        if total_fwd_deformable is None:
            tensor_shape = itk_shape_to_tensor_shape(fixed_primary.shape)
            total_fwd_deformable = np.zeros((1, *tensor_shape, dim), dtype=np.float32)
            total_inv_deformable = np.zeros((1, *tensor_shape, dim), dtype=np.float32)

        fwd_img = disp_tensor_to_itk(total_fwd_deformable, fixed_primary)
        inv_img = disp_tensor_to_itk(total_inv_deformable, fixed_primary)
        
        ants.image_write(fwd_img, fwd_file)
        ants.image_write(inv_img, inv_file)
        
        if affine_file is not None and (initial_transform is None or init_M_phys is not None):
            fwd_transforms = [fwd_file, affine_file]
            inv_transforms = [affine_file, inv_file]
            whichtoinvert_inv = [True, False]
        elif initial_transform is not None:
            fwd_transforms = [fwd_file] + tx_list
            inv_transforms = tx_list + [inv_file]
            whichtoinvert_inv = [True] * len(tx_list) + [False]
        else:
            fwd_transforms = [fwd_file]
            inv_transforms = [inv_file]
            whichtoinvert_inv = [False]
    else:
        if affine_file is not None and (initial_transform is None or init_M_phys is not None):
            fwd_transforms = [affine_file]
            inv_transforms = [affine_file]
            whichtoinvert_inv = [True]
        elif initial_transform is not None:
            fwd_transforms = tx_list
            inv_transforms = tx_list
            whichtoinvert_inv = [True] * len(tx_list)
        else:
            fwd_transforms = []
            inv_transforms = []
            whichtoinvert_inv = []
    
    inverse_identity_errors = {}
    if sum(reg_iterations) > 0 and hasattr(model, 'warp_l2r') and hasattr(model, 'warp_r2l'):
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
            
        inverse_identity_errors['phi_1'] = calculate_inverse_identity_error(w_l2r, w_l2r_inv, fixed_primary.spacing, fixed_primary.origin, fixed_primary.direction)
        inverse_identity_errors['phi_2'] = calculate_inverse_identity_error(w_r2l, w_r2l_inv, fixed_primary.spacing, fixed_primary.origin, fixed_primary.direction)
        
    # 6. Apply transforms to generate warped output images
    warpedmovout = ants.apply_transforms(fixed=fixed_primary, moving=moving_primary, transformlist=fwd_transforms)
    warpedfixout = ants.apply_transforms(fixed=moving_primary, moving=fixed_primary, transformlist=inv_transforms, whichtoinvert=whichtoinvert_inv)
    
    fwd_midpoint_warp = None
    inv_midpoint_warp = None
    midpoint_fixed = None
    midpoint_moving = None

    if sum(reg_iterations) > 0 and hasattr(model, 'midpoint_warp_l2r') and hasattr(model, 'midpoint_warp_r2l'):
        fwd_mid_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
        inv_mid_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name

        fwd_mid_img = disp_tensor_to_itk(model.midpoint_warp_l2r, fixed_primary)
        inv_mid_img = disp_tensor_to_itk(model.midpoint_warp_r2l, fixed_primary)

        ants.image_write(fwd_mid_img, fwd_mid_file)
        ants.image_write(inv_mid_img, inv_mid_file)

        fwd_midpoint_warp = fwd_mid_file
        inv_midpoint_warp = inv_mid_file

        midpoint_fixed = ants.apply_transforms(fixed=fixed_primary, moving=fixed_primary, transformlist=[fwd_midpoint_warp])
        if affine_file is not None and (initial_transform is None or init_M_phys is not None):
            midpoint_moving = ants.apply_transforms(fixed=fixed_primary, moving=moving_primary, transformlist=[inv_midpoint_warp, affine_file])
        elif initial_transform is not None:
            midpoint_moving = ants.apply_transforms(fixed=fixed_primary, moving=moving_primary, transformlist=[inv_midpoint_warp] + tx_list)
        else:
            midpoint_moving = ants.apply_transforms(fixed=fixed_primary, moving=moving_primary, transformlist=[inv_midpoint_warp])

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

    if kwargs.get('in_memory', False) and outprefix is None:
        metadata = {'origin': fixed.origin, 'spacing': fixed.spacing, 'direction': fixed.direction, 'shape': fixed.shape}
        if hasattr(model, 'get_forward_transform'):
            tx = model.get_forward_transform(metadata)
            mem_export = tx.export(outprefix=None)
            ret_dict['fwd_warp'] = mem_export['fwd_warp']
            ret_dict['inv_warp'] = mem_export['inv_warp']
            ret_dict['affine_matrix'] = mem_export['affine_matrix']
            ret_dict['transform'] = tx

    try:
        from .reporting import build_engine_provenance
        fit_time_val = (time.time() - t_start) if 't_start' in locals() else None
        provenance = build_engine_provenance(
            algorithm="syntx.syn",
            backend=backend,
            device=str(device) if 'device' in locals() and device is not None else "cpu",
            fit_time=fit_time_val,
            reg_iterations=reg_iterations,
            solver="SyN",
            fluid_sigma=flow_sigma,
            elastic_sigma=total_sigma,
            learning_rate=grad_step,
            optimizer_type=optimizer,
            optimizer_lr=optimizer_lr,
            similarity_metric=syn_metric,
            syn_sampling=syn_sampling,
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
            fast_smooth=fast_smooth,
            antisymmetric=antisymmetric,
            use_analytical_gradients=use_analytical
        )
        ret_dict['provenance'] = provenance
    except Exception:
        pass
    
    cleanup_gpu(device=device if 'device' in locals() else None, backend=backend)

    return ret_dict


syn = registration


def auto_reg(
    fixed,
    moving,
    type_of_transform=None,
    guided=None,
    cohort_type='auto',
    guided_weight=None,
    robust_affine='auto',
    denoise=False,
    diagnose=True,
    fixed_label=None,
    moving_label=None,
    levels=None,
    reg_iterations=None,
    iterations=None,
    verbose=False,
    seed=42,
    **kwargs
):
    """
    One-call registration with automatic choices -- ``syntx.auto_reg``.

    Inspects the pair, picks a method and pre-processing, runs it with that method's own
    defaults, and reports quality metrics::

        res = syntx.auto_reg(fixed, moving)
        res['warpedmovout'], res['fwdtransforms'], res['metrics']

    Steps:

    1. Diagnosis (``diagnose=True``): ``syntx.diagnose_pair`` + ``syntx.synthesize_policy``
       guess modality / body part and propose a transform type, guidance, denoising,
       similarity metric and (for CT) an intensity window. Anything you pass explicitly wins;
       a diagnosis error is ignored (printed with ``verbose``).
    2. Optional Rician non-local-means denoising (3-D, ``denoise``; needs antstorch).
    3. Registration with the chosen method -- ``syntx.tvf`` (default), ``syntx.syn`` (default
       when ``guided`` is set), ``syntx.syngs``, or ``syntx.robust_affine`` for linear-only
       types -- each with its own defaults, the [100, 100, 20] schedule, and its own
       ``robust_affine`` initial alignment unless ``initial_transform`` is given.
    4. Metrics, added to the result as ``res['metrics']``.

    Parameters
    ----------
    fixed, moving : ANTsImage
        2-D or 3-D images.
    type_of_transform : str or None, default None
        'TVF' (also 'DIRICHLET_TVF', 'DSTI_TVF', 'TIME_VARYING'), 'SyNGS' (also 'GEODESIC',
        'SYN_GS', 'EPDIFF'), any ``syntx.syn`` type ('SyN', 'SyNOnly', 'BSplineSyN', ...;
        others raise ValueError there), or linear only: 'Affine',
        'Rigid', 'Translation', 'AFFINE_ONLY', 'ROBUST_AFFINE'. None: from the diagnosis, else
        'TVF' ('SyN' when ``guided``).
    guided : True, 'sulcal' or None, default None
        SyN only: add a sulcal-probability channel matched with a Dice term.
    cohort_type : {'auto', 'inter', 'intra'}, default 'auto'
        Sulcal-guidance weighting: same-site ('intra': 0.80 / 0.20) or cross-site
        ('inter' / default: 0.30 / 0.70 intensity / sulcal).
    guided_weight : float or None
        Explicit sulcal weight (overrides cohort_type).
    robust_affine : True, 'auto' or a robust_affine mode, default 'auto'
        Initial-alignment mode. TVF / SyN / SyNGS compute their own alignment (``affine_mode``
        for SyN takes this value); linear-only types call ``syntx.robust_affine`` directly.
    denoise : bool or 'auto', default False
        3-D: denoise both images first (antstorch.denoise_image, Rician).
    diagnose : bool, default True
        Run the automatic diagnosis / policy step.
    fixed_label, moving_label : ANTsImage, optional
        Label maps; if both are given the symmetric Dice is reported.
    levels : int or list of int, optional
        Pyramid downsampling shrink factors (e.g. [4, 2, 1] or [8, 4, 2, 1]) forwarded to the
        underlying registration solver.
    reg_iterations : int or list of int, optional
        Iterations per pyramid level. ``iterations`` is accepted as an alias.
    verbose : bool, default False
    seed : int, default 42
    **kwargs
        Passed to the chosen registration function (e.g. ``alpha=``, ``grad_step=``,
        ``flow_sigma=``, ``reg_iterations=``, ``initial_transform=``, ``backend=`` (default
        'pytorch'), ``device=`` (default CUDA, then MPS, then CPU)); unknown keywords raise
        there. For SyN, auto_reg sets ``interpolator='linear'``, ``levels=[4, 2, 1]`` and the
        [100, 100, 20] schedule unless given.

    Returns
    -------
    dict
        The chosen registration's result (``'warpedmovout'``, ``'fwdtransforms'``,
        ``'invtransforms'``, ...) plus ``'metrics'``:
        ``execution_time_seconds``, ``device_used``, ``backend_used``,
        ``type_of_transform_used``; Jacobian statistics of the forward warp
        (``jac_mean / min / max / std``, ``folding_pct`` inside the fixed mask, from
        ``syntx.liouville_determinant`` when the method supports it, else the finite-
        difference Jacobian; ``jac_measure`` says which; ``fd_jac_min`` / ``fd_folding_pct``
        are always the finite-difference values);
        ``smooth_1st`` / ``smooth_2nd`` (mean first / second displacement derivatives);
        ``lncc_score``, ``mse_score``, ``mattes_mi_score`` (fixed vs warped);
        ``inverse_identity_mean_error`` / ``_max_error``; with labels ``dice_fixed``,
        ``dice_moving``, ``dice_symmetric`` (also copied to the top level); with diagnosis
        ``diagnosis`` and ``policy_explanation``.
    """
    import time
    import ants
    import gc
    import torch
    t0 = time.time()

    # 1. Hardware & backend auto-detection
    target_backend = kwargs.pop('backend', 'pytorch')

    if reg_iterations is None:
        reg_iterations = iterations
    if reg_iterations is None and 'iterations' in kwargs:
        reg_iterations = kwargs.pop('iterations')
    elif 'iterations' in kwargs:
        kwargs.pop('iterations')

    target_device = kwargs.pop('device', None)
    if target_device is None:
        if torch.cuda.is_available():
            target_device = 'cuda'
        elif torch.backends.mps.is_available():
            target_device = 'mps'
        else:
            target_device = 'cpu'

    dim = fixed.dimension if hasattr(fixed, 'dimension') else (fixed.ndim if hasattr(fixed, 'ndim') else 3)
    fixed_proc = fixed
    moving_proc = moving

    # 2. Autonomous Diagnosis & Policy Synthesis
    pair_diag = None
    policy = None
    policy_metric = None
    if diagnose:
        try:
            from .diagnose import diagnose_pair
            from .policy import synthesize_policy
            pair_diag = diagnose_pair(fixed, moving, fast=False)
            policy = synthesize_policy(pair_diag)
            if verbose:
                print(f"[auto_reg] Autonomous Diagnosis: {pair_diag.relationship} | Fixed: {pair_diag.fixed.body_part} ({pair_diag.fixed.modality}) | Moving: {pair_diag.moving.body_part} ({pair_diag.moving.modality})")
                print(f"[auto_reg] Synthesized Policy: {policy.explanation}")
        except Exception as e:
            if verbose:
                print(f"[auto_reg] Autonomous diagnosis fallback: {e}")

    # Parameter resolution: Explicit user arguments take strict precedence over synthesized policy
    if policy is not None:
        if type_of_transform is None:
            type_of_transform = policy.transform_type
            if guided is None:
                guided = policy.guided
        if denoise is False:
            denoise = policy.denoise
        if cohort_type == 'auto' and policy.cohort_type != 'auto':
            cohort_type = policy.cohort_type
        if robust_affine == 'auto' and policy.robust_affine != 'auto':
            robust_affine = policy.robust_affine
        policy_metric = (policy.similarity_metric
                         if 'similarity_metric' not in kwargs and 'syn_metric' not in kwargs else None)
        # CT Windowing if diagnosed CT
        if policy.ct_window is not None and pair_diag is not None and pair_diag.fixed.is_ct():
            w_min, w_max = policy.ct_window
            arr_f = np.clip((fixed_proc.numpy() - w_min) / max(w_max - w_min, 1e-4), 0.0, 1.0).astype(np.float32)
            arr_m = np.clip((moving_proc.numpy() - w_min) / max(w_max - w_min, 1e-4), 0.0, 1.0).astype(np.float32)
            fixed_proc = fixed_proc.new_image_like(arr_f)
            moving_proc = moving_proc.new_image_like(arr_m)

    # 3. Adaptive Preprocessing: Non-local means Rician denoising
    if denoise in (True, 'auto') and dim == 3 and hasattr(fixed, 'dimension'):
        try:
            import antstorch
            if verbose:
                print("Applying adaptive non-local means denoising...")
            fixed_proc = antstorch.denoise_image(fixed_proc, shrink_factor=2, p=1, r=1, noise_model="Rician")
            moving_proc = antstorch.denoise_image(moving_proc, shrink_factor=2, p=1, r=1, noise_model="Rician")
        except Exception as e:
            if verbose:
                print(f"[auto_reg] Denoising fallback: {e}")

    # 4. Determine Transform Type
    if type_of_transform is None:
        if guided in (True, 'sulcal'):
            transform_type = 'SyN'
        else:
            transform_type = 'TVF'
    else:
        transform_type = type_of_transform

    transform_type_upper = str(transform_type).upper()
    is_tvf = transform_type_upper in ('TVF', 'DIRICHLET_TVF', 'DSTI_TVF', 'TIME_VARYING')
    is_syngs = transform_type_upper in ('SYNGS', 'GEODESIC', 'SYN_GS', 'EPDIFF')
    is_affine_only = transform_type_upper in ('AFFINE', 'RIGID', 'TRANSLATION', 'AFFINE_ONLY', 'ROBUST_AFFINE')
    # `registration()` (plain SyN/SyNTo), `syngs_registration()`, and `tvf_registration()`
    # all now resolve their own initial affine/rigid alignment via `robust_affine`
    # internally whenever `initial_transform` isn't supplied (see each function's
    # `initial_transform`/`affine_dof`/`affine_mode`/`affine_seed` handling). Precomputing
    # it here too would just be a redundant, wasted second `robust_affine` call for all
    # three, so it's skipped below for all of them -- only `is_affine_only` (which calls
    # `robust_affine` directly itself, not through one of these wrappers) still needs
    # `auto_reg`'s own precomputation.
    is_self_resolving_backend = not is_affine_only
    if policy_metric is not None and not is_affine_only:
        kwargs['syn_metric'] = policy_metric          # the metric keyword all three backends take

    # 4. Deterministic Robust Affine Initialization
    initial_transform = kwargs.pop('initial_transform', None)
    aff_mode = robust_affine if isinstance(robust_affine, str) and robust_affine in ('translation_only', 'com_only', 'pytorch', 'ants_fast') else 'auto'
    should_run_affine = (
        (robust_affine is True) or
        (isinstance(robust_affine, str) and robust_affine in ('auto', 'translation_only', 'com_only', 'pytorch', 'ants_fast')) or
        (robust_affine == 'auto' and dim == 3 and hasattr(fixed, 'dimension'))
    ) and initial_transform is None and not is_affine_only and not is_self_resolving_backend

    if should_run_affine:
        from .robust_affine import robust_affine as run_robust_affine
        if verbose:
            print(f"Computing deterministic multi-start robust affine initialization (mode='{aff_mode}')...")
        aff_res = run_robust_affine(fixed_proc, moving_proc, mode=aff_mode, seed=seed, verbose=verbose)
        initial_transform = aff_res['fwdtransforms']

    # 6. Execute Registration with Proven Best Parameters
    if is_affine_only:
        from .robust_affine import robust_affine as run_robust_affine
        aff_params = {
            'mode': 'auto',
            'verbose': verbose,
            'seed': seed
        }
        if levels is not None:
            aff_params['levels'] = levels
        if reg_iterations is not None:
            aff_params['iterations'] = reg_iterations
        aff_params.update(kwargs)
        res = run_robust_affine(fixed=fixed_proc, moving=moving_proc, **aff_params)
        transform_label = f"Robust Affine ({transform_type})"
    elif is_tvf:
        from .tvf import tvf_registration
        # syntx.tvf's own defaults; only routing and the benchmark schedule here
        tvf_params = {
            'backend': target_backend,
            'device': target_device,
            'reg_iterations': reg_iterations if reg_iterations is not None else [100, 100, 20],
            'initial_transform': initial_transform,
            'verbose': verbose
        }
        if levels is not None:
            tvf_params['levels'] = levels
        tvf_params.update(kwargs)
        res = tvf_registration(fixed=fixed_proc, moving=moving_proc, **tvf_params)
        transform_label = "TVF"
    elif is_syngs:
        from .syngs import syngs_registration
        # syngs's own (canonical, tuned) defaults; only routing and the benchmark schedule here
        syngs_params = {
            'backend': target_backend,
            'device': target_device,
            'reg_iterations': reg_iterations if reg_iterations is not None else [100, 100, 20],
            'initial_transform': initial_transform,
            'verbose': verbose
        }
        if levels is not None:
            syngs_params['levels'] = levels
        syngs_params.update(kwargs)
        res = syngs_registration(fixed=fixed_proc, moving=moving_proc, **syngs_params)
        transform_label = "SyNGS (Riemannian Geodesic)"
    else:
        syn_params = {
            'backend': target_backend,
            'device': target_device,
            'type_of_transform': transform_type if transform_type else 'SyN',
            'levels': levels if levels is not None else [4, 2, 1],
            'reg_iterations': reg_iterations if reg_iterations is not None else [100, 100, 20],
            # regularisation / optimiser parameters: syntx.syn's (tuned, canonical) defaults
            'interpolator': 'linear',
            'bootstrap_mode': 'antithetic',
            'use_analytical_gradients': False,
            'use_ants_pseudo_gradient': False,
            'initial_transform': initial_transform,
            'affine_mode': aff_mode,
            'affine_seed': seed,
            'guided': guided,
            'cohort_type': cohort_type,
            'guided_weight': guided_weight,
            'seed': seed,
            'verbose': verbose
        }
        syn_params.update(kwargs)
        res = registration(fixed=fixed_proc, moving=moving_proc, **syn_params)
        transform_label = "SyN (Eulerian Sobolev Guided)" if guided in (True, 'sulcal') else "SyN (Eulerian Sobolev)"

    t_elapsed = time.time() - t0

    # 5. Compute Comprehensive Metrics
    warpedmovout = res['warpedmovout']
    fwd_tx = res['fwdtransforms']

    metrics = {
        'execution_time_seconds': float(t_elapsed),
        'device_used': str(target_device),
        'backend_used': str(target_backend),
        'type_of_transform_used': transform_label
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

            # Folding: the per-method determinant (syntx.liouville_determinant) when the result
            # supports it, else the finite-difference Jacobian; the FD values are kept as fd_*
            from .deformation_metrics import compute_jacobian_metrics, flow_jacobian_metrics
            fd = compute_jacobian_metrics(fixed, warp_file)
            metrics['fd_jac_min'] = fd['min']
            metrics['fd_folding_pct'] = fd['folding_pct']
            flow = flow_jacobian_metrics(fixed, res)
            src = flow if flow is not None else dict(fd, measure="finite difference of the exported field")
            metrics['jac_mean'] = float(src['mean'])
            metrics['jac_min'] = float(src['min'])
            metrics['jac_max'] = float(src['max'])
            metrics['jac_std'] = float(src.get('std', float('nan')))
            metrics['folding_pct'] = float(src['folding_pct'])
            metrics['jac_measure'] = src['measure']

            if disp_np.ndim == 4:  # 3D image
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
        if 'phi_1' in inv_errs and isinstance(inv_errs['phi_1'], dict):
            metrics['inverse_identity_mean_error'] = float(inv_errs['phi_1'].get('mean_error', float('nan')))
            metrics['inverse_identity_max_error'] = float(inv_errs['phi_1'].get('max_error', float('nan')))
        else:
            err_vals_mean = [v['mean_error'] for v in inv_errs.values() if isinstance(v, dict) and 'mean_error' in v]
            err_vals_max = [v['max_error'] for v in inv_errs.values() if isinstance(v, dict) and 'max_error' in v]
            if err_vals_mean:
                metrics['inverse_identity_mean_error'] = float(np.mean(err_vals_mean))
            if err_vals_max:
                metrics['inverse_identity_max_error'] = float(np.max(err_vals_max))

    # Anatomical Segmentation Evaluation if ground truth labels provided
    if fixed_label is not None and moving_label is not None:
        from .deformation_metrics import compute_bidirectional_dice
        n_inv = len(res.get('invtransforms', []))
        which_inv = res.get('whichtoinvert_inv', [True] if n_inv == 1 else ([True, False] if n_inv == 2 else [True] + [False] * (n_inv - 1)))
        try:
            d_fix, d_mov, d_sym = compute_bidirectional_dice(
                fixed_label, moving_label, fixed_proc, moving_proc,
                fwd_tx, res.get('invtransforms', []), which_inv
            )
            metrics['dice_fixed'] = float(d_fix)
            metrics['dice_moving'] = float(d_mov)
            metrics['dice_symmetric'] = float(d_sym)
            metrics['dice_sym'] = float(d_sym)
            res['dice_fixed'] = float(d_fix)
            res['dice_moving'] = float(d_mov)
            res['dice_symmetric'] = float(d_sym)
        except Exception as e:
            if verbose:
                print(f"[auto_reg] Label DICE calculation skipped: {e}")

    if pair_diag is not None:
        metrics['diagnosis'] = {
            'relationship': pair_diag.relationship,
            'fixed_modality': pair_diag.fixed.modality,
            'fixed_anatomy': pair_diag.fixed.body_part,
            'moving_modality': pair_diag.moving.modality,
            'moving_anatomy': pair_diag.moving.body_part,
            'confidence': pair_diag.confidence,
        }
        if policy is not None:
            metrics['policy_explanation'] = policy.explanation

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

SyNTo.registration = staticmethod(registration)



