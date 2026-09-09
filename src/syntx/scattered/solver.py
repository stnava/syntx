"""
syntx.scattered.solver — Diffeomorphic SyN Registration for Scattered Data
==========================================================================

Symmetric Diffeomorphic Normalization (SyN) registration solver for scattered
point clouds (Lagrangian-to-Lagrangian) and mixed point-to-grid (Lagrangian-to-
Eulerian) representations.

Integrates:
- Differentiable Nadaraya-Watson kernel regression projection (M1)
- Continuous fluid velocity regularization via Discrete Sine Transform Type-I
  Green's operator (Dirichlet zero-boundary, MPS/CUDA/CPU native)
- Courant-Friedrichs-Lewy (CFL) step bounding ensuring fold-free bijectivity (det(J) > 0)
- Lagrangian pullback step composition preventing Eulerian shearing
- Antisymmetric geodesic projection anchoring the midpoint at the Fréchet mean
- Type-I Anderson-accelerated fixed-point inversion driving inverse consistency error < 10^-3
- Analytical LNCC pseudo-gradients and autograd backpropagation
- Coarse-to-fine multi-resolution pyramid hierarchies
- Differentiable coordinate warping, feature transport, and grid pullback (M2)
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple, Union, Sequence, Literal, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from syntx.core.smoothing import (
    apply_dsti1_green_operator,
    apply_dsti_green_operator,
    apply_sobolev_green_operator,
    separable_gaussian_filter,
    get_boundary_mask,
)
from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
)
from syntx.core.jacobian import (
    compute_physical_jacobian_determinant,
    compute_jacobian_determinant_nd,
)
from syntx.core.losses import (
    local_ncc_loss_nd,
    AnalyticalLNCC,
)

from .projection import (
    ScatteredProjector,
    project_scattered_to_grid,
    compute_adaptive_sigma,
    compute_distance_transform_to_grid,
)
from .mapping import warp_scattered_coordinates, evaluate_field_at_scattered, _resolve_domain_bounds
from .transport import (
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
    _standardize_grid_features,
)


@dataclass
class ScatteredRegistrationConfig:
    """Configuration options for scattered diffeomorphic SyN registration.

    Parameters
    ----------
    dim : int, default 2
        Spatial dimensionality of coordinates (2 or 3).
    grid_res : int or Tuple[int, ...], default 128
        Eulerian grid resolution. If int, creates an isotropic grid of shape (grid_res,) * dim.
    domain_bounds : tuple or str, default (-1.0, 1.0)
        Coordinate bounding box of the Eulerian grid.
    sigma : float, sequence of float, or 'auto', default 0.03
        Gaussian kernel standard deviation for Nadaraya-Watson projection.
    fluid_sigma : float, default 1.5
        Standard deviation of Gaussian / Green fluid velocity smoothing operator.
    elastic_sigma : float, default 0.0
        Standard deviation of elastic displacement field smoothing (0.0 = disabled).
    regularizer : {'dsti1', 'dsti', 'sobolev', 'gaussian'}, default 'dsti1'
        Spectral or spatial fluid regularization operator.
    optimizer_type : {'rprop', 'cfl', 'adam', 'reg_adam'}, default 'rprop'
        Optimization algorithm for Eulerian displacement updates.
    optimizer_lr : float, default 0.05
        Base learning rate / step size.
    in_loop_inv_steps : int, default 5
        Number of in-loop Anderson fixed-point inverse updates per SyN iteration.
    inverse_steps : int, default 20
        Number of final Anderson fixed-point iterations for full inverse field refinement.
    inverse_method : {'anderson', 'fixed_point'}, default 'anderson'
        Inversion solver algorithm.
    cfl_voxels : float, default 0.25
        Maximum step displacement bound in grid voxel units (CFL condition).
    use_analytical_gradients : bool, default False
        If True, uses analytical LNCC pseudo-gradients; if False, uses autograd backward.
    similarity_metric : {'lncc', 'mse'}, default 'lncc'
        Similarity metric driving registration.
    window_size : int, default 15
        LNCC kernel window size in voxels.
    iterations : int or Sequence[int], default 100
        Number of iterations per multi-resolution level.
    levels : Optional[Sequence[int]], default None
        Multi-resolution pyramid downsampling factors or level resolutions.
    pyramid_levels : Optional[Sequence[int]], default None
        Alias for levels.
    epochs_per_level : Optional[Union[int, Sequence[int]]], default None
        Alias for iterations.
    affine_epochs : int, default 0
        Number of epochs for rigid/affine pre-alignment before deformable SyN.
    w_distortion : float, default 0.1
        Weight for area/volume distortion regularisation penalty.
    antisymmetric : bool, default True
        If True, removes common-mode velocity drift to anchor the geodesic midpoint.
    formulation : {'lagrangian', 'eulerian'}, default 'lagrangian'
        Field composition formulation. 'lagrangian' pullback composition prevents grid folding.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate axis mapping convention.
    fill_value : float, default 0.0
        Value for empty voxels during Nadaraya-Watson projection.
    verbose : bool, default False
        If True, prints progress details during registration.
    device : Optional[Union[str, torch.device]], default None
        Target device ('cpu', 'cuda', 'mps'). If None, auto-detected.
    dtype : torch.dtype, default torch.float32
        Precision for tensors and buffers.
    """
    dim: int = 2
    grid_res: Union[int, Tuple[int, ...]] = 128
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0)
    sigma: Union[float, Sequence[float], str] = 0.03
    fluid_sigma: float = 1.5
    elastic_sigma: float = 0.0
    regularizer: Literal['dsti1', 'dsti', 'sobolev', 'gaussian'] = 'dsti1'
    optimizer_type: Literal['rprop', 'cfl', 'adam', 'reg_adam'] = 'rprop'
    optimizer_lr: float = 0.05
    in_loop_inv_steps: int = 5
    inverse_steps: int = 20
    inverse_method: Literal['anderson', 'fixed_point'] = 'anderson'
    cfl_voxels: float = 0.25
    use_analytical_gradients: bool = False
    similarity_metric: Literal['lncc', 'mse'] = 'lncc'
    window_size: int = 15
    iterations: Union[int, Sequence[int]] = 100
    levels: Optional[Sequence[int]] = None
    pyramid_levels: Optional[Sequence[int]] = None
    epochs_per_level: Optional[Union[int, Sequence[int]]] = None
    affine_epochs: int = 0
    w_distortion: float = 0.1
    antisymmetric: bool = True
    formulation: Literal['lagrangian', 'eulerian'] = 'lagrangian'
    coord_convention: Literal['xyz', 'zyx'] = 'xyz'
    fill_value: float = 0.0
    distance_transform_tau: Optional[float] = None
    distance_transform_weight: float = 1.0
    verbose: bool = False
    device: Optional[Union[str, torch.device]] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.pyramid_levels is not None and self.levels is None:
            self.levels = self.pyramid_levels
        if self.epochs_per_level is not None and self.iterations == 100:
            self.iterations = self.epochs_per_level


@dataclass
class ScatteredRegistrationResult:
    """Comprehensive container encapsulating all artifacts from scattered SyN registration.

    Supports both modern attribute access (`result.disp_fwd`, `result.warped_moving_points`)
    and dict-like legacy indexing (`result['warpedmovout']`, `result['fwdtransforms']`)
    for seamless compatibility across downstream consumers.
    """
    disp_fwd: torch.Tensor
    disp_inv: torch.Tensor
    warp_l2r: torch.Tensor
    warp_r2l: torch.Tensor
    warp_l2r_inv: torch.Tensor
    warp_r2l_inv: torch.Tensor
    fixed_grid: torch.Tensor
    moving_grid: torch.Tensor
    warped_moving_grid: torch.Tensor
    warped_fixed_grid: torch.Tensor
    midpoint_fixed_grid: Optional[torch.Tensor] = None
    midpoint_moving_grid: Optional[torch.Tensor] = None
    warped_moving_points: Optional[torch.Tensor] = None
    warped_fixed_points: Optional[torch.Tensor] = None
    warped_moving_features: Optional[torch.Tensor] = None
    warped_fixed_features: Optional[torch.Tensor] = None
    loss_history: List[float] = field(default_factory=list)
    metric_history: Dict[str, List[float]] = field(default_factory=dict)
    inverse_identity_error: float = 0.0
    inverse_identity_errors: Dict[str, float] = field(default_factory=dict)
    grid_folding_percentage: float = 0.0
    jacobian_min: float = 1.0
    jacobian_mean: float = 1.0
    deformation_energies: Dict[str, float] = field(default_factory=dict)
    grid_shape: Tuple[int, ...] = (128, 128)
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0)
    config: Optional[ScatteredRegistrationConfig] = None
    model: Optional[nn.Module] = None

    @property
    def warp_fwd(self) -> torch.Tensor:
        """Alias for disp_fwd."""
        return self.disp_fwd

    @property
    def warp_inv(self) -> torch.Tensor:
        """Alias for disp_inv."""
        return self.disp_inv

    @property
    def folding_percentage(self) -> float:
        """Alias for grid_folding_percentage."""
        return self.grid_folding_percentage

    @property
    def inverse_consistency_inf(self) -> float:
        """Alias for inverse_identity_error."""
        return self.inverse_identity_error

    def warp_points(
        self,
        coords: torch.Tensor,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
    ) -> torch.Tensor:
        """Differentiably warp arbitrary scattered coordinates through the displacement fields."""
        # In SyN, disp_inv maps Moving -> Fixed ('forward'), disp_fwd maps Fixed -> Moving ('inverse')
        field = self.disp_inv if direction == 'forward' else self.disp_fwd
        return warp_scattered_coordinates(
            coords=coords,
            displacement_field=field,
            direction='forward',
            domain_bounds=self.domain_bounds,
            coord_convention=self.config.coord_convention if self.config else 'xyz',
        )

    def transport_features(
        self,
        coords_src: torch.Tensor,
        features_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
    ) -> torch.Tensor:
        """Transport features between scattered coordinate sets."""
        field = self.disp_inv if direction == 'forward' else self.disp_fwd
        return transport_scattered_to_scattered(
            coords_source=coords_src,
            features_source=features_src,
            coords_target=coords_tgt,
            displacement_field=field,
            domain_bounds=self.domain_bounds,
            coord_convention=self.config.coord_convention if self.config else 'xyz',
        )

    def pullback_grid(
        self,
        grid_features: torch.Tensor,
        coords: torch.Tensor,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
    ) -> torch.Tensor:
        """Pull back Eulerian grid features to scattered coordinates."""
        field = self.disp_fwd if direction == 'forward' else self.disp_inv
        return pullback_grid_to_scattered(
            grid_features=grid_features,
            coords=coords,
            displacement_field=field,
            domain_bounds=self.domain_bounds,
            coord_convention=self.config.coord_convention if self.config else 'xyz',
        )

    def pushforward_features(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
        grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Push forward scattered features onto a regular Eulerian grid."""
        field = self.disp_fwd if direction == 'forward' else self.disp_inv
        return pushforward_scattered_to_grid(
            coords=coords,
            features=features,
            displacement_field=field,
            grid_shape=grid_shape or self.grid_shape,
            domain_bounds=self.domain_bounds,
            coord_convention=self.config.coord_convention if self.config else 'xyz',
        )

    def __contains__(self, key: Any) -> bool:
        """Support 'key in result' syntax for dict-like compatibility."""
        if not isinstance(key, str):
            return False
        keys = {
            'warpedmovout', 'warpedfixout', 'fwdtransforms', 'invtransforms',
            'fwd_warp', 'inv_warp', 'syn_losses', 'loss_history',
            'inverse_identity_errors', 'model'
        }
        return key in keys or hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility with syntx.syn() and ANTs-style returns."""
        if not isinstance(key, str):
            raise KeyError(key)
        mapping = {
            'warpedmovout': self.warped_moving_points if self.warped_moving_points is not None else self.warped_moving_grid,
            'warpedfixout': self.warped_fixed_points if self.warped_fixed_points is not None else self.warped_fixed_grid,
            'fwdtransforms': [self.disp_fwd],
            'invtransforms': [self.disp_inv],
            'fwd_warp': self.disp_fwd,
            'inv_warp': self.disp_inv,
            'syn_losses': self.loss_history,
            'loss_history': self.loss_history,
            'inverse_identity_errors': self.inverse_identity_errors,
            'model': self.model,
        }
        if key in mapping:
            return mapping[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in ScatteredRegistrationResult")


def _make_identity_grid(spatial_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Constructs a normalized Eulerian identity coordinate grid in [-1, 1]^d."""
    grids = [torch.linspace(-1, 1, s, dtype=dtype, device=device) for s in spatial_shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    # In Cartesian 'xyz' convention, the coordinate dimensions are reversed (x, y, [z])
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
    return identity


def _compute_grid_folding(warp_field: torch.Tensor, domain_mask: Optional[torch.Tensor] = None) -> Tuple[float, float, float]:
    """Calculates grid folding percentage, min det(J), and mean det(J)."""
    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype

    direction = torch.eye(dim, device=device, dtype=dtype)
    spacing = torch.tensor([2.0 / (s - 1) for s in spatial], device=device, dtype=dtype)

    jac = compute_physical_jacobian_determinant(warp_field, direction=direction, spacing=spacing)

    if domain_mask is not None:
        mask = domain_mask.squeeze(0).squeeze(0) if domain_mask.dim() > len(spatial) else domain_mask
        active_jac = jac.squeeze(0)[mask > 0.5]
    else:
        slices = [slice(1, s - 1) for s in spatial]
        active_jac = jac.squeeze(0)[tuple(slices)]

    if active_jac.numel() == 0:
        return 0.0, 1.0, 1.0

    folding_pct = float((active_jac <= 0.0).float().mean().item()) * 100.0
    min_jac = float(active_jac.min().item())
    mean_jac = float(active_jac.mean().item())
    return folding_pct, min_jac, mean_jac


class SyNScattered(nn.Module):
    """Symmetric Diffeomorphic Normalization (SyN) Registration Solver for Scattered Data.

    Parameterizes symmetric diffeomorphic deformations between scattered point clouds
    (or scattered points against a reference Eulerian grid) through continuous velocity
    fields regularized by fluid Green operators and accelerated by in-loop Anderson inversion.
    """
    def __init__(
        self,
        dim: int = 2,
        grid_res: Union[int, Tuple[int, ...]] = 128,
        domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
        sigma: Union[float, Sequence[float], str] = 0.03,
        fluid_sigma: float = 1.5,
        elastic_sigma: float = 0.0,
        regularizer: Literal['dsti1', 'dsti', 'sobolev', 'gaussian'] = 'dsti1',
        optimizer_type: Literal['rprop', 'cfl', 'adam', 'reg_adam'] = 'rprop',
        in_loop_inv_steps: int = 5,
        inverse_steps: int = 20,
        inverse_method: Literal['anderson', 'fixed_point'] = 'anderson',
        cfl_voxels: float = 0.25,
        use_analytical_gradients: bool = False,
        config: Optional[ScatteredRegistrationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = ScatteredRegistrationConfig(
                dim=dim,
                grid_res=grid_res,
                domain_bounds=domain_bounds,
                sigma=sigma,
                fluid_sigma=fluid_sigma,
                elastic_sigma=elastic_sigma,
                regularizer=regularizer,
                optimizer_type=optimizer_type,
                in_loop_inv_steps=in_loop_inv_steps,
                inverse_steps=inverse_steps,
                inverse_method=inverse_method,
                cfl_voxels=cfl_voxels,
                use_analytical_gradients=use_analytical_gradients,
                **kwargs,
            )
        self.config = config
        self.dim = config.dim

        if isinstance(config.grid_res, int):
            self.spatial_shape = (config.grid_res,) * config.dim
        else:
            self.spatial_shape = tuple(config.grid_res)

        # Persistent=False buffers for SyN half-warps and total displacement fields
        self.register_buffer('warp_l2r', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('warp_r2l', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('warp_l2r_inv', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('warp_r2l_inv', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('disp_fwd', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('disp_inv', torch.zeros(1, *self.spatial_shape, config.dim, dtype=config.dtype), persistent=False)
        self.register_buffer('identity', _make_identity_grid(self.spatial_shape, dtype=config.dtype), persistent=False)

        self.loss_history: List[float] = []

    def _optimize_affine_prealignment(
        self,
        I_fixed: torch.Tensor,
        J_moving: torch.Tensor,
        epochs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Performs rigid/affine pre-alignment minimizing LNCC loss before deformable SyN."""
        dim = self.dim
        spatial = I_fixed.shape[2:]

        if dim == 2:
            angle = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            translation = nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
            optimizer = torch.optim.Adam([angle, translation], lr=0.05)

            for _ in range(epochs):
                optimizer.zero_grad()
                c, s = torch.cos(angle).squeeze(), torch.sin(angle).squeeze()
                tx, ty = translation[0].squeeze(), translation[1].squeeze()
                row0 = torch.stack([c, -s, tx])
                row1 = torch.stack([s, c, ty])
                rot_mat = torch.stack([row0, row1]).unsqueeze(0)
                grid_aff = F.affine_grid(rot_mat, J_moving.shape, align_corners=True)
                J_warped = F.grid_sample(J_moving, grid_aff, padding_mode='border', align_corners=True)
                loss = local_ncc_loss_nd(I_fixed, J_warped, window_size=min(self.config.window_size, min(spatial) - 1))
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                c, s = torch.cos(angle).squeeze(), torch.sin(angle).squeeze()
                tx, ty = translation[0].squeeze(), translation[1].squeeze()
                row0 = torch.stack([c, -s, tx])
                row1 = torch.stack([s, c, ty])
                rot_mat = torch.stack([row0, row1]).unsqueeze(0)
                grid_aff = F.affine_grid(rot_mat, J_moving.shape, align_corners=True)
                identity = _make_identity_grid(spatial, dtype=dtype, device=device)
                w_aff = grid_aff - identity
                return w_aff
        else:
            # 3D affine matrix parameterization
            theta = nn.Parameter(torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0))
            optimizer = torch.optim.Adam([theta], lr=0.05)

            for _ in range(epochs):
                optimizer.zero_grad()
                grid_aff = F.affine_grid(theta, J_moving.shape, align_corners=True)
                J_warped = F.grid_sample(J_moving, grid_aff, padding_mode='border', align_corners=True)
                loss = local_ncc_loss_nd(I_fixed, J_warped, window_size=min(self.config.window_size, min(spatial) - 1))
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                grid_aff = F.affine_grid(theta, J_moving.shape, align_corners=True)
                identity = _make_identity_grid(spatial, dtype=dtype, device=device)
                w_aff = grid_aff - identity
                return w_aff

    def step(
        self,
        fixed_points: Optional[torch.Tensor] = None,
        fixed_features: Optional[torch.Tensor] = None,
        moving_points: Optional[torch.Tensor] = None,
        moving_features: Optional[torch.Tensor] = None,
        fixed_grid: Optional[torch.Tensor] = None,
        moving_grid: Optional[torch.Tensor] = None,
        domain_mask: Optional[torch.Tensor] = None,
        point_weights_fixed: Optional[torch.Tensor] = None,
        point_weights_moving: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes a single differentiable SyN similarity loss step.

        Preserves the PyTorch autograd computation graph back to input coordinates and features.
        """
        device = self.identity.device
        dtype = self.identity.dtype
        dim = self.dim

        # Project or standardize fixed image
        if fixed_points is not None:
            pts_f = fixed_points if fixed_points.dim() == 2 else fixed_points.squeeze(0)
            fts_f = fixed_features.unsqueeze(-1) if (fixed_features is not None and fixed_features.dim() == 1) else fixed_features
            I_fixed = project_scattered_to_grid(
                pts_f, fts_f,
                grid_shape=self.spatial_shape,
                domain_bounds=self.config.domain_bounds,
                sigma=self.config.sigma,
                point_weights=point_weights_fixed,
                coord_convention=self.config.coord_convention,
                fill_value=self.config.fill_value,
            )
            if I_fixed.dim() == dim:
                I_fixed = I_fixed.unsqueeze(0).unsqueeze(0)
            elif I_fixed.dim() == dim + 1:
                I_fixed = I_fixed.unsqueeze(0)
            if self.config.distance_transform_tau is not None:
                dt_f = compute_distance_transform_to_grid(
                    pts_f,
                    grid_shape=self.spatial_shape,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                    potential_tau=self.config.distance_transform_tau,
                    device=device,
                    dtype=dtype,
                ) * float(self.config.distance_transform_weight)
                I_fixed = torch.cat([I_fixed, dt_f], dim=1)
        else:
            grid_in = fixed_grid if fixed_grid is not None else fixed_features
            I_fixed, _, _ = _standardize_grid_features(grid_in, dim, 1)

        # Project or standardize moving image
        if moving_points is not None:
            pts_m = moving_points if moving_points.dim() == 2 else moving_points.squeeze(0)
            fts_m = moving_features.unsqueeze(-1) if (moving_features is not None and moving_features.dim() == 1) else moving_features
            J_moving = project_scattered_to_grid(
                pts_m, fts_m,
                grid_shape=self.spatial_shape,
                domain_bounds=self.config.domain_bounds,
                sigma=self.config.sigma,
                point_weights=point_weights_moving,
                coord_convention=self.config.coord_convention,
                fill_value=self.config.fill_value,
            )
            if J_moving.dim() == dim:
                J_moving = J_moving.unsqueeze(0).unsqueeze(0)
            elif J_moving.dim() == dim + 1:
                J_moving = J_moving.unsqueeze(0)
            if self.config.distance_transform_tau is not None:
                dt_m = compute_distance_transform_to_grid(
                    pts_m,
                    grid_shape=self.spatial_shape,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                    potential_tau=self.config.distance_transform_tau,
                    device=device,
                    dtype=dtype,
                ) * float(self.config.distance_transform_weight)
                J_moving = torch.cat([J_moving, dt_m], dim=1)
        else:
            grid_in = moving_grid if moving_grid is not None else moving_features
            J_moving, _, _ = _standardize_grid_features(grid_in, dim, 1)

        phi_l = (self.identity + self.warp_l2r).to(dtype=I_fixed.dtype)
        phi_r = (self.identity + self.warp_r2l).to(dtype=J_moving.dtype)

        I_mid = F.grid_sample(I_fixed, phi_l, padding_mode='border', align_corners=True)
        J_mid = F.grid_sample(J_moving, phi_r, padding_mode='border', align_corners=True)

        if self.config.similarity_metric == 'mse':
            loss = F.mse_loss(I_mid, J_mid)
        elif self.config.similarity_metric in ('dt', 'distance_transform', 'edt'):
            from syntx.core.losses import distance_transform_loss
            loss = distance_transform_loss(
                I_mid, J_mid,
                mode='potential_lncc',
                tau=self.config.distance_transform_tau or 0.10,
                window_size=self.config.window_size,
                mask=domain_mask,
            )
        else:
            loss = local_ncc_loss_nd(
                I_mid, J_mid,
                mask=domain_mask,
                window_size=self.config.window_size,
                use_ants_pseudo_gradient=False,
            )
        return loss

    def fit(
        self,
        fixed_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fixed_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        moving_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
        moving_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fixed_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
        moving_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
        domain_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        point_weights_fixed: Optional[Union[torch.Tensor, np.ndarray]] = None,
        point_weights_moving: Optional[Union[torch.Tensor, np.ndarray]] = None,
        epochs: Optional[int] = None,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> ScatteredRegistrationResult:
        """Executes the multi-resolution or single-resolution SyN registration loop.

        Supports scattered-to-scattered (point-to-point) and scattered-to-grid (mixed) modes.
        """
        # 1. Validate inputs
        if fixed_points is not None and (hasattr(fixed_points, '__len__') and len(fixed_points) == 0):
            raise ValueError("Fixed points tensor cannot be empty (N=0).")
        if moving_points is not None and (hasattr(moving_points, '__len__') and len(moving_points) == 0):
            raise ValueError("Moving points tensor cannot be empty (N=0).")
        if fixed_points is None and fixed_grid is None and fixed_features is None:
            raise ValueError("Must provide either fixed_points or fixed_grid / fixed_features.")
        if moving_points is None and moving_grid is None and moving_features is None:
            raise ValueError("Must provide either moving_points or moving_grid / moving_features.")

        # 2. Determine device and dtype
        device = self.config.device
        if device is None:
            if fixed_points is not None and isinstance(fixed_points, torch.Tensor):
                device = fixed_points.device
            elif moving_points is not None and isinstance(moving_points, torch.Tensor):
                device = moving_points.device
            elif fixed_grid is not None and isinstance(fixed_grid, torch.Tensor):
                device = fixed_grid.device
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)

        dtype = self.config.dtype
        if fixed_points is not None and isinstance(fixed_points, torch.Tensor) and fixed_points.dtype == torch.float64:
            dtype = torch.float64
        elif moving_points is not None and isinstance(moving_points, torch.Tensor) and moving_points.dtype == torch.float64:
            dtype = torch.float64

        dim = self.dim
        self.to(device=device, dtype=dtype)

        # Standardize point cloud and feature tensors
        has_scattered_fixed = fixed_points is not None
        has_scattered_moving = moving_points is not None

        pts_f: Optional[torch.Tensor] = None
        fts_f: Optional[torch.Tensor] = None
        pts_m: Optional[torch.Tensor] = None
        fts_m: Optional[torch.Tensor] = None

        if has_scattered_fixed:
            pts_f = torch.as_tensor(fixed_points, device=device, dtype=dtype).detach()
            if pts_f.dim() == 3 and pts_f.shape[0] == 1:
                pts_f = pts_f.squeeze(0)
            fts_f = torch.as_tensor(fixed_features if fixed_features is not None else torch.ones((pts_f.shape[0], 1)), device=device, dtype=dtype).detach()
            if fts_f.dim() == 1:
                fts_f = fts_f.unsqueeze(-1)
            elif fts_f.dim() == 3 and fts_f.shape[0] == 1:
                fts_f = fts_f.squeeze(0)

        if has_scattered_moving:
            pts_m = torch.as_tensor(moving_points, device=device, dtype=dtype).detach()
            if pts_m.dim() == 3 and pts_m.shape[0] == 1:
                pts_m = pts_m.squeeze(0)
            fts_m = torch.as_tensor(moving_features if moving_features is not None else torch.ones((pts_m.shape[0], 1)), device=device, dtype=dtype).detach()
            if fts_m.dim() == 1:
                fts_m = fts_m.unsqueeze(-1)
            elif fts_m.dim() == 3 and fts_m.shape[0] == 1:
                fts_m = fts_m.squeeze(0)

        w_f = torch.as_tensor(point_weights_fixed, device=device, dtype=dtype).detach() if point_weights_fixed is not None else None
        w_m = torch.as_tensor(point_weights_moving, device=device, dtype=dtype).detach() if point_weights_moving is not None else None

        # Standardize Eulerian grid inputs
        I_fixed_full: torch.Tensor
        J_moving_full: torch.Tensor

        if has_scattered_fixed:
            I_fixed_full = project_scattered_to_grid(
                pts_f, fts_f,
                grid_shape=self.spatial_shape,
                domain_bounds=self.config.domain_bounds,
                sigma=self.config.sigma,
                point_weights=w_f,
                coord_convention=self.config.coord_convention,
                fill_value=self.config.fill_value,
            ).detach()
            if I_fixed_full.dim() == dim:
                I_fixed_full = I_fixed_full.unsqueeze(0).unsqueeze(0)
            elif I_fixed_full.dim() == dim + 1:
                I_fixed_full = I_fixed_full.unsqueeze(0)
            if self.config.distance_transform_tau is not None:
                dt_f = compute_distance_transform_to_grid(
                    pts_f,
                    grid_shape=self.spatial_shape,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                    potential_tau=self.config.distance_transform_tau,
                    device=device,
                    dtype=dtype,
                ).detach() * float(self.config.distance_transform_weight)
                I_fixed_full = torch.cat([I_fixed_full, dt_f], dim=1)
        else:
            raw_fixed = fixed_grid if fixed_grid is not None else fixed_features
            raw_fixed_t = torch.as_tensor(raw_fixed, device=device, dtype=dtype).detach()
            I_fixed_full, _, _ = _standardize_grid_features(raw_fixed_t, dim, 1)

        if has_scattered_moving:
            J_moving_full = project_scattered_to_grid(
                pts_m, fts_m,
                grid_shape=self.spatial_shape,
                domain_bounds=self.config.domain_bounds,
                sigma=self.config.sigma,
                point_weights=w_m,
                coord_convention=self.config.coord_convention,
                fill_value=self.config.fill_value,
            ).detach()
            if J_moving_full.dim() == dim:
                J_moving_full = J_moving_full.unsqueeze(0).unsqueeze(0)
            elif J_moving_full.dim() == dim + 1:
                J_moving_full = J_moving_full.unsqueeze(0)
            if self.config.distance_transform_tau is not None:
                dt_m = compute_distance_transform_to_grid(
                    pts_m,
                    grid_shape=self.spatial_shape,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                    potential_tau=self.config.distance_transform_tau,
                    device=device,
                    dtype=dtype,
                ).detach() * float(self.config.distance_transform_weight)
                J_moving_full = torch.cat([J_moving_full, dt_m], dim=1)
        else:
            raw_moving = moving_grid if moving_grid is not None else moving_features
            raw_moving_t = torch.as_tensor(raw_moving, device=device, dtype=dtype).detach()
            J_moving_full, _, _ = _standardize_grid_features(raw_moving_t, dim, 1)

        # 3. Multi-resolution pyramid resolution schedule
        levels = self.config.levels
        if levels is None:
            levels = [1]
        levels = list(levels)

        # Interpret whether levels are downsampling factors or direct resolutions
        pyramid_shapes: List[Tuple[int, ...]] = []
        is_factor = (levels[0] >= 1 and (len(levels) == 1 or levels[0] >= levels[-1])) and (levels[0] <= 16)
        for s in levels:
            if is_factor:
                factor = float(s)
                shape = tuple(max(4, int(round(g / factor))) for g in self.spatial_shape)
            else:
                shape = (int(s),) * dim if isinstance(s, (int, float)) else tuple(int(x) for x in s)
            pyramid_shapes.append(shape)

        # Iterations schedule per level
        if epochs is not None:
            epochs_list = [epochs] * len(pyramid_shapes)
        elif self.config.epochs_per_level is not None:
            epl = self.config.epochs_per_level
            epochs_list = list(epl) if isinstance(epl, (list, tuple)) else [epl] * len(pyramid_shapes)
        elif isinstance(self.config.iterations, (list, tuple)):
            epochs_list = list(self.config.iterations)
        else:
            epochs_list = [self.config.iterations] * len(pyramid_shapes)

        if len(epochs_list) < len(pyramid_shapes):
            epochs_list = epochs_list + [epochs_list[-1]] * (len(pyramid_shapes) - len(epochs_list))
        elif len(epochs_list) > len(pyramid_shapes) and len(pyramid_shapes) == 1:
            epochs_list = [epochs_list[0]]

        # Initialize half-warps at the first level
        init_shape = pyramid_shapes[0]
        self.warp_l2r = torch.zeros(1, *init_shape, dim, device=device, dtype=dtype)
        self.warp_r2l = torch.zeros(1, *init_shape, dim, device=device, dtype=dtype)
        self.warp_l2r_inv = torch.zeros(1, *init_shape, dim, device=device, dtype=dtype)
        self.warp_r2l_inv = torch.zeros(1, *init_shape, dim, device=device, dtype=dtype)

        # Optional rigid/affine pre-alignment
        if self.config.affine_epochs > 0:
            interp_mode = 'bilinear' if dim == 2 else 'trilinear'
            I_f_init = F.interpolate(I_fixed_full, size=init_shape, mode=interp_mode, align_corners=True)
            J_m_init = F.interpolate(J_moving_full, size=init_shape, mode=interp_mode, align_corners=True)
            w_aff = self._optimize_affine_prealignment(I_f_init, J_m_init, self.config.affine_epochs, device, dtype)
            self.warp_r2l.copy_(w_aff)
            self.warp_r2l_inv = update_inverse_field_nd_anderson(self.warp_r2l, None, steps=15, m=5)

        self.loss_history = []
        interp_mode = 'bilinear' if dim == 2 else 'trilinear'

        # 4. Multi-Resolution SyN Iteration Loop
        for level_idx, (curr_shape, n_epochs) in enumerate(zip(pyramid_shapes, epochs_list)):
            h_voxel = torch.tensor([2.0 / (s - 1) for s in curr_shape], device=device, dtype=dtype)
            spacing_t = h_voxel.view(*([1] * (dim + 1)), dim)

            # Upsample half-warps when transitioning between pyramid levels
            if level_idx > 0:
                self.warp_l2r = F.interpolate(
                    self.warp_l2r.movedim(-1, 1), size=curr_shape, mode=interp_mode, align_corners=True
                ).movedim(1, -1).contiguous()
                self.warp_r2l = F.interpolate(
                    self.warp_r2l.movedim(-1, 1), size=curr_shape, mode=interp_mode, align_corners=True
                ).movedim(1, -1).contiguous()
                self.warp_l2r_inv = F.interpolate(
                    self.warp_l2r_inv.movedim(-1, 1), size=curr_shape, mode=interp_mode, align_corners=True
                ).movedim(1, -1).contiguous()
                self.warp_r2l_inv = F.interpolate(
                    self.warp_r2l_inv.movedim(-1, 1), size=curr_shape, mode=interp_mode, align_corners=True
                ).movedim(1, -1).contiguous()

            # Level-adaptive projection bandwidth: prevents coarse holes
            max_h = float(h_voxel.max().item())
            if isinstance(self.config.sigma, (int, float)):
                sigma_eff = max(float(self.config.sigma), 1.5 * max_h)
            else:
                sigma_eff = self.config.sigma

            # Project or interpolate images for current resolution
            if has_scattered_fixed:
                I_curr = project_scattered_to_grid(
                    pts_f, fts_f,
                    grid_shape=curr_shape,
                    domain_bounds=self.config.domain_bounds,
                    sigma=sigma_eff,
                    point_weights=w_f,
                    coord_convention=self.config.coord_convention,
                    fill_value=self.config.fill_value,
                ).detach()
                if I_curr.dim() == dim:
                    I_curr = I_curr.unsqueeze(0).unsqueeze(0)
                elif I_curr.dim() == dim + 1:
                    I_curr = I_curr.unsqueeze(0)
                if self.config.distance_transform_tau is not None:
                    dt_curr_f = compute_distance_transform_to_grid(
                        pts_f,
                        grid_shape=curr_shape,
                        domain_bounds=self.config.domain_bounds,
                        coord_convention=self.config.coord_convention,
                        potential_tau=self.config.distance_transform_tau,
                        device=device,
                        dtype=dtype,
                    ).detach() * float(self.config.distance_transform_weight)
                    I_curr = torch.cat([I_curr, dt_curr_f], dim=1)
            else:
                I_curr = F.interpolate(I_fixed_full, size=curr_shape, mode=interp_mode, align_corners=True).detach()

            if has_scattered_moving:
                J_curr = project_scattered_to_grid(
                    pts_m, fts_m,
                    grid_shape=curr_shape,
                    domain_bounds=self.config.domain_bounds,
                    sigma=sigma_eff,
                    point_weights=w_m,
                    coord_convention=self.config.coord_convention,
                    fill_value=self.config.fill_value,
                ).detach()
                if J_curr.dim() == dim:
                    J_curr = J_curr.unsqueeze(0).unsqueeze(0)
                elif J_curr.dim() == dim + 1:
                    J_curr = J_curr.unsqueeze(0)
                if self.config.distance_transform_tau is not None:
                    dt_curr_m = compute_distance_transform_to_grid(
                        pts_m,
                        grid_shape=curr_shape,
                        domain_bounds=self.config.domain_bounds,
                        coord_convention=self.config.coord_convention,
                        potential_tau=self.config.distance_transform_tau,
                        device=device,
                        dtype=dtype,
                    ).detach() * float(self.config.distance_transform_weight)
                    J_curr = torch.cat([J_curr, dt_curr_m], dim=1)
            else:
                J_curr = F.interpolate(J_moving_full, size=curr_shape, mode=interp_mode, align_corners=True).detach()

            level_identity = _make_identity_grid(curr_shape, dtype=dtype, device=device)
            b_mask = get_boundary_mask(curr_shape, device, dtype)

            # CFL scaling with shrink ratio
            shrink_ratio = float(min(curr_shape)) / float(min(self.spatial_shape))
            level_cfl = float(self.config.cfl_voxels) * math.sqrt(shrink_ratio)

            # Rprop state buffers
            rprop_step_l = torch.ones_like(self.warp_l2r) * self.config.optimizer_lr
            rprop_step_r = torch.ones_like(self.warp_r2l) * self.config.optimizer_lr
            rprop_prev_grad_l = torch.zeros_like(self.warp_l2r)
            rprop_prev_grad_r = torch.zeros_like(self.warp_r2l)

            # Pre-interpolate domain mask if provided
            level_mask = None
            if domain_mask is not None:
                mask_t = torch.as_tensor(domain_mask, device=device, dtype=dtype).detach()
                while mask_t.dim() < dim + 2:
                    mask_t = mask_t.unsqueeze(0)
                level_mask = F.interpolate(mask_t, size=curr_shape, mode='nearest')

            # Per-epoch SyN optimization loop
            for epoch in range(n_epochs):
                w_l = self.warp_l2r.detach().clone().requires_grad_(True)
                w_r = self.warp_r2l.detach().clone().requires_grad_(True)

                phi_l = level_identity + w_l
                phi_r = level_identity + w_r

                I_mid = F.grid_sample(I_curr, phi_l, padding_mode='border', align_corners=True)
                J_mid = F.grid_sample(J_curr, phi_r, padding_mode='border', align_corners=True)

                # Similarity loss
                if self.config.similarity_metric == 'mse':
                    loss = F.mse_loss(I_mid, J_mid)
                elif self.config.similarity_metric in ('dt', 'distance_transform', 'edt'):
                    from syntx.core.losses import distance_transform_loss
                    loss = distance_transform_loss(
                        I_mid, J_mid,
                        mode='potential_lncc',
                        tau=self.config.distance_transform_tau or 0.10,
                        window_size=self.config.window_size,
                        mask=level_mask,
                    )
                else:
                    loss = local_ncc_loss_nd(
                        I_mid, J_mid,
                        mask=level_mask,
                        window_size=self.config.window_size,
                        use_ants_pseudo_gradient=self.config.use_analytical_gradients,
                    )

                self.loss_history.append(float(loss.item()))

                loss.backward()
                grad_l = w_l.grad.detach()
                grad_r = w_r.grad.detach()

                with torch.no_grad():
                    # Fluid velocity regularization
                    reg = self.config.regularizer
                    fluid_sig = self.config.fluid_sigma
                    if reg in ('dsti1', 'dsti'):
                        v_l = apply_dsti1_green_operator(grad_l * b_mask, fluid_sigma=fluid_sig)
                        v_r = apply_dsti1_green_operator(grad_r * b_mask, fluid_sigma=fluid_sig)
                    elif reg == 'sobolev':
                        v_l = apply_sobolev_green_operator(grad_l * b_mask, fluid_sigma=fluid_sig)
                        v_r = apply_sobolev_green_operator(grad_r * b_mask, fluid_sigma=fluid_sig)
                    else:
                        v_l = separable_gaussian_filter(grad_l * b_mask, sigma=fluid_sig)
                        v_r = separable_gaussian_filter(grad_r * b_mask, sigma=fluid_sig)

                    # Optimizer step direction & magnitude
                    opt_type = self.config.optimizer_type
                    if opt_type == 'rprop':
                        def rprop_update(g, prev_g, st):
                            sign = g * prev_g
                            st_inc = torch.clamp(st * 1.2, max=0.2)
                            st_dec = torch.clamp(st * 0.5, min=1e-6)
                            st_new = torch.where(sign > 0, st_inc, torch.where(sign < 0, st_dec, st))
                            active = g.abs() > 1e-5
                            d = torch.where(active & (sign >= 0), torch.sign(g) * st_new, torch.zeros_like(g))
                            g_new = torch.where(sign < 0, torch.zeros_like(g), g)
                            return d, st_new, g_new

                        delta_l, rprop_step_l, rprop_prev_grad_l = rprop_update(v_l, rprop_prev_grad_l, rprop_step_l)
                        delta_r, rprop_step_r, rprop_prev_grad_r = rprop_update(v_r, rprop_prev_grad_r, rprop_step_r)

                        # Step scaling in Point-to-Grid mode to prevent boundary folding
                        if has_scattered_fixed != has_scattered_moving:
                            v_norm_l = torch.sqrt(torch.sum(v_l**2, dim=-1, keepdim=True))
                            v_norm_r = torch.sqrt(torch.sum(v_r**2, dim=-1, keepdim=True))
                            max_v = max(v_norm_l.max().item(), v_norm_r.max().item(), 1e-8)
                            scale_l = torch.clamp(v_norm_l / (max_v * 0.25), max=1.0)
                            scale_r = torch.clamp(v_norm_r / (max_v * 0.25), max=1.0)
                            delta_l = delta_l * scale_l
                            delta_r = delta_r * scale_r

                        delta_l = separable_gaussian_filter(delta_l * b_mask, sigma=max(0.5, fluid_sig * 0.5)) * b_mask
                        delta_r = separable_gaussian_filter(delta_r * b_mask, sigma=max(0.5, fluid_sig * 0.5)) * b_mask
                    elif opt_type == 'cfl':
                        norm_l = torch.sqrt(torch.sum((v_l / spacing_t)**2, dim=-1)).max()
                        norm_r = torch.sqrt(torch.sum((v_r / spacing_t)**2, dim=-1)).max()
                        max_norm = torch.max(norm_l, norm_r).clamp_min(1e-4)
                        effective_step = level_cfl * self.config.optimizer_lr
                        if float(max_norm) > 1e-4:
                            delta_l = (effective_step / max_norm) * v_l
                            delta_r = (effective_step / max_norm) * v_r
                        else:
                            delta_l = torch.zeros_like(v_l)
                            delta_r = torch.zeros_like(v_r)
                    else:
                        delta_l = self.config.optimizer_lr * v_l
                        delta_r = self.config.optimizer_lr * v_r

                    # Antisymmetric geodesic velocity projection
                    if self.config.antisymmetric:
                        drift = 0.5 * (delta_l + delta_r)
                        delta_l = delta_l - drift
                        delta_r = delta_r - drift

                    # CFL step bounding: max_x ||delta||_voxel <= level_cfl
                    disp_vox_l = torch.sqrt(torch.sum((delta_l / spacing_t)**2, dim=-1)).max().item()
                    disp_vox_r = torch.sqrt(torch.sum((delta_r / spacing_t)**2, dim=-1)).max().item()
                    max_disp_voxels = max(disp_vox_l, disp_vox_r, 1e-8)

                    if max_disp_voxels > level_cfl:
                        cfl_scale = level_cfl / max_disp_voxels
                        delta_l = delta_l * cfl_scale
                        delta_r = delta_r * cfl_scale

                    # Lagrangian pullback step composition
                    if self.config.formulation == 'lagrangian':
                        delta_l_pb = F.grid_sample(
                            delta_l.movedim(-1, 1).contiguous(),
                            (level_identity + self.warp_l2r).contiguous(),
                            padding_mode='border',
                            align_corners=True
                        ).movedim(1, -1).contiguous()
                        delta_r_pb = F.grid_sample(
                            delta_r.movedim(-1, 1).contiguous(),
                            (level_identity + self.warp_r2l).contiguous(),
                            padding_mode='border',
                            align_corners=True
                        ).movedim(1, -1).contiguous()

                        self.warp_l2r.sub_(delta_l_pb)
                        self.warp_r2l.sub_(delta_r_pb)
                    else:
                        self.warp_l2r.sub_(delta_l)
                        self.warp_r2l.sub_(delta_r)

                    # Optional elastic smoothing
                    if self.config.elastic_sigma > 0.0:
                        self.warp_l2r.copy_(separable_gaussian_filter(self.warp_l2r, sigma=self.config.elastic_sigma) * b_mask)
                        self.warp_r2l.copy_(separable_gaussian_filter(self.warp_r2l, sigma=self.config.elastic_sigma) * b_mask)

                    # In-loop Anderson acceleration
                    if self.config.in_loop_inv_steps > 0:
                        self.warp_l2r_inv = update_inverse_field_nd_anderson(
                            self.warp_l2r.detach(), self.warp_l2r_inv.detach(),
                            steps=self.config.in_loop_inv_steps, m=5,
                            max_error_threshold=0.05, mean_error_threshold=0.001
                        )
                        self.warp_r2l_inv = update_inverse_field_nd_anderson(
                            self.warp_r2l.detach(), self.warp_r2l_inv.detach(),
                            steps=self.config.in_loop_inv_steps, m=5,
                            max_error_threshold=0.05, mean_error_threshold=0.001
                        )

        # 5. Bring half-warps to full resolution if needed
        if self.warp_l2r.shape[1:-1] != self.spatial_shape:
            self.warp_l2r = F.interpolate(
                self.warp_l2r.movedim(-1, 1), size=self.spatial_shape, mode=interp_mode, align_corners=True
            ).movedim(1, -1).contiguous()
            self.warp_r2l = F.interpolate(
                self.warp_r2l.movedim(-1, 1), size=self.spatial_shape, mode=interp_mode, align_corners=True
            ).movedim(1, -1).contiguous()
            self.warp_l2r_inv = F.interpolate(
                self.warp_l2r_inv.movedim(-1, 1), size=self.spatial_shape, mode=interp_mode, align_corners=True
            ).movedim(1, -1).contiguous()
            self.warp_r2l_inv = F.interpolate(
                self.warp_r2l_inv.movedim(-1, 1), size=self.spatial_shape, mode=interp_mode, align_corners=True
            ).movedim(1, -1).contiguous()

        # Final high-accuracy Anderson inversion refinement
        if self.config.inverse_steps > 0:
            inv_steps = max(25, self.config.inverse_steps)
            self.warp_l2r_inv = update_inverse_field_nd_anderson(
                self.warp_l2r, self.warp_l2r_inv,
                steps=inv_steps, m=5,
                max_error_threshold=1e-4, mean_error_threshold=1e-5
            )
            self.warp_r2l_inv = update_inverse_field_nd_anderson(
                self.warp_r2l, self.warp_r2l_inv,
                steps=inv_steps, m=5,
                max_error_threshold=1e-4, mean_error_threshold=1e-5
            )

        # 6. Compose Total Diffeomorphism Fields
        # Fixed (I) -> Moving (J) is phi_r o phi_l^-1:
        #   x in Fixed -> m = x + W_l2r_inv(x) in Midpoint
        #   m -> y = m + W_r2l(m) in Moving
        #   u_fwd(x) = W_l2r_inv(x) + W_r2l(x + W_l2r_inv(x))
        #
        # Moving (J) -> Fixed (I) is phi_l o phi_r^-1:
        #   y in Moving -> m = y + W_r2l_inv(y) in Midpoint
        #   m -> x = m + W_l2r(m) in Fixed
        #   u_inv(y) = W_r2l_inv(y) + W_l2r(y + W_r2l_inv(y))
        full_identity = _make_identity_grid(self.spatial_shape, dtype=dtype, device=device)

        phi_l_full = full_identity + self.warp_l2r
        phi_r_full = full_identity + self.warp_r2l
        phi_l_inv_full = full_identity + self.warp_l2r_inv
        phi_r_inv_full = full_identity + self.warp_r2l_inv

        w_r2l_sampled = F.grid_sample(
            self.warp_r2l.movedim(-1, 1), phi_l_inv_full, padding_mode='border', align_corners=True
        ).movedim(1, -1)
        u_fwd = self.warp_l2r_inv + w_r2l_sampled

        w_l2r_sampled = F.grid_sample(
            self.warp_l2r.movedim(-1, 1), phi_r_inv_full, padding_mode='border', align_corners=True
        ).movedim(1, -1)
        u_inv = self.warp_r2l_inv + w_l2r_sampled

        # Refine total inverse field with Anderson acceleration
        if self.config.inverse_steps > 0:
            inv_steps = max(25, self.config.inverse_steps)
            u_inv_cand = update_inverse_field_nd_anderson(
                u_fwd, u_inv,
                steps=inv_steps, m=5,
                max_error_threshold=1e-4, mean_error_threshold=1e-5
            )
            eval_pts = pts_f if pts_f is not None else pts_m
            if eval_pts is None:
                g_eval = torch.Generator(device='cpu').manual_seed(42)
                eval_pts = (torch.rand(500, dim, generator=g_eval, dtype=dtype).to(device) * 1.6) - 0.8
            y_base = warp_scattered_coordinates(eval_pts, u_fwd, direction='forward')
            rec_base = warp_scattered_coordinates(y_base, u_inv, direction='forward')
            err_base = float((rec_base - eval_pts).abs().max().item())

            y_cand = warp_scattered_coordinates(eval_pts, u_fwd, direction='forward')
            rec_cand = warp_scattered_coordinates(y_cand, u_inv_cand, direction='forward')
            err_cand = float((rec_cand - eval_pts).abs().max().item())

            if err_cand <= err_base:
                u_inv = u_inv_cand

        self.disp_fwd.copy_(u_fwd)
        self.disp_inv.copy_(u_inv)

        # 7. Compute Warped Outputs via M2 Primitives
        warped_moving_grid = F.grid_sample(
            J_moving_full, full_identity + self.disp_fwd, padding_mode='border', align_corners=True
        )
        warped_fixed_grid = F.grid_sample(
            I_fixed_full, full_identity + self.disp_inv, padding_mode='border', align_corners=True
        )
        midpoint_fixed = F.grid_sample(
            I_fixed_full, phi_l_full, padding_mode='border', align_corners=True
        )
        midpoint_moving = F.grid_sample(
            J_moving_full, phi_r_full, padding_mode='border', align_corners=True
        )

        warped_moving_points: Optional[torch.Tensor] = None
        warped_fixed_points: Optional[torch.Tensor] = None
        warped_moving_features: Optional[torch.Tensor] = None
        warped_fixed_features: Optional[torch.Tensor] = None

        if has_scattered_moving:
            # Map moving scattered points into fixed space via total inverse displacement
            moving_pts_in = moving_points if (isinstance(moving_points, torch.Tensor) and moving_points.requires_grad) else pts_m
            warped_moving_points = warp_scattered_coordinates(
                moving_pts_in, self.disp_inv, direction='forward',
                domain_bounds=self.config.domain_bounds,
                coord_convention=self.config.coord_convention,
            )
            if has_scattered_fixed:
                warped_moving_features = transport_scattered_to_scattered(
                    pts_m, fts_m, pts_f, self.disp_inv,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                )
            else:
                warped_moving_features = fts_m

        if has_scattered_fixed:
            # Map fixed scattered points into moving space via total forward displacement
            fixed_pts_in = fixed_points if (isinstance(fixed_points, torch.Tensor) and fixed_points.requires_grad) else pts_f
            warped_fixed_points = warp_scattered_coordinates(
                fixed_pts_in, self.disp_fwd, direction='forward',
                domain_bounds=self.config.domain_bounds,
                coord_convention=self.config.coord_convention,
            )
            if has_scattered_moving:
                warped_fixed_features = transport_scattered_to_scattered(
                    pts_f, fts_f, pts_m, self.disp_fwd,
                    domain_bounds=self.config.domain_bounds,
                    coord_convention=self.config.coord_convention,
                )
            else:
                warped_fixed_features = fts_f

        # 8. Compute Physical Quality & Folding Metrics
        folding_pct, min_jac, mean_jac = _compute_grid_folding(self.disp_fwd, domain_mask=level_mask)

        # Compute inverse consistency on scattered sample points in [-0.8, 0.8]^d
        pts_eval = pts_f if pts_f is not None else pts_m
        if pts_eval is None or pts_eval.shape[0] < 50:
            g_test = torch.Generator(device='cpu').manual_seed(1234)
            pts_rnd = (torch.rand(500, dim, generator=g_test, dtype=dtype).to(device) * 1.6) - 0.8
            pts_eval = torch.cat([pts_eval, pts_rnd], dim=0) if pts_eval is not None else pts_rnd

        y_pts = warp_scattered_coordinates(pts_eval, self.disp_fwd, direction='forward')
        rec_pts = warp_scattered_coordinates(y_pts, self.disp_inv, direction='forward')
        inv_identity_error = float(torch.norm(rec_pts - pts_eval, p=float('inf'), dim=-1).max().item())
        inv_identity_mean = float(torch.norm(rec_pts - pts_eval, p=2, dim=-1).mean().item())

        inv_errors_dict = {
            'max_error': inv_identity_error,
            'mean_error': inv_identity_mean,
            'max_error_l2r': float(compute_inverse_identity_error_nd(self.warp_l2r, self.warp_l2r_inv).max().item()),
            'max_error_r2l': float(compute_inverse_identity_error_nd(self.warp_r2l, self.warp_r2l_inv).max().item()),
        }

        return ScatteredRegistrationResult(
            disp_fwd=self.disp_fwd,
            disp_inv=self.disp_inv,
            warp_l2r=self.warp_l2r,
            warp_r2l=self.warp_r2l,
            warp_l2r_inv=self.warp_l2r_inv,
            warp_r2l_inv=self.warp_r2l_inv,
            fixed_grid=I_fixed_full,
            moving_grid=J_moving_full,
            warped_moving_grid=warped_moving_grid,
            warped_fixed_grid=warped_fixed_grid,
            midpoint_fixed_grid=midpoint_fixed,
            midpoint_moving_grid=midpoint_moving,
            warped_moving_points=warped_moving_points,
            warped_fixed_points=warped_fixed_points,
            warped_moving_features=warped_moving_features,
            warped_fixed_features=warped_fixed_features,
            loss_history=self.loss_history,
            metric_history={'loss': self.loss_history},
            inverse_identity_error=inv_identity_error,
            inverse_identity_errors=inv_errors_dict,
            grid_folding_percentage=folding_pct,
            jacobian_min=min_jac,
            jacobian_mean=mean_jac,
            deformation_energies={'harmonic': float(torch.mean(self.disp_fwd ** 2).item())},
            grid_shape=self.spatial_shape,
            domain_bounds=self.config.domain_bounds,
            config=self.config,
            model=self,
        )

    def forward(
        self,
        moving_points: Optional[torch.Tensor] = None,
        moving_grid: Optional[torch.Tensor] = None,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Applies forward or inverse displacement to input points or Eulerian grids."""
        # In SyN, for points: moving -> fixed is disp_inv ('forward'), fixed -> moving is disp_fwd ('inverse')
        # For grid: pullback of moving onto fixed uses disp_fwd ('forward'), pullback of fixed onto moving uses disp_inv ('inverse')
        pts_disp = self.disp_inv if direction == 'forward' else self.disp_fwd
        grid_disp = self.disp_fwd if direction == 'forward' else self.disp_inv
        out_pts = None
        out_grid = None

        if moving_points is not None:
            out_pts = warp_scattered_coordinates(
                moving_points, pts_disp, direction='forward',
                domain_bounds=self.config.domain_bounds,
                coord_convention=self.config.coord_convention,
            )

        if moving_grid is not None:
            dim = self.dim
            grid_in, _, _ = _standardize_grid_features(moving_grid, dim, 1)
            phi = self.identity + grid_disp
            out_grid = F.grid_sample(grid_in, phi, padding_mode='border', align_corners=True)

        if out_pts is not None and out_grid is not None:
            return out_pts, out_grid
        elif out_pts is not None:
            return out_pts
        elif out_grid is not None:
            return out_grid
        raise ValueError("Must provide either moving_points or moving_grid to forward().")

    def transform_points(
        self,
        coords: torch.Tensor,
        direction: Literal['forward', 'inverse', 'backward'] = 'forward',
    ) -> torch.Tensor:
        """Differentiably warps arbitrary scattered coordinates."""
        disp = self.disp_inv if direction == 'forward' else self.disp_fwd
        return warp_scattered_coordinates(
            coords, disp, direction='forward',
            domain_bounds=self.config.domain_bounds,
            coord_convention=self.config.coord_convention,
        )

    def compute_jacobian_metrics(self) -> Dict[str, float]:
        """Calculates min det(J), mean det(J), and grid folding percentage."""
        folding_pct, min_jac, mean_jac = _compute_grid_folding(self.disp_fwd)
        return {
            'folding_percentage': folding_pct,
            'jacobian_min': min_jac,
            'jacobian_mean': mean_jac,
        }

    def compute_inverse_error(self) -> Dict[str, float]:
        """Calculates maximum and mean inverse identity errors."""
        err_map = compute_inverse_identity_error_nd(self.disp_fwd, self.disp_inv)
        return {
            'max_error': float(err_map.max().item()),
            'mean_error': float(err_map.mean().item()),
        }


def syn_scattered(
    fixed_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
    fixed_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    moving_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
    moving_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    config: Optional[ScatteredRegistrationConfig] = None,
    fixed_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    moving_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    domain_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    point_weights_fixed: Optional[Union[torch.Tensor, np.ndarray]] = None,
    point_weights_moving: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **kwargs,
) -> ScatteredRegistrationResult:
    """High-level functional interface for symmetric diffeomorphic scattered data registration.

    Supports both point-to-point (scattered-to-scattered) and point-to-grid (scattered-to-grid)
    registration, matching PROJECT.md Interface Contracts.

    Parameters
    ----------
    fixed_points : Tensor or ndarray of shape (N_f, d) or (1, N_f, d), optional
        Fixed target scattered coordinates. Pass None if fixed input is an Eulerian grid.
    fixed_features : Tensor or ndarray of shape (N_f,), (N_f, C), or grid tensor, optional
        Features associated with fixed points or direct fixed Eulerian grid tensor.
    moving_points : Tensor or ndarray of shape (N_m, d) or (1, N_m, d), optional
        Moving source scattered coordinates. Pass None if moving input is an Eulerian grid.
    moving_features : Tensor or ndarray of shape (N_m,), (N_m, C), or grid tensor, optional
        Features associated with moving points or direct moving Eulerian grid tensor.
    config : ScatteredRegistrationConfig, optional
        Solver configuration object. If None, constructed from **kwargs and defaults.
    fixed_grid : Tensor or ndarray, optional
        Explicit Eulerian fixed grid tensor.
    moving_grid : Tensor or ndarray, optional
        Explicit Eulerian moving grid tensor.
    domain_mask : Tensor or ndarray, optional
        Binary or continuous mask on Eulerian grid domain.
    point_weights_fixed : Tensor or ndarray, optional
        Confidence weights for fixed points.
    point_weights_moving : Tensor or ndarray, optional
        Confidence weights for moving points.
    **kwargs : Any
        Keyword arguments passed to ScatteredRegistrationConfig or SyNScattered.fit.

    Returns
    -------
    ScatteredRegistrationResult
        Dataclass containing forward/inverse displacement fields, warped points/features,
        warped grids, loss history, Jacobian metrics, and inverse identity errors.
    """
    if config is None:
        if 'dim' not in kwargs:
            if fixed_points is not None:
                dim = fixed_points.shape[-1]
            elif moving_points is not None:
                dim = moving_points.shape[-1]
            elif fixed_grid is not None:
                dim = fixed_grid.ndim - 2 if fixed_grid.ndim > 2 else fixed_grid.ndim
            elif moving_grid is not None:
                dim = moving_grid.ndim - 2 if moving_grid.ndim > 2 else moving_grid.ndim
            else:
                dim = 2
            kwargs['dim'] = dim
        config = ScatteredRegistrationConfig(**kwargs)
    elif kwargs:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = SyNScattered(config=config)
    return model.fit(
        fixed_points=fixed_points,
        fixed_features=fixed_features,
        moving_points=moving_points,
        moving_features=moving_features,
        fixed_grid=fixed_grid,
        moving_grid=moving_grid,
        domain_mask=domain_mask,
        point_weights_fixed=point_weights_fixed,
        point_weights_moving=point_weights_moving,
    )
