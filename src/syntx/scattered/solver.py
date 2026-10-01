"""
syntx.scattered.solver -- SyN-style symmetric registration of point sets (or point set vs grid).

Algorithm (``SyNScattered.fit``):

1. Each point set is turned into a grid image with ``project_scattered_to_grid`` (point
   values, or ones; optionally plus a distance-potential channel). A grid input is used as
   is. The grid spans ``domain_bounds``; all displacement fields live on the normalised
   [-1, 1] grid (``F.grid_sample`` units, ``align_corners=True``, (x, y, z) components).
2. Optional initialisation of the moving half-warp: affine from ``syntx.robust_affine`` on
   the projected images (on by default), or a B-spline landmark fit.
3. For each pyramid level: two half-warps w_l (fixed -> midpoint) and w_r (moving ->
   midpoint) are updated so that I(x + w_l(x)) matches J(x + w_r(x)) (LNCC by default). Each
   iteration: autograd gradient of the loss w.r.t. both half-warps, smoothing (DST-I /
   Sobolev / Gaussian / B-spline), an optimizer rule (Rprop by default), optional symmetric
   projection delta_r = -delta_l, a cap on the largest step in voxels, then
   w <- w - delta(x + w(x)) (default 'lagrangian') or w <- w - delta. The half-warp inverses
   are refreshed by Anderson fixed-point iteration. The iterate with the lowest loss on the
   level is kept.
4. Half-warps are upsampled to full size, inverted more accurately, and composed into
   u_fwd = w_l^-1 + w_r o (Id + w_l^-1) (on the fixed grid, pointing into the moving image)
   and u_inv = w_r^-1 + w_l o (Id + w_r^-1).

Nothing guarantees a fold-free result; ``ScatteredRegistrationResult`` reports the folding
fraction and Jacobian range of u_fwd.
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
from syntx.core.grid import compose_grids, resize_field
from syntx.core.inverse import (
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
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
    """Settings for ``SyNScattered`` / ``syn_scattered``.

    Units: "voxels" are grid nodes of the current pyramid level; displacement fields are in
    normalised [-1, 1] units (one voxel = 2 / (n - 1)).

    Parameters
    ----------
    dim : int, default 2
        Spatial dimension (2 or 3).
    grid_res : int or tuple of int, default 128
        Full-resolution grid size in tensor order; an int gives (grid_res,) * dim.
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
        Box of the grid in point-coordinate units (see ``project_scattered_to_grid``). Used
        for projection and for converting displacements to coordinate units when warping
        points. 'auto' is resolved separately by projection and by point warping, with
        different margins, so explicit bounds are safer.
    sigma : float, sequence of float, or 'auto', default 0.03
        Gaussian projection kernel standard deviation, coordinate units. For a single number,
        each level uses max(sigma, 1.5 * voxel size in [-1, 1] units).
    fluid_sigma : float, default 1.5
        Width of the gradient smoother (in voxels for 'gaussian'; for 'dsti1' / 'sobolev' it
        sets the operator's alpha, see ``syntx.core.smoothing``). Also sets the extra
        Gaussian smoothing of Rprop / reg_adam steps.
    elastic_sigma : float, default 0.0
        If > 0, Gaussian sigma (voxels) applied to the whole half-warps after every update.
    regularizer : {'dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'}, default 'dsti1'
        Gradient smoother: 'dsti1' / 'dsti' (same function; DST-I Sobolev operator, zero
        boundary), 'sobolev' (FFT, periodic), 'bspline' (``apply_bspline_fluid_regularizer``,
        mesh = mesh_size if > 2 else 6; needs an int mesh_size), anything else = Gaussian.
    optimizer_type : {'rprop', 'cfl', 'adam', 'reg_adam'}, default 'rprop'
        Step rule applied to the smoothed gradient v (any other value: delta = lr * v).
        'rprop': per-component sign steps, initial size ``optimizer_lr``, grown x1.2 (max
        0.2) / shrunk x0.5 (min 1e-6) on sign agreement / change, then Gaussian-smoothed;
        when exactly one input is a grid, steps are further scaled by |v| / (0.25 max|v|)
        clipped to 1. 'cfl': v scaled so its largest voxel-length is
        level_cfl * optimizer_lr. 'adam': Adam (beta 0.9 / 0.999, state reset per level).
        'reg_adam': Adam then Gaussian smoothing.
    optimizer_lr : float, default 0.05
        Step size parameter of the rule above.
    in_loop_inv_steps : int, default 5
        Anderson iterations used to refresh the half-warp inverses during optimisation
        (0 = no refresh).
    in_loop_inv_interval : int, default 1
        Refresh every this many iterations (and on the last one).
    inverse_steps : int, default 20
        Final Anderson iterations for the half-warp inverses and the total inverse; the code
        uses max(25, inverse_steps). 0 skips the final refinement.
    inverse_method : {'anderson', 'fixed_point'}, default 'anderson'
        Not used: Anderson is always used.
    cfl_voxels : float, default 0.25
        Step cap: the largest step length of either half-warp is limited to
        cfl_voxels * sqrt(min(level shape) / min(full shape)) voxels. This limits step size;
        it does not guarantee det(J) > 0.
    use_analytical_gradients : bool, default False
        Passed to ``local_ncc_loss_nd`` as ``use_ants_pseudo_gradient`` in ``fit`` (LNCC only).
    similarity_metric : str, default 'lncc'
        'mse'; 'dt' / 'distance_transform' / 'edt' (``distance_transform_loss`` in
        'potential_lncc' mode, tau = distance_transform_tau or 0.10); anything else LNCC.
    window_size : int, default 15
        LNCC window in voxels.
    iterations : int or sequence of int, default 100
        Iterations per level (a short list is padded with its last value).
    levels : sequence, optional
        Pyramid. Read as downsampling factors when levels[0] is in [1, 16] and
        (one level or levels[0] >= levels[-1]), e.g. [4, 2, 1]; otherwise as grid sizes, e.g.
        [32, 64, 128] (int = same size on every axis). Factor levels give
        max(4, round(n / factor)) per axis. None = one level at full size.
    pyramid_levels : sequence, optional
        Copied to ``levels`` when ``levels`` is None.
    epochs_per_level : int or sequence, optional
        If set, used instead of ``iterations`` (it wins even when ``iterations`` was set).
    initial_transform : None, False, 'identity', str, list, or ANTsTransform, default None
        Affine initialisation of the moving half-warp:
        - None (default): ``syntx.robust_affine(fixed, moving, dof=affine_dof,
          mode=affine_mode, seed=affine_seed)`` on ANTs images made from channel 0 of the
          projected grids (unit spacing, zero origin, identity direction). Affine
          pre-alignment is therefore ON by default.
        - False or 'identity': no affine.
        - transform path, list of paths, or ``ants.ANTsTransform``: used instead.
    affine_dof : {'affine', 'rigid'}, default 'affine'
        ``robust_affine`` ``dof``.
    affine_mode : str, default 'pytorch'
        ``robust_affine`` ``mode``.
    affine_seed : int, optional
        ``robust_affine`` ``seed``.
    w_distortion : float, default 0.1
        Not used by the solver.
    antisymmetric : bool, default True
        If True, subtract the mean of the two steps from each, which makes delta_r = -delta_l.
    formulation : {'lagrangian', 'eulerian'}, default 'lagrangian'
        'lagrangian': w <- w - delta(x + w(x)); any other value: w <- w - delta(x).
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Component order of the point coordinates (see ``projection``).
    fill_value : float, default 0.0
        Projection fill value for nodes without support.
    projection_method : {'gaussian', 'bspline'}, default 'gaussian'
        Projection engine.
    number_of_fitting_levels, mesh_size, spline_distance :
        B-spline projection / landmark-fit / 'bspline' regulariser settings.
    landmark_init : bool, default False
        With ``initial_landmarks``, initialise the moving half-warp from a B-spline landmark
        fit (this replaces, not composes with, the affine initialisation).
    initial_landmarks : (fixed (N, d), moving (N, d)), optional
        Landmark pairs for ``landmark_init``.
    distance_transform_tau : float, optional
        If set, a channel exp(-D / tau) (D = distance to the nearest point, coordinate units)
        is appended to each projected image.
    distance_transform_weight : float, default 1.0
        Multiplier of that channel.
    verbose : bool, default False
        Passed to ``robust_affine``; the solver itself prints nothing.
    device : str or torch.device, optional
        If None: device of the first tensor among fixed_points, moving_points, fixed_grid;
        else CPU.
    dtype : torch.dtype, default torch.float32
        Working dtype; float64 is used if fixed or moving points are float64 tensors.
    """
    dim: int = 2
    grid_res: Union[int, Tuple[int, ...]] = 128
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0)
    sigma: Union[float, Sequence[float], str] = 0.03
    fluid_sigma: float = 1.5
    elastic_sigma: float = 0.0
    regularizer: Literal['dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'] = 'dsti1'
    optimizer_type: Literal['rprop', 'cfl', 'adam', 'reg_adam'] = 'rprop'
    optimizer_lr: float = 0.05
    in_loop_inv_steps: int = 5
    in_loop_inv_interval: int = 1
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
    initial_transform: Optional[Union[bool, str, list, Any]] = None
    affine_dof: Literal['affine', 'rigid'] = 'affine'
    affine_mode: str = 'pytorch'
    affine_seed: Optional[int] = None
    w_distortion: float = 0.1
    antisymmetric: bool = True
    formulation: Literal['lagrangian', 'eulerian'] = 'lagrangian'
    coord_convention: Literal['xyz', 'zyx'] = 'xyz'
    fill_value: float = 0.0
    projection_method: Literal['gaussian', 'bspline'] = 'gaussian'
    number_of_fitting_levels: int = 4
    mesh_size: Union[int, Sequence[int]] = 1
    spline_distance: Optional[Union[float, Sequence[float]]] = None
    landmark_init: bool = False
    initial_landmarks: Optional[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]] = None
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
    """Output of ``SyNScattered.fit``; attribute access plus a few dict-style keys.

    All fields are normalised [-1, 1]-unit displacement fields (1, *grid_shape, dim) with
    (x, y, z) components (grid_sample convention), at full resolution.

    Attributes
    ----------
    disp_fwd : Tensor
        u_fwd on the fixed grid: J(x + u_fwd(x)) is the moving image in fixed space; moving
        -> fixed for points uses ``disp_inv``.
    disp_inv : Tensor
        u_inv on the moving grid (approximate inverse of u_fwd).
    warp_l2r, warp_r2l : Tensor
        Half-warps w_l (fixed side) and w_r (moving side) toward the midpoint.
    warp_l2r_inv, warp_r2l_inv : Tensor
        Their Anderson inverses.
    fixed_grid, moving_grid : Tensor (1, C, *grid_shape)
        Full-resolution projected (or given) images.
    warped_moving_grid, warped_fixed_grid : Tensor
        Moving image resampled with u_fwd, fixed image resampled with u_inv.
    midpoint_fixed_grid, midpoint_moving_grid : Tensor
        Images resampled with the half-warps.
    warped_moving_points : Tensor or None
        Moving points mapped into fixed space with u_inv (coordinate units).
    warped_fixed_points : Tensor or None
        Fixed points mapped into moving space with u_fwd.
    warped_moving_features : Tensor or None
        Point-to-point: moving values estimated at the fixed points
        (``transport_scattered_to_scattered`` with u_inv, default sigma 0.03); otherwise the
        moving values unchanged. ``warped_fixed_features``: the same in the other direction.
    loss_history : list of float
        Loss before each update, all levels concatenated. ``metric_history`` = {'loss': same}.
    inverse_identity_error : float
        max over test points of max_k |x - u_inv o u_fwd(x)|_k, in normalised [-1, 1] units
        (500 seeded points in [-0.8, 0.8]^d; ``_inverse_consistency_error``).
    inverse_identity_errors : dict
        'max_error' (as above), 'mean_error' (mean Euclidean error), 'max_error_l2r',
        'max_error_r2l' (the same check for each half-warp and its inverse).
    grid_folding_percentage, jacobian_min, jacobian_mean : float
        Percentage of det(J) <= 0, min and mean det(J) of u_fwd, over the interior (or the
        ``domain_mask``).
    deformation_energies : dict
        {'harmonic': mean(u_fwd ** 2)}, i.e. the mean squared displacement in [-1, 1]
        units (not a gradient energy).
    grid_shape : tuple
    domain_bounds : tuple or str
    config : ScatteredRegistrationConfig
    model : SyNScattered
        The fitted module (its buffers are the same tensors as the fields above).

    Dict-style keys: 'warpedmovout' / 'warpedfixout' (warped points if any, else warped
    grids), 'fwdtransforms' ([disp_fwd]), 'invtransforms' ([disp_inv]), 'fwd_warp',
    'inv_warp', 'syn_losses', 'loss_history', 'inverse_identity_errors', 'model', and any
    attribute name.
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
        """Move points: 'forward' = moving -> fixed (x + u_inv(x)), else fixed -> moving (u_fwd).

        Uses ``warp_scattered_coordinates`` with ``domain_bounds`` and the config's
        coord_convention, but without ``vector_convention``, so for 3-D fields its default
        (reverse components) applies, unlike ``SyNScattered.transform_points``.

        Returns
        -------
        Tensor of the shape of ``coords``.
        """
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
        """``transport_scattered_to_scattered`` with u_inv ('forward') or u_fwd (otherwise).

        Default sigma (0.03) and 'direct' method; same 3-D component caveat as
        ``warp_points``.
        """
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
        """``pullback_grid_to_scattered``: G(x + u(x)) with u = u_fwd ('forward') or u_inv.

        Note the field choice is the opposite of ``warp_points`` for the same ``direction``.
        Same 3-D component caveat as ``warp_points``.
        """
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
        """``pushforward_scattered_to_grid`` with u = u_fwd ('forward') or u_inv, onto
        ``grid_shape`` (default: the registration grid). Default sigma 0.03; same 3-D
        component caveat as ``warp_points``."""
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
        """True for the dict-style keys of ``__getitem__`` or any attribute name."""
        if not isinstance(key, str):
            return False
        keys = {
            'warpedmovout', 'warpedfixout', 'fwdtransforms', 'invtransforms',
            'fwd_warp', 'inv_warp', 'syn_losses', 'loss_history',
            'inverse_identity_errors', 'model'
        }
        return key in keys or hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        """Dict-style access (see the class docstring for the keys); KeyError otherwise."""
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
    """Identity sampling grid (1, *spatial_shape, d) in [-1, 1], components (x, y, z)
    (reversed tensor axes), i.e. the ``F.grid_sample`` / ``align_corners=True`` identity."""
    grids = [torch.linspace(-1, 1, s, dtype=dtype, device=device) for s in spatial_shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    # In Cartesian 'xyz' convention, the coordinate dimensions are reversed (x, y, [z])
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
    return identity


def _compute_grid_folding(warp_field: torch.Tensor, domain_mask: Optional[torch.Tensor] = None) -> Tuple[float, float, float]:
    """Folding percentage (det(J) <= 0), min and mean det(J) of a [-1, 1]-unit field.

    ``compute_physical_jacobian_determinant`` with identity direction and spacing
    2 / (n - 1). Statistics are over voxels where ``domain_mask`` > 0.5 (the mask must have
    the field's spatial shape), else over the interior without the one-voxel rim. Returns
    (0.0, 1.0, 1.0) when no voxel is selected.
    """
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



def _inverse_consistency_error(fwd: torch.Tensor, inv: torch.Tensor, pts: Optional[torch.Tensor] = None,
                               n_points: int = 500, seed: int = 1234) -> Dict[str, float]:
    """Inverse consistency of two solver fields, in their own units: normalised [-1, 1]
    coordinates with (x, y, z) components. Test points (N, d) in the same units (default: ``n_points``
    seeded uniform points in [-0.8, 0.8]^d) are moved by ``fwd`` then ``inv``; returns
    {'max_error': max over points of max_k |x_k - rec_k|, 'mean_error': mean |x - rec|}."""
    d = fwd.shape[-1]
    if pts is None:
        g = torch.Generator(device='cpu').manual_seed(seed)
        pts = (torch.rand(n_points, d, generator=g, dtype=fwd.dtype) * 1.6 - 0.8).to(fwd.device)
    y = warp_scattered_coordinates(pts, fwd, direction='forward', vector_convention='xyz')
    rec = warp_scattered_coordinates(y, inv, direction='forward', vector_convention='xyz')
    diff = rec - pts
    return {'max_error': float(diff.abs().max().item()),
            'mean_error': float(torch.norm(diff, dim=-1).mean().item())}


class SyNScattered(nn.Module):
    """SyN-style symmetric registration of two point sets, or a point set and a grid image.

    See the module docstring for the algorithm and ``ScatteredRegistrationConfig`` for the
    settings. Build with a config (``SyNScattered(config)`` or ``SyNScattered(config=...)``)
    or with keyword arguments; the explicit keyword arguments below and ``**kwargs`` all go
    into a new ``ScatteredRegistrationConfig`` when no config is given (ignored otherwise).

    Buffers (non-persistent, (1, *grid_shape, dim), [-1, 1] units, (x, y, z) components):
    ``warp_l2r``, ``warp_r2l``, ``warp_l2r_inv``, ``warp_r2l_inv``, ``disp_fwd``,
    ``disp_inv`` (zeros until ``fit``), and ``identity``.
    """
    def __init__(
        self,
        dim: int = 2,
        grid_res: Union[int, Tuple[int, ...]] = 128,
        domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
        sigma: Union[float, Sequence[float], str] = 0.03,
        fluid_sigma: float = 1.5,
        elastic_sigma: float = 0.0,
        regularizer: Literal['dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'] = 'dsti1',
        optimizer_type: Literal['rprop', 'cfl', 'adam', 'reg_adam'] = 'rprop',
        in_loop_inv_steps: int = 5,
        in_loop_inv_interval: int = 1,
        inverse_steps: int = 20,
        inverse_method: Literal['anderson', 'fixed_point'] = 'anderson',
        cfl_voxels: float = 0.25,
        use_analytical_gradients: bool = False,
        config: Optional[ScatteredRegistrationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(dim, ScatteredRegistrationConfig):
            config = dim
            dim = config.dim
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
                in_loop_inv_interval=in_loop_inv_interval,
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

    def _compute_affine_prealignment(
        self,
        I_fixed_full: torch.Tensor,
        J_moving_full: torch.Tensor,
        init_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Affine initial displacement for ``warp_r2l`` from ``config.initial_transform``.

        - None: ``robust_affine`` on ANTs images made from channel 0 of the full-resolution
          projected grids (axes reversed to ANTs order; unit spacing, zero origin, identity
          direction, so the affine lives in voxel-index space).
        - False / 'identity': return None.
        - otherwise: the given transform path / list / ANTsTransform (interpreted in that
          same voxel-index space).

        The ANTs affine is turned into an ``F.affine_grid`` theta with
        ``syntx.greedy._build_torch_affine_matrix`` and sampled at ``init_shape``.

        Returns
        -------
        Tensor (1, *init_shape, dim) = affine grid - identity ([-1, 1] units, (x, y, z)
        components), or None (skipped, or ``parse_ants_affine`` found no affine).
        """
        import ants
        from syntx.robust_affine import robust_affine
        from syntx.core.affine import parse_ants_affine
        from syntx.greedy import _build_torch_affine_matrix

        dim = self.dim
        init_tx = self.config.initial_transform

        if init_tx is False:
            return None
        if isinstance(init_tx, str) and init_tx.lower() == 'identity':
            return None

        def _to_ants_image(grid: torch.Tensor) -> "ants.ANTsImage":
            # Tensor spatial order (Z,Y,X) -> ANTs/ITK order (X,Y,Z); identity
            # spacing/origin/direction, since this solver has no physical-space concept.
            arr = grid[0, 0].detach().cpu().numpy()
            arr = np.ascontiguousarray(arr.transpose())
            return ants.from_numpy(arr.astype(np.float32))

        fixed_img = _to_ants_image(I_fixed_full)
        moving_img = _to_ants_image(J_moving_full)

        if init_tx is None:
            aff_res = robust_affine(
                fixed_img, moving_img,
                dof=self.config.affine_dof,
                mode=self.config.affine_mode,
                seed=self.config.affine_seed,
                verbose=verbose,
            )
            tx_list = aff_res['fwdtransforms']
        else:
            tx_list = init_tx if isinstance(init_tx, list) else [init_tx]

        M_phys, t_phys = parse_ants_affine(tx_list, dim)
        if M_phys is None:
            return None

        theta, _, _ = _build_torch_affine_matrix(fixed_img, moving_img, M_phys, t_phys, device)
        theta = theta.to(dtype=dtype)

        grid_aff = F.affine_grid(theta, (1, 1, *init_shape), align_corners=True)
        identity = _make_identity_grid(init_shape, dtype=dtype, device=device)
        return (grid_aff - identity).to(dtype=dtype)

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
        """Similarity loss of the current half-warps, differentiable w.r.t. the inputs.

        Projects the points (or reads the grids) at full resolution with the config's
        projection settings (no per-level sigma), samples both images with the current
        ``warp_l2r`` / ``warp_r2l`` and returns the loss. Nothing is updated; this is the
        same loss as in ``fit`` except that LNCC never uses the pseudo-gradient option. The
        half-warps must be at full resolution (true before ``fit`` and after it).

        Parameters
        ----------
        fixed_points, moving_points : Tensor (N, d) or (1, N, d), optional
            Point sets; gradients flow back to them and to their features.
        fixed_features, moving_features : Tensor, optional
            Point values ((N,) or (N, C)), or a grid when the matching points are None.
        fixed_grid, moving_grid : Tensor, optional
            Grid images (channels-first or (*spatial)) used when points are None.
        domain_mask : Tensor, optional
            Mask for LNCC / distance-transform loss (ignored by 'mse').
        point_weights_fixed, point_weights_moving : Tensor, optional
            Projection weights.

        Returns
        -------
        Tensor, scalar loss (lower is better; LNCC is negative).
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
                method=self.config.projection_method,
                number_of_fitting_levels=self.config.number_of_fitting_levels,
                mesh_size=self.config.mesh_size,
                spline_distance=self.config.spline_distance,
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
                method=self.config.projection_method,
                number_of_fitting_levels=self.config.number_of_fitting_levels,
                mesh_size=self.config.mesh_size,
                spline_distance=self.config.spline_distance,
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
        """Register fixed and moving inputs (point-to-point or point-to-grid) and return results.

        See the module docstring for the algorithm. Inputs are detached: ``fit`` does not
        back-propagate to them (the returned warped points keep a graph only when the
        original point tensor had ``requires_grad``). The module's buffers are overwritten.

        Parameters
        ----------
        fixed_points, moving_points : Tensor or ndarray (N, d) or (1, N, d), optional
            Point sets in coordinate units of ``config.domain_bounds``. An empty set raises
            ValueError.
        fixed_features, moving_features : optional
            Values per point ((N,) or (N, C)); ones if None. When the matching points are
            None and no grid is given, this is taken as the grid image instead.
        fixed_grid, moving_grid : Tensor or ndarray, optional
            Grid images used when the matching points are None: (*spatial), (C, *spatial) or
            (1, C, *spatial). Each side needs points, a grid or features (ValueError otherwise).
        domain_mask : Tensor or ndarray, optional
            Grid mask, resized (nearest) to each level, used by the LNCC / distance loss
            and, at the end, to select voxels for the folding statistics (that last step
            needs the final level to be at full resolution).
        point_weights_fixed, point_weights_moving : optional
            Projection weights per point.
        epochs : int, optional
            Iterations per level; overrides the config.
        verbose : bool, optional
            Overrides ``config.verbose`` (only passed to ``robust_affine``).
        **kwargs
            Ignored.

        Returns
        -------
        ScatteredRegistrationResult

        Notes
        -----
        Device / dtype: see ``ScatteredRegistrationConfig.device`` / ``dtype``.

        Initialisation: affine (default) and then, if ``landmark_init``, the landmark fit
        overwrites ``warp_r2l``; in both cases ``warp_r2l_inv`` is set by 15 Anderson
        iterations. The landmark field is copied as is, so it is only consistent with the
        [-1, 1] / (x, y, z) convention of the half-warps when domain_bounds = (-1, 1) and
        coord_convention = 'xyz'.

        Inverse-error report: 500 seeded points in [-0.8, 0.8]^d (normalised units) are moved by
        u_fwd then u_inv with the fields' (x, y, z) component convention; the same check
        chooses between two candidate total inverses.
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
                method=self.config.projection_method,
                number_of_fitting_levels=self.config.number_of_fitting_levels,
                mesh_size=self.config.mesh_size,
                spline_distance=self.config.spline_distance,
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
                method=self.config.projection_method,
                number_of_fitting_levels=self.config.number_of_fitting_levels,
                mesh_size=self.config.mesh_size,
                spline_distance=self.config.spline_distance,
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

        # Optional rigid/affine pre-alignment (backed by syntx.robust_affine)
        effective_verbose = bool(verbose) if verbose is not None else bool(self.config.verbose)
        w_aff = self._compute_affine_prealignment(
            I_fixed_full, J_moving_full, init_shape, device, dtype,
            verbose=effective_verbose,
        )
        if w_aff is not None:
            self.warp_r2l.copy_(w_aff)
            self.warp_r2l_inv = update_inverse_field_nd_anderson(self.warp_r2l, None, steps=15, m=5)

        # Optional landmark warm-start initialization
        if self.config.landmark_init and self.config.initial_landmarks is not None:
            lm_f, lm_m = self.config.initial_landmarks
            from .bspline import fit_bspline_landmark_warp
            w_lm = fit_bspline_landmark_warp(
                fixed_landmarks=lm_f,
                moving_landmarks=lm_m,
                grid_shape=init_shape,
                domain_bounds=self.config.domain_bounds,
                number_of_fitting_levels=self.config.number_of_fitting_levels,
                mesh_size=self.config.mesh_size,
                spline_distance=self.config.spline_distance,
                coord_convention=self.config.coord_convention,
                device=device,
                dtype=dtype,
            )
            self.warp_r2l.copy_(w_lm)
            self.warp_r2l_inv = update_inverse_field_nd_anderson(self.warp_r2l, None, steps=15, m=5)

        self.loss_history = []
        interp_mode = 'bilinear' if dim == 2 else 'trilinear'

        # 4. Multi-Resolution SyN Iteration Loop
        for level_idx, (curr_shape, n_epochs) in enumerate(zip(pyramid_shapes, epochs_list)):
            h_voxel = torch.tensor([2.0 / (s - 1) for s in curr_shape], device=device, dtype=dtype)
            spacing_t = h_voxel.view(*([1] * (dim + 1)), dim)

            # Upsample half-warps when transitioning between pyramid levels
            if level_idx > 0:
                self.warp_l2r = resize_field(self.warp_l2r, size=curr_shape).contiguous()
                self.warp_r2l = resize_field(self.warp_r2l, size=curr_shape).contiguous()
                self.warp_l2r_inv = resize_field(self.warp_l2r_inv, size=curr_shape).contiguous()
                self.warp_r2l_inv = resize_field(self.warp_r2l_inv, size=curr_shape).contiguous()

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
                    method=self.config.projection_method,
                    number_of_fitting_levels=self.config.number_of_fitting_levels,
                    mesh_size=self.config.mesh_size,
                    spline_distance=self.config.spline_distance,
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
                    method=self.config.projection_method,
                    number_of_fitting_levels=self.config.number_of_fitting_levels,
                    mesh_size=self.config.mesh_size,
                    spline_distance=self.config.spline_distance,
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

            # Optimizer state buffers (Rprop & Adam)
            rprop_step_l = torch.ones_like(self.warp_l2r) * self.config.optimizer_lr
            rprop_step_r = torch.ones_like(self.warp_r2l) * self.config.optimizer_lr
            rprop_prev_grad_l = torch.zeros_like(self.warp_l2r)
            rprop_prev_grad_r = torch.zeros_like(self.warp_r2l)
            adam_m_l = torch.zeros_like(self.warp_l2r)
            adam_v_l = torch.zeros_like(self.warp_l2r)
            adam_m_r = torch.zeros_like(self.warp_r2l)
            adam_v_r = torch.zeros_like(self.warp_r2l)

            # Pre-interpolate domain mask if provided
            level_mask = None
            if domain_mask is not None:
                mask_t = torch.as_tensor(domain_mask, device=device, dtype=dtype).detach()
                while mask_t.dim() < dim + 2:
                    mask_t = mask_t.unsqueeze(0)
                level_mask = F.interpolate(mask_t, size=curr_shape, mode='nearest')

            best_level_loss = float('inf')
            best_warp_l2r = None
            best_warp_r2l = None
            best_warp_l2r_inv = None
            best_warp_r2l_inv = None

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

                loss_val = float(loss.item())
                self.loss_history.append(loss_val)

                # Checkpoint best solution at current resolution level
                if loss_val < best_level_loss:
                    best_level_loss = loss_val
                    best_warp_l2r = self.warp_l2r.detach().clone()
                    best_warp_r2l = self.warp_r2l.detach().clone()
                    if hasattr(self, 'warp_l2r_inv') and self.warp_l2r_inv is not None:
                        best_warp_l2r_inv = self.warp_l2r_inv.detach().clone()
                        best_warp_r2l_inv = self.warp_r2l_inv.detach().clone()

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
                    elif reg == 'bspline':
                        from .bspline import apply_bspline_fluid_regularizer
                        res_b = _resolve_domain_bounds(
                            self.config.domain_bounds,
                            pts_f if has_scattered_fixed else (pts_m if has_scattered_moving else torch.zeros((1, dim), device=device, dtype=dtype)),
                            dim,
                        )
                        if res_b is None:
                            res_b = (torch.full((dim,), -1.0, device=device, dtype=dtype), torch.full((dim,), 1.0, device=device, dtype=dtype))
                        reg_mesh = self.config.mesh_size if self.config.mesh_size > 2 else 6
                        v_l = apply_bspline_fluid_regularizer(
                            grad_l * b_mask,
                            grid_shape=curr_shape,
                            domain_bounds=res_b,
                            mesh_size=reg_mesh,
                            spline_distance=self.config.spline_distance,
                            enforce_stationary_boundary=False,
                            coord_convention=self.config.coord_convention,
                        )
                        v_r = apply_bspline_fluid_regularizer(
                            grad_r * b_mask,
                            grid_shape=curr_shape,
                            domain_bounds=res_b,
                            mesh_size=reg_mesh,
                            spline_distance=self.config.spline_distance,
                            enforce_stationary_boundary=False,
                            coord_convention=self.config.coord_convention,
                        )
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
                    elif opt_type in ('adam', 'reg_adam'):
                        beta1, beta2 = 0.9, 0.999
                        step = epoch + 1
                        bias_correction1 = 1.0 - beta1 ** step
                        bias_correction2 = 1.0 - beta2 ** step
                        bias_correction2_sqrt = math.sqrt(bias_correction2)
                        step_size = (self.config.optimizer_lr * bias_correction2_sqrt) / bias_correction1
                        eps_scaled = 1e-8 * bias_correction2_sqrt

                        adam_m_l.mul_(beta1).add_(v_l, alpha=1.0 - beta1)
                        adam_v_l.mul_(beta2).addcmul_(v_l, v_l, value=1.0 - beta2)
                        delta_l = (adam_m_l * step_size).div_(adam_v_l.sqrt().add_(eps_scaled))

                        adam_m_r.mul_(beta1).add_(v_r, alpha=1.0 - beta1)
                        adam_v_r.mul_(beta2).addcmul_(v_r, v_r, value=1.0 - beta2)
                        delta_r = (adam_m_r * step_size).div_(adam_v_r.sqrt().add_(eps_scaled))

                        if opt_type == 'reg_adam':
                            delta_l = separable_gaussian_filter(delta_l * b_mask, sigma=fluid_sig) * b_mask
                            delta_r = separable_gaussian_filter(delta_r * b_mask, sigma=fluid_sig) * b_mask
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
                        delta_l_pb = compose_grids(delta_l, level_identity + self.warp_l2r)
                        delta_r_pb = compose_grids(delta_r, level_identity + self.warp_r2l)

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
                        inv_interval = max(1, getattr(self.config, 'in_loop_inv_interval', 1))
                        if (epoch + 1) % inv_interval == 0 or epoch == n_epochs - 1:
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

            # Evaluate final state after the last optimization step
            def _eval_current_loss():
                with torch.no_grad():
                    phi_l = level_identity + self.warp_l2r
                    phi_r = level_identity + self.warp_r2l
                    I_m = F.grid_sample(I_curr, phi_l, padding_mode='border', align_corners=True)
                    J_m = F.grid_sample(J_curr, phi_r, padding_mode='border', align_corners=True)
                    if self.config.similarity_metric == 'mse':
                        return float(F.mse_loss(I_m, J_m).item())
                    elif self.config.similarity_metric in ('dt', 'distance_transform', 'edt'):
                        from syntx.core.losses import distance_transform_loss
                        return float(distance_transform_loss(
                            I_m, J_m,
                            mode='potential_lncc',
                            tau=self.config.distance_transform_tau or 0.10,
                            window_size=self.config.window_size,
                            mask=level_mask,
                        ).item())
                    else:
                        return float(local_ncc_loss_nd(
                            I_m, J_m,
                            mask=level_mask,
                            window_size=self.config.window_size,
                            use_ants_pseudo_gradient=self.config.use_analytical_gradients,
                        ).item())

            if n_epochs > 0:
                final_loss = _eval_current_loss()
                if final_loss < best_level_loss:
                    best_level_loss = final_loss
                elif best_warp_l2r is not None:
                    self.warp_l2r.copy_(best_warp_l2r)
                    self.warp_r2l.copy_(best_warp_r2l)
                    if best_warp_l2r_inv is not None:
                        self.warp_l2r_inv.copy_(best_warp_l2r_inv)
                    if best_warp_r2l_inv is not None:
                        self.warp_r2l_inv.copy_(best_warp_r2l_inv)

        # 5. Bring half-warps to full resolution if needed
        if self.warp_l2r.shape[1:-1] != self.spatial_shape:
            self.warp_l2r = resize_field(self.warp_l2r, size=self.spatial_shape).contiguous()
            self.warp_r2l = resize_field(self.warp_r2l, size=self.spatial_shape).contiguous()
            self.warp_l2r_inv = resize_field(self.warp_l2r_inv, size=self.spatial_shape).contiguous()
            self.warp_r2l_inv = resize_field(self.warp_r2l_inv, size=self.spatial_shape).contiguous()

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

        w_r2l_sampled = compose_grids(self.warp_r2l, phi_l_inv_full)
        u_fwd = self.warp_l2r_inv + w_r2l_sampled

        w_l2r_sampled = compose_grids(self.warp_l2r, phi_r_inv_full)
        u_inv = self.warp_r2l_inv + w_l2r_sampled

        # Refine total inverse field with Anderson acceleration
        if self.config.inverse_steps > 0:
            inv_steps = max(25, self.config.inverse_steps)
            u_inv_cand = update_inverse_field_nd_anderson(
                u_fwd, u_inv,
                steps=inv_steps, m=5,
                max_error_threshold=1e-4, mean_error_threshold=1e-5
            )
            err_base = _inverse_consistency_error(u_fwd, u_inv, seed=42)['max_error']
            err_cand = _inverse_consistency_error(u_fwd, u_inv_cand, seed=42)['max_error']
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
                vector_convention='xyz',
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
                vector_convention='xyz',
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

        # Inverse consistency in the fields' own units ([-1, 1], x-y-z components)
        total = _inverse_consistency_error(self.disp_fwd, self.disp_inv)
        inv_identity_error = total['max_error']
        inv_errors_dict = {
            'max_error': total['max_error'],
            'mean_error': total['mean_error'],
            'max_error_l2r': _inverse_consistency_error(self.warp_l2r, self.warp_l2r_inv)['max_error'],
            'max_error_r2l': _inverse_consistency_error(self.warp_r2l, self.warp_r2l_inv)['max_error'],
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
        """Apply the fitted transform to points and/or a grid image.

        Parameters
        ----------
        moving_points : Tensor (N, d) or (B, N, d), optional
            'forward': mapped moving -> fixed with u_inv; otherwise fixed -> moving with u_fwd.
        moving_grid : Tensor, optional
            Grid image (channels-first, or (*spatial)) at the full grid size.
            'forward': resampled with u_fwd (moving image into fixed space); otherwise with
            u_inv. Border padding.
        direction : {'forward', 'inverse', 'backward'}, default 'forward'
            Any value other than 'forward' means inverse.

        Returns
        -------
        Warped points, warped grid (1, C, *grid_shape), or (points, grid) if both were given.
        ValueError if neither is given.
        """
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
                vector_convention='xyz',
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
        """Move points: 'forward' = moving -> fixed (u_inv), otherwise fixed -> moving (u_fwd).

        Uses ``domain_bounds`` and coord_convention from the config and
        ``vector_convention='xyz'`` (correct for the solver's fields). Differentiable w.r.t.
        ``coords``.
        """
        disp = self.disp_inv if direction == 'forward' else self.disp_fwd
        return warp_scattered_coordinates(
            coords, disp, direction='forward',
            domain_bounds=self.config.domain_bounds,
            coord_convention=self.config.coord_convention,
            vector_convention='xyz',
        )

    def compute_jacobian_metrics(self) -> Dict[str, float]:
        """Jacobian statistics of ``disp_fwd`` over the interior (no mask).

        Returns
        -------
        dict with 'folding_percentage' (% det(J) <= 0), 'jacobian_min', 'jacobian_mean'.
        """
        folding_pct, min_jac, mean_jac = _compute_grid_folding(self.disp_fwd)
        return {
            'folding_percentage': folding_pct,
            'jacobian_min': min_jac,
            'jacobian_mean': mean_jac,
        }

    def compute_inverse_error(self) -> Dict[str, float]:
        """Inverse consistency of ``disp_fwd`` / ``disp_inv`` in normalised [-1, 1] units
        (``_inverse_consistency_error``: 500 seeded points in [-0.8, 0.8]^d).

        Returns
        -------
        dict with 'max_error' (max over points of the largest component error) and
        'mean_error' (mean Euclidean error).
        """
        return _inverse_consistency_error(self.disp_fwd, self.disp_inv)


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
    """Functional wrapper: build a config, then ``SyNScattered(config).fit(...)``.

    Point-to-point (both point sets) or point-to-grid (one side a grid image).

    Parameters
    ----------
    fixed_points, moving_points : Tensor or ndarray (N, d) or (1, N, d), optional
        Point sets; None when that side is a grid.
    fixed_features, moving_features : optional
        Values per point, or the grid image when the matching points and grid are None.
    config : ScatteredRegistrationConfig, optional
        If None, built from ``**kwargs``; ``dim`` (if not in kwargs) is inferred from the last
        axis of the points, else from the grid as ndim - 2 when ndim > 2 (so an unbatched,
        unchanneled 3-D grid gives the wrong dim; pass ``dim``), else 2. If given and kwargs
        are present, matching attributes are set on this config object in place (unknown
        names are ignored silently).
    fixed_grid, moving_grid : Tensor or ndarray, optional
        Grid images; see ``SyNScattered.fit``.
    domain_mask, point_weights_fixed, point_weights_moving : optional
        See ``SyNScattered.fit``.
    **kwargs
        ``ScatteredRegistrationConfig`` fields only; nothing is passed to ``fit`` (so
        ``epochs`` raises TypeError when config is None).

    Returns
    -------
    ScatteredRegistrationResult
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
