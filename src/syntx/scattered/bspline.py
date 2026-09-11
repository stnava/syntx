"""B-Spline Scattered Data Facilities and SyN Registration Tools.

Integrates ANTsTorch's multi-level B-spline scattered data approximation and
free-form deformation facilities into syntx.scattered.

Provides:
- `has_antstorch`: Safe import check for ANTsTorch B-spline flows.
- `fit_bspline_landmark_warp`: Closed-form C^2 continuous displacement field fitting
  from paired landmark offsets, providing instantaneous topological warm-starts.
- `apply_bspline_fluid_regularizer`: B-spline velocity smoothing operator (BSplineSyN).
- `BSplineScatteredProjector`: PyTorch nn.Module for multi-resolution B-spline projection.
- `bspline_syn_scattered`: High-level diffeomorphic SyN registration configured with
  B-spline projection and/or B-spline regularization.
"""

from typing import Optional, Sequence, Tuple, Union, Literal, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def has_antstorch() -> bool:
    """Check if ANTsTorch B-spline flows are available."""
    try:
        import sys
        orig_backend = None
        if 'matplotlib' in sys.modules:
            import matplotlib
            orig_backend = matplotlib.get_backend()
        from antstorch.bspline_flows import fit_bspline_object_to_scattered_data, ImageDomain
        if orig_backend and orig_backend != 'Agg':
            try:
                import matplotlib
                matplotlib.use(orig_backend)
            except Exception:
                pass
        return True
    except ImportError:
        return False


def _require_antstorch():
    if not has_antstorch():
        raise ImportError(
            "ANTsTorch B-spline facilities require 'antstorch'. "
            "Please install it via 'pip install antstorch'."
        )


def fit_bspline_landmark_warp(
    fixed_landmarks: Union[torch.Tensor, np.ndarray],
    moving_landmarks: Union[torch.Tensor, np.ndarray],
    grid_shape: Union[int, Tuple[int, ...]],
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    enforce_stationary_boundary: bool = True,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fit a smooth C^2 B-spline displacement field from paired anatomical landmarks.

    Computes the displacement vectors u(x_f) = x_m - x_f mapping fixed landmarks x_f
    to moving landmarks x_m, and fits a multi-level cubic B-spline displacement field
    over the specified Eulerian grid domain using `antstorch.bspline_flows.fit_bspline_displacement_field`.

    Parameters
    ----------
    fixed_landmarks : Tensor or ndarray of shape (N, d) or (B, N, d)
        Coordinates of landmarks in fixed/target space.
    moving_landmarks : Tensor or ndarray of shape (N, d) or (B, N, d)
        Coordinates of corresponding landmarks in moving/source space.
    grid_shape : int or Tuple[int, ...]
        Output grid resolution, e.g. 64 or (64, 64) or (64, 64, 64).
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
        Coordinate bounding box of the Eulerian grid domain.
    number_of_fitting_levels : int, default 4
        Number of multi-resolution B-spline refinement levels.
    mesh_size : int or Sequence[int], default 1
        Base level B-spline mesh size per axis.
    spline_distance : float or Sequence[float], optional
        Physical knot distance in coordinate units. Overrides `mesh_size` if given.
    enforce_stationary_boundary : bool, default True
        If True, locks the boundary voxels of the domain to zero displacement.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate axis ordering convention.
    weights : Tensor or ndarray, optional
        Confidence weights for individual landmarks of shape (N,) or (B, N).
    device : str or torch.device, optional
    dtype : torch.dtype, default torch.float32

    Returns
    -------
    torch.Tensor
        Eulerian displacement field of shape (B, *spatial_shape, d) (or (1, *spatial_shape, d))
        in physical coordinate units matching syntx conventions.
    """
    _require_antstorch()
    from antstorch.bspline_flows import (
        fit_bspline_displacement_field,
        ImageDomain,
        mesh_size_for_spline_distance,
    )

    if device is None:
        if isinstance(fixed_landmarks, torch.Tensor):
            device = fixed_landmarks.device
        else:
            device = torch.device('cpu')

    if not isinstance(fixed_landmarks, torch.Tensor):
        fixed_landmarks = torch.as_tensor(fixed_landmarks, dtype=dtype, device=device)
    else:
        fixed_landmarks = fixed_landmarks.to(device=device, dtype=dtype)

    if not isinstance(moving_landmarks, torch.Tensor):
        moving_landmarks = torch.as_tensor(moving_landmarks, dtype=dtype, device=device)
    else:
        moving_landmarks = moving_landmarks.to(device=device, dtype=dtype)

    unbatched = fixed_landmarks.dim() == 2
    if unbatched:
        fixed_landmarks = fixed_landmarks.unsqueeze(0)
        moving_landmarks = moving_landmarks.unsqueeze(0)

    B, N, d = fixed_landmarks.shape
    if moving_landmarks.shape != fixed_landmarks.shape:
        raise ValueError(
            f"fixed_landmarks shape {fixed_landmarks.shape} does not match moving_landmarks {moving_landmarks.shape}"
        )

    if isinstance(grid_shape, int):
        spatial_shape = (grid_shape,) * d
    elif len(grid_shape) != d:
        raise ValueError(f"grid_shape length {len(grid_shape)} must match landmark dimension {d}")
    else:
        spatial_shape = tuple(grid_shape)

    # Resolve domain bounds
    if domain_bounds is None or domain_bounds == 'auto':
        all_pts = torch.cat([fixed_landmarks, moving_landmarks], dim=1)
        min_b = all_pts.amin(dim=(0, 1)) - 0.10
        max_b = all_pts.amax(dim=(0, 1)) + 0.10
    elif isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2:
        if isinstance(domain_bounds[0], (int, float)):
            min_b = torch.full((d,), float(domain_bounds[0]), device=device, dtype=dtype)
            max_b = torch.full((d,), float(domain_bounds[1]), device=device, dtype=dtype)
        else:
            min_b = torch.as_tensor(domain_bounds[0], device=device, dtype=dtype)
            max_b = torch.as_tensor(domain_bounds[1], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported domain_bounds specification: {domain_bounds}")

    # Build ITK ImageDomain
    if coord_convention == 'xyz':
        size_itk = tuple(reversed(spatial_shape))
        origin_itk = tuple(float(min_b[i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[i] - min_b[i]).item()) for i in range(d))
    elif coord_convention == 'zyx':
        size_itk = tuple(spatial_shape)
        origin_itk = tuple(float(min_b[d - 1 - i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[d - 1 - i] - min_b[d - 1 - i]).item()) for i in range(d))
    else:
        raise ValueError(f"Unknown coord_convention: '{coord_convention}', expected 'xyz' or 'zyx'")

    spacing_itk = tuple(extent / max(1, s - 1) for extent, s in zip(extent_itk, size_itk))
    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        mesh_size_itk = mesh_size_for_spline_distance(domain, spline_distance)
    elif isinstance(mesh_size, int):
        mesh_size_itk = (mesh_size,) * d
    else:
        mesh_size_itk = tuple(reversed(mesh_size)) if coord_convention == 'zyx' else tuple(mesh_size)

    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=dtype, device=device)
        else:
            weights = weights.to(device=device, dtype=dtype)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand(B, -1)

    batch_warps = []
    for b in range(B):
        pts_f_b = fixed_landmarks[b]      # (N, d)
        pts_m_b = moving_landmarks[b]     # (N, d)
        disp_b = pts_m_b - pts_f_b        # (N, d)

        if coord_convention == 'zyx':
            pts_f_b = pts_f_b.flip(dims=[-1])
            disp_b = disp_b.flip(dims=[-1])

        w_b = weights[b] if weights is not None else None

        # fit_bspline_displacement_field returns (1, d, *domain.torch_size)
        warp_itk = fit_bspline_displacement_field(
            displacement_origins=pts_f_b,
            displacements=disp_b,
            displacement_weights=w_b,
            domain=domain,
            number_of_fitting_levels=number_of_fitting_levels,
            mesh_size=mesh_size_itk,
            enforce_stationary_boundary=enforce_stationary_boundary,
        )

        # Convert (1, d, *spatial) to (1, *spatial, d)
        warp_syntx = warp_itk.movedim(1, -1)

        if coord_convention == 'zyx':
            # Flip vector channels back to zyx
            warp_syntx = warp_syntx.flip(dims=[-1])

        batch_warps.append(warp_syntx)

    out_warp = torch.cat(batch_warps, dim=0)  # (B, *spatial_shape, d)
    setattr(out_warp, 'is_physical', True)
    setattr(out_warp, 'vector_convention', coord_convention)
    return out_warp


def apply_bspline_fluid_regularizer(
    velocity_field: torch.Tensor,
    grid_shape: Tuple[int, ...],
    domain_bounds: Tuple[torch.Tensor, torch.Tensor],
    mesh_size: Union[int, Sequence[int]] = 6,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    enforce_stationary_boundary: bool = False,
    coord_convention: str = 'xyz',
) -> torch.Tensor:
    """Smooth a velocity field using single-level cubic B-spline fitting (BSplineSyN regularizer).

    Parameters
    ----------
    velocity_field : torch.Tensor
        Velocity field of shape (1, *spatial, d) or (*spatial, d).
    grid_shape : Tuple[int, ...]
        Spatial dimensions of the field.
    domain_bounds : Tuple[torch.Tensor, torch.Tensor]
        (min_coords, max_coords) bounding box.
    mesh_size : int or Sequence[int], default 2
        Number of control intervals per axis.
    spline_distance : float or Sequence[float], optional
    enforce_stationary_boundary : bool, default True
    coord_convention : str, default 'xyz'

    Returns
    -------
    torch.Tensor
        Smoothed velocity field of identical shape, dtype, and device.
    """
    _require_antstorch()
    from antstorch.bspline_flows import (
        fit_bspline_displacement_field,
        ImageDomain,
        mesh_size_for_spline_distance,
    )

    unbatched = velocity_field.ndim == len(grid_shape) + 1
    if unbatched:
        v = velocity_field.unsqueeze(0)
    else:
        v = velocity_field

    B = v.shape[0]
    d = v.shape[-1]
    min_b, max_b = domain_bounds

    if coord_convention == 'xyz':
        size_itk = tuple(reversed(grid_shape))
        origin_itk = tuple(float(min_b[i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[i] - min_b[i]).item()) for i in range(d))
    else:
        size_itk = tuple(grid_shape)
        origin_itk = tuple(float(min_b[d - 1 - i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[d - 1 - i] - min_b[d - 1 - i]).item()) for i in range(d))

    spacing_itk = tuple(extent / max(1, s - 1) for extent, s in zip(extent_itk, size_itk))
    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        mesh_size_itk = mesh_size_for_spline_distance(domain, spline_distance)
    elif isinstance(mesh_size, int):
        mesh_size_itk = (mesh_size,) * d
    else:
        mesh_size_itk = tuple(reversed(mesh_size)) if coord_convention == 'zyx' else tuple(mesh_size)

    smoothed_batches = []
    for b in range(B):
        vb = v[b:b+1]  # (1, *spatial, d)
        # Convert to (1, d, *spatial)
        if coord_convention == 'zyx':
            vb_itk = vb.movedim(-1, 1).flip(1)
        else:
            vb_itk = vb.movedim(-1, 1)

        sm = fit_bspline_displacement_field(
            displacement_field=vb_itk,
            domain=domain,
            number_of_fitting_levels=1,
            mesh_size=mesh_size_itk,
            enforce_stationary_boundary=enforce_stationary_boundary,
        )

        if coord_convention == 'zyx':
            sm_syntx = sm.flip(1).movedim(1, -1)
        else:
            sm_syntx = sm.movedim(1, -1)

        smoothed_batches.append(sm_syntx)

    out = torch.cat(smoothed_batches, dim=0)
    return out.squeeze(0) if unbatched else out


class BSplineScatteredProjector(nn.Module):
    """Multi-resolution B-spline scattered data projector module.

    Precomputes domain bounds and configuration to efficiently project
    scattered points and features onto Eulerian grids using ANTsTorch B-splines.
    """
    def __init__(
        self,
        grid_shape: Union[int, Tuple[int, ...]] = 64,
        domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
        number_of_fitting_levels: int = 4,
        mesh_size: Union[int, Sequence[int]] = 1,
        spline_distance: Optional[Union[float, Sequence[float]]] = None,
        coord_convention: Literal['xyz', 'zyx'] = 'xyz',
        fill_value: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        _require_antstorch()
        self.grid_shape = grid_shape
        self.domain_bounds = domain_bounds
        self.number_of_fitting_levels = number_of_fitting_levels
        self.mesh_size = mesh_size
        self.spline_distance = spline_distance
        self.coord_convention = coord_convention
        self.fill_value = fill_value
        self.target_device = device
        self.target_dtype = dtype

    def forward(
        self,
        points: Union[torch.Tensor, np.ndarray],
        values: Union[torch.Tensor, np.ndarray],
        point_weights: Optional[torch.Tensor] = None,
        return_density: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Project scattered points and features to grid using multi-level B-splines."""
        from .projection import project_scattered_to_grid

        return project_scattered_to_grid(
            points=points,
            values=values,
            grid_shape=self.grid_shape,
            domain_bounds=self.domain_bounds,
            point_weights=point_weights,
            coord_convention=self.coord_convention,
            fill_value=self.fill_value,
            return_density=return_density,
            method='bspline',
            number_of_fitting_levels=self.number_of_fitting_levels,
            mesh_size=self.mesh_size,
            spline_distance=self.spline_distance,
        )


def bspline_syn_scattered(
    fixed_points: Union[torch.Tensor, np.ndarray],
    moving_points: Union[torch.Tensor, np.ndarray],
    fixed_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    moving_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    fixed_landmarks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    moving_landmarks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    grid_res: Union[int, Tuple[int, ...]] = 128,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
    projection_method: Literal['gaussian', 'bspline'] = 'bspline',
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    regularizer: Literal['dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'] = 'dsti1',
    iterations: Union[int, Sequence[int]] = 50,
    fluid_sigma: float = 1.5,
    elastic_sigma: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Dict[str, Any]:
    """Execute SyN registration with ANTsTorch B-spline projection, regularization, or landmark initialization.

    Parameters
    ----------
    fixed_points : Tensor or ndarray of shape (N, d) or (B, N, d)
        Fixed point coordinates.
    moving_points : Tensor or ndarray of shape (M, d) or (B, M, d)
        Moving point coordinates.
    fixed_features : Tensor or ndarray, optional
    moving_features : Tensor or ndarray, optional
    fixed_landmarks : Tensor or ndarray, optional
        Landmark coordinates for initial B-spline landmark pre-warping.
    moving_landmarks : Tensor or ndarray, optional
        Corresponding moving landmark coordinates.
    grid_res : int or Tuple[int, ...], default 128
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
    projection_method : {'gaussian', 'bspline'}, default 'bspline'
    number_of_fitting_levels : int, default 4
    mesh_size : int or Sequence[int], default 1
    spline_distance : float or Sequence[float], optional
    regularizer : {'dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'}, default 'dsti1'
    iterations : int or Sequence[int], default 50
    fluid_sigma : float, default 1.5
    elastic_sigma : float, default 0.0
    device : str or torch.device, optional
    dtype : torch.dtype, default torch.float32
    **kwargs : Additional arguments forwarded to ScatteredRegistrationConfig.

    Returns
    -------
    dict
        Dictionary containing registration outputs:
        - 'warp_l2r', 'warp_r2l', 'warp_l2r_inv', 'warp_r2l_inv'
        - 'warped_moving_points', 'warped_fixed_points'
        - 'fixed_grid', 'moving_grid', 'warped_grid'
        - 'loss_history'
        - 'model'
    """
    from .solver import SyNScattered, ScatteredRegistrationConfig
    from .mapping import warp_scattered_coordinates

    pts_f_tensor = torch.as_tensor(fixed_points, dtype=dtype)
    dim = pts_f_tensor.shape[-1]

    initial_landmarks = None
    landmark_init = False
    if fixed_landmarks is not None and moving_landmarks is not None:
        initial_landmarks = (fixed_landmarks, moving_landmarks)
        landmark_init = True

    config = ScatteredRegistrationConfig(
        dim=dim,
        grid_res=grid_res,
        domain_bounds=domain_bounds,
        projection_method=projection_method,
        number_of_fitting_levels=number_of_fitting_levels,
        mesh_size=mesh_size,
        spline_distance=spline_distance,
        regularizer=regularizer,
        iterations=iterations,
        fluid_sigma=fluid_sigma,
        elastic_sigma=elastic_sigma,
        landmark_init=landmark_init,
        initial_landmarks=initial_landmarks,
        device=device,
        dtype=dtype,
        **kwargs,
    )

    model = SyNScattered(config)
    result = model.fit(
        fixed_points=fixed_points,
        moving_points=moving_points,
        fixed_features=fixed_features,
        moving_features=moving_features,
    )
    setattr(result, 'model', model)
    return result
