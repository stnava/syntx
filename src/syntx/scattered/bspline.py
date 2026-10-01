"""syntx.scattered.bspline -- ANTsTorch B-spline helpers for scattered-data registration.

All functions except ``has_antstorch`` need ``antstorch.bspline_flows``.

- ``has_antstorch``: import check.
- ``fit_bspline_landmark_warp``: dense displacement field from landmark pairs, by
  multi-level cubic B-spline approximation (``fit_bspline_displacement_field``). It is a
  smooth approximation, not an interpolation: landmarks are generally not hit exactly.
- ``apply_bspline_fluid_regularizer``: single-level B-spline fit of a dense field, used as
  the 'bspline' smoother in ``SyNScattered`` (as in ANTs BSplineSyN).
- ``BSplineScatteredProjector``: module wrapper of ``project_scattered_to_grid(method=
  'bspline')``.
- ``bspline_syn_scattered``: ``SyNScattered`` with B-spline defaults and optional landmark
  initialisation.

ITK domain construction (shared by these functions): size and spacing are taken in ITK
(x, y, z) order, origin = lower bound, spacing = extent / (n - 1).
"""

from typing import Optional, Sequence, Tuple, Union, Literal, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def has_antstorch() -> bool:
    """Return True if ``antstorch.bspline_flows`` (with the scattered-data fit) imports.

    Restores a previously selected matplotlib backend after the import. Only ImportError is
    caught. Same behaviour as ``projection.has_antstorch``.
    """
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
    """Raise ImportError if antstorch is not importable."""
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
    """Dense displacement field approximating landmark offsets u(x_f) = x_m - x_f.

    The offsets are placed at the fixed landmarks and fitted with
    ``antstorch.bspline_flows.fit_bspline_displacement_field`` (multi-level cubic B-spline
    approximation) over a grid spanning ``domain_bounds``. The field is defined on the fixed
    grid and points to the moving position, i.e. x + u(x) ~ moving location.

    Parameters
    ----------
    fixed_landmarks, moving_landmarks : Tensor or ndarray (N, d) or (B, N, d)
        Corresponding landmarks (same shape, else ValueError), in coordinate units.
    grid_shape : int or tuple of int
        Output grid size in tensor order; length must equal d.
    domain_bounds : tuple, 'auto' or None, default (-1.0, 1.0)
        Grid box: ``(lo, hi)``, ``(lo_seq, hi_seq)``, or 'auto' / None = extent of all
        landmarks +- 0.10 coordinate units.
    number_of_fitting_levels : int, default 4
        B-spline mesh doublings.
    mesh_size : int or sequence of int, default 1
        Spans of the coarsest mesh per axis (sequence in ``coord_convention`` order).
    spline_distance : float or sequence of float, optional
        Knot spacing in coordinate units; replaces ``mesh_size`` when given. A sequence is
        passed unchanged, i.e. in ITK (x, y, z) order.
    enforce_stationary_boundary : bool, default True
        Adds zero-displacement observations with weight 1e10 on the outer voxel layer
        (ITK behaviour), which pulls the whole fit toward zero near the box edge.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Component order of the landmarks and of the output vectors; 'xyz' also means
        component 0 runs along the last tensor axis.
    weights : Tensor or ndarray (N,) or (B, N), optional
        Per-landmark weights.
    device : str or torch.device, optional
        Defaults to the landmarks' device (CPU for NumPy).
    dtype : torch.dtype, default torch.float32

    Returns
    -------
    Tensor (B, *grid_shape, d), B = 1 for unbatched input. Displacements in coordinate units,
    components in ``coord_convention`` order; the tensor carries attributes
    ``is_physical = True`` and ``vector_convention = coord_convention``.
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
        size_itk = tuple(reversed(spatial_shape))      # the grid is tensor (z, y, x) either way
        origin_itk = tuple(float(min_b[d - 1 - i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[d - 1 - i] - min_b[d - 1 - i]).item()) for i in range(d))
    else:
        raise ValueError(f"Unknown coord_convention: '{coord_convention}', expected 'xyz' or 'zyx'")

    spacing_itk = tuple(extent / max(1, s - 1) for extent, s in zip(extent_itk, size_itk))
    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        mesh_size_itk = mesh_size_for_spline_distance(domain, tuple(reversed(spline_distance)) if (coord_convention == 'zyx' and not np.isscalar(spline_distance)) else spline_distance)
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
    """Smooth a dense vector field by a single-level cubic B-spline least-squares fit.

    Each batch element is fitted with ``fit_bspline_displacement_field(number_of_fitting_levels
    =1)`` on a mesh of ``mesh_size`` spans and resampled on the same grid. This is the
    BSplineSyN-style smoother used by ``SyNScattered`` (regularizer='bspline').

    Parameters
    ----------
    velocity_field : Tensor (*spatial, d) or (B, *spatial, d)
        Channels-last field; unbatched when ndim == len(grid_shape) + 1.
    grid_shape : tuple of int
        Spatial shape of the field (tensor order).
    domain_bounds : (Tensor (d,), Tensor (d,))
        (min, max) box in coordinate units; only sets the spacing used by
        ``spline_distance``.
    mesh_size : int or sequence of int, default 6
        Spans per axis (sequence in ``coord_convention`` order).
    spline_distance : float or sequence of float, optional
        Knot spacing in coordinate units; replaces ``mesh_size`` when given. A sequence is
        passed unchanged, i.e. in ITK (x, y, z) order.
    enforce_stationary_boundary : bool, default False
        If True, the outer voxel layer is fitted as zero with weight 1e10.
    coord_convention : str, default 'xyz'
        'xyz': components and axes as in ``projection``; any other value is treated as
        'zyx' (components flipped to ITK order before the fit and back after).

    Returns
    -------
    Tensor with the same shape as ``velocity_field``.
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
        size_itk = tuple(reversed(grid_shape))      # the grid is tensor (z, y, x) either way
        origin_itk = tuple(float(min_b[d - 1 - i].item()) for i in range(d))
        extent_itk = tuple(float((max_b[d - 1 - i] - min_b[d - 1 - i]).item()) for i in range(d))

    spacing_itk = tuple(extent / max(1, s - 1) for extent, s in zip(extent_itk, size_itk))
    domain = ImageDomain(size=size_itk, spacing=spacing_itk, origin=origin_itk)

    if spline_distance is not None:
        mesh_size_itk = mesh_size_for_spline_distance(domain, tuple(reversed(spline_distance)) if (coord_convention == 'zyx' and not np.isscalar(spline_distance)) else spline_distance)
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
    """Module that stores B-spline projection settings and calls ``project_scattered_to_grid``.

    Nothing is precomputed: each ``forward`` call runs
    ``project_scattered_to_grid(..., method='bspline')`` with the stored settings. Raises
    ImportError at construction if antstorch is missing.

    Parameters
    ----------
    grid_shape, domain_bounds, number_of_fitting_levels, mesh_size, spline_distance,
    coord_convention, fill_value :
        As in ``project_scattered_to_grid`` (B-spline engine).
    device, dtype :
        Inputs are moved / cast to these (``target_device`` / ``target_dtype``) before the
        fit, so the output has them; None keeps the input's.
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
        """B-spline projection of ``values`` at ``points``; returns (B, C, *spatial), or
        (values, density) if ``return_density`` (see ``project_scattered_to_grid``)."""
        from .projection import project_scattered_to_grid

        if self.target_device is not None or self.target_dtype is not None:
            dev = self.target_device
            points = torch.as_tensor(points).to(device=dev, dtype=self.target_dtype)
            values = torch.as_tensor(values).to(device=dev, dtype=self.target_dtype)
            if point_weights is not None:
                point_weights = torch.as_tensor(point_weights).to(device=dev, dtype=self.target_dtype)
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
) -> "ScatteredRegistrationResult":
    """Run ``SyNScattered`` with B-spline projection by default and optional landmark init.

    Builds a ``ScatteredRegistrationConfig`` from the arguments (``dim`` from the last axis of
    ``fixed_points``) and calls ``SyNScattered(config).fit(...)``. When both landmark sets are
    given, ``landmark_init=True`` and ``fit`` replaces the initial ``warp_r2l`` with
    ``fit_bspline_landmark_warp`` of the landmarks (see ``SyNScattered.fit``).

    Parameters
    ----------
    fixed_points : Tensor or ndarray (N, d) or (1, N, d)
    moving_points : Tensor or ndarray (M, d) or (1, M, d)
    fixed_features, moving_features : optional
        Per-point values; ones if None.
    fixed_landmarks, moving_landmarks : optional
        Corresponding landmarks (N_l, d); used only if both are given.
    grid_res : int or tuple of int, default 128
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
    projection_method : {'gaussian', 'bspline'}, default 'bspline'
    number_of_fitting_levels : int, default 4
    mesh_size : int or sequence of int, default 1
    spline_distance : float or sequence of float, optional
    regularizer : {'dsti1', 'dsti', 'sobolev', 'gaussian', 'bspline'}, default 'dsti1'
    iterations : int or sequence of int, default 50
    fluid_sigma : float, default 1.5
    elastic_sigma : float, default 0.0
    device : str or torch.device, optional
    dtype : torch.dtype, default torch.float32
        All of these are ``ScatteredRegistrationConfig`` fields; see there.
    **kwargs
        Other ``ScatteredRegistrationConfig`` fields (unknown names raise TypeError).

    Returns
    -------
    ScatteredRegistrationResult
        The ``fit`` result (attribute access, plus the dict-style keys listed in
        ``ScatteredRegistrationResult.__getitem__``); ``result.model`` is the fitted
        ``SyNScattered``.
    """
    from .solver import SyNScattered, ScatteredRegistrationConfig

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
