"""
syntx.scattered.transport — Differentiable Feature Pullback, Pushforward & Transport
====================================================================================

Provides mathematically rigorous, autograd-differentiable feature transport
utilities bridging Lagrangian scattered point sets and Eulerian regular lattices:

1. pullback_grid_to_scattered (F10):
   Evaluates Eulerian grid features at warped scattered point coordinates:
   Phi^* G_B(x_i) = G_B(Phi(x_i)) = G_B(x_i + u(x_i))

2. pushforward_scattered_to_grid (F11):
   Pushes forward scattered features onto an Eulerian regular grid lattice via
   warped coordinate transformation and Nadaraya-Watson kernel regression:
   Phi_* F_A(g_j) = project_scattered_to_grid(Phi(X_A), F_A, grid_shape, ...)

3. transport_scattered_to_scattered (F12):
   Directly transports features from source scattered points to target query
   coordinates through the diffeomorphism, supporting both direct Lagrangian
   Nadaraya-Watson kernel regression (Method A) and Eulerian grid bridge (Method B).
"""

from typing import Optional, Tuple, Union, Sequence, Literal
import torch
import torch.nn.functional as F

from syntx.spatial import reverse_components
from .mapping import warp_scattered_coordinates, _resolve_domain_bounds
from .projection import project_scattered_to_grid, compute_adaptive_sigma


def _standardize_grid_features(
    grid_features: torch.Tensor,
    spatial_dim: int,
    batch_size: int,
    channel_dim: Optional[int] = 1,
) -> Tuple[torch.Tensor, bool, bool]:
    """Standardize input grid tensor to channels-first (B, C, *spatial).

    Returns
    -------
    grid_cf : torch.Tensor
        Standardized tensor of shape (B, C, *spatial).
    was_unbatched : bool
        True if the input lacked an explicit batch dimension.
    was_unchanneled : bool
        True if the input lacked an explicit channel dimension.
    """
    dim = spatial_dim
    g_dim = grid_features.dim()
    was_unbatched = False
    was_unchanneled = False

    if g_dim == dim:
        # Scalar unbatched: (*spatial) -> (1, 1, *spatial)
        was_unbatched = True
        was_unchanneled = True
        grid_cf = grid_features.unsqueeze(0).unsqueeze(0)
    elif g_dim == dim + 1:
        if channel_dim in (-1, g_dim - 1):
            # Channels-last unbatched: (*spatial, C) -> (1, C, *spatial)
            was_unbatched = True
            grid_cf = torch.movedim(grid_features, -1, 0).unsqueeze(0)
        elif channel_dim is None or channel_dim == 0:
            # Explicitly batched scalar: (B, *spatial) -> (B, 1, *spatial)
            was_unchanneled = True
            grid_cf = grid_features.unsqueeze(1)
        else:
            # Channels-first unbatched: (C, *spatial) -> (1, C, *spatial)
            was_unbatched = True
            grid_cf = grid_features.unsqueeze(0)
    elif g_dim == dim + 2:
        if channel_dim in (-1, g_dim - 1):
            # Channels-last batched: (B, *spatial, C) -> (B, C, *spatial)
            grid_cf = torch.movedim(grid_features, -1, 1)
        else:
            # (B, C, *spatial) - standard PyTorch format
            grid_cf = grid_features
    else:
        raise ValueError(
            f"Unsupported grid_features dimension {g_dim} for spatial dimension {dim}. "
            f"Expected {dim} (scalar), {dim + 1} (batched scalar or unbatched multi-channel), "
            f"or {dim + 2} (batched multi-channel)."
        )

    return grid_cf, was_unbatched, was_unchanneled


def pullback_grid_to_scattered(
    grid_features: torch.Tensor,
    coords: torch.Tensor,
    displacement_field: Optional[torch.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = True,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = None,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    channel_dim: Optional[int] = 1,
) -> torch.Tensor:
    """Pull back Eulerian grid features to scattered coordinates under diffeomorphism.

    Evaluates:
        Phi^* G_B(x_i) = G_B(Phi(x_i)) = G_B(x_i + u(x_i))

    If displacement_field is None, evaluates G_B directly at coords (identity warp).

    Parameters
    ----------
    grid_features : torch.Tensor
        Eulerian feature tensor of shape (*spatial), (C, *spatial), (B, *spatial),
        or (B, C, *spatial). If channels-last, set channel_dim=-1.
    coords : torch.Tensor
        Scattered coordinates of shape (N, d) or (B, N, d).
    displacement_field : torch.Tensor, optional
        Eulerian displacement field of shape (*spatial, d) or (B, *spatial, d).
        If None, evaluates grid at unwarped coordinates.
    mode : str, default 'bilinear'
        Interpolation mode: 'bilinear' (linear for 2D/3D) or 'nearest'.
    padding_mode : str, default 'border'
        Padding mode: 'border', 'zeros', or 'reflection'.
    align_corners : bool, default True
        Coordinate alignment convention for PyTorch grid_sample.
    domain_bounds : tuple or 'auto', optional
        Physical bounding box of the grid. If provided, coords are assumed to be
        in physical units and will be affine-normalized to [-1, 1]^d before sampling.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate axis mapping convention matching projection.py.
    channel_dim : int, optional, default 1
        Axis of feature channels in `grid_features`. 1 for channels-first (default),
        -1 for channels-last, 0 or None for batched scalar.

    Returns
    -------
    torch.Tensor
        Sampled features at scattered coordinates, with shape (N, C), (B, N, C),
        or (N,) / (B, N) if input was an unchanneled scalar.
    """
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords, dtype=torch.float32)
    if not isinstance(grid_features, torch.Tensor):
        grid_features = torch.as_tensor(grid_features, dtype=coords.dtype, device=coords.device)
    else:
        grid_features = grid_features.to(device=coords.device)

    unbatched_coords = (coords.dim() == 2)
    coords_b = coords.unsqueeze(0) if unbatched_coords else coords
    B_coords, N, d = coords_b.shape

    # 1. Warp coordinates if displacement field is provided
    if displacement_field is not None:
        warped_coords = warp_scattered_coordinates(
            coords=coords_b,
            displacement_field=displacement_field,
            direction='forward',
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            domain_bounds=domain_bounds,
            coord_convention=coord_convention,
        )
    else:
        warped_coords = coords_b

    # Ensure warped_coords is 3D (B_warp, N, d)
    if warped_coords.dim() == 2:
        warped_coords = warped_coords.unsqueeze(0)
    B_warp = warped_coords.shape[0]

    # 2. Normalize to [-1, 1]^d if physical domain_bounds are specified
    bounds = _resolve_domain_bounds(domain_bounds, warped_coords, d)
    if bounds is not None:
        min_b, max_b = bounds
        span = (max_b - min_b).clamp_min(1e-8)
        query_norm = 2.0 * (warped_coords - min_b) / span - 1.0
    else:
        query_norm = warped_coords

    # 3. Handle coordinate convention ('xyz' vs 'zyx')
    if coord_convention == 'zyx':
        query_norm = reverse_components(query_norm)
    elif coord_convention != 'xyz':
        raise ValueError(f"Unknown coord_convention: '{coord_convention}', expected 'xyz' or 'zyx'")

    # 4. Standardize grid_features to (B_grid, C, *spatial)
    grid_cf, was_unbatched_grid, was_unchanneled_grid = _standardize_grid_features(
        grid_features, spatial_dim=d, batch_size=B_warp, channel_dim=channel_dim
    )
    B_grid = grid_cf.shape[0]
    was_any_batched = (
        (not unbatched_coords)
        or (displacement_field is not None and displacement_field.dim() == d + 2)
        or (not was_unbatched_grid)
    )

    # Resolve effective batch size B and broadcast
    if B_warp == B_grid:
        B = B_warp
    elif B_warp == 1 and B_grid > 1:
        B = B_grid
        query_norm = query_norm.expand(B, -1, -1)
    elif B_grid == 1 and B_warp > 1:
        B = B_warp
        grid_cf = grid_cf.expand(B, -1, *([-1] * d))
    else:
        raise ValueError(
            f"Batch dimension mismatch: grid_features has batch {B_grid}, "
            f"coordinates/displacement has batch {B_warp}"
        )

    # Match dtypes for grid_sample
    if grid_cf.dtype != query_norm.dtype:
        grid_cf = grid_cf.to(query_norm.dtype)

    # 5. Build singleton query grid: (B, 1, ..., 1, N, d)
    query_tensor = query_norm.reshape(B, *([1] * (d - 1)), N, d)

    # 6. Sample grid differentiably
    sampled = F.grid_sample(
        grid_cf,
        query_tensor,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )  # (B, C, 1, ..., 1, N)

    C = grid_cf.shape[1]
    out = sampled.view(B, C, N).movedim(1, -1)  # (B, N, C)

    # 7. Formatting unravelling
    if was_unchanneled_grid and C == 1:
        out = out.squeeze(-1)  # (B, N)
    if not was_any_batched and B == 1:
        out = out.squeeze(0)   # (N, C) or (N,)

    return out


def pushforward_scattered_to_grid(
    coords: torch.Tensor,
    features: torch.Tensor,
    displacement_field: Optional[torch.Tensor] = None,
    grid_shape: Union[int, Tuple[int, ...]] = 64,
    sigma: Union[float, Sequence[float], torch.Tensor, str] = 0.03,
    domain_mask: Optional[torch.Tensor] = None,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
    point_weights: Optional[torch.Tensor] = None,
    direction: Literal['forward', 'backward'] = 'forward',
    epsilon: float = 1e-8,
    chunk_size: int = 0,
    target_memory_mb: float = 256.0,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    fill_value: float = 0.0,
    return_density: bool = False,
    mode: str = 'bilinear',
    align_corners: bool = True,
    method: Literal['gaussian', 'bspline'] = 'gaussian',
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Push forward scattered features onto a regular Eulerian grid lattice.

    Warps source coordinates through the displacement field, then projects the
    transported point features onto the grid via differentiable Nadaraya-Watson
    normalized Gaussian kernel regression.

    Parameters
    ----------
    coords : torch.Tensor
        Source coordinates of shape (N, d) or (B, N, d).
    features : torch.Tensor
        Source features of shape (N,), (N, C), (B, N), or (B, N, C).
    displacement_field : torch.Tensor, optional
        Eulerian displacement field of shape (*spatial, d) or (B, *spatial, d).
        If None, projects unwarped coordinates directly onto the grid.
    grid_shape : int or Tuple[int, ...], default 64
        Target Eulerian grid resolution.
    sigma : float, sequence of float, Tensor, or 'auto', default 0.03
        Gaussian kernel bandwidth.
    domain_mask : torch.Tensor, optional
        Eulerian continuous or binary domain mask.
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
        Grid coordinate bounding box.
    point_weights : torch.Tensor, optional
        Confidence weights per point.
    direction : {'forward', 'backward'}, default 'forward'
        Coordinate warp direction.
    epsilon : float, default 1e-8
        Numerical stability denominator floor.
    chunk_size : int, default 0
        Chunk size in voxels (0 = auto-calculate from memory ceiling).
    target_memory_mb : float, default 256.0
        Memory ceiling for GEMM operations in megabytes.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate mapping convention.
    fill_value : float, default 0.0
        Value for empty voxels.
    return_density : bool, default False
        If True, returns (grid_features, density).
    mode : str, default 'bilinear'
        Interpolation mode for coordinate warping.
    align_corners : bool, default True
        Coordinate alignment convention for coordinate warping.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Projected Eulerian feature tensor of shape (B, C, *spatial), and
        optionally the density tensor of shape (B, 1, *spatial).
    """
    if displacement_field is not None:
        warped_coords = warp_scattered_coordinates(
            coords=coords,
            displacement_field=displacement_field,
            direction=direction,
            mode=mode,
            align_corners=align_corners,
            domain_bounds=domain_bounds,
            coord_convention=coord_convention,
        )
    else:
        warped_coords = coords

    # Batch broadcasting for coordinates and features
    B_coords = 1 if warped_coords.dim() == 2 else warped_coords.shape[0]
    N_pts = warped_coords.shape[-2]

    # Determine batch size from features
    is_batched_scalar = False
    if features.dim() == 3:
        B_feat = features.shape[0]
    elif features.dim() == 2:
        if features.shape[1] == N_pts and features.shape[0] != N_pts:
            is_batched_scalar = True
            features = features.unsqueeze(-1)
            B_feat = features.shape[0]
        elif features.shape[0] == N_pts and features.shape[1] == N_pts:
            is_batched_scalar = True
            features = features.unsqueeze(-1)
            B_feat = N_pts
        elif features.shape[0] == N_pts and features.shape[1] != N_pts:
            B_feat = 1
        else:
            B_feat = 1
    else:
        B_feat = 1

    # Resolve effective batch size
    if B_coords == B_feat:
        B = B_coords
    elif B_coords == 1 and B_feat > 1:
        B = B_feat
    elif B_feat == 1 and B_coords > 1:
        B = B_coords
    else:
        raise ValueError(
            f"Batch dimension mismatch in pushforward_scattered_to_grid: "
            f"coordinates has batch {B_coords}, features has batch {B_feat}"
        )

    # When features is batched scalar (B, N, 1), ensure warped_coords is promoted to (B, N, d) even when B=1
    if is_batched_scalar and warped_coords.dim() == 2:
        warped_coords = warped_coords.unsqueeze(0).expand(B, -1, -1)
    elif B > 1:
        if warped_coords.dim() == 2:
            warped_coords = warped_coords.unsqueeze(0).expand(B, -1, -1)
        elif warped_coords.shape[0] == 1:
            warped_coords = warped_coords.expand(B, -1, -1)

        # Expand features if passed with singleton batch in 3D
        if features.dim() == 3 and features.shape[0] == 1:
            features = features.expand(B, -1, -1)

        # Expand point_weights if provided with singleton batch
        if point_weights is not None:
            if point_weights.dim() == 3 and point_weights.shape[0] == 1:
                point_weights = point_weights.expand(B, -1, -1)

    return project_scattered_to_grid(
        points=warped_coords,
        values=features,
        grid_shape=grid_shape,
        domain_bounds=domain_bounds,
        sigma=sigma,
        mask=domain_mask,
        point_weights=point_weights,
        epsilon=epsilon,
        chunk_size=chunk_size,
        target_memory_mb=target_memory_mb,
        coord_convention=coord_convention,
        fill_value=fill_value,
        return_density=return_density,
        method=method,
        number_of_fitting_levels=number_of_fitting_levels,
        mesh_size=mesh_size,
        spline_distance=spline_distance,
    )


def _direct_scattered_kernel_regression(
    coords_src: torch.Tensor,       # (B, N_src, d)
    features_src: torch.Tensor,     # (B, N_src, C)
    coords_tgt: torch.Tensor,       # (B, N_tgt, d)
    sigma_t: torch.Tensor,          # (d,)
    point_weights: Optional[torch.Tensor] = None,  # (B, N_src, 1) or None
    epsilon: float = 1e-8,
    chunk_size: int = 0,
    target_memory_mb: float = 256.0,
    fill_value: float = 0.0,
    return_density: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Direct Lagrangian Nadaraya-Watson kernel regression between point clouds."""
    B, N_src, d = coords_src.shape
    _, N_tgt, _ = coords_tgt.shape
    C = features_src.shape[-1]

    elem_bytes = coords_src.element_size()
    if chunk_size <= 0:
        chunk_size = max(1, int(target_memory_mb * (1024 ** 2) / (4 * max(1, N_src) * elem_bytes)))
    chunk_size = min(N_tgt, max(1, chunk_size))

    batch_outputs = []
    batch_densities = []

    for b in range(B):
        # Center coordinates relative to source domain midpoint to prevent float32 cancellation
        if N_src > 0:
            domain_center = 0.5 * (coords_src[b].amin(dim=0) + coords_src[b].amax(dim=0))
        else:
            domain_center = torch.zeros(d, device=coords_src.device, dtype=coords_src.dtype)

        X_s = (coords_src[b] - domain_center) / sigma_t     # (N_src, d)
        X_t = (coords_tgt[b] - domain_center) / sigma_t     # (N_tgt, d)
        F_s = features_src[b]                               # (N_src, C)

        NX_s = (X_s ** 2).sum(dim=-1, keepdim=True).t()    # (1, N_src)

        num_chunks = []
        den_chunks = []

        for t_start in range(0, N_tgt, chunk_size):
            t_end = min(t_start + chunk_size, N_tgt)
            Y_c = X_t[t_start:t_end]                        # (chunk, d)
            NY_c = (Y_c ** 2).sum(dim=-1, keepdim=True)     # (chunk, 1)

            # Vectorized GEMM distance expansion with clamp_min
            d2_raw = NY_c - 2.0 * (Y_c @ X_s.t()) + NX_s
            d2 = torch.clamp_min(d2_raw, 0.0)
            W = torch.exp(-0.5 * d2)                        # (chunk, N_src)

            if point_weights is not None:
                pw_b = point_weights[b]
                if pw_b.dim() == 2:
                    W = W * pw_b.t()
                elif pw_b.dim() == 1:
                    W = W * pw_b.unsqueeze(0)

            num_chunks.append(W @ F_s)                      # (chunk, C)
            den_chunks.append(W.sum(dim=-1, keepdim=True))  # (chunk, 1)

        num = torch.cat(num_chunks, dim=0)                  # (N_tgt, C)
        den = torch.cat(den_chunks, dim=0)                  # (N_tgt, 1)

        tgt_vals = num / (den + epsilon)                    # (N_tgt, C)

        if fill_value != 0.0:
            empty_mask = den < (epsilon * 10.0)
            tgt_vals = torch.where(empty_mask, torch.full_like(tgt_vals, fill_value), tgt_vals)

        batch_outputs.append(tgt_vals)
        batch_densities.append(den)

    out = torch.stack(batch_outputs, dim=0)       # (B, N_tgt, C)
    density = torch.stack(batch_densities, dim=0) # (B, N_tgt, 1)

    if return_density:
        return out, density
    return out


def transport_scattered_to_scattered(
    coords_source: torch.Tensor,
    features_source: torch.Tensor,
    coords_target: torch.Tensor,
    displacement_field: Optional[torch.Tensor] = None,
    sigma: Union[float, Sequence[float], torch.Tensor, str] = 0.03,
    direction: Literal['forward', 'backward'] = 'forward',
    method: Literal['direct', 'grid_bridge'] = 'direct',
    grid_shape: Optional[Union[int, Tuple[int, ...]]] = None,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = None,
    point_weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
    chunk_size: int = 0,
    target_memory_mb: float = 256.0,
    fill_value: float = 0.0,
    mode: str = 'bilinear',
    align_corners: bool = True,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    return_density: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Transport features from source scattered points to target query coordinates.

    Parameters
    ----------
    coords_source : torch.Tensor
        Source coordinates of shape (N_src, d) or (B, N_src, d).
    features_source : torch.Tensor
        Source features of shape (N_src,), (N_src, C), (B, N_src), or (B, N_src, C).
    coords_target : torch.Tensor
        Target query coordinates of shape (N_tgt, d) or (B, N_tgt, d).
    displacement_field : torch.Tensor, optional
        Eulerian displacement field. If None, performs direct kernel interpolation
        between unwarped coordinate sets.
    sigma : float, sequence of float, Tensor, or 'auto', default 0.03
        Gaussian kernel bandwidth.
    direction : {'forward', 'backward'}, default 'forward'
        Warp direction.
    method : {'direct', 'grid_bridge'}, default 'direct'
        'direct': Direct Lagrangian Nadaraya-Watson regression in target domain.
        'grid_bridge': Rasterizes to Eulerian grid then samples at target points.
    grid_shape : int or tuple, optional
        Required when method='grid_bridge'. Defaults to 64 if not specified.
    domain_bounds : tuple or 'auto', optional
        Domain bounds for physical coordinate spaces.
    point_weights : torch.Tensor, optional
        Confidence weights for source points.
    epsilon : float, default 1e-8
        Numerical stability denominator floor.
    chunk_size : int, default 0
        Chunk size for distance matrix evaluation.
    target_memory_mb : float, default 256.0
        Target memory ceiling in megabytes.
    fill_value : float, default 0.0
        Value for target points with zero density.
    mode : str, default 'bilinear'
        Interpolation mode.
    align_corners : bool, default True
        Coordinate alignment convention.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate convention.
    return_density : bool, default False
        If True, returns (features, density).

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Transported features at target coordinates, with shape (N_tgt, C),
        (B, N_tgt, C), or scalar equivalents.
    """
    if not isinstance(coords_source, torch.Tensor):
        coords_source = torch.as_tensor(coords_source, dtype=torch.float32)
    if not isinstance(features_source, torch.Tensor):
        features_source = torch.as_tensor(features_source, dtype=coords_source.dtype, device=coords_source.device)
    else:
        features_source = features_source.to(device=coords_source.device, dtype=coords_source.dtype)
    if not isinstance(coords_target, torch.Tensor):
        coords_target = torch.as_tensor(coords_target, dtype=coords_source.dtype, device=coords_source.device)
    else:
        coords_target = coords_target.to(device=coords_source.device, dtype=coords_source.dtype)

    unbatched_src = (coords_source.dim() == 2)
    unbatched_tgt = (coords_target.dim() == 2)
    coords_src_b = coords_source.unsqueeze(0) if unbatched_src else coords_source
    coords_tgt_b = coords_target.unsqueeze(0) if unbatched_tgt else coords_target

    B, N_src, d = coords_src_b.shape
    B_tgt, N_tgt, _ = coords_tgt_b.shape

    if B != B_tgt:
        if B == 1 and B_tgt > 1:
            coords_src_b = coords_src_b.expand(B_tgt, -1, -1)
            B = B_tgt
        elif B_tgt == 1 and B > 1:
            coords_tgt_b = coords_tgt_b.expand(B, -1, -1)
            B_tgt = B
        else:
            raise ValueError(f"Batch dimension mismatch: source {B} vs target {B_tgt}")

    # Standardize features to (B, N_src, C)
    was_unchanneled_feat = False
    if features_source.dim() == 1:
        was_unchanneled_feat = True
        features_b = features_source.view(1, N_src, 1).expand(B, -1, -1)
    elif features_source.dim() == 2:
        if features_source.shape[1] == N_src and features_source.shape[0] != N_src:
            was_unchanneled_feat = True
            features_b = features_source.unsqueeze(-1)
        elif not unbatched_src and features_source.shape == (B, N_src):
            was_unchanneled_feat = True
            features_b = features_source.unsqueeze(-1)
        elif features_source.shape[0] == N_src:
            features_b = features_source.unsqueeze(0).expand(B, -1, -1)
        else:
            raise ValueError(
                f"Cannot align features_source shape {features_source.shape} with N_src={N_src}"
            )
    elif features_source.dim() == 3:
        features_b = features_source
    else:
        raise ValueError(f"Unsupported features_source dimension {features_source.dim()}")

    was_any_batched = (
        (not unbatched_src)
        or (not unbatched_tgt)
        or (displacement_field is not None and displacement_field.dim() == d + 2)
        or (features_source.dim() == 3)
        or (features_source.dim() == 2 and features_source.shape[1] == N_src and features_source.shape[0] != N_src)
    )

    # 1. Warp source coordinates to target space
    if displacement_field is not None:
        warped_src = warp_scattered_coordinates(
            coords=coords_src_b,
            displacement_field=displacement_field,
            direction=direction,
            mode=mode,
            align_corners=align_corners,
            domain_bounds=domain_bounds,
            coord_convention=coord_convention,
        )
    else:
        warped_src = coords_src_b

    # Resolve effective batch size after coordinate warping
    B_eff = warped_src.shape[0]

    # Reconcile batch sizes across warped_src, features_b, and coords_tgt_b
    if features_b.shape[0] > 1:
        if B_eff == 1:
            B_eff = features_b.shape[0]
            warped_src = warped_src.expand(B_eff, -1, -1)
        elif features_b.shape[0] != B_eff:
            raise ValueError(
                f"Batch dimension mismatch: features have batch {features_b.shape[0]}, "
                f"warped coordinates have batch {B_eff}"
            )
    if features_b.shape[0] == 1 and B_eff > 1:
        features_b = features_b.expand(B_eff, -1, -1)

    if coords_tgt_b.shape[0] == 1 and B_eff > 1:
        coords_tgt_b = coords_tgt_b.expand(B_eff, -1, -1)
    elif coords_tgt_b.shape[0] != B_eff:
        raise ValueError(
            f"Batch dimension mismatch: target coordinates have batch {coords_tgt_b.shape[0]}, "
            f"expected {B_eff}"
        )

    # Standardize point_weights to (B_eff, N_src, 1) if provided
    if point_weights is not None:
        if not isinstance(point_weights, torch.Tensor):
            point_weights = torch.as_tensor(point_weights, device=coords_source.device, dtype=coords_source.dtype)
        else:
            point_weights = point_weights.to(device=coords_source.device, dtype=coords_source.dtype)
        if point_weights.dim() == 1:
            if point_weights.shape[0] != N_src:
                raise ValueError(f"point_weights length {point_weights.shape[0]} does not match N_src {N_src}")
            point_weights = point_weights.view(1, N_src, 1).expand(B_eff, -1, -1)
        elif point_weights.dim() == 2:
            if point_weights.shape == (N_src, 1):
                point_weights = point_weights.unsqueeze(0).expand(B_eff, -1, -1)
            elif point_weights.shape == (1, N_src):
                point_weights = point_weights.unsqueeze(-1).expand(B_eff, -1, -1)
            elif point_weights.shape == (B_eff, N_src):
                point_weights = point_weights.unsqueeze(-1)
            elif point_weights.shape[0] == 1 and point_weights.shape[1] == N_src:
                point_weights = point_weights.unsqueeze(-1).expand(B_eff, -1, -1)
            elif unbatched_src and point_weights.shape[0] == N_src:
                point_weights = point_weights.view(1, N_src, -1).expand(B_eff, -1, -1)
            else:
                raise ValueError(
                    f"Cannot align point_weights shape {point_weights.shape} with ({B_eff}, {N_src})"
                )
        elif point_weights.dim() == 3:
            if point_weights.shape[0] == 1 and B_eff > 1:
                point_weights = point_weights.expand(B_eff, -1, -1)
            elif point_weights.shape[0] != B_eff or point_weights.shape[1] != N_src:
                raise ValueError(
                    f"point_weights shape {point_weights.shape} does not match ({B_eff}, {N_src})"
                )

    # 2. Execute transport via chosen method
    if method == 'direct':
        # Resolve sigma
        if isinstance(sigma, str) and sigma == 'auto':
            s_val = compute_adaptive_sigma(warped_src)
            sigma_t = torch.full((d,), s_val, device=coords_source.device, dtype=coords_source.dtype)
        elif isinstance(sigma, (int, float)):
            sigma_t = torch.full((d,), float(sigma), device=coords_source.device, dtype=coords_source.dtype)
        else:
            sigma_t = torch.as_tensor(sigma, device=coords_source.device, dtype=coords_source.dtype)

        res = _direct_scattered_kernel_regression(
            coords_src=warped_src,
            features_src=features_b,
            coords_tgt=coords_tgt_b,
            sigma_t=sigma_t,
            point_weights=point_weights,
            epsilon=epsilon,
            chunk_size=chunk_size,
            target_memory_mb=target_memory_mb,
            fill_value=fill_value,
            return_density=return_density,
        )
        if return_density:
            out, density = res
        else:
            out = res
            density = None

    elif method == 'grid_bridge':
        if grid_shape is None:
            grid_shape = 64

        if domain_bounds is None:
            # Auto-detect if coordinates exceed [-1, 1] without autograd scalar conversion warnings
            c_min = min(float(warped_src.detach().amin()), float(coords_tgt_b.detach().amin()))
            c_max = max(float(warped_src.detach().amax()), float(coords_tgt_b.detach().amax()))
            gb_bounds = 'auto' if (c_min < -1.05 or c_max > 1.05) else (-1.0, 1.0)
        else:
            gb_bounds = domain_bounds

        # Step A: Push forward to intermediate grid
        grid_b = pushforward_scattered_to_grid(
            coords=warped_src,
            features=features_b,
            displacement_field=None,
            grid_shape=grid_shape,
            sigma=sigma,
            domain_bounds=gb_bounds,
            point_weights=point_weights,
            epsilon=epsilon,
            chunk_size=chunk_size,
            target_memory_mb=target_memory_mb,
            coord_convention=coord_convention,
            fill_value=fill_value,
            return_density=False,
        )
        # Step B: Pull back from intermediate grid to target points
        out = pullback_grid_to_scattered(
            grid_features=grid_b,
            coords=coords_tgt_b,
            displacement_field=None,
            mode=mode,
            align_corners=align_corners,
            domain_bounds=gb_bounds,
            coord_convention=coord_convention,
        )
        density = None
    else:
        raise ValueError(f"Unknown transport method: '{method}', expected 'direct' or 'grid_bridge'")

    # Formatting unravelling
    if was_unchanneled_feat and out.shape[-1] == 1:
        out = out.squeeze(-1)
    if not was_any_batched and B_eff == 1:
        out = out.squeeze(0)
        if density is not None:
            density = density.squeeze(0)

    if return_density and density is not None:
        return out, density
    return out
