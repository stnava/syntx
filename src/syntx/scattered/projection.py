"""
syntx.scattered.projection — Differentiable Scattered-to-Grid Projection
========================================================================

High-performance, autograd-differentiable Nadaraya-Watson normalized Gaussian
kernel regression mapping Lagrangian scattered coordinates to Eulerian regular
grids across arbitrary spatial dimensions.

Key Features & Numerical Safeguards:
- Vectorized GEMM distance expansion D^2 = NY - 2 * (Y @ X^T) + NX with
  mandatory torch.clamp_min(..., 0.0) eliminating negative roundoff noise.
- Coordinate centering relative to domain midpoint preventing catastrophic
  float32 cancellation on physical millimeter/DICOM coordinate systems.
- Dynamic auto-chunking memory ceiling (<= 256 MB) to prevent OOM on dense 3D grids.
- Continuous C^inf Nadaraya-Watson division with positive epsilon floor.
- Strict spatial dimension and coordinate ordering support ('xyz' Cartesian/ITK
  and 'zyx' tensor-index conventions).
- Pre-cached Eulerian grid geometry and norms via ScatteredProjector(nn.Module).
- Backward compatibility wrapper matching sulceye.flatmap.utils signature.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence, Literal
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ProjectionConfig:
    """Configuration for differentiable scattered-to-grid projection.

    Parameters
    ----------
    grid_shape : int or Tuple[int, ...]
        Output grid resolution, e.g. 64, (64, 64), or (32, 64, 64).
    domain_bounds : tuple or str, optional
        Physical/coordinate bounds of the grid. Can be:
        - (-1.0, 1.0): isotropic scalar range
        - ([min_x, min_y], [max_x, max_y]): per-axis bounding box
        - 'auto': automatically derived from point extent with 3*sigma padding
    sigma : float, sequence of float, or 'auto'
        Gaussian kernel bandwidth.
    sigma_scale : float
        Multiplier when deriving bandwidth from grid spacing or k-NN distance.
    epsilon : float
        Numerical stability denominator floor.
    chunk_size : int
        Number of grid voxels per chunk (0 = auto-calculate from memory budget).
    target_memory_mb : float
        Target memory ceiling for intermediate GEMM tensors in megabytes.
    coord_convention : {'xyz', 'zyx'}
        Coordinate mapping convention.
    fill_value : float
        Value assigned to empty voxels (density < epsilon * 10). If 0.0, uses
        pure C^inf smooth division without branching.
    kernel : {'gaussian'}
        Kernel type.
    method : {'gaussian', 'bspline'}, default 'gaussian'
        Projection engine: 'gaussian' (Nadaraya-Watson kernel regression) or
        'bspline' (ANTsTorch multi-level cubic B-splines).
    number_of_fitting_levels : int, default 4
        Number of hierarchical refinement levels for B-spline projection.
    mesh_size : int or sequence of int, default 1
        Base control point lattice spans per axis for B-spline projection.
    spline_distance : float or sequence of float, optional
        Physical knot spacing in mm for B-spline projection (overrides mesh_size).
    """
    grid_shape: Union[int, Tuple[int, ...]] = 64
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0)
    sigma: Union[float, Sequence[float], str] = 0.03
    sigma_scale: float = 1.5
    epsilon: float = 1e-8
    chunk_size: int = 0
    target_memory_mb: float = 256.0
    coord_convention: Literal['xyz', 'zyx'] = 'xyz'
    fill_value: float = 0.0
    kernel: Literal['gaussian'] = 'gaussian'
    method: Literal['gaussian', 'bspline'] = 'gaussian'
    number_of_fitting_levels: int = 4
    mesh_size: Union[int, Sequence[int]] = 1
    spline_distance: Optional[Union[float, Sequence[float]]] = None


def has_antstorch() -> bool:
    """Check if antstorch package (antstorch.bspline_flows) is available."""
    try:
        import sys
        orig_backend = None
        if 'matplotlib' in sys.modules:
            import matplotlib
            orig_backend = matplotlib.get_backend()
        import antstorch.bspline_flows  # noqa: F401
        if orig_backend and orig_backend != 'Agg':
            try:
                import matplotlib
                matplotlib.use(orig_backend)
            except Exception:
                pass
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def compute_adaptive_sigma(
    points: Union[torch.Tensor, np.ndarray],
    k: int = 6,
    scale: float = 2.0,
    floor: float = 0.01,
    cap: float = 0.20,
) -> float:
    """Compute adaptive Gaussian bandwidth from median k-NN distance.

    Adapts the kernel standard deviation to local point density using the median
    k-nearest-neighbor distance scaled by `scale`.

    Parameters
    ----------
    points : (N, d) or (B, N, d) array or tensor
        Scattered coordinates.
    k : int, default 6
        Number of nearest neighbors to consider (excluding self).
    scale : float, default 2.0
        Multiplier on median k-NN distance.
    floor : float, default 0.01
        Minimum bandwidth clamp.
    cap : float, default 0.20
        Maximum bandwidth clamp.

    Returns
    -------
    float
        Adaptive sigma value clamped to [floor, cap].
    """
    if isinstance(points, np.ndarray):
        pts = torch.from_numpy(points)
    else:
        pts = points.detach()
    if pts.dim() == 3:
        pts = pts[0]
    pts = pts.float()
    N = pts.shape[0]
    if N <= 1:
        return floor

    # Subsample points if N is large for speed
    if N > 2000:
        indices = torch.randperm(N, device=pts.device)[:2000]
        sample = pts[indices]
    else:
        sample = pts

    dists = torch.cdist(sample, sample)
    k_val = min(k + 1, dists.shape[-1])
    topk_vals, _ = torch.topk(dists, k=k_val, dim=-1, largest=False)
    # topk_vals[:, 0] is self distance (0.0); take columns 1..k
    nn_dists = topk_vals[:, 1:]
    median_nn = float(torch.median(nn_dists).item())
    sigma = float(np.clip(scale * median_nn, floor, cap))
    return sigma


def _build_eulerian_grid(
    grid_shape: Tuple[int, ...],
    domain_bounds: Tuple[torch.Tensor, torch.Tensor],
    coord_convention: str = 'xyz',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Construct flattened Eulerian grid coordinates matching spatial tensor ordering.

    Parameters
    ----------
    grid_shape : Tuple[int, ...]
        Spatial dimensions of the grid, e.g. (H, W) or (D, H, W).
    domain_bounds : Tuple[torch.Tensor, torch.Tensor]
        (min_coords, max_coords) where each is a 1D tensor of shape (d,).
    coord_convention : {'xyz', 'zyx'}
        'xyz': Cartesian/ITK convention where coordinate 0 is X (cols), 1 is Y (rows),
               2 is Z (depth). Output spatial tensor has shape (H, W) or (D, H, W).
        'zyx': Tensor index convention where coordinate index directly matches
               spatial axis index.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    grid_coords : (G, d) torch.Tensor
        Flattened Eulerian grid coordinates.
    spatial_shape : Tuple[int, ...]
        Output grid shape (H, W) or (D, H, W).
    """
    dim = len(grid_shape)
    min_b, max_b = domain_bounds

    if coord_convention == 'xyz':
        # In spatial tensor (D, H, W) or (H, W):
        # Spatial axis k corresponds to coordinate component c_k = dim - 1 - k.
        # X is spatial axis dim - 1 (cols), Y is spatial axis dim - 2 (rows), Z is spatial axis dim - 3 (depth).
        spatial_shape = grid_shape
        axes_coords = []
        for d_idx in range(dim):
            spatial_axis = dim - 1 - d_idx
            num_voxels = spatial_shape[spatial_axis]
            coords_1d = torch.linspace(min_b[d_idx], max_b[d_idx], num_voxels, device=device, dtype=dtype)
            axes_coords.append((spatial_axis, coords_1d))
        axes_coords.sort(key=lambda item: item[0])
        mesh_inputs = [item[1] for item in axes_coords]
        mesh = torch.meshgrid(*mesh_inputs, indexing='ij')
        xyz_grids = [mesh[dim - 1 - d_idx] for d_idx in range(dim)]
        grid_coords = torch.stack(xyz_grids, dim=-1)
    elif coord_convention == 'zyx':
        spatial_shape = grid_shape
        axes_coords = [
            torch.linspace(min_b[d_idx], max_b[d_idx], spatial_shape[d_idx], device=device, dtype=dtype)
            for d_idx in range(dim)
        ]
        mesh = torch.meshgrid(*axes_coords, indexing='ij')
        grid_coords = torch.stack(mesh, dim=-1)
    else:
        raise ValueError(f"Unknown coord_convention: '{coord_convention}', expected 'xyz' or 'zyx'")

    return grid_coords.reshape(-1, dim), spatial_shape


def _project_scattered_kernel_engine(
    points: torch.Tensor,                # (B, N, d)
    values: torch.Tensor,                # (B, N, C)
    Y_scaled: torch.Tensor,              # (G, d)
    NY: torch.Tensor,                    # (G, 1)
    sigma_t: torch.Tensor,               # (d,)
    domain_center: torch.Tensor,         # (d,)
    spatial_shape: Tuple[int, ...],      # (*spatial_shape)
    point_weights: Optional[torch.Tensor] = None,  # (B, N, 1) or None
    chunk_size: int = 0,
    target_memory_mb: float = 256.0,
    epsilon: float = 1e-8,
    mask: Optional[torch.Tensor] = None, # (*spatial_shape) or (B, 1, *spatial_shape)
    fill_value: float = 0.0,
    return_density: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Core vectorized GEMM Nadaraya-Watson projection engine.

    Evaluates normalized Gaussian kernel regression from scattered points to an Eulerian grid.
    Utilizes:
    - Coordinate centering relative to domain center to prevent float32 catastrophic cancellation
    - Vectorized GEMM expansion D^2 = NY - 2 * (Y @ X^T) + NX with torch.clamp_min(..., 0.0)
    - Dynamic chunking to respect target memory ceiling
    - Continuous C^inf Nadaraya-Watson division with epsilon floor
    """
    B, N, d = points.shape
    C = values.shape[-1]
    G = Y_scaled.shape[0]

    elem_bytes = points.element_size()
    if chunk_size <= 0:
        chunk_size = max(1, int(target_memory_mb * (1024 ** 2) / (4 * max(1, N) * elem_bytes)))
    chunk_size = min(G, max(1, chunk_size))

    batch_outputs = []
    batch_densities = []

    for b in range(B):
        X_b = points[b]  # (N, d)
        F_b = values[b]  # (N, C)

        # Coordinate centering safeguard
        X_centered = X_b - domain_center
        X_scaled = X_centered / sigma_t
        NX = (X_scaled ** 2).sum(dim=-1, keepdim=True).t()  # (1, N)

        num_chunks = []
        den_chunks = []

        for g_start in range(0, G, chunk_size):
            g_end = min(g_start + chunk_size, G)
            Y_c = Y_scaled[g_start:g_end]  # (chunk, d)
            NY_c = NY[g_start:g_end]        # (chunk, 1)

            # Vectorized GEMM distance expansion with mandatory clamp_min
            d2_raw = NY_c - 2.0 * (Y_c @ X_scaled.t()) + NX
            d2 = torch.clamp_min(d2_raw, 0.0)
            W = torch.exp(-0.5 * d2)  # (chunk, N)

            if point_weights is not None:
                pw_b = point_weights[b]
                if pw_b.dim() == 2:
                    W = W * pw_b.t()
                elif pw_b.dim() == 1:
                    W = W * pw_b.unsqueeze(0)

            num_chunks.append(W @ F_b)                     # (chunk, C)
            den_chunks.append(W.sum(dim=-1, keepdim=True))  # (chunk, 1)

        num = torch.cat(num_chunks, dim=0)  # (G, C)
        den = torch.cat(den_chunks, dim=0)  # (G, 1)

        # Standard C^inf smooth Nadaraya-Watson division
        grid_vals = num / (den + epsilon)  # (G, C)

        # Permute channel to front: (C, *spatial_shape)
        reshaped_vals = grid_vals.view(*spatial_shape, C)
        perm = (d,) + tuple(range(d))
        out_b = reshaped_vals.permute(*perm)
        den_b = den.view(*spatial_shape, 1).permute(*perm)

        batch_outputs.append(out_b)
        batch_densities.append(den_b)

    out = torch.stack(batch_outputs, dim=0)      # (B, C, *spatial_shape)
    density = torch.stack(batch_densities, dim=0)  # (B, 1, *spatial_shape)

    # Apply Eulerian mask
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=out.dtype, device=out.device)
        else:
            mask = mask.to(device=out.device, dtype=out.dtype)
        while mask.dim() < out.dim():
            mask = mask.unsqueeze(0)
        out = out * mask
        density = density * mask

    # Empty voxel replacement when fill_value != 0.0
    if fill_value != 0.0:
        empty_mask = density < (epsilon * 10.0)
        out = torch.where(empty_mask, torch.full_like(out, fill_value), out)

    if return_density:
        return out, density
    return out


def _project_scattered_bspline_engine(
    points: torch.Tensor,                # (B, N, d)
    values: torch.Tensor,                # (B, N, C)
    spatial_shape: Tuple[int, ...],      # (*spatial)
    min_coords: torch.Tensor,            # (d,)
    max_coords: torch.Tensor,            # (d,)
    point_weights: Optional[torch.Tensor] = None,  # (B, N, 1) or None
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    coord_convention: str = 'xyz',
    mask: Optional[torch.Tensor] = None,
    fill_value: float = 0.0,
    return_density: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Projects scattered point features to an Eulerian grid using ANTsTorch multi-level B-splines.

    Leverages `antstorch.bspline_flows.fit_bspline_object_to_scattered_data` for
    compact O(N * 4^d) support accumulation, multi-resolution hierarchical refinement,
    and memory-bounded synthesis.
    """
    if not has_antstorch():
        raise ImportError(
            "B-spline scattered data projection requires 'antstorch' (antstorch.bspline_flows). "
            "Please install it via 'pip install antstorch'."
        )
    from antstorch.bspline_flows import (
        fit_bspline_object_to_scattered_data,
        ImageDomain,
        mesh_size_for_spline_distance,
    )

    B, N, d = points.shape
    C = values.shape[-1]

    # If no points, return empty grid with fill_value
    if N == 0:
        empty_out = torch.full((B, C, *spatial_shape), fill_value, device=points.device, dtype=points.dtype)
        if return_density:
            empty_den = torch.zeros((B, 1, *spatial_shape), device=points.device, dtype=points.dtype)
            return empty_out, empty_den
        return empty_out

    # Construct ITK domain specification
    # In ANTsTorch, size, origin, spacing use ITK axis order (X, Y, Z)
    if coord_convention == 'xyz':
        size_itk = tuple(reversed(spatial_shape))
        origin_itk = tuple(float(min_coords[i].item()) for i in range(d))
        extent_itk = tuple(float((max_coords[i] - min_coords[i]).item()) for i in range(d))
    elif coord_convention == 'zyx':
        size_itk = tuple(spatial_shape)
        origin_itk = tuple(float(min_coords[d - 1 - i].item()) for i in range(d))
        extent_itk = tuple(float((max_coords[d - 1 - i] - min_coords[d - 1 - i]).item()) for i in range(d))
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

    batch_outputs = []
    batch_densities = []

    for b in range(B):
        pts_b = points[b]  # (N, d)
        vals_b = values[b]  # (N, C)

        # Coordinate alignment: ANTsTorch expects physical points in (X, Y[, Z]) order
        if coord_convention == 'zyx':
            pts_b = pts_b.flip(dims=[-1])

        w_b = None
        if point_weights is not None:
            w_b = point_weights[b].squeeze(-1)  # (N,)

        grid_b = fit_bspline_object_to_scattered_data(
            scattered_data=vals_b,
            parametric_data=pts_b,
            parametric_domain_origin=domain.origin,
            parametric_domain_spacing=domain.spacing,
            parametric_domain_size=domain.size,
            data_weights=w_b,
            number_of_fitting_levels=number_of_fitting_levels,
            mesh_size=mesh_size_itk,
            device=points.device,
            dtype=points.dtype,
        )  # Returns (1, C, *reversed(size_itk)) = (1, C, *spatial_shape)
        batch_outputs.append(grid_b.squeeze(0))

        if return_density:
            ones_b = torch.ones((N, 1), device=points.device, dtype=points.dtype)
            den_b = fit_bspline_object_to_scattered_data(
                scattered_data=ones_b,
                parametric_data=pts_b,
                parametric_domain_origin=domain.origin,
                parametric_domain_spacing=domain.spacing,
                parametric_domain_size=domain.size,
                data_weights=w_b,
                number_of_fitting_levels=max(1, min(2, number_of_fitting_levels)),
                mesh_size=mesh_size_itk,
                device=points.device,
                dtype=points.dtype,
            )
            batch_densities.append(den_b.squeeze(0))

    out = torch.stack(batch_outputs, dim=0)  # (B, C, *spatial_shape)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=out.dtype, device=out.device)
        else:
            mask = mask.to(device=out.device, dtype=out.dtype)
        while mask.dim() < out.dim():
            mask = mask.unsqueeze(0)
        out = out * mask

    if fill_value != 0.0 and return_density:
        density = torch.stack(batch_densities, dim=0)
        empty_mask = density < 1e-4
        out = torch.where(empty_mask, torch.full_like(out, fill_value), out)

    if return_density:
        density = torch.stack(batch_densities, dim=0)
        return out, density
    return out


def project_scattered_to_grid(
    points: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    grid_shape: Union[int, Tuple[int, ...]] = 64,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
    sigma: Union[float, Sequence[float], torch.Tensor, str] = 0.03,
    mask: Optional[torch.Tensor] = None,
    point_weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
    chunk_size: int = 0,
    target_memory_mb: float = 256.0,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    fill_value: float = 0.0,
    return_density: bool = False,
    method: Literal['gaussian', 'bspline'] = 'gaussian',
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
    config: Optional[ProjectionConfig] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Differentiably project scattered point features onto an Eulerian regular grid.

    Maps scattered coordinates {x_i} with features {f_i} onto a regular grid
    lattice via Nadaraya-Watson Gaussian kernel regression or ANTsTorch multi-level
    cubic B-splines. Supports arbitrary dimensions (2D, 3D), batched and unbatched
    inputs, arbitrary domain bounding boxes, auto-bounding, Eulerian masking, and
    point confidence weights.

    Parameters
    ----------
    points : Tensor or ndarray of shape (N, d) or (B, N, d)
        Scattered coordinates in d-dimensional space.
    values : Tensor or ndarray of shape (N,), (N, C), (B, N), or (B, N, C)
        Scalar or multi-channel vector features associated with each point.
    grid_shape : int or Tuple[int, ...], default 64
        Spatial resolution of output grid.
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
        Coordinate domain bounds.
    sigma : float, sequence of float, Tensor, or 'auto', default 0.03
        Gaussian bandwidth.
    mask : Tensor, optional
        Eulerian domain mask of shape (*spatial_shape) or (B, 1, *spatial_shape).
    point_weights : Tensor or ndarray, optional
        Confidence weights per point, shape (N,), (N, 1), (B, N), or (B, N, 1).
    epsilon : float, default 1e-8
        Numerical stability denominator floor.
    chunk_size : int, default 0
        Chunk size in voxels. If <= 0, automatically determined from memory ceiling.
    target_memory_mb : float, default 256.0
        Target memory ceiling for chunking in MB.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate to spatial tensor mapping convention.
    fill_value : float, default 0.0
        Value for empty voxels.
    return_density : bool, default False
        If True, returns (projected_features, kernel_density).
    config : ProjectionConfig, optional
        Configuration dataclass overriding default arguments.

    Returns
    -------
    Tensor of shape (B, C, *spatial_shape) or Tuple[Tensor, Tensor]
        Projected Eulerian feature tensor, and optionally the density tensor.
    """
    if config is not None:
        grid_shape = config.grid_shape
        domain_bounds = config.domain_bounds
        sigma = config.sigma
        epsilon = config.epsilon
        chunk_size = config.chunk_size
        target_memory_mb = config.target_memory_mb
        coord_convention = config.coord_convention
        fill_value = config.fill_value
        method = getattr(config, 'method', method)
        number_of_fitting_levels = getattr(config, 'number_of_fitting_levels', number_of_fitting_levels)
        mesh_size = getattr(config, 'mesh_size', mesh_size)
        spline_distance = getattr(config, 'spline_distance', spline_distance)

    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon}")

    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, dtype=torch.float32)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=points.dtype, device=points.device)
    else:
        values = values.to(device=points.device, dtype=points.dtype)

    unbatched = points.dim() == 2
    if unbatched:
        points = points.unsqueeze(0)  # (1, N, d)
    B, N, d = points.shape

    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,) * d
    elif len(grid_shape) != d:
        raise ValueError(f"grid_shape length {len(grid_shape)} must match point dimension {d}")

    # Standardize values to (B, N, C)
    if values.dim() == 1:
        if values.shape[0] != N:
            raise ValueError(f"values length {values.shape[0]} does not match points count {N}")
        values = values.view(1, N, 1).expand(B, -1, -1)
    elif values.dim() == 2:
        if unbatched:
            if values.shape[0] != N:
                raise ValueError(f"values shape {values.shape} does not match points count {N}")
            values = values.unsqueeze(0)  # (1, N, C)
        else:
            if values.shape == (B, N):
                values = values.unsqueeze(-1)  # (B, N, 1)
            elif values.shape[0] == N:
                values = values.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
            else:
                raise ValueError(f"Cannot align values shape {values.shape} with points shape {points.shape}")
    elif values.dim() == 3:
        if values.shape[:2] != (B, N):
            raise ValueError(f"values shape {values.shape} does not match points shape {points.shape}")
    else:
        raise ValueError(f"Unsupported values dimension {values.dim()}, expected 1D, 2D, or 3D tensor")

    # Standardize point_weights to (B, N, 1)
    if point_weights is not None:
        if not isinstance(point_weights, torch.Tensor):
            point_weights = torch.as_tensor(point_weights, dtype=points.dtype, device=points.device)
        else:
            point_weights = point_weights.to(device=points.device, dtype=points.dtype)
        if point_weights.dim() == 1:
            if point_weights.shape[0] != N:
                raise ValueError(f"point_weights length {point_weights.shape[0]} does not match points count {N}")
            point_weights = point_weights.view(1, N, 1).expand(B, -1, -1)
        elif point_weights.dim() == 2:
            if unbatched:
                point_weights = point_weights.unsqueeze(0)
            else:
                if point_weights.shape == (B, N):
                    point_weights = point_weights.unsqueeze(-1)
                elif point_weights.shape[0] == N and point_weights.shape[1] == 1:
                    point_weights = point_weights.unsqueeze(0).expand(B, -1, -1)
                else:
                    raise ValueError(f"Cannot align point_weights shape {point_weights.shape} with points {points.shape}")
        elif point_weights.dim() == 3:
            if point_weights.shape[:2] != (B, N):
                raise ValueError(f"point_weights shape {point_weights.shape} does not match points shape {points.shape}")

    # Resolve domain bounds
    if domain_bounds is None or domain_bounds == 'auto':
        if N == 0:
            min_coords = torch.full((d,), -1.0, device=points.device, dtype=points.dtype)
            max_coords = torch.full((d,), 1.0, device=points.device, dtype=points.dtype)
        else:
            min_coords = points.amin(dim=(0, 1))
            max_coords = points.amax(dim=(0, 1))
            margin = 3.0 * (float(sigma) if isinstance(sigma, (int, float)) else 0.05)
            min_coords = min_coords - margin
            max_coords = max_coords + margin
            span = max_coords - min_coords
            pad = torch.clamp_min(1e-4 - span, 0.0) * 0.5
            min_coords = min_coords - pad
            max_coords = max_coords + pad
        min_coords = min_coords.detach()
        max_coords = max_coords.detach()
    elif isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2:
        if isinstance(domain_bounds[0], (int, float)):
            min_coords = torch.full((d,), float(domain_bounds[0]), device=points.device, dtype=points.dtype)
            max_coords = torch.full((d,), float(domain_bounds[1]), device=points.device, dtype=points.dtype)
        else:
            min_coords = torch.as_tensor(domain_bounds[0], device=points.device, dtype=points.dtype)
            max_coords = torch.as_tensor(domain_bounds[1], device=points.device, dtype=points.dtype)
    else:
        raise ValueError(f"Unsupported domain_bounds specification: {domain_bounds}")

    # Build grid
    grid_coords, spatial_shape = _build_eulerian_grid(
        grid_shape, (min_coords, max_coords), coord_convention, points.device, points.dtype
    )

    if method == 'bspline':
        return _project_scattered_bspline_engine(
            points=points,
            values=values,
            spatial_shape=spatial_shape,
            min_coords=min_coords,
            max_coords=max_coords,
            point_weights=point_weights,
            number_of_fitting_levels=number_of_fitting_levels,
            mesh_size=mesh_size,
            spline_distance=spline_distance,
            coord_convention=coord_convention,
            mask=mask,
            fill_value=fill_value,
            return_density=return_density,
        )

    # Resolve bandwidth sigma
    if isinstance(sigma, str) and sigma == 'auto':
        if N >= 2:
            s_val = compute_adaptive_sigma(points)
            sigma_t = torch.full((d,), s_val, device=points.device, dtype=points.dtype)
        else:
            if coord_convention == 'xyz':
                shape_t = torch.tensor([grid_shape[d - 1 - i] for i in range(d)], device=points.device, dtype=points.dtype)
            else:
                shape_t = torch.tensor(list(grid_shape), device=points.device, dtype=points.dtype)
            spacing = (max_coords - min_coords) / shape_t.clamp_min(1.0)
            sigma_scale = config.sigma_scale if config is not None else 1.5
            sigma_t = spacing * sigma_scale
    elif isinstance(sigma, (int, float)):
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        sigma_t = torch.full((d,), float(sigma), device=points.device, dtype=points.dtype)
    else:
        sigma_t = torch.as_tensor(sigma, device=points.device, dtype=points.dtype)
        if (sigma_t <= 0.0).any():
            raise ValueError(f"All components of sigma must be strictly positive, got {sigma_t}")

    # Coordinate centering safeguard
    domain_center = 0.5 * (min_coords + max_coords)
    Y_centered = grid_coords - domain_center
    Y_scaled = Y_centered / sigma_t
    NY = (Y_scaled ** 2).sum(-1, keepdim=True)

    return _project_scattered_kernel_engine(
        points=points,
        values=values,
        Y_scaled=Y_scaled,
        NY=NY,
        sigma_t=sigma_t,
        domain_center=domain_center,
        spatial_shape=spatial_shape,
        point_weights=point_weights,
        chunk_size=chunk_size,
        target_memory_mb=target_memory_mb,
        epsilon=epsilon,
        mask=mask,
        fill_value=fill_value,
        return_density=return_density,
    )


class ScatteredProjector(nn.Module):
    """Pre-cached Eulerian grid projector for iterative registration loops.

    Precomputes and caches the Eulerian grid geometry, coordinate centering,
    scaled grid coordinates, and squared norms in persistent=False buffers,
    delivering significant speedups during iterative SyN loops.
    """
    def __init__(
        self,
        grid_shape: Union[int, Tuple[int, ...]] = 64,
        domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
        sigma: Union[float, Sequence[float], str] = 0.03,
        mask: Optional[torch.Tensor] = None,
        epsilon: float = 1e-8,
        chunk_size: int = 0,
        target_memory_mb: float = 256.0,
        coord_convention: Literal['xyz', 'zyx'] = 'xyz',
        fill_value: float = 0.0,
        method: Literal['gaussian', 'bspline'] = 'gaussian',
        number_of_fitting_levels: int = 4,
        mesh_size: Union[int, Sequence[int]] = 1,
        spline_distance: Optional[Union[float, Sequence[float]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        config: Optional[ProjectionConfig] = None,
    ):
        super().__init__()
        if config is not None:
            grid_shape = config.grid_shape
            domain_bounds = config.domain_bounds
            sigma = config.sigma
            epsilon = config.epsilon
            chunk_size = config.chunk_size
            target_memory_mb = config.target_memory_mb
            coord_convention = config.coord_convention
            fill_value = config.fill_value
            method = config.method
            number_of_fitting_levels = config.number_of_fitting_levels
            mesh_size = config.mesh_size
            spline_distance = config.spline_distance

        if isinstance(device, str):
            device = torch.device(device)

        self.grid_shape = grid_shape
        self.domain_bounds = domain_bounds
        self.sigma = sigma
        self.epsilon = epsilon
        self.chunk_size = chunk_size
        self.target_memory_mb = target_memory_mb
        self.coord_convention = coord_convention
        self.fill_value = fill_value
        self.method = method
        self.number_of_fitting_levels = number_of_fitting_levels
        self.mesh_size = mesh_size
        self.spline_distance = spline_distance

        is_static = (method == 'gaussian' and domain_bounds is not None and domain_bounds != 'auto' and sigma != 'auto')
        self.is_static = is_static

        if is_static:
            if isinstance(grid_shape, int):
                d = 2
                if isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2 and hasattr(domain_bounds[0], '__len__'):
                    d = len(domain_bounds[0])
                dim_grid_shape = (grid_shape,) * d
            else:
                d = len(grid_shape)
                dim_grid_shape = tuple(grid_shape)

            if isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2:
                if isinstance(domain_bounds[0], (int, float)):
                    min_b = torch.full((d,), float(domain_bounds[0]), device=device, dtype=dtype)
                    max_b = torch.full((d,), float(domain_bounds[1]), device=device, dtype=dtype)
                else:
                    min_b = torch.as_tensor(domain_bounds[0], device=device, dtype=dtype)
                    max_b = torch.as_tensor(domain_bounds[1], device=device, dtype=dtype)
            else:
                raise ValueError(f"Invalid domain_bounds: {domain_bounds}")

            if isinstance(sigma, (int, float)):
                sigma_t = torch.full((d,), float(sigma), device=device, dtype=dtype)
            else:
                sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)

            grid_coords, spatial_shape = _build_eulerian_grid(
                dim_grid_shape, (min_b, max_b), coord_convention, device, dtype
            )
            domain_center = 0.5 * (min_b + max_b)
            Y_centered = grid_coords - domain_center
            Y_scaled = Y_centered / sigma_t
            NY = (Y_scaled ** 2).sum(-1, keepdim=True)

            self.spatial_shape = spatial_shape
            self.register_buffer('grid_coords', grid_coords, persistent=False)
            self.register_buffer('domain_center', domain_center, persistent=False)
            self.register_buffer('sigma_t', sigma_t, persistent=False)
            self.register_buffer('Y_scaled', Y_scaled, persistent=False)
            self.register_buffer('NY', NY, persistent=False)
        else:
            self.spatial_shape = None
            self.grid_coords = None
            self.domain_center = None
            self.sigma_t = None
            self.Y_scaled = None
            self.NY = None

        if mask is not None:
            mask_t = torch.as_tensor(mask, device=device, dtype=dtype)
            self.register_buffer('mask', mask_t, persistent=False)
        else:
            self.mask = None

    def forward(
        self,
        points: Union[torch.Tensor, np.ndarray],
        values: Union[torch.Tensor, np.ndarray],
        point_weights: Optional[torch.Tensor] = None,
        return_density: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Project scattered points and features using cached grid geometry."""
        if not self.is_static:
            return project_scattered_to_grid(
                points=points,
                values=values,
                grid_shape=self.grid_shape,
                domain_bounds=self.domain_bounds,
                sigma=self.sigma,
                mask=self.mask,
                point_weights=point_weights,
                epsilon=self.epsilon,
                chunk_size=self.chunk_size,
                target_memory_mb=self.target_memory_mb,
                coord_convention=self.coord_convention,
                fill_value=self.fill_value,
                return_density=return_density,
                method=self.method,
                number_of_fitting_levels=self.number_of_fitting_levels,
                mesh_size=self.mesh_size,
                spline_distance=self.spline_distance,
            )

        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)
        else:
            points = points.to(device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)

        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)
        else:
            values = values.to(device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)

        unbatched = points.dim() == 2
        if unbatched:
            points = points.unsqueeze(0)
        B, N, d = points.shape

        if values.dim() == 1:
            if values.shape[0] != N:
                raise ValueError(f"values length {values.shape[0]} does not match points count {N}")
            values = values.view(1, N, 1).expand(B, -1, -1)
        elif values.dim() == 2:
            if unbatched:
                if values.shape[0] != N:
                    raise ValueError(f"values shape {values.shape} does not match points count {N}")
                values = values.unsqueeze(0)
            else:
                if values.shape == (B, N):
                    values = values.unsqueeze(-1)
                elif values.shape[0] == N:
                    values = values.unsqueeze(0).expand(B, -1, -1)
                else:
                    raise ValueError(f"Cannot align values shape {values.shape} with points {points.shape}")
        elif values.dim() == 3:
            if values.shape[:2] != (B, N):
                raise ValueError(f"values shape {values.shape} does not match points {points.shape}")

        if point_weights is not None:
            if not isinstance(point_weights, torch.Tensor):
                point_weights = torch.as_tensor(point_weights, device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)
            else:
                point_weights = point_weights.to(device=self.Y_scaled.device, dtype=self.Y_scaled.dtype)
            if point_weights.dim() == 1:
                point_weights = point_weights.view(1, N, 1).expand(B, -1, -1)
            elif point_weights.dim() == 2:
                if unbatched:
                    point_weights = point_weights.unsqueeze(0)
                else:
                    if point_weights.shape == (B, N):
                        point_weights = point_weights.unsqueeze(-1)
                    elif point_weights.shape[0] == N and point_weights.shape[1] == 1:
                        point_weights = point_weights.unsqueeze(0).expand(B, -1, -1)

        return _project_scattered_kernel_engine(
            points=points,
            values=values,
            Y_scaled=self.Y_scaled,
            NY=self.NY,
            sigma_t=self.sigma_t,
            domain_center=self.domain_center,
            spatial_shape=self.spatial_shape,
            point_weights=point_weights,
            chunk_size=self.chunk_size,
            target_memory_mb=self.target_memory_mb,
            epsilon=self.epsilon,
            mask=self.mask,
            fill_value=self.fill_value,
            return_density=return_density,
        )


def differentiable_grid_projection(
    u_coords: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    grid_res: int = 64,
    sigma: Union[float, str] = 0.03,
    epsilon: float = 1e-8,
    nw_chunk_size: int = 0,
    grid_bounds: Tuple[float, float] = (-1.0, 1.0),
    return_density: bool = False,
    method: Literal['gaussian', 'bspline'] = 'gaussian',
    number_of_fitting_levels: int = 4,
    mesh_size: Union[int, Sequence[int]] = 1,
    spline_distance: Optional[Union[float, Sequence[float]]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Drop-in backward compatibility wrapper matching sulceye.flatmap.utils signature.

    Parameters
    ----------
    u_coords : Tensor or ndarray of shape (N, 2)
        UV point cloud coordinates.
    values : Tensor or ndarray of shape (N,) or (N, C)
        Scalar or multi-channel values at points.
    grid_res : int, default 64
        Output grid resolution (grid_res x grid_res).
    sigma : float or 'auto', default 0.03
        Gaussian bandwidth.
    epsilon : float, default 1e-8
        Numerical stability denominator floor.
    nw_chunk_size : int, default 0
        Number of grid pixels processed per chunk. 0 enables auto-memory chunking.
    grid_bounds : Tuple[float, float], default (-1.0, 1.0)
        Coordinate bounds of the grid.
    return_density : bool, default False
        If True, returns (grid_vals, density).
    method : {'gaussian', 'bspline'}, default 'gaussian'
        Projection engine.
    number_of_fitting_levels : int, default 4
    mesh_size : int or Sequence[int], default 1
    spline_distance : float or Sequence[float], optional

    Returns
    -------
    Tensor of shape (1, 1, grid_res, grid_res) or (1, C, grid_res, grid_res),
    or tuple with density.
    """
    return project_scattered_to_grid(
        points=u_coords,
        values=values,
        grid_shape=grid_res,
        domain_bounds=grid_bounds,
        sigma=sigma,
        epsilon=epsilon,
        chunk_size=nw_chunk_size,
        coord_convention='xyz',
        return_density=return_density,
        method=method,
        number_of_fitting_levels=number_of_fitting_levels,
        mesh_size=mesh_size,
        spline_distance=spline_distance,
    )


def compute_distance_transform_to_grid(
    points: Union[torch.Tensor, np.ndarray],
    grid_shape: Union[int, Tuple[int, ...]],
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = (-1.0, 1.0),
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    potential_tau: Optional[float] = None,
    chunk_size: int = 32768,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute Euclidean Distance Transform (or smooth potential) from scattered points to an Eulerian grid.

    Calculates the Euclidean distance from every voxel in an Eulerian grid to the closest
    scattered point using memory-bounded chunked distance calculations. Optionally converts
    the distance field into a smooth exponential potential P(x) = exp(-D(x) / tau).

    Parameters
    ----------
    points : Tensor or ndarray of shape (N, d) or (B, N, d)
        Scattered point coordinates.
    grid_shape : int or Tuple[int, ...]
        Resolution of the Eulerian grid, e.g. 64 or (64, 64, 64).
    domain_bounds : tuple or 'auto', default (-1.0, 1.0)
        Physical bounding box of the grid.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Axis ordering convention.
    potential_tau : float, optional
        If specified, returns the smooth distance potential P(x) = exp(-D(x) / tau) in [0, 1].
        If None, returns raw Euclidean distance D(x) in physical coordinate units.
    chunk_size : int, default 32768
        Chunk size for memory-safe distance evaluation.
    device : str or torch.device, optional
        Target device for computation.
    dtype : torch.dtype, default torch.float32
        Precision.

    Returns
    -------
    Tensor of shape (B, 1, *spatial_shape)
        Eulerian distance transform or distance potential map.
    """
    if device is None:
        if isinstance(points, torch.Tensor):
            device = points.device
        else:
            device = torch.device('cpu')

    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, dtype=dtype, device=device)
    else:
        points = points.to(device=device, dtype=dtype)

    unbatched = points.dim() == 2
    if unbatched:
        points = points.unsqueeze(0)
    B, N, d = points.shape

    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,) * d
    elif len(grid_shape) != d:
        raise ValueError(f"grid_shape length {len(grid_shape)} must match point dimension {d}")

    # Resolve domain bounds
    if domain_bounds is None or domain_bounds == 'auto':
        if N == 0:
            min_coords = torch.full((d,), -1.0, device=device, dtype=dtype)
            max_coords = torch.full((d,), 1.0, device=device, dtype=dtype)
        else:
            min_coords = points.amin(dim=(0, 1)) - 0.05
            max_coords = points.amax(dim=(0, 1)) + 0.05
    elif isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2:
        if isinstance(domain_bounds[0], (int, float)):
            min_coords = torch.full((d,), float(domain_bounds[0]), device=device, dtype=dtype)
            max_coords = torch.full((d,), float(domain_bounds[1]), device=device, dtype=dtype)
        else:
            min_coords = torch.as_tensor(domain_bounds[0], device=device, dtype=dtype)
            max_coords = torch.as_tensor(domain_bounds[1], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported domain_bounds specification: {domain_bounds}")

    # Build grid
    grid_coords, spatial_shape = _build_eulerian_grid(
        grid_shape, (min_coords, max_coords), coord_convention, device, dtype
    )
    G = grid_coords.shape[0]

    batch_maps = []
    chunk_sz = max(1, chunk_size)

    for b in range(B):
        X_b = points[b]  # (N, d)
        min_dists_chunks = []
        for g_start in range(0, G, chunk_sz):
            g_end = min(g_start + chunk_sz, G)
            Y_c = grid_coords[g_start:g_end]  # (chunk, d)
            dists = torch.cdist(Y_c, X_b)     # (chunk, N)
            min_d, _ = torch.min(dists, dim=-1)  # (chunk,)
            min_dists_chunks.append(min_d)
        D_b = torch.cat(min_dists_chunks, dim=0).view(spatial_shape)
        if potential_tau is not None:
            if potential_tau <= 0.0:
                raise ValueError(f"potential_tau must be strictly positive, got {potential_tau}")
            D_b = torch.exp(-D_b / potential_tau)
        batch_maps.append(D_b.unsqueeze(0))  # (1, *spatial_shape)

    out = torch.stack(batch_maps, dim=0)  # (B, 1, *spatial_shape)
    return out

