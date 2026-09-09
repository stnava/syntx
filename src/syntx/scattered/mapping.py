"""
syntx.scattered.mapping — Lagrangian Coordinate Mapping & Field Evaluation
==========================================================================

Differentiable Eulerian displacement field evaluation at arbitrary scattered
coordinates and bidirectional coordinate warping (forward and backward).

Key Features & Mathematical Formulations:
- Differentiable Eulerian field evaluation at scattered points via singleton grid
  query insertion with torch.nn.functional.grid_sample in O(N) operations.
- Dimension-agnostic query construction for 2D (B, 1, N, 2) and 3D (B, 1, 1, N, 3).
- Bidirectional coordinate warping:
    phi(x) = x + u(x)        (forward)
    phi^-1(y) = y + v(y)     (backward / inverse)
- Support for physical bounding boxes [b_min, b_max], unit domain [-1, 1], and auto-bounding.
- Exact coordinate convention handling ('xyz' Cartesian/ITK and 'zyx' tensor-index).
- Automatic device/dtype alignment and autograd gradient preservation.
- Full compatibility with Anderson-accelerated displacement field inversion.
"""

from typing import Optional, Tuple, Union, Sequence, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_domain_bounds(
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]],
    coords: torch.Tensor,
    d: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Parse and resolve domain bounding boxes into (min_coords, max_coords) tensors."""
    if domain_bounds is None or domain_bounds == 'none':
        return None

    if isinstance(domain_bounds, str) and domain_bounds == 'auto':
        if coords.shape[0] == 0 or (coords.dim() == 3 and coords.shape[1] == 0):
            min_coords = torch.full((d,), -1.0, device=coords.device, dtype=coords.dtype)
            max_coords = torch.full((d,), 1.0, device=coords.device, dtype=coords.dtype)
        else:
            reduce_dims = (0, 1) if coords.dim() == 3 else (0,)
            min_coords = coords.amin(dim=reduce_dims)
            max_coords = coords.amax(dim=reduce_dims)
            margin = 0.05 * (max_coords - min_coords).clamp_min(1.0)
            min_coords = min_coords - margin
            max_coords = max_coords + margin
            span = max_coords - min_coords
            pad = torch.clamp_min(1e-4 - span, 0.0) * 0.5
            min_coords = min_coords - pad
            max_coords = max_coords + pad
        return min_coords.detach(), max_coords.detach()

    if isinstance(domain_bounds, (tuple, list)) and len(domain_bounds) == 2:
        if isinstance(domain_bounds[0], (int, float)):
            min_coords = torch.full((d,), float(domain_bounds[0]), device=coords.device, dtype=coords.dtype)
            max_coords = torch.full((d,), float(domain_bounds[1]), device=coords.device, dtype=coords.dtype)
        else:
            min_coords = torch.as_tensor(domain_bounds[0], device=coords.device, dtype=coords.dtype)
            max_coords = torch.as_tensor(domain_bounds[1], device=coords.device, dtype=coords.dtype)
        return min_coords.detach(), max_coords.detach()

    raise ValueError(f"Unsupported domain_bounds format: {domain_bounds}")


def evaluate_field_at_scattered(
    field: torch.Tensor,
    coords: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = True,
    channel_dim: int = -1,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = None,
) -> torch.Tensor:
    """Differentiably evaluate an Eulerian vector or scalar field at scattered coordinates.

    Uses singleton grid query insertion with `torch.nn.functional.grid_sample` to evaluate
    the continuous field at arbitrary scattered coordinates in O(N) operations without
    allocating intermediate full-grid tensors.

    Parameters
    ----------
    field : torch.Tensor
        Eulerian tensor representing a displacement or feature field:
        - Channels-last (default, channel_dim=-1): shape (B, *spatial, C) or (*spatial, C).
        - Channels-first (channel_dim=1): shape (B, C, *spatial) or (C, *spatial).
        - Unbatched scalar: shape (*spatial).
    coords : torch.Tensor
        Scattered coordinates of shape (B, N, d) or (N, d).
    mode : {'bilinear', 'nearest', 'bicubic'}, default 'bilinear'
        Interpolation mode. Note that for 3D inputs, 'bilinear' performs trilinear interpolation.
    padding_mode : {'border', 'zeros', 'reflection'}, default 'border'
        Padding mode for outside-grid coordinates.
    align_corners : bool, default True
        Grid sample alignment convention. Matches syntx Eulerian grid lattice standard.
    channel_dim : int, default -1
        Axis of vector/feature channels in `field`. -1 for channels-last, 1 for channels-first.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate convention of `coords`:
        - 'xyz': Cartesian / ITK order. Matches F.grid_sample directly (no flipping).
        - 'zyx': Tensor indexing order. Flipped along last axis to match F.grid_sample.
    domain_bounds : tuple or 'auto', optional
        Physical domain bounding box [min_coords, max_coords]. If provided, `coords` are
        affinely normalized to [-1, 1]^d before grid sampling. If None, `coords` are
        assumed to already reside in [-1, 1]^d.

    Returns
    -------
    torch.Tensor
        Sampled field values of shape (B, N, C) or (N, C).
    """
    if not isinstance(field, torch.Tensor):
        field = torch.as_tensor(field)
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords, device=field.device, dtype=field.dtype)
    else:
        if coords.device != field.device or coords.dtype != field.dtype:
            coords = coords.to(device=field.device, dtype=field.dtype)

    d = coords.shape[-1]
    if d not in (2, 3):
        raise ValueError(f"Expected coords spatial dimension d in (2, 3), got d={d}")

    coords_unbatched = (coords.dim() == 2)
    coords_b = coords.unsqueeze(0) if coords_unbatched else coords
    if coords_b.dim() != 3:
        raise ValueError(f"Expected coords tensor of dimension 2 or 3, got dim={coords.dim()}")

    B_coords, N, _ = coords_b.shape

    # 1. Coordinate normalization if domain bounds specified
    bounds = _resolve_domain_bounds(domain_bounds, coords_b, d)
    if bounds is not None:
        min_b, max_b = bounds
        span = (max_b - min_b).clamp_min(1e-8)
        norm_coords = 2.0 * (coords_b - min_b) / span - 1.0
    else:
        norm_coords = coords_b

    # 2. Coordinate ordering alignment for F.grid_sample
    if coord_convention == 'xyz':
        grid_coords = norm_coords
    elif coord_convention == 'zyx':
        grid_coords = torch.flip(norm_coords, dims=[-1])
    else:
        raise ValueError(f"Unknown coord_convention: '{coord_convention}', expected 'xyz' or 'zyx'")

    # 3. Standardize field to channels-first format: (B_field, C, *spatial)
    field_unbatched = False
    if channel_dim is None:
        if field.dim() == d + 1:
            # Batched scalar: (B, *spatial)
            field_unbatched = False
            field_cf = field.unsqueeze(1)
        elif field.dim() == d:
            # Unbatched scalar: (*spatial)
            field_unbatched = True
            field_cf = field.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Incompatible scalar field shape {field.shape} for spatial dimension d={d}")
    elif channel_dim in (-1, field.dim() - 1):
        if field.dim() == d + 2:
            field_unbatched = False
            field_cf = torch.movedim(field, -1, 1)
        elif field.dim() == d + 1:
            field_unbatched = True
            field_cf = torch.movedim(field, -1, 0).unsqueeze(0)
        else:
            raise ValueError(f"Incompatible field shape {field.shape} for spatial dimension d={d}")
    elif channel_dim in (1, 0):
        if field.dim() == d + 2:
            field_unbatched = False
            field_cf = field
        elif field.dim() == d + 1:
            field_unbatched = True
            field_cf = field.unsqueeze(0)
        else:
            raise ValueError(f"Incompatible field shape {field.shape} for spatial dimension d={d}")
    else:
        raise ValueError(f"Unsupported channel_dim {channel_dim}")

    B_field = field_cf.shape[0]
    C = field_cf.shape[1]

    # 4. Batch broadcasting
    if B_field == 1 and B_coords > 1:
        field_cf = field_cf.expand(B_coords, -1, *([-1] * d))
        B = B_coords
    elif B_coords == 1 and B_field > 1:
        grid_coords = grid_coords.expand(B_field, -1, -1)
        B = B_field
    elif B_field == B_coords:
        B = B_field
    else:
        raise ValueError(f"Batch dimension mismatch: field has batch {B_field}, coords has batch {B_coords}")

    # 5. Singleton grid query insertion: shape (B, *([1]*(d-1)), N, d)
    query = grid_coords.view(B, *([1] * (d - 1)), N, d)

    # 6. Differentiable grid sampling
    sampled = F.grid_sample(
        field_cf, query, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    # 7. Reshape output to canonical (B, N, C)
    out = sampled.view(B, C, N).movedim(1, -1)

    if coords_unbatched and B == 1:
        out = out.squeeze(0)

    return out


def warp_scattered_coordinates(
    coords: torch.Tensor,
    displacement_field: torch.Tensor,
    direction: Literal['forward', 'backward', 'inverse'] = 'forward',
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: bool = True,
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = None,
    scale_displacement: Optional[bool] = None,
    auto_invert: bool = False,
    inversion_steps: int = 20,
    coord_convention: Literal['xyz', 'zyx'] = 'xyz',
    is_physical: Optional[bool] = None,
) -> torch.Tensor:
    """Warp scattered coordinates through an Eulerian displacement field.

    Applies forward warping phi(x) = x + u(x) or backward warping phi^-1(y) = y + v(y).
    Supports arbitrary domain bounding boxes, physical vs. normalized displacement units,
    and optional on-the-fly Anderson displacement field inversion.

    Parameters
    ----------
    coords : torch.Tensor
        Scattered coordinates of shape (B, N, d) or (N, d).
    displacement_field : torch.Tensor
        Eulerian displacement field of shape (B, *spatial, d) or (*spatial, d).
    direction : {'forward', 'backward', 'inverse'}, default 'forward'
        Warping direction:
        - 'forward': computes phi(x) = x + u(x).
        - 'backward' or 'inverse': computes phi^-1(y) = y + v(y). If `auto_invert=True`,
          inverts `displacement_field` via Anderson acceleration; otherwise assumes
          `displacement_field` is already the inverse field v.
    mode : str, default 'bilinear'
        Interpolation mode ('bilinear', 'nearest', 'bicubic').
    padding_mode : str, default 'border'
        Padding mode for coordinates outside the grid bounds.
    align_corners : bool, default True
        Grid sample alignment convention.
    domain_bounds : tuple or 'auto', optional
        Domain bounds [min_coords, max_coords]. If provided, `coords` are normalized
        to [-1, 1]^d for field evaluation and denormalized accordingly.
    scale_displacement : bool, optional
        Whether to scale normalized displacement vectors by (max_b - min_b) / 2.0.
        If None, automatically set to False if displacement_field has `is_physical=True`,
        and True if domain_bounds is specified.
    auto_invert : bool, default False
        If True and direction is 'backward', computes the inverse displacement field
        using Type-I Anderson acceleration before sampling.
    inversion_steps : int, default 20
        Number of iterations if auto_invert is performed.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Coordinate mapping convention.
    is_physical : bool, optional
        Explicitly declare whether displacement vectors are in physical millimeters.
        If None, inferred from displacement_field attribute `is_physical`.

    Returns
    -------
    torch.Tensor
        Warped coordinates of shape (B, N, d) or (N, d).
    """
    if direction not in ('forward', 'backward', 'inverse'):
        raise ValueError(f"Unknown direction '{direction}', expected 'forward', 'backward', or 'inverse'")

    d = coords.shape[-1]
    if d not in (2, 3):
        raise ValueError(f"Expected coords spatial dimension d in (2, 3), got d={d}")

    # Handle automatic inversion when direction is backward and auto_invert is requested
    if direction in ('backward', 'inverse') and auto_invert:
        from syntx.core.inverse import update_inverse_field_nd_anderson
        disp_input = displacement_field
        if disp_input.dim() == d + 1:
            disp_input = disp_input.unsqueeze(0)
        disp_field = update_inverse_field_nd_anderson(
            disp_input, None, steps=inversion_steps, max_error_threshold=1e-5, mean_error_threshold=1e-6
        )
        if displacement_field.dim() == d + 1:
            disp_field = disp_field.squeeze(0)
    else:
        if hasattr(displacement_field, 'direction') or isinstance(displacement_field, str):
            from syntx.spatial import disp_itk_to_tensor
            disp_field = disp_itk_to_tensor(displacement_field, device=coords.device)
            disp_field.is_physical = True
        else:
            disp_field = displacement_field

    # Resolve domain bounds and displacement scaling factors
    bounds = _resolve_domain_bounds(domain_bounds, coords, d)
    if bounds is not None:
        min_b, max_b = bounds
        span = (max_b - min_b).clamp_min(1e-8)
        half_span = span * 0.5
    else:
        min_b = None
        max_b = None
        half_span = None

    # Determine whether displacement vectors require domain span scaling
    if scale_displacement is None:
        field_is_physical = is_physical if is_physical is not None else getattr(disp_field, 'is_physical', False)
        scale_displacement = (not field_is_physical) and (half_span is not None)
    elif is_physical is not None:
        scale_displacement = (not is_physical) and (half_span is not None)

    # Sample displacement vectors at coordinates
    u_eval = evaluate_field_at_scattered(
        field=disp_field,
        coords=coords,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        channel_dim=-1,
        coord_convention=coord_convention,
        domain_bounds=domain_bounds,
    )

    # Scale displacement if operating under normalized field coordinates with physical bounds
    if scale_displacement and half_span is not None:
        s = half_span.to(device=coords.device, dtype=coords.dtype)
        if u_eval.dim() == 3:
            s = s.view(1, 1, d)
        elif u_eval.dim() == 2:
            s = s.view(1, d)
        u_scaled = u_eval * s
    else:
        u_scaled = u_eval

    # Align displacement channels with coordinate convention
    # PyTorch displacement fields are stored with tensor component order (dz, dy, dx) or (dy, dx)
    if coord_convention == 'xyz':
        u_disp = torch.flip(u_scaled, dims=[-1])
    else:
        u_disp = u_scaled

    # Align batch dimension if coords is unbatched but u_disp is batched
    if coords.dim() == 2 and u_disp.dim() == 3:
        warped = coords.unsqueeze(0) + u_disp
    else:
        warped = coords + u_disp

    return warped


class ScatteredWarper(nn.Module):
    """Reusable PyTorch module for warping scattered coordinates through cached displacement fields."""
    def __init__(
        self,
        displacement_field: torch.Tensor,
        inverse_field: Optional[torch.Tensor] = None,
        domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'border',
        align_corners: bool = True,
        coord_convention: Literal['xyz', 'zyx'] = 'xyz',
        scale_displacement: Optional[bool] = None,
        inversion_steps: int = 20,
    ):
        super().__init__()
        self.register_buffer('displacement_field', displacement_field, persistent=False)
        if inverse_field is not None:
            self.register_buffer('inverse_field', inverse_field, persistent=False)
        else:
            self.inverse_field = None

        self.domain_bounds = domain_bounds
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.coord_convention = coord_convention
        self.scale_displacement = scale_displacement
        self.inversion_steps = inversion_steps

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward coordinate warp: phi(x) = x + u(x)."""
        return warp_scattered_coordinates(
            coords=coords,
            displacement_field=self.displacement_field,
            direction='forward',
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            domain_bounds=self.domain_bounds,
            scale_displacement=self.scale_displacement,
            coord_convention=self.coord_convention,
        )

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        """Backward coordinate warp: phi^-1(y) = y + v(y)."""
        if self.inverse_field is not None:
            field = self.inverse_field
            auto_inv = False
        else:
            field = self.displacement_field
            auto_inv = True

        return warp_scattered_coordinates(
            coords=coords,
            displacement_field=field,
            direction='backward',
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            domain_bounds=self.domain_bounds,
            scale_displacement=self.scale_displacement,
            auto_invert=auto_inv,
            inversion_steps=self.inversion_steps,
            coord_convention=self.coord_convention,
        )
