"""
syntx.scattered.mapping -- sample grid fields at points and move points through a field.

``evaluate_field_at_scattered`` samples a (B, *spatial, C) or (B, C, *spatial) grid at N
points with one ``F.grid_sample`` call (the points form a (B, 1, N, d) or (B, 1, 1, N, d)
query grid). ``warp_scattered_coordinates`` returns x + u(x) for a displacement field u,
handling coordinate bounds, displacement units and component order. ``ScatteredWarper``
stores a field (and optionally its inverse) for repeated use.

Coordinates: without ``domain_bounds`` points must already be in grid_sample units
[-1, 1]^d (-1 / +1 = first / last node, ``align_corners=True``). With bounds they are mapped
affinely from [min, max] to [-1, 1]. ``coord_convention='xyz'``: point component 0 runs along
the last tensor axis (grid_sample order); 'zyx': component k runs along tensor axis k.
"""

from typing import Optional, Tuple, Union, Sequence, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from syntx.spatial import reverse_components


def _resolve_domain_bounds(
    domain_bounds: Optional[Union[str, Tuple[float, float], Tuple[Sequence[float], Sequence[float]]]],
    coords: torch.Tensor,
    d: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Turn a ``domain_bounds`` spec into detached (min (d,), max (d,)) tensors, or None.

    None / 'none' -> None (coordinates are taken as already in [-1, 1]). 'auto' -> extent of
    ``coords`` (all batches) +- 5% of the span, the margin being at least 0.05 coordinate
    units per side ([-1, 1] if there are no points). Note that 'auto' derives the grid box
    from the query points themselves, so different point sets imply different boxes.
    ``(lo, hi)`` scalars or ``(lo_seq, hi_seq)`` per component are used as given.
    """
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
    """Sample a grid field (scalar, feature or displacement) at scattered points.

    Differentiable w.r.t. both the field and the coordinates (through ``F.grid_sample``).
    No component reordering or unit conversion is applied to the sampled values.

    Parameters
    ----------
    field : Tensor
        Grid of values; its spatial axes must be the d axes after batch / channel:
        - channel_dim=-1 (default): (B, *spatial, C) or (*spatial, C);
        - channel_dim=1 or 0: (B, C, *spatial) or (C, *spatial);
        - channel_dim=None: scalar (B, *spatial) or (*spatial).
        Non-tensor input is converted with ``torch.as_tensor``.
    coords : Tensor (N, d) or (B, N, d), d in {2, 3}
        Query points; cast to the field's device and dtype.
    mode : {'bilinear', 'nearest', 'bicubic'}, default 'bilinear'
        ``F.grid_sample`` mode ('bilinear' is trilinear in 3-D; 'bicubic' is 2-D only).
    padding_mode : {'border', 'zeros', 'reflection'}, default 'border'
    align_corners : bool, default True
        True matches the grids built in this package (nodes on the bounds).
    channel_dim : int or None, default -1
        See ``field``. Other values raise ValueError.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        'xyz': coords are passed to grid_sample as is (component 0 along the last tensor
        axis); 'zyx': components are reversed first.
    domain_bounds : tuple, 'auto' or None, default None
        If given, coords are mapped from [min, max] to [-1, 1] first (see
        ``_resolve_domain_bounds``; 'auto' uses the extent of ``coords``). None: coords are
        already in [-1, 1].

    Returns
    -------
    Tensor (B, N, C); (N, C) when coords were unbatched and the field batch is 1. A field
    batch of 1 is broadcast to the coords batch and vice versa; other mismatches raise.
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
        grid_coords = reverse_components(norm_coords)
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


def _invert_displacement(source, domain_bounds, coords, d, coord_convention, src_physical, src_vc,
                         inversion_steps):
    """Anderson inverse of a displacement field in its own units: a physical field on the grid
    of ``domain_bounds`` (required) with tensor-order components, a normalised field with
    (x, y, z) components. Keeps ``is_physical`` / ``vector_convention``."""
    from syntx.core.inverse import update_inverse_field_nd_anderson
    disp_input = source if source.dim() == d + 2 else source.unsqueeze(0)
    if src_physical:
        b = _resolve_domain_bounds(domain_bounds, coords, d)
        if b is None:
            raise ValueError("auto_invert of a physical field needs domain_bounds (its box)")
        lo, hi = (x.detach().cpu().double().numpy() for x in b)
        if coord_convention == 'zyx':
            lo, hi = lo[::-1], hi[::-1]                     # -> (x, y, z)
        n_xyz = np.asarray(list(reversed(disp_input.shape[1:-1])), dtype=float)
        sp_xyz = (hi - lo) / np.maximum(n_xyz - 1, 1)
        comp_tensor = (src_vc or 'zyx') == 'zyx'
        w = disp_input if comp_tensor else torch.flip(disp_input, dims=[-1])
        w_inv = update_inverse_field_nd_anderson(
            w, None, steps=inversion_steps, max_error_threshold=1e-5, mean_error_threshold=1e-6,
            spacing=tuple(sp_xyz), origin=tuple(lo), direction=np.eye(d))
        out = w_inv if comp_tensor else torch.flip(w_inv, dims=[-1])
    else:
        comp_xyz = (src_vc or ('zyx' if (d == 3 and coord_convention == 'xyz') else 'xyz')) == 'xyz'
        w = disp_input if comp_xyz else torch.flip(disp_input, dims=[-1])
        w_inv = update_inverse_field_nd_anderson(
            w, None, steps=inversion_steps, max_error_threshold=1e-5, mean_error_threshold=1e-6)
        out = w_inv if comp_xyz else torch.flip(w_inv, dims=[-1])
    if source.dim() == d + 1:
        out = out.squeeze(0)
    out.is_physical = src_physical
    if src_vc is not None:
        out.vector_convention = src_vc
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
    vector_convention: Optional[Literal['xyz', 'zyx']] = None,
) -> torch.Tensor:
    """Move points through a displacement field: return x + u(x).

    Steps: (1) pick the field (optionally invert it, or convert an ANTs image); (2) sample it
    at ``coords`` with ``evaluate_field_at_scattered`` (channels-last); (3) optionally scale
    the sampled vectors from [-1, 1] units to coordinate units by (max - min) / 2; (4) reorder
    components to ``coord_convention``; (5) add to ``coords``. Differentiable w.r.t. coords
    and field.

    Parameters
    ----------
    coords : Tensor (N, d) or (B, N, d), d in {2, 3}
        Points, in coordinate units of ``domain_bounds`` (or [-1, 1] when it is None).
    displacement_field : Tensor (*spatial, d) or (B, *spatial, d), ANTsImage or file path
        Channels-last displacement. An object with a ``direction`` attribute (ANTsImage) or a
        str is converted with ``syntx.spatial.disp_itk_to_tensor`` (mm, tensor-order
        components, ``is_physical`` set); it must be axis-aligned (ValueError otherwise) and,
        when ``domain_bounds`` is None, its box comes from the image origin / spacing / size.
        Tensor attributes ``is_physical`` and ``vector_convention`` are read if present.
    direction : {'forward', 'backward', 'inverse'}, default 'forward'
        Only matters together with ``auto_invert``: with 'backward' / 'inverse' and
        ``auto_invert=True`` the field is first inverted; otherwise the given field is used
        as is in every direction (pass the inverse field yourself).
    mode : str, default 'bilinear'
        Sampling mode for the field ('bilinear', 'nearest'; 'bicubic' 2-D only).
    padding_mode : str, default 'border'
    align_corners : bool, default True
    domain_bounds : tuple, 'auto' or None, default None
        Box of the field's grid in coordinate units (see ``_resolve_domain_bounds``). None:
        coordinates are already [-1, 1] (or, for an ANTs image, its own box). 'auto' raises
        ValueError (the box of a field cannot be derived from the query points).
    scale_displacement : bool, optional
        Multiply sampled vectors by (max - min) / 2 per component (converts [-1, 1] units to
        coordinate units). If None: True when bounds are given and the field is not physical.
        Giving both this and a contradicting ``is_physical`` raises ValueError.
    auto_invert : bool, default False
        With 'backward' / 'inverse': invert the field with
        ``syntx.core.inverse.update_inverse_field_nd_anderson`` in its own units (thresholds
        1e-5 / 1e-6): a physical field on the grid of ``domain_bounds`` (required) with
        tensor-order components, a normalised field with (x, y, z) components. The result
        keeps ``is_physical`` / ``vector_convention``.
    inversion_steps : int, default 20
        Maximum Anderson iterations when ``auto_invert`` is used.
    coord_convention : {'xyz', 'zyx'}, default 'xyz'
        Component order of ``coords`` (see module docstring).
    is_physical : bool, optional
        Declares the field to be in coordinate units already (no scaling). None: read the
        field's ``is_physical`` attribute (False if absent).
    vector_convention : {'xyz', 'zyx'}, optional
        Component order of the field's vectors. Components are reversed when it differs from
        ``coord_convention``. If None: the field's ``vector_convention`` attribute; else 'zyx'
        for ANTs input; else 'zyx' when d == 3 and coord_convention == 'xyz'; else 'xyz'. So
        for 3-D fields without the attribute, components are always reversed relative to the
        coordinates; pass ``vector_convention`` explicitly for (x, y, z)-component fields
        such as the ones made by ``SyNScattered``.

    Returns
    -------
    Tensor (N, d) or (B, N, d)
        Warped points; (B, N, d) if coords were unbatched but the field batch is > 1.
    """
    if direction not in ('forward', 'backward', 'inverse'):
        raise ValueError(f"Unknown direction '{direction}', expected 'forward', 'backward', or 'inverse'")

    d = coords.shape[-1]
    if d not in (2, 3):
        raise ValueError(f"Expected coords spatial dimension d in (2, 3), got d={d}")

    if isinstance(domain_bounds, str) and domain_bounds == 'auto':
        raise ValueError("domain_bounds='auto' would derive the field's box from the query points; "
                         "pass the box of the displacement field's grid")
    if scale_displacement is not None and is_physical is not None and bool(scale_displacement) == bool(is_physical):
        raise ValueError(f"scale_displacement={scale_displacement} contradicts is_physical={is_physical} "
                         "(a physical field is not scaled); pass one of them")

    # ANTs images / files: tensor-order mm displacement; the box comes from the image geometry
    if hasattr(displacement_field, 'direction') or isinstance(displacement_field, str):
        import ants
        from syntx.spatial import disp_itk_to_tensor
        img = ants.image_read(displacement_field) if isinstance(displacement_field, str) else displacement_field
        if not np.allclose(np.asarray(img.direction), np.eye(img.dimension), atol=1e-6):
            raise ValueError("warp_scattered_coordinates needs an axis-aligned displacement image (the "
                             "grid box cannot describe an oblique one); resample it first")
        if domain_bounds is None:
            lo_xyz = np.asarray(img.origin, dtype=float)
            hi_xyz = lo_xyz + np.asarray(img.spacing, dtype=float) * (np.asarray(img.shape) - 1)
            domain_bounds = ((tuple(lo_xyz), tuple(hi_xyz)) if coord_convention == 'xyz'
                             else (tuple(lo_xyz[::-1]), tuple(hi_xyz[::-1])))
        source = disp_itk_to_tensor(img, device=coords.device)
        source.is_physical = True
        source.vector_convention = 'zyx'
    else:
        source = displacement_field
    src_physical = bool(is_physical) if is_physical is not None else bool(getattr(source, 'is_physical', False))
    src_vc = vector_convention or getattr(source, 'vector_convention', None)

    # Handle automatic inversion when direction is backward and auto_invert is requested
    if direction in ('backward', 'inverse') and auto_invert:
        disp_field = _invert_displacement(source, domain_bounds, coords, d, coord_convention,
                                          src_physical, src_vc, inversion_steps)
    else:
        disp_field = source

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
    if vector_convention is None:
        vector_convention = getattr(disp_field, 'vector_convention', None)
        if vector_convention is None:
            vector_convention = getattr(displacement_field, 'vector_convention', None)

    if vector_convention is None:
        if hasattr(displacement_field, 'direction') or isinstance(displacement_field, str):
            vector_convention = 'zyx'
        elif d == 3 and coord_convention == 'xyz':
            vector_convention = 'zyx'
        else:
            vector_convention = 'xyz'

    if coord_convention == 'xyz' and vector_convention == 'zyx':
        u_disp = reverse_components(u_scaled)
    elif coord_convention == 'zyx' and vector_convention == 'xyz':
        u_disp = reverse_components(u_scaled)
    else:
        u_disp = u_scaled

    # Align batch dimension if coords is unbatched but u_disp is batched
    if coords.dim() == 2 and u_disp.dim() == 3:
        warped = coords.unsqueeze(0) + u_disp
    else:
        warped = coords + u_disp

    return warped


class ScatteredWarper(nn.Module):
    """Module holding a displacement field (and optionally its inverse) for warping points.

    Parameters
    ----------
    displacement_field : Tensor (*spatial, d) or (B, *spatial, d)
        Forward field, channels-last; stored as a non-persistent buffer (moving the module
        to another device makes a new tensor, which drops ``is_physical`` /
        ``vector_convention`` attributes).
    inverse_field : Tensor, optional
        Inverse field. If None, ``inverse`` inverts ``displacement_field`` once (on first use;
        cached until ``displacement_field`` changes).
    domain_bounds, mode, padding_mode, align_corners, coord_convention, scale_displacement :
        Passed to ``warp_scattered_coordinates``.
    inversion_steps : int, default 20
        Anderson iterations for the on-the-fly inverse.

    is_physical, vector_convention : optional
        Passed to ``warp_scattered_coordinates`` (default: the field tensor's attributes at
        construction, so they survive moving the module to another device).
    """
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
        is_physical: Optional[bool] = None,
        vector_convention: Optional[Literal['xyz', 'zyx']] = None,
    ):
        super().__init__()
        self.is_physical = is_physical if is_physical is not None else getattr(displacement_field, 'is_physical', None)
        self.vector_convention = vector_convention or getattr(displacement_field, 'vector_convention', None)
        self._inverse_cache = None
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
        """Return coords + u(coords) using ``displacement_field``."""
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
            is_physical=self.is_physical,
            vector_convention=self.vector_convention,
        )

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        """Return coords + v(coords), v = ``inverse_field`` or, if None, the Anderson inverse
        of ``displacement_field`` (computed once and cached)."""
        if self.inverse_field is not None:
            field = self.inverse_field
        else:
            u = self.displacement_field
            key = (u.data_ptr(), u._version, u.device, u.dtype)
            if self._inverse_cache is None or self._inverse_cache[0] != key:
                d = u.shape[-1]
                inv = _invert_displacement(u, self.domain_bounds, coords, d, self.coord_convention,
                                           bool(self.is_physical), self.vector_convention, self.inversion_steps)
                self._inverse_cache = (key, inv)
            field = self._inverse_cache[1]
        return warp_scattered_coordinates(
            coords=coords,
            displacement_field=field,
            direction='forward',
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            domain_bounds=self.domain_bounds,
            scale_displacement=self.scale_displacement,
            coord_convention=self.coord_convention,
            is_physical=self.is_physical,
            vector_convention=self.vector_convention,
        )
