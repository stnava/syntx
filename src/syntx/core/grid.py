"""
Sampling primitives on normalised [-1, 1] grids (``F.grid_sample`` conventions, coordinates in
(x, y, z) order, ``align_corners=True`` unless stated): ``grid_sample_nd`` (interpolator /
padding dispatch, ITK-style padding, label sampling), cubic B-spline sampling, an analytical
grid-gradient autograd function, a bit-reproducible bilinear backward for GPU / MPS, field
composition / resizing helpers and the SyN midpoint image / gradient preparation step. The
physical <-> normalised grid helpers re-exported here live in ``syntx.spatial``.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def grid_sample_bspline_torch(
    image: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = 'border',
    align_corners: bool = True
) -> torch.Tensor:
    """
    Sample an image with cubic B-spline weights at normalised grid points (2-D / 3-D).

    The 4 x 4 (x 4) neighbourhood of voxel values is weighted with the cubic B-spline basis
    directly; there is no B-spline prefilter, so the result is a smoothing approximation
    (it does not reproduce the voxel values at voxel centres), not an interpolation.

    Parameters
    ----------
    image : Tensor (B, C, H, W) or (B, C, D, H, W)
        Must be viewable as (B, C, N) (contiguous).
    grid : Tensor (B, *out_spatial, 2 or 3)
        Normalised coordinates in (x, y[, z]) order.
    padding_mode : {'border', 'zeros'}, default 'border'
        'zeros': taps outside the image contribute 0. 'border': taps are clamped to the edge
        voxel. Other values raise ValueError.
    align_corners : bool, default True
        Same meaning as in ``F.grid_sample``.

    Returns
    -------
    Tensor (B, C, *out_spatial). Differentiable with respect to ``grid`` and ``image`` through
    autograd.

    Raises
    ------
    ValueError
        If the image is not 2-D or 3-D, or for an unknown ``padding_mode``.
    """
    ndim = image.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(f"Only 2D and 3D grid sampling supported, got ndim={ndim}")
    if padding_mode not in ('zeros', 'border'):
        raise ValueError(f"padding_mode must be 'zeros' or 'border', got {padding_mode!r}")

    B, C = image.shape[:2]
    device = image.device
    dtype = image.dtype
    spatial_target = grid.shape[1:-1]

    def get_w(u):
        u2 = u * u
        u3 = u2 * u
        w0 = (1.0 - u)**3 / 6.0
        w1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
        w2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
        w3 = u3 / 6.0
        return [w0, w1, w2, w3]

    if ndim == 2:
        H, W = image.shape[2:]
        gx, gy = grid[..., 0], grid[..., 1]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        ix, iy = torch.floor(vx), torch.floor(vy)
        ux, uy = vx - ix, vy - iy
        wx, wy = get_w(ux), get_w(uy)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, H * W)

        for ky in range(4):
            jy_raw = (iy + ky - 1).long()
            jy = jy_raw.clamp(0, H - 1)
            valid_y = (jy_raw >= 0) & (jy_raw < H)
            w_y = wy[ky].unsqueeze(1)
            for kx in range(4):
                jx_raw = (ix + kx - 1).long()
                jx = jx_raw.clamp(0, W - 1)
                valid_x = (jx_raw >= 0) & (jx_raw < W)
                w_yx = w_y * wx[kx].unsqueeze(1)
                idx = (jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                sampled_flat = torch.gather(img_flat, 2, idx)
                sampled = sampled_flat.view(B, C, *spatial_target)
                if padding_mode == 'zeros':
                    valid_mask = (valid_y & valid_x).view(B, 1, *spatial_target)
                    sampled = sampled * valid_mask
                out = out + w_yx * sampled
        return out
    else:
        D, H, W = image.shape[2:]
        gx, gy, gz = grid[..., 0], grid[..., 1], grid[..., 2]
        vx = (gx + 1.0) * (W - 1) / 2.0 if align_corners else (gx + 1.0) * W / 2.0 - 0.5
        vy = (gy + 1.0) * (H - 1) / 2.0 if align_corners else (gy + 1.0) * H / 2.0 - 0.5
        vz = (gz + 1.0) * (D - 1) / 2.0 if align_corners else (gz + 1.0) * D / 2.0 - 0.5
        ix, iy, iz = torch.floor(vx), torch.floor(vy), torch.floor(vz)
        ux, uy, uz = vx - ix, vy - iy, vz - iz
        wx, wy, wz = get_w(ux), get_w(uy), get_w(uz)

        out = torch.zeros((B, C, *spatial_target), device=device, dtype=dtype)
        img_flat = image.view(B, C, D * H * W)

        for kz in range(4):
            jz_raw = (iz + kz - 1).long()
            jz = jz_raw.clamp(0, D - 1)
            valid_z = (jz_raw >= 0) & (jz_raw < D)
            w_z = wz[kz].unsqueeze(1)
            for ky in range(4):
                jy_raw = (iy + ky - 1).long()
                jy = jy_raw.clamp(0, H - 1)
                valid_y = (jy_raw >= 0) & (jy_raw < H)
                w_zy = w_z * wy[ky].unsqueeze(1)
                for kx in range(4):
                    jx_raw = (ix + kx - 1).long()
                    jx = jx_raw.clamp(0, W - 1)
                    valid_x = (jx_raw >= 0) & (jx_raw < W)
                    w_zyx = w_zy * wx[kx].unsqueeze(1)
                    idx = (jz * (H * W) + jy * W + jx).view(B, 1, -1).expand(B, C, -1)
                    sampled_flat = torch.gather(img_flat, 2, idx)
                    sampled = sampled_flat.view(B, C, *spatial_target)
                    if padding_mode == 'zeros':
                        valid_mask = (valid_z & valid_y & valid_x).view(B, 1, *spatial_target)
                        sampled = sampled * valid_mask
                    out = out + w_zyx * sampled
        return out


def _image_spatial_gradient(image):
    """Image gradient in voxel units, (B, C, dim, *spatial) with the dim axis in (x, y, z)
    order: ``torch.gradient`` (central differences, one-sided at the border -- no wrap-around).
    None for images that are not 2-D / 3-D."""
    dim = image.dim() - 2
    if dim not in (2, 3):
        return None
    grads = torch.gradient(image, dim=tuple(range(2, 2 + dim)))   # tensor order (z, y, x)
    return torch.stack(grads[::-1], dim=2)


class AnalyticalGridSample(torch.autograd.Function):
    """
    ``F.grid_sample`` whose backward returns an approximate gradient for ``grid`` only.

    Forward: ``F.grid_sample(input, grid, mode, padding_mode, align_corners)`` (``input`` cast
    to the grid dtype). Backward: the image gradient (``precomputed_grad_I`` or
    ``_image_spatial_gradient(input)``, central differences, one-sided at the border) is
    sampled at ``grid`` with the same mode / padding, contracted with ``grad_output`` over
    channels and scaled from voxels to normalised units ((n - 1) / 2, or n / 2 when
    ``align_corners=False``). This is the gradient of the interpolated image gradient, not the
    exact derivative of the bilinear sampler; no gradient flows to ``input``.

    ``precomputed_grad_I`` must have the ``_image_spatial_gradient`` layout
    (B, C, dim, *spatial); computing it once is worthwhile when the same image is sampled in
    many optimiser iterations.
    """

    @staticmethod
    def forward(ctx, input, grid, mode='bilinear', padding_mode='border', align_corners=True,
                precomputed_grad_I=None):
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.save_for_backward(input, grid, precomputed_grad_I)
        if input.dtype != grid.dtype:
            input = input.to(grid.dtype)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, precomputed_grad_I = ctx.saved_tensors
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners

        dim = input.dim() - 2
        spatial_shape = input.shape[2:]
        B, C = input.shape[:2]

        # 1. Spatial gradients of the source input image: dI/dx, dI/dy, dI/dz. The image is
        # typically held FIXED across many optimizer iterations (only the sampling grid
        # changes), so callers doing repeated grid_sample_nd calls against the same image
        # (e.g. an LBFGS/Adam inner loop) should compute this ONCE via
        # `_image_spatial_gradient` and pass it as `precomputed_grad_I` -- recomputing it on
        # every backward call is a real, measured cost (the dominant per-iteration cost in
        # syntx.motion_batched's batched solver before this was added).
        if precomputed_grad_I is not None:
            grad_I = precomputed_grad_I
        else:
            grad_I = _image_spatial_gradient(input)  # (B, C, dim, *spatial_shape)

        # 2. Sample source gradients at grid lookup coordinates G (matching dtype with grid)
        grad_I_flat = grad_I.reshape(B, C * dim, *spatial_shape).to(dtype=grid.dtype)
        grad_I_sampled = F.grid_sample(grad_I_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        grad_I_sampled = grad_I_sampled.reshape(B, C, dim, *grid.shape[1:-1])  # (B, C, dim, *spatial_grid)

        # 3. Inner product with incoming loss gradient grad_output (B, C, *spatial_grid)
        grad_out_cast = grad_output.to(dtype=grid.dtype)
        grad_grid = torch.sum(grad_out_cast.unsqueeze(2) * grad_I_sampled, dim=1).movedim(1, -1)  # (B, *spatial_grid, dim)

        # 4. Apply voxel-to-normalized grid coordinate scaling
        scales = []
        for d in range(dim):
            size = spatial_shape[dim - 1 - d]  # X is dim - 1, Y is dim - 2, Z is dim - 3
            s = (size - 1) / 2.0 if align_corners else size / 2.0
            scales.append(s)
        scale_t = torch.tensor(scales, dtype=grad_grid.dtype, device=grad_grid.device)
        grad_grid = grad_grid * scale_t

        return None, grad_grid, None, None, None, None


def itk_inside_mask(grid, spatial_shape):
    """
    1 where a normalised ``align_corners=True`` point is inside the ITK image buffer, else 0.

    A point is inside when its continuous index lies in ``[-0.5, N - 0.5)`` on every axis
    (half a voxel beyond the edge voxel centres); axes of size 1 accept every point.

    Parameters
    ----------
    grid : Tensor (B, *out_spatial, dim)
        (x, y[, z]) order.
    spatial_shape : sequence of int
        Image spatial shape in tensor order (z, y, x).

    Returns
    -------
    Tensor (B, *out_spatial, 1) in the grid dtype.
    """
    sizes = torch.tensor(list(reversed(tuple(spatial_shape))), device=grid.device, dtype=grid.dtype)
    half = torch.where(sizes > 1, 1.0 / (sizes - 1).clamp(min=1), torch.full_like(sizes, float('inf')))
    inside = ((grid >= -1.0 - half) & (grid < 1.0 + half)).all(dim=-1, keepdim=True)
    return inside.to(grid.dtype)


def _generic_label_sample(input, grid, padding_mode='itk', align_corners=True):
    """
    Label sampling in the style of ITK's LabelImageGenericInterpolateImageFunction
    (ANTs 'genericLabel').

    Each label's indicator image is sampled bilinearly / trilinearly and the label with the
    largest weight wins (ties go to the lowest label: strict ``>`` over ascending labels).
    With ``padding_mode='itk'`` the indicators use border padding and points outside
    ``itk_inside_mask`` are set to 0; with 'zeros', points where every indicator weight is 0
    (outside the image) get 0; with 'border', the nearest edge labels. Returns the input
    dtype. Not differentiable.
    """
    labels = torch.unique(input)
    best_w = None
    best_l = None
    for lab in labels:  # ascending
        ind = (input == lab).to(grid.dtype)
        w = F.grid_sample(ind, grid, mode='bilinear',
                          padding_mode='border' if padding_mode == 'itk' else padding_mode,
                          align_corners=align_corners)
        if best_w is None:
            best_w = w
            best_l = torch.full_like(w, float(lab))
        else:
            better = w > best_w
            best_w = torch.where(better, w, best_w)
            best_l = torch.where(better, torch.full_like(w, float(lab)), best_l)
    if padding_mode == 'itk':
        best_l = best_l * torch.movedim(itk_inside_mask(grid, input.shape[2:]), -1, 1)
    elif padding_mode == 'zeros':
        best_l = torch.where(best_w > 0, best_l, torch.zeros_like(best_l))
    return best_l.to(input.dtype)


_KNOWN_INTERPOLATORS = frozenset({
    None, 'linear', 'bilinear', 'trilinear', 'bicubic', 'bspline',
    'nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor',
    'genericLabel', 'generic_label', 'GenericLabel',
})


def grid_sample_nd(input, grid, mode='bilinear', padding_mode='border', align_corners=True,
                    interpolator='linear', use_analytical_gradients=True, precomputed_grad_I=None):
    """
    Sample ``input`` at normalised grid points, choosing the sampler from the options.

    Checked in this order:

    1. ``interpolator`` in {'genericLabel', 'generic_label', 'GenericLabel'} or ``mode`` in
       {'genericLabel', 'generic_label'}: ``_generic_label_sample`` (no gradient).
    2. ``padding_mode='itk'``: the call is repeated with 'border' padding and multiplied by
       ``itk_inside_mask`` (values edge-clamped out to half a voxel beyond the edge voxel
       centres, 0 beyond that, as ITK resamplers do).
    3. nearest-neighbour names in ``interpolator`` or ``mode`` set ``mode='nearest'``.
    4. ``interpolator`` or ``mode`` 'bspline': ``grid_sample_bspline_torch``.
    5. ``input`` is cast to the grid dtype. If ``use_analytical_gradients`` and the grid
       requires grad and the input does not: ``AnalyticalGridSample`` (approximate grid
       gradient, any ``mode``).
    6. If the input requires grad (grad mode on), ``mode='bilinear'``, ``align_corners``,
       padding 'border' / 'zeros' and the device type is in ``DETERMINISTIC_SAMPLE_DEVICES``:
       ``DeterministicGridSample``.
    7. Otherwise ``F.grid_sample``.

    The linear names ('linear', 'bilinear', 'trilinear') and None leave ``mode`` unchanged;
    'bicubic' sets ``mode='bicubic'``; any other ``interpolator`` raises ValueError.

    Parameters
    ----------
    input : Tensor (B, C, *spatial)
    grid : Tensor (B, *out_spatial, dim)
        Normalised coordinates, (x, y[, z]) order.
    mode : str, default 'bilinear'
        ``F.grid_sample`` mode, or one of the names above.
    padding_mode : str, default 'border'
        Any ``F.grid_sample`` padding mode, or 'itk'.
    align_corners : bool, default True
    interpolator : str, default 'linear'
        ANTs-style interpolator name; see the order above.
    use_analytical_gradients : bool, default True
        See step 5.
    precomputed_grad_I : Tensor, optional
        ``_image_spatial_gradient(input)``, used only by ``AnalyticalGridSample``. Pass it
        when the same ``input`` is sampled in many iterations while only ``grid`` changes.

    Returns
    -------
    Tensor (B, C, *out_spatial).
    """
    if interpolator not in _KNOWN_INTERPOLATORS:
        raise ValueError(f"unknown interpolator {interpolator!r}; expected one of "
                         f"{sorted(i for i in _KNOWN_INTERPOLATORS if i is not None)} or None")
    if interpolator == 'bicubic':
        mode = 'bicubic'
    if interpolator in ('genericLabel', 'generic_label', 'GenericLabel') or mode in ('genericLabel', 'generic_label'):
        return _generic_label_sample(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    if padding_mode == 'itk':
        mask = itk_inside_mask(grid, input.shape[2:])
        out = grid_sample_nd(input, grid, mode=mode, padding_mode='border',
                             align_corners=align_corners, interpolator=interpolator,
                             use_analytical_gradients=use_analytical_gradients,
                             precomputed_grad_I=precomputed_grad_I)
        return out * torch.movedim(mask, -1, 1)
    if interpolator in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor') or mode in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor'):
        mode = 'nearest'
    if interpolator == 'bspline' or mode == 'bspline':
        return grid_sample_bspline_torch(input, grid, padding_mode=padding_mode, align_corners=align_corners)
    if input.dtype != grid.dtype:
        input = input.to(grid.dtype)
    if use_analytical_gradients and grid.requires_grad and not input.requires_grad:
        return AnalyticalGridSample.apply(input, grid, mode, padding_mode, align_corners, precomputed_grad_I)
    if (input.requires_grad and torch.is_grad_enabled() and mode == 'bilinear' and align_corners
            and padding_mode in ('border', 'zeros') and input.device.type in DETERMINISTIC_SAMPLE_DEVICES):
        return DeterministicGridSample.apply(input, grid, padding_mode)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def compose_grids(grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
    """
    Compose two normalised coordinate grids: ``grid1(grid2(x))``.

    ``grid1`` is treated as a field and sampled bilinearly / trilinearly with border padding
    and ``align_corners=True`` at the normalised positions ``grid2`` (via
    ``sample_field_cf``).

    Parameters
    ----------
    grid1 : Tensor (B, *spatial1, dim)
        Coordinates (any units) stored on a grid; ``grid2`` indexes this grid.
    grid2 : Tensor (B, *spatial2, dim)
        Normalised positions, (x, y[, z]) order. ``spatial2`` may differ from ``spatial1``.

    Returns
    -------
    Tensor (B, *spatial2, dim)
    """
    grid1_cf = torch.movedim(grid1, -1, 1)   # → (B, dim, *spatial) channel-first
    return sample_field_cf(grid1_cf, grid2, mode='bilinear', padding_mode='border')  # → last-channel


def resize_field(field: torch.Tensor, size, mode: str = None) -> torch.Tensor:
    """
    Resize a channels-last field to a new spatial size with ``F.interpolate``
    (``align_corners=True``).

    Values are not rescaled, so this is correct for normalised or physical-unit fields but
    not for voxel-unit displacements.

    Parameters
    ----------
    field : Tensor (B, *spatial, dim)
    size : sequence of int
        Target spatial size, tensor order (z, y, x).
    mode : str, optional
        ``F.interpolate`` mode. None: 'trilinear' when the component count is 3, otherwise
        'bilinear'.

    Returns
    -------
    Tensor (B, *size, dim)
    """
    dim = field.shape[-1]
    if mode is None:
        mode = 'trilinear' if dim == 3 else 'bilinear'
    field_cf = torch.movedim(field, -1, 1)                                       # (B, dim, *spatial)
    resized_cf = F.interpolate(field_cf, size=size, mode=mode, align_corners=True)
    return torch.movedim(resized_cf, 1, -1)                                       # (B, *size, dim)


def sample_field_cf(field_cf: torch.Tensor, grid: torch.Tensor,
                    mode: str = 'bilinear', padding_mode: str = 'border') -> torch.Tensor:
    """
    Sample a channels-first field at normalised positions and return it channels-last.

    ``align_corners=True``. When either input requires grad (grad mode on), ``mode`` is
    'bilinear', padding is 'border' / 'zeros' and the device type is in
    ``DETERMINISTIC_SAMPLE_DEVICES``, ``DeterministicGridSample`` is used; otherwise
    ``F.grid_sample``.

    Parameters
    ----------
    field_cf : Tensor (B, C, *spatial)
    grid : Tensor (B, *spatial_out, dim)
        Normalised coordinates, (x, y[, z]) order.
    mode : str, default 'bilinear'
    padding_mode : str, default 'border'

    Returns
    -------
    Tensor (B, *spatial_out, C)
    """
    if ((field_cf.requires_grad or grid.requires_grad) and torch.is_grad_enabled() and mode == 'bilinear'
            and padding_mode in ('border', 'zeros') and field_cf.device.type in DETERMINISTIC_SAMPLE_DEVICES):
        sampled_cf = DeterministicGridSample.apply(field_cf, grid, padding_mode)
    else:
        sampled_cf = F.grid_sample(field_cf, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return torch.movedim(sampled_cf, 1, -1)   # → (B, *spatial_out, dim)


# Devices on which sample_field_cf routes differentiable fields through DeterministicGridSample.
# CPU grid_sample is already deterministic; set this to () to use the stock kernel everywhere.
DETERMINISTIC_SAMPLE_DEVICES = ('mps', 'cuda')


def _fixed_point_scale(vmax, n_terms):
    """Power-of-two scale for exact int64 accumulation of n_terms values bounded by vmax:
    every partial sum stays below n_terms * vmax * scale <= 2^62."""
    return 2.0 ** math.floor(math.log2(2.0 ** 62 / (vmax * n_terms)))


def _fixed_point_scatter_add(n_bins, index, values):
    """Order-independent (hence bit-reproducible) ``zeros(n_bins).index_add_(0, index, values)``.

    Float atomics sum in whatever order threads arrive, so the float result varies from run to
    run on GPU/MPS. Integer addition is associative, so the values are quantised to int64 with a
    power-of-two scale chosen so that no partial sum can overflow, summed exactly, and converted
    back. The quantum is at most 2 * max|v| * N * 2^-62. Zero or non-finite inputs fall back to a
    plain float ``index_add_``.
    """
    vmax = float(values.abs().max()) if values.numel() else 0.0
    if vmax == 0.0 or not math.isfinite(vmax):
        return torch.zeros(n_bins, dtype=values.dtype, device=values.device).index_add_(0, index, values)
    scale = _fixed_point_scale(vmax, values.numel())
    acc = torch.zeros(n_bins, dtype=torch.int64, device=values.device)
    acc.index_add_(0, index, torch.round(values * scale).to(torch.int64))
    return acc.to(values.dtype) / scale


class DeterministicGridSample(torch.autograd.Function):
    """``F.grid_sample(input, grid, 'bilinear', padding_mode, align_corners=True)`` with
    run-to-run reproducible gradients.

    The stock GPU/MPS backward (``grid_sampler_{2,3}d_backward``) is non-deterministic: the
    input gradient is a float-atomic scatter, and on MPS the grid gradient varies too.
    Optimisers that sample their trainable field many times per iteration (e.g. syngs'
    geodesic shooting) amplify the ~1e-7 differences into different registrations. Here the
    forward is the stock kernel; the backward is the Metal kernel
    ``mps_kernels.grid_sample_backward_mps`` for float32 on MPS, otherwise
    ``_deterministic_grid_sample_backward`` (grid gradient from gathered corner values in a
    fixed order, input gradient by int64 fixed-point accumulation). ``padding_mode`` must be
    'border' or 'zeros' (ValueError otherwise).
    """

    @staticmethod
    def forward(ctx, input, grid, padding_mode):
        if padding_mode not in ('zeros', 'border'):
            raise ValueError(f"padding_mode must be 'zeros' or 'border', got {padding_mode!r}")
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = padding_mode
        return F.grid_sample(input, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)

    @staticmethod
    def backward(ctx, grad_out):
        input, grid = ctx.saved_tensors
        if input.device.type == 'mps' and input.dtype == torch.float32 and grid.dtype == torch.float32:
            from .mps_kernels import grid_sample_backward_mps
            grad_input, grad_grid = grid_sample_backward_mps(
                grad_out, input, grid, ctx.padding_mode,
                need_input=ctx.needs_input_grad[0], need_grid=ctx.needs_input_grad[1])
            return grad_input, grad_grid, None
        grad_input, grad_grid = _deterministic_grid_sample_backward(
            grad_out, input, grid, ctx.padding_mode,
            need_input=ctx.needs_input_grad[0], need_grid=ctx.needs_input_grad[1])
        return grad_input, grad_grid, None


def _deterministic_grid_sample_backward(grad_out, input, grid, padding_mode, need_input=True, need_grid=True):
    """Gradients of bilinear/trilinear grid_sample (align_corners=True, 'zeros'/'border' padding)
    w.r.t. input (exact int64 fixed-point scatter) and grid (gather-only), both bit-reproducible.
    Out-of-bounds corners carry value 0, and border-clipped coordinates get zero grid gradient,
    following ATen's conventions. Any padding_mode other than 'border' is handled as 'zeros'.
    A non-finite grad_out gives an all-NaN input gradient."""
    input_shape = input.shape
    B, C = input_shape[:2]
    spatial = input_shape[2:]                      # (D,) H, W
    nd = len(spatial)
    n_pts = grid[..., 0].numel() // B
    g = grid.reshape(B, n_pts, nd)
    go = grad_out.reshape(B, C, n_pts)
    # grid[..., 0] indexes the LAST spatial axis (x -> W), grid[..., -1] the first.
    coords, sizes, dudg = [], [], []
    for k in range(nd):
        size = spatial[nd - 1 - k]
        u = (g[..., k] + 1.0) * 0.5 * (size - 1)
        d = torch.full_like(u, 0.5 * (size - 1))
        if padding_mode == 'border':
            d = torch.where((u > 0) & (u < size - 1), d, torch.zeros_like(d))
            u = u.clamp(0.0, size - 1)
        coords.append(u)
        sizes.append(size)
        dudg.append(d)
    floors = [torch.floor(u) for u in coords]
    fracs = [u - f for u, f in zip(coords, floors)]
    floors = [f.to(torch.int64) for f in floors]
    strides = []                                   # flat stride of axis k (x first)
    s = 1
    for size in sizes:
        strides.append(s)
        s *= size
    n_vox = s

    def corner(c):
        """(flat voxel index, validity, per-axis factor list) of corner bitmask c."""
        flat = torch.zeros_like(floors[0])
        valid = torch.ones_like(floors[0], dtype=torch.bool)
        facs = []
        for k in range(nd):
            hi = (c >> k) & 1
            i = floors[k] + hi
            facs.append(fracs[k] if hi else (1.0 - fracs[k]))
            valid = valid & (i >= 0) & (i < sizes[k])
            flat = flat + i.clamp(0, sizes[k] - 1) * strides[k]
        return flat, valid, facs

    grad_input = grad_grid = None
    if need_grid:
        inp = input.reshape(B, C, n_vox)
        gu = [torch.zeros_like(fracs[0]) for _ in range(nd)]
        for c in range(2 ** nd):
            flat, valid, facs = corner(c)
            val = torch.gather(inp, 2, flat.unsqueeze(1).expand(B, C, n_pts))
            # sum_c grad_out_c * value_c (fixed channel order), zero for out-of-bounds corners
            gv = (go * val).sum(1) * valid.to(go.dtype)
            for k in range(nd):
                dw = torch.ones_like(fracs[0]) if (c >> k) & 1 else -torch.ones_like(fracs[0])
                for j in range(nd):
                    if j != k:
                        dw = dw * facs[j]
                gu[k] = gu[k] + gv * dw
        grad_grid = torch.stack([gu[k] * dudg[k] for k in range(nd)], dim=-1).view(grid.shape)
    if need_input:
        b_off = (torch.arange(B, device=grid.device) * (C * n_vox)).view(B, 1, 1)
        c_off = (torch.arange(C, device=grid.device) * n_vox).view(1, C, 1)
        # One scale for all 2^nd corners (weights are in [0, 1], so max|grad_out| bounds every
        # term), accumulated corner by corner to keep peak memory at one corner's indices.
        vmax = float(go.abs().max()) if go.numel() else 0.0
        if vmax == 0.0 or not math.isfinite(vmax):
            grad_input = torch.full(input_shape, 0.0 if vmax == 0.0 else float('nan'),
                                    dtype=grad_out.dtype, device=grad_out.device)
        else:
            scale = _fixed_point_scale(vmax, go.numel() * 2 ** nd)
            acc = torch.zeros(B * C * n_vox, dtype=torch.int64, device=grad_out.device)
            for c in range(2 ** nd):
                flat, valid, facs = corner(c)
                w = facs[0]
                for f in facs[1:]:
                    w = w * f
                w = torch.where(valid, w, torch.zeros_like(w))
                acc.index_add_(0, (b_off + c_off + flat.unsqueeze(1)).reshape(-1),
                               torch.round((go * w.unsqueeze(1)) * scale).to(torch.int64).reshape(-1))
            grad_input = (acc.to(grad_out.dtype) / scale).view(input_shape)
    return grad_input, grad_grid


from ..spatial import (
    _get_physical_grid_torch_yfirst,
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    _physical_to_normalized_torch_yfirst,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    get_identity_grid_torch,
)



def prepare_mid_images_and_gradients_torch(
    warp_l2r, warp_r2l, I_curr, J_curr,
    X_phys,
    fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t,
    moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t,
    fixed_spacing, moving_spacing,
    M_phys, t_phys, initial_grid_level,
    interpolator='linear',
    grad_I_curr=None, grad_J_curr=None,
    use_analytical_gradients=True
):
    """
    Warp the fixed and moving images to the SyN midpoint and (optionally) sample their
    gradients there.

    ``I_mid = I(X + warp_l2r)`` (fixed image); ``J_mid = J(M_phys (X + warp_r2l) + t_phys)``
    (moving image after the initial affine; when ``initial_grid_level`` is given, the affine
    point is converted to fixed-grid normalised coordinates and looked up in that grid
    instead of going to moving coordinates directly). Sampling uses ``grid_sample_nd`` with
    border padding.

    Parameters
    ----------
    warp_l2r, warp_r2l : Tensor (B, *spatial, dim)
        Physical (mm) displacements on the midpoint grid, components in tensor order (z, y, x).
    I_curr, J_curr : Tensor (B, C, *spatial)
        Fixed / moving images at the current level.
    X_phys : Tensor (1, *spatial, dim)
        Physical grid points (``get_physical_grid_torch``), tensor order.
    fixed_*_t, moving_*_t : Tensor
        Shape, spacing, origin, direction in tensor order (reversed), as expected by
        ``physical_to_normalized_torch_cached``.
    fixed_spacing, moving_spacing : sequence of float
        ANTs (x, y, z) order; used for the image gradients.
    M_phys, t_phys : Tensor (dim, dim), (dim,)
        Initial affine in tensor-order physical coordinates.
    initial_grid_level : Tensor (B, *spatial, dim) or None
        Normalised moving coordinates of each fixed voxel (``compute_initial_grid``, resized).
    interpolator : str, default 'linear'
        Passed to ``grid_sample_nd``.
    grad_I_curr, grad_J_curr : Tensor, optional
        Precomputed image gradients, (B, *spatial, dim) in the ``_spatial_jacobian_nd`` layout
        with the channel axis squeezed (derivative axis (x, y, z), per mm along each grid
        axis); computed here when None.
    use_analytical_gradients : bool, default True
        Also passed to ``grid_sample_nd``. When True the gradients are computed and sampled;
        this needs single-channel images (ValueError otherwise).

    Returns
    -------
    (I_mid, J_mid, grad_I_mid, grad_J_mid, in_bounds_mask)
        I_mid, J_mid : (B, C, *spatial).
        grad_I_mid, grad_J_mid : (B, *spatial, dim) or None (when
        ``use_analytical_gradients`` is False). Physical (mm) image gradients in tensor order
        (z, y, x), matching the warps: the sampled per-axis derivatives g give
        ``g @ inv(direction_t)`` (for J then ``@ M_phys``, the chain rule through the affine).
        in_bounds_mask : (B, 1, *spatial) float, 1 where both sampling points lie in [-1, 1]
        on every axis.
    """
    from .jacobian import _spatial_jacobian_nd
    
    phi_l2r_phys = X_phys + warp_l2r
    coords_norm = physical_to_normalized_torch_cached(
        phi_l2r_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
    )
    I_mid = grid_sample_nd(I_curr, coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)
    
    phi_r2l_phys = X_phys + warp_r2l
    y_phys = phi_r2l_phys @ M_phys.t() + t_phys
    if initial_grid_level is not None:
        y_norm_fixed = physical_to_normalized_torch_cached(
            y_phys, fixed_shape_t, fixed_spacing_t, fixed_origin_t, fixed_direction_t
        )
        y_norm = compose_grids(initial_grid_level, y_norm_fixed)
    else:
        y_norm = physical_to_normalized_torch_cached(
            y_phys, moving_shape_t, moving_spacing_t, moving_origin_t, moving_direction_t
        )
        
    J_mid = grid_sample_nd(J_curr, y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients)
    
    grad_I_mid_sampled = None
    grad_J_mid_sampled = None
    if use_analytical_gradients:
        if I_curr.shape[1] != 1 or J_curr.shape[1] != 1:
            raise ValueError("analytical gradients need single-channel images, got "
                             f"{I_curr.shape[1]} / {J_curr.shape[1]} channels")
        if grad_I_curr is None:
            grad_I_curr = _spatial_jacobian_nd(I_curr.movedim(1, -1), physical_spacing=tuple(reversed(fixed_spacing)))
            if I_curr.shape[1] == 1:
                grad_I_curr = grad_I_curr.squeeze(-2)
        if grad_J_curr is None:
            grad_J_curr = _spatial_jacobian_nd(J_curr.movedim(1, -1), physical_spacing=tuple(reversed(moving_spacing)))
            if J_curr.shape[1] == 1:
                grad_J_curr = grad_J_curr.squeeze(-2)
        
        # per-axis derivatives (x, y, z) -> tensor order; physical gradient = g @ inv(D)
        # (p = origin + D S i, so g = D^T grad_p)
        grad_I_mid_sampled = grid_sample_nd(grad_I_curr.movedim(-1, 1), coords_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1)
        grad_I_mid_sampled = torch.matmul(torch.flip(grad_I_mid_sampled, dims=[-1]),
                                          torch.linalg.inv(fixed_direction_t.to(grad_I_mid_sampled.dtype)))

        grad_J_mid_sampled = grid_sample_nd(grad_J_curr.movedim(-1, 1), y_norm, padding_mode='border', align_corners=True, interpolator=interpolator, use_analytical_gradients=use_analytical_gradients).movedim(1, -1)
        grad_J_mid_sampled = torch.matmul(torch.flip(grad_J_mid_sampled, dims=[-1]),
                                          torch.linalg.inv(moving_direction_t.to(grad_J_mid_sampled.dtype)))
        grad_J_mid_sampled = torch.matmul(grad_J_mid_sampled, M_phys.to(grad_J_mid_sampled.dtype))

    dim = coords_norm.shape[-1]
    mask_I = (coords_norm[..., 0] >= -1.0) & (coords_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_I = mask_I & (coords_norm[..., d] >= -1.0) & (coords_norm[..., d] <= 1.0)
        
    mask_J = (y_norm[..., 0] >= -1.0) & (y_norm[..., 0] <= 1.0)
    for d in range(1, dim):
        mask_J = mask_J & (y_norm[..., d] >= -1.0) & (y_norm[..., d] <= 1.0)
        
    in_bounds_mask = (mask_I & mask_J).unsqueeze(1).to(dtype=I_mid.dtype)
    
    return I_mid, J_mid, grad_I_mid_sampled, grad_J_mid_sampled, in_bounds_mask
