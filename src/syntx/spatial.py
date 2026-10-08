"""
Conversions between ANTs / ITK image space and the tensor layout used inside syntx:
displacement fields, affine matrices, physical and normalised coordinate grids, scalar
images, plus Jacobian / deformation-gradient utilities on NumPy arrays.

Conventions used throughout:

- ANTs / ITK: ``ANTsImage.numpy()`` arrays have spatial axes in (x, y, z) order (shape
  ``img.shape``); displacement components are (dx, dy, dz); spacing and origin are
  (x, y, z); a physical point is ``direction @ (spacing * index) + origin``.
- Tensor (torch / JAX): spatial axes in tensor order (z, y, x); physical-vector components
  in (z, y, x); "reversed" metadata means spacing / origin reversed and the direction matrix
  flipped as ``direction[::-1, ::-1]``.
- Normalised grid coordinates (``F.grid_sample``): [-1, 1] per axis with
  ``align_corners=True`` (-1 and 1 are the edge voxel centres), components in (x, y, z)
  order.

Each function's docstring states which layout it expects; there is no general
auto-detection.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import ants
except ImportError:
    ants = None


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Detection & Type Coercion Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _to_numpy(x):
    """Convert an ANTsImage, torch / JAX array or array-like to a NumPy array (no axis
    changes; an ANTsImage gives its (x, y, z) ``.numpy()``)."""
    if x is None:
        return None
    if ants is not None and isinstance(x, ants.ANTsImage):
        return x.numpy()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):  # jax arrays or similar
        val = x.numpy()
        return val() if callable(val) else np.asarray(val)
    return np.asarray(x)


def _is_tensor(x):
    """Check if x is a PyTorch tensor or JAX array (tensor-domain component order)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return True
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            return True
    except ImportError:
        pass
    return False


def _squeeze_batch(arr):
    """Drop axis 0 when the array has >= 3 axes and ``shape[0] == 1``."""
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _get_spacing(ref_image=None, spacing=None, ndim=None):
    """Spacing tuple: ``spacing`` if given, else ``ref_image.spacing`` (ANTsImage), else
    ``(1.0,) * ndim``, else None."""
    if spacing is not None:
        if hasattr(spacing, "tolist"):
            spacing = spacing.tolist()
        return tuple(spacing)
    if (
        ref_image is not None
        and ants is not None
        and isinstance(ref_image, ants.ANTsImage)
    ):
        return tuple(ref_image.spacing)
    if ndim is not None:
        return (1.0,) * ndim
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Component & Metadata Reversal Primitives
# ═══════════════════════════════════════════════════════════════════════════════


def reverse_components(disp):
    """Reverse vector component order along the last axis.

    Converts between ITK (dx, dy, dz) and tensor (dz, dy, dx) component order; applying it
    twice returns the input. Spatial axes are not touched.

    Parameters
    ----------
    disp : Tensor, jax.Array, ndarray or array-like (..., dim)

    Returns
    -------
    Same type for torch (``torch.flip``, keeps device / dtype / autograd) and JAX; a NumPy
    copy for anything else.
    """
    if torch is not None and torch.is_tensor(disp):
        return torch.flip(disp, dims=[-1])
    try:
        import jax.numpy as jnp

        if isinstance(disp, jnp.ndarray):
            return jnp.flip(disp, axis=-1)
    except ImportError:
        pass
    arr = _to_numpy(disp)
    return arr[..., ::-1].copy()


def reverse_metadata(spacing, origin, direction):
    """Reverse ANTs (x, y, z) metadata to tensor (z, y, x) order (also its own inverse).

    Parameters
    ----------
    spacing, origin : sequence or array-like (dim,)
    direction : array-like (dim, dim) or flat (dim * dim,)
        ITK direction matrix (columns are the physical directions of the index axes).

    Returns
    -------
    (spacing_rev, origin_rev, direction_rev)
        Two tuples and an ndarray ``direction[::-1, ::-1]``.
    """
    if hasattr(spacing, "tolist"):
        spacing = spacing.tolist()
    if hasattr(origin, "tolist"):
        origin = origin.tolist()
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = _to_numpy(direction)
    if dir_arr.ndim == 1:
        dim = len(spacing_rev)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = np.asarray(dir_arr)[::-1, ::-1].copy()
    return spacing_rev, origin_rev, direction_rev


def itk_shape_to_tensor_shape(shape):
    """Reverse a sequence: ANTs shape (Nx, Ny[, Nz]) -> tensor shape ([Nz,] Ny, Nx).

    Works the same for any per-axis tuple (callers also use it for spacing).

    Returns
    -------
    tuple
    """
    if hasattr(shape, "tolist"):
        shape = shape.tolist()
    return tuple(reversed(shape))


def get_image_metadata(img):
    """Geometry of an ANTsImage as a dict.

    Returns
    -------
    dict
        ``{'origin': tuple, 'spacing': tuple, 'direction': ndarray, 'shape': tuple}``, all
        in ANTs (x, y, z) order (the layout ``SyNToTransform`` metadata uses).
    """
    return {
        "origin": tuple(img.origin),
        "spacing": tuple(img.spacing),
        "direction": np.array(img.direction),
        "shape": tuple(img.shape),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Displacement Field Conversions
# ═══════════════════════════════════════════════════════════════════════════════


def export_ants_displacement_field(
    disp, origin=None, spacing=None, direction=None, ref_image=None
):
    """Convert a tensor-layout physical displacement field to an ANTs displacement image.

    Spatial axes go from tensor order (z, y, x) to ANTs (x, y, z) and components from
    (dz, dy, dx) to (dx, dy, dz). Values are not rescaled (they should already be in mm).

    Parameters
    ----------
    disp : ndarray, Tensor or jax.Array (1, ..., 1, *spatial, dim) or (*spatial, dim)
        ``dim`` is taken from the last axis; leading size-1 axes are dropped. A batch larger
        than 1 is not supported (use ``disp_tensor_to_itk``). Only 2-D / 3-D fields are
        transposed.
    origin, spacing, direction : optional
        ANTs (x, y, z) order. ``origin`` may also be an ANTsImage, used as ``ref_image``.
    ref_image : ANTsImage, optional
        Fills whichever of origin / spacing / direction are None.

    Returns
    -------
    ANTsImage with ``has_components=True``.

    Raises
    ------
    ImportError
        If ANTsPy is not installed.
    """
    if ants is None:
        raise ImportError("ANTsPy is required to export ANTs displacement fields.")

    if ref_image is not None and isinstance(ref_image, ants.ANTsImage):
        if origin is None:
            origin = ref_image.origin
        if spacing is None:
            spacing = ref_image.spacing
        if direction is None:
            direction = ref_image.direction
    elif origin is not None and isinstance(origin, ants.ANTsImage):
        ref_image = origin
        origin = ref_image.origin
        spacing = ref_image.spacing
        direction = ref_image.direction

    disp_np = _to_numpy(disp)
    dim = disp_np.shape[-1]

    while disp_np.ndim > (dim + 1) and disp_np.shape[0] == 1:
        disp_np = disp_np[0]

    if dim == 2:
        disp_np = np.transpose(disp_np, (1, 0, 2))
    elif dim == 3:
        disp_np = np.transpose(disp_np, (2, 1, 0, 3))

    # Reverse vector components from PyTorch ZYX order [v_z, v_y, v_x] to ITK XYZ order [v_x, v_y, v_z]
    disp_xyz = np.ascontiguousarray(disp_np[..., ::-1].copy())

    return ants.from_numpy(
        disp_xyz,
        origin=origin,
        spacing=spacing,
        direction=direction,
        has_components=True,
    )


def disp_tensor_to_itk(disp, ref_image):
    """Convert a tensor-domain displacement field to an ANTs displacement image.

    Spatial axes go from tensor order (z, y, x) to ANTs (x, y, z) and components from
    (dz, dy, dx) to (dx, dy, dz) (``export_ants_displacement_field`` per batch item).

    Parameters
    ----------
    disp : torch.Tensor, jax.Array, or np.ndarray
        Displacement field in tensor spatial + component order.
        Shape: (B, *spatial_tensor, dim), (1, *spatial_tensor, dim) or (*spatial_tensor, dim).
    ref_image : ants.ANTsImage or dict
        Reference image providing origin, spacing, direction metadata,
        or metadata dictionary with 'origin', 'spacing', 'direction'.

    Returns
    -------
    ants.ANTsImage or list of ants.ANTsImage
        Multi-component ANTs displacement image(s) with ITK ordering.
        If batch dimension B > 1, returns a list of ANTsImage objects.
        If B = 1 or unbatched, returns a single ANTsImage.

    Examples
    --------
    >>> warp_itk = syntx.spatial.disp_tensor_to_itk(model.warp_l2r, fixed)
    >>> ants.image_write(warp_itk, 'warp.nii.gz')
    """
    arr = _to_numpy(disp)
    if isinstance(ref_image, dict):
        origin = ref_image.get("origin")
        spacing = ref_image.get("spacing")
        direction = ref_image.get("direction")
    else:
        origin = ref_image.origin
        spacing = ref_image.spacing
        direction = ref_image.direction

    dim = arr.shape[-1]
    spatial_ndim = dim

    if arr.ndim == spatial_ndim + 2:
        batch_size = arr.shape[0]
        if batch_size == 1:
            return export_ants_displacement_field(
                arr[0],
                origin=origin,
                spacing=spacing,
                direction=direction,
            )
        else:
            return [
                export_ants_displacement_field(
                    arr[b],
                    origin=origin,
                    spacing=spacing,
                    direction=direction,
                )
                for b in range(batch_size)
            ]
    else:
        return export_ants_displacement_field(
            arr,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )


def _single_disp_itk_to_tensor(disp_img, device="cpu"):
    """ANTs displacement image (or file path) -> torch tensor (1, *spatial, dim) in tensor
    layout (axes and components reversed), in the image's dtype."""
    if isinstance(disp_img, (str, os.PathLike)):
        disp_img = ants.image_read(str(disp_img))
    arr = disp_img.numpy() if hasattr(disp_img, "numpy") else np.asarray(disp_img)
    dim = arr.shape[-1]
    if dim == 2:
        arr = np.transpose(arr, (1, 0, 2))
    elif dim == 3:
        arr = np.transpose(arr, (2, 1, 0, 3))
    arr = arr[..., ::-1]

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
    return tensor


def disp_itk_to_tensor(disp_img, device="cpu"):
    """Convert ANTs displacement image(s) to a tensor-domain displacement field.

    Parameters
    ----------
    disp_img : ants.ANTsImage, str, PathLike, or sequence (list/tuple)
        ANTs displacement image, path to NIfTI displacement field file, or sequence
        (list/tuple) of ANTsImage objects or file paths.
    device : str or torch.device
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Displacement field tensor of shape (B, *spatial_tensor, dim) with tensor
        component order and matching spatial layout. B=1 for single input, B=N for sequence
        (the fields must then share one shape). Values are not rescaled.

    Raises
    ------
    ValueError
        For an empty list / tuple.
    """
    if isinstance(disp_img, (list, tuple)):
        if len(disp_img) == 0:
            raise ValueError("Empty list/tuple provided to disp_itk_to_tensor.")
        tensors = [_single_disp_itk_to_tensor(item, device=device) for item in disp_img]
        return torch.cat(tensors, dim=0)
    return _single_disp_itk_to_tensor(disp_img, device=device)


def normalized_to_physical_disp(
    disp,
    shape,
    spacing,
    direction,
    origin=None,
    device=None,
    dtype=torch.float32 if torch is not None else None,
):
    """Convert a normalised-coordinate displacement to a tensor-layout physical (mm) one.

    Components are reversed to (z, y, x), scaled by ``(n - 1) / 2 * spacing`` per axis and
    rotated by the reversed direction matrix.

    Parameters
    ----------
    disp : torch.Tensor or np.ndarray
        Normalized grid displacement of shape `(1, *shape, dim)` or `(*shape, dim)`
        with vector channels in normalized (x, y[, z]) order.
    shape : tuple of int
        Tensor spatial shape (e.g. (Nz, Ny, Nx) or (Ny, Nx)).
    spacing : tuple of float
        Physical voxel spacing in ITK (sx, sy[, sz]) order.
    direction : np.ndarray or array-like
        Direction matrix in ITK order.
    origin : tuple or list, optional
        Ignored (a displacement does not depend on the origin).
    device : str or torch.device, optional
        None: the input's device.
    dtype : torch.dtype, default torch.float32
        None: the input dtype (float32 for integer input).

    Returns
    -------
    torch.Tensor
        Same shape as ``disp``; components in (dz, dy, dx) order, mm.
    """
    if torch is None:
        raise ImportError("PyTorch is required for normalized_to_physical_disp.")
    if not isinstance(disp, torch.Tensor):
        disp = torch.from_numpy(np.asarray(disp))
    if device is None:
        device = disp.device
    if dtype is None:
        dtype = disp.dtype if disp.is_floating_point() else torch.float32
    disp = disp.to(device=device, dtype=dtype)
    dim = len(shape)

    # Convert component channels from (x, y, z) grid order to (z, y, x) tensor order
    disp_tensor = torch.flip(disp, dims=[-1])

    shape_t = torch.tensor(list(shape), device=device, dtype=dtype)
    dummy_origin = (0.0,) * dim if origin is None else origin
    spacing_rev, _, direction_rev = reverse_metadata(spacing, dummy_origin, direction)
    spacing_t = torch.tensor(list(spacing_rev), device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)

    scale_t = (shape_t - 1.0) / 2.0 * spacing_t
    scaled = disp_tensor * scale_t

    orig_shape = disp_tensor.shape
    flat_scaled = scaled.reshape(-1, dim)
    flat_phys = flat_scaled @ direction_t.t()
    return flat_phys.reshape(orig_shape)


# ═══════════════════════════════════════════════════════════════════════════════
# Affine Parameter Conversions & Exports
# ═══════════════════════════════════════════════════════════════════════════════


def create_ants_affine(M_phys, t_phys=None, dim: int = None, fixed_params=None):
    """Build a float ANTs 'AffineTransform' from a physical matrix and translation.

    The transform maps ``x -> M_phys @ (x - c) + t_phys + c`` with ``c = fixed_params``
    (ITK semantics), in ANTs physical (x, y, z) coordinates. Values are cast to float32.

    Parameters
    ----------
    M_phys : ndarray or Tensor
        (dim, dim) linear part; or, when ``t_phys`` is None, an augmented (dim, dim+1) matrix
        or a homogeneous 3 x 3 / 4 x 4 matrix (so a 3 x 3 with ``t_phys=None`` is read as a
        2-D homogeneous matrix).
    t_phys : ndarray or Tensor (dim,), optional
    dim : int, optional
        Defaults to the number of rows of the linear part.
    fixed_params : ndarray or Tensor (dim,), optional
        Centre of rotation; zeros when None.

    Returns
    -------
    ants.ANTsTransform

    Raises
    ------
    ValueError
        ``t_phys`` is None and ``M_phys`` is neither augmented nor homogeneous.
    ImportError
        If ANTsPy is not installed.
    """
    if ants is None:
        raise ImportError("ANTsPy is required for create_ants_affine.")
    M_phys = np.asarray(_to_numpy(M_phys), dtype=np.float32)

    if t_phys is None:
        if M_phys.ndim == 2 and M_phys.shape[1] == M_phys.shape[0] + 1:
            t_phys = M_phys[:, -1]
            M_phys = M_phys[:, :-1]
        elif (
            M_phys.ndim == 2
            and M_phys.shape[0] == M_phys.shape[1]
            and M_phys.shape[0] in (3, 4)
        ):
            # Homogeneous matrix (dim+1, dim+1)
            t_phys = M_phys[:-1, -1]
            M_phys = M_phys[:-1, :-1]
        else:
            raise ValueError(
                "t_phys must be provided if M_phys is not an augmented/homogeneous matrix."
            )
    else:
        t_phys = np.asarray(_to_numpy(t_phys), dtype=np.float32).ravel()

    if dim is None:
        dim = M_phys.shape[0]

    tx = ants.new_ants_transform(
        precision="float", dimension=dim, transform_type="AffineTransform"
    )
    tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    if fixed_params is None:
        tx.set_fixed_parameters(np.zeros(dim))
    else:
        tx.set_fixed_parameters(
            np.asarray(_to_numpy(fixed_params), dtype=np.float32).ravel()
        )
    return tx


def export_ants_affine_transform(M_phys, t_phys, dim: int = None, filename: str = None):
    """Forward and inverse ANTs affine transforms (zero centre) from ``y = M_phys @ x + t_phys``.

    The inverse is ``(M^-1, -M^-1 t)`` computed in float32. Both are built with
    ``create_ants_affine``; parameters are ``M.ravel()`` (row-major) followed by ``t``.

    Parameters
    ----------
    M_phys : ndarray or Tensor (dim, dim)
        ANTs physical (x, y, z) coordinates.
    t_phys : ndarray or Tensor (dim,)
    dim : int, optional
        Defaults to ``M_phys.shape[0]``.
    filename : str, optional
        If given, the forward transform only is written there with ``ants.write_transform``.

    Returns
    -------
    tx_fwd : ants.ANTsTransform
        Forward ANTs transform object.
    tx_inv : ants.ANTsTransform
        Inverse ANTs transform object.
    """
    if ants is None:
        raise ImportError("ANTsPy is required for export_ants_affine_transform.")
    M_phys = np.asarray(_to_numpy(M_phys), dtype=np.float32)
    t_phys = np.asarray(_to_numpy(t_phys), dtype=np.float32).ravel()

    if dim is None:
        dim = M_phys.shape[0]

    tx_fwd = create_ants_affine(M_phys, t_phys, dim=dim)

    M_phys_inv = np.linalg.inv(M_phys)
    t_phys_inv = -M_phys_inv @ t_phys
    tx_inv = create_ants_affine(M_phys_inv, t_phys_inv, dim=dim)

    if filename is not None:
        ants.write_transform(tx_fwd, filename)

    return tx_fwd, tx_inv


def _grid_to_physical_affine_torch_yfirst(
    T_grid,
    fixed_shape,
    fixed_spacing,
    fixed_origin,
    fixed_direction,
    moving_shape,
    moving_spacing,
    moving_origin,
    moving_direction,
):
    """Normalised-grid affine -> physical (M, t), everything in tensor order (z, y, x).

    ``T_grid`` (dim+1 or dim rows, (z, y, x) order) maps fixed normalised coordinates to
    moving normalised ones; the result maps fixed physical points to moving physical points,
    ``M = V_y A W_x``, ``t = V_y (A b_x + t_grid) + c_y``. Computed in float32 (returned in
    ``T_grid``'s dtype); the fixed direction is inverted exactly (``torch.inverse``)."""
    dim = len(fixed_shape)
    device = T_grid.device
    orig_dtype = T_grid.dtype
    calc_dtype = torch.float32

    Nx = torch.tensor(fixed_shape, device=device, dtype=calc_dtype)
    Ny = torch.tensor(moving_shape, device=device, dtype=calc_dtype)
    Sx = torch.tensor(fixed_spacing, device=device, dtype=calc_dtype)
    Sy = torch.tensor(moving_spacing, device=device, dtype=calc_dtype)
    Ox = torch.tensor(fixed_origin, device=device, dtype=calc_dtype)
    Oy = torch.tensor(moving_origin, device=device, dtype=calc_dtype)
    Dx = torch.tensor(fixed_direction, device=device, dtype=calc_dtype)
    Dy = torch.tensor(moving_direction, device=device, dtype=calc_dtype)

    Kx = torch.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = torch.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Kx_inv = torch.inverse(Kx)
    Sx_inv = torch.inverse(torch.diag(Sx))
    Dx_inv = torch.inverse(Dx)              # ITK's inverse direction (not the transpose)
    Wx = Kx_inv @ Sx_inv @ Dx_inv
    bx = -Kx_inv @ Sx_inv @ Dx_inv @ Ox - Kx_inv @ Cx

    Vy = Dy @ torch.diag(Sy) @ Ky
    cy = Dy @ torch.diag(Sy) @ Cy + Oy

    A_grid = T_grid[:dim, :dim].to(calc_dtype)
    t_grid = T_grid[:dim, dim].to(calc_dtype)

    M_phys = (Vy @ A_grid @ Wx).to(orig_dtype)
    t_phys = (Vy @ (A_grid @ bx + t_grid) + cy).to(orig_dtype)
    return M_phys, t_phys


def grid_to_physical_affine_torch(
    T_grid,
    fixed_shape,
    fixed_spacing,
    fixed_origin,
    fixed_direction,
    moving_shape,
    moving_spacing,
    moving_origin,
    moving_direction,
):
    """Normalised-grid affine -> physical (M_phys, t_phys) in tensor order, in torch.

    Parameters
    ----------
    T_grid : Tensor (dim+1, dim+1), (1, dim+1, dim+1) or (dim, dim+1)
        Maps fixed normalised coordinates to moving normalised coordinates, (x, y, z) order
        (``F.affine_grid`` / ``HierarchicalAffine`` convention).
    fixed_shape, moving_shape : sequence of int
        Tensor order (z, y, x).
    fixed_spacing, fixed_origin, moving_spacing, moving_origin : sequence of float
        ANTs (x, y, z) order (reversed internally).
    fixed_direction, moving_direction : array-like (dim, dim)
        ANTs direction matrices (NumPy-convertible; flipped internally).

    Returns
    -------
    (M_phys, t_phys)
        Tensors (dim, dim) and (dim,) in tensor (z, y, x) physical order, mapping fixed
        physical points to moving physical points; differentiable with respect to
        ``T_grid``.
    """
    if T_grid.ndim == 3 and T_grid.shape[0] == 1:
        T_grid = T_grid[0]
    dim = len(fixed_shape)
    # T_grid operates in grid_sample's XY order; permute to YX for _yfirst
    perm = list(range(dim - 1, -1, -1))  # [1,0] for 2D, [2,1,0] for 3D
    T_yx = T_grid.clone()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]
    fs_rev = tuple(reversed(fixed_spacing))
    fo_rev = tuple(reversed(fixed_origin))
    fd_rev = np.asarray(fixed_direction)[::-1, ::-1].copy()
    ms_rev = tuple(reversed(moving_spacing))
    mo_rev = tuple(reversed(moving_origin))
    md_rev = np.asarray(moving_direction)[::-1, ::-1].copy()
    M_phys_zyx, t_phys_zyx = _grid_to_physical_affine_torch_yfirst(
        T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev
    )

    # Return ZYX physical affine matrices directly to match PyTorch tensor coordinate ordering (Z, Y, X)
    return M_phys_zyx, t_phys_zyx


def grid_to_physical_affine(T_grid, fixed, moving):
    """Normalised-grid affine -> physical (M_phys, t_phys) in ANTs (x, y, z) order (NumPy).

    Same algebra as ``grid_to_physical_affine_torch``, with the geometry read from the
    images; inverse of ``physical_to_grid_affine``.

    Parameters
    ----------
    T_grid : Tensor or array-like (dim+1, dim+1) or (dim, dim+1)
        Fixed normalised -> moving normalised coordinates, (x, y, z) order.
    fixed, moving : ANTsImage

    Returns
    -------
    (M_phys, t_phys)
        ndarrays (dim, dim) and (dim,) mapping fixed physical points to moving physical
        points, ANTs (x, y, z) order (the layout ``create_ants_affine`` takes).
    """
    if hasattr(T_grid, "detach"):
        T_grid = T_grid.detach().cpu().numpy()
    T_grid = np.asarray(T_grid, dtype=np.float32)

    dim = len(fixed.shape)
    Nx = np.array(list(reversed(fixed.shape)), dtype=np.float32)
    Ny = np.array(list(reversed(moving.shape)), dtype=np.float32)

    # Reverse spacing, origin, and direction to match PyTorch/JAX (z, y, x) order
    Sx = np.array(fixed.spacing)[::-1]
    Sy = np.array(moving.spacing)[::-1]
    Ox = np.array(fixed.origin)[::-1]
    Oy = np.array(moving.origin)[::-1]
    Dx = np.array(fixed.direction)[::-1, ::-1]
    Dy = np.array(moving.direction)[::-1, ::-1]

    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0

    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Kx_inv = np.linalg.inv(Kx)
    Sx_inv = np.linalg.inv(np.diag(Sx))
    Dx_inv = np.linalg.inv(Dx)              # ITK's inverse direction (not the transpose)
    Wx = Kx_inv @ Sx_inv @ Dx_inv
    bx = -Kx_inv @ Sx_inv @ Dx_inv @ Ox - Kx_inv @ Cx

    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy

    perm = list(range(dim - 1, -1, -1))
    T_yx = T_grid.copy()
    T_yx[:dim, :dim] = T_grid[:dim, :dim][perm][:, perm]
    T_yx[:dim, dim] = T_grid[:dim, dim][perm]

    A_grid = T_yx[:dim, :dim]
    t_grid = T_yx[:dim, dim]

    # Compute in (z, y, x) space
    M_phys = Vy @ A_grid @ Wx
    t_phys = Vy @ (A_grid @ bx + t_grid) + cy

    # Permute from (z, y, x) to (x, y, z) for ITK physical space
    P = np.eye(dim)[::-1]
    M_phys_xyz = P @ M_phys @ P
    t_phys_xyz = P @ t_phys

    return M_phys_xyz, t_phys_xyz


def physical_to_grid_affine(M_phys, t_phys, fixed_img, moving_img):
    """Physical affine ``y = M_phys @ x + t_phys`` (ANTs (x, y, z) order) -> normalised-grid
    affine.

    Inverse of ``grid_to_physical_affine``; the fixed direction is inverted exactly.

    Parameters
    ----------
    M_phys, t_phys : ndarray or Tensor (dim, dim), (dim,)
        Fixed physical points -> moving physical points.
    fixed_img, moving_img : ANTsImage

    Returns
    -------
    ndarray float32 (dim+1, dim+1)
        Homogeneous matrix mapping fixed normalised coordinates to moving normalised
        coordinates ((x, y, z) order, ``align_corners=True``).
    """
    if hasattr(M_phys, "detach"):
        M_phys = M_phys.detach().cpu().numpy()
    if hasattr(t_phys, "detach"):
        t_phys = t_phys.detach().cpu().numpy()
    M_phys = np.asarray(M_phys, dtype=np.float32)
    t_phys = np.asarray(t_phys, dtype=np.float32).ravel()

    dim = fixed_img.dimension
    Nx = np.array(fixed_img.shape)
    Ny = np.array(moving_img.shape)
    Sx = np.array(fixed_img.spacing)
    Sy = np.array(moving_img.spacing)
    Ox = np.array(fixed_img.origin)
    Oy = np.array(moving_img.origin)
    Dx = np.array(fixed_img.direction)
    Dy = np.array(moving_img.direction)

    Kx = np.diag((Nx - 1) / 2.0)
    Cx = (Nx - 1) / 2.0
    Ky = np.diag((Ny - 1) / 2.0)
    Cy = (Ny - 1) / 2.0

    Wx_inv = Dx @ np.diag(Sx) @ Kx
    bx = (
        -np.linalg.inv(Kx) @ np.linalg.inv(np.diag(Sx)) @ np.linalg.inv(Dx) @ Ox
        - np.linalg.inv(Kx) @ Cx
    )

    Vy = Dy @ np.diag(Sy) @ Ky
    cy = Dy @ np.diag(Sy) @ Cy + Oy
    Vy_inv = np.linalg.inv(Vy)

    A_grid = Vy_inv @ M_phys @ Wx_inv
    t_grid = Vy_inv @ (t_phys - cy) - A_grid @ bx

    T_grid = np.eye(dim + 1, dtype=np.float32)
    T_grid[:dim, :dim] = A_grid
    T_grid[:dim, dim] = t_grid

    return T_grid


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate Grid Primitives
# ═══════════════════════════════════════════════════════════════════════════════


def _get_physical_grid_torch_yfirst(
    shape, spacing, origin, direction, device="cpu", dtype=torch.float32
):
    """Physical point of every voxel, ``(index * spacing) @ direction.T + origin``, with all
    inputs and the output components in the same (tensor) order; shape (1, *shape, dim)."""
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing="ij")
    idxs = torch.stack(meshgrid, dim=-1)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)

    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)


def get_physical_grid_torch(
    shape, spacing, origin, direction, device="cpu", dtype=torch.float32
):
    """Physical coordinates of every voxel of a grid, in tensor layout.

    Parameters
    ----------
    shape : sequence of int
        Tensor order (z, y, x).
    spacing, origin : sequence of float
        ANTs (x, y, z) order.
    direction : array-like (dim, dim) or flat (dim * dim,)
        ANTs direction matrix.
    device : default 'cpu'
    dtype : default torch.float32

    Returns
    -------
    Tensor (1, *shape, dim)
        Physical points (mm), components in tensor order (z, y, x).
    """
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _get_physical_grid_torch_yfirst(
        shape, spacing_rev, origin_rev, direction_rev, device, dtype
    )


def _physical_to_normalized_torch_yfirst(
    phys_coords, target_shape, spacing, origin, direction
):
    """Physical points -> normalised [-1, 1] coordinates with all inputs in tensor order; the
    output components are flipped to (x, y, z) for ``grid_sample``. Uses a general matrix
    inverse of the direction; axes of size 1 divide by zero."""
    device = phys_coords.device
    dtype = phys_coords.dtype
    dim = len(target_shape)

    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)

    flat_phys = phys_coords.view(-1, dim)
    diff = flat_phys - origin_t
    inv_direction_t = torch.inverse(direction_t.t())
    rotated = diff @ inv_direction_t
    voxel_coords = rotated / spacing_t

    shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
    norm_coords = (voxel_coords / (shape_t - 1)) * 2.0 - 1.0
    # Flip from internal YX order to grid_sample's expected XY order
    norm_coords = torch.flip(norm_coords, dims=[-1])
    return norm_coords.view(phys_coords.shape)


def physical_to_normalized_torch(phys_coords, target_shape, spacing, origin, direction):
    """Physical points -> normalised ``grid_sample`` coordinates of an image grid.

    Parameters
    ----------
    phys_coords : Tensor (..., dim)
        Physical points, components in tensor order (z, y, x). Must be viewable as
        (-1, dim).
    target_shape : sequence of int
        Image shape in tensor order (z, y, x).
    spacing, origin : sequence of float
        ANTs (x, y, z) order.
    direction : array-like (dim, dim) or flat
        ANTs direction matrix.

    Returns
    -------
    Tensor of the input shape
        Normalised coordinates (``align_corners=True``), components in (x, y, z) order.
        Points outside the image fall outside [-1, 1].
    """
    # target_shape is in tensor order (Z, Y, X). _yfirst expects all params in Z-first order.
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(target_shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(
        phys_coords, target_shape, spacing_rev, origin_rev, direction_rev
    )


_ANATOMICAL_AXIS_LABELS = {
    "lr": 0,
    "rl": 0,
    "l": 0,
    "r": 0,
    "x": 0,
    "ap": 1,
    "pa": 1,
    "a": 1,
    "p": 1,
    "y": 1,
    "si": 2,
    "is": 2,
    "s": 2,
    "i_": 2,  # 'i_' avoids colliding with BIDS 'i' voxel-axis key
    "z": 2,
}
_BIDS_VOXEL_AXIS_INDEX = {"i": 0, "j": 1, "k": 2}


def restriction_from_orientation(
    image,
    *,
    anatomical_axis=None,
    bids_phase_encoding_direction=None,
    json_sidecar=None,
    obliquity_warn_threshold=0.98,
) -> tuple:
    """One-hot ``restrict_transformation`` weights for a physical axis named by anatomy or by
    a BIDS phase-encoding direction.

    Restriction weights act on physical axes, so an oblique acquisition (phase-encode axis
    not aligned with a physical axis) can only be approximated; the BIDS path warns when the
    approximation is poor.

    Give exactly one of ``anatomical_axis``, ``bids_phase_encoding_direction`` or
    ``json_sidecar`` (``PhaseEncodingDirection``); several raise ValueError.

    Parameters
    ----------
    image : ANTsImage
        Its direction matrix gives ``dim`` (always) and, for the BIDS path, the voxel-axis to
        physical-axis mapping.
    anatomical_axis : str, optional
        Physical (ITK LPS) axis, case-insensitive: 'LR', 'RL', 'L', 'R', 'x' -> 0; 'AP', 'PA',
        'A', 'P', 'y' -> 1; 'SI', 'IS', 'S', 'I' (uppercase; lowercase 'i' is the BIDS voxel
        axis), 'i_', 'z' -> 2. No direction lookup is done.
    bids_phase_encoding_direction : str, optional
        'i', 'j' or 'k', optionally with a trailing '-' (sign is ignored). The voxel axis is
        mapped to the physical axis with the largest absolute component in that axis's
        column of ``image.direction``.
    json_sidecar : str or Path, optional
        BIDS JSON sidecar read for ``PhaseEncodingDirection``. A missing file or key raises
        ValueError.
    obliquity_warn_threshold : float, default 0.98
        BIDS path only: a ``UserWarning`` is issued when that largest absolute component is
        below this (0.98 is about 11.5 degrees).

    Returns
    -------
    tuple of float
        Length ``dim``: 1.0 at the resolved physical axis, 0.0 elsewhere, for
        ``restrict_transformation=``.

    Raises
    ------
    ValueError
        Unknown label, no or several inputs, or an axis beyond ``dim`` (e.g. 'z' for a 2-D
        image).

    Examples
    --------
    >>> w = restriction_from_orientation(t1_image, anatomical_axis="AP")
    >>> w = restriction_from_orientation(dwi_b0_image, json_sidecar="sub-01_dwi.json")
    """
    import warnings

    dim = int(np.asarray(image.direction).shape[0])

    n_given = sum(x is not None for x in (anatomical_axis, bids_phase_encoding_direction, json_sidecar))
    if n_given > 1:
        raise ValueError("give exactly one of anatomical_axis, bids_phase_encoding_direction, "
                         "json_sidecar")
    if anatomical_axis is not None:
        key = str(anatomical_axis).strip()
        key = "i_" if key == "I" else key.lower()
        if key not in _ANATOMICAL_AXIS_LABELS:
            raise ValueError(
                f"Unknown anatomical_axis {anatomical_axis!r}; expected one of "
                f"LR/RL, AP/PA, SI/IS (or L/R/A/P/S/I), or x/y/z."
            )
        physical_axis = _ANATOMICAL_AXIS_LABELS[key]
        if physical_axis >= dim:
            raise ValueError(f"anatomical_axis {anatomical_axis!r} does not exist in a {dim}-D image")
        weights = [0.0] * dim
        weights[physical_axis] = 1.0
        return tuple(weights)

    pe_dir = bids_phase_encoding_direction
    if pe_dir is None and json_sidecar is not None:
        import json
        from pathlib import Path

        sidecar_path = Path(json_sidecar)
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            pe_dir = metadata.get("PhaseEncodingDirection")
        if pe_dir is None:
            raise ValueError(
                f"No PhaseEncodingDirection found in sidecar {json_sidecar!r}, and no "
                f"anatomical_axis or bids_phase_encoding_direction given as a fallback."
            )
    if pe_dir is None:
        raise ValueError(
            "Provide exactly one of anatomical_axis, bids_phase_encoding_direction, or "
            "json_sidecar (with a readable PhaseEncodingDirection)."
        )

    voxel_axis_key = str(pe_dir).strip().lower().rstrip("-")
    if voxel_axis_key not in _BIDS_VOXEL_AXIS_INDEX:
        raise ValueError(
            f"Unrecognized BIDS PhaseEncodingDirection {pe_dir!r}; expected 'i', 'j', 'k' "
            f"(optionally with a trailing '-')."
        )
    voxel_axis = _BIDS_VOXEL_AXIS_INDEX[voxel_axis_key]
    if voxel_axis >= dim:
        raise ValueError(
            f"PhaseEncodingDirection {pe_dir!r} implies voxel axis {voxel_axis}, "
            f"but image direction matrix has only {dim} dimensions."
        )

    direction = np.asarray(image.direction)
    column = direction[:, voxel_axis]
    physical_axis = int(np.argmax(np.abs(column)))
    alignment = float(np.abs(column[physical_axis]))
    if alignment < obliquity_warn_threshold:
        warnings.warn(
            f"restriction_from_orientation: voxel axis '{pe_dir}' (index {voxel_axis}) is "
            f"obliquely acquired relative to physical axis {physical_axis} (alignment "
            f"{alignment:.4f} < {obliquity_warn_threshold}). Axis-aligned "
            f"restrict_transformation is an approximation for oblique acquisitions -- "
            f"the true phase-encode direction has real components on more than one "
            f"physical axis, which this weight vector cannot represent exactly.",
            UserWarning,
            stacklevel=2,
        )

    weights = [0.0] * dim
    weights[physical_axis] = 1.0
    return tuple(weights)


def get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t):
    """Matrix and bias mapping physical points to normalised grid coordinates.

    ``n = D^-1 (x - origin) * 2 / (spacing * (shape - 1)) - 1`` written as ``x @ M + b``
    (the true inverse direction, as ITK uses).

    Parameters
    ----------
    shape_t, spacing_t, origin_t : Tensor (dim,)
        Tensor order (z, y, x).
    direction_t : Tensor (dim, dim)
        Reversed direction, ``direction[::-1, ::-1]``.

    Returns
    -------
    (M, b)
        Tensors (dim, dim) and (dim,). For row vectors ``x`` of physical points in tensor
        order (z, y, x), ``x @ M + b`` gives normalised coordinates in (x, y, z) order.
    """
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    # x_row @ inv(D)^T == (inv(D) x)^T; equals x_row @ D for an orthonormal D
    M = torch.linalg.inv(direction_t).transpose(-1, -2) * scale_t.unsqueeze(0)
    b = -(origin_t @ M) - 1.0
    M_norm = torch.flip(M, dims=[-1])
    b_norm = torch.flip(b, dims=[-1])
    return M_norm, b_norm


def get_physical_to_normalized_affine_xyz(
    shape, spacing, origin, direction, device="cpu", dtype=torch.float32
):
    """``get_physical_to_normalized_affine`` for ANTs (x, y, z)-ordered inputs and points.

    Computed in float64, then cast. Works for 2-D as well as 3-D.

    Parameters
    ----------
    shape, spacing, origin : sequence, length dim
        ANTs order (Nx, Ny[, Nz]), (sx, sy[, sz]), (ox, oy[, oz]).
    direction : array-like (dim, dim)
        As ``ANTsImage.direction`` reports it (not reversed).
    device : str or torch.device
        Target PyTorch device for the returned tensors.
    dtype : torch.dtype
        Target dtype for the returned tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        M (dim, dim) and b (dim,) such that, for physical points ``x_phys`` in (x, y, z)
        order, ``x_phys @ M + b`` gives ``grid_sample`` coordinates in (x, y, z) order.
    """
    shape_t = torch.as_tensor(np.asarray(shape)[::-1].copy(), dtype=torch.float64)
    spacing_t = torch.as_tensor(np.asarray(spacing)[::-1].copy(), dtype=torch.float64)
    origin_t = torch.as_tensor(np.asarray(origin)[::-1].copy(), dtype=torch.float64)
    direction_t = torch.as_tensor(
        np.asarray(direction)[::-1, ::-1].copy(), dtype=torch.float64
    )
    M, b = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
    M = torch.flip(M, dims=[0])
    return M.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def lps_to_ras(coords):
    """Convert LPS millimetre coordinates (ANTs / ITK) to RAS (NIfTI / nibabel, .tck / .trk)
    by negating x and y.

    Accepts any array-like (..., dim >= 2); returns a NumPy copy (floating input keeps its
    dtype, other input becomes float64). Apply at file I/O boundaries.
    """
    out = np.array(coords, copy=True)
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    out[..., 0] = -out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


def ras_to_lps(coords):
    """Convert coordinates from RAS millimeters to LPS millimeters (flips x and y).

    Same operation as :func:`lps_to_ras` (it is its own inverse); returns a copy, dtype as
    there.
    """
    out = np.array(coords, copy=True)
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    out[..., 0] = -out[..., 0]
    out[..., 1] = -out[..., 1]
    return out


def physical_to_normalized_fast(phys_coords, M, b):
    """``phys_coords @ M + b`` over the last axis, for (M, b) from
    ``get_physical_to_normalized_affine``; returns the input shape (input must be viewable
    as (-1, dim))."""
    dim = phys_coords.shape[-1]
    flat_phys = phys_coords.view(-1, dim)
    flat_norm = flat_phys @ M + b
    return flat_norm.view(phys_coords.shape)


def physical_to_normalized_torch_cached(
    phys_coords, shape_t, spacing_t, origin_t, direction_t
):
    """Physical points (tensor order) -> normalised (x, y, z) coordinates from reversed
    metadata tensors (see ``get_physical_to_normalized_affine``).

    Nothing is cached here: (M, b) are rebuilt on every call; "cached" refers to the caller
    keeping the metadata as tensors on the device.
    """
    M, b = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
    return physical_to_normalized_fast(phys_coords, M, b)


def get_identity_grid_torch(target_shape, device="cpu", dtype=torch.float32):
    """Generate normalized identity coordinate grid [-1, 1] for torch.nn.functional.grid_sample.

    Coordinates along the component axis are in (x, y) or (x, y, z) order.

    Parameters
    ----------
    target_shape : tuple or list of int
        Image spatial shape in Tensor order (Z, Y, X) or (Y, X).
    device : str or torch.device, default 'cpu'
        Target compute device.
    dtype : torch.dtype, default torch.float32
        Target data type.

    Returns
    -------
    torch.Tensor
        Identity coordinate grid of shape (1, *target_shape, dim).
    """
    grids = [
        torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in target_shape
    ]
    meshgrid = torch.meshgrid(*grids, indexing="ij")
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
    return identity


def compute_grid_to_physical_reference_matrix(
    shape, spacing, origin, direction, device=None, dtype=None
) -> torch.Tensor:
    """Homogeneous matrix H mapping normalised grid coordinates to physical points.

    ``x_phys = D diag(spacing) ((shape - 1) / 2 * (n + 1)) + origin``, i.e.
    ``H[:dim, :dim] = D diag(spacing) diag((shape - 1) / 2)`` and ``H[:dim, dim]`` the
    physical centre of the image. Inputs and both coordinate vectors are in ANTs (x, y, z)
    order (normalised (x, y, z) is the ``grid_sample`` component order).

    Parameters
    ----------
    shape : tuple of int
        ANTs order (``ANTsImage.shape``).
    spacing : tuple of float
        Voxel spacing in XYZ order.
    origin : tuple of float
        Image origin in XYZ order.
    direction : np.ndarray or list of list
        Direction matrix in XYZ order.
    device : str or torch.device, optional
        Default 'cpu'.
    dtype : torch.dtype, optional
        Default torch.float32.

    Returns
    -------
    torch.Tensor (dim+1, dim+1)
    """
    dim = len(shape)
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float32

    N_t = torch.tensor(list(shape), device=device, dtype=dtype)
    S_t = torch.tensor(list(spacing), device=device, dtype=dtype)
    O_t = torch.tensor(list(origin), device=device, dtype=dtype)
    D_t = torch.tensor(np.asarray(direction), device=device, dtype=dtype)

    com_fov = D_t @ (S_t * (N_t - 1) / 2.0) + O_t

    H = torch.eye(dim + 1, device=device, dtype=dtype)
    H[:dim, :dim] = D_t @ torch.diag(S_t) @ torch.diag((N_t - 1) / 2.0)
    H[:dim, dim] = com_fov
    return H


def compute_autograd_physical_scale(shape, spacing, device=None, dtype=None):
    """Millimetres per normalised unit along each axis, ``flip((shape - 1) * spacing / 2)``.

    The flip turns tensor order into the (x, y, z) component order of normalised grids, so
    the result scales grid-ordered quantities between normalised and physical units.

    Parameters
    ----------
    shape : tuple, list, or torch.Tensor
        Image spatial shape in Tensor order (Z, Y, X) or (Y, X).
    spacing : tuple, list, or torch.Tensor
        Voxel spacing aligned with shape (tensor order).
    device : str or torch.device, optional
        Target device for returned tensor.
    dtype : torch.dtype, optional
        Target dtype (defaults to torch.float32).

    Returns
    -------
    torch.Tensor
        Shape (dim,), (x, y, z) order.
    """
    if dtype is None:
        dtype = torch.float32

    if not isinstance(shape, torch.Tensor):
        shape_t = torch.tensor(list(shape), device=device, dtype=dtype)
    else:
        shape_t = shape.to(
            device=device if device is not None else shape.device, dtype=dtype
        )

    if not isinstance(spacing, torch.Tensor):
        spacing_t = torch.tensor(list(spacing), device=device, dtype=dtype)
    else:
        spacing_t = spacing.to(
            device=device if device is not None else spacing.device, dtype=dtype
        )

    scale = (shape_t - 1.0) * spacing_t / 2.0
    return torch.flip(scale, dims=[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Image Conversions
# ═══════════════════════════════════════════════════════════════════════════════


def _format_tensor_batch_channel(t):
    """Add batch / channel axes by rank: 0-3 axes -> (1, 1, *spatial) (a 3-axis tensor is a
    3-D volume); 4 axes -> unchanged if ``shape[1] == 1`` else (B, 1, D, H, W); 5+ unchanged."""
    while t.ndim < 2:
        t = t.unsqueeze(0)
    if t.ndim == 2:
        # Unbatched 2D image: (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        # Unbatched 3D image: (D, H, W) -> (1, 1, D, H, W)
        return t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 4:
        # If shape is (B, 1, H, W), channel dimension is already present
        if t.shape[1] == 1:
            return t
        # If shape is (B, D, H, W) without channel, insert channel dimension -> (B, 1, D, H, W)
        return t.unsqueeze(1)
    elif t.ndim >= 5:
        # Already batched with channel: (B, C, D, H, W)
        return t
    return t


def image_to_tensor(img, device="cpu", dtype=None, to_zyx=False):
    """Convert an ANTsImage, tensor or array to a float tensor with batch / channel axes.

    Parameters
    ----------
    img : ANTsImage, Tensor or array-like
        Scalar image. Arrays are taken to be in the same axis order as ``ANTsImage.numpy()``.
    device : str or torch.device, default 'cpu'
    dtype : torch.dtype, optional
        Default float32.
    to_zyx : bool, default False
        True: reverse the spatial axes, ANTs (x, y, z) -> tensor order (z, y, x) (for tensors
        also with a (B, 1, ...) prefix). False keeps the ANTs axis order, which is NOT the
        tensor order the rest of syntx uses.

    Returns
    -------
    Tensor (1, 1, *spatial), or (B, 1, *spatial) for batched tensor input (see
    ``_format_tensor_batch_channel``).
    """
    if dtype is None:
        dtype = torch.float32
    if ants is not None and isinstance(img, ants.ANTsImage):
        arr = img.numpy().astype(np.float32)
        if to_zyx:
            if arr.ndim == 2:
                arr = arr.T
            elif arr.ndim == 3:
                arr = np.transpose(arr, (2, 1, 0))
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)
        return _format_tensor_batch_channel(t)
    elif torch is not None and isinstance(img, torch.Tensor):
        t = img.to(device=device, dtype=dtype)
        if to_zyx:
            if t.ndim == 2:
                t = t.t().contiguous()
            elif t.ndim == 3:
                t = t.permute(2, 1, 0).contiguous()
            elif t.ndim == 4 and t.shape[1] == 1:
                t = t.transpose(-2, -1).contiguous()
            elif t.ndim == 5 and t.shape[1] == 1:
                t = t.permute(0, 1, 4, 3, 2).contiguous()
        return _format_tensor_batch_channel(t)
    else:
        arr = np.asarray(_to_numpy(img), dtype=np.float32)
        if to_zyx:
            if arr.ndim == 2:
                arr = arr.T
            elif arr.ndim == 3:
                arr = np.transpose(arr, (2, 1, 0))
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)
        return _format_tensor_batch_channel(t)


def tensor_to_image(tensor, ref_image):
    """Convert a tensor / array to a float32 scalar ANTsImage.

    No axis transpose is done: the spatial axes must already be in ANTs (x, y, z) order (the
    ``image_to_tensor(..., to_zyx=False)`` layout). Leading size-1 axes are dropped; if the
    shape still differs from ``ref_image.shape`` all size-1 axes are squeezed.

    Parameters
    ----------
    tensor : Tensor or array-like
        (1, 1, *spatial), (1, *spatial) or (*spatial).
    ref_image : ANTsImage or None
        Supplies origin / spacing / direction. None: squeezed array with default geometry.

    Returns
    -------
    ANTsImage
    """
    if ants is None:
        raise ImportError("ANTsPy is required for tensor_to_image.")
    arr = _to_numpy(tensor)
    if ref_image is not None and isinstance(ref_image, ants.ANTsImage):
        target_shape = tuple(ref_image.shape)
        while arr.ndim > len(target_shape) and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape != target_shape:
            arr = arr.squeeze()
        return ants.from_numpy(
            arr.astype(np.float32),
            origin=ref_image.origin,
            spacing=ref_image.spacing,
            direction=ref_image.direction,
        )
    arr = arr.squeeze()
    return ants.from_numpy(arr.astype(np.float32))


def get_spatial_coordinate_grid(img, level=1, device="cpu"):
    """Physical points (x, y, z order) of a subsampled voxel lattice of an ANTsImage.

    Voxel indices ``0, level, 2 * level, ...`` (``img.shape[k] // level`` per axis, no
    half-voxel offset) are mapped with ``origin + (index * spacing) @ direction.T``.

    Points are ordered with x varying fastest (tensor (z, y, x) raster order).

    Parameters
    ----------
    img : ANTsImage
    level : int, default 1
        Subsampling factor.
    device : str or torch.device, default 'cpu'

    Returns
    -------
    (phys_coords_xyz, shape_zyx)
        float32 Tensor (N, dim) of physical points, and the subsampled grid shape in tensor
        (z, y, x) order.
    """
    dim = img.dimension
    device_obj = torch.device(device) if isinstance(device, str) else device
    sp_xyz = torch.tensor(img.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(img.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(img.direction, dtype=torch.float32, device=device_obj)

    shape_zyx = tuple(s // level for s in reversed(tuple(img.shape)))   # ANTs (x, y, z) -> (z, y, x)
    if dim == 3:
        grid_z = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
        grid_y = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
        grid_x = torch.linspace(0, shape_zyx[2] - 1, shape_zyx[2], device=device_obj)
        mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing="ij")
        vox_coords_xyz = (
            torch.stack([mesh_x, mesh_y, mesh_z], dim=-1).reshape(-1, 3) * level
        )
    else:
        grid_y = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
        grid_x = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        vox_coords_xyz = torch.stack([mesh_x, mesh_y], dim=-1).reshape(-1, 2) * level

    phys_coords_xyz = orig_xyz + (vox_coords_xyz * sp_xyz) @ dir_xyz.t()
    return phys_coords_xyz, shape_zyx


# ═══════════════════════════════════════════════════════════════════════════════
# Jacobian Determinant
# ═══════════════════════════════════════════════════════════════════════════════


def jacobian_determinant(disp, spacing=None, ref_image=None):
    """Jacobian determinant map det(I + grad u) of a physical (mm) displacement field (NumPy).

    Input handling:

    - ANTsImage: its ``.numpy()`` (ANTs layout: axes and components in (x, y, z) order).
    - torch / JAX tensor with ``ref_image`` given and ``shape[0] == 1``: converted from tensor
      layout with ``disp_tensor_to_itk`` (ANTs layout).
    - anything else: used as is after dropping a leading size-1 axis (no axis or component
      reversal, so a raw tensor-layout field is not converted).
    - a components-first array (dim, *spatial) is moved to channels-last when
      ``shape[0]`` is 2 or 3, ``shape[1] > 4`` and ``shape[-1]`` is not 2 or 3.
    - an array with ``dim + 2`` axes is treated as a batch and stacked.

    Computed as det(F) with F = ``deformation_gradient(disp, spacing, ref_image)``: physical
    du/dx from ``np.gradient`` (central, one-sided at borders) along each array axis divided
    by that axis's spacing, rotated by the full direction matrix (any orientation, including
    oblique), in 2-D and 3-D. Direction: an ANTsImage ``disp``'s own header, else
    ``ref_image``'s, else identity.

    Parameters
    ----------
    disp : ANTsImage, Tensor, jax.Array or ndarray
        See above.
    spacing : sequence of float, optional
        ANTs (x, y, z) order. None: ``ref_image.spacing``, else ones.
    ref_image : ANTsImage, optional
        Spacing, direction, and the tensor conversion described above.

    Returns
    -------
    ndarray (*spatial) (or (B, *spatial) for batched input). Values <= 0 mark folding.
    """
    F = deformation_gradient(disp, spacing=spacing, ref_image=ref_image)
    return np.linalg.det(F)


def jacobian_determinant_image(disp, ref_image):
    """``jacobian_determinant(disp, ref_image=ref_image)`` as a float32 ANTsImage with
    ``ref_image``'s geometry.

    The determinant array must come out in ANTs (x, y, z) axis order (ANTsImage input, or a
    (1, *spatial, dim) tensor, which is converted using ``ref_image``).

    Parameters
    ----------
    disp : ANTsImage, Tensor or ndarray
        Displacement field (see ``jacobian_determinant``).
    ref_image : ants.ANTsImage
        Spacing, direction and output geometry.

    Returns
    -------
    ants.ANTsImage
        Scalar ANTs image containing Jacobian determinant values.
    """
    if ants is None:
        raise ImportError("ANTsPy is required for jacobian_determinant_image.")
    detJ = jacobian_determinant(disp, ref_image=ref_image)
    return ants.from_numpy(
        detJ.astype(np.float32),
        origin=ref_image.origin,
        spacing=ref_image.spacing,
        direction=ref_image.direction,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Deformation Gradient & Polar Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


def polar_rotation_field(F: np.ndarray) -> np.ndarray:
    """Per-element polar-decomposition rotation ``R`` of an arbitrary batch of square
    matrices ``F`` (``F = R U``, via SVD: ``R = U @ Vh`` with the last row of ``Vh``
    negated wherever that would otherwise give a reflection, i.e. ``det(R) < 0``).

    Extracted as a standalone, reusable primitive (used internally by
    ``deformation_gradient(..., to_rotation=True)``, which computes the rotation of a
    single deformable warp's own Jacobian) so callers composing MULTIPLE Jacobians first
    -- e.g. an affine stage's constant rotation matrix with a deformable stage's own
    per-voxel Jacobian field, as in a `robust_affine` + `greedy` two-stage registration --
    get the mathematically correct rotation of the COMPOSED Jacobian, not the (generally
    different) composition of each stage's own separately-extracted rotation. Verified
    numerically (2026-10-07, antsxdwi tract-prior reorientation work): composing each
    stage's own polar rotation separately (``R_affine @ R_warp``) disagreed with the true
    empirical rotation (measured via finite differences through the real composed
    transform) by up to 0.15 in matrix entries -- composing the full Jacobians FIRST
    (``R_affine @ F_warp``, i.e. the affine's rotation times the warp's full, not yet
    decomposed, Jacobian) and polar-decomposing that PRODUCT matched to within ~0.01
    (consistent with finite-difference noise).

    Parameters
    ----------
    F : ndarray, shape (..., dim, dim)
        Any batch of square matrices (a single affine linear part broadcast against a
        per-voxel Jacobian field, a raw per-voxel Jacobian field on its own, etc.).

    Returns
    -------
    ndarray, same shape as ``F`` -- the per-element rotation.
    """
    U, _, Vh = np.linalg.svd(F)
    R = U @ Vh
    dets = np.linalg.det(R)
    reflection_mask = dets < 0
    if np.any(reflection_mask):
        Vh_copy = Vh.copy()
        Vh_copy[reflection_mask, -1, :] *= -1
        R[reflection_mask] = U[reflection_mask] @ Vh_copy[reflection_mask]
    return R


def polar_rotation_field_torch(
    F,
    device: str = "cpu",
    chunk_size: int = 65536,
    dtype=None,
):
    """Chunked, batched torch equivalent of :func:`polar_rotation_field` -- same math,
    numerically verified against it on real data (max abs diff ~1e-15 in float64 on CPU,
    ~6e-7 in float32; see ``tests/test_spatial_polar_rotation_torch.py``).

    Processes the flattened batch dimension in chunks of ``chunk_size`` so peak memory
    never scales with the full input size -- only one chunk's ``U``/``S``/``Vh`` (each
    ``chunk_size x dim x dim``) is ever materialized on ``device`` at a time, for the case
    this needs to run over a full volume's worth of per-voxel 3x3 matrices (hundreds of
    thousands to tens of millions).

    **Not currently GPU-accelerated on Apple Silicon**: measured (2026-10-07) that
    ``torch.linalg.svd`` has no native MPS kernel in this torch version -- it silently
    falls back to CPU internally, ~150x SLOWER than calling this function with
    ``device="cpu"`` directly, due to fallback-dispatch overhead. ``device="mps"`` is
    therefore redirected to ``"cpu"`` automatically inside this function (not left to
    silently eat that penalty). ``"cuda"`` is untested here but should genuinely
    accelerate once available, since CUDA does have a native batched SVD kernel. The
    chunking itself (bounding peak memory) is the real, always-applicable speed/safety
    win this function provides over a naive whole-volume ``polar_rotation_field`` call or
    a Python per-voxel loop, independent of which device ends up actually running the math.

    Parameters
    ----------
    F : ndarray or torch.Tensor, shape (..., dim, dim)
        Same contract as ``polar_rotation_field``. A numpy input is chunked and returned
        as numpy; a torch input is chunked and returned as a torch tensor on its original
        device, to avoid a surprising host<->device round trip for an already-GPU-resident
        caller.
    device : str, default "cpu"
        Compute device for the per-chunk SVD. ``mdf_distance_matrix``'s chunking
        convention (this module's own sibling pattern in
        ``antsxdwi.streamlines.clustering``) is followed here: never materialize more than
        one chunk's intermediates on-device at once.
    chunk_size : int, default 65536
        Voxels (flattened batch elements) processed per chunk.
    dtype : torch.dtype, optional
        Defaults to float64 for SVD numerical stability (matching
        ``polar_rotation_field``'s implicit numpy float64 behavior), unless the input is
        already a torch tensor of a specific dtype, in which case that dtype is kept.
    """
    if torch is None:
        raise ImportError("polar_rotation_field_torch requires torch to be installed")

    is_torch_input = isinstance(F, torch.Tensor)
    orig_device = F.device if is_torch_input else None
    orig_dtype = F.dtype if is_torch_input else None

    F_np = F.detach().cpu().numpy() if is_torch_input else np.asarray(F)
    orig_shape = F_np.shape
    dim = orig_shape[-1]
    F_flat = F_np.reshape(-1, dim, dim)
    n = F_flat.shape[0]

    dev = torch.device(device)
    if dev.type == "mps":
        # Measured (2026-10-07): torch.linalg.svd has NO native MPS kernel in this torch
        # version -- it silently dispatches to a CPU fallback internally (with a
        # UserWarning), ~150x slower than just calling this function with device="cpu"
        # directly (1.86s vs 0.012s for the same 8000-voxel chunk). Requesting "mps" here
        # would silently be much SLOWER than "cpu", not faster -- route to cpu explicitly
        # rather than let a caller pay that penalty by default in an ecosystem where MPS
        # is usually the right choice for everything else.
        dev = torch.device("cpu")
    if dtype is not None:
        compute_dtype = dtype
    elif dev.type == "mps":
        # Unreachable today (mps is redirected to cpu above) but kept for when/if torch
        # adds a native MPS SVD kernel -- float64 is unsupported on MPS regardless, and
        # float32 was verified (2026-10-07) to still agree with the float64 numpy
        # reference to ~1e-6, far tighter than this rotation field is ever consumed at
        # (a cosine-similarity direction score).
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.float64
    out_flat = np.empty_like(F_flat)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.as_tensor(F_flat[start:end], device=dev, dtype=compute_dtype)
        U, _, Vh = torch.linalg.svd(chunk)
        R = U @ Vh
        dets = torch.linalg.det(R)
        neg = dets < 0
        if bool(neg.any()):
            Vh_fixed = Vh.clone()
            Vh_fixed[neg, -1, :] *= -1
            R = torch.where(neg[:, None, None], U @ Vh_fixed, R)
        out_flat[start:end] = R.detach().cpu().numpy()

    result = out_flat.reshape(orig_shape)
    if is_torch_input:
        return_dtype = orig_dtype if orig_dtype is not None else compute_dtype
        return torch.as_tensor(result, device=orig_device, dtype=return_dtype)
    return result


def deformation_gradient(
    warp,
    to_rotation: bool = False,
    to_inverse_rotation: bool = False,
    spacing: tuple | None = None,
    direction: np.ndarray | None = None,
    ref_image: Any = None,
) -> np.ndarray:
    """Physical deformation gradient field F = I + du/dx of a displacement field (NumPy).

    ``F = I + G @ direction.T`` with ``G[k, j] = (du_k / d index_j) / spacing_j``
    (``np.gradient``), which is du/dx in physical coordinates for an orthonormal direction.
    The field must be in ANTs layout (spatial axes and components in (x, y, z) order).

    Parameters
    ----------
    warp : ANTsImage, Tensor or ndarray
        - ANTsImage: spacing and direction come from its header (passing ``spacing`` /
          ``direction`` too raises ValueError).
        - Tensor with an ANTsImage ``ref_image`` and ``shape[0] == 1``: converted from tensor
          layout with ``disp_tensor_to_itk`` and handled as an ANTsImage (passing ``spacing`` /
          ``direction`` too raises ValueError).
        - otherwise (ndarray, or tensor without such a ref_image): used as is (no layout
          conversion) after dropping a leading size-1 axis; components-first input is moved
          to channels-last by the same rule as ``jacobian_determinant``; ``dim + 2`` axes are
          treated as a batch.
    to_rotation : bool, default False
        Return the rotation R of the polar decomposition F = R U instead (SVD, ``U @ Vh`` with
        the last row of Vh negated where the determinant would be -1).
    to_inverse_rotation : bool, default False
        Return R.T (takes effect even if ``to_rotation`` is False).
    spacing : sequence of float, optional
        ANTs order. None: ``ref_image.spacing``, else ones.
    direction : array-like (dim, dim), optional
        None: ``ref_image.direction``, else identity.
    ref_image : ANTsImage, optional

    Returns
    -------
    ndarray float64 (*spatial, dim, dim) (or (B, *spatial, dim, dim)).
    """
    if ants is not None and isinstance(warp, ants.ANTsImage):
        if spacing is not None or direction is not None:
            raise ValueError("deformation_gradient: an ANTsImage carries its own geometry; do not "
                             "also pass spacing / direction")
        dim = warp.dimension
        spc = tuple(warp.spacing)
        tdir = np.asarray(warp.direction, dtype=np.float64)
        warpnp = warp.numpy()
    elif _is_tensor(warp):
        if (
            ref_image is not None
            and ants is not None
            and isinstance(ref_image, ants.ANTsImage)
            and warp.ndim >= 3
            and warp.shape[0] == 1
        ):
            if spacing is not None or direction is not None:
                raise ValueError("deformation_gradient: with a tensor and ref_image the geometry comes "
                                 "from ref_image; do not also pass spacing / direction")
            disp_img = disp_tensor_to_itk(warp, ref_image=ref_image)
            return deformation_gradient(
                disp_img,
                to_rotation=to_rotation,
                to_inverse_rotation=to_inverse_rotation,
            )
        warpnp = _to_numpy(warp)
        warpnp = _squeeze_batch(warpnp)
        if (
            warpnp.ndim >= 3
            and warpnp.shape[-1] not in (2, 3)
            and warpnp.shape[0] in (2, 3)
            and warpnp.shape[1] > 4
        ):
            warpnp = np.moveaxis(warpnp, 0, -1)
        dim = warpnp.shape[-1]
        spc = _get_spacing(ref_image=ref_image, spacing=spacing, ndim=dim)
        if direction is not None:
            tdir = np.asarray(direction, dtype=np.float64)
        elif (
            ref_image is not None
            and ants is not None
            and isinstance(ref_image, ants.ANTsImage)
        ):
            tdir = np.asarray(ref_image.direction, dtype=np.float64)
        else:
            tdir = np.eye(dim, dtype=np.float64)
    else:
        warpnp = _to_numpy(warp)
        warpnp = _squeeze_batch(warpnp)
        if (
            warpnp.ndim >= 3
            and warpnp.shape[-1] not in (2, 3)
            and warpnp.shape[0] in (2, 3)
            and warpnp.shape[1] > 4
        ):
            warpnp = np.moveaxis(warpnp, 0, -1)
        dim = warpnp.shape[-1]
        spc = _get_spacing(ref_image=ref_image, spacing=spacing, ndim=dim)
        if direction is not None:
            tdir = np.asarray(direction, dtype=np.float64)
        elif (
            ref_image is not None
            and ants is not None
            and isinstance(ref_image, ants.ANTsImage)
        ):
            tdir = np.asarray(ref_image.direction, dtype=np.float64)
        else:
            tdir = np.eye(dim, dtype=np.float64)

    if spc is None:
        spc = (1.0,) * dim

    if warpnp.ndim == dim + 2:
        return np.stack(
            [
                deformation_gradient(
                    warpnp[b],
                    to_rotation=to_rotation,
                    to_inverse_rotation=to_inverse_rotation,
                    spacing=spc,
                    direction=tdir,
                    ref_image=ref_image,
                )
                for b in range(warpnp.shape[0])
            ],
            axis=0,
        )

    gradient_list = [
        np.gradient(warpnp[..., k], *spc, axis=range(dim)) for k in range(dim)
    ]
    dg = np.stack([np.stack(grad_k, axis=-1) for grad_k in gradient_list], axis=-1)
    dg = (tdir @ dg).swapaxes(-1, -2)
    dg += np.eye(dim, dtype=dg.dtype)

    if to_rotation or to_inverse_rotation:
        dg = polar_rotation_field(dg)
        if to_inverse_rotation:
            dg = np.swapaxes(dg, -1, -2)

    return dg


# ═══════════════════════════════════════════════════════════════════════════════
# Deformation Statistics
# ═══════════════════════════════════════════════════════════════════════════════


def deformation_stats(disp, spacing=None, ref_image=None):
    """Summary statistics of a displacement field and its Jacobian determinant.

    The magnitude statistics use the raw array (leading size-1 axis dropped, components-first
    moved to channels-last when ``shape[0]`` is 2 or 3 and ``shape[1] > 4``); the
    determinant is ``jacobian_determinant(disp, spacing, ref_image)`` with that function's
    layout rules (no auto-detection of tensor vs ANTs layout beyond those).

    Parameters
    ----------
    disp : ANTsImage, Tensor or ndarray
        Displacement field; mm values give mm statistics.
    spacing : sequence of float, optional
        ANTs (x, y, z) order.
    ref_image : ants.ANTsImage, optional
        Passed to ``jacobian_determinant``.

    Returns
    -------
    dict
        Statistics dictionary containing:
        - 'detJ': np.ndarray — Jacobian determinant map
        - 'min_j': float — minimum det(J)
        - 'max_j': float — maximum det(J)
        - 'mean_j': float — mean det(J)
        - 'std_j': float — std dev of det(J)
        - 'folding_pct': float — percentage of voxels with det(J) ≤ 0
        - 'l2_norm': float — sqrt of the sum of squares over all voxels and components
        - 'mean_displacement': float — mean per-voxel displacement magnitude
    """
    arr = _to_numpy(disp)
    arr = _squeeze_batch(arr)

    # Handle component-first format
    if arr.ndim >= 3 and arr.shape[0] in (2, 3) and arr.shape[1] > 4:
        arr = np.moveaxis(arr, 0, -1)

    mag = np.sqrt(np.sum(arr**2, axis=-1))
    l2_norm = float(np.sqrt(np.sum(arr**2)))
    mean_disp = float(np.mean(mag))

    detJ = jacobian_determinant(disp, spacing=spacing, ref_image=ref_image)

    folding_mask = detJ <= 0.0
    folding_pct = float(np.mean(folding_mask) * 100.0)

    return {
        "detJ": detJ,
        "min_j": float(np.min(detJ)),
        "max_j": float(np.max(detJ)),
        "mean_j": float(np.mean(detJ)),
        "std_j": float(np.std(detJ)),
        "folding_pct": folding_pct,
        "l2_norm": l2_norm,
        "mean_displacement": mean_disp,
    }


__all__ = [
    "_to_numpy",
    "_is_tensor",
    "_squeeze_batch",
    "_get_spacing",
    "_get_physical_grid_torch_yfirst",
    "_physical_to_normalized_torch_yfirst",
    "_grid_to_physical_affine_torch_yfirst",
    "reverse_components",
    "reverse_metadata",
    "itk_shape_to_tensor_shape",
    "get_image_metadata",
    "export_ants_displacement_field",
    "disp_tensor_to_itk",
    "disp_itk_to_tensor",
    "create_ants_affine",
    "export_ants_affine_transform",
    "grid_to_physical_affine",
    "grid_to_physical_affine_torch",
    "physical_to_grid_affine",
    "get_physical_grid_torch",
    "get_physical_to_normalized_affine",
    "get_physical_to_normalized_affine_xyz",
    "lps_to_ras",
    "ras_to_lps",
    "physical_to_normalized_fast",
    "physical_to_normalized_torch",
    "physical_to_normalized_torch_cached",
    "get_identity_grid_torch",
    "compute_grid_to_physical_reference_matrix",
    "compute_autograd_physical_scale",
    "image_to_tensor",
    "tensor_to_image",
    "get_spatial_coordinate_grid",
    "jacobian_determinant",
    "jacobian_determinant_image",
    "deformation_gradient",
    "polar_rotation_field",
    "polar_rotation_field_torch",
    "deformation_stats",
    "normalized_to_physical_disp",
    "restriction_from_orientation",
]
