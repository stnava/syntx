"""
syntx.spatial — Centralized ITK/ANTs ↔ PyTorch/JAX Spatial Conversion Suite

This module provides the single source of truth for all coordinate and displacement
field conversions between ITK/ANTs physical space and PyTorch/JAX tensor space.

Two coordinate domains exist in syntx:

    ITK/ANTs domain:
        - Spatial axes: C-contiguous (Z, Y, X) in numpy arrays
        - Vector components: (dx, dy, dz) — physical coordinate order
        - Metadata (spacing, origin): (sx, sy, sz) — physical coordinate order
        - Direction matrix: maps physical (x, y, z) to voxel (x, y, z)

    Tensor domain (PyTorch / JAX):
        - Spatial axes: C-contiguous (Z, Y, X) — same memory layout
        - Vector components: (dz, dy, dx) — reversed tensor-index order
        - Metadata: reversed to (sz, sy, sx) for internal grid builders
        - Direction matrix: reversed [::-1, ::-1] for tensor-order operations

The ONLY differences are:
    1. Vector component order: ITK (dx,dy,dz) vs Tensor (dz,dy,dx) → [..., ::-1]
    2. Metadata ordering: ITK (x,y,z) vs Tensor (z,y,x) → reversed()

All public functions in this module accept mixed input types (torch.Tensor,
np.ndarray, ants.ANTsImage, jax.Array) and auto-detect the domain.
"""

import os
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
    """Convert any array-like to numpy, stripping batch dimensions from tensors."""
    if x is None:
        return None
    if ants is not None and isinstance(x, ants.ANTsImage):
        return x.numpy()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):  # jax arrays or similar
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
    """Remove leading batch dimension if present: (1, *spatial, dim) → (*spatial, dim)."""
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _get_spacing(ref_image=None, spacing=None, ndim=None):
    """Extract spacing tuple from ref_image or explicit spacing argument."""
    if spacing is not None:
        if hasattr(spacing, 'tolist'):
            spacing = spacing.tolist()
        return tuple(spacing)
    if ref_image is not None and ants is not None and isinstance(ref_image, ants.ANTsImage):
        return tuple(ref_image.spacing)
    if ndim is not None:
        return (1.0,) * ndim
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Component & Metadata Reversal Primitives
# ═══════════════════════════════════════════════════════════════════════════════

def reverse_components(disp):
    """Reverse vector component order along the last axis.

    Converts between ITK (dx, dy, dz) and Tensor (dz, dy, dx) orderings.
    This is an involution (self-inverting): applying it twice returns the original.
    Preserves input container type (torch.Tensor vs np.ndarray), device, dtype,
    and autograd computation graph.

    Parameters
    ----------
    disp : Tensor, ndarray, or array-like
        Displacement field or coordinates with vector components in the last dimension.
        Shape: (*spatial, dim) or (batch, *spatial, dim). Accepts torch.Tensor,
        np.ndarray, jax.Array, or list.

    Returns
    -------
    Tensor or ndarray
        Array with reversed component order matching the input type.
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
    """Reverse ITK (x,y,z) metadata to tensor (z,y,x) order.

    Parameters
    ----------
    spacing : tuple or array-like
        Voxel spacing in ITK order (sx, sy, sz).
    origin : tuple or array-like
        Image origin in ITK order (ox, oy, oz).
    direction : np.ndarray or array-like
        Direction cosine matrix mapping physical (x,y,z) to voxel (x,y,z).

    Returns
    -------
    tuple
        (spacing_rev, origin_rev, direction_rev) in tensor (z,y,x) order.
    """
    if hasattr(spacing, 'tolist'):
        spacing = spacing.tolist()
    if hasattr(origin, 'tolist'):
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
    """Convert ITK spatial shape (Nx, Ny, Nz) to Tensor spatial shape (Nz, Ny, Nx).

    Parameters
    ----------
    shape : tuple, list, or array-like
        Spatial shape in ITK order (Nx, Ny[, Nz]).

    Returns
    -------
    tuple
        Spatial shape in Tensor order (Nz, Ny[, Nx]).
    """
    if hasattr(shape, 'tolist'):
        shape = shape.tolist()
    return tuple(reversed(shape))


def get_image_metadata(img):
    """Extract spatial metadata dictionary from an ANTsImage.

    Returns a dict compatible with SyNToTransform and other syntx internals:
    {'origin': tuple, 'spacing': tuple, 'direction': np.ndarray, 'shape': tuple}

    Parameters
    ----------
    img : ants.ANTsImage
        Input ANTs image.

    Returns
    -------
    dict
        Metadata dictionary with origin, spacing, direction, shape.
    """
    return {
        'origin': tuple(img.origin),
        'spacing': tuple(img.spacing),
        'direction': np.array(img.direction),
        'shape': tuple(img.shape),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Displacement Field Conversions
# ═══════════════════════════════════════════════════════════════════════════════

def export_ants_displacement_field(disp, origin=None, spacing=None, direction=None, ref_image=None):
    """Standardized conversion of PyTorch/JAX physical displacement arrays into ITK-compatible ANTsImage displacement fields.

    Parameters
    ----------
    disp : np.ndarray, torch.Tensor, or jax.Array
        Array of shape `(1, *spatial, dim)` or `(*spatial, dim)` containing ZYX physical displacement vectors.
    origin : tuple or list, optional
        Image origin in XYZ order.
    spacing : tuple or list, optional
        Voxel spacing in XYZ order.
    direction : np.ndarray or list of list, optional
        Direction matrix in XYZ order.
    ref_image : ants.ANTsImage, optional
        Reference image providing origin, spacing, direction.

    Returns
    -------
    ants.ANTsImage
        ANTs vector image with `has_components=True`.
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

    Performs TWO coordinate domain transformations:
    1. Spatial axis transposition: tensor order (Z,Y,X) → ANTs order (X,Y,Z)
    2. Component reversal: tensor (dz,dy,dx) → ITK (dx,dy,dz)

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
        origin = ref_image.get('origin')
        spacing = ref_image.get('spacing')
        direction = ref_image.get('direction')
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


def _single_disp_itk_to_tensor(disp_img, device='cpu'):
    if isinstance(disp_img, (str, os.PathLike)):
        disp_img = ants.image_read(str(disp_img))
    arr = disp_img.numpy() if hasattr(disp_img, 'numpy') else np.asarray(disp_img)
    dim = arr.shape[-1]
    if dim == 2:
        arr = np.transpose(arr, (1, 0, 2))
    elif dim == 3:
        arr = np.transpose(arr, (2, 1, 0, 3))
    arr = arr[..., ::-1]

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
    return tensor


def disp_itk_to_tensor(disp_img, device='cpu'):
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
        component order and matching spatial layout. B=1 for single input, B=N for sequence.
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
    """Convert a normalized grid displacement field [-1, 1] to a tensor-domain physical displacement field.

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
        Image origin in ITK order (unused for displacement, but passed for metadata compatibility).
    device : str or torch.device, optional
        Target compute device.
    dtype : torch.dtype, optional
        Target data type. Defaults to torch.float32.

    Returns
    -------
    torch.Tensor
        Physical displacement tensor in Tensor domain (components in (dz, dy, dx) order, mm units).
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
    """Create an ITK AffineTransform ANTsTransform object from physical matrix and translation.

    Parameters
    ----------
    M_phys : np.ndarray or torch.Tensor
        Physical rotation/scale/shear matrix (dim x dim), or full (dim+1 x dim+1) / (dim x dim+1) affine matrix.
    t_phys : np.ndarray or torch.Tensor, optional
        Physical translation vector (dim,). If None, extracted from M_phys if M_phys is (dim, dim+1) or (dim+1, dim+1).
    dim : int, optional
        Spatial dimension (2 or 3). Inferred if None.
    fixed_params : np.ndarray or torch.Tensor, optional
        Fixed parameters (center of rotation), default zeros(dim).

    Returns
    -------
    ants.ANTsTransform
        Configured ANTsTransform object.
    """
    if ants is None:
        raise ImportError("ANTsPy is required for create_ants_affine.")
    M_phys = np.asarray(_to_numpy(M_phys), dtype=np.float32)

    if t_phys is None:
        if M_phys.ndim == 2 and M_phys.shape[1] == M_phys.shape[0] + 1:
            t_phys = M_phys[:, -1]
            M_phys = M_phys[:, :-1]
        elif M_phys.ndim == 2 and M_phys.shape[0] == M_phys.shape[1] and M_phys.shape[0] in (3, 4):
            # Homogeneous matrix (dim+1, dim+1)
            t_phys = M_phys[:-1, -1]
            M_phys = M_phys[:-1, :-1]
        else:
            raise ValueError("t_phys must be provided if M_phys is not an augmented/homogeneous matrix.")
    else:
        t_phys = np.asarray(_to_numpy(t_phys), dtype=np.float32).ravel()

    if dim is None:
        dim = M_phys.shape[0]

    tx = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    if fixed_params is None:
        tx.set_fixed_parameters(np.zeros(dim))
    else:
        tx.set_fixed_parameters(np.asarray(_to_numpy(fixed_params), dtype=np.float32).ravel())
    return tx


def export_ants_affine_transform(M_phys, t_phys, dim: int = None, filename: str = None):
    """Standardized export of physical affine parameters `(M_phys, t_phys)` into ITK-compatible ANTs transforms.

    Guarantees exact ITK parameter layout (`M_phys.ravel()` for forward, `M_phys_inv.ravel()` for inverse).

    Parameters
    ----------
    M_phys : np.ndarray or torch.Tensor
        Physical rotation/scale/shear matrix (`2x2` or `3x3`).
    t_phys : np.ndarray or torch.Tensor
        Physical translation vector.
    dim : int, optional
        Spatial dimensionality (2 or 3). Inferred if None.
    filename : str, optional
        File path to write forward transform matrix file.

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


def _grid_to_physical_affine_torch_yfirst(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
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
    Wx = Kx_inv @ Sx_inv @ Dx.t()
    bx = - Kx_inv @ Sx_inv @ Dx.t() @ Ox - Kx_inv @ Cx

    Vy = Dy @ torch.diag(Sy) @ Ky
    cy = Dy @ torch.diag(Sy) @ Cy + Oy

    A_grid = T_grid[:dim, :dim].to(calc_dtype)
    t_grid = T_grid[:dim, dim].to(calc_dtype)

    M_phys = (Vy @ A_grid @ Wx).to(orig_dtype)
    t_phys = (Vy @ (A_grid @ bx + t_grid) + cy).to(orig_dtype)
    return M_phys, t_phys


def grid_to_physical_affine_torch(T_grid, fixed_shape, fixed_spacing, fixed_origin, fixed_direction, moving_shape, moving_spacing, moving_origin, moving_direction):
    """Convert normalized grid affine matrix T_grid into physical (M_phys, t_phys) in PyTorch tensor space."""
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
    M_phys_zyx, t_phys_zyx = _grid_to_physical_affine_torch_yfirst(T_yx, fixed_shape, fs_rev, fo_rev, fd_rev, moving_shape, ms_rev, mo_rev, md_rev)

    # Return ZYX physical affine matrices directly to match PyTorch tensor coordinate ordering (Z, Y, X)
    return M_phys_zyx, t_phys_zyx


def grid_to_physical_affine(T_grid, fixed, moving):
    """Convert normalized grid affine matrix T_grid into physical (M_phys, t_phys) in ITK XYZ order."""
    if hasattr(T_grid, 'detach'):
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
    Wx = Kx_inv @ Sx_inv @ Dx.T
    bx = - Kx_inv @ Sx_inv @ Dx.T @ Ox - Kx_inv @ Cx

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
    """Convert physical affine parameters (M_phys, t_phys) to normalized grid affine matrix T_grid."""
    if hasattr(M_phys, 'detach'):
        M_phys = M_phys.detach().cpu().numpy()
    if hasattr(t_phys, 'detach'):
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
    bx = - np.linalg.inv(Kx) @ np.linalg.inv(np.diag(Sx)) @ Dx.T @ Ox - np.linalg.inv(Kx) @ Cx

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

def _get_physical_grid_torch_yfirst(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    dim = len(shape)
    grids = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    idxs = torch.stack(meshgrid, dim=-1)
    spacing_t = torch.tensor(spacing, device=device, dtype=dtype)
    origin_t = torch.tensor(origin, device=device, dtype=dtype)
    direction_t = torch.tensor(direction, device=device, dtype=dtype)

    scaled = idxs * spacing_t
    flat_scaled = scaled.view(-1, dim)
    flat_phys = flat_scaled @ direction_t.t() + origin_t
    return flat_phys.view(*shape, dim).unsqueeze(0)


def get_physical_grid_torch(shape, spacing, origin, direction, device='cpu', dtype=torch.float32):
    """Generate physical coordinate grid tensor of shape `(1, *shape, dim)` from spatial metadata."""
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _get_physical_grid_torch_yfirst(shape, spacing_rev, origin_rev, direction_rev, device, dtype)


def _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing, origin, direction):
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
    """Map physical coordinates to normalized grid coordinates in [-1, 1]."""
    # target_shape is in tensor order (Z, Y, X). _yfirst expects all params in Z-first order.
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    dir_arr = np.asarray(direction)
    if dir_arr.ndim == 1:
        dim = len(target_shape)
        dir_arr = dir_arr.reshape(dim, dim)
    direction_rev = dir_arr[::-1, ::-1].copy()
    return _physical_to_normalized_torch_yfirst(phys_coords, target_shape, spacing_rev, origin_rev, direction_rev)


def physical_to_normalized_torch_cached(phys_coords, shape_t, spacing_t, origin_t, direction_t):
    """Fast inner-loop physical coordinate normalization to [-1, 1] using pre-cached metadata tensors."""
    dim = phys_coords.shape[-1]
    flat_phys = phys_coords.view(-1, dim)
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = - (origin_t @ M) - 1.0
    flat_norm = flat_phys @ M + b
    norm_coords = torch.flip(flat_norm, dims=[-1])
    return norm_coords.view(phys_coords.shape)


def get_identity_grid_torch(target_shape, device='cpu', dtype=torch.float32):
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
    grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in target_shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
    return identity


def compute_grid_to_physical_reference_matrix(shape, spacing, origin, direction, device=None, dtype=None) -> torch.Tensor:
    """Computes homogeneous transformation matrix H mapping normalized grid coordinates [-1, 1] to physical scanner space.

    Parameters
    ----------
    shape : tuple of int
        Image grid shape.
    spacing : tuple of float
        Voxel spacing in XYZ order.
    origin : tuple of float
        Image origin in XYZ order.
    direction : np.ndarray or list of list
        Direction matrix in XYZ order.
    device : str or torch.device, optional
        Target PyTorch compute device.
    dtype : torch.dtype, optional
        Target PyTorch data type.

    Returns
    -------
    torch.Tensor
        (dim+1, dim+1) homogeneous transformation matrix mapping normalized grid to physical space.
    """
    dim = len(shape)
    if device is None:
        device = 'cpu'
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
    """Compute coordinate scaling vector converting autograd normalized grid gradients to physical displacement gradients.

    Follows Syntx Registration Guardrails:
        s_phys = torch.flip((shape_t - 1.0) * spacing_t / 2.0, dims=[0])

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
        Scaling vector of shape (dim,) converting autograd gradients to physical units.
    """
    if dtype is None:
        dtype = torch.float32

    if not isinstance(shape, torch.Tensor):
        shape_t = torch.tensor(list(shape), device=device, dtype=dtype)
    else:
        shape_t = shape.to(device=device if device is not None else shape.device, dtype=dtype)

    if not isinstance(spacing, torch.Tensor):
        spacing_t = torch.tensor(list(spacing), device=device, dtype=dtype)
    else:
        spacing_t = spacing.to(device=device if device is not None else spacing.device, dtype=dtype)

    scale = (shape_t - 1.0) * spacing_t / 2.0
    return torch.flip(scale, dims=[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Image Conversions
# ═══════════════════════════════════════════════════════════════════════════════

def _format_tensor_batch_channel(t):
    """Format tensor to have leading batch and channel dimensions (B, 1, *spatial)."""
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


def image_to_tensor(img, device='cpu', dtype=None, to_zyx=False):
    """Convert an ANTsImage or array to a PyTorch tensor with batch/channel dims.

    Parameters
    ----------
    img : ants.ANTsImage, torch.Tensor, or array-like
        Input image.
    device : str or torch.device
        Target device.
    dtype : torch.dtype, optional
        Target dtype. Defaults to float32.
    to_zyx : bool, optional
        If True, transpose spatial axes from ITK XYZ to PyTorch ZYX.
        Defaults to False for backward compatibility.

    Returns
    -------
    torch.Tensor
        Shape (1, 1, *spatial) for use with PyTorch convolution and grid_sample.
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
    """Convert a tensor back to an ANTsImage with reference metadata.

    Parameters
    ----------
    tensor : torch.Tensor or array-like
        Shape (1, 1, *spatial), (1, *spatial), or (*spatial).
    ref_image : ants.ANTsImage
        Reference image providing origin, spacing, direction.

    Returns
    -------
    ants.ANTsImage
        Scalar ANTs image with proper metadata.
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


def get_spatial_coordinate_grid(img, level=1, device='cpu'):
    """Generates physical coordinate grid points in XYZ order for an ANTsImage at a given pyramid level.

    Parameters
    ----------
    img : ants.ANTsImage
        Reference image providing shape, spacing, origin, direction.
    level : int
        Pyramid downsampling factor.
    device : str or torch.device
        Target PyTorch device.

    Returns
    -------
    tuple
        (phys_coords_xyz, shape_zyx)
        - phys_coords_xyz: torch.Tensor of shape (N_voxels, dim) in physical XYZ coordinates.
        - shape_zyx: tuple of downsampled image shape (Z_lev, Y_lev, X_lev).
    """
    dim = img.dimension
    device_obj = torch.device(device) if isinstance(device, str) else device
    sp_xyz = torch.tensor(img.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(img.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(img.direction, dtype=torch.float32, device=device_obj)

    shape_zyx = tuple(s // level for s in img.shape)
    if dim == 3:
        grid_z = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
        grid_y = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
        grid_x = torch.linspace(0, shape_zyx[2] - 1, shape_zyx[2], device=device_obj)
        mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
        vox_coords_xyz = torch.stack([mesh_x, mesh_y, mesh_z], dim=-1).reshape(-1, 3) * level
    else:
        grid_y = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
        grid_x = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        vox_coords_xyz = torch.stack([mesh_x, mesh_y], dim=-1).reshape(-1, 2) * level

    phys_coords_xyz = orig_xyz + (vox_coords_xyz * sp_xyz) @ dir_xyz.t()
    return phys_coords_xyz, shape_zyx


# ═══════════════════════════════════════════════════════════════════════════════
# Jacobian Determinant — ANTs-validated (r > 0.999)
# ═══════════════════════════════════════════════════════════════════════════════

def jacobian_determinant(disp, spacing=None, ref_image=None):
    """Compute the Jacobian determinant map from a displacement field.

    Validated against ANTs C++ ITK reference (ants.create_jacobian_determinant_image):
    - 2D: Pearson r > 0.999 on ANTsPy r16↔r64 benchmark
    - 3D: Pearson r > 0.999 on Mindboggle Pair 08 benchmark

    The displacement field must be in ITK component order (dx, dy[, dz]).
    If a PyTorch/JAX tensor is passed, components are auto-reversed from
    tensor order (dz, dy, dx) to ITK order before computation.

    Parameters
    ----------
    disp : array-like
        Displacement field. Accepted formats:
        - np.ndarray of shape (*spatial, dim): ITK component order assumed
        - torch.Tensor of shape (1, *spatial, dim): tensor order, auto-reversed
        - ants.ANTsImage: ITK component order, extracted via .numpy()
    spacing : tuple, optional
        Voxel spacing in ITK physical order (sx, sy[, sz]).
        Extracted from ref_image if not provided. Defaults to (1.0,)*dim.
    ref_image : ants.ANTsImage, optional
        Reference image for spacing extraction.

    Returns
    -------
    np.ndarray
        Jacobian determinant map of shape (*spatial).
        Values > 1.0 indicate local expansion, < 1.0 indicate compression,
        ≤ 0.0 indicate topology-violating grid folding.
    """
    # Auto-detect and convert input
    if ants is not None and isinstance(disp, ants.ANTsImage):
        arr = disp.numpy()
    elif _is_tensor(disp):
        # Convert tensor if ref_image is provided and batch size is 1
        if ref_image is not None and disp.ndim >= 3 and disp.shape[0] == 1:
            # Raw model displacement tensor in PyTorch domain (1, *spatial, dim)
            disp_img = disp_tensor_to_itk(disp, ref_image=ref_image)
            arr = disp_img.numpy()
        else:
            arr = _to_numpy(disp)
            arr = _squeeze_batch(arr)
    else:
        arr = _to_numpy(disp)
        arr = _squeeze_batch(arr)

    # Handle component-first format: (dim, *spatial) → (*spatial, dim)
    if arr.ndim >= 3 and arr.shape[-1] not in (2, 3) and arr.shape[0] in (2, 3) and arr.shape[1] > 4:
        arr = np.moveaxis(arr, 0, -1)

    dim = arr.shape[-1]
    sp = _get_spacing(ref_image=ref_image, spacing=spacing, ndim=dim)
    if sp is None:
        sp = (1.0,) * dim

    if arr.ndim == dim + 2:
        return np.stack(
            [jacobian_determinant(arr[b], spacing=sp, ref_image=ref_image) for b in range(arr.shape[0])],
            axis=0
        )

    if dim == 3 and arr.ndim == 4:
        ref_obj = ref_image if (ref_image is not None and isinstance(ref_image, ants.ANTsImage)) else (disp if isinstance(disp, ants.ANTsImage) else None)
        dir_diag = np.diag(ref_obj.direction) if ref_obj is not None else np.ones(3)
        sp_XYZ = [sp[0], sp[1], sp[2]]

        J = np.zeros((*arr.shape[:3], 3, 3), dtype=np.float32)
        for i in range(3):
            sign_i = float(dir_diag[i]) if i < len(dir_diag) else 1.0
            for j in range(3):
                deriv = np.gradient(arr[..., i], axis=j) / sp_XYZ[j]
                if i == j:
                    J[..., i, j] = 1.0 + sign_i * deriv
                else:
                    J[..., i, j] = sign_i * deriv

        return np.linalg.det(J)

    elif dim == 2 and arr.ndim == 3:
        sp_axis = [sp[1], sp[0]]  # spacing per axis: [sp_y, sp_x]

        du_0_d0 = np.gradient(arr[..., 0], axis=0) / sp_axis[0]  # d(dy)/dY
        du_0_d1 = np.gradient(arr[..., 0], axis=1) / sp_axis[1]  # d(dy)/dX
        du_1_d0 = np.gradient(arr[..., 1], axis=0) / sp_axis[0]  # d(dx)/dY
        du_1_d1 = np.gradient(arr[..., 1], axis=1) / sp_axis[1]  # d(dx)/dX

        return (1.0 + du_0_d0) * (1.0 + du_1_d1) - du_0_d1 * du_1_d0

    else:
        return np.ones(arr.shape[:-1], dtype=np.float32)


def jacobian_determinant_image(disp, ref_image):
    """Compute Jacobian determinant and return as ANTsImage.

    Parameters
    ----------
    disp : array-like
        Displacement field.
    ref_image : ants.ANTsImage
        Reference image for spacing and metadata.

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
# Deformation Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def deformation_stats(disp, spacing=None, ref_image=None):
    """Compute comprehensive deformation statistics from a displacement field.

    Parameters
    ----------
    disp : array-like
        Displacement field (ITK or tensor domain — auto-detected).
    spacing : tuple, optional
        Voxel spacing in ITK physical order.
    ref_image : ants.ANTsImage, optional
        Reference image for spacing extraction.

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
        - 'l2_norm': float — L2 norm of displacement field
        - 'mean_displacement': float — mean displacement magnitude (mm)
    """
    arr = _to_numpy(disp)
    arr = _squeeze_batch(arr)

    # Handle component-first format
    if arr.ndim >= 3 and arr.shape[0] in (2, 3) and arr.shape[1] > 4:
        arr = np.moveaxis(arr, 0, -1)

    mag = np.sqrt(np.sum(arr ** 2, axis=-1))
    l2_norm = float(np.sqrt(np.sum(arr ** 2)))
    mean_disp = float(np.mean(mag))

    detJ = jacobian_determinant(disp, spacing=spacing, ref_image=ref_image)

    folding_mask = detJ <= 0.0
    folding_pct = float(np.mean(folding_mask) * 100.0)

    return {
        'detJ': detJ,
        'min_j': float(np.min(detJ)),
        'max_j': float(np.max(detJ)),
        'mean_j': float(np.mean(detJ)),
        'std_j': float(np.std(detJ)),
        'folding_pct': folding_pct,
        'l2_norm': l2_norm,
        'mean_displacement': mean_disp,
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
    "deformation_stats",
    "normalized_to_physical_disp",
]
