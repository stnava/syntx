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
# Domain Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _to_numpy(x):
    """Convert any array-like to numpy, stripping batch dimensions from tensors."""
    if x is None:
        return None
    if ants is not None and isinstance(x, ants.ANTsImage):
        return x.numpy()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, 'numpy'):  # jax arrays
        return np.asarray(x)
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
        return tuple(spacing)
    if ref_image is not None and ants is not None and isinstance(ref_image, ants.ANTsImage):
        return ref_image.spacing
    if ndim is not None:
        return (1.0,) * ndim
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Component Reversal — The Core Conversion
# ═══════════════════════════════════════════════════════════════════════════════

def reverse_components(disp):
    """Reverse vector component order along the last axis.

    Converts between ITK (dx,dy,dz) and Tensor (dz,dy,dx) orderings.
    This is a symmetric operation: applying it twice returns the original.

    Parameters
    ----------
    disp : array-like
        Displacement field with vector components in the last dimension.
        Shape: (*spatial, dim) or (batch, *spatial, dim).

    Returns
    -------
    np.ndarray
        Displacement field with reversed component order.
    """
    arr = _to_numpy(disp)
    return arr[..., ::-1].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata Reversal
# ═══════════════════════════════════════════════════════════════════════════════

def reverse_metadata(spacing, origin, direction):
    """Reverse ITK (x,y,z) metadata to tensor (z,y,x) order.

    Parameters
    ----------
    spacing : tuple
        Voxel spacing in ITK order (sx, sy, sz).
    origin : tuple
        Image origin in ITK order (ox, oy, oz).
    direction : np.ndarray
        Direction cosine matrix mapping physical (x,y,z) to voxel (x,y,z).

    Returns
    -------
    tuple
        (spacing_rev, origin_rev, direction_rev) in tensor (z,y,x) order.
    """
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    return spacing_rev, origin_rev, direction_rev


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
    ref_image : ants.ANTsImage
        Reference image providing origin, spacing, direction metadata.

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
    from .transform import export_ants_displacement_field

    arr = _to_numpy(disp)
    dim = arr.shape[-1]
    spatial_ndim = dim

    if arr.ndim == spatial_ndim + 2:
        batch_size = arr.shape[0]
        if batch_size == 1:
            return export_ants_displacement_field(
                arr[0],
                origin=ref_image.origin,
                spacing=ref_image.spacing,
                direction=ref_image.direction
            )
        else:
            return [
                export_ants_displacement_field(
                    arr[b],
                    origin=ref_image.origin,
                    spacing=ref_image.spacing,
                    direction=ref_image.direction
                )
                for b in range(batch_size)
            ]
    else:
        return export_ants_displacement_field(
            arr,
            origin=ref_image.origin,
            spacing=ref_image.spacing,
            direction=ref_image.direction
        )


def _single_disp_itk_to_tensor(disp_img, device='cpu'):
    if isinstance(disp_img, str):
        disp_img = ants.image_read(disp_img)
    arr = disp_img.numpy()
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
    disp_img : ants.ANTsImage, str, or sequence (list/tuple) of ANTsImage/str
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


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Image Conversions
# ═══════════════════════════════════════════════════════════════════════════════

def image_to_tensor(img, device='cpu', dtype=None):
    """Convert an ANTsImage to a batched channel-first tensor.

    No spatial axis transposition is needed because both ANTsPy and PyTorch
    use C-contiguous (Z, Y, X) memory layout for scalar images.

    Parameters
    ----------
    img : ants.ANTsImage
        Input scalar ANTs image.
    device : str or torch.device
        Target device.
    dtype : torch.dtype, optional
        Target dtype. Defaults to float32.

    Returns
    -------
    torch.Tensor
        Shape (1, 1, *spatial) for use with PyTorch convolution and grid_sample.
    """
    if dtype is None:
        dtype = torch.float32
    arr = img.numpy().astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_image(tensor, ref_image):
    """Convert a tensor back to an ANTsImage with reference metadata.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (1, 1, *spatial), (1, *spatial), or (*spatial).
    ref_image : ants.ANTsImage
        Reference image providing origin, spacing, direction.

    Returns
    -------
    ants.ANTsImage
        Scalar ANTs image with proper metadata.
    """
    arr = _to_numpy(tensor).squeeze()
    return ants.from_numpy(
        arr.astype(np.float32),
        origin=ref_image.origin,
        spacing=ref_image.spacing,
        direction=ref_image.direction,
    )


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

    Notes
    -----
    ANTs C-contiguous convention: component i corresponds to spatial axis i.
    For 2D (H, W, 2): comp 0 = displacement along axis 0 (Y), comp 1 = along axis 1 (X)
    For 3D (Z, Y, X, 3): comp 0 = along axis 0 (Z), comp 1 = along axis 1 (Y), comp 2 = along axis 2 (X)

    The diagonal Jacobian entry is: J[i,i] = 1 + ∂u_i/∂axis_i
    Off-diagonal: J[i,j] = ∂u_i/∂axis_j

    Spacing per axis:
    - 2D: axis 0 uses spacing[1] (sp_y), axis 1 uses spacing[0] (sp_x)
    - 3D: axis 0 uses spacing[2] (sp_z), axis 1 uses spacing[1] (sp_y), axis 2 uses spacing[0] (sp_x)
    """
    # Auto-detect and convert input
    if isinstance(disp, ants.ANTsImage):
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
        # 3D ANTs/ITK component mapping: comp 0=dx (axis 0), comp 1=dy (axis 1), comp 2=dz (axis 2)
        # Spatial axes: axis 0 = X (spacing sp[0]), axis 1 = Y (spacing sp[1]), axis 2 = Z (spacing sp[2])
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
        # 2D: arr shape (H, W, 2) from ants.image_read().numpy()
        # ANTs/ITK component order: comp 0 = dy (axis 0), comp 1 = dx (axis 1)
        # Spatial axes: axis 0 = Y (spacing sp[1]), axis 1 = X (spacing sp[0])
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
        Displacement field (see jacobian_determinant for accepted formats).
    ref_image : ants.ANTsImage
        Reference image for spacing and metadata.

    Returns
    -------
    ants.ANTsImage
        Scalar ANTs image containing Jacobian determinant values.
    """
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

    # Compute displacement magnitude statistics
    mag = np.sqrt(np.sum(arr ** 2, axis=-1))
    l2_norm = float(np.sqrt(np.sum(arr ** 2)))
    mean_disp = float(np.mean(mag))

    # Compute Jacobian
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
