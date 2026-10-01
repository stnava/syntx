"""
Finite-difference spatial Jacobians and Jacobian determinants of displacement fields (torch).

``_spatial_jacobian_nd`` returns the full derivative matrix of a channels-last field (also used
for image gradients by SyN / TVF); ``compute_jacobian_determinant_nd`` and
``compute_physical_jacobian_determinant`` return det(I + grad u) maps;
``compute_jacobian_hinge_penalty`` is a fold penalty built on the determinant.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _spatial_jacobian_nd(field: torch.Tensor, physical_spacing=None, method='central') -> torch.Tensor:
    """
    Spatial derivative matrix of a channels-last field, by finite differences.

    Parameters
    ----------
    field : Tensor (B, *spatial, d)
        Any number of components ``d`` (``d = 1`` gives an image gradient).
    physical_spacing : sequence of float, optional
        Grid step per spatial axis in tensor order (z, y, x). None: normalised coordinates,
        ``2 / (n - 1)`` per axis.
    method : str, default 'central'
        - 'central': ``torch.gradient`` (second-order central differences, one-sided at the
          border).
        - 'bspline': 5-tap kernel ``[-1, -8, 0, 8, 1] / 12`` with replicate padding. Note: this
          kernel returns 5/3 times the true slope of a linear field (it is not the standard
          ``[1, -8, 0, 8, -1] / 12`` stencil).
        Any other value falls through to 'central'.

    Returns
    -------
    Tensor (B, *spatial, d, n_spatial)
        ``J[..., i, j] = d field_i / d x_j`` with the derivative axis ``j`` in (x, y, z) order
        (the last tensor axis first). The component axis ``i`` is left in the input order.
    """
    dim = field.shape[-1]
    spatial = field.shape[1:-1]
    if physical_spacing is not None:
        spacings = list(physical_spacing)
    else:
        spacings = [2.0 / (s - 1) for s in spatial]
    
    if method == 'bspline':
        grads = []
        for i, sp in enumerate(spacings):
            k_np = np.array([-1/12, -8/12, 0.0, 8/12, 1/12], dtype=np.float32) / sp
            k_t = torch.from_numpy(k_np).to(device=field.device, dtype=field.dtype)
            
            # Permute spatial dim i to the last dimension
            perm = [0] + [j + 1 for j in range(len(spatial)) if j != i] + [len(spatial) + 1, i + 1]
            field_perm = field.permute(perm)
            orig_shape = field_perm.shape
            flat_in = field_perm.reshape(-1, 1, orig_shape[-1])
            padded = F.pad(flat_in, (2, 2), mode='replicate')
            k_view = k_t.view(1, 1, 5)
            conv_out = F.conv1d(padded, k_view)
            conv_restored = conv_out.view(*orig_shape[:-1], conv_out.shape[-1])
            
            inv_perm = [0] * len(perm)
            for new_pos, orig_pos in enumerate(perm):
                inv_perm[orig_pos] = new_pos
            g_i = conv_restored.permute(inv_perm)
            grads.append(g_i)
        return torch.stack(list(reversed(grads)), dim=-1)
    
    # torch.gradient returns a list of gradients, one per spatial dimension (ij order)
    grads = torch.gradient(field, spacing=spacings, dim=list(range(1, len(spatial) + 1)))
    
    # grads[k] shape: (B, *spatial, d) = derivative of all field components w.r.t. spatial dim k
    # Reverse to match our (x,y,[z]) component ordering convention
    return torch.stack(list(reversed(grads)), dim=-1)  # (B, *spatial, d, d)


def compute_jacobian_determinant_nd(warp_field: torch.Tensor, physical_spacing=None, method: str = 'central', **kwargs) -> torch.Tensor:
    """
    Jacobian determinant det(I + grad u) of a displacement field u.

    Three code paths, chosen by ``method`` and by whether the field is physical:

    - ``method='bspline'``: derivatives from ``_spatial_jacobian_nd(..., 'bspline')`` with
      ``physical_spacing`` passed through unchanged (so read as tensor order (z, y, x)), or
      ``2 / (n - 1)`` when None. Components are taken in (x, y, z) order. See the gain caveat
      on ``_spatial_jacobian_nd``. Any dimension (``torch.linalg.det`` above 3-D).
    - physical (``warp_field.is_physical`` if that attribute exists, otherwise
      ``physical_spacing is not None``): displacement in mm with components in tensor order
      (z, y, x); ``physical_spacing`` in ANTs (x, y, z) order (reversed internally), unit
      spacing when None. Central differences. The direction matrix is not used.
    - normalised (otherwise): displacement in normalised [-1, 1] coordinates with components
      in (x, y, z) order (``grid_sample`` convention); the identity grid is added and
      central differences are taken with step ``2 / (n - 1)``, or the reversed
      ``physical_spacing`` if one is given together with ``is_physical=False``.

    Parameters
    ----------
    warp_field : Tensor (B, *spatial, dim) or (B, dim, *spatial)
        Channels-first is assumed when ``shape[1]`` is 2 or 3 and ``shape[-1]`` is not, so a
        channels-first field whose last spatial size is 2 or 3 is misread as channels-last.
        A batch axis is required.
    physical_spacing : sequence of float, optional
        See the paths above. Passing it switches a plain tensor to the physical path.
    method : str, default 'central'
        'bspline' or anything else (central differences).
    **kwargs
        Ignored.

    Returns
    -------
    Tensor (B, *spatial), or (B, 1, *spatial) when the input was channels-first.

    Raises
    ------
    ValueError
        Central-difference paths with a dimension other than 2 or 3.
    """
    channels_first = False
    if warp_field.dim() >= 3 and warp_field.shape[1] in [2, 3] and warp_field.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, warp_field.dim())) + (1,)
        warp_field = warp_field.permute(perm)

    dim = warp_field.shape[-1]
    spatial = warp_field.shape[1:-1]
    device = warp_field.device
    dtype = warp_field.dtype
    
    if warp_field.dim() == dim:
        warp_field = warp_field.unsqueeze(0)

    if method == 'bspline':
        if physical_spacing is not None:
            spacings = list(physical_spacing)
        else:
            spacings = [2.0 / (size - 1) for size in spatial]
        J_disp = _spatial_jacobian_nd(warp_field, physical_spacing=spacings, method='bspline')
        F_mat = J_disp + torch.eye(dim, device=device, dtype=dtype)
        if dim == 2:
            res = F_mat[..., 0, 0] * F_mat[..., 1, 1] - F_mat[..., 0, 1] * F_mat[..., 1, 0]
        elif dim == 3:
            a, b, c = F_mat[..., 0, 0], F_mat[..., 0, 1], F_mat[..., 0, 2]
            d, e, f = F_mat[..., 1, 0], F_mat[..., 1, 1], F_mat[..., 1, 2]
            g, h, i = F_mat[..., 2, 0], F_mat[..., 2, 1], F_mat[..., 2, 2]
            res = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        else:
            res = torch.linalg.det(F_mat)
        return res.unsqueeze(1) if channels_first else res

    is_physical = getattr(warp_field, 'is_physical', physical_spacing is not None)
    
    if is_physical:
        if physical_spacing is not None:
            spacings = tuple(float(s) for s in reversed(physical_spacing))
        else:
            spacings = tuple(1.0 for _ in range(dim))
            
        grads = torch.gradient(warp_field, spacing=spacings, dim=tuple(range(1, dim + 1)))
        
        if dim == 2:
            # grads[0] is d/dy (spatial axis 1), grads[1] is d/dx (spatial axis 2)
            # In PyTorch internal tensor order: channel 0 is u_y, channel 1 is u_x
            du_x_dy = grads[0][..., 1]
            du_x_dx = grads[1][..., 1]
            du_y_dy = grads[0][..., 0]
            du_y_dx = grads[1][..., 0]

            j00 = 1.0 + du_x_dx
            j11 = 1.0 + du_y_dy
            j01 = du_x_dy
            j10 = du_y_dx
            res = j00 * j11 - j01 * j10
            return res.unsqueeze(1) if channels_first else res
        elif dim == 3:
            # grads[0]=d/dz, grads[1]=d/dy, grads[2]=d/dx
            # In PyTorch internal tensor order: channel 0 is u_z, channel 1 is u_y, channel 2 is u_x
            du_x_dz = grads[0][..., 2]
            du_x_dy = grads[1][..., 2]
            du_x_dx = grads[2][..., 2]

            du_y_dz = grads[0][..., 1]
            du_y_dy = grads[1][..., 1]
            du_y_dx = grads[2][..., 1]

            du_z_dz = grads[0][..., 0]
            du_z_dy = grads[1][..., 0]
            du_z_dx = grads[2][..., 0]

            j00 = 1.0 + du_x_dx
            j01 = du_x_dy
            j02 = du_x_dz

            j10 = du_y_dx
            j11 = 1.0 + du_y_dy
            j12 = du_y_dz

            j20 = du_z_dx
            j21 = du_z_dy
            j22 = 1.0 + du_z_dz

            res = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
            return res.unsqueeze(1) if channels_first else res
        else:
            raise ValueError("Only 2D and 3D are supported.")
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(warp_field.shape[0], *([-1] * (dim + 1)))
        
        phi = identity + warp_field
        if physical_spacing is not None:
            spacings = tuple(float(s) for s in reversed(physical_spacing))
        else:
            spacings = [2.0 / (size - 1) for size in spatial]
        grads = torch.gradient(phi, spacing=spacings, dim=list(range(1, dim + 1)))
        
        if dim == 2:
            j00 = grads[1][..., 0]
            j01 = grads[0][..., 0]
            j10 = grads[1][..., 1]
            j11 = grads[0][..., 1]
            res = j00 * j11 - j01 * j10
            return res.unsqueeze(1) if channels_first else res
        elif dim == 3:
            j00 = grads[2][..., 0]
            j01 = grads[1][..., 0]
            j02 = grads[0][..., 0]
            
            j10 = grads[2][..., 1]
            j11 = grads[1][..., 1]
            j12 = grads[0][..., 1]
            
            j20 = grads[2][..., 2]
            j21 = grads[1][..., 2]
            j22 = grads[0][..., 2]
            
            res = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
            return res.unsqueeze(1) if channels_first else res
        else:
            raise ValueError("Only 2D and 3D are supported.")


def compute_jacobian_hinge_penalty(warp_field: torch.Tensor, physical_spacing=None, epsilon: float = 0.05) -> torch.Tensor:
    """
    Differentiable fold penalty ``mean(relu(epsilon - det J) ** 2)`` over all voxels.

    ``det J`` comes from ``compute_jacobian_determinant_nd(warp_field, physical_spacing)``
    (central differences), so passing ``physical_spacing`` makes a plain tensor be read as a
    physical-mm displacement with that spacing in ANTs (x, y, z) order.

    Returns
    -------
    Tensor, scalar.
    """
    det_J = compute_jacobian_determinant_nd(warp_field, physical_spacing=physical_spacing)
    hinge = F.relu(epsilon - det_J)
    return torch.mean(hinge ** 2)



def compute_physical_jacobian_determinant(
    warp_field: torch.Tensor,
    direction: torch.Tensor = None,
    spacing: torch.Tensor = None,
    origin: torch.Tensor | None = None,
    method: str = 'central',
    **kwargs
) -> torch.Tensor:
    """
    Jacobian determinant map det(I + grad u) of a displacement field, with image geometry.

    Values <= 0 mark folding. Two paths:

    - ``warp_field.is_physical`` is True: delegates to
      ``compute_jacobian_determinant_nd(warp_field, physical_spacing=spacing, method)``
      (mm displacement, components in tensor order (z, y, x), ``spacing`` in ANTs order);
      ``direction`` is not used.
    - otherwise (plain tensors): ``u`` is a normalised-coordinate displacement with components
      in (x, y, z) order. The derivative matrix ``J`` is taken in normalised coordinates
      (step ``2 / (n - 1)``), mapped as ``M J M^-1`` with ``M = direction @ diag(spacing)``,
      and ``det(I + M J M^-1)`` is returned. Because this is a similarity transform the
      determinant equals ``det(I + J)``, so ``direction`` and ``spacing`` change the result only
      by rounding.

    Parameters
    ----------
    warp_field : Tensor (B, *spatial, dim) or (B, dim, *spatial)
        Same channels-first detection as ``compute_jacobian_determinant_nd``.
    direction : Tensor or array-like (dim, dim), optional
        Identity when None.
    spacing : Tensor or array-like (dim,), optional
        ANTs (x, y, z) order. Ones when None.
    origin : optional
        Ignored.
    method : str, default 'central'
        'bspline' uses ``_spatial_jacobian_nd(..., 'bspline')`` (see its gain caveat); anything
        else uses central differences.
    **kwargs
        Ignored.

    Returns
    -------
    Tensor (B, *spatial), or (B, 1, *spatial) when the input was channels-first. 2-D and 3-D
    determinants are written out explicitly; other dimensions use ``torch.linalg.det``.
    """
    channels_first = False
    if warp_field.dim() >= 3 and warp_field.shape[1] in [2, 3] and warp_field.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, warp_field.dim())) + (1,)
        warp_field = warp_field.permute(perm)

    dim = warp_field.shape[-1]
    if direction is None:
        direction = torch.eye(dim, device=warp_field.device, dtype=warp_field.dtype)
    if spacing is None:
        spacing = torch.ones(dim, device=warp_field.device, dtype=warp_field.dtype)

    is_physical = getattr(warp_field, 'is_physical', False)
    if is_physical:
        res = compute_jacobian_determinant_nd(warp_field, physical_spacing=spacing, method=method)
        return res.unsqueeze(1) if channels_first else res
        
    device = warp_field.device
    dtype = warp_field.dtype
    spatial = warp_field.shape[1:-1]
    
    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction, device=device, dtype=dtype)
    else:
        direction = direction.to(device=device, dtype=dtype)
        
    if not isinstance(spacing, torch.Tensor):
        spacing = torch.tensor(spacing, device=device, dtype=dtype)
    else:
        spacing = spacing.to(device=device, dtype=dtype)
        
    # 1. Compute J_voxel using spatial gradients with normalized spacing
    normalized_spacings = [2.0 / (s - 1) for s in spatial]
    if method == 'bspline':
        J_voxel = _spatial_jacobian_nd(warp_field, physical_spacing=normalized_spacings, method='bspline')
    else:
        grads = torch.gradient(warp_field, spacing=normalized_spacings, dim=list(range(1, dim + 1)))
        # Reverse gradient list to align with (x, y, [z]) component convention
        J_voxel = torch.stack(list(reversed(grads)), dim=-1)  # (B, *spatial, dim, dim)
    
    # 2. Construct voxel-to-physical matrices M and M_inv
    # M = D @ diag(S) -> column-wise scaling
    M = direction * spacing.unsqueeze(0)  # (dim, dim)
    # M_inv = diag(1/S) @ D^T -> row-wise scaling
    M_inv = direction.t() * (1.0 / spacing).unsqueeze(1)  # (dim, dim)
    
    # 3. Compute similarity transform J_phys = M @ J_voxel @ M_inv
    J_phys = torch.einsum('ij,b...jk,kl->b...il', M, J_voxel, M_inv)
    
    # 4. Compute deformation gradient F = J_phys + I
    F = J_phys + torch.eye(dim, device=device, dtype=dtype)
    
    # 5. Compute determinant of F analytically to avoid MPS batch LU decomposition deadlocks
    if dim == 2:
        a = F[..., 0, 0]
        b = F[..., 0, 1]
        c = F[..., 1, 0]
        d = F[..., 1, 1]
        jac_det_phys = a * d - b * c
    elif dim == 3:
        a = F[..., 0, 0]
        b = F[..., 0, 1]
        c = F[..., 0, 2]
        d = F[..., 1, 0]
        e = F[..., 1, 1]
        f = F[..., 1, 2]
        g = F[..., 2, 0]
        h = F[..., 2, 1]
        i = F[..., 2, 2]
        jac_det_phys = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    else:
        jac_det_phys = torch.linalg.det(F)
        
    return jac_det_phys.unsqueeze(1) if channels_first else jac_det_phys
