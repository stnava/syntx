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
        - 'bspline': fourth-order 5-tap central difference ``[1, -8, 0, 8, -1] / 12`` with
          replicate padding (exact for polynomials up to degree 4 in the interior).
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
            k_np = np.array([1/12, -8/12, 0.0, 8/12, -1/12], dtype=np.float32) / sp  # conv1d correlates
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


def compute_jacobian_determinant_nd(warp_field: torch.Tensor, physical_spacing=None, method: str = 'central',
                                    is_physical=None) -> torch.Tensor:
    """
    Jacobian determinant det(I + grad u) of a displacement field u.

    One convention for every path: the components are put in tensor order, derivatives are
    taken along the tensor axes (``method``) with tensor-order steps, and det(I + G) is
    returned with ``G[i, j] = d u_i / d axis_j``.

    - physical (``is_physical`` True): u in mm with components in tensor order (z, y, x);
      ``physical_spacing`` in ANTs (x, y, z) order (unit spacing when None).
    - normalised (``is_physical`` False): u in normalised [-1, 1] coordinates with components
      in (x, y, z) order (``grid_sample`` convention); steps ``2 / (n - 1)``, or the reversed
      ``physical_spacing`` when one is given.

    Parameters
    ----------
    warp_field : Tensor (B, *spatial, dim), (B, dim, *spatial) or (*spatial, dim)
        Channels-first is assumed when ``shape[1]`` is 2 or 3 and ``shape[-1]`` is not (a
        channels-first field whose last spatial size is 2 or 3 is therefore misread).
    physical_spacing : sequence of float, optional
    method : {'central', 'bspline'}, default 'central'
        Central differences (``torch.gradient``, one-sided at the border) or the 5-tap
        fourth-order stencil (``_spatial_jacobian_nd``). Other values raise ValueError.
    is_physical : bool, optional
        None: the field's ``is_physical`` attribute if it has one, else
        ``physical_spacing is not None``.

    Returns
    -------
    Tensor (B, *spatial) ((*spatial) for unbatched input), or (B, 1, *spatial) when the input
    was channels-first.
    """
    G, channels_first, unbatched = _displacement_gradient(warp_field, physical_spacing, method, is_physical)
    res = _det_identity_plus(G)
    if channels_first:
        return res.unsqueeze(1)
    return res[0] if unbatched else res


def _det_identity_plus(G: torch.Tensor) -> torch.Tensor:
    """det(I + G) over the last two axes (2-D / 3-D written out, else ``torch.linalg.det``)."""
    dim = G.shape[-1]
    F_mat = G + torch.eye(dim, device=G.device, dtype=G.dtype)
    if dim > 3:
        return torch.linalg.det(F_mat)
    if dim == 2:
        return F_mat[..., 0, 0] * F_mat[..., 1, 1] - F_mat[..., 0, 1] * F_mat[..., 1, 0]
    a, b, c = F_mat[..., 0, 0], F_mat[..., 0, 1], F_mat[..., 0, 2]
    d, e, f = F_mat[..., 1, 0], F_mat[..., 1, 1], F_mat[..., 1, 2]
    g, h, k = F_mat[..., 2, 0], F_mat[..., 2, 1], F_mat[..., 2, 2]
    return a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)


def _displacement_gradient(warp_field, physical_spacing, method, is_physical):
    """(G, channels_first, unbatched): G (B, *spatial, dim, dim) = d u_i / d axis_j with
    components and axes in tensor order (see ``compute_jacobian_determinant_nd``)."""
    if method not in ('central', 'bspline'):
        raise ValueError(f"method must be 'central' or 'bspline', got {method!r}")
    channels_first = False
    if warp_field.dim() >= 3 and warp_field.shape[1] in [2, 3] and warp_field.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, warp_field.dim())) + (1,)
        warp_field = warp_field.permute(perm)

    dim = warp_field.shape[-1]
    if dim < 2 or (method == 'bspline' and dim not in (2, 3)):
        raise ValueError(f"unsupported dimension {dim} for method {method!r} "
                         "('bspline': 2-D / 3-D; 'central': any dim >= 2)")
    unbatched = warp_field.dim() == dim + 1
    if unbatched:
        warp_field = warp_field.unsqueeze(0)
    spatial = warp_field.shape[1:-1]
    device, dtype = warp_field.device, warp_field.dtype

    if is_physical is None:
        is_physical = getattr(warp_field, 'is_physical', physical_spacing is not None)
    comps = warp_field if is_physical else torch.flip(warp_field, dims=[-1])   # -> tensor order
    if physical_spacing is not None:
        steps = [float(sp) for sp in reversed(tuple(physical_spacing))]
    elif is_physical:
        steps = [1.0] * dim
    else:
        steps = [2.0 / (n - 1) for n in spatial]

    if method == 'bspline':
        # _spatial_jacobian_nd returns derivative axes in (x, y, z) order: flip to tensor order
        G = torch.flip(_spatial_jacobian_nd(comps, physical_spacing=steps, method='bspline'), dims=[-1])
    else:
        grads = torch.gradient(comps, spacing=steps, dim=tuple(range(1, dim + 1)))
        G = torch.stack(grads, dim=-1)                     # (..., comp i, axis j), tensor order
    return G, channels_first, unbatched


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
    direction=None,
    spacing=None,
    method: str = 'central',
) -> torch.Tensor:
    """
    Jacobian determinant map det(I + du/dx) of a displacement field, with image geometry.
    Values <= 0 mark folding.

    - ``warp_field.is_physical`` True: u in mm (world components, tensor order (z, y, x)).
      With p = origin + D S i, du/dp = (du/di S^-1) D^-1, so det(I + G D^-1) is returned, G
      from ``compute_jacobian_determinant_nd``'s physical convention (``spacing`` ANTs order)
      and D the direction in tensor order -- correct for oblique images.
    - otherwise: u in normalised [-1, 1] coordinates with (x, y, z) components;
      det(I + du/dn) in normalised coordinates. ``spacing`` / ``direction`` do not change this
      determinant (a similarity transform) and are not used.

    Parameters
    ----------
    warp_field : Tensor (B, *spatial, dim) or (B, dim, *spatial)
        Same channels-first detection as ``compute_jacobian_determinant_nd``.
    direction : Tensor or array-like (dim, dim), optional
        ANTs order; identity when None.
    spacing : Tensor or array-like (dim,), optional
        ANTs (x, y, z) order; ones when None.
    method : {'central', 'bspline'}, default 'central'

    Returns
    -------
    Tensor (B, *spatial), or (B, 1, *spatial) when the input was channels-first.
    """
    if not getattr(warp_field, 'is_physical', False):
        return compute_jacobian_determinant_nd(warp_field, method=method, is_physical=False)
    dim = warp_field.shape[1] if (warp_field.dim() >= 3 and warp_field.shape[1] in [2, 3]
                                  and warp_field.shape[-1] not in [2, 3]) else warp_field.shape[-1]
    sp = [1.0] * dim if spacing is None else [float(x) for x in (spacing.tolist() if isinstance(spacing, torch.Tensor) else spacing)]
    G, channels_first, unbatched = _displacement_gradient(warp_field, sp, method, True)
    D = np.eye(dim) if direction is None else np.asarray(
        direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else direction, dtype=float)
    D_rev_inv = torch.as_tensor(np.linalg.inv(D[::-1, ::-1].copy()), device=G.device, dtype=G.dtype)
    res = _det_identity_plus(G @ D_rev_inv)
    if channels_first:
        return res.unsqueeze(1)
    return res[0] if unbatched else res
