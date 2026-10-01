"""
Linear-transform pieces shared by SyN / TVF / SyNGS: Lie-algebra rotation matrices, the
learnable ``HierarchicalAffine`` module, reading ANTs affine transforms into physical (M, t),
and evaluating an initial transform list as a normalised sampling grid. The grid <-> physical
affine helpers re-exported here live in ``syntx.spatial``.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import ants


def get_rotation_matrix(omega: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Rotation matrix from a rotation vector (exponential map of so(2) / so(3)).

    2-D: ``[[cos w, -sin w], [sin w, cos w]]`` with ``w = omega[0]`` (radians). 3-D: Rodrigues'
    formula ``I + sin(t) K + (1 - cos(t)) K^2`` with ``t = |omega|`` and ``K`` the skew matrix
    of ``omega / t``; when ``|omega|^2 < 1e-16`` the first-order form ``I + skew(omega)`` is
    used instead, which avoids the division by zero and keeps a gradient at identity.

    Parameters
    ----------
    omega : Tensor
        Rotation parameters: element 0 used in 2-D; 3 elements (axis * angle) in 3-D.
    dim : int
        2 or 3.

    Returns
    -------
    Tensor (dim, dim), same device / dtype as ``omega``.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3.
    """
    device = omega.device
    dtype = omega.dtype
    if dim == 2:
        theta = omega[0]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t])
        ])
    elif dim == 3:
        theta2 = torch.sum(omega**2)
        is_zero = theta2 < 1e-16
        safe_theta2 = torch.where(is_zero, 1e-16, theta2)
        theta = torch.sqrt(safe_theta2)
        
        safe_theta = torch.where(is_zero, 1.0, theta)
        omega_norm = omega / safe_theta
        
        K_raw = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega[2], omega[1]]),
            torch.stack([omega[2], torch.tensor(0.0, device=device, dtype=dtype), -omega[0]]),
            torch.stack([-omega[1], omega[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[2], omega_norm[1]]),
            torch.stack([omega_norm[2], torch.tensor(0.0, device=device, dtype=dtype), -omega_norm[0]]),
            torch.stack([-omega_norm[1], omega_norm[0], torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)
        R_small = I + K_raw
        return torch.where(is_zero, R_small, R)
    else:
        raise ValueError("Only 2D and 3D are supported.")


class HierarchicalAffine(nn.Module):
    """
    Learnable linear transform, initialised to the identity, as a (dim+1, dim+1) matrix.

    The matrix is ``[[A, translation], [0, 1]] @ T_init`` with

    - 'Affine': ``A = R(omega) @ diag(anisotropic_scale * scale) @ Sh``, ``Sh`` unit upper
      triangular with ``shear`` above the diagonal;
    - any other ``transform_type``: ``A = R(omega) * scale``.

    Which entries are trainable ``nn.Parameter`` and which are fixed buffers:

    - ``translation`` (dim,) and ``omega`` (dim*(dim-1)/2,): always parameters, so
      'Translation' also has a trainable rotation; 'Translation' and 'Rigid' build the same
      module.
    - ``scale`` (1,): parameter for 'Similarity' and 'Affine', else a buffer of 1.
    - ``anisotropic_scale`` (dim,) and ``shear`` (dim*(dim-1)/2,): parameters for 'Affine',
      else buffers of 1 / 0.
    - ``T_init``: buffer, None at construction; callers (``syntx.tvf``, ``syntx.syngs``) set
      it to an initial transform in normalised grid coordinates. It is applied first.

    The callers use the matrix in normalised ``grid_sample`` coordinates (x, y, z order).

    Parameters
    ----------
    dim : int, default 3
        2 or 3 (``get_rotation_matrix`` raises otherwise).
    transform_type : str, default 'Affine'
        'Translation', 'Rigid', 'Similarity' or 'Affine'; not validated.
    """

    def __init__(self, dim: int = 3, transform_type: str = 'Affine'):
        super().__init__()
        self.dim = dim
        self.type = transform_type
        
        # Translation
        self.translation = nn.Parameter(torch.zeros(dim))
        
        # Rotation (Lie Algebra SO(d))
        num_rot = dim * (dim - 1) // 2
        self.omega = nn.Parameter(torch.zeros(num_rot))
        
        # Scale (Similarity)
        if transform_type in ['Similarity', 'Affine']:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
            
        # Shear/Anisotropic Scale
        if transform_type == 'Affine':
            self.anisotropic_scale = nn.Parameter(torch.ones(dim))
            self.shear = nn.Parameter(torch.zeros(num_rot))
        else:
            self.register_buffer('anisotropic_scale', torch.ones(dim))
            self.register_buffer('shear', torch.zeros(num_rot))
            
        self.register_buffer('T_init', None)

    def clamp_parameters(self):
        """Clamp, in place, trainable scale / anisotropic_scale to [0.05, 20], shear to [-5, 5]
        and omega to [-pi, pi]."""
        with torch.no_grad():
            if isinstance(self.scale, nn.Parameter):
                self.scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.anisotropic_scale, nn.Parameter):
                self.anisotropic_scale.clamp_(min=0.05, max=20.0)
            if isinstance(self.shear, nn.Parameter):
                self.shear.clamp_(min=-5.0, max=5.0)
            if isinstance(self.omega, nn.Parameter):
                self.omega.clamp_(min=-3.14159265, max=3.14159265)

    def get_matrix(self):
        """Homogeneous matrix (dim+1, dim+1): ``[[A, translation], [0, 1]]``, right-multiplied
        by ``T_init`` when it is set."""
        R = get_rotation_matrix(self.omega, self.dim)
        
        if self.type == 'Affine':
            S = torch.diag(self.anisotropic_scale * self.scale)
            Sh = torch.eye(self.dim, device=self.shear.device, dtype=self.shear.dtype)
            triu_indices = torch.triu_indices(self.dim, self.dim, offset=1)
            Sh[triu_indices[0], triu_indices[1]] = self.shear
            A = R @ S @ Sh
        else:
            A = R * self.scale
            
        T = torch.eye(self.dim + 1, device=self.translation.device, dtype=self.translation.dtype)
        T[:self.dim, :self.dim] = A
        T[:self.dim, self.dim] = self.translation
        
        if hasattr(self, 'T_init') and self.T_init is not None:
            return T @ self.T_init.to(device=T.device, dtype=T.dtype)
        return T

    def get_affine_grid_matrix(self):
        """Top ``dim`` rows of ``get_matrix()``, shape (dim, dim+1) (the ``F.affine_grid``
        theta layout without the batch axis)."""
        T = self.get_matrix()
        return T[:self.dim, :self.dim + 1]


from ..spatial import (
    _grid_to_physical_affine_torch_yfirst,
    grid_to_physical_affine_torch,
    grid_to_physical_affine,
    physical_to_grid_affine,
    export_ants_affine_transform,
    create_ants_affine,
)



def parse_ants_affine(tx_list, dim):
    """
    Read ANTs / ITK linear transforms into one physical map ``y = M_phys @ x + t_phys``.

    Each item's ITK parameters are used directly when their count is ``dim*dim + dim``
    (matrix row-major, then translation) or ``dim`` (translation only); the centre ``C`` in
    ``fixed_parameters`` is folded in as ``t + C - M @ C``. Otherwise, for an existing file
    path only, ``M`` and ``t`` are recovered by mapping the origin and unit points with
    ``ants.apply_transforms_to_points``. Anything else (unreadable paths, other objects, other
    parameter counts such as Euler / similarity transforms passed as objects) is skipped
    without a warning.

    Items are composed in list order: the first item is applied to the point first. (This is
    the reverse of the ``ants.apply_transforms`` transformlist convention, where the last
    entry is applied first; it only matters for lists of more than one transform.)

    Parameters
    ----------
    tx_list : str, ANTsTransform, or list / tuple of them
        A single item is wrapped in a list.
    dim : int
        2 or 3.

    Returns
    -------
    (M_phys, t_phys)
        float32 tensors (dim, dim) and (dim,) in ANTs physical (x, y, z) coordinates, or
        ``(None, None)`` when the list is empty or nothing could be parsed.
    """
    import ants
    
    if not isinstance(tx_list, (list, tuple)):
        tx_list = [tx_list]
    if len(tx_list) == 0:
        return None, None
    M_composed = np.eye(dim, dtype=np.float32)
    t_composed = np.zeros(dim, dtype=np.float32)
    parsed_any = False

    for tx_item in tx_list:
        tx = None
        try:
            if hasattr(tx_item, 'parameters') and hasattr(tx_item, 'fixed_parameters'):
                tx = tx_item
            elif isinstance(tx_item, str):
                try:
                    tx = ants.read_transform(tx_item)
                except Exception:
                    continue
        except Exception:
            continue

        if tx is None:
            continue

        params = None
        fixed_params = None
        try:
            params = tx.parameters
            fixed_params = tx.fixed_parameters
        except Exception:
            params = None

        if params is not None and len(params) == 12 and dim == 3:
            M = np.array(params[:9], dtype=np.float32).reshape(3, 3)
            t = np.array(params[9:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 3 else np.zeros(3, dtype=np.float32)
        elif params is not None and len(params) == 6 and dim == 2:
            M = np.array(params[:4], dtype=np.float32).reshape(2, 2)
            t = np.array(params[4:], dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == 2 else np.zeros(2, dtype=np.float32)
        elif params is not None and len(params) == dim:  # TranslationTransform (2D: 2, 3D: 3)
            M = np.eye(dim, dtype=np.float32)
            t = np.array(params, dtype=np.float32)
            C = np.array(fixed_params, dtype=np.float32) if len(fixed_params) == dim else np.zeros(dim, dtype=np.float32)
        elif isinstance(tx_item, str) and os.path.exists(tx_item):
            # Robust fallback: extract linear mapping via point transformations
            try:
                import pandas as pd
                if dim == 3:
                    pts = pd.DataFrame({'x': [0.0, 1.0, 0.0, 0.0], 'y': [0.0, 0.0, 1.0, 0.0], 'z': [0.0, 0.0, 0.0, 1.0]})
                    w_pts = ants.apply_transforms_to_points(dim=3, points=pts, transformlist=[tx_item])
                    p0 = np.array(w_pts.iloc[0])
                    p1 = np.array(w_pts.iloc[1]) - p0
                    p2 = np.array(w_pts.iloc[2]) - p0
                    p3 = np.array(w_pts.iloc[3]) - p0
                    M = np.column_stack([p1, p2, p3]).astype(np.float32)
                    t = p0.astype(np.float32)
                    C = np.zeros(3, dtype=np.float32)
                else:
                    pts = pd.DataFrame({'x': [0.0, 1.0, 0.0], 'y': [0.0, 0.0, 1.0]})
                    w_pts = ants.apply_transforms_to_points(dim=2, points=pts, transformlist=[tx_item])
                    p0 = np.array(w_pts.iloc[0])
                    p1 = np.array(w_pts.iloc[1]) - p0
                    p2 = np.array(w_pts.iloc[2]) - p0
                    M = np.column_stack([p1, p2]).astype(np.float32)
                    t = p0.astype(np.float32)
                    C = np.zeros(2, dtype=np.float32)
            except Exception:
                continue
        else:
            continue

        t_new = t + C - M @ C
        t_composed = M @ t_composed + t_new
        M_composed = M @ M_composed
        parsed_any = True

    if not parsed_any:
        return None, None

    M_phys = torch.from_numpy(M_composed).to(torch.float32)
    t_phys = torch.from_numpy(t_composed).to(torch.float32)
    return M_phys, t_phys


def compute_initial_grid(fixed, moving, tx_list):
    """
    Initial transform evaluated at every fixed voxel, as normalised moving-image coordinates.

    ``ants.apply_transforms(..., compose=...)`` turns ``tx_list`` into one displacement field
    on the fixed grid (written to and read back from a temporary directory, removed
    afterwards). Each fixed voxel's physical point plus that displacement is converted to a
    moving continuous index and then to [-1, 1] (``align_corners=True``). Points that map
    outside the moving image get coordinates outside [-1, 1] rather than being clamped.

    Parameters
    ----------
    fixed, moving : ANTsImage
        Fixed and moving images (only their geometry is used for the conversion).
    tx_list : list
        Transform list in ``ants.apply_transforms`` order.

    Returns
    -------
    ndarray float32 (1, *spatial, dim)
        Spatial axes in tensor order (z, y, x) of the fixed grid; the last axis holds moving
        coordinates in (x, y, z) order, ready for ``F.grid_sample``. For ``dim`` other than
        2 / 3 the spatial axes are left in ANTs order.
    """
    import os
    import shutil
    import tempfile
    import ants
    dim = moving.dimension

    tmpdir = tempfile.mkdtemp(prefix="syntx_initgrid_")
    try:
        comp_path = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=tx_list,
            compose=os.path.join(tmpdir, "comp"))
        disp = ants.image_read(comp_path).numpy().astype(np.float64)  # (*fixed.shape, dim)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Fixed-grid physical points (ITK xyz order) and their images in moving space.
    f_dir = np.array(fixed.direction, dtype=np.float64)
    f_sp = np.array(fixed.spacing, dtype=np.float64)
    f_org = np.array(fixed.origin, dtype=np.float64)
    idxs = np.stack(np.meshgrid(*[np.arange(s) for s in fixed.shape], indexing='ij'), axis=-1)
    x_phys = (idxs.reshape(-1, dim) * f_sp) @ f_dir.T + f_org
    y_phys = x_phys + disp.reshape(-1, dim)

    # Moving physical -> moving continuous index -> normalized [-1, 1] (align_corners=True).
    m_dir = np.array(moving.direction, dtype=np.float64)
    m_sp = np.array(moving.spacing, dtype=np.float64)
    m_org = np.array(moving.origin, dtype=np.float64)
    voxel_idx = ((y_phys - m_org) @ np.linalg.inv(m_dir).T) / m_sp
    n_minus_1 = np.maximum(np.array(moving.shape, dtype=np.float64) - 1.0, 1.0)
    normalized = voxel_idx / n_minus_1 * 2.0 - 1.0

    grid = normalized.reshape(tuple(fixed.shape) + (dim,))
    if dim == 2:
        grid = np.transpose(grid, (1, 0, 2))
    elif dim == 3:
        grid = np.transpose(grid, (2, 1, 0, 3))
    return np.expand_dims(grid.astype(np.float32), axis=0)
