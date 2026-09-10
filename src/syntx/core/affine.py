import os
import math
import numpy as np
import torch
import torch.nn as nn
import ants


def get_rotation_matrix(omega: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes a 2D or 3D rotation matrix from a Lie Algebra parameterization ($so(2)$ or $so(3)$).

    Uses a first-order Taylor expansion near $\\omega = 0$ to prevent zero-angle gradient locking
    and division-by-zero singularities during automatic differentiation (GEMINI.md Rule 6).

    Parameters
    ----------
    omega : torch.Tensor
        Lie algebra rotation vector (1 element for 2D angle; 3 elements `[w0, w1, w2]` for 3D axis-angle).
    dim : int
        Spatial dimensionality (2 or 3).

    Returns
    -------
    torch.Tensor
        Rotation matrix $R \\in SO(d)$ of shape `(2, 2)` or `(3, 3)`.

    Raises
    ------
    ValueError
        If `dim` is not 2 or 3.
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
    Hierarchical Differentiable Linear Transformation Module in PyTorch.

    Parameterizes physical linear transformations using Lie Algebra $SO(d)$ rotation representation
    to eliminate gimbal lock and maintain continuous gradient flow at identity initialization.

    Supported Transformation Hierarchy (`transform_type`):
    - `'Translation'`: $d$-dimensional physical shift vector.
    - `'Rigid'`: Translation + $SO(d)$ Lie algebra rotation.
    - `'Similarity'`: Rigid + isotropic scaling factor $s$.
    - `'Affine'`: Similarity + anisotropic scaling $S$ + upper-triangular shear matrix $Sh$.

    Parameters
    ----------
    dim : int, default=3
        Spatial dimensionality (2 or 3).
    transform_type : str, default='Affine'
        Linear transformation model ('Translation', 'Rigid', 'Similarity', 'Affine').

    Attributes
    ----------
    translation : nn.Parameter
        Translation parameter vector of shape `(dim,)`.
    omega : nn.Parameter
        Lie algebra rotation vector of shape `(dim*(dim-1)//2,)`.
    scale : nn.Parameter or torch.Tensor
        Isotropic scaling factor.
    anisotropic_scale : nn.Parameter or torch.Tensor
        Per-axis scaling factor vector of shape `(dim,)`.
    shear : nn.Parameter or torch.Tensor
        Upper-triangular shear parameter vector.
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
    Parses a single ANTs affine transform (path string or ANTsTransform) into M_phys and t_phys tensors.
    Takes into account the center of rotation C as per rule:
    t_new = t + C - M @ C
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
    Computes an initial_grid (representing the mapping from fixed space to moving space
    under the initial transform) using coordinate warping.
    """
    import ants
    dim = moving.dimension
    
    # 1. Get moving physical coordinates via numpy meshgrid
    shape = moving.shape
    grids = [np.arange(s) for s in shape]
    meshgrid_idxs = np.meshgrid(*grids, indexing='ij')
    idxs = np.stack(meshgrid_idxs, axis=-1)
    
    direction = np.array(moving.direction)
    spacing = np.array(moving.spacing)
    origin = np.array(moving.origin)
    
    idxs_flat = idxs.reshape(-1, dim)
    scaled_idxs = idxs_flat * spacing
    phys_flat = (direction @ scaled_idxs.T).T + origin
    coord_np = phys_flat.reshape(shape + (dim,)).astype(np.float32)
    
    # 2. Warp each coordinate component image to the fixed space
    warped_coords = []
    for d in range(dim):
        c_img = ants.from_numpy(coord_np[..., d], origin=moving.origin, spacing=moving.spacing, direction=moving.direction)
        w_c_img = ants.apply_transforms(fixed=fixed, moving=c_img, transformlist=tx_list)
        warped_coords.append(w_c_img.numpy())
        
    moving_phys_at_fixed = np.stack(warped_coords, axis=-1)
    
    # 3. Map physical coordinates to voxel indices in moving space
    shape = moving_phys_at_fixed.shape
    phys_flat = moving_phys_at_fixed.reshape(-1, dim)
    
    direction_inv = np.linalg.inv(direction)
    diff = phys_flat - origin
    sp_idx = diff @ direction_inv.T
    voxel_idx = sp_idx / spacing
    
    # 4. Normalize voxel indices to [-1, 1] for grid_sample (x, y, [z]) convention
    normalized_coords = []
    for d in range(dim):
        N = moving.shape[d]
        norm_d = (voxel_idx[:, d] / (N - 1)) * 2.0 - 1.0
        normalized_coords.append(norm_d)
        
    normalized_grid_flat = np.stack(normalized_coords, axis=-1)
    
    grid = normalized_grid_flat.reshape(fixed.shape + (dim,))
    if dim == 2:
        grid = np.transpose(grid, (1, 0, 2))
    elif dim == 3:
        grid = np.transpose(grid, (2, 1, 0, 3))
    initial_grid = np.expand_dims(grid.astype(np.float32), axis=0)
    return initial_grid
