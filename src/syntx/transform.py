"""
transform.py — Coordinate System Bridging & ANTs Physical Transform Exporters
==============================================================================

This module provides object containers (`SyNToTransform`) and export utilities bridging PyTorch
native normalized coordinate grids `[-1, 1]` to ITK/ANTs physical coordinate spaces (LPS mm).

Key Features & Rule Compliance
------------------------------
- PyTorch ZYX -> ITK XYZ Array Order Parity (GEMINI.md Rule 3): Vector component channels and
  spatial array dimensions are systematically mapped to preserve exact physical alignment.
- Single Interpolation Policy: Composes Affine and SyN displacement fields into a single unified step.
- Physical Space Awareness: Converts normalized displacement vectors into absolute physical mm shifts.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import ants


class SyNToTransform:
    """
    Object container bridging PyTorch native normalized matrices to ITK physical formats.

    Encapsulates PyTorch normalized affine grids, displacement fields, and image metadata
    (origin, spacing, direction matrix), providing native GPU resampling, Jacobian determinant
    calculation, and composite NIfTI warp file exports.

    Parameters
    ----------
    affine_grid : torch.Tensor or np.ndarray
        Normalized affine coordinate grid of shape `(1, *spatial, dim)` in `[-1, 1]`.
    warp_field : torch.Tensor or np.ndarray
        Displacement field of shape `(1, *spatial, dim)`.
    metadata : dict
        Image spatial metadata containing `'origin'`, `'spacing'`, `'direction'`, and optionally `'shape'`.
    device : str or torch.device, default='cpu'
        Compute device ('cpu', 'cuda', 'mps').
    T_grid : torch.Tensor or np.ndarray, optional
        Physical affine matrix representation.
    is_physical : bool, default=False
        If True, indicates `warp_field` is already in physical mm units.

    Attributes
    ----------
    dim : int
        Spatial dimensionality (2 or 3).
    spatial : tuple of int
        Spatial shape of the warp field tensor.
    target_shape : tuple of int
        Target resampling shape (from metadata or spatial).
    """

    def __init__(self, affine_grid, warp_field, metadata: dict, device='cpu', T_grid=None, is_physical=False):
        if not isinstance(affine_grid, torch.Tensor):
            if hasattr(affine_grid, 'numpy'):
                affine_grid = torch.from_numpy(np.array(affine_grid))
            else:
                affine_grid = torch.from_numpy(np.asarray(affine_grid))
        if not isinstance(warp_field, torch.Tensor):
            if hasattr(warp_field, 'numpy'):
                warp_field = torch.from_numpy(np.array(warp_field))
            else:
                warp_field = torch.from_numpy(np.asarray(warp_field))

        self.affine_grid = affine_grid
        self.metadata = metadata
        self.device = device
        self.dim = warp_field.shape[-1]
        self.spatial = warp_field.shape[1:-1]
        self.target_shape = tuple(metadata['shape']) if 'shape' in metadata else self.spatial
        self.T_grid = T_grid

        is_physical = is_physical or getattr(warp_field, 'is_physical', False)
        if not is_physical:
            # Convert normalized coordinate field to physical mm coordinates
            spatial_shape_t = torch.tensor(list(reversed(self.spatial)), dtype=torch.float32, device=device)
            voxel_disp = warp_field * (spatial_shape_t - 1) / 2.0

            direction = torch.tensor(metadata['direction'], dtype=torch.float32, device=device)
            spacing = torch.tensor(metadata['spacing'], dtype=torch.float32, device=device)

            # voxel to physical spacing
            phys_disp = voxel_disp * spacing
            # physical rotation
            phys_disp_flat = phys_disp.reshape(-1, self.dim)
            phys_disp_flat = phys_disp_flat @ direction.t()
            phys_disp = phys_disp_flat.reshape(warp_field.shape)

            self.warp_field = phys_disp
            self.warp_field.is_physical = True
        else:
            self.warp_field = warp_field
            self.warp_field.is_physical = True

    def to(self, device):
        """
        Moves internal transformation tensors to the specified compute device.

        Parameters
        ----------
        device : str or torch.device
            Target PyTorch device ('cpu', 'cuda', 'mps').

        Returns
        -------
        SyNToTransform
            Self reference with updated device tensors.
        """
        self.device = device
        self.affine_grid = self.affine_grid.to(device)
        self.warp_field = self.warp_field.to(device)
        if self.T_grid is not None:
            self.T_grid = self.T_grid.to(device)
        return self

    def apply(self, image_tensor: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        """
        Applies the composite transformation directly to an image tensor on GPU/CPU.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image tensor of shape `(1, 1, *spatial)`.
        mode : str, default='bilinear'
            Interpolation mode ('bilinear', 'nearest').

        Returns
        -------
        torch.Tensor
            Resampled / warped image tensor of shape `(1, 1, *target_shape)`.
        """
        from .syn import compose_grids, get_physical_grid_torch, physical_to_normalized_torch, grid_to_physical_affine_torch

        device = self.device
        dtype = self.warp_field.dtype
        dim = self.dim

        spacing = tuple(reversed(self.metadata['spacing']))
        origin = tuple(reversed(self.metadata['origin']))
        direction = self.metadata['direction'][::-1, ::-1].copy()

        X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)

        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field

        phi_l2r_phys = X_phys + warp_resampled

        if self.T_grid is not None:
            moving_shape = image_tensor.shape[2:]
            moving_spacing = spacing
            moving_origin = origin
            moving_direction = direction

            M_phys, t_phys = grid_to_physical_affine_torch(
                self.T_grid, self.target_shape, spacing, origin, direction,
                moving_shape, moving_spacing, moving_origin, moving_direction
            )
            y_phys = phi_l2r_phys @ M_phys.t() + t_phys
            composed_grid = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
        else:
            if self.target_shape != self.spatial:
                affine_resampled = F.interpolate(
                    torch.movedim(self.affine_grid, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                affine_resampled = self.affine_grid
            phi_l2r_norm = physical_to_normalized_torch(phi_l2r_phys, self.target_shape, spacing, origin, direction)
            composed_grid = compose_grids(affine_resampled, phi_l2r_norm)

        return F.grid_sample(image_tensor, composed_grid, mode=mode, padding_mode='border', align_corners=True)

    def get_jacobian_determinant(self) -> np.ndarray:
        """
        Computes the Jacobian determinant map of the total composite deformation natively in PyTorch.

        Returns
        -------
        np.ndarray
            NumPy array of shape `(*spatial)` containing physical Jacobian determinants $\\det(J(x))$.
        """
        from .syn import compute_physical_jacobian_determinant, compose_grids, get_physical_grid_torch, physical_to_normalized_torch

        device = self.device
        dtype = self.warp_field.dtype
        dim = self.dim

        spacing = tuple(reversed(self.metadata['spacing']))
        origin = tuple(reversed(self.metadata['origin']))
        direction = np.array(self.metadata['direction'])[::-1, ::-1].copy()

        X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)

        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field

        phi_l2r_phys = X_phys + warp_resampled

        if self.T_grid is not None:
            moving_shape = self.target_shape
            moving_spacing = spacing
            moving_origin = origin
            moving_direction = direction

            from .syn import grid_to_physical_affine_torch
            M_phys, t_phys = grid_to_physical_affine_torch(
                self.T_grid, self.target_shape, spacing, origin, direction,
                moving_shape, moving_spacing, moving_origin, moving_direction
            )
            y_phys = phi_l2r_phys @ M_phys.t() + t_phys
            composed_grid = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
        else:
            if self.target_shape != self.spatial:
                affine_resampled = F.interpolate(
                    torch.movedim(self.affine_grid, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                affine_resampled = self.affine_grid
            phi_l2r_norm = physical_to_normalized_torch(phi_l2r_phys, self.target_shape, spacing, origin, direction)
            composed_grid = compose_grids(affine_resampled, phi_l2r_norm)

        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

        total_normalized_disp = composed_grid - identity
        return compute_physical_jacobian_determinant(
            total_normalized_disp,
            direction=self.metadata['direction'],
            spacing=self.metadata['spacing']
        ).squeeze(0).detach().cpu().numpy()

    def _to_physical_displacement(self, disp: torch.Tensor, is_physical: bool = False) -> ants.ANTsImage:
        """Helper to convert displacement tensor to ANTsImage with correct vector channel order."""
        if is_physical:
            phys_disp = disp.squeeze(0).detach().cpu().numpy()
        else:
            spatial_shape = torch.tensor(list(reversed(self.target_shape)), dtype=torch.float32, device=self.device)
            voxel_disp = disp * (spatial_shape - 1) / 2.0

            direction = np.array(self.metadata['direction'])
            spacing = np.array(self.metadata['spacing'])

            phys_disp = voxel_disp.squeeze(0).detach().cpu().numpy() * spacing
            phys_disp_flat = phys_disp.reshape(-1, self.dim)
            phys_disp_flat = phys_disp_flat @ direction.T
            phys_disp = phys_disp_flat.reshape(tuple(self.target_shape) + (self.dim,))

        if self.dim == 2:
            phys_disp = phys_disp[..., [1, 0]]
        elif self.dim == 3:
            phys_disp = phys_disp[..., [2, 1, 0]]

        return ants.from_numpy(
            phys_disp,
            origin=self.metadata['origin'],
            spacing=self.metadata['spacing'],
            direction=self.metadata['direction'],
            has_components=True
        )

    def to_composite_warp(self, filename: str) -> str:
        """
        Exports combined Affine + SyN transformation fields into a single ITK CompositeWarp NIfTI file.

        Parameters
        ----------
        filename : str
            Target output file path (`CompositeWarp.nii.gz`).

        Returns
        -------
        str
            Absolute file path of written NIfTI file.
        """
        from .syn import compose_grids, get_physical_grid_torch, physical_to_normalized_torch

        device = self.device
        dtype = self.warp_field.dtype
        dim = self.dim

        spacing = tuple(reversed(self.metadata['spacing']))
        origin = tuple(reversed(self.metadata['origin']))
        direction = np.array(self.metadata['direction'])[::-1, ::-1].copy()

        X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)

        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field

        phi_l2r_phys = X_phys + warp_resampled

        if self.T_grid is not None:
            moving_shape = self.target_shape
            moving_spacing = spacing
            moving_origin = origin
            moving_direction = direction

            from .syn import grid_to_physical_affine_torch
            M_phys, t_phys = grid_to_physical_affine_torch(
                self.T_grid, self.target_shape, spacing, origin, direction,
                moving_shape, moving_spacing, moving_origin, moving_direction
            )
            y_phys = phi_l2r_phys @ M_phys.t() + t_phys
            composed_grid = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
        else:
            if self.target_shape != self.spatial:
                affine_resampled = F.interpolate(
                    torch.movedim(self.affine_grid, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                affine_resampled = self.affine_grid
            phi_l2r_norm = physical_to_normalized_torch(phi_l2r_phys, self.target_shape, spacing, origin, direction)
            composed_grid = compose_grids(affine_resampled, phi_l2r_norm)

        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

        total_normalized_disp = composed_grid - identity
        ants_disp = self._to_physical_displacement(total_normalized_disp, is_physical=False)

        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        ants.image_write(ants_disp, filename)
        return filename

    def export_classic(self, prefix: str) -> list:
        """
        Exports transformations separated into physical 0AffineWarp and 1SyNWarp NIfTI displacement fields.

        Parameters
        ----------
        prefix : str
            Filename prefix for saved NIfTI files.

        Returns
        -------
        list of str
            File paths `[1SyNWarp.nii.gz, 0AffineWarp.nii.gz]`.
        """
        from .syn import compose_grids, get_physical_grid_torch, physical_to_normalized_torch

        device = self.device
        dtype = self.warp_field.dtype
        dim = self.dim

        spacing = tuple(reversed(self.metadata['spacing']))
        origin = tuple(reversed(self.metadata['origin']))
        direction = np.array(self.metadata['direction'])[::-1, ::-1].copy()

        X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)

        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)

            affine_resampled = F.interpolate(
                torch.movedim(self.affine_grid, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field
            affine_resampled = self.affine_grid

        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

        affine_disp = affine_resampled - identity
        ants_affine = self._to_physical_displacement(affine_disp, is_physical=False)
        ants_warp = self._to_physical_displacement(warp_resampled, is_physical=True)

        ants.image_write(ants_affine, f"{prefix}0AffineWarp.nii.gz")
        ants.image_write(ants_warp, f"{prefix}1SyNWarp.nii.gz")

        return [f"{prefix}1SyNWarp.nii.gz", f"{prefix}0AffineWarp.nii.gz"]


def export_ants_displacement_field(disp_np: np.ndarray, origin, spacing, direction) -> ants.ANTsImage:
    """
    Standardized conversion of PyTorch/JAX physical displacement arrays into ITK-compatible ANTsImage displacement fields.

    Parameters
    ----------
    disp_np : np.ndarray
        Array of shape `(1, *spatial, dim)` or `(*spatial, dim)` containing ZYX physical displacement vectors.
    origin : tuple or list
        Image origin in XYZ order.
    spacing : tuple or list
        Voxel spacing in XYZ order.
    direction : np.ndarray or list of list
        Direction matrix in XYZ order.

    Returns
    -------
    ants.ANTsImage
        ANTs vector image with `has_components=True`.
    """
    while disp_np.ndim > 3 and disp_np.shape[0] == 1:
        disp_np = disp_np[0]

    # Reverse vector components from PyTorch ZYX order [v_z, v_y, v_x] to ITK XYZ order [v_x, v_y, v_z]
    disp_xyz = np.ascontiguousarray(disp_np[..., ::-1].copy())

    return ants.from_numpy(
        disp_xyz,
        origin=origin,
        spacing=spacing,
        direction=direction,
        has_components=True
    )


def export_ants_affine_transform(M_phys, t_phys, dim: int, filename: str = None):
    """
    Standardized export of physical affine parameters `(M_phys, t_phys)` into ITK-compatible ANTs transforms.

    Guarantees exact ITK parameter layout (`M_phys.ravel()` for forward, `M_phys_inv.T.ravel()` for inverse).

    Parameters
    ----------
    M_phys : np.ndarray or torch.Tensor
        Physical rotation/scale/shear matrix (`2x2` or `3x3`).
    t_phys : np.ndarray or torch.Tensor
        Physical translation vector.
    dim : int
        Spatial dimensionality (2 or 3).
    filename : str, optional
        File path to write forward transform matrix file.

    Returns
    -------
    tx_fwd : ants.ANTsTransform
        Forward ANTs transform object.
    tx_inv : ants.ANTsTransform
        Inverse ANTs transform object.
    """
    if hasattr(M_phys, 'detach'):
        M_phys = M_phys.detach().cpu().numpy()
    if hasattr(t_phys, 'detach'):
        t_phys = t_phys.detach().cpu().numpy()

    tx_fwd = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx_fwd.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    tx_fwd.set_fixed_parameters(np.zeros(dim))

    M_phys_inv = np.linalg.inv(M_phys)
    t_phys_inv = -M_phys_inv @ t_phys
    tx_inv = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx_inv.set_parameters(np.concatenate([M_phys_inv.ravel(), t_phys_inv]))
    tx_inv.set_fixed_parameters(np.zeros(dim))

    if filename is not None:
        ants.write_transform(tx_fwd, filename)

    return tx_fwd, tx_inv


def compute_grid_to_physical_reference_matrix(shape, spacing, origin, direction, device=None, dtype=None) -> torch.Tensor:
    """
    Computes homogeneous transformation matrix $H$ mapping normalized grid coordinates `[-1, 1]` to physical scanner space.

    Parameters
    ----------
    shape : tuple of int
        Image grid shape in XYZ or ZYX order.
    spacing : tuple of float
        Voxel spacing in XYZ order.
    origin : tuple of float
        Image origin in XYZ order.
    direction : np.ndarray
        Direction matrix in XYZ order.
    device : str or torch.device, optional
        Target PyTorch compute device.
    dtype : torch.dtype, optional
        Target PyTorch data type.

    Returns
    -------
    H : torch.Tensor
        `(dim+1, dim+1)` homogeneous transformation matrix mapping normalized grid to physical space.
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
