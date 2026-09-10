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

from .spatial import (
    export_ants_displacement_field,
    export_ants_affine_transform,
    create_ants_affine,
    compute_grid_to_physical_reference_matrix,
    grid_to_physical_affine,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    get_identity_grid_torch,
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    jacobian_determinant,
    jacobian_determinant_image,
    deformation_stats,
    normalized_to_physical_disp,
    _to_numpy,
)
from .core.grid import compose_grids
from .core.jacobian import compute_physical_jacobian_determinant
from .core.inverse import update_inverse_field_nd


def _invert_t_grid(T_grid, dim: int):
    """Invert normalized grid affine matrix T_grid preserving tensor/array type and batch dim."""
    if T_grid is None:
        return None
    if isinstance(T_grid, torch.Tensor):
        T = T_grid
        is_batched = (T.ndim == 3)
        if not is_batched:
            T_mat = T.unsqueeze(0)
        else:
            T_mat = T
        batch_size = T_mat.shape[0]
        if T_mat.shape[1] == dim:
            homo = torch.eye(dim + 1, dtype=T.dtype, device=T.device).unsqueeze(0).repeat(batch_size, 1, 1)
            homo[:, :dim, :] = T_mat
        else:
            homo = T_mat
        inv_homo = torch.linalg.inv(homo)
        inv_T = inv_homo[:, :dim, :] if T_mat.shape[1] == dim else inv_homo
        return inv_T if is_batched else inv_T.squeeze(0)
    else:
        T_arr = np.asarray(T_grid)
        is_batched = (T_arr.ndim == 3)
        if not is_batched:
            T_mat = np.expand_dims(T_arr, 0)
        else:
            T_mat = T_arr
        batch_size = T_mat.shape[0]
        if T_mat.shape[1] == dim:
            homo = np.eye(dim + 1, dtype=T_arr.dtype)[np.newaxis, ...].repeat(batch_size, axis=0)
            homo[:, :dim, :] = T_mat
        else:
            homo = T_mat
        inv_homo = np.linalg.inv(homo).astype(T_arr.dtype)
        inv_T = inv_homo[:, :dim, :] if T_mat.shape[1] == dim else inv_homo
        return inv_T if is_batched else inv_T[0]


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

    def __init__(self, affine_grid=None, warp_field=None, metadata: dict = None, device='cpu', T_grid=None, is_physical=False, warp_inv_field=None, affine_matrix=None):
        if affine_grid is not None and not isinstance(affine_grid, torch.Tensor):
            if hasattr(affine_grid, 'numpy'):
                affine_grid = torch.from_numpy(np.array(affine_grid))
            else:
                affine_grid = torch.from_numpy(np.asarray(affine_grid))
        if warp_field is not None and not isinstance(warp_field, torch.Tensor):
            if hasattr(warp_field, 'numpy'):
                warp_field = torch.from_numpy(np.array(warp_field))
            else:
                warp_field = torch.from_numpy(np.asarray(warp_field))
        if warp_inv_field is not None and not isinstance(warp_inv_field, torch.Tensor):
            if hasattr(warp_inv_field, 'numpy'):
                warp_inv_field = torch.from_numpy(np.array(warp_inv_field))
            else:
                warp_inv_field = torch.from_numpy(np.asarray(warp_inv_field))

        if warp_field is not None and warp_field.dim() >= 3 and warp_field.shape[1] in [2, 3] and warp_field.shape[-1] not in [2, 3]:
            perm = (0,) + tuple(range(2, warp_field.dim())) + (1,)
            warp_field = warp_field.permute(perm)
        if warp_inv_field is not None and warp_inv_field.dim() >= 3 and warp_inv_field.shape[1] in [2, 3] and warp_inv_field.shape[-1] not in [2, 3]:
            perm = (0,) + tuple(range(2, warp_inv_field.dim())) + (1,)
            warp_inv_field = warp_inv_field.permute(perm)
        if affine_grid is not None and affine_grid.dim() >= 3 and affine_grid.shape[1] in [2, 3] and affine_grid.shape[-1] not in [2, 3]:
            perm = (0,) + tuple(range(2, affine_grid.dim())) + (1,)
            affine_grid = affine_grid.permute(perm)

        self.affine_grid = affine_grid
        self.warp_inv_field = warp_inv_field
        self.affine_matrix = affine_matrix
        self.metadata = metadata or {}
        self.device = device
        ref_tensor = warp_field if warp_field is not None else affine_grid
        if ref_tensor is not None:
            self.dim = ref_tensor.shape[-1]
            self.spatial = tuple(ref_tensor.shape[1:-1])
            if 'shape' in self.metadata:
                raw_shape = tuple(self.metadata['shape'])
                # If metadata shape is in ITK order (reversed compared to spatial) or already matches spatial,
                # retain tensor spatial shape to prevent spurious dimension-swapping resampling.
                if raw_shape == tuple(reversed(self.spatial)) or raw_shape == self.spatial:
                    self.target_shape = self.spatial
                else:
                    self.target_shape = raw_shape
            else:
                self.target_shape = self.spatial
        else:
            if 'shape' in self.metadata:
                self.dim = len(self.metadata['shape'])
                self.spatial = tuple(reversed(self.metadata['shape']))
            elif 'spacing' in self.metadata:
                self.dim = len(self.metadata['spacing'])
                self.spatial = (64,) * self.dim
            elif 'origin' in self.metadata:
                self.dim = len(self.metadata['origin'])
                self.spatial = (64,) * self.dim
            else:
                self.dim = 3
                self.spatial = (64,) * self.dim
            self.target_shape = self.spatial
        self.T_grid = T_grid

        self.is_physical = is_physical or (warp_field is not None and getattr(warp_field, 'is_physical', False))
        self.warp_field = warp_field
        if isinstance(self.warp_field, torch.Tensor):
            self.warp_field.is_physical = self.is_physical
        if self.warp_inv_field is not None and isinstance(self.warp_inv_field, torch.Tensor):
            self.warp_inv_field.is_physical = self.is_physical

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
        if self.affine_grid is not None:
            self.affine_grid = self.affine_grid.to(device)
        if self.warp_field is not None:
            self.warp_field = self.warp_field.to(device)
        if self.warp_inv_field is not None:
            self.warp_inv_field = self.warp_inv_field.to(device)
        if self.T_grid is not None:
            self.T_grid = self.T_grid.to(device)
        return self

    def _ensure_affine(self):
        """Lazily evaluate and cache self.affine_matrix = (M_itk, t_itk) from self.T_grid if affine_matrix is None."""
        if getattr(self, 'affine_matrix', None) is None and getattr(self, 'T_grid', None) is not None:
            dim = self.dim
            t_grid_tensor = self.T_grid if isinstance(self.T_grid, torch.Tensor) else torch.from_numpy(np.asarray(self.T_grid))
            spacing = self.metadata.get('spacing', tuple([1.0] * dim))
            origin = self.metadata.get('origin', tuple([0.0] * dim))
            direction = self.metadata.get('direction', np.eye(dim, dtype=np.float32))
            M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
                t_grid_tensor, self.target_shape, spacing, origin, direction,
                self.target_shape, spacing, origin, direction
            )
            P = np.eye(dim, dtype=np.float32)[::-1]
            M_itk = P @ _to_numpy(M_phys_zyx) @ P
            t_itk = P @ _to_numpy(t_phys_zyx)
            self.affine_matrix = (M_itk.astype(np.float32), t_itk.astype(np.float32))
        return getattr(self, 'affine_matrix', None)

    def _get_physical_affine_zyx(self):
        """
        Extracts physical affine parameters (M_phys_zyx, t_phys_zyx) in PyTorch tensor (ZYX) space
        as torch.Tensor on self.device with self.warp_field.dtype (or torch.float32).
        """
        device = self.device
        if self.warp_field is not None and isinstance(self.warp_field, torch.Tensor):
            dtype = self.warp_field.dtype
        elif self.affine_grid is not None and isinstance(self.affine_grid, torch.Tensor):
            dtype = self.affine_grid.dtype
        elif self.T_grid is not None and isinstance(self.T_grid, torch.Tensor):
            dtype = self.T_grid.dtype
        else:
            dtype = torch.float32
        dim = self.dim
        P = np.eye(dim, dtype=np.float32)[::-1]

        # Priority 1: self.T_grid
        if getattr(self, 'T_grid', None) is not None:
            t_grid_tensor = self.T_grid if isinstance(self.T_grid, torch.Tensor) else torch.from_numpy(np.asarray(self.T_grid))
            t_grid_tensor = t_grid_tensor.to(device=device, dtype=dtype)
            spacing = self.metadata.get('spacing', tuple([1.0] * dim))
            origin = self.metadata.get('origin', tuple([0.0] * dim))
            direction = self.metadata.get('direction', np.eye(dim, dtype=np.float32))
            M_phys_zyx, t_phys_zyx = grid_to_physical_affine_torch(
                t_grid_tensor, self.target_shape, spacing, origin, direction,
                self.target_shape, spacing, origin, direction
            )
            return M_phys_zyx.to(device=device, dtype=dtype), t_phys_zyx.to(device=device, dtype=dtype)

        # Priority 2: self.affine_matrix
        if getattr(self, 'affine_matrix', None) is not None:
            aff = self.affine_matrix
            if isinstance(aff, ants.ANTsTransform):
                params = np.asarray(aff.parameters, dtype=np.float32)
                M_itk = params[:dim * dim].reshape(dim, dim)
                t_itk = params[dim * dim:]
                if hasattr(aff, 'fixed_parameters') and len(aff.fixed_parameters) == dim:
                    C = np.asarray(aff.fixed_parameters, dtype=np.float32)
                    t_itk = t_itk + C - M_itk @ C
            elif isinstance(aff, tuple) and len(aff) == 2:
                M_itk = np.asarray(_to_numpy(aff[0]), dtype=np.float32)
                t_itk = np.asarray(_to_numpy(aff[1]), dtype=np.float32).ravel()
            else:
                mat = np.asarray(_to_numpy(aff), dtype=np.float32)
                if mat.shape == (dim + 1, dim + 1):
                    M_itk = mat[:dim, :dim]
                    t_itk = mat[:dim, dim]
                elif mat.shape == (dim, dim + 1):
                    M_itk = mat[:dim, :dim]
                    t_itk = mat[:dim, -1]
                elif mat.shape == (dim, dim):
                    M_itk = mat
                    t_itk = np.zeros(dim, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported affine_matrix shape: {mat.shape}")

            M_zyx = P @ M_itk @ P
            t_zyx = P @ t_itk
            return torch.as_tensor(M_zyx, device=device, dtype=dtype), torch.as_tensor(t_zyx, device=device, dtype=dtype)

        return None, None

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
        device = self.device
        if self.warp_field is not None and isinstance(self.warp_field, torch.Tensor):
            dtype = self.warp_field.dtype
        elif self.affine_grid is not None and isinstance(self.affine_grid, torch.Tensor):
            dtype = self.affine_grid.dtype
        elif self.T_grid is not None and isinstance(self.T_grid, torch.Tensor):
            dtype = self.T_grid.dtype
        else:
            dtype = torch.float32
        dim = self.dim

        spacing = self.metadata.get('spacing', tuple([1.0] * dim))
        origin = self.metadata.get('origin', tuple([0.0] * dim))
        direction = self.metadata.get('direction', np.eye(dim, dtype=np.float32))

        X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)

        if self.warp_field is not None:
            if self.target_shape != self.spatial:
                warp_resampled = F.interpolate(
                    torch.movedim(self.warp_field, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                warp_resampled = self.warp_field
        else:
            warp_resampled = torch.zeros(1, *self.target_shape, dim, device=device, dtype=dtype)

        if not self.is_physical:
            if self.affine_grid is not None:
                if self.target_shape != self.spatial:
                    affine_resampled = F.interpolate(
                        torch.movedim(self.affine_grid, -1, 1),
                        size=self.target_shape,
                        mode='bilinear' if dim == 2 else 'trilinear',
                        align_corners=True
                    ).movedim(1, -1)
                else:
                    affine_resampled = self.affine_grid
            elif self.T_grid is not None:
                T = self.T_grid if self.T_grid.ndim == 3 else self.T_grid.unsqueeze(0)
                if T.shape[-1] == dim + 1 and T.shape[-2] == dim + 1:
                    T = T[:, :dim, :]
                affine_resampled = F.affine_grid(T.to(device=device, dtype=dtype), [1, 1, *self.target_shape], align_corners=True)
            elif getattr(self, 'affine_matrix', None) is not None:
                moving_shape = image_tensor.shape[2:]
                M_zyx, t_zyx = self._get_physical_affine_zyx()
                y_phys = X_phys @ M_zyx.t() + t_zyx
                affine_resampled = physical_to_normalized_torch(y_phys, moving_shape, spacing, origin, direction)
            else:
                affine_resampled = None

            identity = get_identity_grid_torch(self.target_shape, device=device, dtype=dtype)
            phi = identity + warp_resampled
            composed_grid = compose_grids(affine_resampled, phi) if affine_resampled is not None else phi
            return F.grid_sample(image_tensor, composed_grid, mode=mode, padding_mode='border', align_corners=True)

        phi_l2r_phys = X_phys + warp_resampled

        moving_shape = image_tensor.shape[2:]
        moving_spacing = spacing
        moving_origin = origin
        moving_direction = direction

        M_zyx, t_zyx = self._get_physical_affine_zyx()
        if M_zyx is not None:
            y_phys = phi_l2r_phys @ M_zyx.t() + t_zyx
            composed_grid = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
        else:
            if self.affine_grid is not None:
                if self.target_shape != self.spatial:
                    affine_resampled = F.interpolate(
                        torch.movedim(self.affine_grid, -1, 1),
                        size=self.target_shape,
                        mode='bilinear' if dim == 2 else 'trilinear',
                        align_corners=True
                    ).movedim(1, -1)
                else:
                    affine_resampled = self.affine_grid
            else:
                affine_resampled = None
            phi_l2r_norm = physical_to_normalized_torch(phi_l2r_phys, self.target_shape, spacing, origin, direction)
            composed_grid = compose_grids(affine_resampled, phi_l2r_norm) if affine_resampled is not None else phi_l2r_norm

        return F.grid_sample(image_tensor, composed_grid, mode=mode, padding_mode='border', align_corners=True)

    def _get_composite_normalized_displacement(self) -> torch.Tensor:
        """Computes the total composite displacement field in normalized coordinates [-1, 1]."""
        device = self.device
        if self.warp_field is not None and isinstance(self.warp_field, torch.Tensor):
            dtype = self.warp_field.dtype
        elif self.affine_grid is not None and isinstance(self.affine_grid, torch.Tensor):
            dtype = self.affine_grid.dtype
        elif self.T_grid is not None and isinstance(self.T_grid, torch.Tensor):
            dtype = self.T_grid.dtype
        else:
            dtype = torch.float32
        dim = self.dim

        spacing = self.metadata.get('spacing', tuple([1.0] * dim))
        origin = self.metadata.get('origin', tuple([0.0] * dim))
        direction = self.metadata.get('direction', np.eye(dim, dtype=np.float32))

        if self.warp_field is not None:
            if self.target_shape != self.spatial:
                warp_resampled = F.interpolate(
                    torch.movedim(self.warp_field, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                warp_resampled = self.warp_field
        else:
            warp_resampled = torch.zeros(1, *self.target_shape, dim, device=device, dtype=dtype)

        if not self.is_physical:
            if self.affine_grid is not None:
                if self.target_shape != self.spatial:
                    affine_resampled = F.interpolate(
                        torch.movedim(self.affine_grid, -1, 1),
                        size=self.target_shape,
                        mode='bilinear' if dim == 2 else 'trilinear',
                        align_corners=True
                    ).movedim(1, -1)
                else:
                    affine_resampled = self.affine_grid
            elif self.T_grid is not None:
                T = self.T_grid if self.T_grid.ndim == 3 else self.T_grid.unsqueeze(0)
                if T.shape[-1] == dim + 1 and T.shape[-2] == dim + 1:
                    T = T[:, :dim, :]
                affine_resampled = F.affine_grid(T.to(device=device, dtype=dtype), [1, 1, *self.target_shape], align_corners=True)
            elif getattr(self, 'affine_matrix', None) is not None:
                M_zyx, t_zyx = self._get_physical_affine_zyx()
                X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)
                y_phys = X_phys @ M_zyx.t() + t_zyx
                affine_resampled = physical_to_normalized_torch(y_phys, self.target_shape, spacing, origin, direction)
            else:
                affine_resampled = None

            identity = get_identity_grid_torch(self.target_shape, device=device, dtype=dtype)
            phi = identity + warp_resampled
            composed_grid = compose_grids(affine_resampled, phi) if affine_resampled is not None else phi
            return composed_grid - identity
        else:
            X_phys = get_physical_grid_torch(self.target_shape, spacing, origin, direction, device=device, dtype=dtype)
            phi_l2r_phys = X_phys + warp_resampled

            moving_shape = self.target_shape
            moving_spacing = spacing
            moving_origin = origin
            moving_direction = direction

            M_zyx, t_zyx = self._get_physical_affine_zyx()
            if M_zyx is not None:
                y_phys = phi_l2r_phys @ M_zyx.t() + t_zyx
                composed_grid = physical_to_normalized_torch(y_phys, moving_shape, moving_spacing, moving_origin, moving_direction)
            else:
                if self.affine_grid is not None:
                    if self.target_shape != self.spatial:
                        affine_resampled = F.interpolate(
                            torch.movedim(self.affine_grid, -1, 1),
                            size=self.target_shape,
                            mode='bilinear' if dim == 2 else 'trilinear',
                            align_corners=True
                        ).movedim(1, -1)
                    else:
                        affine_resampled = self.affine_grid
                else:
                    affine_resampled = None
                phi_l2r_norm = physical_to_normalized_torch(phi_l2r_phys, self.target_shape, spacing, origin, direction)
                composed_grid = compose_grids(affine_resampled, phi_l2r_norm) if affine_resampled is not None else phi_l2r_norm

            identity = get_identity_grid_torch(self.target_shape, device=device, dtype=dtype)
            return composed_grid - identity

    def get_jacobian_determinant(self, method: str = 'central', as_numpy: bool | None = None):
        """
        Computes the Jacobian determinant map of the total composite deformation natively in PyTorch.

        Parameters
        ----------
        method : str, default 'central'
            Derivative evaluation method ('central' or 'bspline').
        as_numpy : bool, optional
            If True, returns NumPy array. If False, returns PyTorch Tensor.
            If None (default), returns NumPy array for 'central' and PyTorch Tensor for 'bspline'.

        Returns
        -------
        np.ndarray or torch.Tensor
            Physical Jacobian determinants map.
        """
        total_normalized_disp = self._get_composite_normalized_displacement()
        jac = compute_physical_jacobian_determinant(
            total_normalized_disp,
            direction=self.metadata['direction'],
            spacing=self.metadata['spacing'],
            method=method
        )
        return_numpy = (method != 'bspline') if as_numpy is None else bool(as_numpy)
        if return_numpy and isinstance(jac, torch.Tensor):
            return jac.squeeze(0).detach().cpu().numpy()
        return jac

    def jacobian(self, method: str = 'central', as_numpy: bool | None = None):
        """Alias for get_jacobian_determinant."""
        return self.get_jacobian_determinant(method=method, as_numpy=as_numpy)

    def _to_physical_displacement(self, disp: torch.Tensor, is_physical: bool = False) -> ants.ANTsImage:
        """Helper to convert displacement tensor to ANTsImage with correct vector channel order and spatial layout."""
        if not is_physical:
            disp = normalized_to_physical_disp(
                disp,
                shape=self.target_shape,
                spacing=self.metadata['spacing'],
                direction=self.metadata['direction'],
                origin=self.metadata.get('origin'),
                device=self.device,
                dtype=disp.dtype if isinstance(disp, torch.Tensor) else torch.float32,
            )
        return export_ants_displacement_field(
            disp,
            origin=self.metadata.get('origin'),
            spacing=self.metadata.get('spacing'),
            direction=self.metadata.get('direction')
        )

    def to_displacement_field(self) -> ants.ANTsImage:
        """Computes and exports the total composite transformation as an ANTs displacement field."""
        has_affine = (
            getattr(self, 'T_grid', None) is not None or
            getattr(self, 'affine_grid', None) is not None or
            getattr(self, 'affine_matrix', None) is not None
        )
        if self.is_physical and not has_affine:
            if self.target_shape != self.spatial:
                warp_resampled = F.interpolate(
                    torch.movedim(self.warp_field, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if self.dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                warp_resampled = self.warp_field
            return self._to_physical_displacement(warp_resampled, is_physical=True)

        total_normalized_disp = self._get_composite_normalized_displacement()
        return self._to_physical_displacement(total_normalized_disp, is_physical=False)

    def jacobian_determinant_image(self, ref_image=None) -> ants.ANTsImage:
        """Computes the Jacobian determinant map and returns as an ANTsImage."""
        disp_img = self.to_displacement_field()
        ref = ref_image if ref_image is not None else disp_img
        return jacobian_determinant_image(disp_img, ref_image=ref)

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
        ants_disp = self.to_displacement_field()
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
        device = self.device
        dtype = self.warp_field.dtype
        dim = self.dim

        spacing = self.metadata['spacing']
        origin = self.metadata['origin']
        direction = self.metadata['direction']

        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field

        ants_warp = self._to_physical_displacement(warp_resampled, is_physical=self.is_physical)
        ants.image_write(ants_warp, f"{prefix}1SyNWarp.nii.gz")

        if self.affine_grid is not None:
            if self.target_shape != self.spatial:
                affine_resampled = F.interpolate(
                    torch.movedim(self.affine_grid, -1, 1),
                    size=self.target_shape,
                    mode='bilinear' if dim == 2 else 'trilinear',
                    align_corners=True
                ).movedim(1, -1)
            else:
                affine_resampled = self.affine_grid

            identity = get_identity_grid_torch(self.target_shape, device=device, dtype=dtype)

            affine_disp = affine_resampled - identity
            ants_affine = self._to_physical_displacement(affine_disp, is_physical=False)
            ants.image_write(ants_affine, f"{prefix}0AffineWarp.nii.gz")
            return [f"{prefix}1SyNWarp.nii.gz", f"{prefix}0AffineWarp.nii.gz"]

        return [f"{prefix}1SyNWarp.nii.gz"]

    def export(self, outprefix: str | None = None):
        """
        Dual-mode ITK transform export:
        - When outprefix is provided: writes standard ITK 0GenericAffine.mat, 1Warp.nii.gz,
          and 1InverseWarp.nii.gz files matching ants.apply_transforms convention.
        - When outprefix is None: returns in-memory PyTorch tensors without disk I/O.
        """
        self._ensure_affine()
        if outprefix is None:
            return {
                'affine_grid': self.affine_grid,
                'T_grid': getattr(self, 'T_grid', None),
                'warp_field': self.warp_field,
                'warp_inv_field': getattr(self, 'warp_inv_field', None),
                'fwd_warp': self.warp_field,
                'inv_warp': getattr(self, 'warp_inv_field', None),
                'affine_matrix': getattr(self, 'affine_matrix', None),
                'metadata': self.metadata
            }

        os.makedirs(os.path.dirname(outprefix) or '.', exist_ok=True)
        fwd_transforms = []
        inv_transforms = []

        # 1. Forward and Inverse Warp fields
        warp_path = None
        inv_warp_path = None
        if self.warp_field is not None:
            warp_path = f"{outprefix}1Warp.nii.gz"
            ants_warp = self._to_physical_displacement(self.warp_field, is_physical=self.is_physical)
            ants.image_write(ants_warp, warp_path)
            fwd_transforms.append(warp_path)

            inv_warp_path = f"{outprefix}1InverseWarp.nii.gz"
            if getattr(self, 'warp_inv_field', None) is not None:
                ants_inv_warp = self._to_physical_displacement(self.warp_inv_field, is_physical=self.is_physical)
            else:
                spacing = self.metadata.get('spacing', None)
                origin = self.metadata.get('origin', None)
                direction = self.metadata.get('direction', None)
                if self.is_physical and spacing is not None:
                    inv_disp = update_inverse_field_nd(
                        self.warp_field,
                        spacing=spacing,
                        origin=origin if origin is not None else (0.0,) * self.dim,
                        direction=direction if direction is not None else np.eye(self.dim)
                    )
                else:
                    inv_disp = update_inverse_field_nd(self.warp_field)
                ants_inv_warp = self._to_physical_displacement(inv_disp, is_physical=self.is_physical)
            ants.image_write(ants_inv_warp, inv_warp_path)

        # 2. Affine transform (.mat)
        affine_path = None
        has_affine = getattr(self, 'affine_matrix', None) is not None

        if has_affine:
            affine_path = f"{outprefix}0GenericAffine.mat"
            if isinstance(self.affine_matrix, ants.ANTsTransform):
                ants.write_transform(self.affine_matrix, affine_path)
            elif isinstance(self.affine_matrix, tuple) and len(self.affine_matrix) == 2:
                M_phys, t_phys = self.affine_matrix
                export_ants_affine_transform(M_phys, t_phys, dim=self.dim, filename=affine_path)
            else:
                mat = np.asarray(_to_numpy(self.affine_matrix))
                if mat.shape == (self.dim + 1, self.dim + 1):
                    M_phys = mat[:self.dim, :self.dim]
                    t_phys = mat[:self.dim, self.dim]
                    export_ants_affine_transform(M_phys, t_phys, dim=self.dim, filename=affine_path)
                elif mat.shape == (self.dim, self.dim + 1):
                    M_phys = mat[:self.dim, :self.dim]
                    t_phys = mat[:self.dim, -1]
                    export_ants_affine_transform(M_phys, t_phys, dim=self.dim, filename=affine_path)
                else:
                    raise ValueError(f"Unsupported affine_matrix shape: {mat.shape}")
            fwd_transforms.append(affine_path)
            inv_transforms.append(affine_path)

        if inv_warp_path is not None:
            inv_transforms.append(inv_warp_path)

        return {
            'fwdtransforms': fwd_transforms,
            'invtransforms': inv_transforms,
            'fwd_transforms': fwd_transforms,
            'inv_transforms': inv_transforms,
            'affine': affine_path,
            'warp': warp_path,
            'inverse_warp': inv_warp_path
        }

    def invert(self) -> "SyNToTransform":
        """Returns an inverted SyNToTransform object."""
        self._ensure_affine()
        if self.warp_inv_field is not None:
            inv_warp = self.warp_inv_field
        else:
            if self.warp_field is not None:
                if self.is_physical:
                    spacing = self.metadata.get('spacing', None)
                    origin = self.metadata.get('origin', (0.0,) * self.dim)
                    direction = self.metadata.get('direction', np.eye(self.dim))
                    inv_warp = update_inverse_field_nd(
                        self.warp_field,
                        spacing=spacing,
                        origin=origin,
                        direction=direction
                    )
                else:
                    inv_warp = update_inverse_field_nd(self.warp_field)
            else:
                inv_warp = None

        inv_affine_mat = None
        if getattr(self, 'affine_matrix', None) is not None:
            if isinstance(self.affine_matrix, ants.ANTsTransform):
                inv_affine_mat = (
                    ants.invert_ants_transform(self.affine_matrix)
                    if hasattr(ants, 'invert_ants_transform')
                    else self.affine_matrix.invert()
                )
            elif isinstance(self.affine_matrix, tuple) and len(self.affine_matrix) == 2:
                M = np.asarray(_to_numpy(self.affine_matrix[0]), dtype=np.float64)
                t = np.asarray(_to_numpy(self.affine_matrix[1]), dtype=np.float64).ravel()
                M_inv = np.linalg.inv(M)
                t_inv = -M_inv @ t
                inv_affine_mat = (M_inv.astype(np.float32), t_inv.astype(np.float32))
            else:
                mat = np.asarray(_to_numpy(self.affine_matrix), dtype=np.float64)
                if mat.shape == (self.dim + 1, self.dim + 1):
                    inv_affine_mat = np.linalg.inv(mat).astype(mat.dtype)
                elif mat.shape == (self.dim, self.dim + 1):
                    homo = np.eye(self.dim + 1, dtype=np.float64)
                    homo[:self.dim, :] = mat
                    inv_homo = np.linalg.inv(homo)
                    inv_affine_mat = inv_homo[:self.dim, :].astype(mat.dtype)
                else:
                    inv_affine_mat = np.linalg.inv(mat).astype(mat.dtype)

        inv_T_grid = None
        if getattr(self, 'T_grid', None) is not None:
            inv_T_grid = _invert_t_grid(self.T_grid, self.dim)

        inv_affine_grid = None
        if getattr(self, 'affine_grid', None) is not None:
            if inv_T_grid is not None:
                theta = inv_T_grid if isinstance(inv_T_grid, torch.Tensor) else torch.from_numpy(np.asarray(inv_T_grid))
                if theta.ndim == 2:
                    theta = theta.unsqueeze(0)
                if theta.shape[1] == self.dim + 1:
                    theta = theta[:, :self.dim, :]
                size = [1, 1] + list(self.target_shape)
                inv_affine_grid = F.affine_grid(
                    theta.to(device=self.device, dtype=torch.float32),
                    size=size,
                    align_corners=True
                )
            else:
                id_grid = get_identity_grid_torch(self.spatial, device=self.affine_grid.device, dtype=self.affine_grid.dtype)
                X = id_grid.reshape(-1, self.dim)
                X_homo = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], dim=-1)
                Y = self.affine_grid.reshape(-1, self.dim)
                theta_t = torch.linalg.lstsq(X_homo, Y).solution
                theta = theta_t.t()
                homo = torch.eye(self.dim + 1, device=theta.device, dtype=theta.dtype)
                homo[:self.dim, :] = theta
                inv_homo = torch.linalg.inv(homo)
                inv_T_grid = inv_homo[:self.dim, :].unsqueeze(0)
                size = [1, 1] + list(self.target_shape)
                inv_affine_grid = F.affine_grid(
                    inv_T_grid.to(device=self.device, dtype=torch.float32),
                    size=size,
                    align_corners=True
                )
        elif inv_T_grid is not None:
            theta = inv_T_grid if isinstance(inv_T_grid, torch.Tensor) else torch.from_numpy(np.asarray(inv_T_grid))
            if theta.ndim == 2:
                theta = theta.unsqueeze(0)
            if theta.shape[1] == self.dim + 1:
                theta = theta[:, :self.dim, :]
            size = [1, 1] + list(self.target_shape)
            inv_affine_grid = F.affine_grid(
                theta.to(device=self.device, dtype=torch.float32),
                size=size,
                align_corners=True
            )

        return SyNToTransform(
            affine_grid=inv_affine_grid,
            warp_field=inv_warp,
            metadata=self.metadata,
            device=self.device,
            T_grid=inv_T_grid,
            is_physical=self.is_physical,
            warp_inv_field=self.warp_field,
            affine_matrix=inv_affine_mat
        )

    def to_ants(self, outprefix: str | None = None):
        """Converts to ANTs format, either on-disk (if outprefix provided) or in-memory ANTs objects."""
        if outprefix is not None:
            return self.export(outprefix=outprefix)
        self._ensure_affine()
        if self.warp_field is not None:
            ants_warp = self._to_physical_displacement(self.warp_field, is_physical=self.is_physical)
        else:
            ants_warp = None
        ants_tx = None
        if getattr(self, 'affine_matrix', None) is not None:
            if isinstance(self.affine_matrix, ants.ANTsTransform):
                ants_tx = self.affine_matrix
            elif isinstance(self.affine_matrix, tuple) and len(self.affine_matrix) == 2:
                ants_tx = create_ants_affine(self.affine_matrix[0], self.affine_matrix[1], dim=self.dim)
            else:
                ants_tx = create_ants_affine(self.affine_matrix, dim=self.dim)
        res = {'warp': ants_warp, 'affine': ants_tx}
        if self.warp_field is None:
            res['inverse_warp'] = None
        elif getattr(self, 'warp_inv_field', None) is not None:
            res['inverse_warp'] = self._to_physical_displacement(self.warp_inv_field, is_physical=self.is_physical)
        return res


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Coordinate & Displacement Export Bridges (Re-exported from syntx.spatial)
# ═══════════════════════════════════════════════════════════════════════════════

from .spatial import (
    export_ants_displacement_field,
    export_ants_affine_transform,
    create_ants_affine,
    compute_grid_to_physical_reference_matrix,
    grid_to_physical_affine,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    get_identity_grid_torch,
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    jacobian_determinant,
    jacobian_determinant_image,
    deformation_stats,
    normalized_to_physical_disp,
)

__all__ = [
    "SyNToTransform",
    "export_ants_displacement_field",
    "export_ants_affine_transform",
    "create_ants_affine",
    "compute_grid_to_physical_reference_matrix",
    "grid_to_physical_affine",
    "grid_to_physical_affine_torch",
    "physical_to_grid_affine",
    "get_physical_grid_torch",
    "physical_to_normalized_torch",
    "physical_to_normalized_torch_cached",
    "get_identity_grid_torch",
    "disp_tensor_to_itk",
    "disp_itk_to_tensor",
    "jacobian_determinant",
    "jacobian_determinant_image",
    "deformation_stats",
    "normalized_to_physical_disp",
]

