import torch
import torch.nn.functional as F
import numpy as np
import ants
import os

class SyNToTransform:
    def __init__(self, affine_grid, warp_field, metadata, device='cpu', T_grid=None, is_physical=False):
        """
        Object holder bridging PyTorch native normalized matrices to ITK physical formats.
        affine_grid: (1, *spatial, dim) normalized grid
        warp_field: (1, *spatial, dim) normalized displacement
        metadata: dict containing origin, spacing, direction, and optionally shape
        """
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
        """Moves internal tensors to the specified device and updates self.device."""
        self.device = device
        self.affine_grid = self.affine_grid.to(device)
        self.warp_field = self.warp_field.to(device)
        if self.T_grid is not None:
            self.T_grid = self.T_grid.to(device)
        return self

    def apply(self, image_tensor, mode='bilinear'):
        """Native GPU PyTorch application"""
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

    def get_jacobian_determinant(self):
        """Computes the Jacobian determinant map of the total composite deformation natively in PyTorch."""
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
            # Note: Jacobian is computed in physical space using total displacement
            # Wait, compute_physical_jacobian_determinant takes (normalized_disp, spacing, direction).
            # But wait, in transform.py, the original code computes it from composed_grid - identity:
            # Let's rebuild the composed_grid.
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

    def _to_physical_displacement(self, disp, is_physical=False):
        """Converts a displacement field to an ITK physical LPS field."""
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

    def to_composite_warp(self, filename):
        """Exports the combined Affine + SyN fields into a single classic ITK-compatible CompositeWarp.nii.gz"""
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

    def export_classic(self, prefix):
        """
        Exports the transformations separated as physical displacement fields.
        This provides perfect isolation of the Affine and SyN fields for TBM analysis
        without suffering from PyTorch-to-ITK matrix coordinate translation errors.
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
