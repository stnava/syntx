import torch
import torch.nn.functional as F
import numpy as np
import ants
import os

class SyNToTransform:
    def __init__(self, affine_grid, warp_field, metadata, device='cpu'):
        """
        Object holder bridging PyTorch native normalized matrices to ITK physical formats.
        affine_grid: (1, *spatial, dim) normalized grid
        warp_field: (1, *spatial, dim) normalized displacement
        metadata: dict containing origin, spacing, direction, and optionally shape
        """
        if not hasattr(affine_grid, 'device'):
            if hasattr(affine_grid, 'numpy'):
                affine_grid = torch.from_numpy(np.array(affine_grid))
            else:
                affine_grid = torch.from_numpy(np.asarray(affine_grid))
        if not hasattr(warp_field, 'device'):
            if hasattr(warp_field, 'numpy'):
                warp_field = torch.from_numpy(np.array(warp_field))
            else:
                warp_field = torch.from_numpy(np.asarray(warp_field))

        self.affine_grid = affine_grid
        self.warp_field = warp_field
        self.metadata = metadata
        self.device = device
        self.dim = warp_field.shape[-1]
        self.spatial = warp_field.shape[1:-1]
        self.target_shape = tuple(metadata['shape']) if 'shape' in metadata else self.spatial
        
    def to(self, device):
        """Moves internal tensors to the specified device and updates self.device."""
        self.device = device
        self.affine_grid = self.affine_grid.to(device)
        self.warp_field = self.warp_field.to(device)
        return self

    def apply(self, image_tensor, mode='bilinear'):
        """Native GPU PyTorch application"""
        from .syn import compose_grids
        
        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
            
            affine_resampled = F.interpolate(
                torch.movedim(self.affine_grid, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field
            affine_resampled = self.affine_grid
            
        grids = [torch.linspace(-1, 1, size, device=self.device, dtype=self.warp_field.dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        phi = identity + warp_resampled
        composed_grid = compose_grids(affine_resampled, phi)
        
        return F.grid_sample(image_tensor, composed_grid, mode=mode, padding_mode='border', align_corners=True)

    def get_jacobian_determinant(self):
        """Computes the Jacobian determinant map of the total composite deformation natively in PyTorch."""
        from .syn import compute_physical_jacobian_determinant, compose_grids
        
        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
            
            affine_resampled = F.interpolate(
                torch.movedim(self.affine_grid, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field
            affine_resampled = self.affine_grid
            
        grids = [torch.linspace(-1, 1, size, device=self.device, dtype=self.warp_field.dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        phi = identity + warp_resampled
        composed_grid = compose_grids(affine_resampled, phi)
        
        # Calculate total normalized displacement
        total_normalized_disp = composed_grid - identity
        return compute_physical_jacobian_determinant(
            total_normalized_disp,
            direction=self.metadata['direction'],
            spacing=self.metadata['spacing']
        ).squeeze(0).detach().cpu().numpy()

    def _to_physical_displacement(self, normalized_disp):
        """Converts a normalized PyTorch [-1, 1] displacement field to an ITK physical LPS field."""
        spatial_shape = torch.tensor(list(reversed(self.target_shape)), dtype=torch.float32, device=self.device)
        voxel_disp = normalized_disp * (spatial_shape - 1) / 2.0
        
        direction = np.array(self.metadata['direction'])
        spacing = np.array(self.metadata['spacing'])
        
        phys_disp = voxel_disp.squeeze(0).detach().cpu().numpy() * spacing
        phys_disp_flat = phys_disp.reshape(-1, self.dim)
        phys_disp_flat = phys_disp_flat @ direction.T
        phys_disp = phys_disp_flat.reshape(tuple(self.target_shape) + (self.dim,))
        
        return ants.from_numpy(
            phys_disp, 
            origin=self.metadata['origin'], 
            spacing=self.metadata['spacing'], 
            direction=self.metadata['direction'], 
            has_components=True
        )

    def to_composite_warp(self, filename):
        """Exports the combined Affine + SyN fields into a single classic ITK-compatible CompositeWarp.nii.gz"""
        from .syn import compose_grids
        
        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
            
            affine_resampled = F.interpolate(
                torch.movedim(self.affine_grid, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field
            affine_resampled = self.affine_grid
            
        grids = [torch.linspace(-1, 1, size, device=self.device, dtype=self.warp_field.dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        phi = identity + warp_resampled
        composed_grid = compose_grids(affine_resampled, phi)
        
        total_normalized_disp = composed_grid - identity
        ants_disp = self._to_physical_displacement(total_normalized_disp)
        
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        ants.image_write(ants_disp, filename)
        return filename

    def export_classic(self, prefix):
        """
        Exports the transformations separated as physical displacement fields.
        This provides perfect isolation of the Affine and SyN fields for TBM analysis
        without suffering from PyTorch-to-ITK matrix coordinate translation errors.
        """
        if self.target_shape != self.spatial:
            warp_resampled = F.interpolate(
                torch.movedim(self.warp_field, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
            
            affine_resampled = F.interpolate(
                torch.movedim(self.affine_grid, -1, 1),
                size=self.target_shape,
                mode='bilinear' if self.dim == 2 else 'trilinear',
                align_corners=True
            ).movedim(1, -1)
        else:
            warp_resampled = self.warp_field
            affine_resampled = self.affine_grid
            
        grids = [torch.linspace(-1, 1, size, device=self.device, dtype=self.warp_field.dtype) for size in self.target_shape]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
        
        affine_disp = affine_resampled - identity
        ants_affine = self._to_physical_displacement(affine_disp)
        
        ants_warp = self._to_physical_displacement(warp_resampled)
        
        ants.image_write(ants_affine, f"{prefix}0AffineWarp.nii.gz")
        ants.image_write(ants_warp, f"{prefix}1SyNWarp.nii.gz")
        
        return [f"{prefix}1SyNWarp.nii.gz", f"{prefix}0AffineWarp.nii.gz"]
