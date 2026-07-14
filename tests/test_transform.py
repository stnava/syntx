import os
import pytest
import numpy as np
import torch
import ants
from syntx.transform import SyNToTransform

def test_synto_transform_flow(synthetic_ants_intensity, tmp_path):
    device = torch.device('cpu')
    fixed_img = synthetic_ants_intensity
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.from_numpy(fixed_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Create dummy identity-like grids and displacement fields
    shape = fixed_img.shape
    dim = fixed_img.dimension
    
    grids = [torch.linspace(-1, 1, size, device=device) for size in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)
    
    # Translate by 1 voxel along first axis
    # u = grid_new - identity
    # Shifted identity: grid_new = identity + offset
    # In normalized space, offset of 1 voxel is 2 / (32 - 1)
    offset = 2.0 / (shape[0] - 1)
    warp_field = torch.zeros_like(identity)
    warp_field[..., 0] = offset
    
    metadata = {
        'origin': fixed_img.origin,
        'spacing': fixed_img.spacing,
        'direction': fixed_img.direction,
        'shape': shape
    }
    
    tx = SyNToTransform(
        affine_grid=identity,
        warp_field=warp_field,
        metadata=metadata,
        device=device
    )
    
    # Apply warp
    warped_tensor = tx.apply(fixed_tensor, mode='bilinear')
    assert warped_tensor.shape == fixed_tensor.shape
    
    # Check Jacobian determinant
    jac_np = tx.get_jacobian_determinant()
    assert jac_np.shape == shape
    # Since it's a constant translation, Jacobian determinant should be approx 1 in the interior.
    # We slice out the outer 2 voxels to avoid grid_sample border clamping artifacts which propagate
    # 2 voxels deep due to central differences.
    assert np.allclose(jac_np[2:-2, 2:-2, 2:-2], 1.0, atol=1e-3)
    
    # Export to ITK/ANTs files
    prefix = os.path.join(tmp_path, "tx_test_")
    tx.export_classic(prefix)
    
    affine_warp = f"{prefix}0AffineWarp.nii.gz"
    syn_warp = f"{prefix}1SyNWarp.nii.gz"
    
    assert os.path.exists(affine_warp)
    assert os.path.exists(syn_warp)
    
    composite_warp_path = os.path.join(tmp_path, "CompositeWarp.nii.gz")
    tx.to_composite_warp(composite_warp_path)
    assert os.path.exists(composite_warp_path)
    
    field = ants.image_read(composite_warp_path)
    assert field.components == dim
    assert field.shape == shape


def test_synto_transform_extra(synthetic_ants_intensity, tmp_path):
    import torch.nn.functional as F
    device = torch.device('cpu')
    fixed_img = synthetic_ants_intensity
    shape = fixed_img.shape
    dim = fixed_img.dimension
    
    resampled_shape = (24, 24, 24)
    metadata = {
        'origin': fixed_img.origin,
        'spacing': fixed_img.spacing,
        'direction': fixed_img.direction,
        'shape': resampled_shape
    }

    grids = [torch.linspace(-1, 1, size, device=device) for size in shape]
    meshgrid = torch.meshgrid(*grids, indexing='ij')
    identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

    class MockArray:
        def __init__(self, arr):
            self.arr = arr
        @property
        def numpy(self):
            return self.arr
        def __array__(self, dtype=None, copy=None):
            return self.arr

    # 1. Use MockArray to test __init__ hasattr(..., 'numpy') handling
    # Cover MockArray for affine_grid, numpy for warp_field
    affine_np = MockArray(identity.numpy())
    warp_np = torch.zeros_like(identity).numpy()
    
    tx = SyNToTransform(
        affine_grid=affine_np,
        warp_field=warp_np,
        metadata=metadata,
        device=device
    )
    
    # Cover numpy for affine_grid, MockArray for warp_field
    affine_np2 = identity.numpy()
    warp_np2 = MockArray(torch.zeros_like(identity).numpy())
    _ = SyNToTransform(
        affine_grid=affine_np2,
        warp_field=warp_np2,
        metadata=metadata,
        device=device
    )
    
    tx = SyNToTransform(
        affine_grid=affine_np,
        warp_field=warp_np,
        metadata=metadata,
        device=device
    )
    
    # 3. Test .to(device)
    tx = tx.to(device)
    
    # 4. Test apply with resampling
    input_tensor = torch.from_numpy(fixed_img.numpy()).unsqueeze(0).unsqueeze(0).float()
    resampled_fixed_tensor = F.interpolate(
        input_tensor,
        size=resampled_shape,
        mode='trilinear',
        align_corners=True
    )
    warped_tensor = tx.apply(resampled_fixed_tensor, mode='bilinear')
    assert warped_tensor.shape == resampled_fixed_tensor.shape
    
    # 5. Test get_jacobian_determinant with resampling
    jac_np = tx.get_jacobian_determinant()
    assert jac_np.shape == resampled_shape
    assert np.allclose(jac_np[2:-2, 2:-2, 2:-2], 1.0, atol=1e-3)
    
    # 6. Test classic export with resampling
    prefix = os.path.join(tmp_path, "tx_resample_")
    tx.export_classic(prefix)
    assert os.path.exists(f"{prefix}0AffineWarp.nii.gz")
    assert os.path.exists(f"{prefix}1SyNWarp.nii.gz")
    
    # 7. Test to_composite_warp with resampling
    composite_warp_path = os.path.join(tmp_path, "CompositeWarp_resample.nii.gz")
    tx.to_composite_warp(composite_warp_path)
    assert os.path.exists(composite_warp_path)

