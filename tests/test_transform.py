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
    # Since it's a constant translation, Jacobian determinant should be approx 1 everywhere
    assert np.allclose(jac_np, 1.0, atol=1e-3)
    
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
