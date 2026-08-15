import pytest
import ants
import numpy as np
import torch
from syntx.tvf_adj import (
    get_physical_grid_torch,
    physical_to_normalized_torch,
    image_gradient,
    fluid_smooth,
    integrate_svf,
    integrate_forward,
    TVFRegistrationAdjoint,
    tvf_registration_adjoint
)

def test_tvf_adj_2d_basic():
    """Test the basic execution of tvf_registration_adjoint in 2D."""
    # Use ants data for a quick 2D test
    fi = ants.image_read(ants.get_data('r16')).resample_image((32, 32), use_voxels=True)
    mi = ants.image_read(ants.get_data('r64')).resample_image((32, 32), use_voxels=True)
    
    # Normalizing images slightly for faster convergence
    fi = (fi - fi.mean()) / fi.std()
    mi = (mi - mi.mean()) / mi.std()

    # Run adjoint optimization
    res = tvf_registration_adjoint(
        fixed=fi,
        moving=mi,
        flow_sigma=1.0,
        total_sigma=0.0,
        lr=50.0,
        levels=[1],
        reg_iterations=[2],
        device='cpu'
    )
    
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
    
    warp = ants.image_read(res['fwdtransforms'][0])
    assert warp.shape == (32, 32)
    assert warp.components == 2

def test_tvf_adj_helpers():
    """Test helper functions in tvf_adj.py to boost coverage."""
    device = 'cpu'
    
    # 1. get_physical_grid_torch 2D
    grid2d = get_physical_grid_torch((16, 16), (1.0, 1.0), (0.0, 0.0), np.eye(2), device=device)
    assert grid2d.shape == (16, 16, 2)
    
    # 2. get_physical_grid_torch 3D
    grid3d = get_physical_grid_torch((16, 16, 16), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3), device=device)
    assert grid3d.shape == (16, 16, 16, 3)
    
    # 3. physical_to_normalized_torch
    norm_grid = physical_to_normalized_torch(grid2d, (16, 16), (1.0, 1.0), (0.0, 0.0), np.eye(2))
    assert norm_grid.min() >= -1.0
    assert norm_grid.max() <= 1.0
    
    # 4. image_gradient 2D
    img2d = torch.randn(1, 1, 16, 16)
    grad2d = image_gradient(img2d)
    assert grad2d.shape == (1, 2, 16, 16)
    
    # 5. image_gradient 3D
    img3d = torch.randn(1, 1, 16, 16, 16)
    grad3d = image_gradient(img3d)
    assert grad3d.shape == (1, 3, 16, 16, 16)
    
    # 6. fluid_smooth
    smooth2d = fluid_smooth(img2d.expand(1, 2, 16, 16), sigma=1.0, dim=2)
    assert smooth2d.shape == (1, 2, 16, 16)
    
    smooth2d_nosmooth = fluid_smooth(img2d.expand(1, 2, 16, 16), sigma=0.0, dim=2)
    assert torch.allclose(img2d.expand(1, 2, 16, 16), smooth2d_nosmooth)
    
    # 7. integrate_svf
    phi_svf = integrate_svf(torch.zeros((1, 2, 16, 16)), n_steps=1)
    assert phi_svf.shape == (1, 2, 16, 16)
    
    # 8. integrate_forward 2D
    v_list_2d = [torch.zeros((1, 2, 16, 16)), torch.zeros((1, 2, 16, 16))]
    phi_hist_2d = integrate_forward(v_list_2d, (16, 16))
    assert len(phi_hist_2d) == 3 # init + 2 steps
    
    # 9. integrate_forward 3D
    v_list_3d = [torch.zeros((1, 3, 16, 16, 16))]
    phi_hist_3d = integrate_forward(v_list_3d, (16, 16, 16))
    assert len(phi_hist_3d) == 2

def test_tvf_adj_3d_basic():
    """Test the basic execution of tvf_registration_adjoint in 3D."""
    # Create random 3D images
    fi = ants.from_numpy(np.random.randn(8, 8, 8).astype(np.float32))
    mi = ants.from_numpy(np.random.randn(8, 8, 8).astype(np.float32))
    
    res = tvf_registration_adjoint(
        fixed=fi,
        moving=mi,
        flow_sigma=1.0,
        total_sigma=1.0,
        lr=10.0,
        levels=[2], # downsample to 4x4x4
        reg_iterations=[1],
        device='cpu'
    )
    
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
    
    warp = ants.image_read(res['fwdtransforms'][0])
    assert warp.shape == (8, 8, 8)
    assert warp.components == 3

def test_tvf_adj_initial_transform():
    """Test tvf_registration_adjoint with an initial transform."""
    import tempfile
    
    fi = ants.from_numpy(np.random.randn(8, 8, 8).astype(np.float32))
    mi = ants.from_numpy(np.random.randn(8, 8, 8).astype(np.float32))
    
    # Create a dummy initial transform (identity)
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tx_path = tmp.name
        
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    ants.write_transform(tx, tx_path)
    
    res = tvf_registration_adjoint(
        fixed=fi,
        moving=mi,
        initial_transform=tx_path,
        flow_sigma=1.0,
        total_sigma=1.0,
        lr=10.0,
        levels=[2],
        reg_iterations=[1],
        device='cpu'
    )
    
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
    assert len(res['fwdtransforms']) == 2 # non-linear + affine

