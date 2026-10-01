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



def test_tvf_adj_svf_grid_helpers_and_export_are_consistent():
    # scaling and squaring of a constant field is the field itself
    v = torch.zeros((1, 2, 12, 14))
    v[:, 0] = 1.5
    u = integrate_svf(v, n_steps=4)
    np.testing.assert_allclose(u[0, 0, 3:-3, 3:-3].numpy(), 1.5, atol=1e-4)
    # physical grid <-> normalised grid are inverses (oblique, anisotropic, 2-D)
    th = 0.3
    D = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    g = get_physical_grid_torch((10, 7), (0.5, 2.0), (3.0, -1.0), D)
    n = physical_to_normalized_torch(g, (10, 7), (0.5, 2.0), (3.0, -1.0), D)
    assert np.allclose(n[0, 0].numpy(), [-1, -1], atol=1e-5) and np.allclose(n[-1, -1].numpy(), [1, 1], atol=1e-5)
    # one voxel of x velocity moves the points by 2 / (W - 1) in normalised units
    phi = integrate_forward([torch.cat([torch.ones(1, 1, 9, 11), torch.zeros(1, 1, 9, 11)], 1)], (9, 11))
    assert np.allclose((phi[0] - phi[1])[0, 4, 5].numpy(), [2.0 / 10, 0.0], atol=1e-6)


def test_tvf_adj_upsampling_rescales_voxel_velocities_and_skipped_last_level():
    fi = ants.from_numpy(np.random.default_rng(0).random((32, 32)).astype(np.float32))
    adj = TVFRegistrationAdjoint(fi, fi, levels=[2, 1], reg_iterations=[1, 0], device='cpu')
    adj.v = torch.ones((3, 1, 2, 16, 16))
    out = adj._resample_velocity([32, 32])
    assert torch.allclose(out, torch.full_like(out, 31 / 15))
    v = adj.fit()                                   # last level skipped: no crash
    assert v.shape == (3, 1, 2, 32, 32)


def test_tvf_adj_export_components_and_spacing(monkeypatch):
    import syntx.tvf_adj as ta
    fi = ants.from_numpy(np.zeros((20, 24), np.float32), spacing=(2.0, 1.0))   # ANTs (x, y)

    def fake_fit(self):
        v = torch.zeros((3, 1, 2, *self.shape))
        v[:, :, 0] = -1.0                       # -1 voxel along x per unit time -> phi moves +x
        self.v = v
        return v

    monkeypatch.setattr(ta.TVFRegistrationAdjoint, "fit", fake_fit)
    res = ta.tvf_registration_adjoint(fi, fi, levels=[1], reg_iterations=[1], device='cpu')
    w = ants.image_read(res['fwdtransforms'][0]).numpy()        # (x, y, 2) physical
    np.testing.assert_allclose(w[10, 12], [2.0, 0.0], atol=1e-4)  # 1 voxel x 2 mm, along x
