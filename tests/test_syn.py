import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F
import ants

# Avoid dependency on external procedural generators by simulating images inline or importing siq
try:
    import siq
except ImportError:
    siq = None

from syntx.syn import (
    SyNTo,
    compute_jacobian_determinant_nd,
    local_ncc_loss_nd,
    mattes_mi_loss_nd,
    separable_gaussian_filter,
    registration
)

def compute_pearson_correlation(x, y):
    x_flat = x.ravel()
    y_flat = y.ravel()
    return float(np.corrcoef(x_flat, y_flat)[0, 1])

def get_test_image_2d():
    if siq is not None:
        return siq.simulate_image(shaper=[64, 64], n_levels=8)
    else:
        # Create a simple geometric phantom if siq is not available
        vol = np.zeros((64, 64), dtype=float)
        # Draw some circles/rectangles
        y, x = np.ogrid[:64, :64]
        mask1 = (x - 32)**2 + (y - 32)**2 < 15**2
        mask2 = (x - 20)**2 + (y - 20)**2 < 8**2
        vol[mask1] = 10
        vol[mask2] = 20
        # Add blur
        img = ants.from_numpy(vol, spacing=(1, 1), origin=(0, 0))
        img = ants.smooth_image(img, 1.5)
        return img

def get_test_image_3d():
    if siq is not None:
        return siq.simulate_image(shaper=[32, 32, 32], n_levels=6)
    else:
        # Create a simple geometric phantom in 3D
        vol = np.zeros((32, 32, 32), dtype=float)
        z, y, x = np.ogrid[:32, :32, :32]
        mask1 = (x - 16)**2 + (y - 16)**2 + (z - 16)**2 < 8**2
        vol[mask1] = 15
        img = ants.from_numpy(vol, spacing=(1, 1, 1), origin=(0, 0, 0))
        img = ants.smooth_image(img, 1.0)
        return img

def run_test_2d(similarity_metric):
    print(f"\n--- Running 2D test with metric: {similarity_metric} ---")
    fixed_img = get_test_image_2d()
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.tensor(fixed_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    fi_mean = fixed_np.mean()
    fi_std = fixed_np.std() + 1e-8
    fi_norm = (fixed_np - fi_mean) / fi_std
    
    device = torch.device('cpu')
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, 64, device=device),
        torch.linspace(-1, 1, 64, device=device),
        indexing='ij'
    )
    disp_x = 0.08 * torch.sin(math.pi * grid_y)
    disp_y = 0.04 * torch.cos(math.pi * grid_x)
    
    grid = torch.stack([grid_x + disp_x, grid_y + disp_y], dim=-1).unsqueeze(0)
    moving_tensor = F.grid_sample(fixed_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    moving_np = moving_tensor.squeeze().numpy()
    
    mi_mean = moving_np.mean()
    mi_std = moving_np.std() + 1e-8
    mi_norm = (moving_np - mi_mean) / mi_std
    
    moving_img = ants.from_numpy(moving_np, spacing=fixed_img.spacing, origin=fixed_img.origin, direction=fixed_img.direction)
    
    fi_norm_tensor = torch.tensor(fi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_norm_tensor = torch.tensor(mi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run ANTs SyN registration for reference
    reg_ants = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')
    warped_ants = reg_ants['warpedmovout'].numpy()
    corr_ants = compute_pearson_correlation(fixed_np, warped_ants)
    
    # Run SyNTo registration
    model = SyNTo(dim=2, grid_shape=(64, 64), fluid_sigma=2.0, elastic_sigma=1.0)
    model.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=30, 
        cfl_voxels=0.15,
        affine_epochs=30, 
        affine_lr=1e-2,
        similarity_metric=similarity_metric
    )
    
    with torch.no_grad():
        warped_py_tensor = model(mi_norm_tensor)
        warped_py = warped_py_tensor.squeeze().numpy()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_py)
    
    with torch.no_grad():
        jac = compute_jacobian_determinant_nd(model.warp_l2r)
        min_jac = torch.min(jac[..., 1:-1, 1:-1]).item()
        
    assert corr_py > 0.60
    assert min_jac >= -1e-5

def run_test_3d(similarity_metric):
    print(f"\n--- Running 3D test with metric: {similarity_metric} ---")
    fixed_img = get_test_image_3d()
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.tensor(fixed_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    fi_mean = fixed_np.mean()
    fi_std = fixed_np.std() + 1e-8
    fi_norm = (fixed_np - fi_mean) / fi_std
    
    theta = math.radians(8.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    A = torch.tensor([
        [cos_t, -sin_t, 0.0, 0.04],
        [sin_t,  cos_t, 0.0, -0.02],
        [0.0,    0.0,   1.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0)
    
    grid = F.affine_grid(A, size=[1, 1, 32, 32, 32], align_corners=True)
    moving_tensor = F.grid_sample(fixed_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    moving_np = moving_tensor.squeeze().numpy()
    
    mi_mean = moving_np.mean()
    mi_std = moving_np.std() + 1e-8
    mi_norm = (moving_np - mi_mean) / mi_std
    
    moving_img = ants.from_numpy(moving_np, spacing=fixed_img.spacing, origin=fixed_img.origin, direction=fixed_img.direction)
    
    fi_norm_tensor = torch.tensor(fi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_norm_tensor = torch.tensor(mi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run SyNTo registration
    model = SyNTo(dim=3, grid_shape=(32, 32, 32), fluid_sigma=2.0, elastic_sigma=1.0)
    model.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=30, 
        cfl_voxels=0.15,
        affine_epochs=30, 
        affine_lr=1e-2,
        similarity_metric=similarity_metric
    )
    
    with torch.no_grad():
        warped_py_tensor = model(mi_norm_tensor)
        warped_py = warped_py_tensor.squeeze().numpy()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_py)
    
    with torch.no_grad():
        jac = compute_jacobian_determinant_nd(model.warp_l2r)
        min_jac = torch.min(jac[..., 1:-1, 1:-1, 1:-1]).item()
        
    assert corr_py > 0.60
    assert min_jac >= -1e-5

def test_pytorch_syn_2d_lncc():
    run_test_2d('lncc')

def test_pytorch_syn_2d_mattes_mi():
    run_test_2d('mattes_mi')

@pytest.mark.slow
def test_pytorch_syn_3d_lncc():
    run_test_3d('lncc')

@pytest.mark.slow
def test_pytorch_syn_3d_mattes_mi():
    run_test_3d('mattes_mi')

def test_pytorch_syn_2d_vgg19():
    run_test_2d('vgg19')

def test_pytorch_syn_2d_vgg19_lncc():
    print("\n--- Running 2D test with metric: vgg19, mode: lncc ---")
    fixed_img = get_test_image_2d()
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.tensor(fixed_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    fi_mean = fixed_np.mean()
    fi_std = fixed_np.std() + 1e-8
    fi_norm = (fixed_np - fi_mean) / fi_std
    
    device = torch.device('cpu')
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, 64, device=device),
        torch.linspace(-1, 1, 64, device=device),
        indexing='ij'
    )
    disp_x = 0.08 * torch.sin(math.pi * grid_y)
    disp_y = 0.04 * torch.cos(math.pi * grid_x)
    
    grid = torch.stack([grid_x + disp_x, grid_y + disp_y], dim=-1).unsqueeze(0)
    moving_tensor = F.grid_sample(fixed_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    moving_np = moving_tensor.squeeze().numpy()
    
    mi_mean = moving_np.mean()
    mi_std = moving_np.std() + 1e-8
    mi_norm = (moving_np - mi_mean) / mi_std
    
    fi_norm_tensor = torch.tensor(fi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_norm_tensor = torch.tensor(mi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    model = SyNTo(dim=2, grid_shape=(64, 64), fluid_sigma=2.0, elastic_sigma=1.0)
    model.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=10, 
        cfl_voxels=0.15,
        affine_epochs=10, 
        affine_lr=1e-2,
        similarity_metric='vgg19',
        vgg_mode='lncc',
        vgg_layers=[4]
    )
    
    with torch.no_grad():
        warped_py_tensor = model(mi_norm_tensor)
        warped_py = warped_py_tensor.squeeze().numpy()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_py)
    assert corr_py > 0.50

@pytest.mark.slow
def test_pytorch_syn_3d_vgg19():
    run_test_3d('vgg19')

@pytest.mark.slow
def test_pytorch_syn_3d_vgg19_lncc3d():
    print("\n--- Running 3D test with metric: vgg19, mode: lncc_3d ---")
    fixed_img = get_test_image_3d()
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.tensor(fixed_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    fi_mean = fixed_np.mean()
    fi_std = fixed_np.std() + 1e-8
    fi_norm = (fixed_np - fi_mean) / fi_std
    
    theta = math.radians(8.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    A = torch.tensor([
        [cos_t, -sin_t, 0.0, 0.04],
        [sin_t,  cos_t, 0.0, -0.02],
        [0.0,    0.0,   1.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0)
    
    grid = F.affine_grid(A, size=[1, 1, 32, 32, 32], align_corners=True)
    moving_tensor = F.grid_sample(fixed_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    moving_np = moving_tensor.squeeze().numpy()
    
    mi_mean = moving_np.mean()
    mi_std = moving_np.std() + 1e-8
    mi_norm = (moving_np - mi_mean) / mi_std
    
    moving_img = ants.from_numpy(moving_np, spacing=fixed_img.spacing, origin=fixed_img.origin, direction=fixed_img.direction)
    
    fi_norm_tensor = torch.tensor(fi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_norm_tensor = torch.tensor(mi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    model = SyNTo(dim=3, grid_shape=(32, 32, 32), fluid_sigma=2.0, elastic_sigma=1.0)
    model.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=10, 
        cfl_voxels=0.15,
        affine_epochs=10, 
        affine_lr=1e-2,
        similarity_metric='vgg19',
        vgg_mode='lncc_3d'
    )
    
    with torch.no_grad():
        warped_py_tensor = model(mi_norm_tensor)
        warped_py = warped_py_tensor.squeeze().numpy()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_py)
    assert corr_py > 0.50

def test_high_level_registration():
    # Test top-level API
    fixed = get_test_image_2d()
    moving = get_test_image_2d()
    tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2, translation=(1.0, 1.0))
    moving = tx.apply_to_image(moving)
    
    res = registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNTo',
        backend='pytorch',
        reg_iterations=[10, 5],
        affine_iterations=[10, 5],
        levels=[2, 1]
    )
    
    assert 'warpedmovout' in res
    assert 'warpedfixout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
