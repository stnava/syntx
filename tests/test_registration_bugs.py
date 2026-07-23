import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import ants
import tempfile
from syntx.syn import get_rotation_matrix, physical_to_grid_affine, registration
from syntx.syn_jax import get_rotation_matrix_jax, get_affine_matrix_jax

def test_zero_rotation_gradient_pytorch():
    """
    Test that the gradient of the rotation matrix at omega=0 is not zero in PyTorch.
    This prevents the Lie algebra parameterization from locking the optimizer.
    """
    for dim in [2, 3]:
        omega_size = 1 if dim == 2 else 3
        omega = torch.zeros(omega_size, dtype=torch.float32, requires_grad=True)
        R = get_rotation_matrix(omega, dim)
        
        # Define a dummy loss that depends on off-diagonal elements
        if dim == 2:
            loss = R[0, 1] - R[1, 0]  # -sin - sin = -2sin -> grad is not 0
        else:
            loss = R[0, 1] - R[1, 2] + R[2, 0]
            
        loss.backward()
        
        assert omega.grad is not None, f"Gradient is None for dim {dim}"
        assert not torch.allclose(omega.grad, torch.zeros_like(omega)), f"Gradient is zero at omega=0 for dim {dim}"

def test_zero_rotation_gradient_jax():
    """
    Test that the gradient of the rotation matrix at omega=0 is not zero in JAX.
    """
    for dim in [2, 3]:
        omega_size = 1 if dim == 2 else 3
        
        def dummy_loss(omega):
            R = get_rotation_matrix_jax(omega, dim)
            if dim == 2:
                return R[0, 1] - R[1, 0]
            else:
                return R[0, 1] - R[1, 2] + R[2, 0]
                
        grad_fn = jax.grad(dummy_loss)
        g = grad_fn(jnp.zeros(omega_size, dtype=jnp.float32))
        
        assert not jnp.allclose(g, jnp.zeros_like(g)), f"Gradient is zero at omega=0 for dim {dim}"

def test_physical_space_mapping():
    """
    Test that physical_to_grid_affine maps physical translation correctly 
    into [-1, 1] grid space, accounting for origin, spacing, and directions.
    """
    dim = 3
    shape = (100, 100, 100)
    spacing = (2.0, 2.0, 2.0)
    origin = (-50.0, -50.0, -50.0)
    
    img = ants.from_numpy(np.zeros(shape, dtype=np.float32), spacing=spacing, origin=origin)
    
    # Let's say we have a physical translation of 50mm in x, y, z
    t_phys = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    M_phys = np.eye(dim, dtype=np.float32)
    
    T_grid = physical_to_grid_affine(M_phys, t_phys, img, img)
    
    # Full physical width of the image is (100-1) * 2 = 198 mm
    # A physical translation of 50mm should map to 50 / (198/2) = 50 / 99 ≈ 0.505 in normalized [-1, 1] space
    expected_translation = 50.0 / 99.0
    
    # In T_grid (PyTorch order z, y, x), the translation vector is in the last column
    t_grid = T_grid[:dim, dim]
    
    assert np.allclose(t_grid, expected_translation, atol=1e-4), f"Physical translation not correctly mapped to grid. Got {t_grid}"

@pytest.fixture
def synthetic_2d_circles():
    # Create a 64x64 grid
    y, x = np.ogrid[-32:32, -32:32]
    
    # Fixed image: Circle at center, radius 15
    mask_f = x**2 + y**2 <= 15**2
    fi_arr = np.zeros((64, 64), dtype=np.float32)
    fi_arr[mask_f] = 1.0
    
    # Moving image: Circle shifted by 5 pixels, radius 15
    mask_m = (x-5)**2 + (y-5)**2 <= 15**2
    mi_arr = np.zeros((64, 64), dtype=np.float32)
    mi_arr[mask_m] = 1.0
    
    fi = ants.from_numpy(fi_arr, origin=(0., 0.), spacing=(1., 1.))
    mi = ants.from_numpy(mi_arr, origin=(0., 0.), spacing=(1., 1.))
    
    # Labels are just the masks themselves
    fl = ants.from_numpy(mask_f.astype(np.float32), origin=(0., 0.), spacing=(1., 1.))
    ml = ants.from_numpy(mask_m.astype(np.float32), origin=(0., 0.), spacing=(1., 1.))
    
    return fi, mi, fl, ml

def compute_mean_dice_2d(fixed_label, moving_label, transformlist):
    fl = ants.resample_image(fixed_label, (2, 2), use_voxels=False, interp_type=1)
    ml = ants.resample_image(moving_label, (2, 2), use_voxels=False, interp_type=1)
    warped_ml = ants.apply_transforms(fixed=fl, moving=ml, transformlist=transformlist, interpolator='nearestNeighbor')
    overlap = ants.label_overlap_measures(fl, warped_ml)
    return overlap['MeanOverlap'][0]

def test_2d_grid_folding_pytorch(synthetic_2d_circles):
    fi, mi, fl, ml = synthetic_2d_circles
    
    # We test deformable registration and check the folding rate
    res = registration(
        fixed=fi, moving=mi, type_of_transform='SyN',
        backend='pytorch', syn_metric='mattes_mi',
        affine_iterations=[0], reg_iterations=[20, 0, 0],
        grad_step=0.7 # Aggressive step that caused folding
    )
    
    # In PyTorch backend, the warp_l2r field is saved as an ITK displacement field
    warp_file = [t for t in res['fwdtransforms'] if 'warp' in t or 'Warp' in t or t.endswith('.nii.gz')][0]
    warp_img = ants.image_read(warp_file)
    
    jac = ants.create_jacobian_determinant_image(fi, warp_img)
    jac_arr = jac.numpy()
    
    folding_rate = np.mean(jac_arr <= 0)
    assert folding_rate < 0.001, f"Severe folding detected in 2D PyTorch SyN: {folding_rate*100}%"

def test_2d_vgg_lncc_regression(synthetic_2d_circles):
    fi, mi, fl, ml = synthetic_2d_circles
    
    # Baseline intensity LNCC
    res_base = registration(
        fixed=fi, moving=mi, type_of_transform='SyN',
        backend='pytorch', syn_metric='lncc',
        affine_iterations=[50, 0], reg_iterations=[50, 0]
    )
    dice_base = compute_mean_dice_2d(fl, ml, res_base['fwdtransforms'])
    
    # VGG LNCC
    res_vgg = registration(
        fixed=fi, moving=mi, type_of_transform='SyN',
        backend='pytorch', syn_metric='vgg19', vgg_mode='lncc',
        affine_iterations=[50, 0], reg_iterations=[50, 0]
    )
    dice_vgg = compute_mean_dice_2d(fl, ml, res_vgg['fwdtransforms'])
    
    print(f"\nBase LNCC DICE: {dice_base:.4f}")
    print(f"VGG19 LNCC DICE: {dice_vgg:.4f}")
    
    assert dice_vgg >= dice_base - 0.05, f"VGG LNCC caused major regression: {dice_vgg:.4f} vs {dice_base:.4f}"
