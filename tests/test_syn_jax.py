import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F
import ants

try:
    import siq
except ImportError:
    siq = None

from syntx.syn_jax import (
    SyNTo,
    compute_jacobian_determinant_nd_jax,
    local_ncc_loss_nd_jax,
    mattes_mi_loss_nd_jax,
    separable_gaussian_filter_jax
)
from syntx.syn import registration

def compute_pearson_correlation(x, y):
    x_flat = x.ravel()
    y_flat = y.ravel()
    return float(np.corrcoef(x_flat, y_flat)[0, 1])

def get_test_image_2d():
    if siq is not None:
        return siq.simulate_image(shaper=[64, 64], n_levels=8)
    else:
        vol = np.zeros((64, 64), dtype=float)
        y, x = np.ogrid[:64, :64]
        mask1 = (x - 32)**2 + (y - 32)**2 < 15**2
        mask2 = (x - 20)**2 + (y - 20)**2 < 8**2
        vol[mask1] = 10
        vol[mask2] = 20
        img = ants.from_numpy(vol, spacing=(1, 1), origin=(0, 0))
        img = ants.smooth_image(img, 1.5)
        return img

def get_test_image_3d():
    if siq is not None:
        return siq.simulate_image(shaper=[32, 32, 32], n_levels=6)
    else:
        vol = np.zeros((32, 32, 32), dtype=float)
        z, y, x = np.ogrid[:32, :32, :32]
        mask1 = (x - 16)**2 + (y - 16)**2 + (z - 16)**2 < 8**2
        vol[mask1] = 15
        img = ants.from_numpy(vol, spacing=(1, 1, 1), origin=(0, 0, 0))
        img = ants.smooth_image(img, 1.0)
        return img

def run_test_2d(similarity_metric):
    print(f"\n--- Running JAX 2D test with metric: {similarity_metric} ---")
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
    
    # Run JAX SyNTo registration
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
    
    # Apply forward warp
    warped_jax_tensor = model.forward(mi_norm_tensor)
    if hasattr(warped_jax_tensor, 'detach'):
        warped_jax = warped_jax_tensor.squeeze().detach().cpu().numpy()
    else:
        warped_jax = np.array(warped_jax_tensor).squeeze()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_jax)
    
    # Compute Jacobian determinants
    jac = compute_jacobian_determinant_nd_jax(model.warp_l2r)
    min_jac = float(np.min(jac))
    
    assert corr_py > 0.60
    assert min_jac > 0.0

def run_test_3d(similarity_metric):
    print(f"\n--- Running JAX 3D test with metric: {similarity_metric} ---")
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
    
    # Run JAX SyNTo registration
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
    
    # Apply forward warp
    warped_jax_tensor = model.forward(mi_norm_tensor)
    if hasattr(warped_jax_tensor, 'detach'):
        warped_jax = warped_jax_tensor.squeeze().detach().cpu().numpy()
    else:
        warped_jax = np.array(warped_jax_tensor).squeeze()
        
    corr_py = compute_pearson_correlation(fi_norm, warped_jax)
    
    # Compute Jacobian determinants
    jac = compute_jacobian_determinant_nd_jax(model.warp_l2r)
    min_jac = float(np.min(jac))
    
    assert corr_py > 0.60
    assert min_jac > 0.0

def test_jax_syn_2d_lncc():
    run_test_2d('lncc')

def test_jax_syn_2d_mattes_mi():
    run_test_2d('mattes_mi')

@pytest.mark.slow
def test_jax_syn_3d_lncc():
    run_test_3d('lncc')

@pytest.mark.slow
def test_jax_syn_3d_mattes_mi():
    run_test_3d('mattes_mi')

def test_high_level_registration_jax():
    fixed = get_test_image_2d()
    moving = get_test_image_2d()
    tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2, translation=(1.0, 1.0))
    moving = tx.apply_to_image(moving)
    
    res = registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNTo',
        backend='jax',
        reg_iterations=[10, 5],
        affine_iterations=[10, 5],
        levels=[2, 1]
    )
    
    assert 'warpedmovout' in res
    assert 'warpedfixout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res


def test_new_jax_helpers():
    import jax.numpy as jnp
    from syntx.syn_jax import (
        prepare_mid_images_and_gradients_jax,
        syn_update_step_jax,
        upscale_initial_grid,
        to_jax_array
    )
    from syntx.syn import compute_initial_grid
    import ants
    import numpy as np

    shape = (16, 16)
    dim = 2
    warp_l2r = jnp.zeros((1, 16, 16, 2))
    warp_r2l = jnp.zeros((1, 16, 16, 2))
    warp_l2r_inv = jnp.zeros((1, 16, 16, 2))
    warp_r2l_inv = jnp.zeros((1, 16, 16, 2))
    
    I_curr = jnp.ones((1, 1, 16, 16))
    J_curr = jnp.ones((1, 1, 16, 16))
    
    grids = [jnp.linspace(-1.0, 1.0, size) for size in shape]
    meshgrid = jnp.meshgrid(*grids, indexing='ij')
    identity = jnp.stack(list(reversed(meshgrid)), axis=-1)
    identity = jnp.expand_dims(identity, axis=0)
    
    # 1. test prepare_mid_images_and_gradients_jax
    spacing_arg = (1.0, 1.0)
    I_mid, J_mid, grad_I, grad_J = prepare_mid_images_and_gradients_jax(
        warp_l2r, warp_r2l, I_curr, J_curr,
        True, spacing_arg, identity
    )
    assert I_mid.shape == (1, 1, 16, 16)
    assert J_mid.shape == (1, 1, 16, 16)
    assert grad_I.shape == (1, 16, 16, 2)
    assert grad_J.shape == (1, 16, 16, 2)
    
    # 2. test syn_update_step_jax
    grad_l_raw = jnp.zeros((1, 16, 16, 2))
    grad_r_raw = jnp.zeros((1, 16, 16, 2))
    b_mask = jnp.ones((1, 16, 16, 1))
    
    w_l2r, w_r2l, w_l2r_inv, w_r2l_inv = syn_update_step_jax(
        warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv,
        grad_l_raw, grad_r_raw, identity, b_mask,
        True, spacing_arg, 1.0, 1.0, 0.75,
        5, 'fixed_point'
    )
    assert w_l2r.shape == (1, 16, 16, 2)
    assert w_r2l_inv.shape == (1, 16, 16, 2)
    
    # 3. test upscale_initial_grid
    grid_upscaled = upscale_initial_grid(warp_l2r, (32, 32))
    assert grid_upscaled.shape == (1, 32, 32, 2)
    
    # 4. test compute_initial_grid
    fixed_img = ants.from_numpy(np.ones((16, 16), dtype=np.float32))
    moving_img = ants.from_numpy(np.ones((16, 16), dtype=np.float32))
    tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2, translation=(0.5, 0.5))
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        tx_path = os.path.join(tmpdir, "init_tx_test.mat")
        ants.write_transform(tx, tx_path)
        init_grid = compute_initial_grid(fixed_img, moving_img, [tx_path])
        assert init_grid.shape == (1, 16, 16, 2)


def test_pytorch_loss_jax_jit():
    import jax
    import jax.numpy as jnp
    import torch
    from syntx.syn_jax import make_pytorch_loss_jax

    class DummyPyTorchLoss(torch.nn.Module):
        def forward(self, m, f):
            return torch.mean((m - f) ** 2)

    py_loss = DummyPyTorchLoss()
    jax_loss = make_pytorch_loss_jax(py_loss)

    @jax.jit
    def jit_loss(m, f):
        return jax_loss(m, f)

    m_val = jnp.ones((2, 2))
    f_val = jnp.zeros((2, 2))
    
    # Test forward
    val = jit_loss(m_val, f_val)
    assert float(val) == 1.0

    # Test backward (custom VJP backward callback)
    grad_fn = jax.grad(jit_loss, argnums=(0, 1))
    g_m, g_f = grad_fn(m_val, f_val)
    assert g_m.shape == (2, 2)
    assert g_f.shape == (2, 2)


def test_dlpack_empty_tensor():
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import to_jax_array_dl, to_torch_tensor
    
    # Empty torch tensor to JAX
    t_empty = torch.empty((0, 5))
    j_empty = to_jax_array_dl(t_empty)
    assert j_empty.shape == (0, 5)

    # Empty JAX array to torch
    j_arr = jnp.empty((0, 3))
    t_arr = to_torch_tensor(j_arr)
    assert t_arr.shape == (0, 3)


def test_update_inverse_field_neumann():
    import jax.numpy as jnp
    from syntx.syn_jax import update_inverse_field_nd_jax
    w = jnp.zeros((1, 8, 8, 2))
    w_inv = jnp.zeros((1, 8, 8, 2))
    res = update_inverse_field_nd_jax(w, w_inv, steps=2, method='neumann')
    assert res.shape == (1, 8, 8, 2)


def test_mattes_mi_sampling():
    import jax.numpy as jnp
    from syntx.syn_jax import mattes_mi_loss_nd_jax
    x = jnp.ones((1, 1, 10, 10))
    y = jnp.zeros((1, 1, 10, 10))
    loss = mattes_mi_loss_nd_jax(x, y, num_bins=8, sampling_percentage=0.5)
    assert float(loss) is not None


def test_compute_physical_jacobian_jax():
    import jax.numpy as jnp
    from syntx.syn_jax import compute_physical_jacobian_determinant_jax
    w = jnp.zeros((1, 8, 8, 2))
    direction = jnp.eye(2)
    spacing = jnp.array([1.0, 1.0])
    jac = compute_physical_jacobian_determinant_jax(w, direction, spacing)
    assert jac.shape == (1, 8, 8)


def test_synto_jax_affine_grids():
    from syntx.syn_jax import SyNTo
    import torch
    model = SyNTo(dim=2, grid_shape=(8, 8))
    g = model.get_affine_grid((8, 8))
    g_inv = model.get_inverse_affine_grid((8, 8))
    assert isinstance(g, torch.Tensor)
    assert isinstance(g_inv, torch.Tensor)


def test_synto_jax_forward_inverse():
    from syntx.syn_jax import SyNTo
    import torch
    model = SyNTo(dim=2, grid_shape=(8, 8))
    img = torch.ones((1, 1, 8, 8))
    warped = model.forward(img)
    warped_inv = model.forward_inverse(img)
    assert warped.shape == (1, 1, 8, 8)
    assert warped_inv.shape == (1, 1, 8, 8)

def test_ants_parity_2d_jax():
    import ants
    from syntx.syn import registration
    
    # Load the standard 2D phantoms used in comparison reports
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r27'))
    
    # Run registration using JAX backend with standard settings
    res = registration(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[30, 20],
        reg_iterations=[30, 20],
        grad_step=0.5,
        flow_sigma=1.0
    )
    
    # Define local helper for overlap calculation
    def compute_tissue_overlap(fixed_img, warped_img):
        fixed_seg = ants.threshold_image(fixed_img, 'Otsu', 3)
        warped_seg = ants.threshold_image(warped_img, 'Otsu', 3)
        overlap = ants.label_overlap_measures(fixed_seg, warped_seg)
        if 'MeanOverlap' in overlap.columns:
            return float(overlap.loc[overlap['Label'] == 'All', 'MeanOverlap'].values[0])
        return 0.0

    dice = compute_tissue_overlap(fi, res['warpedmovout'])
    # Verify that we achieve high quality registration alignment (DICE >= 0.55)
    assert dice >= 0.55
